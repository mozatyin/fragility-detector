"""Golden Set Evaluation: run fragility detection on 50 hand-labeled real user samples.

This is the TRUE accuracy metric. Unlike hand-crafted eval cases, these are:
- Real user messages from production
- Manually labeled by reading actual text + context
- Include neutral/no-vulnerability cases
- Include multiple languages (en, ar, bs, hi, es, ru)

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_golden_eval.py
    python run_golden_eval.py --behavioral-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.models import FragilityPattern, STAR_LABELS


# For neutral cases: if max LLM signal < threshold, we call it "neutral detected"
NEUTRAL_CONFIDENCE_THRESHOLD = 0.2


def load_golden_set(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def build_conversation_from_golden(sample: dict) -> list[dict]:
    """Build minimal conversation from golden set sample."""
    turns = []
    if sample.get("context"):
        turns.append({"role": "chatter", "text": sample["context"]})
    turns.append({"role": "speaker", "text": sample["user_text"]})
    return turns


def run_behavioral_eval(samples: list[dict]):
    """Behavioral-only evaluation."""
    results = []
    for s in samples:
        features = extract_features(s["user_text"])
        scores = classify_from_features(features)
        detected = max(scores, key=scores.get)

        # For neutral: behavioral can't detect neutral, so it's always a pattern
        expected = s["pattern"]
        correct = detected == expected

        results.append({
            "id": s["id"],
            "expected": expected,
            "detected": detected,
            "correct": correct,
            "scores": {k: round(v, 3) for k, v in scores.items()},
        })
    return results


def run_llm_eval(samples: list[dict], api_key: str):
    """LLM evaluation with neutral detection."""
    from fragility_detector.detector import FragilityDetector
    from fragility_detector.star_map import generate_star_map

    detector = FragilityDetector(api_key)
    results = []

    for i, s in enumerate(samples):
        conversation = build_conversation_from_golden(s)
        try:
            snapshot = detector.detect(conversation, turn=0)
            detected = snapshot.pattern.value
            conf = snapshot.confidence

            # Neutral detection: if confidence is very low (insufficient signal),
            # classify as neutral
            is_neutral_detected = conf < NEUTRAL_CONFIDENCE_THRESHOLD

            expected = s["pattern"]
            if expected == "neutral":
                correct = is_neutral_detected
                detected_label = "neutral" if is_neutral_detected else detected
            else:
                correct = detected == expected and not is_neutral_detected
                detected_label = "neutral" if is_neutral_detected else detected

            status = "OK" if correct else "MISS"
            lang = s.get("language", "?")
            print(f"  [{i+1:2d}/{len(samples)}] {status} {s['id']:6s} [{lang:2s}] "
                  f"exp={expected:12s} det={detected_label:12s} conf={conf:.3f} | {s['user_text'][:50]}...")

            results.append({
                "id": s["id"],
                "expected": expected,
                "detected": detected_label,
                "raw_detected": detected,
                "confidence": round(conf, 3),
                "correct": correct,
                "scores": {k: round(v, 3) for k, v in snapshot.pattern_scores.items()},
                "evidence": snapshot.evidence,
                "label_confidence": s.get("confidence", ""),
                "language": lang,
                "note": s.get("note", ""),
            })

        except Exception as e:
            print(f"  [{i+1:2d}/{len(samples)}] ERROR {s['id']}: {e}")
            results.append({
                "id": s["id"], "expected": s["pattern"],
                "detected": "error", "correct": False, "error": str(e),
            })

        time.sleep(0.5)

    return results


def print_report(results: list[dict], title: str):
    """Print evaluation report."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total: {total} | Correct: {correct} | Accuracy: {accuracy:.1%}")

    # Per-pattern metrics
    patterns = ["open", "defensive", "masked", "denial", "neutral"]
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()

    for r in results:
        support[r["expected"]] += 1
        if r["correct"]:
            tp[r["expected"]] += 1
        else:
            fp[r["detected"]] += 1
            fn[r["expected"]] += 1

    print(f"\n{'Pattern':<12s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>4s}")
    print("-" * 36)
    for pattern in patterns:
        if support[pattern] == 0:
            continue
        p = tp[pattern] / (tp[pattern] + fp[pattern]) if (tp[pattern] + fp[pattern]) > 0 else 0
        r = tp[pattern] / (tp[pattern] + fn[pattern]) if (tp[pattern] + fn[pattern]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{pattern:<12s} {p:>6.1%} {r:>6.1%} {f1:>6.1%} {support[pattern]:>4d}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    active_patterns = [p for p in patterns if support[p] > 0]
    print(f"{'Expected':<12s} | " + " | ".join(f"{p:>10s}" for p in active_patterns))
    print("-" * (14 + 13 * len(active_patterns)))
    confusion = defaultdict(Counter)
    for r in results:
        confusion[r["expected"]][r["detected"]] += 1
    for exp in active_patterns:
        row = confusion[exp]
        cells = " | ".join(f"{row.get(p, 0):>10d}" for p in active_patterns)
        print(f"{exp:<12s} | {cells}")

    # Errors breakdown
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for r in errors:
            lang = r.get("language", "?")
            conf_label = r.get("label_confidence", "")
            print(f"  {r['id']:6s} [{lang:2s}] [{conf_label:6s}] exp={r['expected']:12s} det={r['detected']:12s} "
                  f"conf={r.get('confidence', 0):.3f}")
            if r.get("note"):
                print(f"    note: {r['note']}")

    # Accuracy by label confidence
    high_conf = [r for r in results if r.get("label_confidence") == "high"]
    med_conf = [r for r in results if r.get("label_confidence") == "medium"]
    if high_conf:
        acc_h = sum(1 for r in high_conf if r["correct"]) / len(high_conf)
        print(f"\nAccuracy on high-confidence labels: {acc_h:.1%} ({sum(1 for r in high_conf if r['correct'])}/{len(high_conf)})")
    if med_conf:
        acc_m = sum(1 for r in med_conf if r["correct"]) / len(med_conf)
        print(f"Accuracy on medium-confidence labels: {acc_m:.1%} ({sum(1 for r in med_conf if r['correct'])}/{len(med_conf)})")

    # Accuracy by language
    lang_groups = defaultdict(list)
    for r in results:
        lang_groups[r.get("language", "?")].append(r)
    print(f"\nAccuracy by language:")
    for lang, group in sorted(lang_groups.items(), key=lambda x: -len(x[1])):
        acc = sum(1 for r in group if r["correct"]) / len(group)
        print(f"  {lang}: {acc:.1%} ({sum(1 for r in group if r['correct'])}/{len(group)})")

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral-only", action="store_true")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    golden_path = Path("data/golden_set.jsonl")
    if not golden_path.exists():
        print(f"Error: {golden_path} not found")
        sys.exit(1)

    samples = load_golden_set(golden_path)
    print(f"Loaded {len(samples)} golden set samples")

    # Distribution
    pattern_counts = Counter(s["pattern"] for s in samples)
    print(f"Distribution: {dict(pattern_counts)}")

    if args.behavioral_only:
        beh_results = run_behavioral_eval(samples)
        print_report(beh_results, "Behavioral-Only (Golden Set)")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print(f"\n--- Running LLM Detection ---")
    results = run_llm_eval(samples, api_key)
    accuracy = print_report(results, "LLM Detection (Golden Set)")

    # Save
    os.makedirs(args.output, exist_ok=True)
    save_data = {
        "eval": "golden_set",
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": accuracy,
        "neutral_threshold": NEUTRAL_CONFIDENCE_THRESHOLD,
        "results": results,
    }
    with open(os.path.join(args.output, "golden_eval.json"), "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}golden_eval.json")


if __name__ == "__main__":
    main()
