"""Stratified validation: select balanced samples across pattern hints, then run LLM detection.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_stratified_validation.py [--max-per-pattern 12]
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


LABEL_PATTERN_HINTS = {
    "distressed": "open", "overwhelmed": "open", "hurt": "open",
    "sad": "open", "lonely": "open", "vulnerable": "open",
    "crying": "open", "heartbroken": "open", "desperate": "open",
    "defensive": "defensive", "guarded": "defensive", "deflecting": "defensive",
    "avoidant": "defensive", "withdrawn": "defensive", "evasive": "defensive",
    "lighthearted": "masked", "playful": "masked", "sarcastic": "masked",
    "joking": "masked", "self-deprecating": "masked",
    "resigned": "denial", "detached": "denial", "dismissive": "denial",
    "stoic": "denial", "indifferent": "denial",
}


def load_and_stratify(path: Path, max_per_pattern: int = 12) -> list[dict]:
    """Load samples and stratify by expected pattern."""
    buckets: dict[str, list[dict]] = defaultdict(list)

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            for label in data.get("prod_labels", []):
                pattern = LABEL_PATTERN_HINTS.get(label.lower().strip())
                if pattern and len(buckets[pattern]) < max_per_pattern:
                    buckets[pattern].append(data)
                    break  # only add once per sample

    # Also add some ambiguous/unlabeled as "unknown" for diversity
    samples = []
    for pattern, items in sorted(buckets.items()):
        print(f"  {pattern:12s}: {len(items)} samples")
        samples.extend(items)
    return samples


def build_conversation(sample: dict) -> list[dict]:
    """Build conversation from context."""
    context = sample.get("context", "")
    turns = []
    for line in context.strip().split("\n"):
        line = line.strip()
        if line.startswith("User:"):
            turns.append({"role": "speaker", "text": line[5:].strip()})
        elif line.startswith("Assistant:"):
            turns.append({"role": "chatter", "text": line[10:].strip()})
    if not turns:
        turns.append({"role": "speaker", "text": sample["user_text"]})
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-pattern", type=int, default=12)
    parser.add_argument("--behavioral-only", action="store_true")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    data_path = Path("/Users/michael/emotion-detector/data/real_user/critical.jsonl")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    print(f"Stratified sampling (max {args.max_per_pattern} per pattern):")
    samples = load_and_stratify(data_path, args.max_per_pattern)
    print(f"Total: {len(samples)} samples\n")

    # --- Behavioral ---
    print("=== Behavioral Classification ===")
    behavioral_results = []
    pattern_counts = Counter()
    for sample in samples:
        features = extract_features(sample["user_text"])
        scores = classify_from_features(features)
        detected = max(scores, key=scores.get)
        pattern_counts[detected] += 1

        expected = None
        for label in sample.get("prod_labels", []):
            expected = LABEL_PATTERN_HINTS.get(label.lower().strip())
            if expected:
                break

        behavioral_results.append({
            "session_id": sample["session_id"],
            "text_preview": sample["user_text"][:60],
            "expected": expected,
            "detected": detected,
            "match": detected == expected if expected else None,
            "scores": {k: round(v, 3) for k, v in scores.items()},
        })

    matches = sum(1 for r in behavioral_results if r.get("match") is True)
    total_hinted = sum(1 for r in behavioral_results if r.get("match") is not None)
    print(f"Distribution: {dict(pattern_counts)}")
    print(f"Accuracy: {matches}/{total_hinted} = {matches/total_hinted:.1%}\n")

    if args.behavioral_only:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "stratified_behavioral.json"), "w") as f:
            json.dump(behavioral_results, f, indent=2, ensure_ascii=False)
        return

    # --- LLM ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("No ANTHROPIC_API_KEY. Stopping.")
        return

    from fragility_detector.detector import FragilityDetector
    from fragility_detector.star_map import generate_star_map

    detector = FragilityDetector(api_key)
    llm_results = []
    llm_pattern_counts = Counter()
    llm_matches = 0
    llm_total_hinted = 0

    print("=== LLM Classification ===")
    for i, sample in enumerate(samples):
        conversation = build_conversation(sample)
        try:
            snapshot = detector.detect(conversation, turn=sample["turn"])
            star = generate_star_map(snapshot)
            detected = snapshot.pattern.value
            llm_pattern_counts[detected] += 1

            expected = None
            for label in sample.get("prod_labels", []):
                expected = LABEL_PATTERN_HINTS.get(label.lower().strip())
                if expected:
                    break

            match = None
            if expected:
                llm_total_hinted += 1
                match = detected == expected
                if match:
                    llm_matches += 1

            llm_results.append({
                "session_id": sample["session_id"],
                "text_preview": sample["user_text"][:60],
                "expected": expected,
                "detected": detected,
                "confidence": round(snapshot.confidence, 3),
                "match": match,
                "scores": {k: round(v, 3) for k, v in snapshot.pattern_scores.items()},
                "star_label": star.star_label,
                "evidence": snapshot.evidence,
            })

            status = "✓" if match else ("✗" if match is False else " ")
            print(f"  [{i+1:2d}/{len(samples)}] {status} expected={expected or '?':12s} detected={detected:12s} "
                  f"conf={snapshot.confidence:.3f} | {sample['user_text'][:50]}...")

        except Exception as e:
            print(f"  [{i+1:2d}/{len(samples)}] ERROR: {e}")
            llm_results.append({"session_id": sample["session_id"], "error": str(e)})

        time.sleep(0.5)

    print(f"\n=== LLM Results ===")
    print(f"Distribution: {dict(llm_pattern_counts)}")
    print(f"Accuracy: {llm_matches}/{llm_total_hinted} = {llm_matches/llm_total_hinted:.1%}")

    # Confusion matrix
    print("\n--- Confusion Matrix ---")
    confusion = defaultdict(Counter)
    for r in llm_results:
        if r.get("expected") and r.get("detected"):
            confusion[r["expected"]][r["detected"]] += 1

    patterns = ["open", "defensive", "masked", "denial"]
    print(f"{'Expected':<12s} | " + " | ".join(f"{p:>10s}" for p in patterns))
    print("-" * 60)
    for exp in patterns:
        row = confusion[exp]
        cells = " | ".join(f"{row.get(p, 0):>10d}" for p in patterns)
        print(f"{exp:<12s} | {cells}")

    # Star label summary
    print("\n--- Star Labels ---")
    for pattern in patterns:
        count = llm_pattern_counts.get(pattern, 0)
        star = STAR_LABELS[FragilityPattern(pattern)]["star_label"]
        print(f"  {pattern:12s}: {count:3d} users → {star}")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "stratified_llm.json"), "w") as f:
        json.dump(llm_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}stratified_llm.json")


if __name__ == "__main__":
    main()
