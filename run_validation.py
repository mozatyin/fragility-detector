"""Validation: run fragility detection on real user high-distress samples.

Extracts high-distress sessions from 584 critical samples, runs fragility
detection, cross-validates with production emotion labels.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_validation.py [--max-samples 50] [--behavioral-only]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.models import FragilityPattern, STAR_LABELS


# Map production labels to expected fragility patterns (heuristic ground truth)
LABEL_PATTERN_HINTS = {
    # Open pattern indicators
    "distressed": "open", "overwhelmed": "open", "hurt": "open",
    "sad": "open", "lonely": "open", "vulnerable": "open",
    "crying": "open", "heartbroken": "open", "desperate": "open",

    # Defensive pattern indicators
    "defensive": "defensive", "guarded": "defensive", "deflecting": "defensive",
    "avoidant": "defensive", "withdrawn": "defensive", "evasive": "defensive",

    # Masked pattern indicators
    "lighthearted": "masked", "playful": "masked", "sarcastic": "masked",
    "joking": "masked", "self-deprecating": "masked",

    # Denial pattern indicators
    "resigned": "denial", "detached": "denial", "dismissive": "denial",
    "stoic": "denial", "indifferent": "denial",
}

# Labels that indicate high emotional content (fragility-relevant)
HIGH_DISTRESS_LABELS = {
    "distressed", "overwhelmed", "hurt", "sad", "lonely", "crying",
    "heartbroken", "desperate", "anxious", "frustrated", "angry",
    "betrayed", "conflicted", "resigned", "defensive", "worried",
}


def load_critical_samples(path: Path, min_distress: float = 0.3) -> list[dict]:
    """Load high-distress samples from critical.jsonl."""
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("prod_distress", 0) >= min_distress:
                samples.append(data)
    return samples


def infer_expected_pattern(prod_labels: list[str]):
    """Infer expected fragility pattern from production emotion labels."""
    for label in prod_labels:
        label_lower = label.lower().strip()
        if label_lower in LABEL_PATTERN_HINTS:
            return LABEL_PATTERN_HINTS[label_lower]
    return None


def build_conversation(sample: dict) -> list[dict]:
    """Build conversation list from sample context."""
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


def run_behavioral_validation(samples: list[dict]) -> dict:
    """Run behavioral-only validation (no LLM cost)."""
    results = []
    pattern_counts = Counter()
    match_count = 0
    total_with_hint = 0

    for sample in samples:
        text = sample["user_text"]
        features = extract_features(text)
        scores = classify_from_features(features)
        detected = max(scores, key=scores.get)
        pattern_counts[detected] += 1

        expected = infer_expected_pattern(sample.get("prod_labels", []))

        result = {
            "session_id": sample["session_id"],
            "turn": sample["turn"],
            "text_preview": text[:80],
            "prod_distress": sample.get("prod_distress", 0),
            "prod_labels": sample.get("prod_labels", []),
            "detected_pattern": detected,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "expected_pattern": expected,
        }

        if expected:
            total_with_hint += 1
            if detected == expected:
                match_count += 1
                result["match"] = True
            else:
                result["match"] = False

        results.append(result)

    accuracy = match_count / total_with_hint if total_with_hint > 0 else 0

    return {
        "total_samples": len(samples),
        "pattern_distribution": dict(pattern_counts),
        "samples_with_label_hint": total_with_hint,
        "matches": match_count,
        "accuracy_on_hinted": round(accuracy, 3),
        "results": results,
    }


def run_llm_validation(samples: list[dict], api_key: str, max_samples: int = 50) -> dict:
    """Run LLM-based validation on a subset of samples."""
    from fragility_detector.detector import FragilityDetector
    from fragility_detector.star_map import generate_star_map

    detector = FragilityDetector(api_key)
    results = []
    pattern_counts = Counter()
    match_count = 0
    total_with_hint = 0

    subset = samples[:max_samples]
    print(f"Running LLM validation on {len(subset)} samples...")

    for i, sample in enumerate(subset):
        conversation = build_conversation(sample)
        try:
            snapshot = detector.detect(conversation, turn=sample["turn"])
            star = generate_star_map(snapshot)

            detected = snapshot.pattern.value
            pattern_counts[detected] += 1

            expected = infer_expected_pattern(sample.get("prod_labels", []))

            result = {
                "session_id": sample["session_id"],
                "turn": sample["turn"],
                "text_preview": sample["user_text"][:80],
                "prod_distress": sample.get("prod_distress", 0),
                "prod_labels": sample.get("prod_labels", []),
                "detected_pattern": detected,
                "confidence": round(snapshot.confidence, 3),
                "scores": {k: round(v, 3) for k, v in snapshot.pattern_scores.items()},
                "star_label": star.star_label,
                "expected_pattern": expected,
                "evidence": snapshot.evidence,
            }

            if expected:
                total_with_hint += 1
                if detected == expected:
                    match_count += 1
                    result["match"] = True
                else:
                    result["match"] = False

            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(subset)}] {detected:12s} conf={snapshot.confidence:.3f} | {sample['user_text'][:50]}...")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            results.append({
                "session_id": sample["session_id"],
                "turn": sample["turn"],
                "error": str(e),
            })

        # Rate limiting
        time.sleep(0.5)

    accuracy = match_count / total_with_hint if total_with_hint > 0 else 0

    return {
        "total_samples": len(subset),
        "pattern_distribution": dict(pattern_counts),
        "samples_with_label_hint": total_with_hint,
        "matches": match_count,
        "accuracy_on_hinted": round(accuracy, 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--min-distress", type=float, default=0.3)
    parser.add_argument("--behavioral-only", action="store_true")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    data_path = Path("/Users/michael/emotion-detector/data/real_user/critical.jsonl")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    samples = load_critical_samples(data_path, args.min_distress)
    print(f"Loaded {len(samples)} high-distress samples (distress >= {args.min_distress})")

    # Always run behavioral first
    print("\n=== Behavioral-Only Validation ===")
    behavioral_results = run_behavioral_validation(samples)
    print(f"Pattern distribution: {behavioral_results['pattern_distribution']}")
    print(f"Samples with label hints: {behavioral_results['samples_with_label_hint']}")
    print(f"Matches: {behavioral_results['matches']}/{behavioral_results['samples_with_label_hint']}")
    print(f"Accuracy on hinted: {behavioral_results['accuracy_on_hinted']:.1%}")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "behavioral_validation.json"), "w") as f:
        json.dump(behavioral_results, f, indent=2, ensure_ascii=False)

    if not args.behavioral_only:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\nNo ANTHROPIC_API_KEY set. Skipping LLM validation.")
            return

        print(f"\n=== LLM Validation (max {args.max_samples} samples) ===")
        llm_results = run_llm_validation(samples, api_key, args.max_samples)
        print(f"\nPattern distribution: {llm_results['pattern_distribution']}")
        print(f"Samples with label hints: {llm_results['samples_with_label_hint']}")
        print(f"Matches: {llm_results['matches']}/{llm_results['samples_with_label_hint']}")
        print(f"Accuracy on hinted: {llm_results['accuracy_on_hinted']:.1%}")

        # Print per-pattern breakdown
        print("\n--- Per-Pattern Results ---")
        for pattern in ["open", "defensive", "masked", "denial"]:
            pattern_results = [r for r in llm_results["results"]
                              if r.get("detected_pattern") == pattern and "error" not in r]
            if pattern_results:
                avg_conf = sum(r["confidence"] for r in pattern_results) / len(pattern_results)
                star = STAR_LABELS[FragilityPattern(pattern)]["star_label"]
                print(f"  {pattern:12s}: {len(pattern_results):3d} samples, avg conf={avg_conf:.3f} | {star}")

        with open(os.path.join(args.output, "llm_validation.json"), "w") as f:
            json.dump(llm_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
