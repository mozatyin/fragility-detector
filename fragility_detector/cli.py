"""CLI entry point for fragility detection.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python -m fragility_detector --smoke
    ANTHROPIC_API_KEY=sk-or-... python -m fragility_detector --detect "I feel like giving up"
    ANTHROPIC_API_KEY=sk-or-... python -m fragility_detector --detect-file input.jsonl
"""

import argparse
import json
import os
import sys

from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.models import FragilityPattern


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Fragility Pattern Detector")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with sample texts")
    parser.add_argument("--detect", type=str, help="Detect fragility pattern from a single text")
    parser.add_argument("--detect-file", type=str, help="Detect from JSONL file of conversations")
    parser.add_argument("--features-only", action="store_true", help="Only extract behavioral features (no LLM)")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    return parser.parse_args(args)


SMOKE_SAMPLES = [
    {
        "label": "Open",
        "conversation": [
            {"role": "speaker", "text": "I feel like I'm falling apart. I don't know how much longer I can take this. Everything hurts and I'm scared."},
        ],
    },
    {
        "label": "Defensive",
        "conversation": [
            {"role": "chatter", "text": "How are you feeling about the breakup?"},
            {"role": "speaker", "text": "I'm fine, it's not a big deal. Anyway, did you see that new movie? Let's talk about something else."},
        ],
    },
    {
        "label": "Masked",
        "conversation": [
            {"role": "chatter", "text": "I heard you got laid off..."},
            {"role": "speaker", "text": "haha yeah I got dumped by my job AND my girlfriend in the same week. I should get a loyalty card for losses at this point lol. At least I'm consistent 😂"},
        ],
    },
    {
        "label": "Denial",
        "conversation": [
            {"role": "chatter", "text": "That must have been really hard for you."},
            {"role": "speaker", "text": "I don't need anyone's sympathy. I'm strong, I can handle anything. Feelings are just weakness. I dealt with it and moved on."},
        ],
    },
]


def main():
    args = parse_args()

    if args.features_only and args.detect:
        features = extract_features(args.detect)
        scores = classify_from_features(features)
        print("Behavioral features:")
        for k, v in sorted(features.items()):
            print(f"  {k}: {v:.4f}")
        print("\nPattern scores (behavioral only):")
        for k, v in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.3f}")
        return

    if args.smoke:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Fallback: behavioral features only
            print("No API key found. Running behavioral-only smoke test.\n")
            for sample in SMOKE_SAMPLES:
                texts = [t["text"] for t in sample["conversation"] if t["role"] == "speaker"]
                combined = " ".join(texts)
                features = extract_features(combined)
                scores = classify_from_features(features)
                best = max(scores, key=scores.get)
                print(f"Expected: {sample['label']:12s} | Detected: {best:12s} | Scores: {json.dumps({k: round(v, 3) for k, v in scores.items()})}")
            return

        from fragility_detector.detector import FragilityDetector
        from fragility_detector.star_map import generate_star_map

        detector = FragilityDetector(api_key)
        print("Running smoke test with LLM detection...\n")
        for sample in SMOKE_SAMPLES:
            snapshot = detector.detect(sample["conversation"], turn=0)
            star = generate_star_map(snapshot)
            print(f"Expected: {sample['label']:12s} | Detected: {snapshot.pattern.value:12s} | "
                  f"Confidence: {snapshot.confidence:.3f} | Star: {star.star_label}")
            print(f"  Scores: { {k: round(v, 3) for k, v in snapshot.pattern_scores.items()} }")
            print(f"  Evidence: {snapshot.evidence}")
            print()
        return

    if args.detect:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)

        from fragility_detector.detector import FragilityDetector
        from fragility_detector.star_map import generate_star_map

        detector = FragilityDetector(api_key)
        conversation = [{"role": "speaker", "text": args.detect}]
        snapshot = detector.detect(conversation, turn=0)
        star = generate_star_map(snapshot)
        print(f"Pattern: {snapshot.pattern.value}")
        print(f"Star: {star.star_label} ({star.star_sublabel})")
        print(f"Confidence: {snapshot.confidence:.3f}")
        print(f"Scores: {json.dumps({k: round(v, 3) for k, v in snapshot.pattern_scores.items()}, indent=2)}")
        return

    print("Use --smoke, --detect TEXT, or --detect-file FILE. See --help for details.")


if __name__ == "__main__":
    main()
