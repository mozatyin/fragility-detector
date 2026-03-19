"""Real Literary Dialogue Evaluation.

Tests the detector on ACTUAL character quotes extracted from novels and TV scripts.
No synthetic generation — these are real words from known characters with known patterns.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_real_literary_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from fragility_detector.detector import FragilityDetector


# Characters and their patterns
CHARACTERS = {
    "chandler_bing": {"pattern": "masked", "source": "Friends"},
    "don_draper": {"pattern": "denial", "source": "Mad Men"},
    "scrooge": {"pattern": "denial", "source": "A Christmas Carol"},
    "elinor_dashwood": {"pattern": "defensive", "source": "Sense & Sensibility"},
    "jane_eyre": {"pattern": "open", "source": "Jane Eyre"},
    "mr_darcy": {"pattern": "defensive", "source": "Pride & Prejudice"},
    "pip": {"pattern": "open", "source": "Great Expectations"},
    "catherine_earnshaw": {"pattern": "open", "source": "Wuthering Heights"},
    "heathcliff": {"pattern": "denial", "source": "Wuthering Heights"},
    "miss_havisham": {"pattern": "denial", "source": "Great Expectations"},
}

# Minimum quote length to be meaningful for detection
MIN_QUOTE_LENGTH = 30  # characters
MAX_QUOTES_PER_CHAR = 20  # limit API calls


def load_quotes(char_name: str, data_dir: Path) -> list[str]:
    """Load and filter quotes for a character."""
    path = data_dir / f"{char_name}_quotes.txt"
    if not path.exists():
        return []

    quotes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if len(line) >= MIN_QUOTE_LENGTH:
                quotes.append(line)
    return quotes


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    data_dir = Path("data/literary")
    detector = FragilityDetector(api_key)

    print("Real Literary Dialogue Evaluation")
    print("=" * 60)

    all_results = []

    for char_name, info in CHARACTERS.items():
        quotes = load_quotes(char_name, data_dir)
        if not quotes:
            print(f"\n{char_name}: NO QUOTES FOUND")
            continue

        # Sample up to MAX_QUOTES_PER_CHAR
        sample = quotes[:MAX_QUOTES_PER_CHAR]
        expected = info["pattern"]

        print(f"\n{char_name} ({info['source']}) — expected: {expected} — {len(sample)} quotes")

        correct = 0
        pattern_counts = Counter()

        for i, quote in enumerate(sample):
            conv = [{"role": "speaker", "text": quote}]
            try:
                snap = detector.detect(conv, turn=0)
                detected = snap.pattern.value
                conf = snap.confidence

                # For very low confidence, we might consider it "neutral"
                is_neutral = conf < 0.2

                ok = detected == expected and not is_neutral
                if ok:
                    correct += 1
                pattern_counts[detected] += 1

                if i < 3 or not ok:  # Show first 3 + errors
                    status = "OK" if ok else "MISS"
                    print(f"  [{i+1:2d}] {status} → {detected:12s} conf={conf:.3f} | {quote[:60]}...")

            except Exception as e:
                print(f"  [{i+1:2d}] ERROR: {e}")

            time.sleep(0.3)

        acc = correct / len(sample) if sample else 0
        print(f"  RESULT: {correct}/{len(sample)} = {acc:.1%}")
        print(f"  Distribution: {dict(pattern_counts)}")

        all_results.append({
            "character": char_name,
            "pattern": expected,
            "source": info["source"],
            "n_quotes": len(sample),
            "correct": correct,
            "accuracy": round(acc, 3),
            "distribution": dict(pattern_counts),
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_quotes = sum(r["n_quotes"] for r in all_results)
    total_correct = sum(r["correct"] for r in all_results)
    overall_acc = total_correct / total_quotes if total_quotes > 0 else 0
    print(f"Overall: {total_correct}/{total_quotes} = {overall_acc:.1%}")

    # Per-pattern
    print(f"\n{'Pattern':<12s} {'Acc':>8s} {'N':>4s}")
    print("-" * 28)
    for pattern in ["open", "defensive", "masked", "denial"]:
        p_results = [r for r in all_results if r["pattern"] == pattern]
        if not p_results:
            continue
        p_total = sum(r["n_quotes"] for r in p_results)
        p_correct = sum(r["correct"] for r in p_results)
        p_acc = p_correct / p_total if p_total > 0 else 0
        print(f"{pattern:<12s} {p_acc:>7.1%} {p_total:>4d}")

    # Per-character
    print(f"\n{'Character':<20s} {'Pattern':<12s} {'Acc':>8s}")
    print("-" * 44)
    for r in all_results:
        print(f"{r['character']:<20s} {r['pattern']:<12s} {r['accuracy']:>7.1%}")

    # Save
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "real_literary_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/real_literary_eval.json")


if __name__ == "__main__":
    main()
