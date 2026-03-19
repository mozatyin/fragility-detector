"""Filtered Literary Eval: only test quotes where vulnerability is relevant.

Step 1: LLM filters quotes to those showing vulnerability-relevant content
Step 2: Detector classifies filtered quotes
This matches real product behavior — only detect when emotional context exists.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_filtered_literary_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from fragility_detector.api_retry import make_client, retry_api_call
from fragility_detector.detector import FragilityDetector

CHARACTERS = {
    "chandler_bing": {"pattern": "masked", "source": "Friends"},
    "don_draper": {"pattern": "denial", "source": "Mad Men"},
    "scrooge": {"pattern": "denial", "source": "A Christmas Carol"},
    "elinor_dashwood": {"pattern": "defensive", "source": "Sense & Sensibility"},
    "jane_eyre": {"pattern": "open", "source": "Jane Eyre"},
    "mr_darcy": {"pattern": "defensive", "source": "Pride & Prejudice"},
    "pip": {"pattern": "open", "source": "Great Expectations"},
    "heathcliff": {"pattern": "denial", "source": "Wuthering Heights"},
}

MIN_QUOTE_LENGTH = 50
MAX_QUOTES = 30


def load_quotes(char_name: str) -> list[str]:
    path = Path(f"data/literary/{char_name}_quotes.txt")
    if not path.exists():
        return []
    quotes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and len(line) >= MIN_QUOTE_LENGTH:
                quotes.append(line)
    return quotes[:MAX_QUOTES]


def filter_vulnerability_relevant(quotes: list[str], api_key: str) -> list[str]:
    """Use LLM to filter quotes that involve emotional/vulnerability topics."""
    client = make_client(api_key)

    # Batch filter in one call
    numbered = "\n".join(f"{i+1}. {q[:150]}" for i, q in enumerate(quotes))

    response = retry_api_call(
        lambda: client.messages.create(
            model="anthropic/claude-sonnet-4",
            max_tokens=500,
            temperature=0.0,
            system=(
                "You are filtering dialogue quotes. Return ONLY the numbers of quotes that involve "
                "emotional content — pain, vulnerability, relationships, loss, fear, humor about pain, "
                "denial of feelings, deflection from emotional topics, or any emotional self-disclosure. "
                "Exclude quotes that are purely factual, mundane, or about non-emotional topics. "
                "Return as comma-separated numbers only. Example: 1,3,5,8"
            ),
            messages=[{"role": "user", "content": f"Which quotes have emotional content?\n\n{numbered}"}],
        )
    )

    text = response.content[0].text.strip()
    try:
        indices = [int(x.strip()) - 1 for x in text.split(",") if x.strip().isdigit()]
        return [quotes[i] for i in indices if 0 <= i < len(quotes)]
    except (ValueError, IndexError):
        return quotes  # fallback: use all


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    detector = FragilityDetector(api_key)

    print("Filtered Literary Eval (vulnerability-relevant quotes only)")
    print("=" * 60)

    all_results = []

    for char_name, info in CHARACTERS.items():
        quotes = load_quotes(char_name)
        if not quotes:
            continue

        expected = info["pattern"]
        print(f"\n{char_name} ({info['source']}) — expected: {expected}")
        print(f"  Total quotes: {len(quotes)}")

        # Step 1: Filter
        filtered = filter_vulnerability_relevant(quotes, api_key)
        print(f"  Vulnerability-relevant: {len(filtered)}")

        if not filtered:
            print("  No relevant quotes found, skipping")
            continue

        # Step 2: Detect
        correct = 0
        pattern_counts = Counter()

        for i, quote in enumerate(filtered[:15]):  # max 15 per character
            conv = [{"role": "speaker", "text": quote}]
            snap = detector.detect(conv, turn=0)
            detected = snap.pattern.value
            ok = detected == expected
            if ok:
                correct += 1
            pattern_counts[detected] += 1

            status = "OK" if ok else "MISS"
            if i < 3 or not ok:
                print(f"  [{i+1:2d}] {status} → {detected:12s} conf={snap.confidence:.3f} | {quote[:65]}...")
            time.sleep(0.3)

        n = min(len(filtered), 15)
        acc = correct / n if n > 0 else 0
        print(f"  RESULT: {correct}/{n} = {acc:.1%}  dist={dict(pattern_counts)}")

        all_results.append({
            "character": char_name, "pattern": expected,
            "total_quotes": len(quotes), "filtered_quotes": len(filtered),
            "tested": n, "correct": correct, "accuracy": round(acc, 3),
            "distribution": dict(pattern_counts),
        })

    # Summary
    print("\n" + "=" * 60)
    total = sum(r["tested"] for r in all_results)
    correct = sum(r["correct"] for r in all_results)
    print(f"Overall: {correct}/{total} = {correct/total:.1%}")

    for pat in ["open", "defensive", "masked", "denial"]:
        pr = [r for r in all_results if r["pattern"] == pat]
        if not pr:
            continue
        pt = sum(r["tested"] for r in pr)
        pc = sum(r["correct"] for r in pr)
        print(f"  {pat:12s}: {pc}/{pt} = {pc/pt:.1%}")

    print(f"\n{'Character':<20s} {'Pattern':<12s} {'Filtered':>8s} {'Acc':>8s}")
    print("-" * 52)
    for r in all_results:
        print(f"{r['character']:<20s} {r['pattern']:<12s} {r['filtered_quotes']:>8d} {r['accuracy']:>7.1%}")

    with open("results/filtered_literary_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
