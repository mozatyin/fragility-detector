"""Literary Character Evaluation: generate conversations with fictional characters,
then detect their fragility patterns.

This creates a large, diverse evaluation set from known ground truth:
- 12 characters × 6 scenarios = 72 conversations
- Ground truth = character's known fragility pattern
- Tests both single-turn (last message) and session-level detection

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_literary_eval.py [--n-turns 4] [--max-chars 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from fragility_detector.literary_characters import (
    ALL_CHARACTERS, VULNERABILITY_SCENARIOS, CHARACTERS_BY_PATTERN,
)
from fragility_detector.speaker import FragilitySpeaker, Chatter
from fragility_detector.detector import FragilityDetector
from fragility_detector.session_detector import SessionFragilityDetector


def run_conversation(
    character, scenario, api_key: str, n_turns: int = 4,
) -> list[dict]:
    """Generate a multi-turn conversation with a literary character."""
    speaker = FragilitySpeaker(character, api_key)
    chatter = Chatter(api_key, scenario_prompt=scenario["context"])

    conversation = []
    for turn in range(n_turns):
        # Chatter speaks first
        if turn == 0:
            chatter_text = scenario["prompt"]
        else:
            chatter_text = chatter.generate(conversation, turn, n_turns)
        conversation.append({"role": "chatter", "text": chatter_text})

        # Speaker (character) responds
        speaker_text = speaker.generate(conversation, turn)
        conversation.append({"role": "speaker", "text": speaker_text})

        time.sleep(0.3)

    return conversation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-turns", type=int, default=4, help="Turns per conversation")
    parser.add_argument("--max-chars", type=int, default=12, help="Max characters to test")
    parser.add_argument("--max-scenarios", type=int, default=3, help="Max scenarios per character")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    characters = ALL_CHARACTERS[:args.max_chars]
    scenarios = VULNERABILITY_SCENARIOS[:args.max_scenarios]

    total = len(characters) * len(scenarios)
    print(f"Literary Character Evaluation")
    print(f"Characters: {len(characters)} × Scenarios: {len(scenarios)} = {total} conversations")
    print(f"Turns per conversation: {args.n_turns}")
    print()

    single_detector = FragilityDetector(api_key)
    session_detector = SessionFragilityDetector(api_key)

    results = []
    single_correct = 0
    session_correct = 0

    for i, char in enumerate(characters):
        for j, scenario in enumerate(scenarios):
            idx = i * len(scenarios) + j + 1
            print(f"[{idx:2d}/{total}] {char.name:20s} × {scenario['id']:20s} (exp: {char.fragility_pattern})")

            # Generate conversation
            try:
                conv = run_conversation(char, scenario, api_key, args.n_turns)
            except Exception as e:
                print(f"  ERROR generating: {e}")
                results.append({
                    "character": char.name, "scenario": scenario["id"],
                    "expected": char.fragility_pattern, "error": str(e),
                })
                continue

            # Show conversation
            for t in conv:
                role = "CHAR" if t["role"] == "speaker" else "FRND"
                print(f"  [{role}] {t['text'][:80]}{'...' if len(t['text']) > 80 else ''}")

            # Single-turn detection (last speaker message)
            last_idx = max(k for k, t in enumerate(conv) if t["role"] == "speaker")
            single_window = conv[max(0, last_idx - 1):last_idx + 1]
            single_snap = single_detector.detect(single_window, turn=0)

            time.sleep(0.3)

            # Session detection
            session_snap = session_detector.detect_session(conv, detect_interval=2)

            s_ok = single_snap.pattern.value == char.fragility_pattern
            ss_ok = session_snap.pattern.value == char.fragility_pattern
            if s_ok: single_correct += 1
            if ss_ok: session_correct += 1

            s_status = "OK" if s_ok else "MISS"
            ss_status = "OK" if ss_ok else "MISS"
            print(f"  Single:  {s_status} → {single_snap.pattern.value:12s} conf={single_snap.confidence:.3f}")
            print(f"  Session: {ss_status} → {session_snap.pattern.value:12s} conf={session_snap.confidence:.3f}")
            print()

            results.append({
                "character": char.name,
                "scenario": scenario["id"],
                "expected": char.fragility_pattern,
                "conversation": conv,
                "single": {
                    "pattern": single_snap.pattern.value,
                    "confidence": round(single_snap.confidence, 3),
                    "correct": s_ok,
                },
                "session": {
                    "pattern": session_snap.pattern.value,
                    "confidence": round(session_snap.confidence, 3),
                    "correct": ss_ok,
                    "evidence": session_snap.evidence,
                },
            })

            time.sleep(0.5)

    # Summary
    valid = [r for r in results if "error" not in r]
    n = len(valid)

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total conversations: {n}")
    print(f"Single-turn: {single_correct}/{n} = {single_correct/n:.1%}")
    print(f"Session:     {session_correct}/{n} = {session_correct/n:.1%}")

    # Per-pattern
    print(f"\n{'Pattern':<12s} {'Single':>8s} {'Session':>8s} {'N':>4s}")
    print("-" * 36)
    for pattern in ["open", "defensive", "masked", "denial"]:
        p_results = [r for r in valid if r["expected"] == pattern]
        if not p_results:
            continue
        s_acc = sum(1 for r in p_results if r["single"]["correct"]) / len(p_results)
        ss_acc = sum(1 for r in p_results if r["session"]["correct"]) / len(p_results)
        print(f"{pattern:<12s} {s_acc:>7.1%} {ss_acc:>8.1%} {len(p_results):>4d}")

    # Per-character
    print(f"\n{'Character':<20s} {'Pattern':<12s} {'Single':>8s} {'Session':>8s}")
    print("-" * 52)
    for char in characters:
        c_results = [r for r in valid if r["character"] == char.name]
        if not c_results:
            continue
        s_acc = sum(1 for r in c_results if r["single"]["correct"]) / len(c_results)
        ss_acc = sum(1 for r in c_results if r["session"]["correct"]) / len(c_results)
        print(f"{char.name:<20s} {char.fragility_pattern:<12s} {s_acc:>7.1%} {ss_acc:>8.1%}")

    # Save
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "literary_eval.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}literary_eval.json")


if __name__ == "__main__":
    main()
