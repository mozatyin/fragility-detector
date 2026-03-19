"""Session-level evaluation: test session detector on multi-turn conversations.

Compares:
1. Single-turn detector (last speaker message only)
2. Session-level detector (full conversation trajectory)

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_session_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fragility_detector.detector import FragilityDetector
from fragility_detector.session_detector import SessionFragilityDetector
from fragility_detector.eval_sessions import SESSION_CASES


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("Session-Level Evaluation")
    print("=" * 70)
    print(f"Cases: {len(SESSION_CASES)} multi-turn conversations\n")

    single_detector = FragilityDetector(api_key)
    session_detector = SessionFragilityDetector(api_key)

    single_results = []
    session_results = []

    for i, case in enumerate(SESSION_CASES):
        conv = case["conversation"]
        expected = case["pattern"]
        n_turns = sum(1 for t in conv if t["role"] == "speaker")

        print(f"[{i+1}/{len(SESSION_CASES)}] {case['id']} ({n_turns} speaker turns) — expected: {expected}")
        print(f"  desc: {case['description']}")

        # Single-turn: only last speaker message + previous chatter
        last_speaker_idx = max(j for j, t in enumerate(conv) if t["role"] == "speaker")
        single_window = conv[max(0, last_speaker_idx - 1):last_speaker_idx + 1]
        single_snap = single_detector.detect(single_window, turn=0)
        single_correct = single_snap.pattern.value == expected
        single_results.append({"id": case["id"], "expected": expected,
                               "detected": single_snap.pattern.value,
                               "confidence": single_snap.confidence,
                               "correct": single_correct})

        time.sleep(0.3)

        # Session-level: full conversation
        session_snap = session_detector.detect_session(conv, detect_interval=2)
        session_correct = session_snap.pattern.value == expected
        session_results.append({"id": case["id"], "expected": expected,
                                "detected": session_snap.pattern.value,
                                "confidence": session_snap.confidence,
                                "correct": session_correct,
                                "evidence": session_snap.evidence})

        s_status = "OK" if single_correct else "MISS"
        ss_status = "OK" if session_correct else "MISS"
        print(f"  Single:  {s_status} → {single_snap.pattern.value:12s} conf={single_snap.confidence:.3f}")
        print(f"  Session: {ss_status} → {session_snap.pattern.value:12s} conf={session_snap.confidence:.3f}")
        ev = session_snap.evidence
        print(f"    turns={ev.get('n_turns_analyzed','?')} peak_distress={ev.get('peak_distress','?')} "
              f"vuln_trend={ev.get('vulnerability_trend','?')} "
              f"open={ev.get('open_turns','?')} defl={ev.get('deflection_turns','?')} "
              f"humor={ev.get('humor_turns','?')} deny={ev.get('denial_turns','?')}")
        print()

        time.sleep(0.5)

    # Summary
    single_acc = sum(1 for r in single_results if r["correct"]) / len(single_results)
    session_acc = sum(1 for r in session_results if r["correct"]) / len(session_results)

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Single-turn accuracy: {single_acc:.1%} ({sum(1 for r in single_results if r['correct'])}/{len(single_results)})")
    print(f"Session-level accuracy: {session_acc:.1%} ({sum(1 for r in session_results if r['correct'])}/{len(session_results)})")
    print(f"Improvement: {session_acc - single_acc:+.1%}")

    # Per-pattern breakdown
    from collections import defaultdict
    print(f"\n{'Pattern':<12s} {'Single':>8s} {'Session':>8s}")
    print("-" * 32)
    for pattern in ["open", "defensive", "masked", "denial"]:
        s_cases = [r for r in single_results if r["expected"] == pattern]
        ss_cases = [r for r in session_results if r["expected"] == pattern]
        s_acc = sum(1 for r in s_cases if r["correct"]) / len(s_cases) if s_cases else 0
        ss_acc = sum(1 for r in ss_cases if r["correct"]) / len(ss_cases) if ss_cases else 0
        print(f"{pattern:<12s} {s_acc:>7.1%} {ss_acc:>8.1%}")

    # Detailed errors
    print(f"\nSingle-turn errors:")
    for r in single_results:
        if not r["correct"]:
            print(f"  {r['id']}: expected={r['expected']} detected={r['detected']} conf={r['confidence']:.3f}")

    print(f"\nSession errors:")
    for r in session_results:
        if not r["correct"]:
            print(f"  {r['id']}: expected={r['expected']} detected={r['detected']} conf={r['confidence']:.3f}")

    # Save
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    save_data = {
        "single_accuracy": single_acc,
        "session_accuracy": session_acc,
        "single_results": single_results,
        "session_results": session_results,
    }
    with open(output_dir / "session_eval.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_dir}/session_eval.json")


if __name__ == "__main__":
    main()
