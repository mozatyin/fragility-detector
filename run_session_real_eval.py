"""Real-data session-level evaluation.

Picks multi-turn sessions from critical.jsonl, runs session detector,
compares with single-turn detector on last message only.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_session_real_eval.py [--max-sessions 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from fragility_detector.detector import FragilityDetector
from fragility_detector.session_detector import SessionFragilityDetector


def load_sessions(path: Path, min_messages: int = 3) -> dict[str, list[dict]]:
    """Load multi-turn sessions from critical.jsonl."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sessions[d["session_id"]].append(d)

    # Filter and sort
    multi = {}
    for sid, msgs in sessions.items():
        if len(msgs) >= min_messages:
            msgs.sort(key=lambda x: x["turn"])
            multi[sid] = msgs
    return multi


def build_session_conversation(messages: list[dict]) -> list[dict]:
    """Build full conversation from the LAST message's context + all user messages.

    Uses the context field from the last message (which has the most history),
    then fills in user messages from the data.
    """
    last_msg = messages[-1]
    context = last_msg.get("context", "")

    turns = []
    for line in context.strip().split("\n"):
        line = line.strip()
        if line.startswith("User:"):
            turns.append({"role": "speaker", "text": line[5:].strip()})
        elif line.startswith("Assistant:"):
            turns.append({"role": "chatter", "text": line[10:].strip()})

    if not turns:
        # Fallback: just user messages
        for msg in messages:
            turns.append({"role": "speaker", "text": msg["user_text"]})

    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument("--min-messages", type=int, default=4)
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    data_path = Path("/Users/michael/emotion-detector/data/real_user/critical.jsonl")
    sessions = load_sessions(data_path, args.min_messages)
    print(f"Found {len(sessions)} sessions with {args.min_messages}+ messages")

    # Pick sessions with highest average distress (more interesting)
    ranked = sorted(
        sessions.items(),
        key=lambda x: sum(m.get("prod_distress", 0) for m in x[1]) / len(x[1]),
        reverse=True,
    )[:args.max_sessions]

    single_detector = FragilityDetector(api_key)
    session_detector = SessionFragilityDetector(api_key)

    results = []
    print(f"\nRunning on top {len(ranked)} sessions by avg distress:\n")

    for i, (sid, msgs) in enumerate(ranked):
        avg_dist = sum(m.get("prod_distress", 0) for m in msgs) / len(msgs)
        n_msgs = len(msgs)
        print(f"[{i+1}/{len(ranked)}] Session {sid[:8]}... ({n_msgs} msgs, avg_dist={avg_dist:.2f})")

        # Build conversation
        conversation = build_session_conversation(msgs)
        n_speaker = sum(1 for t in conversation if t["role"] == "speaker")
        print(f"  Conversation: {len(conversation)} turns ({n_speaker} speaker)")

        # Show first and last speaker messages
        speaker_msgs = [t["text"] for t in conversation if t["role"] == "speaker"]
        if speaker_msgs:
            print(f"  First: {speaker_msgs[0][:60]}...")
            if len(speaker_msgs) > 1:
                print(f"  Last:  {speaker_msgs[-1][:60]}...")

        # Single-turn: last speaker message only
        last_speaker_idx = max(
            (j for j, t in enumerate(conversation) if t["role"] == "speaker"),
            default=0,
        )
        single_window = conversation[max(0, last_speaker_idx - 1):last_speaker_idx + 1]
        single_snap = single_detector.detect(single_window, turn=0)

        time.sleep(0.3)

        # Session-level
        session_snap = session_detector.detect_session(conversation, detect_interval=2)

        print(f"  Single:  {single_snap.pattern.value:12s} conf={single_snap.confidence:.3f}")
        print(f"  Session: {session_snap.pattern.value:12s} conf={session_snap.confidence:.3f}")

        ev = session_snap.evidence
        print(f"    turns={ev.get('n_turns_analyzed','?')} vuln_trend={ev.get('vulnerability_trend','?')} "
              f"open={ev.get('open_turns','?')} defl={ev.get('deflection_turns','?')} "
              f"humor={ev.get('humor_turns','?')} deny={ev.get('denial_turns','?')}")

        # Agreement
        agree = single_snap.pattern.value == session_snap.pattern.value
        print(f"  Agreement: {'YES' if agree else 'NO — patterns differ'}")
        print()

        results.append({
            "session_id": sid,
            "n_messages": n_msgs,
            "avg_distress": round(avg_dist, 2),
            "prod_labels": [l for m in msgs for l in m.get("prod_labels", [])],
            "single": {
                "pattern": single_snap.pattern.value,
                "confidence": round(single_snap.confidence, 3),
            },
            "session": {
                "pattern": session_snap.pattern.value,
                "confidence": round(session_snap.confidence, 3),
                "evidence": session_snap.evidence,
            },
            "agree": agree,
            "first_text": speaker_msgs[0][:100] if speaker_msgs else "",
            "last_text": speaker_msgs[-1][:100] if len(speaker_msgs) > 1 else "",
        })

        time.sleep(0.5)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    single_counts = Counter(r["single"]["pattern"] for r in results)
    session_counts = Counter(r["session"]["pattern"] for r in results)
    agree_count = sum(1 for r in results if r["agree"])

    print(f"Agreement: {agree_count}/{len(results)} ({agree_count/len(results):.1%})")
    print(f"\nSingle-turn distribution: {dict(single_counts)}")
    print(f"Session-level distribution: {dict(session_counts)}")

    # Show disagreements
    disagreements = [r for r in results if not r["agree"]]
    if disagreements:
        print(f"\n--- Disagreements ({len(disagreements)}) ---")
        for r in disagreements:
            print(f"  {r['session_id'][:8]}... single={r['single']['pattern']} session={r['session']['pattern']}")
            print(f"    first: {r['first_text'][:70]}...")
            print(f"    last:  {r['last_text'][:70]}...")

    # Save
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "session_real_eval.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}session_real_eval.json")


if __name__ == "__main__":
    main()
