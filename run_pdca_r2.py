"""PDCA R2: Disambiguation improvements.

Plan:
  - Fix denial over-detection (was inflated by (1-vuln)*0.2 on neutral text)
  - Add disambiguation rules to LLM prompt (defensive vs denial, masked vs genuine humor)
  - Add distress threshold for insufficient signal detection
  - Add 6 hard/ambiguous cases to eval set (now 30 total)

Do:   Run improved detector on expanded eval set.
Check: Compare accuracy vs R1 baseline (95.8%).
Act:   Identify remaining failure modes for R3.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_pdca_r2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fragility_detector.detector import FragilityDetector
from fragility_detector.eval_cases import EVAL_CASES
from fragility_detector.eval_fragility import EvalResult, evaluate


def run_llm_eval(api_key: str) -> list[EvalResult]:
    """Run LLM evaluation on all ground-truth cases."""
    detector = FragilityDetector(api_key)
    results = []

    # Separate original (R1) cases from new R2 cases
    r1_cases = [c for c in EVAL_CASES if not c["id"].startswith("hard_")]
    r2_cases = [c for c in EVAL_CASES if c["id"].startswith("hard_")]

    print(f"Eval set: {len(r1_cases)} original + {len(r2_cases)} new R2 hard cases = {len(EVAL_CASES)} total\n")

    for i, case in enumerate(EVAL_CASES):
        try:
            snapshot = detector.detect(case["conversation"], turn=0)
            detected = snapshot.pattern.value

            results.append(EvalResult(
                case_id=case["id"],
                expected=case["pattern"],
                detected=detected,
                confidence=snapshot.confidence,
                scores={k: round(v, 3) for k, v in snapshot.pattern_scores.items()},
                correct=detected == case["pattern"],
                evidence=snapshot.evidence,
            ))

            tag = "R2" if case["id"].startswith("hard_") else "  "
            status = "OK" if detected == case["pattern"] else "MISS"
            print(f"  [{i+1:2d}/{len(EVAL_CASES)}] {tag} {status} {case['id']:20s} "
                  f"exp={case['pattern']:12s} det={detected:12s} conf={snapshot.confidence:.3f}")

        except Exception as e:
            print(f"  [{i+1:2d}/{len(EVAL_CASES)}] ERROR {case['id']}: {e}")
            results.append(EvalResult(
                case_id=case["id"],
                expected=case["pattern"],
                detected="error",
                confidence=0,
                scores={},
                correct=False,
            ))

        time.sleep(0.5)

    return results


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("PDCA R2: Disambiguation Improvements")
    print("=" * 60)
    print("Changes:")
    print("  1. Disambiguation rules in LLM prompt (defensive vs denial, masked vs genuine humor)")
    print("  2. Fixed denial derive formula (removed (1-vuln)*0.2 inflation)")
    print("  3. Distress threshold for insufficient signal detection")
    print("  4. +6 hard/ambiguous test cases (30 total)")
    print()

    # Load R1 baseline
    r1_path = output_dir / "pdca_r1.json"
    r1_accuracy = None
    if r1_path.exists():
        with open(r1_path) as f:
            r1_data = json.load(f)
            r1_accuracy = r1_data.get("llm", {}).get("accuracy")
            print(f"R1 baseline accuracy: {r1_accuracy:.1%} (24 cases)")

    # Run R2
    print(f"\n--- R2 LLM Detection ---")
    results = run_llm_eval(api_key)

    # Full eval
    summary = evaluate(results)
    summary.print_report()

    # Separate eval on original R1 cases only (for fair comparison)
    r1_only = [r for r in results if not r.case_id.startswith("hard_")]
    r1_summary = evaluate(r1_only)
    print(f"\n--- R1 Cases Only (fair comparison) ---")
    print(f"R1 baseline: {r1_accuracy:.1%}" if r1_accuracy else "R1 baseline: N/A")
    print(f"R2 on R1 cases: {r1_summary.accuracy:.1%} ({r1_summary.correct}/{r1_summary.total})")

    # R2 hard cases only
    r2_only = [r for r in results if r.case_id.startswith("hard_")]
    if r2_only:
        r2_summary = evaluate(r2_only)
        print(f"\n--- R2 Hard Cases Only ---")
        print(f"R2 hard cases: {r2_summary.accuracy:.1%} ({r2_summary.correct}/{r2_summary.total})")

    # Save
    save_data = {
        "round": "R2",
        "changes": [
            "Disambiguation rules in LLM prompt",
            "Fixed denial derive formula",
            "Distress threshold for insufficient signal",
            "+6 hard/ambiguous test cases",
        ],
        "total_cases": len(EVAL_CASES),
        "overall": {
            "accuracy": summary.accuracy,
            "per_pattern": summary.per_pattern,
            "confusion": summary.confusion,
        },
        "r1_cases_only": {
            "accuracy": r1_summary.accuracy,
            "r1_baseline": r1_accuracy,
        },
        "r2_hard_cases": {
            "accuracy": evaluate(r2_only).accuracy if r2_only else None,
        },
        "results": [
            {
                "case_id": r.case_id,
                "expected": r.expected,
                "detected": r.detected,
                "confidence": r.confidence,
                "scores": r.scores,
                "correct": r.correct,
                "evidence": r.evidence,
            }
            for r in results
        ],
    }

    with open(output_dir / "pdca_r2.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_dir}/pdca_r2.json")


if __name__ == "__main__":
    main()
