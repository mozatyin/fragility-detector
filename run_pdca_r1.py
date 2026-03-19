"""PDCA R1: Baseline evaluation on hand-crafted ground-truth cases.

Plan: Establish baseline accuracy on 24 hand-crafted cases (6 per pattern).
Do:   Run current detector on all cases.
Check: Compute accuracy, confusion matrix, per-pattern F1.
Act:   Identify failure modes for R2 improvements.

Usage:
    ANTHROPIC_API_KEY=sk-or-... python run_pdca_r1.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fragility_detector.detector import FragilityDetector
from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.eval_cases import EVAL_CASES
from fragility_detector.eval_fragility import EvalResult, evaluate


def run_behavioral_eval() -> list[EvalResult]:
    """Run behavioral-only evaluation (no LLM cost)."""
    results = []
    for case in EVAL_CASES:
        texts = [t["text"] for t in case["conversation"] if t["role"] == "speaker"]
        combined = " ".join(texts)
        features = extract_features(combined)
        scores = classify_from_features(features)
        detected = max(scores, key=scores.get)

        results.append(EvalResult(
            case_id=case["id"],
            expected=case["pattern"],
            detected=detected,
            confidence=scores[detected],
            scores=scores,
            correct=detected == case["pattern"],
        ))
    return results


def run_llm_eval(api_key: str) -> list[EvalResult]:
    """Run LLM evaluation on all ground-truth cases."""
    detector = FragilityDetector(api_key)
    results = []

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

            status = "OK" if detected == case["pattern"] else "MISS"
            print(f"  [{i+1:2d}/{len(EVAL_CASES)}] {status} {case['id']:10s} "
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

    # Phase 1: Behavioral baseline
    print("=" * 60)
    print("PDCA R1: Baseline Evaluation")
    print("=" * 60)

    print("\n--- Behavioral Only ---")
    beh_results = run_behavioral_eval()
    beh_summary = evaluate(beh_results)
    beh_summary.print_report()

    # Phase 2: LLM evaluation
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nNo ANTHROPIC_API_KEY. Skipping LLM evaluation.")
        return

    print("\n--- LLM Detection ---")
    llm_results = run_llm_eval(api_key)
    llm_summary = evaluate(llm_results)
    llm_summary.print_report()

    # Save results
    save_data = {
        "round": "R1",
        "description": "Baseline evaluation on 24 hand-crafted cases",
        "behavioral": {
            "accuracy": beh_summary.accuracy,
            "per_pattern": beh_summary.per_pattern,
            "confusion": beh_summary.confusion,
        },
        "llm": {
            "accuracy": llm_summary.accuracy,
            "per_pattern": llm_summary.per_pattern,
            "confusion": llm_summary.confusion,
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
                for r in llm_results
            ],
        },
    }

    with open(output_dir / "pdca_r1.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_dir}/pdca_r1.json")

    # Identify failure modes for R2
    errors = [r for r in llm_results if not r.correct]
    if errors:
        print(f"\n{'='*60}")
        print("FAILURE MODE ANALYSIS (for R2)")
        print(f"{'='*60}")

        # Group errors by expected pattern
        from collections import defaultdict
        error_patterns = defaultdict(list)
        for r in errors:
            error_patterns[r.expected].append(r)

        for pattern, errs in sorted(error_patterns.items()):
            misclassified_as = [r.detected for r in errs]
            print(f"\n  {pattern} misclassified as: {misclassified_as}")
            for r in errs:
                print(f"    {r.case_id}: → {r.detected} (conf={r.confidence:.3f})")
                if r.evidence:
                    print(f"      evidence: {r.evidence}")


if __name__ == "__main__":
    main()
