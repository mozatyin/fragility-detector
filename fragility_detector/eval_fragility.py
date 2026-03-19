"""Evaluation framework for fragility pattern detection.

Metrics: accuracy, per-pattern precision/recall/F1, confusion matrix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result of a single evaluation case."""
    case_id: str
    expected: str
    detected: str
    confidence: float
    scores: dict[str, float]
    correct: bool
    evidence: dict = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Aggregate evaluation summary."""
    total: int
    correct: int
    accuracy: float
    per_pattern: dict[str, dict[str, float]]  # pattern -> {precision, recall, f1, support}
    confusion: dict[str, dict[str, int]]       # expected -> {detected -> count}
    results: list[EvalResult]

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"Fragility Detection Evaluation Report")
        print(f"{'='*60}")
        print(f"Total: {self.total} | Correct: {self.correct} | Accuracy: {self.accuracy:.1%}")

        print(f"\n{'Pattern':<12s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>4s}")
        print("-" * 36)
        for pattern in ["open", "defensive", "masked", "denial"]:
            m = self.per_pattern.get(pattern, {})
            print(f"{pattern:<12s} {m.get('precision', 0):>6.1%} {m.get('recall', 0):>6.1%} "
                  f"{m.get('f1', 0):>6.1%} {m.get('support', 0):>4.0f}")

        print(f"\nConfusion Matrix:")
        patterns = ["open", "defensive", "masked", "denial"]
        print(f"{'Expected':<12s} | " + " | ".join(f"{p:>10s}" for p in patterns))
        print("-" * 60)
        for exp in patterns:
            row = self.confusion.get(exp, {})
            cells = " | ".join(f"{row.get(p, 0):>10d}" for p in patterns)
            print(f"{exp:<12s} | {cells}")

        # Errors
        errors = [r for r in self.results if not r.correct]
        if errors:
            print(f"\n--- Errors ({len(errors)}) ---")
            for r in errors:
                print(f"  {r.case_id}: expected={r.expected} detected={r.detected} conf={r.confidence:.3f}")


def evaluate(results: list[EvalResult]) -> EvalSummary:
    """Compute evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0

    # Per-pattern metrics
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()

    for r in results:
        support[r.expected] += 1
        if r.correct:
            tp[r.expected] += 1
        else:
            fp[r.detected] += 1
            fn[r.expected] += 1

    per_pattern = {}
    for pattern in ["open", "defensive", "masked", "denial"]:
        p = tp[pattern] / (tp[pattern] + fp[pattern]) if (tp[pattern] + fp[pattern]) > 0 else 0
        r = tp[pattern] / (tp[pattern] + fn[pattern]) if (tp[pattern] + fn[pattern]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_pattern[pattern] = {
            "precision": p, "recall": r, "f1": f1, "support": support[pattern],
        }

    # Confusion matrix
    confusion = defaultdict(Counter)
    for r in results:
        confusion[r.expected][r.detected] += 1

    return EvalSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        per_pattern=per_pattern,
        confusion={k: dict(v) for k, v in confusion.items()},
        results=results,
    )
