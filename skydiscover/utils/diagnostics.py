"""Deterministic diagnosis of why a child under-performed its parent.

No LLM calls and no I/O -- this only compares a child's metrics against its
parent's and renders the per-metric deltas as text.  Two consumers:

- ``DiscoveryController._attach_diagnostics`` stores
  :func:`build_regression_report` on the child as a ``regression_report``
  artifact, so it persists in checkpoints, shows on the dashboard, and reaches
  the prompt when that program is later selected as a parent.
- ``DefaultContextBuilder._determine_outcome`` uses
  :func:`describe_metric_deltas` to turn the bare "Regression in
  combined_score" outcome line into one that names *which* metric moved.

The second path matters more than the first: artifacts render only for the
program currently being mutated, so a regressed child that the search never
re-selects would otherwise never show its report to anybody.
"""

from typing import Any, Dict, List, NamedTuple, Optional

from skydiscover.utils.metrics import get_score, is_numeric_metric

#: The metric that drives selection.  Always reported first when present.
FITNESS_KEY = "combined_score"

#: Metrics that describe how an evaluation went rather than how good the
#: solution is.  Reporting a delta on them is noise.
_NON_FITNESS_KEYS = frozenset({"timeout", "validity", "error", "error_message"})


class MetricDelta(NamedTuple):
    """One metric's movement from parent to child."""

    name: str
    child: float
    parent: float
    delta: float
    #: Change relative to the parent, in percent.  ``None`` when the parent
    #: value was zero and a ratio would divide by zero.
    pct: Optional[float]


def metric_deltas(
    child_metrics: Optional[Dict[str, Any]],
    parent_metrics: Optional[Dict[str, Any]],
) -> List[MetricDelta]:
    """Per-metric movement for every metric numeric on *both* sides.

    Ordered fitness-first, then by largest relative change, so truncating the
    list keeps the interesting entries.  ``bool`` is excluded via
    ``is_numeric_metric``, so ``timeout: True`` cannot masquerade as a delta.
    """
    child_metrics = child_metrics or {}
    parent_metrics = parent_metrics or {}

    deltas: List[MetricDelta] = []
    for name, value in child_metrics.items():
        if name in _NON_FITNESS_KEYS:
            continue
        previous = parent_metrics.get(name)
        if not is_numeric_metric(value) or not is_numeric_metric(previous):
            continue
        child_value, parent_value = float(value), float(previous)
        delta = child_value - parent_value
        pct = (delta / abs(parent_value) * 100.0) if parent_value else None
        deltas.append(MetricDelta(name, child_value, parent_value, delta, pct))

    def rank(d: MetricDelta) -> tuple:
        magnitude = abs(d.pct) if d.pct is not None else abs(d.delta)
        return (d.name != FITNESS_KEY, -magnitude, d.name)

    return sorted(deltas, key=rank)


def _format_delta(d: MetricDelta) -> str:
    if d.delta == 0:
        return f"{d.name} unchanged ({d.child:.4g})"
    movement = f"{d.name} {d.delta:+.4g} ({d.parent:.4g} -> {d.child:.4g}"
    if d.pct is not None:
        movement += f", {d.pct:+.1f}%"
    return movement + ")"


def describe_metric_deltas(
    child_metrics: Optional[Dict[str, Any]],
    parent_metrics: Optional[Dict[str, Any]],
    *,
    limit: int = 4,
) -> str:
    """One-line summary of the largest metric movements.

    Returns ``""`` when no metric is comparable across the two, which is the
    normal case for the very first program in a run.
    """
    deltas = metric_deltas(child_metrics, parent_metrics)[:limit]
    return "; ".join(_format_delta(d) for d in deltas)


def build_regression_report(
    child_metrics: Optional[Dict[str, Any]],
    parent_metrics: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Explain a child that failed to beat its parent.

    Returns ``None`` when the child improved (an improvement needs no
    explaining) or when there is nothing comparable to report.  Fitness is
    taken from ``get_score``, so this agrees with the selection rule even for
    evaluators that never emit ``combined_score``.
    """
    deltas = metric_deltas(child_metrics, parent_metrics)
    if not deltas:
        return None

    child_score = get_score(child_metrics or {})
    parent_score = get_score(parent_metrics or {})
    if child_score > parent_score:
        return None

    drop = child_score - parent_score
    if drop == 0:
        headline = f"This solution matched its parent's score ({child_score:.4g}); the change bought nothing."
    else:
        pct = f", {drop / abs(parent_score) * 100.0:+.1f}%" if parent_score else ""
        headline = (
            f"This solution scored below its parent: "
            f"{parent_score:.4g} -> {child_score:.4g} ({drop:+.4g}{pct})."
        )

    detail = describe_metric_deltas(child_metrics, parent_metrics, limit=6)
    if not detail:
        return headline
    return f"{headline}\nMetric changes vs parent: {detail}."
