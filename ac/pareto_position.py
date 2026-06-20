"""
Pareto position classifier.

Given a baseline + candidate (both EvaluatedCandidate from optimizer.py) plus
the conditioning hardware + constraints, classify the candidate's location
relative to the baseline-conditioned modifier Pareto frontier.

We re-use modifier.run_modifier_search() as the source of the frontier rather
than reimplementing search. The first call for a given (baseline, hw,
constraints) tuple is memoized so repeated evaluate_delta calls don't pay the
full search cost.

Pareto position vocabulary:

    DOMINATES_BASELINE   — candidate strictly better than the baseline on all
                           Pareto axes (one of which must be strict).
    DOMINATED_BY_BASELINE — candidate strictly worse than the baseline on all
                           Pareto axes.
    EXPANDS_FRONTIER     — candidate is not dominated by ANY frontier point
                           AND dominates at least one frontier point. The
                           frontier shifts.
    ON_FRONTIER          — candidate is not dominated by any frontier point
                           but also dominates none (lies on the existing
                           frontier surface in some axis).
    INTERIOR             — dominated by at least one frontier point but better
                           than the baseline on at least one axis (i.e. it's
                           an interior trade-off — useful trade but not
                           Pareto-optimal under the chosen objectives).
    EQUIVALENT           — within an epsilon ball of the baseline on every
                           axis.

The Pareto axes are:
    predicted_loss   (lower is better)
    serving_tbt_ms   (lower is better)
    prefill_time_ms  (lower is better)
    memory_per_gpu_gb (lower is better)
    -training_tps    (lower is better, negated so all axes are "lower better")
"""

from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- repo path bootstrap ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from optimizer import (  # noqa: E402
    CandidateArch,
    DeploymentConstraints,
    EvaluatedCandidate,
)
from modifier import (  # noqa: E402
    ModifierRecord,
    ModifierResult,
    run_modifier_search,
)
from baseline import BaselineModel  # noqa: E402


# =============================================================================
# Position vocabulary
# =============================================================================

PARETO_POSITION_KIND = (
    "DOMINATES_BASELINE",
    "DOMINATED_BY_BASELINE",
    "EXPANDS_FRONTIER",
    "ON_FRONTIER",
    "INTERIOR",
    "EQUIVALENT",
    "UNKNOWN",
)


# Pareto axes: (label, attribute_path, lower_is_better)
# attribute_path is read via _get(); supports "throughput.prefill_time_ms".
PARETO_AXES: Tuple[Tuple[str, str, bool], ...] = (
    ("predicted_loss",    "predicted_loss",    True),
    ("serving_tbt_ms",    "serving_tbt_ms",    True),
    ("prefill_time_ms",   "throughput.prefill_time_ms", True),
    ("memory_per_gpu_gb", "memory_per_gpu_gb", True),
    ("training_tps",      "training_tps",      False),  # higher is better
)


@dataclass
class ParetoPosition:
    """Detailed Pareto-position report for one candidate."""
    position: str = "UNKNOWN"
    distance: float = 0.0
    axes: List[str] = field(default_factory=list)
    dominated_by_count: int = 0    # frontier points that dominate the candidate
    dominates_count: int = 0       # frontier points the candidate dominates
    frontier_size: int = 0
    notes: List[str] = field(default_factory=list)


# =============================================================================
# Helpers
# =============================================================================

def _get(ev: EvaluatedCandidate, attr_path: str) -> float:
    obj: Any = ev
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return float(obj)


def _objectives_vector(ev: EvaluatedCandidate) -> Tuple[float, ...]:
    """Return the canonical Pareto-objective vector for one EvaluatedCandidate.

    All axes are normalized so "lower is better" (we negate higher-better axes).
    """
    out = []
    for label, attr, lower_better in PARETO_AXES:
        v = _get(ev, attr)
        if not lower_better:
            v = -v
        out.append(v)
    return tuple(out)


def _record_objectives(rec: ModifierRecord) -> Tuple[float, ...]:
    """Objectives for a ModifierRecord (wraps an EvaluatedCandidate)."""
    return _objectives_vector(rec.evaluated)


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...],
               eps: float = 1e-9) -> bool:
    """Return True if vector `a` dominates vector `b` (all axes minimize).

    Strict on at least one axis, no-worse on all axes.
    """
    strictly_better = False
    for ax, bx in zip(a, b):
        if ax > bx + eps:
            return False
        if ax < bx - eps:
            strictly_better = True
    return strictly_better


def _equal_within_eps(a: Tuple[float, ...], b: Tuple[float, ...],
                       rel_eps: float = 0.01) -> bool:
    """All axes within ``rel_eps`` relative distance (or absolute ε)."""
    for ax, bx in zip(a, b):
        denom = max(abs(ax), abs(bx), 1e-9)
        if abs(ax - bx) / denom > rel_eps and abs(ax - bx) > 1e-6:
            return False
    return True


# =============================================================================
# Frontier construction
# =============================================================================

# Memoize the modifier search keyed by (baseline_fingerprint, hardware, tp,
# context_length). Repeated evaluate_delta calls with the same baseline reuse
# the frontier.
_FRONTIER_CACHE: Dict[Tuple, ModifierResult] = {}


def _baseline_fingerprint(c: CandidateArch) -> Tuple:
    return (
        c.d_model, c.n_layers, c.n_heads, c.d_head, c.n_kv_heads,
        c.ffn_dim, c.vocab_size, c.weight_precision, c.ffn_precision,
        tuple(sorted(c.attn_precision.items())), c.kv_cache_bits,
    )


def _constraints_fingerprint(con: DeploymentConstraints) -> Tuple:
    return (
        int(con.tp), int(con.pp), int(con.dp),
        int(con.target_params_b * 1000),
        int(con.context_length or 0),
        int(con.prompt_len or 0),
        int(con.training_tokens or 0),
        int(con.serving_batch or 0),
    )


def _get_or_build_frontier(
    baseline_candidate: CandidateArch,
    hardware: str,
    constraints: DeploymentConstraints,
) -> ModifierResult:
    """Run (or look up) a modifier search for this baseline; return its result.

    The frontier we use is `result.pareto_frontier` from the modifier; this
    is the same frontier surface the modifier ranks against, so a delta that
    sits on it is genuinely Pareto-optimal under the modifier's objectives.
    """
    key = (_baseline_fingerprint(baseline_candidate),
           hardware,
           _constraints_fingerprint(constraints))
    if key in _FRONTIER_CACHE:
        return _FRONTIER_CACHE[key]

    bm = BaselineModel(
        path="<inline>", name="frontier_baseline",
        config={}, candidate=copy.deepcopy(baseline_candidate),
    )
    # Run the search (this also evaluates the baseline as a record).
    result = run_modifier_search(
        baseline_model=bm,
        hw_name=hardware,
        constraints=copy.deepcopy(constraints),
        tp_options=[int(constraints.tp)],
    )
    _FRONTIER_CACHE[key] = result
    return result


def clear_frontier_cache() -> None:
    """Drop all cached modifier searches. Test-friendly hook."""
    _FRONTIER_CACHE.clear()


# =============================================================================
# Distance metric
# =============================================================================

def _normalize_vector(
    v: Tuple[float, ...],
    scales: Tuple[float, ...],
) -> Tuple[float, ...]:
    out = []
    for x, s in zip(v, scales):
        out.append(x / s if s > 1e-9 else x)
    return tuple(out)


def _scales_from_frontier(frontier_vecs: List[Tuple[float, ...]]
                           ) -> Tuple[float, ...]:
    """Per-axis range = max-min (with a floor) so distance is dimensionless.

    Infers axis count from the first vector so this function works for
    arbitrary-dimension inputs (production uses len(PARETO_AXES); tests may
    pass 2-D toy vectors).
    """
    if not frontier_vecs:
        return tuple(1.0 for _ in PARETO_AXES)
    n_axes = len(frontier_vecs[0])
    scales = []
    for i in range(n_axes):
        vals = [v[i] for v in frontier_vecs if i < len(v)]
        if not vals:
            scales.append(1.0)
            continue
        rng = max(vals) - min(vals)
        scales.append(max(rng, abs(max(vals, key=abs)), 1e-6))
    return tuple(scales)


def _signed_distance(
    candidate_vec: Tuple[float, ...],
    frontier_vecs: List[Tuple[float, ...]],
    scales: Tuple[float, ...],
) -> float:
    """Min normalized Euclidean distance from candidate to any frontier point.

    The sign convention: negative when the candidate dominates the closest
    frontier point (Pareto-expansion side); positive when dominated; zero
    when on the frontier surface or equivalent.
    """
    if not frontier_vecs:
        return 0.0
    cn = _normalize_vector(candidate_vec, scales)
    best = float("inf")
    best_vec = None
    for fv in frontier_vecs:
        fn = _normalize_vector(fv, scales)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(cn, fn)))
        if d < best:
            best = d
            best_vec = fn
    # Determine sign by checking dominance against the nearest frontier point.
    if best_vec is None:
        return 0.0
    if _dominates(cn, best_vec):
        return -best
    if _dominates(best_vec, cn):
        return best
    return best  # mixed: positive (interior trade-off)


def signed_distance_to_frontier(
    cand_ev: EvaluatedCandidate,
    frontier: List[ModifierRecord],
) -> float:
    """Public helper: signed distance from `cand_ev` to the given frontier."""
    cand_vec = _objectives_vector(cand_ev)
    frontier_vecs = [_record_objectives(r) for r in frontier]
    scales = _scales_from_frontier(frontier_vecs + [cand_vec])
    return _signed_distance(cand_vec, frontier_vecs, scales)


# =============================================================================
# Classifier
# =============================================================================

def _which_axes_differ(
    base_vec: Tuple[float, ...],
    cand_vec: Tuple[float, ...],
    rel_eps: float = 0.001,
) -> List[str]:
    """Names of axes where baseline → candidate is a measurable change."""
    out = []
    for (label, _, _), bv, cv in zip(PARETO_AXES, base_vec, cand_vec):
        denom = max(abs(bv), abs(cv), 1e-9)
        if abs(bv - cv) / denom > rel_eps or abs(bv - cv) > 1e-6:
            out.append(label)
    return out


def classify_position(
    *,
    base_ev: EvaluatedCandidate,
    cand_ev: EvaluatedCandidate,
    baseline_candidate: CandidateArch,
    hardware: str,
    constraints: DeploymentConstraints,
) -> Tuple[str, float, List[str], int, int, int]:
    """Return (position, signed_distance, axes_changed, dominated_by_count,
    dominates_count, frontier_size).

    Implementation:
      1. Build (or look up) a baseline-conditioned modifier Pareto frontier.
      2. Compute the candidate's objective vector in the same axes.
      3. Compare against the baseline directly + against the frontier set.
    """
    base_vec = _objectives_vector(base_ev)
    cand_vec = _objectives_vector(cand_ev)
    axes_changed = _which_axes_differ(base_vec, cand_vec)

    # 1) baseline-direct comparison (fast path that doesn't depend on frontier)
    if _equal_within_eps(base_vec, cand_vec):
        return ("EQUIVALENT", 0.0, axes_changed, 0, 0, 0)

    # 2) frontier construction
    try:
        result = _get_or_build_frontier(
            baseline_candidate, hardware, constraints)
        frontier = list(result.pareto_frontier)
    except Exception:
        frontier = []

    frontier_vecs = [_record_objectives(r) for r in frontier]
    scales = _scales_from_frontier(frontier_vecs + [base_vec, cand_vec])
    distance = _signed_distance(cand_vec, frontier_vecs, scales)

    dominated_by = sum(1 for fv in frontier_vecs if _dominates(fv, cand_vec))
    dominates_n = sum(1 for fv in frontier_vecs if _dominates(cand_vec, fv))

    # 3) classification cascade
    if _dominates(cand_vec, base_vec):
        # Strictly better than baseline on all axes.
        if dominated_by == 0 and dominates_n > 0:
            return ("EXPANDS_FRONTIER", distance, axes_changed,
                    dominated_by, dominates_n, len(frontier))
        if dominated_by == 0:
            return ("ON_FRONTIER", distance, axes_changed,
                    dominated_by, dominates_n, len(frontier))
        return ("DOMINATES_BASELINE", distance, axes_changed,
                dominated_by, dominates_n, len(frontier))

    if _dominates(base_vec, cand_vec):
        return ("DOMINATED_BY_BASELINE", distance, axes_changed,
                dominated_by, dominates_n, len(frontier))

    # Mixed (better on some, worse on others) — classify by frontier relation.
    if dominated_by == 0 and dominates_n > 0:
        return ("EXPANDS_FRONTIER", distance, axes_changed,
                dominated_by, dominates_n, len(frontier))
    if dominated_by == 0:
        return ("ON_FRONTIER", distance, axes_changed,
                dominated_by, dominates_n, len(frontier))
    return ("INTERIOR", distance, axes_changed,
            dominated_by, dominates_n, len(frontier))
