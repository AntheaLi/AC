"""Wave 18b — Budget-matched comparisons and serving Pareto frontiers.

Per plan/redesign/18b-budget-and-serving-pareto.md. Provides:

  * `ComparisonBudget` — enum of the five budget projections.
  * `ParetoScenario` — enum of the four serving/training frontiers.
  * `CandidateMetrics` — the auditable derived metrics per evaluated candidate
    (active_params, total_params, training_flops, training_gpu_seconds,
    prefill_gpu_seconds, decode_gpu_seconds,
    serving_gpu_seconds_per_request, replica_gpus,
    aggregate_output_tokens_per_second, requests_per_second) plus topology
    (tp/pp/ep/cp/dp) and serving_batch.
  * `extract_metrics(ev, constraints)` — derive `CandidateMetrics` from an
    `EvaluatedCandidate` plus the surrounding `DeploymentConstraints`.
  * `match_budget(m, budget, target)` — ±5% (training) or ±10% (serving)
    tolerance test per spec.
  * `pareto_frontier(candidates, scenario)` — ε-Pareto per scenario. ε values:
    predicted_loss 1%, TTFT/TBT/throughput/cost 5%, mem/replica_gpus 5%.
  * `BudgetMatrix` — driver that receives a list of `EvaluatedCandidate`s and
    emits all budget projections + scenario frontiers.

The module is additive and standalone: it consumes existing
`EvaluatedCandidate` / `DeploymentConstraints` objects and produces new
projections without touching `optimizer.py`. Per the roadmap: "prefer new
modules over unrelated edits to optimizer.py."

Design principles enforced here:

  1. A candidate is never removed for TTFT/TBT/cost. Operationally extreme
     points remain in the diagnostic frontier; scenario frontiers annotate
     them with operational flags (18c) but keep them included pre-dominance
     so downstream consumers can see the trade-off surface.
  2. Actual `active_params` and `total_params` are always displayed. A
     120B-total / 5B-active MoE lands in the 5B active bucket AND the 120B
     total bucket — never the 120B active bucket.
  3. The same candidate may win one budget view and lose another. That is
     the intended behavior, not a bug. See spec acceptance gate.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # 18a's canonical parameter accounting when available.
    from ac.architecture import parameter_ledger as _parameter_ledger  # type: ignore
except Exception:  # pragma: no cover - graceful degradation for older builds
    _parameter_ledger = None  # type: ignore


# =============================================================================
# Enums
# =============================================================================


class ComparisonBudget(str, enum.Enum):
    """Five budget-projection views per 18b spec §Comparison views."""
    EQUAL_ACTIVE_PARAMS = "equal_active_params"
    EQUAL_TOTAL_PARAMS = "equal_total_params"
    EQUAL_TRAINING_FLOPS = "equal_training_flops"
    EQUAL_TRAINING_GPU_SECONDS = "equal_training_gpu_seconds"
    EQUAL_SERVING_GPU_SECONDS_PER_REQUEST = "equal_serving_gpu_seconds_per_request"


class ParetoScenario(str, enum.Enum):
    """Four Pareto frontier views per 18b spec §Pareto interfaces.

    UNCONSTRAINED_DIAGNOSTIC keeps every physically feasible candidate for
    research inspection; the three named frontiers apply epsilon dominance
    but never delete operationally extreme points before frontier
    construction.
    """
    INTERACTIVE_SERVING = "interactive_serving"
    THROUGHPUT_SERVING = "throughput_serving"
    TRAINING = "training"
    UNCONSTRAINED_DIAGNOSTIC = "unconstrained_diagnostic"


# =============================================================================
# CandidateMetrics
# =============================================================================


@dataclass
class Topology:
    """Parallelism topology for a single evaluated candidate.

    Kept separate from CandidateMetrics so consumers can print it as a
    single row-annotation ("TP=8 PP=1 EP=8 CP=1 DP=64 batch=16") without
    having to reach into six separate CandidateMetrics fields.
    """
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    dp: int = 1
    serving_batch: int = 1

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)

    def label(self) -> str:
        return (f"TP={self.tp} PP={self.pp} EP={self.ep} CP={self.cp} "
                f"DP={self.dp} batch={self.serving_batch}")


@dataclass
class CandidateMetrics:
    """Auditable derived metrics per 18b spec §Metrics.

    Every field except identity_label is a plain scalar computed from
    (arch, throughput, constraints) — no residual model, no calibration
    lookup. This makes the metrics reproducible from a saved
    EvaluatedCandidate + DeploymentConstraints pair.
    """
    # Identity — the label AC would use to describe this candidate. The
    # actual family classification comes from 18a; until 18a lands we
    # store the legacy family_label ("dense" | "moe" | "hybrid" |
    # "moe_hybrid") so 18b's tests can verify labeling without depending
    # on 18a's factorized axes.
    identity_label: str = "dense"

    # Sizes (18b spec: always display both).
    active_params: int = 0
    total_params: int = 0

    # Training-side derived metrics.
    training_flops: float = 0.0
    training_gpu_seconds: float = 0.0

    # Serving-side derived metrics.
    prefill_gpu_seconds: float = 0.0
    decode_gpu_seconds: float = 0.0
    serving_gpu_seconds_per_request: float = 0.0
    replica_gpus: int = 1
    aggregate_output_tokens_per_second: float = 0.0
    requests_per_second: float = 0.0

    # Latency and quality axes (used by Pareto scenarios).
    predicted_loss: float = 0.0
    ttft_ms: float = 0.0
    tbt_ms: float = 0.0
    memory_per_gpu_gb: float = 0.0

    # Topology (18b spec: show alongside every frontier point).
    topology: Topology = field(default_factory=Topology)

    # Wave 18c (Jun 2026): extended feasibility (physical + operational).
    # Populated lazily by matrix drivers or `attach_operational_flags()`.
    # `None` = not yet evaluated; callers should not assume the absence of
    # this field means "feasible".
    extended_feasibility: Optional[Any] = None

    def as_row(self) -> Dict[str, Any]:
        """Flat, JSON-safe view used by renderers and tests."""
        d = asdict(self)
        d["topology"] = self.topology.as_dict()
        if self.extended_feasibility is not None:
            try:
                d["extended_feasibility"] = self.extended_feasibility.as_dict()
            except AttributeError:
                # If the attached object doesn't expose as_dict, drop it
                # gracefully rather than break rendering.
                d["extended_feasibility"] = None
        else:
            d["extended_feasibility"] = None
        return d


# =============================================================================
# Metric extraction
# =============================================================================


# Chinchilla-style training-FLOP coefficient (6N per training token).
# We deliberately use active_params for MoE — the compute-per-token bill
# scales with active-only, matching what the throughput model already
# reports.
_TRAINING_FLOPS_PER_TOKEN_PER_PARAM = 6.0


def extract_metrics(
    ev: Any,
    constraints: Any,
    *,
    identity_label: Optional[str] = None,
) -> CandidateMetrics:
    """Derive a `CandidateMetrics` from an `EvaluatedCandidate` + `constraints`.

    Non-EvaluatedCandidate inputs (e.g. plain dicts from a snapshot) are
    also accepted so tests can construct synthetic candidates without
    running the optimizer end-to-end.
    """
    arch = getattr(ev, "arch", ev)

    # Prefer 18a's ParameterLedger — the single canonical accounting shared by
    # search, cost, and reporting. Falls back to direct arch attributes only
    # when the ledger cannot be built (e.g. synthetic test fixtures).
    active = 0
    total = 0
    if _parameter_ledger is not None:
        try:
            ledger = _parameter_ledger(arch)
            active = int(ledger.active_params)
            total = int(ledger.total_params)
        except Exception:
            active = 0
            total = 0
    if active <= 0:
        active = int(getattr(arch, "active_params", 0) or 0)
    if total <= 0:
        total = int(getattr(arch, "total_params", 0) or 0)
    if active <= 0:
        active = total
    if total <= 0:
        total = active
    # Never let active exceed total. MoE candidates carry active < total
    # by construction; a dense/state candidate has active == total.
    if total < active:
        total = active

    tp = max(1, int(getattr(arch, "tp_degree", 1) or 1))
    pp = max(1, int(getattr(arch, "pp_degree", 1) or 1))
    ep = max(1, int(getattr(arch, "ep_degree", 1) or 1))
    cp = max(1, int(getattr(arch, "cp_degree", 1) or 1))
    dp = max(1, int(getattr(constraints, "dp", 1) or 1))
    serving_batch = max(1, int(getattr(constraints, "serving_batch", 1) or 1))
    output_len = max(1, int(getattr(constraints, "output_len", 512) or 512))
    training_tokens = max(1, int(getattr(constraints, "training_tokens", 0) or 0))

    # A "replica" is the set of GPUs that jointly serve one inference
    # request. TP × PP × CP is the canonical serving replica size. EP is
    # co-located with MoE FFN parallelism inside the replica for MoE
    # (rank shares TP × PP GPUs); it does not multiply serving_gpus/req.
    replica_gpus = tp * pp * cp

    predicted_loss = float(getattr(ev, "predicted_loss", 0.0) or 0.0)
    ttft_ms = float(getattr(ev, "serving_ttft_ms", 0.0) or 0.0)
    if ttft_ms <= 0.0:
        # Fall back to throughput.prefill_time_ms if the top-level
        # accessor isn't populated (older EvaluatedCandidate shapes).
        tput = getattr(ev, "throughput", None)
        ttft_ms = float(getattr(tput, "prefill_time_ms", 0.0) or 0.0)
    tbt_ms = float(getattr(ev, "serving_tbt_ms", 0.0) or 0.0)
    mem_gb = float(getattr(ev, "memory_per_gpu_gb", 0.0) or 0.0)
    train_tps = float(getattr(ev, "training_tps", 0.0) or 0.0)

    training_flops = _TRAINING_FLOPS_PER_TOKEN_PER_PARAM * active * training_tokens
    # training_gpu_seconds is what a physical training cluster would bill:
    # (training_time_per_step_s × n_steps) × cluster_gpus, but since
    # training_tps is the per-replica tokens/sec, we can shortcut to
    # (training_tokens / train_tps) × (replica_gpus × dp).
    training_wall_s = training_tokens / train_tps if train_tps > 0 else 0.0
    training_gpu_seconds = training_wall_s * replica_gpus * dp

    prefill_s = ttft_ms / 1000.0
    decode_s = (tbt_ms / 1000.0) * output_len
    prefill_gpu_seconds = prefill_s * replica_gpus
    decode_gpu_seconds = decode_s * replica_gpus

    # Batched serving: prefill + decode cost is amortized over serving_batch
    # concurrent requests using the same replica.
    per_request_wall_s = prefill_s + decode_s
    serving_gpu_seconds_per_request = (
        per_request_wall_s * replica_gpus / max(1, serving_batch)
    )

    # Aggregate output tokens/sec across all DP replicas.
    per_replica_output_tps = (1000.0 / tbt_ms * serving_batch) if tbt_ms > 0 else 0.0
    aggregate_output_tps = per_replica_output_tps * dp
    requests_per_second = (
        aggregate_output_tps / output_len if output_len > 0 else 0.0
    )

    # Wave 18a: family label comes from the canonical ArchitectureSignature.
    # 18b readers continue to consume ``identity_label`` as the legacy
    # coarse label; new consumers should read ``signature`` (populated
    # below) for the factorized axes.
    if identity_label is None:
        try:
            from ac.architecture import architecture_signature
            identity_label = architecture_signature(arch).legacy_family
        except (ValueError, ImportError):
            has_moe = bool(getattr(arch, "moe", None))
            has_state = (bool(getattr(arch, "state_config", None))
                         and int(getattr(arch, "n_state_layers", 0) or 0) > 0)
            if has_moe and has_state:
                identity_label = "moe_hybrid"
            elif has_moe:
                identity_label = "moe"
            elif has_state:
                identity_label = "hybrid"
            else:
                identity_label = "dense"

    return CandidateMetrics(
        identity_label=identity_label,
        active_params=active,
        total_params=total,
        training_flops=training_flops,
        training_gpu_seconds=training_gpu_seconds,
        prefill_gpu_seconds=prefill_gpu_seconds,
        decode_gpu_seconds=decode_gpu_seconds,
        serving_gpu_seconds_per_request=serving_gpu_seconds_per_request,
        replica_gpus=replica_gpus,
        aggregate_output_tokens_per_second=aggregate_output_tps,
        requests_per_second=requests_per_second,
        predicted_loss=predicted_loss,
        ttft_ms=ttft_ms,
        tbt_ms=tbt_ms,
        memory_per_gpu_gb=mem_gb,
        topology=Topology(
            tp=tp, pp=pp, ep=ep, cp=cp, dp=dp,
            serving_batch=serving_batch,
        ),
    )


# =============================================================================
# Budget matching
# =============================================================================


# Per spec: training budgets ±5%, serving cost ±10%.
_TRAINING_BUDGET_TOL = 0.05
_SERVING_BUDGET_TOL = 0.10


def _budget_axis(budget: ComparisonBudget, m: CandidateMetrics) -> float:
    """Return the metric on which a budget is compared."""
    if budget == ComparisonBudget.EQUAL_ACTIVE_PARAMS:
        return float(m.active_params)
    if budget == ComparisonBudget.EQUAL_TOTAL_PARAMS:
        return float(m.total_params)
    if budget == ComparisonBudget.EQUAL_TRAINING_FLOPS:
        return float(m.training_flops)
    if budget == ComparisonBudget.EQUAL_TRAINING_GPU_SECONDS:
        return float(m.training_gpu_seconds)
    if budget == ComparisonBudget.EQUAL_SERVING_GPU_SECONDS_PER_REQUEST:
        return float(m.serving_gpu_seconds_per_request)
    raise ValueError(f"unknown budget: {budget}")


def _budget_tol(budget: ComparisonBudget) -> float:
    if budget == ComparisonBudget.EQUAL_SERVING_GPU_SECONDS_PER_REQUEST:
        return _SERVING_BUDGET_TOL
    return _TRAINING_BUDGET_TOL


def match_budget(
    m: CandidateMetrics,
    budget: ComparisonBudget,
    target: float,
) -> bool:
    """True iff `m` falls within the budget's tolerance around `target`.

    `target` is the reference value (typically the dense candidate's budget
    at that column). Tolerance is ±5% for training views, ±10% for serving
    cost per the spec §Comparison views.
    """
    if target <= 0:
        return False
    tol = _budget_tol(budget)
    value = _budget_axis(budget, m)
    return abs(value - target) / target <= tol


# =============================================================================
# Epsilon-Pareto per scenario
# =============================================================================


# Per spec §Pareto interfaces:
#   predicted loss: 1%
#   TTFT / TBT / throughput / cost: 5%
#   memory and replica size: 5%
_EPS_LOSS = 0.01
_EPS_SERVING = 0.05
_EPS_RESOURCE = 0.05


def _better_by_eps(
    a_val: float, b_val: float, eps: float, *, minimize: bool = True
) -> bool:
    """True iff b beats a by more than a fractional `eps` on this axis.

    Uses relative epsilon around max(|a|, |b|, 1e-12) so scale-free axes
    stay well-behaved. Zero values are treated as no-worse (epsilon-tied).
    """
    denom = max(abs(a_val), abs(b_val), 1e-12)
    delta = (a_val - b_val) if minimize else (b_val - a_val)
    return (delta / denom) > eps


def _no_worse_by_eps(
    a_val: float, b_val: float, eps: float, *, minimize: bool = True
) -> bool:
    denom = max(abs(a_val), abs(b_val), 1e-12)
    delta = (a_val - b_val) if minimize else (b_val - a_val)
    # b is no-worse than a if the (minimizing) delta is > -eps (i.e., b
    # doesn't lose by more than eps).
    return (delta / denom) >= -eps


def _scenario_axes(scenario: ParetoScenario) -> List[Tuple[str, str, float]]:
    """Return list of (attr_name, direction, epsilon) tuples per scenario.

    direction is "min" (lower is better) or "max" (higher is better).
    """
    if scenario == ParetoScenario.INTERACTIVE_SERVING:
        return [
            ("predicted_loss", "min", _EPS_LOSS),
            ("ttft_ms", "min", _EPS_SERVING),
            ("tbt_ms", "min", _EPS_SERVING),
            ("serving_gpu_seconds_per_request", "min", _EPS_SERVING),
            ("replica_gpus", "min", _EPS_RESOURCE),
        ]
    if scenario == ParetoScenario.THROUGHPUT_SERVING:
        return [
            ("predicted_loss", "min", _EPS_LOSS),
            ("serving_gpu_seconds_per_request", "min", _EPS_SERVING),
            ("memory_per_gpu_gb", "min", _EPS_RESOURCE),
            ("replica_gpus", "min", _EPS_RESOURCE),
            ("aggregate_output_tokens_per_second", "max", _EPS_SERVING),
            ("requests_per_second", "max", _EPS_SERVING),
        ]
    if scenario == ParetoScenario.TRAINING:
        return [
            ("predicted_loss", "min", _EPS_LOSS),
            ("training_flops", "min", _EPS_SERVING),
            ("training_gpu_seconds", "min", _EPS_SERVING),
            ("memory_per_gpu_gb", "min", _EPS_RESOURCE),
        ]
    if scenario == ParetoScenario.UNCONSTRAINED_DIAGNOSTIC:
        # Diagnostic view: everything survives; consumers filter later.
        return []
    raise ValueError(f"unknown scenario: {scenario}")


def epsilon_dominated(
    a: CandidateMetrics, b: CandidateMetrics, scenario: ParetoScenario,
) -> bool:
    """True iff `b` epsilon-dominates `a` for `scenario`.

    b epsilon-dominates a iff:
      - b is no-worse than a on every scenario axis (within the axis eps), AND
      - b is strictly better than a on at least one axis (beyond the axis eps).
    """
    axes = _scenario_axes(scenario)
    if not axes:
        return False  # diagnostic view: nothing is dominated
    strict_wins = 0
    for attr, direction, eps in axes:
        a_val = float(getattr(a, attr, 0.0) or 0.0)
        b_val = float(getattr(b, attr, 0.0) or 0.0)
        minimize = (direction == "min")
        if not _no_worse_by_eps(a_val, b_val, eps, minimize=minimize):
            return False
        if _better_by_eps(a_val, b_val, eps, minimize=minimize):
            strict_wins += 1
    return strict_wins >= 1


def pareto_frontier(
    candidates: Sequence[CandidateMetrics], scenario: ParetoScenario,
) -> List[CandidateMetrics]:
    """Return the epsilon-Pareto frontier for a scenario.

    Per spec §Pareto interfaces: "Do not delete operationally extreme
    points before frontier construction." All candidates are considered;
    scenario axes (only) drive dominance. Operational flags from 18c can
    be attached to surviving frontier points by callers.
    """
    surviving: List[CandidateMetrics] = []
    for i, cand in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if epsilon_dominated(cand, other, scenario):
                dominated = True
                break
        if not dominated:
            surviving.append(cand)
    return surviving


# =============================================================================
# BudgetMatrix driver
# =============================================================================


@dataclass
class BudgetView:
    """One (budget, target) projection over a set of candidates.

    The `matched` list is populated with candidates whose budget-axis
    value falls inside the budget's tolerance around `target`. `all_evaluated`
    is preserved verbatim so consumers can see which candidates were
    considered but excluded from this view.
    """
    budget: ComparisonBudget
    target: float
    matched: List[CandidateMetrics] = field(default_factory=list)
    all_evaluated: List[CandidateMetrics] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget.value,
            "target": self.target,
            "tolerance_pct": _budget_tol(self.budget) * 100.0,
            "matched": [m.as_row() for m in self.matched],
            "n_matched": len(self.matched),
            "n_evaluated": len(self.all_evaluated),
        }


@dataclass
class ScenarioFrontier:
    scenario: ParetoScenario
    frontier: List[CandidateMetrics] = field(default_factory=list)
    n_evaluated: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "frontier": [m.as_row() for m in self.frontier],
            "n_frontier": len(self.frontier),
            "n_evaluated": self.n_evaluated,
        }


@dataclass
class MatrixCell:
    """One matrix cell's Wave 18b output: five budget views + four scenario
    frontiers over the same underlying candidate pool.

    A cell is identified by a (hardware, context_length, training_tokens,
    reference_active_b) tuple. `reference_active_b` is the target the
    equal-active view centers on; it's used verbatim so downstream code
    can pick the reference from either the user-requested target or an
    audit-picked anchor.
    """
    hardware: str = ""
    context_length: int = 0
    training_tokens: int = 0
    reference_active_b: float = 0.0
    reference_total_b: float = 0.0
    reference_training_flops: float = 0.0
    reference_training_gpu_seconds: float = 0.0
    reference_serving_gpu_seconds_per_request: float = 0.0

    budget_views: Dict[str, BudgetView] = field(default_factory=dict)
    scenario_frontiers: Dict[str, ScenarioFrontier] = field(default_factory=dict)
    all_evaluated: List[CandidateMetrics] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hardware": self.hardware,
            "context_length": self.context_length,
            "training_tokens": self.training_tokens,
            "reference": {
                "active_b": self.reference_active_b,
                "total_b": self.reference_total_b,
                "training_flops": self.reference_training_flops,
                "training_gpu_seconds": self.reference_training_gpu_seconds,
                "serving_gpu_seconds_per_request":
                    self.reference_serving_gpu_seconds_per_request,
            },
            "budget_views": {k: v.as_dict() for k, v in self.budget_views.items()},
            "scenario_frontiers": {
                k: v.as_dict() for k, v in self.scenario_frontiers.items()
            },
            "n_candidates_evaluated": len(self.all_evaluated),
        }


class BudgetMatrix:
    """Reusable matrix driver that receives evaluated candidates and emits
    all budget/scenario projections.

    Usage:
        driver = BudgetMatrix(hardware="h100", context_length=131072,
                              training_tokens=int(20e12))
        # anchor is the "dense reference" whose budgets set the ±5%/±10%
        # windows for all four budget-matched views.
        anchor = extract_metrics(dense_ev, constraints)
        for ev in all_evaluated:
            driver.add(extract_metrics(ev, constraints))
        cell = driver.build_cell(anchor)
    """

    def __init__(
        self,
        *,
        hardware: str = "",
        context_length: int = 0,
        training_tokens: int = 0,
    ) -> None:
        self.hardware = hardware
        self.context_length = context_length
        self.training_tokens = training_tokens
        self._candidates: List[CandidateMetrics] = []

    def add(self, m: CandidateMetrics) -> None:
        self._candidates.append(m)

    def extend(self, ms: Iterable[CandidateMetrics]) -> None:
        for m in ms:
            self.add(m)

    def build_cell(self, anchor: CandidateMetrics) -> MatrixCell:
        """Build one MatrixCell using `anchor` as the reference for all
        four budget-matched projections. The anchor itself does not have
        to be in the candidate pool (it can be an audit-only reference)."""
        cell = MatrixCell(
            hardware=self.hardware,
            context_length=self.context_length,
            training_tokens=self.training_tokens,
            reference_active_b=anchor.active_params / 1e9,
            reference_total_b=anchor.total_params / 1e9,
            reference_training_flops=anchor.training_flops,
            reference_training_gpu_seconds=anchor.training_gpu_seconds,
            reference_serving_gpu_seconds_per_request=
                anchor.serving_gpu_seconds_per_request,
            all_evaluated=list(self._candidates),
        )
        # Budget views.
        budget_targets: Dict[ComparisonBudget, float] = {
            ComparisonBudget.EQUAL_ACTIVE_PARAMS: float(anchor.active_params),
            ComparisonBudget.EQUAL_TOTAL_PARAMS: float(anchor.total_params),
            ComparisonBudget.EQUAL_TRAINING_FLOPS: float(anchor.training_flops),
            ComparisonBudget.EQUAL_TRAINING_GPU_SECONDS:
                float(anchor.training_gpu_seconds),
            ComparisonBudget.EQUAL_SERVING_GPU_SECONDS_PER_REQUEST:
                float(anchor.serving_gpu_seconds_per_request),
        }
        for budget, target in budget_targets.items():
            matched = [m for m in self._candidates if match_budget(m, budget, target)]
            cell.budget_views[budget.value] = BudgetView(
                budget=budget, target=target, matched=matched,
                all_evaluated=list(self._candidates),
            )
        # Scenario frontiers.
        for scenario in ParetoScenario:
            if scenario == ParetoScenario.UNCONSTRAINED_DIAGNOSTIC:
                # Diagnostic keeps every physically feasible candidate.
                frontier = list(self._candidates)
            else:
                frontier = pareto_frontier(self._candidates, scenario)
            cell.scenario_frontiers[scenario.value] = ScenarioFrontier(
                scenario=scenario, frontier=frontier,
                n_evaluated=len(self._candidates),
            )
        return cell


# =============================================================================
# Renderers
# =============================================================================


def render_json(cell: MatrixCell) -> Dict[str, Any]:
    """JSON-safe dict rendering of a MatrixCell."""
    return cell.as_dict()


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(int(n))


def _fmt_flops(x: float) -> str:
    if x >= 1e24:
        return f"{x / 1e24:.2f}Y"
    if x >= 1e21:
        return f"{x / 1e21:.2f}Z"
    if x >= 1e18:
        return f"{x / 1e18:.2f}E"
    if x >= 1e15:
        return f"{x / 1e15:.2f}P"
    return f"{x:.2e}"


# =============================================================================
# Wave 18c operational-flag attachment
# =============================================================================


def attach_operational_flags(
    metrics_list: Sequence[CandidateMetrics],
    workload: Any,
    *,
    hardware: Any = None,
    hbm_gb: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
    gpu_seconds_budget: Optional[float] = None,
) -> None:
    """Populate ``m.extended_feasibility`` for every metric in `metrics_list`.

    Uses Wave 18c's `build_extended_feasibility` so operational flags are
    consistent across BudgetMatrix cells. Per 18c: operational flags NEVER
    remove a candidate from any frontier — they only annotate.
    """
    try:
        from ac.operational_flags import build_extended_feasibility
    except Exception:  # pragma: no cover
        return
    for m in metrics_list:
        m.extended_feasibility = build_extended_feasibility(
            m, workload,
            hardware=hardware, hbm_gb=hbm_gb,
            thresholds=thresholds, gpu_seconds_budget=gpu_seconds_budget,
        )


# =============================================================================
# Cross-cell matrix aggregation
# =============================================================================


@dataclass
class MatrixKey:
    """Coordinates of one cell inside a Wave 18b matrix.

    Deliberately typed rather than a tuple so future integrations (hardware
    calibration, workload registry, etc.) can extend the key without
    breaking rendering call-sites.
    """
    hardware: str
    context_length: int
    reference_active_b: float

    def as_tuple(self) -> Tuple[str, int, float]:
        return (self.hardware, self.context_length, self.reference_active_b)


@dataclass
class Matrix:
    """Multi-cell Wave 18b matrix.

    Groups `MatrixCell`s by `(hardware, context_length, reference_active_b)`
    so the main-matrix renderer can display one row per context and one
    column per reference active size. Detailed budget/scenario sections
    remain per-cell.

    The main matrix intentionally displays `contender_status` per Wave 18d
    (winner/unresolved/out_of_domain/no_physically_feasible_candidate/
    model_validation_failure). Until 18d lands as an owner, the driver
    accepts a callable `status_fn(cell) -> str` so 18b can render whatever
    label the downstream integrator supplies. Default is loss-argmin from
    the interactive-serving frontier — clearly a *compatibility* projection,
    not a scientific claim.
    """
    training_tokens: int = 0
    cells: Dict[Tuple[str, int, float], MatrixCell] = field(default_factory=dict)

    def add_cell(self, key: MatrixKey, cell: MatrixCell) -> None:
        self.cells[key.as_tuple()] = cell

    def get_cell(self, key: MatrixKey) -> Optional[MatrixCell]:
        return self.cells.get(key.as_tuple())

    def hardware_names(self) -> List[str]:
        return sorted({k[0] for k in self.cells.keys()})

    def context_lengths(self) -> List[int]:
        return sorted({k[1] for k in self.cells.keys()})

    def reference_active_bs(self) -> List[float]:
        return sorted({k[2] for k in self.cells.keys()})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "training_tokens": self.training_tokens,
            "cells": [
                {"key": {"hardware": k[0], "context_length": k[1],
                         "reference_active_b": k[2]},
                 "cell": v.as_dict()}
                for k, v in self.cells.items()
            ],
            "hardware": self.hardware_names(),
            "contexts": self.context_lengths(),
            "reference_active_bs": self.reference_active_bs(),
        }


# Default contender_status fallback — used when 18d has not been wired in.
# Explicitly labeled as a compatibility projection in the rendered output.
def _default_contender_status(cell: MatrixCell) -> str:
    """Loss-argmin over the interactive-serving frontier.

    Not a scientific decision — 18d owns the real contender-status field.
    This exists so 18b's matrix renderer produces a non-empty main matrix
    before 18d lands. The status label always includes the `[loss-argmin]`
    tag so a reviewer can tell it isn't the confidence-aware decision.
    """
    front = cell.scenario_frontiers.get(ParetoScenario.INTERACTIVE_SERVING.value)
    if front is None or not front.frontier:
        return "no physically feasible candidate"
    winner = min(front.frontier, key=lambda m: m.predicted_loss)
    return f"{winner.identity_label} [loss-argmin]"


def render_matrix_markdown(
    matrix: Matrix,
    *,
    status_fn=None,
) -> str:
    """Render a Wave 18b matrix as (hardware × context) × reference-active-B.

    Main matrix: contender status per cell (Wave 18d owns this; see the
    module docstring). Detailed sections: per-cell budget views and
    scenario frontiers.
    """
    if status_fn is None:
        status_fn = _default_contender_status
    lines: List[str] = []
    lines.append("# Wave 18b matrix — comparison views + scenario Pareto")
    lines.append("")
    lines.append(
        f"Training tokens: {matrix.training_tokens:,} | "
        f"hardware: {', '.join(matrix.hardware_names())} | "
        f"reference active-B: {matrix.reference_active_bs()}"
    )
    lines.append("")
    ref_bs = matrix.reference_active_bs()
    contexts = matrix.context_lengths()
    for hw in matrix.hardware_names():
        lines.append(f"## Main matrix — {hw}")
        lines.append("")
        header = "| ctx | " + " | ".join(f"{b:g}B" for b in ref_bs) + " |"
        sep = "|" + "---|" * (len(ref_bs) + 1)
        lines.append(header)
        lines.append(sep)
        for ctx in contexts:
            row = [f"{ctx}"]
            for b in ref_bs:
                cell = matrix.get_cell(MatrixKey(hw, ctx, b))
                if cell is None:
                    row.append("—")
                else:
                    row.append(status_fn(cell))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # Detailed per-cell sections.
    lines.append("## Detailed cells")
    lines.append("")
    for key_tuple, cell in matrix.cells.items():
        lines.append(render_markdown(cell))
        lines.append("")
    return "\n".join(lines)


def render_matrix_json(matrix: Matrix) -> Dict[str, Any]:
    return matrix.as_dict()


def render_markdown(cell: MatrixCell) -> str:
    """Compact markdown rendering of a MatrixCell for CLI / report output.

    Main matrix will surface contender status (18d owns that field); this
    renderer exposes the detailed frontier + budget-view sections so a
    reviewer can audit the projections directly.
    """
    lines: List[str] = []
    lines.append(f"# Matrix cell — {cell.hardware}, "
                 f"ctx={cell.context_length}, tokens={cell.training_tokens}")
    lines.append("")
    lines.append(
        f"Reference: active={_fmt_params(int(cell.reference_active_b * 1e9))}, "
        f"total={_fmt_params(int(cell.reference_total_b * 1e9))}, "
        f"training_flops={_fmt_flops(cell.reference_training_flops)}, "
        f"training_gpu_s={cell.reference_training_gpu_seconds:.2e}, "
        f"serving_gpu_s/req={cell.reference_serving_gpu_seconds_per_request:.4f}"
    )
    lines.append("")
    lines.append("## Budget views")
    for name, view in cell.budget_views.items():
        lines.append(f"### {name} (±{_budget_tol(view.budget) * 100:.0f}%)")
        if not view.matched:
            lines.append("_no candidates matched_")
            lines.append("")
            continue
        lines.append("| identity | active | total | loss | TBT (ms) | "
                     "TTFT (ms) | mem (GB) | gpu-s/req | topology |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for m in view.matched:
            lines.append(
                f"| {m.identity_label} | {_fmt_params(m.active_params)} | "
                f"{_fmt_params(m.total_params)} | {m.predicted_loss:.4f} | "
                f"{m.tbt_ms:.2f} | {m.ttft_ms:.1f} | "
                f"{m.memory_per_gpu_gb:.1f} | "
                f"{m.serving_gpu_seconds_per_request:.4f} | "
                f"{m.topology.label()} |"
            )
        lines.append("")
    lines.append("## Scenario frontiers")
    for name, sf in cell.scenario_frontiers.items():
        lines.append(f"### {name} ({len(sf.frontier)} of {sf.n_evaluated} candidates)")
        if not sf.frontier:
            lines.append("_frontier empty_")
            lines.append("")
            continue
        lines.append("| identity | active | total | loss | TBT (ms) | "
                     "gpu-s/req | agg tok/s | topology |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for m in sf.frontier:
            lines.append(
                f"| {m.identity_label} | {_fmt_params(m.active_params)} | "
                f"{_fmt_params(m.total_params)} | {m.predicted_loss:.4f} | "
                f"{m.tbt_ms:.2f} | "
                f"{m.serving_gpu_seconds_per_request:.4f} | "
                f"{m.aggregate_output_tokens_per_second:.1f} | "
                f"{m.topology.label()} |"
            )
        lines.append("")
    return "\n".join(lines)
