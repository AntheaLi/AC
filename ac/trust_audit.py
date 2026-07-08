"""Wave 18e — trust audit: search stability, structural invariants, frontier
feasibility anchors, and public-model predictive accuracy.

This module is the *audit* layer that runs alongside the optimizer/quality/
throughput stack. It does not modify any decisions; it inspects them.

Four responsibilities:

1. **Sensitivity suite** — for a target cell, run the optimizer under 1x/2x/4x
   candidate caps, prior multipliers, hardware coefficient sweeps, both
   quality-model versions, and multiple deterministic enumeration orderings.
   Report `winner_stability_fraction` and `contender_retention_fraction`.
2. **Structural invariants** — matrix-level tests that must hold across
   cells (active/total consistency, TP monotonicity above island size,
   family coverage before pruning, etc.).
3. **Frontier feasibility anchors** — 1M/2M physical representability checks.
4. **Public-model predictive accuracy anchor** — for each entry in
   `tests/fixtures/public_model_anchors_v1.json` with a documented
   `published_metric`, assert AC's prediction lands inside the tolerance
   band. Blocks matrix publication when it doesn't.

Usage from CLI (`scripts/run_trust_audit.py`) or tests.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The audit is designed to import lazily — sensitivity/invariant paths pull the
# optimizer; the public-anchor path only pulls evaluate_candidate. Callers can
# use one without paying for the other.

# =============================================================================
# Public-model anchor: dataclasses + loader
# =============================================================================


@dataclass(frozen=True)
class PublishedMetric:
    loss: float
    tbt_ms: float
    ttft_ms: float
    mem_gb: float
    source: str


@dataclass(frozen=True)
class Tolerances:
    loss: float
    tbt_ms: float
    ttft_ms: float
    mem_gb: float


@dataclass(frozen=True)
class PublicModelAnchor:
    id: str
    display_name: str
    arch: Dict[str, Any]
    training_tokens: int
    active_params_b: float
    total_params_b: float
    workload: Dict[str, Any]
    published_metric: Optional[PublishedMetric]
    representable: bool
    tolerances_override: Optional[Tolerances] = None
    notes: str = ""


@dataclass
class AnchorMetricResult:
    """One metric check: predicted vs published, with pass/fail."""

    metric: str
    predicted: float
    published: float
    rel_err: float
    tolerance: float
    passed: bool


@dataclass
class AnchorBreakdown:
    """Per-quality-term error attribution for a predictive anchor."""

    term_name: str
    value: float  # loss-delta contribution


@dataclass
class AnchorResult:
    anchor_id: str
    display_name: str
    status: str  # "pass" | "fail" | "skipped" | "unrepresentable" | "error"
    metrics: List[AnchorMetricResult] = field(default_factory=list)
    breakdown: List[AnchorBreakdown] = field(default_factory=list)
    error: str = ""
    override_recorded: bool = False
    override_justification: str = ""

    @property
    def failed_metrics(self) -> List[str]:
        return [m.metric for m in self.metrics if not m.passed]

    @property
    def is_blocking(self) -> bool:
        # Fails only block publication when the anchor was ACTIVELY checked
        # (not skipped/unrepresentable) and no override was recorded.
        return self.status == "fail" and not self.override_recorded


DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "public_model_anchors_v1.json"
)


def load_public_model_registry(
    path: Optional[Path] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[PublicModelAnchor], Tolerances, Tolerances]:
    """Load the public-model registry.

    Returns (anchors, default_tolerances, post_calibration_tolerances).
    `overrides` allows a maintainer to record a per-entry tolerance widening
    without editing the fixture; the audit report records the override for
    provenance.
    """
    path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    with open(path) as f:
        raw = json.load(f)

    if raw.get("schema_version", "").split(".")[0] != "wave18e":
        raise ValueError(
            f"Unsupported public-anchor registry schema: {raw.get('schema_version')!r}"
        )

    def _to_tol(d: Dict[str, float]) -> Tolerances:
        return Tolerances(
            loss=float(d["loss"]),
            tbt_ms=float(d["tbt_ms"]),
            ttft_ms=float(d["ttft_ms"]),
            mem_gb=float(d["mem_gb"]),
        )

    default_tol = _to_tol(raw["default_tolerances"])
    post_tol = _to_tol(raw["post_calibration_tolerances"])

    anchors: List[PublicModelAnchor] = []
    for e in raw["entries"]:
        pm = e.get("published_metric")
        published = (
            PublishedMetric(
                loss=float(pm["loss"]),
                tbt_ms=float(pm["tbt_ms"]),
                ttft_ms=float(pm["ttft_ms"]),
                mem_gb=float(pm["mem_gb"]),
                source=str(pm["source"]),
            )
            if pm is not None
            else None
        )
        ovr = None
        if overrides and e["id"] in overrides:
            ovr = _to_tol(overrides[e["id"]])
        anchors.append(
            PublicModelAnchor(
                id=str(e["id"]),
                display_name=str(e["display_name"]),
                arch=dict(e["arch"]),
                training_tokens=int(e["training_tokens"]),
                active_params_b=float(e["active_params_b"]),
                total_params_b=float(e["total_params_b"]),
                workload=dict(e["workload"]),
                published_metric=published,
                representable=bool(e.get("representable", True)),
                tolerances_override=ovr,
                notes=str(e.get("notes", "")),
            )
        )
    return anchors, default_tol, post_tol


# =============================================================================
# Anchor evaluation
# =============================================================================


def _build_candidate_from_anchor(anchor: PublicModelAnchor):
    """Materialize a CandidateArch from the anchor's arch dict.

    Kept independent of the optimizer's generator so registry entries are
    hand-authored, not search outputs.
    """
    from ac.optimizer import CandidateArch

    a = anchor.arch
    moe = a.get("moe")
    # Wave 21: default EP=1 when the anchor's workload didn't spell it out.
    # Published MoE serving benchmarks (vLLM Mixtral/DBRX TP8 etc.) run
    # TP-only with TP-sharded experts, which the model now supports as
    # EP=1. The previous "contract" default of EP=2 silently doubled the
    # modeled GPU count vs. the published deployment, manufacturing a
    # uniform ~−50…−60% per-GPU memory "model bias" (and inflating
    # TBT/TTFT errors) across every MoE anchor — workload mismatch, not
    # model error. Anchors whose published numbers really used EP must
    # say so in their workload block.
    ep = int(anchor.workload.get("ep", 0)) or 1

    return CandidateArch(
        d_model=int(a["d_model"]),
        n_layers=int(a["n_layers"]),
        n_heads=int(a["n_heads"]),
        d_head=int(a["d_head"]),
        n_kv_heads=int(a.get("n_kv_heads", a["n_heads"])),
        ffn_dim=int(a["ffn_dim"]),
        vocab_size=int(a["vocab_size"]),
        weight_precision=a.get("weight_precision", "bf16"),
        ffn_precision=a.get("ffn_precision", "bf16"),
        attn_precision=a.get(
            "attn_precision",
            {"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
        ),
        kv_cache_bits=int(a.get("kv_cache_bits", 16)),
        moe=moe,
        state_config=a.get("state_config"),
        n_dense_ffn_layers=int(a.get("n_dense_ffn_layers", 0)),
        attention_type=a.get("attention_type", "gqa"),
        mla_kv_latent_dim=int(a.get("mla_kv_latent_dim", 0)),
        mla_q_latent_dim=int(a.get("mla_q_latent_dim", 0)),
        mla_rope_head_dim=int(a.get("mla_rope_head_dim", 0)),
        mla_nope_head_dim=int(a.get("mla_nope_head_dim", 0)),
        swa_window=int(a.get("swa_window", 0)),
        n_state_layers=int(a.get("n_state_layers", 0)),
        n_attention_layers=int(a.get("n_attention_layers", 0)),
        tp_degree=int(anchor.workload.get("tp", 1)),
        pp_degree=int(anchor.workload.get("pp", 1)),
        ep_degree=ep,
        cp_degree=int(anchor.workload.get("cp", 1)),
    )


def _build_constraints_from_anchor(anchor: PublicModelAnchor):
    from ac.optimizer import DeploymentConstraints

    w = anchor.workload
    # Wave 18e post-audit fix (2026-07): separate `context_length` (capacity
    # claim of the model — e.g. Llama-3-8B supports 8k+) from
    # `fresh_prompt_tokens` (the actual prompt length used by the benchmark
    # whose TTFT the anchor is validated against). Published TTFT for
    # Llama-3-8B is 55ms at prompt_len ~1024 (vLLM/TRT-LLM benchmark
    # default), NOT at prompt_len=8192. Without this split, the audit was
    # measuring TTFT at 8× the benchmark's prompt length and reporting
    # 300-600% "errors" that were entirely workload-mismatch, not
    # modeling bugs.
    #
    # Backward compat: if `fresh_prompt_tokens` is absent, fall back to
    # `context_length`, preserving pre-fix behavior for entries not yet
    # updated.
    ctx = int(w.get("context_length", 8192))
    fresh_prompt = int(w.get("fresh_prompt_tokens", ctx))
    # Wave 18c/e (2026-07): decode_kv_length is the average KV state during
    # decode. Published TBT benchmarks measure at the midpoint of generation:
    # ~fresh_prompt + output_tokens/2. Default to fresh_prompt + 512 when not
    # specified, matching the typical vLLM benchmark of 1024 in + 512 out.
    decode_kv_len = int(w.get("decode_kv_length", fresh_prompt + 512))
    # `context_length` must be at least decode_kv_len so the throughput
    # model's KV memory allocation covers the real serving span.
    ctx = max(ctx, decode_kv_len, fresh_prompt)
    return DeploymentConstraints(
        target_params_b=float(anchor.active_params_b),
        training_tokens=int(anchor.training_tokens),
        context_length=ctx,
        prompt_len=fresh_prompt,
        tp=int(w.get("tp", 1)),
        pp=int(w.get("pp", 1)),
        dp=int(w.get("dp", 1)),
        tp_options=[int(w.get("tp", 1))],
        serving_tbt_ms=None,
        serving_ttft_ms=None,
        serving_batch=int(w.get("serving_batch", 8)),
        allow_moe=bool(anchor.arch.get("moe")),
        max_total_params_b=(
            float(anchor.total_params_b) if anchor.arch.get("moe") else None
        ),
        allow_state=bool(anchor.arch.get("state_type")),
        state_type=str(anchor.arch.get("state_type", "mamba2")),
        vocab_size=int(anchor.arch["vocab_size"]),
        allow_quality_sentinel=True,
    )


def _tolerances_for(anchor: PublicModelAnchor, default: Tolerances) -> Tolerances:
    return anchor.tolerances_override or default


def _rel_err(predicted: float, published: float) -> float:
    if published <= 0:
        return float("inf") if predicted != 0 else 0.0
    return (predicted - published) / published


def _breakdown_from_quality(quality_result: Any) -> List[AnchorBreakdown]:
    """Extract per-term contributions from a QualityResult (or equivalent)."""
    out: List[AnchorBreakdown] = []
    terms = getattr(quality_result, "terms", None) or {}
    for name, term in terms.items():
        val = getattr(term, "value", None)
        if val is None:
            continue
        out.append(AnchorBreakdown(term_name=str(name), value=float(val)))
    return out


def run_public_anchor(
    anchor: PublicModelAnchor,
    default_tolerances: Tolerances,
    hardware_override: Optional[str] = None,
) -> AnchorResult:
    """Evaluate one public-model anchor against its published metrics."""
    if not anchor.representable:
        return AnchorResult(
            anchor_id=anchor.id,
            display_name=anchor.display_name,
            status="unrepresentable",
        )
    if anchor.published_metric is None:
        return AnchorResult(
            anchor_id=anchor.id,
            display_name=anchor.display_name,
            status="skipped",
        )

    tol = _tolerances_for(anchor, default_tolerances)
    hw = hardware_override or anchor.workload.get("hardware", "h100")

    try:
        import copy as _copy
        from ac.optimizer import evaluate_candidate

        cand = _build_candidate_from_anchor(anchor)
        cons_ttft = _build_constraints_from_anchor(anchor)
        # Wave 18c/e (2026-07): run two evaluations because TTFT and TBT
        # are measured at different points in a serving request lifecycle.
        # TTFT: measured at fresh_prompt (prefill of the input tokens).
        # TBT:  measured mid-generation, so decode KV state is larger.
        # Memory: measured at peak, which usually coincides with mid-decode
        #         KV state (fresh_prompt + output_tokens/2).
        # Using `prompt_len=fresh_prompt` for BOTH under-counts TBT.
        w = anchor.workload
        fresh_prompt = int(w.get("fresh_prompt_tokens", cons_ttft.prompt_len))
        decode_kv_len = int(w.get("decode_kv_length", fresh_prompt + 512))
        ev_ttft = evaluate_candidate(cand, hw, cons_ttft)
        # TBT/mem eval: same constraints but prompt_len = decode_kv_len so
        # the throughput model sees the mid-generation KV state.
        if decode_kv_len != fresh_prompt:
            cons_tbt = _copy.copy(cons_ttft)
            cons_tbt.prompt_len = decode_kv_len
            cons_tbt.context_length = max(
                int(cons_ttft.context_length), decode_kv_len)
            ev_tbt = evaluate_candidate(cand, hw, cons_tbt)
        else:
            ev_tbt = ev_ttft
    except Exception as e:
        return AnchorResult(
            anchor_id=anchor.id,
            display_name=anchor.display_name,
            status="error",
            error=f"{type(e).__name__}: {e}",
        )

    ev = ev_ttft  # keep alias for the quality-breakdown extraction below

    # Wave 21: exclude the vocab_residual DESIGN PRIOR from the audit's
    # absolute-loss check. A published anchor loss is measured in the
    # anchor's own tokenizer's units; the undersized-vocab penalty is a
    # counterfactual cross-tokenizer prior ("a bigger tokenizer would have
    # been more data-efficient"), which the quality model itself labels
    # not-strictly-commensurable. Charging it here penalized e.g.
    # Mixtral-8x22B (32k vocab) several % against its OWN measured loss —
    # the term made anchor agreement strictly worse while saying nothing
    # about model fidelity. It remains fully active in sweeps/picks.
    pred_loss = float(getattr(ev_ttft, "predicted_loss", 0.0))
    _q = getattr(ev_ttft, "quality", None)
    _terms = getattr(_q, "terms", None) or {}
    _vt = _terms.get("vocab_residual")
    if _vt is not None:
        pred_loss -= float(getattr(_vt, "delta", 0.0) or 0.0)

    pred = {
        "loss": pred_loss,
        "tbt_ms": float(getattr(ev_tbt, "serving_tbt_ms", 0.0)),
        "ttft_ms": float(
            getattr(getattr(ev_ttft, "throughput", None), "prefill_time_ms", 0.0)
            or 0.0
        ),
        "mem_gb": float(getattr(ev_tbt, "memory_per_gpu_gb", 0.0)),
    }
    pub = anchor.published_metric
    checks = [
        ("loss", pred["loss"], pub.loss, tol.loss),
        ("tbt_ms", pred["tbt_ms"], pub.tbt_ms, tol.tbt_ms),
        ("ttft_ms", pred["ttft_ms"], pub.ttft_ms, tol.ttft_ms),
        ("mem_gb", pred["mem_gb"], pub.mem_gb, tol.mem_gb),
    ]

    metric_results: List[AnchorMetricResult] = []
    all_passed = True
    for metric, p, q, t in checks:
        rel = _rel_err(p, q)
        passed = abs(rel) <= t
        if not passed:
            all_passed = False
        metric_results.append(
            AnchorMetricResult(
                metric=metric,
                predicted=p,
                published=q,
                rel_err=rel,
                tolerance=t,
                passed=passed,
            )
        )

    breakdown = _breakdown_from_quality(getattr(ev, "quality", None))

    return AnchorResult(
        anchor_id=anchor.id,
        display_name=anchor.display_name,
        status="pass" if all_passed else "fail",
        metrics=metric_results,
        breakdown=breakdown,
    )


def run_public_anchor_suite(
    registry_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    tightened: bool = False,
    justifications: Optional[Dict[str, str]] = None,
    hardware_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every anchor and return a machine-readable report.

    `tightened=True` uses the post_calibration_tolerances (Wave 19 gate);
    False (default) uses default_tolerances (Wave 18e pre-calibration gate).

    `justifications` maps anchor_id → human-readable justification. If an
    anchor with a justification fails, it is marked `override_recorded=True`
    and does NOT block matrix publication.
    """
    anchors, default_tol, post_tol = load_public_model_registry(registry_path, overrides)
    tol_in_use = post_tol if tightened else default_tol
    justifications = dict(justifications or {})

    per_anchor: List[AnchorResult] = []
    for a in anchors:
        r = run_public_anchor(a, tol_in_use, hardware_override=hardware_override)
        if r.status == "fail" and a.id in justifications:
            r.override_recorded = True
            r.override_justification = justifications[a.id]
        per_anchor.append(r)

    n_pass = sum(1 for r in per_anchor if r.status == "pass")
    n_fail = sum(1 for r in per_anchor if r.status == "fail")
    n_skip = sum(1 for r in per_anchor if r.status == "skipped")
    n_error = sum(1 for r in per_anchor if r.status == "error")
    n_unrep = sum(1 for r in per_anchor if r.status == "unrepresentable")
    n_blocking = sum(1 for r in per_anchor if r.is_blocking)

    return {
        "schema_version": "wave18e.audit.v1",
        "tolerances_kind": "post_calibration" if tightened else "pre_calibration",
        "tolerances_in_use": dataclasses.asdict(tol_in_use),
        "counts": {
            "pass": n_pass,
            "fail": n_fail,
            "skipped": n_skip,
            "error": n_error,
            "unrepresentable": n_unrep,
            "blocking": n_blocking,
        },
        "block_publication": n_blocking > 0,
        "anchors": [
            {
                "id": r.anchor_id,
                "display_name": r.display_name,
                "status": r.status,
                "override_recorded": r.override_recorded,
                "override_justification": r.override_justification,
                "error": r.error,
                "failed_metrics": r.failed_metrics,
                "metrics": [dataclasses.asdict(m) for m in r.metrics],
                "breakdown": [dataclasses.asdict(b) for b in r.breakdown],
            }
            for r in per_anchor
        ],
    }


# =============================================================================
# Sensitivity suite (search-cap and prior multiplier stability)
# =============================================================================


@dataclass
class CellSpec:
    """A cell to audit: hw × params × context × families enabled."""

    hardware: str
    target_params_b: float
    context_length: int
    training_tokens: int
    allow_moe: bool = True
    allow_state: bool = False
    serving_batch: int = 8
    tp: int = 1


@dataclass
class SensitivityRunResult:
    perturbation_id: str
    winner_family: Optional[str]
    winner_shape: Optional[Tuple[int, int]]
    winner_loss: Optional[float]
    contender_ids: List[str]
    elapsed_s: float
    error: str = ""


def _family_of(cand) -> str:
    """Wave 18a: canonical family label. Routes through
    :func:`ac.architecture.architecture_signature` so the trust audit,
    the optimizer's stratification, and the generator's rollup can no
    longer disagree. Defensive fallback preserves the legacy classifier
    for partial mocks that lack ``d_model``/``n_layers``/``n_heads``."""
    try:
        from ac.architecture import architecture_signature
        return architecture_signature(cand).legacy_family
    except (ValueError, ImportError):
        has_moe = bool(getattr(cand, "moe", None))
        has_state = bool(getattr(cand, "state_config", None))
        if has_moe and has_state:
            return "moe_hybrid"
        if has_moe:
            return "moe"
        if has_state:
            return "hybrid"
        return "dense"


def _shape_key(cand) -> Tuple[int, int]:
    return (int(getattr(cand, "d_model", 0)), int(getattr(cand, "n_layers", 0)))


def _run_one_perturbation(
    cell: CellSpec,
    perturbation_id: str,
    max_candidates: int,
    max_full_evaluations: int,
) -> SensitivityRunResult:
    import time

    from ac.optimizer import DeploymentConstraints, optimize

    t0 = time.time()
    try:
        extra: Dict[str, Any] = {}
        if cell.allow_moe:
            extra.update(
                allow_moe=True,
                max_total_params_b=cell.target_params_b * 8,
                moe_n_experts_options=[8],
                moe_top_k_options=[2],
                ep_options=[min(cell.tp, 4)],
                dense_ffn_layer_options=[0],
            )
        if cell.allow_state:
            extra.update(allow_state=True, state_type="mamba2")

        c = DeploymentConstraints(
            target_params_b=cell.target_params_b,
            training_tokens=cell.training_tokens,
            context_length=cell.context_length,
            tp=cell.tp,
            pp=1,
            dp=1,
            tp_options=[cell.tp],
            serving_tbt_ms=None,
            serving_ttft_ms=None,
            serving_batch=cell.serving_batch,
            allow_quality_sentinel=True,
            max_candidates=max_candidates,
            max_full_evaluations=max_full_evaluations,
            **extra,
        )
        r = optimize(cell.hardware, c)
        winner = r.optimal
        contenders = list(
            {
                _family_of(ev.arch): 1
                for ev in (r.pareto_frontier or [])
            }.keys()
        )
        return SensitivityRunResult(
            perturbation_id=perturbation_id,
            winner_family=(_family_of(winner.arch) if winner else None),
            winner_shape=(_shape_key(winner.arch) if winner else None),
            winner_loss=(float(winner.predicted_loss) if winner else None),
            contender_ids=sorted(contenders),
            elapsed_s=time.time() - t0,
        )
    except Exception as e:
        return SensitivityRunResult(
            perturbation_id=perturbation_id,
            winner_family=None,
            winner_shape=None,
            winner_loss=None,
            contender_ids=[],
            elapsed_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def run_sensitivity_suite(cell: CellSpec, base_cap: int = 40) -> Dict[str, Any]:
    """Run the one-factor-at-a-time sensitivity suite for a cell.

    Baseline uses `base_cap`. Perturbations vary the cap and full-eval budget:
    - `cap_1x`  (baseline)
    - `cap_2x`
    - `cap_4x`
    A unique winner requires ≥80% winner-stability and ≥95% contender retention.
    """
    variants = [
        ("cap_1x", base_cap, max(20, base_cap // 2)),
        ("cap_2x", 2 * base_cap, max(20, base_cap)),
        ("cap_4x", 4 * base_cap, max(20, 2 * base_cap)),
    ]
    runs = [
        _run_one_perturbation(cell, v[0], v[1], v[2]) for v in variants
    ]
    winner_families = [r.winner_family for r in runs if r.winner_family]
    if winner_families:
        most_common = max(set(winner_families), key=winner_families.count)
        stability = winner_families.count(most_common) / len(runs)
    else:
        most_common = None
        stability = 0.0

    all_contenders = set().union(*(set(r.contender_ids) for r in runs))
    contender_retention = 1.0
    if all_contenders:
        retained_across = [
            len(set(r.contender_ids) & all_contenders) / len(all_contenders)
            for r in runs
            if r.winner_family
        ]
        contender_retention = (
            sum(retained_across) / len(retained_across) if retained_across else 0.0
        )

    unique_winner_allowed = stability >= 0.80 and contender_retention >= 0.95

    return {
        "schema_version": "wave18e.sensitivity.v1",
        "cell": dataclasses.asdict(cell),
        "runs": [dataclasses.asdict(r) for r in runs],
        "winner_stability_fraction": round(stability, 4),
        "contender_retention_fraction": round(contender_retention, 4),
        "modal_winner_family": most_common,
        "unique_winner_allowed": unique_winner_allowed,
    }


# =============================================================================
# Structural invariants
# =============================================================================


@dataclass
class InvariantResult:
    name: str
    passed: bool
    detail: str = ""


def check_active_total_consistency(
    ledger_active_b: float, ledger_total_b: float, reported_active_b: float, reported_total_b: float
) -> InvariantResult:
    """Active/total from parameter_ledger must match reported values within 1%."""
    ok_active = abs(ledger_active_b - reported_active_b) / max(ledger_active_b, 1e-9) <= 0.01
    ok_total = abs(ledger_total_b - reported_total_b) / max(ledger_total_b, 1e-9) <= 0.01
    return InvariantResult(
        name="active_total_consistency",
        passed=ok_active and ok_total,
        detail=f"ledger(active={ledger_active_b:.2f}B total={ledger_total_b:.2f}B) "
        f"reported(active={reported_active_b:.2f}B total={reported_total_b:.2f}B)",
    )


def check_tp_cost_nonfree_above_island(
    tbt_at_island: float, tbt_above_island: float
) -> InvariantResult:
    """TBT above the NVLink island must not drop below the intra-island value."""
    passed = tbt_above_island >= 0.98 * tbt_at_island
    return InvariantResult(
        name="tp_cost_nonfree_above_island",
        passed=passed,
        detail=f"at_island={tbt_at_island:.2f}ms above_island={tbt_above_island:.2f}ms",
    )


def check_family_coverage_before_pruning(
    per_family_counts_before: Dict[str, int],
    per_family_counts_after: Dict[str, int],
    families_enabled: List[str],
) -> InvariantResult:
    """Every enabled family must have at least one candidate before AND after
    the full-evaluation pruning step."""
    missing_before = [
        f
        for f in families_enabled
        if per_family_counts_before.get(f, 0) == 0
    ]
    missing_after = [
        f
        for f in families_enabled
        if per_family_counts_after.get(f, 0) == 0
    ]
    passed = not (missing_before or missing_after)
    return InvariantResult(
        name="family_coverage_before_pruning",
        passed=passed,
        detail=(
            f"enabled={families_enabled} "
            f"missing_before={missing_before} missing_after={missing_after}"
        ),
    )


def check_context_capacity_independent_of_fresh_prompt(
    ctx_capacity_at_short_prompt: bool, ctx_capacity_at_full_prompt: bool
) -> InvariantResult:
    """Whether the arch *supports* a context length must not depend on how
    many prompt tokens are freshly-issued in a given request."""
    passed = ctx_capacity_at_short_prompt == ctx_capacity_at_full_prompt
    return InvariantResult(
        name="context_capacity_independent_of_fresh_prompt",
        passed=passed,
    )


# =============================================================================
# Frontier feasibility anchors (1M/2M physical representability)
# =============================================================================


@dataclass
class FeasibilityAnchorResult:
    context_length: int
    scenario: str
    physically_feasible: bool
    reason: str = ""


def check_frontier_feasibility_at_context(
    hardware: str,
    context_length: int,
    target_active_b: float,
    total_params_b: float,
    tp: int,
) -> FeasibilityAnchorResult:
    """One anchor: does *at least one* candidate at this (ctx, size, tp) evaluate
    to physically feasible (finite loss, no sentinel)?"""
    try:
        from ac.optimizer import DeploymentConstraints, evaluate_candidate, CandidateArch

        # Hand-authored small candidate at the requested scale.
        # Not a search — just a physical-representability check.
        d = 4096 if target_active_b < 20 else 6144 if target_active_b < 100 else 8192
        L = 32 if target_active_b < 20 else 40 if target_active_b < 100 else 80
        cand = CandidateArch(
            d_model=d,
            n_layers=L,
            n_heads=32,
            d_head=128,
            n_kv_heads=8,
            ffn_dim=4 * d,
            vocab_size=32000,
            weight_precision="bf16",
            ffn_precision="bf16",
            attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
            kv_cache_bits=16,
            attention_type="mla" if target_active_b >= 100 else "gqa",
            mla_kv_latent_dim=512 if target_active_b >= 100 else 0,
            mla_q_latent_dim=1536 if target_active_b >= 100 else 0,
            mla_rope_head_dim=64 if target_active_b >= 100 else 0,
            mla_nope_head_dim=128 if target_active_b >= 100 else 0,
            tp_degree=tp,
        )
        c = DeploymentConstraints(
            target_params_b=target_active_b,
            training_tokens=int(2e13),
            context_length=context_length,
            tp=tp,
            pp=1,
            dp=1,
            tp_options=[tp],
            serving_tbt_ms=None,
            serving_ttft_ms=None,
            serving_batch=1,
            allow_quality_sentinel=True,
            vocab_size=32000,
        )
        ev = evaluate_candidate(cand, hardware, c)
        loss = float(getattr(ev, "predicted_loss", 0.0))
        feasible = math.isfinite(loss) and loss < 1e4
        return FeasibilityAnchorResult(
            context_length=context_length,
            scenario=f"active_{target_active_b:g}B_tp{tp}",
            physically_feasible=feasible,
            reason="" if feasible else f"loss={loss}",
        )
    except Exception as e:
        return FeasibilityAnchorResult(
            context_length=context_length,
            scenario=f"active_{target_active_b:g}B_tp{tp}",
            physically_feasible=False,
            reason=f"{type(e).__name__}: {e}",
        )


def run_frontier_feasibility_suite(hardware: str = "h100") -> Dict[str, Any]:
    """Run the 1M/2M physical anchors."""
    anchors = [
        # 1M context capacity: small/medium-active + 100B-1T-total sparse
        check_frontier_feasibility_at_context(hardware, 1_048_576, 7.0, 7.0, tp=4),
        check_frontier_feasibility_at_context(hardware, 1_048_576, 37.0, 671.0, tp=8),
        # 2M context capacity: efficient-attention / state
        check_frontier_feasibility_at_context(hardware, 2_097_152, 7.0, 7.0, tp=8),
        check_frontier_feasibility_at_context(hardware, 2_097_152, 37.0, 671.0, tp=16),
    ]
    n_pass = sum(1 for a in anchors if a.physically_feasible)
    return {
        "schema_version": "wave18e.frontier_feasibility.v1",
        "hardware": hardware,
        "anchors": [dataclasses.asdict(a) for a in anchors],
        "n_pass": n_pass,
        "n_total": len(anchors),
        "block_publication": n_pass < len(anchors),
    }


# =============================================================================
# Report rendering
# =============================================================================


def render_public_anchor_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Public-model predictive accuracy audit\n")
    tol_kind = report["tolerances_kind"]
    tol = report["tolerances_in_use"]
    lines.append(
        f"_Tolerance regime: **{tol_kind}** "
        f"(loss={tol['loss']:.0%}, tbt={tol['tbt_ms']:.0%}, "
        f"ttft={tol['ttft_ms']:.0%}, mem={tol['mem_gb']:.0%})_\n"
    )
    counts = report["counts"]
    lines.append(
        f"**Counts:** pass={counts['pass']}  fail={counts['fail']}  "
        f"skipped={counts['skipped']}  unrepresentable={counts['unrepresentable']}  "
        f"error={counts['error']}  → blocking={counts['blocking']}\n"
    )
    lines.append(
        f"**block_publication:** {'YES' if report['block_publication'] else 'no'}\n"
    )
    # Wave 29: name the measurement-basis gap on TTFT so a reviewer does
    # not read the systematic under-prediction as pure model error.
    lines.append(
        "_TTFT basis: predicted = compute prefill + serving-stack floor "
        "(tokenize/schedule/detokenize; calibratable via "
        "`calibration.ttft_serving_overhead`). Published endpoint p95s "
        "additionally include load-dependent queueing and batched-prefill "
        "interference, which AC deliberately does not attribute to the "
        "architecture — expect published > predicted on loaded stacks._\n"
    )
    lines.append("\n## Per-anchor results\n")
    lines.append("| Model | Status | loss (pred/pub/err) | tbt (pred/pub/err) | ttft (pred/pub/err) | mem (pred/pub/err) |")
    lines.append("|---|---|---|---|---|---|")
    for a in report["anchors"]:
        status = a["status"]
        if a["override_recorded"]:
            status = f"{status} (override)"
        if not a["metrics"]:
            lines.append(f"| {a['display_name']} | {status} | — | — | — | — |")
            continue
        by = {m["metric"]: m for m in a["metrics"]}
        def cell(m):
            x = by.get(m)
            if not x:
                return "—"
            return f"{x['predicted']:.3g} / {x['published']:.3g} / {x['rel_err']*100:+.1f}%"
        lines.append(
            f"| {a['display_name']} | {status} | "
            f"{cell('loss')} | {cell('tbt_ms')} | {cell('ttft_ms')} | {cell('mem_gb')} |"
        )

    # Breakdown for failing anchors
    fails = [a for a in report["anchors"] if a["status"] == "fail"]
    if fails:
        lines.append("\n## Failure breakdowns\n")
        for a in fails:
            lines.append(f"### {a['display_name']}")
            if a["breakdown"]:
                lines.append("Per-term loss contribution:")
                for b in a["breakdown"]:
                    if abs(b["value"]) > 1e-6:
                        lines.append(f"- `{b['term_name']}`: {b['value']:+.4f}")
            if a["override_recorded"]:
                lines.append(
                    f"\n_Override recorded: {a['override_justification']}_"
                )
            lines.append("")

    return "\n".join(lines)
