"""
Baseline-aware Pareto modifier for Architecture Compiler v0.5.

The greenfield optimizer remains the default compiler ability. This module is
an additional local-search layer that starts from a supplied dense baseline and
asks which nearby architecture changes move it toward the Pareto frontier.
"""

import copy
import csv
import io
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .lattice_engine import HARDWARE as LATTICE_HW, estimate_params
    from .optimizer import (
        CandidateArch, DeploymentConstraints, EvaluatedCandidate,
        evaluate_candidate, get_precision_configs_for_hardware,
    )
    from .schema import build_config
    from .baseline import BaselineModel
except ImportError:
    from lattice_engine import HARDWARE as LATTICE_HW, estimate_params
    from optimizer import (
        CandidateArch, DeploymentConstraints, EvaluatedCandidate,
        evaluate_candidate, get_precision_configs_for_hardware,
    )
    from schema import build_config
    from baseline import BaselineModel


@dataclass
class ModificationChange:
    """One changed knob relative to the baseline."""

    field: str
    old: Any
    new: Any

    @property
    def label(self) -> str:
        return f"{self.field}: {self.old} -> {self.new}"


@dataclass
class ModifierRecord:
    """An evaluated baseline or local modification."""

    evaluated: EvaluatedCandidate
    tp: int
    changes: List[ModificationChange] = field(default_factory=list)
    risk_score: float = 0.0
    risk_label: str = "baseline"
    kv_cache_gb: float = 0.0
    is_baseline: bool = False
    quality_preserving: bool = False
    move_class: str = "architecture"

    @property
    def change_summary(self) -> str:
        if self.is_baseline or not self.changes:
            return "baseline"
        return "; ".join(c.label for c in self.changes)

    @property
    def quality_risk_pct(self) -> float:
        return 0.0


@dataclass
class ModifierResult:
    """Output of baseline-aware local Pareto modification."""

    baseline: ModifierRecord
    selected: ModifierRecord
    pareto_frontier: List[ModifierRecord] = field(default_factory=list)
    performance_dominating: List[ModifierRecord] = field(default_factory=list)
    near_dominating: List[ModifierRecord] = field(default_factory=list)
    all_records: List[ModifierRecord] = field(default_factory=list)
    feasible_records: List[ModifierRecord] = field(default_factory=list)
    baseline_model: Optional[BaselineModel] = None
    hardware: str = ""
    constraints: Optional[DeploymentConstraints] = None
    quality_risk_budget_pct: float = 1.0
    allow_quality_spending: bool = False
    search_time_sec: float = 0.0
    candidates_generated: int = 0
    candidates_evaluated: int = 0

    @property
    def baseline_risk_aware_pareto_optimal(self) -> bool:
        return self.baseline in self.pareto_frontier

    @property
    def baseline_performance_dominated(self) -> bool:
        return bool(self.performance_dominating)


def run_modifier_search(
    baseline_model: BaselineModel,
    hw_name: str,
    constraints: DeploymentConstraints,
    tp_options: Optional[List[int]] = None,
    quality_risk_budget_pct: float = 1.0,
    allow_quality_spending: bool = False,
    top_near_dominating: int = 12,
) -> ModifierResult:
    """Evaluate a baseline and nearby architecture modifications."""

    t0 = time.time()
    base = baseline_model.candidate
    if tp_options is None or not tp_options:
        tp_options = [constraints.tp]
    tp_options = sorted(set(int(tp) for tp in tp_options if int(tp) >= 1))

    if constraints.target_params_b <= 0:
        constraints = copy.deepcopy(constraints)
        constraints.target_params_b = base.total_params / 1e9

    # Evaluate the baseline at the user-selected/default TP.
    base_constraints = copy.deepcopy(constraints)
    base_constraints.tp = constraints.tp
    base_eval = evaluate_candidate(base, hw_name, base_constraints)
    baseline_record = _make_record(
        base_eval, base, base, constraints.tp, constraints.tp,
        constraints, is_baseline=True
    )

    candidates = _generate_local_candidates(base, hw_name, constraints, tp_options)
    records: List[ModifierRecord] = [baseline_record]
    seen = {
        _record_key(base, constraints.tp)
    }

    for cand, tp in candidates:
        key = _record_key(cand, tp)
        if key in seen:
            continue
        seen.add(key)
        eval_constraints = copy.deepcopy(constraints)
        eval_constraints.tp = tp
        try:
            ev = evaluate_candidate(cand, hw_name, eval_constraints)
        except Exception:
            continue
        records.append(_make_record(ev, base, cand, constraints.tp, tp, constraints))

    feasible = [r for r in records if r.evaluated.meets_constraints]
    pareto = compute_modifier_pareto(feasible)

    perf_dom = [
        r for r in feasible
        if not r.is_baseline and _performance_dominates(baseline_record, r)
    ]
    perf_dom.sort(key=lambda r: _modifier_sort_key(r, baseline_record))

    near = [
        r for r in feasible
        if not r.is_baseline
        and r not in perf_dom
        and (
            r.quality_preserving
            or (allow_quality_spending and _within_quality_budget(r, baseline_record, quality_risk_budget_pct))
        )
        and _improves_any_resource(r, baseline_record)
    ]
    near.sort(key=lambda r: _modifier_sort_key(r, baseline_record))

    selected = _select_modified_candidate(
        pareto, baseline_record, quality_risk_budget_pct, allow_quality_spending
    )

    return ModifierResult(
        baseline=baseline_record,
        selected=selected,
        pareto_frontier=pareto,
        performance_dominating=perf_dom[:top_near_dominating],
        near_dominating=near[:top_near_dominating],
        all_records=records,
        feasible_records=feasible,
        baseline_model=baseline_model,
        hardware=hw_name,
        constraints=constraints,
        quality_risk_budget_pct=quality_risk_budget_pct,
        allow_quality_spending=allow_quality_spending,
        search_time_sec=round(time.time() - t0, 2),
        candidates_generated=len(candidates),
        candidates_evaluated=len(records),
    )


def _generate_local_candidates(
    base: CandidateArch,
    hw_name: str,
    constraints: DeploymentConstraints,
    tp_options: List[int],
) -> List[Tuple[CandidateArch, int]]:
    """Generate product of v0.5 local mutation options around the baseline.

    Wave 7a.3 audit note. The optimizer's `_filter_lattice_to_budget`
    (added in Wave 5) bounds `generate_candidates()` to the N points
    closest to the Chinchilla-anchor shape in (d_model, d_head) log-space.
    The modifier does NOT pass through that filter — it constructs
    CandidateArch instances directly from `base.d_model`, `base.d_head`,
    `base.n_heads` (mutating only n_layers, n_kv_heads, kv_bits,
    ffn_precision, ffn_dim) and evaluates them via the throughput/quality
    models directly. Consequence: a baseline whose (d_model, d_head)
    sits outside the lattice-filter neighborhood is still freely
    explored by the modifier; the filter cannot exclude its proposals.
    No code change needed for Wave 7. (Note for future work: if we ever
    re-route modifier candidates through `generate_candidates()` or
    `optimize()`, we'd need to either widen the lattice filter for that
    call or have the modifier pin a per-call anchor at the baseline's
    own shape; both are out of scope today.)"""

    candidates = []
    kv_options = constraints.kv_bits_options or [16, 8, 4]
    n_kv_options = _valid_kv_head_options(base.n_heads)
    layer_options = [
        base.n_layers + delta for delta in (-2, -1, 0, 1, 2)
        if base.n_layers + delta >= 4
    ]

    ffn_precision_options = ["bf16"]
    hw_prec = get_precision_configs_for_hardware(hw_name)
    requested = constraints.precision_configs or hw_prec
    if "ffn_fp8" in requested and "ffn_fp8" in hw_prec:
        ffn_precision_options.append("fp8")

    # Wave 19 (loop finding L4): carry the baseline's architecture-family
    # fields into every variant. The old constructor dropped moe / ep /
    # MLA / local:global entirely, so an MoE baseline produced dense-ified
    # 100B+ variants that all failed validation — modifier mode silently
    # evaluated exactly ONE candidate (the baseline) for every MoE config.
    def _family_kwargs() -> dict:
        kw: dict = {}
        if getattr(base, "moe", None):
            kw["moe"] = copy.deepcopy(base.moe)
            kw["ep_degree"] = int(getattr(base, "ep_degree", 2) or 2)
            kw["moe_style"] = getattr(base, "moe_style", "fine")
            kw["n_dense_ffn_layers"] = int(
                getattr(base, "n_dense_ffn_layers", 0) or 0)
        if getattr(base, "attention_type", "full") != "full":
            kw["attention_type"] = base.attention_type
            for f in ("mla_kv_latent_dim", "mla_q_latent_dim",
                      "mla_rope_head_dim", "mla_nope_head_dim"):
                if getattr(base, f, None) is not None:
                    kw[f] = getattr(base, f)
        for f in ("swa_window", "n_local_attn_layers", "cp_degree",
                  "cp_method"):
            v = getattr(base, f, None)
            if v:
                kw[f] = v
        return kw

    def _params_for(cand: CandidateArch) -> int:
        """MoE-aware total params (estimate_params is dense-only; using it
        on an MoE variant books expert_dim as a dense FFN width)."""
        dense_total = estimate_params(
            cand.d_model, cand.n_heads, cand.d_head,
            cand.ffn_dim, cand.n_layers, cand.n_kv_heads, cand.vocab_size,
        )
        moe = getattr(cand, "moe", None)
        if not moe:
            return dense_total
        # Replace the dense-FFN mass with the expert + shared mass.
        per_dense_ffn = 3 * cand.d_model * cand.ffn_dim
        n_experts = int(moe.get("n_experts", 1))
        expert_dim = int(moe.get("expert_dim", cand.ffn_dim))
        shared = moe.get("shared_expert")
        shared_dim = int(shared.get("ffn_dim", 0)) if isinstance(shared, dict) else 0
        per_moe_ffn = 3 * cand.d_model * (n_experts * expert_dim + shared_dim)
        return dense_total + (per_moe_ffn - per_dense_ffn) * cand.n_layers

    def _active_for(cand: CandidateArch) -> int:
        moe = getattr(cand, "moe", None)
        dense_total = estimate_params(
            cand.d_model, cand.n_heads, cand.d_head,
            cand.ffn_dim, cand.n_layers, cand.n_kv_heads, cand.vocab_size,
        )
        if not moe:
            return dense_total
        per_dense_ffn = 3 * cand.d_model * cand.ffn_dim
        top_k = int(moe.get("top_k", 1))
        expert_dim = int(moe.get("expert_dim", cand.ffn_dim))
        shared = moe.get("shared_expert")
        shared_dim = int(shared.get("ffn_dim", 0)) if isinstance(shared, dict) else 0
        per_active_ffn = 3 * cand.d_model * (top_k * expert_dim + shared_dim)
        return dense_total + (per_active_ffn - per_dense_ffn) * cand.n_layers

    base_is_moe = bool(getattr(base, "moe", None))
    for tp in tp_options:
        # For MoE baselines the local FFN knob is the EXPERT width.
        _base_width = (int(base.moe.get("expert_dim", base.ffn_dim))
                       if base_is_moe else base.ffn_dim)
        ffn_options = _nearby_ffn_dims(_base_width, hw_name, tp)
        for n_kv in n_kv_options:
            for kv_bits in kv_options:
                for ffn_precision in ffn_precision_options:
                    for ffn_dim in ffn_options:
                        for n_layers in layer_options:
                            fam_kw = _family_kwargs()
                            if base_is_moe:
                                fam_kw["moe"] = dict(fam_kw["moe"])
                                fam_kw["moe"]["expert_dim"] = int(ffn_dim)
                            cand = CandidateArch(
                                d_model=base.d_model,
                                n_layers=n_layers,
                                n_heads=base.n_heads,
                                d_head=base.d_head,
                                n_kv_heads=n_kv,
                                ffn_dim=(base.ffn_dim if base_is_moe
                                         else ffn_dim),
                                vocab_size=base.vocab_size,
                                weight_precision=base.weight_precision,
                                ffn_precision=ffn_precision,
                                attn_precision=copy.deepcopy(base.attn_precision),
                                kv_cache_bits=int(kv_bits),
                                **fam_kw,
                            )
                            cand.total_params = _params_for(cand)
                            cand.total_params_b = round(cand.total_params / 1e9, 2)
                            if not _candidate_valid(cand, hw_name, constraints,
                                                    tp, base=base):
                                continue
                            if base_is_moe:
                                cand.active_params = _active_for(cand)
                                cand.active_params_b = round(
                                    cand.active_params / 1e9, 2)
                            candidates.append((cand, tp))

    return candidates


def _candidate_valid(
    cand: CandidateArch,
    hw_name: str,
    constraints: DeploymentConstraints,
    tp: int,
    base: Optional[CandidateArch] = None,
) -> bool:
    """Cheap structural checks before expensive evaluation."""

    # Wave 19 (loop finding L4): the hard `d_model == n_heads * d_head`
    # check silently rejected EVERY variant of a baseline with decoupled
    # attention width (GPT-OSS: d_model 2880, 64 heads x 64 = 4096) — the
    # same invariant Wave 18g already relaxed in the schema loader. When
    # the BASELINE itself is decoupled, variants inherit its geometry and
    # only the schema's ±4x typo bound applies.
    attn_width = cand.n_heads * cand.d_head
    if base is not None and base.d_model == base.n_heads * base.d_head:
        if cand.d_model != attn_width:
            return False
    else:
        if not (0.25 * cand.d_model <= attn_width <= 4.0 * cand.d_model):
            return False
    if cand.n_heads % cand.n_kv_heads != 0:
        return False
    if cand.n_kv_heads < tp and tp % cand.n_kv_heads != 0:
        return False
    if cand.d_model % tp != 0 or cand.n_heads % tp != 0 or cand.ffn_dim % tp != 0:
        return False
    if constraints.pp > 1 and cand.n_layers % constraints.pp != 0:
        return False

    # Wave 19 (L4): the param-band check must use the candidate's own
    # (family-aware) ledger, not the dense estimate — for an MoE variant
    # the dense formula books only the per-expert width and lands ~40x
    # under the target band, rejecting everything.
    total = float(getattr(cand, "total_params", 0) or 0)
    if total <= 0:
        total = estimate_params(
            cand.d_model, cand.n_heads, cand.d_head, cand.ffn_dim,
            cand.n_layers, cand.n_kv_heads, cand.vocab_size,
        )
    target = constraints.target_params_b * 1e9
    lo = target * (1 - constraints.param_tolerance)
    hi = target * (1 + constraints.param_tolerance)
    if total < lo or total > hi:
        return False

    hw = LATTICE_HW.get(hw_name)
    if hw is None:
        return False
    tile = hw.tiles.get("bf16") or next(iter(hw.tiles.values()))
    return (
        cand.d_model % tile.cta_k == 0
        and cand.d_model % tile.cta_n == 0
        and cand.d_head % tile.cta_k == 0
        and cand.d_head % tile.cta_n == 0
        and cand.ffn_dim % tile.cta_n == 0
    )


def _valid_kv_head_options(n_heads: int) -> List[int]:
    return sorted([n for n in range(1, n_heads + 1) if n_heads % n == 0])


def _nearby_ffn_dims(base_ffn: int, hw_name: str, tp: int) -> List[int]:
    hw = LATTICE_HW[hw_name]
    tile = hw.tiles.get("bf16") or next(iter(hw.tiles.values()))
    quantum = math.lcm(int(tile.cta_n), int(max(tp, 1)))
    values = set()
    for frac in (-0.10, -0.05, 0.0, 0.05, 0.10):
        raw = base_ffn * (1 + frac)
        rounded = max(quantum, int(round(raw / quantum) * quantum))
        values.add(rounded)
    return sorted(values)


def _make_record(
    ev: EvaluatedCandidate,
    base: CandidateArch,
    cand: CandidateArch,
    base_tp: int,
    tp: int,
    constraints: DeploymentConstraints,
    is_baseline: bool = False,
) -> ModifierRecord:
    changes = [] if is_baseline else _changes(base, cand, base_tp, tp)
    risk_score, risk_label = _risk_for_changes(changes, cand)
    quality_preserving = is_baseline or _changes_preserve_model_quality(changes)
    move_class = _move_class_for_changes(changes)
    if is_baseline:
        risk_score, risk_label, move_class = 0.0, "baseline", "baseline"
    return ModifierRecord(
        evaluated=ev,
        tp=tp,
        changes=changes,
        risk_score=risk_score,
        risk_label=risk_label,
        kv_cache_gb=_kv_cache_gb(cand, constraints, tp),
        is_baseline=is_baseline,
        quality_preserving=quality_preserving,
        move_class=move_class,
    )


def _changes(
    base: CandidateArch,
    cand: CandidateArch,
    base_tp: int,
    tp: int,
) -> List[ModificationChange]:
    changes = []
    for field_name in ("n_layers", "n_kv_heads", "ffn_dim", "ffn_precision", "kv_cache_bits"):
        old = getattr(base, field_name)
        new = getattr(cand, field_name)
        if old != new:
            changes.append(ModificationChange(field_name, old, new))
    if base_tp != tp:
        changes.append(ModificationChange("tp", base_tp, tp))
    return changes


def _risk_for_changes(
    changes: List[ModificationChange],
    cand: CandidateArch,
) -> Tuple[float, str]:
    score = 0.0
    for ch in changes:
        if ch.field == "kv_cache_bits":
            if int(ch.new) == 8:
                score += 0.5
            elif int(ch.new) == 4:
                score += 1.5
        elif ch.field == "ffn_precision" and ch.new == "fp8":
            score += 0.5
        elif ch.field == "n_kv_heads":
            if int(ch.new) > int(ch.old):
                score += 0.25
            else:
                group_size = cand.n_heads // int(ch.new)
                if int(ch.new) == 1:
                    score += 2.0
                elif group_size <= 8:
                    score += 0.75
                else:
                    score += 1.25
        elif ch.field == "n_layers":
            score += 0.75 * abs(int(ch.new) - int(ch.old))
        elif ch.field == "ffn_dim":
            frac = abs(float(ch.new) - float(ch.old)) / max(float(ch.old), 1.0)
            score += 0.5 if frac <= 0.05 else 1.0
        elif ch.field == "tp":
            score += 0.25

    if score <= 1.0:
        label = "low"
    elif score <= 2.5:
        label = "medium"
    else:
        label = "high"
    return round(score, 2), label


def _changes_preserve_model_quality(changes: List[ModificationChange]) -> bool:
    """Return True for changes that keep the learned function/numerics intact."""

    if not changes:
        return True
    quality_preserving_fields = {"tp"}
    return all(ch.field in quality_preserving_fields for ch in changes)


def _move_class_for_changes(changes: List[ModificationChange]) -> str:
    if not changes:
        return "baseline"
    if _changes_preserve_model_quality(changes):
        return "deployment"
    if any(ch.field in {"ffn_precision", "kv_cache_bits"} for ch in changes):
        return "precision"
    if any(ch.field in {"n_layers", "ffn_dim", "n_kv_heads"} for ch in changes):
        return "architecture"
    return "modifier"


def _kv_precision(kv_bits: int) -> str:
    if kv_bits == 16:
        return "bf16"
    if kv_bits == 8:
        return "int8"
    if kv_bits == 4:
        return "fp4"
    return "bf16"


def _kv_cache_gb(cand: CandidateArch, constraints: DeploymentConstraints, tp: int) -> float:
    kv_bpe = {"bf16": 2, "int8": 1, "fp4": 0.5}.get(_kv_precision(cand.kv_cache_bits), 2)
    context = constraints.prompt_len or constraints.context_length
    # Keep this aligned with the current throughput model, which estimates
    # single active decode-stream memory rather than full scheduler residency.
    batch = 1
    layers_per_stage = cand.n_layers // max(constraints.pp, 1)
    kv_heads_per_gpu = max(1, math.ceil(cand.n_kv_heads / max(tp, 1)))
    bytes_total = (
        batch * context * layers_per_stage * 2 * kv_heads_per_gpu * cand.d_head * kv_bpe
    )
    return bytes_total / (1024**3)


def compute_modifier_pareto(records: List[ModifierRecord]) -> List[ModifierRecord]:
    """Compute a risk-aware Pareto frontier for modifier records."""

    frontier = []
    for rec in records:
        dominated = False
        for other in records:
            if other is rec:
                continue
            if _dominates(rec, other):
                dominated = True
                break
        if not dominated:
            frontier.append(rec)
    frontier.sort(key=lambda r: (
        r.evaluated.predicted_loss,
        r.evaluated.serving_tbt_ms,
        -r.evaluated.training_tps,
        r.risk_score,
    ))
    return frontier


def _objectives(rec: ModifierRecord, include_risk: bool = True) -> Tuple[float, ...]:
    ev = rec.evaluated
    objs = [
        ev.predicted_loss,
        ev.serving_tbt_ms,
        ev.throughput.prefill_time_ms,
        -ev.training_tps,
        ev.memory_per_gpu_gb,
        rec.kv_cache_gb,
    ]
    if include_risk:
        objs.append(rec.risk_score)
    return tuple(objs)


def _dominates(a: ModifierRecord, b: ModifierRecord) -> bool:
    """Return True if b dominates a."""

    objs_a = _objectives(a, include_risk=True)
    objs_b = _objectives(b, include_risk=True)
    better = False
    for oa, ob in zip(objs_a, objs_b):
        if ob > oa:
            return False
        if ob < oa:
            better = True
    return better


def _performance_dominates(a: ModifierRecord, b: ModifierRecord) -> bool:
    """Dominance ignoring risk, useful for baseline delta reporting."""

    objs_a = _objectives(a, include_risk=False)
    objs_b = _objectives(b, include_risk=False)
    better = False
    for oa, ob in zip(objs_a, objs_b):
        if ob > oa:
            return False
        if ob < oa:
            better = True
    return better


def _quality_risk_pct(rec: ModifierRecord, baseline: ModifierRecord) -> float:
    base_loss = baseline.evaluated.predicted_loss
    if base_loss <= 0:
        return 0.0
    return ((rec.evaluated.predicted_loss - base_loss) / base_loss) * 100


def _within_quality_budget(
    rec: ModifierRecord,
    baseline: ModifierRecord,
    quality_risk_budget_pct: float,
) -> bool:
    return _quality_risk_pct(rec, baseline) <= quality_risk_budget_pct


def _improves_any_resource(rec: ModifierRecord, baseline: ModifierRecord) -> bool:
    ev = rec.evaluated
    base = baseline.evaluated
    return (
        ev.serving_tbt_ms < base.serving_tbt_ms
        or ev.throughput.prefill_time_ms < base.throughput.prefill_time_ms
        or ev.training_tps > base.training_tps
        or ev.memory_per_gpu_gb < base.memory_per_gpu_gb
        or rec.kv_cache_gb < baseline.kv_cache_gb
    )


def _modifier_score(rec: ModifierRecord, baseline: ModifierRecord) -> float:
    ev = rec.evaluated
    base = baseline.evaluated
    score = 0.0
    if base.serving_tbt_ms > 0:
        score += (base.serving_tbt_ms - ev.serving_tbt_ms) / base.serving_tbt_ms * 100
    if base.throughput.prefill_time_ms > 0:
        score += 0.5 * (base.throughput.prefill_time_ms - ev.throughput.prefill_time_ms) / base.throughput.prefill_time_ms * 100
    if base.training_tps > 0:
        score += 0.25 * (ev.training_tps - base.training_tps) / base.training_tps * 100
    if base.memory_per_gpu_gb > 0:
        score += 0.5 * (base.memory_per_gpu_gb - ev.memory_per_gpu_gb) / base.memory_per_gpu_gb * 100
    score -= rec.risk_score * 5
    score -= max(_quality_risk_pct(rec, baseline), 0) * 10
    return score


def _modifier_sort_key(rec: ModifierRecord, baseline: ModifierRecord) -> Tuple[float, float, float, float]:
    return (
        -_modifier_score(rec, baseline),
        rec.risk_score,
        rec.evaluated.predicted_loss,
        rec.evaluated.serving_tbt_ms,
    )


def _select_modified_candidate(
    pareto: List[ModifierRecord],
    baseline: ModifierRecord,
    quality_risk_budget_pct: float,
    allow_quality_spending: bool,
) -> ModifierRecord:
    # Tier 1 — "free wins": candidates that performance-dominate the baseline
    # AND don't worsen predicted quality beyond a tiny tolerance. These are
    # strict Pareto improvements and should always be preferred over a
    # quality-spending change, regardless of allow_quality_spending.
    quality_tolerance_pct = 0.05  # treat <=0.05% loss drift as noise
    free_wins = [
        r for r in pareto
        if not r.is_baseline
        and r.evaluated.meets_constraints
        and _performance_dominates(baseline, r)
        and _quality_risk_pct(r, baseline) <= quality_tolerance_pct
    ]
    if free_wins:
        free_wins.sort(key=lambda r: _modifier_sort_key(r, baseline))
        return free_wins[0]

    # Tier 2 — strictly quality-preserving deployment edits (TP/EP placement,
    # etc.) that improve at least one resource axis.
    pool = [
        r for r in pareto
        if not r.is_baseline
        and r.evaluated.meets_constraints
        and r.quality_preserving
        and _improves_any_resource(r, baseline)
    ]
    if pool:
        pool.sort(key=lambda r: _modifier_sort_key(r, baseline))
        return pool[0]

    # Tier 3 — quality-spending changes within the user's risk budget.
    if allow_quality_spending:
        pool = [
            r for r in pareto
            if not r.is_baseline
            and r.evaluated.meets_constraints
            and _within_quality_budget(r, baseline, quality_risk_budget_pct)
            and _improves_any_resource(r, baseline)
        ]
        if pool:
            pool.sort(key=lambda r: _modifier_sort_key(r, baseline))
            return pool[0]

    return baseline


def _record_key(cand: CandidateArch, tp: int) -> Tuple[Any, ...]:
    return (
        cand.d_model, cand.n_layers, cand.n_heads, cand.d_head,
        cand.n_kv_heads, cand.ffn_dim, cand.weight_precision,
        cand.ffn_precision, tuple(sorted(cand.attn_precision.items())),
        cand.kv_cache_bits, tp,
    )


def modifier_result_to_config(result: ModifierResult) -> Dict[str, Any]:
    """Build compiler JSON for the selected modifier candidate."""

    rec = result.selected
    c = rec.evaluated.arch
    con = result.constraints
    terms = getattr(rec.evaluated.quality, "terms", {})
    arch_term = terms.get("architecture_residual")
    precision_term = terms.get("precision_residual")
    risk_term = terms.get("risk_residual")
    assert con is not None
    return build_config(
        d_model=c.d_model,
        n_layers=c.n_layers,
        n_heads=c.n_heads,
        d_head=c.d_head,
        n_kv_heads=c.n_kv_heads,
        ffn_dim=c.ffn_dim,
        vocab_size=c.vocab_size,
        weight_precision=c.weight_precision,
        attn_precision=c.attn_precision,
        ffn_precision=c.ffn_precision,
        kv_cache_bits=c.kv_cache_bits,
        tp=rec.tp,
        pp=con.pp,
        dp=con.dp,
        hardware_name=result.hardware,
        input_constraints={
            "mode": "baseline_modifier",
            "baseline": result.baseline_model.name if result.baseline_model else "",
            "target_params": f"{con.target_params_b:.2f}B",
            "training_tokens": f"{con.training_tokens / 1e12:.1f}T",
            "context_length": con.context_length,
            "prompt_len": con.prompt_len,
            "output_len": con.output_len,
            "serving_tbt_ms": con.serving_tbt_ms,
            "serving_ttft_ms": con.serving_ttft_ms,
            "serving_batch": con.serving_batch,
            "quality_risk_budget_pct": result.quality_risk_budget_pct,
            "allow_quality_spending": result.allow_quality_spending,
        },
        predicted={
            "quality_rank_score": round(-rec.evaluated.predicted_loss, 4),
            "predicted_loss": round(rec.evaluated.predicted_loss, 4),
            "training_throughput_tokens_per_sec": round(rec.evaluated.training_tps),
            "serving_tbt_ms": round(rec.evaluated.serving_tbt_ms, 1),
            "serving_ttft_ms": round(rec.evaluated.throughput.prefill_time_ms, 1),
            "memory_per_gpu_gb": round(rec.evaluated.memory_per_gpu_gb, 1),
            "kv_cache_gb": round(rec.kv_cache_gb, 2),
            "risk_score": rec.risk_score,
            "risk_label": rec.risk_label,
            "quality_preserving": rec.quality_preserving,
            "move_class": rec.move_class,
            "baseline_delta_loss_pct": round(_quality_risk_pct(rec, result.baseline), 3),
            "baseline_change_summary": rec.change_summary,
            "confidence": rec.evaluated.quality.confidence,
            "scaling_spine_loss": round(rec.evaluated.quality.chinchilla_baseline, 4),
            "spine_active_params_b": round(getattr(rec.evaluated.quality, "spine_active_params", 0) / 1e9, 3),
            "total_residual_pct": round(rec.evaluated.quality.total_penalty_fraction * 100, 2),
            "architecture_residual_pct": round((arch_term.value if arch_term else 0.0) * 100, 3),
            "precision_residual_pct": round((precision_term.value if precision_term else 0.0) * 100, 3),
            "risk_uncertainty_pct": round((risk_term.uncertainty if risk_term else 0.0) * 100, 3),
            "total_penalty_pct": round(rec.evaluated.quality.total_penalty_fraction * 100, 2),
            "dominant_penalty": rec.evaluated.quality.dominant_penalty,
            "uncertainty_total_pct": round(getattr(rec.evaluated.quality, "uncertainty_total", 0.0) * 100, 2),
            "uncertainty_breakdown": {
                k: round(v * 100, 3)
                for k, v in getattr(rec.evaluated.quality, "uncertainty_breakdown", {}).items()
            },
            "quality_model_version": getattr(rec.evaluated.quality, "quality_model_version", "quality_v0"),
            "quality_terms": {
                k: {
                    "value_pct": round(v.value * 100, 4),
                    "uncertainty_pct": round(v.uncertainty * 100, 4),
                    "confidence": v.confidence,
                    "source": v.source,
                    "notes": v.notes,
                    "features": v.features,
                }
                for k, v in getattr(rec.evaluated.quality, "terms", {}).items()
                if v.confidence != "not_applicable" or abs(v.value) > 0 or v.uncertainty > 0
            },
            "binding_serving_regime": rec.evaluated.binding_serving_regime,
        },
        search_stats={
            "modifier_candidates_generated": result.candidates_generated,
            "modifier_candidates_evaluated": result.candidates_evaluated,
            "modifier_feasible": len(result.feasible_records),
            "modifier_pareto_size": len(result.pareto_frontier),
            "search_time_sec": result.search_time_sec,
        },
    )


def modifier_pareto_to_csv(result: ModifierResult) -> str:
    """Serialize modifier Pareto frontier to CSV."""

    output = io.StringIO()
    writer = csv.writer(output)
    # Schema parity with greenfield pareto.csv: include the loss CI band and
    # total uncertainty percentage so the "treat two rows as
    # quality-equivalent if their loss CIs overlap" recipe from the README
    # works on the modifier output too.
    writer.writerow([
        "rank", "selected", "is_baseline", "changes", "risk_label", "risk_score",
        "quality_preserving", "move_class", "quality_risk_pct", "tp", "d_model", "n_layers", "n_heads", "d_head",
        "n_kv_heads", "ffn_dim", "ffn_precision", "kv_bits", "params_B",
        "predicted_loss", "loss_ci_low", "loss_ci_high", "uncertainty_total_pct",
        "training_tps", "serving_tbt_ms", "serving_ttft_ms",
        "memory_gb", "kv_cache_gb", "confidence",
    ])
    for idx, rec in enumerate(result.pareto_frontier, 1):
        c = rec.evaluated.arch
        loss = float(rec.evaluated.predicted_loss)
        # uncertainty_total is stored as a fraction (e.g. 0.031 = 3.1%).
        u_frac = float(getattr(rec.evaluated.quality, "uncertainty_total", 0.0) or 0.0)
        half = abs(loss) * u_frac
        loss_ci_low = loss - half
        loss_ci_high = loss + half
        writer.writerow([
            idx,
            rec is result.selected,
            rec.is_baseline,
            rec.change_summary,
            rec.risk_label,
            rec.risk_score,
            rec.quality_preserving,
            rec.move_class,
            round(_quality_risk_pct(rec, result.baseline), 3),
            rec.tp,
            c.d_model,
            c.n_layers,
            c.n_heads,
            c.d_head,
            c.n_kv_heads,
            c.ffn_dim,
            c.ffn_precision,
            c.kv_cache_bits,
            c.total_params_b,
            round(loss, 4),
            round(loss_ci_low, 4),
            round(loss_ci_high, 4),
            round(u_frac * 100, 3),
            round(rec.evaluated.training_tps),
            round(rec.evaluated.serving_tbt_ms, 2),
            round(rec.evaluated.throughput.prefill_time_ms, 2),
            round(rec.evaluated.memory_per_gpu_gb, 2),
            round(rec.kv_cache_gb, 3),
            rec.evaluated.quality.confidence,
        ])
    return output.getvalue()


def quality_risk_pct(rec: ModifierRecord, baseline: ModifierRecord) -> float:
    """Public helper for report generation."""

    return _quality_risk_pct(rec, baseline)


def modifier_score(rec: ModifierRecord, baseline: ModifierRecord) -> float:
    """Public helper for report generation."""

    return _modifier_score(rec, baseline)
