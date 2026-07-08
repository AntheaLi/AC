"""
Quality Stress Reporter — 7-axis named decomposition of the quality residual.

Reads the existing v1 modular quality backbone (TermResult-tagged sub-terms)
and re-exposes them through a fixed-schema vector that mirrors StressVector
on the throughput side.

Axes (per instruction §3):
    shape_law_loss        — width / depth / MLP-attention shape residuals
    attention_residual    — d_head, query_heads, kv_heads, gqa, bottleneck
    moe_residual          — terms["moe_residual"].value
    state_residual        — terms["state_residual"].value  (SIGNED, see below)
    precision_loss        — terms["precision_residual"].value
    fa_underrun           — FlashAttention SRAM-underrun penalty (v1-fix Part 3)
    context_extrapolation — placeholder (0 until trained-context tracking lands)

Values are **fractional residuals relative to the spine** — same units as
the underlying TermResult.value field. Multiplying by `chinchilla_baseline`
gives absolute loss-proxy contributions; we surface both.

Sign convention (post-Wave-5):
  Most axes are *penalties* and stay non-negative. The exception is
  `state_residual`. After Wave 5 the state-residual sub-term can return a
  **negative** value for long-context Jamba-/Samba-like hybrids where the
  state mechanism gives a quality benefit relative to a pure-attention
  baseline at the same active params and tokens. Concretely:
      state_residual > 0  → state added a quality cost (e.g. p_attn too low
                            at short ctx, or out-of-band hybrid mix).
      state_residual == 0 → not applicable (dense / MoE-only candidate) or
                            exactly at the crossover.
      state_residual < 0  → state contributed a quality benefit at long ctx
                            (this is the Wave-5 sign change). Consumers that
                            previously assumed a non-negative penalty must
                            preserve the sign — clipping to ≥ 0 would erase
                            the benefit signal.
  `axis_value("state_residual")` returns the raw signed value so a
  descending sort by "stress" puts the *largest cost* candidates first and
  a sort by `absolute["state_residual"]` ranks negative-benefit candidates
  lowest (most-benefit on top when ascending). The rendering layer
  (`pretty()`) labels the sign so a reader sees "state benefit at long ctx"
  instead of just a minus sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    from .quality_model import (
        ArchConfig as QArchConfig, QualityResult, TrainingConfig,
        estimate_quality, quality,
    )
except ImportError:
    from quality_model import (
        ArchConfig as QArchConfig, QualityResult, TrainingConfig,
        estimate_quality, quality,
    )


# =============================================================================
# Quality stress dataclass
# =============================================================================

QUALITY_STRESS_AXES = (
    "shape_law_loss",
    "attention_residual",
    "moe_residual",
    "state_residual",
    "precision_loss",
    "fa_underrun",
    "context_extrapolation",
)


@dataclass
class QualityStressVector:
    schema_version: int = 1
    arch_name: str = ""

    # Fractional residuals (units: quality_proxy fraction of baseline)
    shape_law_loss: float = 0.0
    attention_residual: float = 0.0
    moe_residual: float = 0.0
    state_residual: float = 0.0
    precision_loss: float = 0.0
    fa_underrun: float = 0.0
    context_extrapolation: float = 0.0

    # Absolute contribution (axis × baseline) for each axis, computed once.
    absolute: Dict[str, float] = field(default_factory=dict)

    # Provenance — confidence string carried through from the source TermResult.
    confidence: Dict[str, str] = field(default_factory=dict)

    # Raw spine baseline so callers can convert axis fractions to absolute loss
    chinchilla_baseline: float = 0.0
    total_residual: float = 0.0
    notes: List[str] = field(default_factory=list)

    def axis_value(self, axis: str) -> float:
        if axis not in QUALITY_STRESS_AXES:
            raise KeyError(f"unknown quality stress axis: {axis!r}")
        return float(getattr(self, axis))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        rows = []
        for axis in QUALITY_STRESS_AXES:
            v = self.axis_value(axis)
            conf = self.confidence.get(axis, "?")
            abs_v = self.absolute.get(axis, 0.0)
            # Wave 7a.1: surface the sign on state_residual so a negative
            # value reads as "state benefit at long ctx" rather than as a
            # bare minus sign (which post-Wave-5 looks like a regression
            # to a reader that hasn't read the docstring). Other axes are
            # non-negative penalties by construction; they get no tag.
            tag = ""
            if axis == "state_residual":
                if v < -1e-6:
                    tag = "  (state benefit at long ctx)"
                elif v > 1e-6:
                    tag = "  (state cost)"
            rows.append(
                f"  {axis:22s} {v: 8.5f}  (Δabs={abs_v: 7.4f})  conf={conf}{tag}"
            )
        return (
            f"QualityStressVector  arch={self.arch_name}\n"
            f"  baseline={self.chinchilla_baseline:.4f}  "
            f"total_residual={self.total_residual:.4f}\n"
            + "\n".join(rows)
        )


# =============================================================================
# Decomposition
# =============================================================================

def _split_architecture_residual(arch_term) -> Dict[str, float]:
    """Split the architecture_residual term into shape-law vs attention-head
    sub-buckets using the per-subterm features attached by quality_model.

    Mirrors the subterm keys created in `_architecture_residual` in
    v0-quality/quality_model.py (around line 800).
    """
    if arch_term is None or not arch_term.features:
        return {"shape_law": 0.0, "attention_residual": 0.0, "fa_underrun": 0.0}
    sub = arch_term.features.get("subterms", {})
    shape_law = sub.get("width_depth", 0.0) + sub.get("mlp_attention_ratio", 0.0)
    attention = (
        sub.get("d_head", 0.0)
        + sub.get("query_heads", 0.0)
        + sub.get("kv_heads", 0.0)
        + sub.get("gqa_sharing", 0.0)
        + sub.get("attention_bottleneck", 0.0)
    )
    fa = sub.get("attn_kernel_underrun", 0.0)
    return {"shape_law": shape_law, "attention_residual": attention, "fa_underrun": fa}


def _context_extrapolation_penalty(
    arch: QArchConfig,
    training: TrainingConfig,
    workload_spec: Optional[Dict[str, Any]],
) -> float:
    """Placeholder for serving context > trained context.

    Returns a flat 0.0 for v1; the calibration data to estimate this
    properly lands in Phase 2. Surface here so the axis exists in the
    vector and downstream code doesn't need branching when it's wired up.
    """
    if workload_spec is None:
        return 0.0
    serve_ctx = int(workload_spec.get("context_length", training.sequence_length))
    train_ctx = int(training.sequence_length)
    if serve_ctx <= train_ctx:
        return 0.0
    # 2026-06-16: keep this as 0.0 until Phase 2 calibration. Future:
    #   return max(0.0, k * (math.log2(serve_ctx / train_ctx)) ** 2)
    return 0.0


# =============================================================================
# Main entry point
# =============================================================================

def compute_quality_stress(
    arch: QArchConfig,
    training: Optional[TrainingConfig] = None,
    workload_spec: Optional[Dict[str, Any]] = None,
    arch_name: str = "",
    _quality_result: Optional[QualityResult] = None,
) -> QualityStressVector:
    """Compute the 7-axis QualityStressVector for an architecture.

    Calls quality_model.estimate_quality (the v1 modular backbone) and reads
    the named TermResult sub-buckets out of the result. Each axis is exactly
    a slice of an existing term — no new quality math is introduced here.
    """
    training = training or TrainingConfig(training_tokens=20_000_000_000_000)
    if _quality_result is None:
        qr = estimate_quality(arch, training, workload_spec=workload_spec)
    else:
        qr = _quality_result

    baseline = qr.chinchilla_baseline
    terms = qr.terms

    arch_split = _split_architecture_residual(terms.get("architecture_residual"))
    moe_v = terms.get("moe_residual").value if terms.get("moe_residual") else 0.0
    state_v = terms.get("state_residual").value if terms.get("state_residual") else 0.0
    precision_v = terms.get("precision_residual").value if terms.get("precision_residual") else 0.0
    ctx_ext = _context_extrapolation_penalty(arch, training, workload_spec)

    sv = QualityStressVector(
        arch_name=arch_name,
        shape_law_loss=arch_split["shape_law"],
        attention_residual=arch_split["attention_residual"],
        moe_residual=moe_v,
        state_residual=state_v,
        precision_loss=precision_v,
        fa_underrun=arch_split["fa_underrun"],
        context_extrapolation=ctx_ext,
        chinchilla_baseline=baseline,
        total_residual=qr.total_penalty_fraction,
    )
    sv.absolute = {a: sv.axis_value(a) * baseline for a in QUALITY_STRESS_AXES}
    sv.confidence = {
        "shape_law_loss": terms.get("architecture_residual").confidence
                           if terms.get("architecture_residual") else "n/a",
        "attention_residual": terms.get("architecture_residual").confidence
                               if terms.get("architecture_residual") else "n/a",
        "moe_residual": terms.get("moe_residual").confidence
                         if terms.get("moe_residual") else "not_applicable",
        "state_residual": terms.get("state_residual").confidence
                           if terms.get("state_residual") else "not_applicable",
        "precision_loss": terms.get("precision_residual").confidence
                           if terms.get("precision_residual") else "n/a",
        "fa_underrun": "high",
        "context_extrapolation": "placeholder",
    }
    return sv
