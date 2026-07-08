"""Wave 18h — paired-ablation residual fitting (zero-compute calibration).

Consumes a machine-readable corpus of PUBLISHED paired ablations (same data,
same scale, one architecture axis flipped — see
tests/fixtures/public_ablation_pairs_v1.json) and:

  1. Scores both arms of every pair with the quality model and compares the
     predicted loss delta with the published delta.
  2. Fits a per-term scale factor by attributing each pair's residual to the
     term the pair targets (least squares over the pairs touching a term).
  3. Emits a COVERAGE AUDIT: which residual terms have zero constraining
     pairs, and which operating-point regions of a constrained term are
     unanchored. This is the check that would have caught the pure-SSM bug
     mechanically — the state-benefit term had no anchor below p_attn=0.07,
     so its behavior there was pure extrapolation.

This is curation, not compute: it upgrades hand-tuned residual constants
into a reproducible fit with per-term error bars, without a single training
run. Cross-paper deltas carry datamix/tokenizer confounds, so fitted scales
are written as an *overlay pack* (like ac-auto-calibrate's) rather than
edited into the defaults.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .quality_model import ArchConfig as QArch, estimate_quality
except ImportError:  # pragma: no cover
    from quality_model import ArchConfig as QArch, estimate_quality


# Residual terms a pair may target. Terms in the quality model but absent
# here can never be constrained by this corpus format — they are reported
# as uncovered in the audit.
KNOWN_TERMS = (
    "state_residual", "architecture_residual", "attention_locality",
    "effective_capacity", "moe_residual", "mtp_residual", "context_utility",
    "precision_residual", "large_shape_stability_prior", "risk_residual",
)

# Some corpus terms live inside a differently-named term in the model's
# breakdown; map corpus terms to model term names for delta attribution.
# (The SWA/interleave locality penalty is booked under context_utility —
# verified by term-delta inspection, Wave 18h.)
_MODEL_TERM_ALIAS = {"attention_locality": "context_utility"}


def _arm_to_arch(arm: Dict[str, Any]) -> QArch:
    kind = str(arm.get("kind", "dense"))
    kw: Dict[str, Any] = dict(
        d_model=int(arm["d_model"]), n_layers=int(arm["n_layers"]),
        n_heads=int(arm["n_heads"]), d_head=int(arm["d_head"]),
        n_kv_heads=int(arm["n_kv_heads"]), ffn_dim=int(arm["ffn_dim"]),
        vocab_size=int(arm.get("vocab_size", 32000)),
    )
    if kind in ("hybrid", "pure_state"):
        kw.update(model_type="hybrid",
                  state_config={
                      "enabled": True,
                      "state_type": arm.get("state_type", "mamba2"),
                      "d_state": int(arm.get("d_state", 128)),
                      "state_layers": int(arm["state_layers"]),
                      "attention_layers": int(arm["attention_layers"]),
                  })
    elif kind == "mla":
        kw.update(attention_type="mla",
                  mla_latent_dim=int(arm.get("mla_kv_latent", 512)),
                  mla_q_latent_dim=int(arm.get("mla_q_latent", 1536)),
                  mla_rope_head_dim=64)
    elif kind == "moe":
        kw.update(model_type="moe",
                  moe_config={
                      "n_experts": int(arm["n_experts"]),
                      "top_k": int(arm["top_k"]),
                      "expert_dim": int(arm.get("expert_dim", arm["ffn_dim"])),
                  })
    elif kind in ("local_global", "swa"):
        kw.update(local_window=int(arm["local_window"]),
                  local_attention_fraction=float(arm.get("local_fraction", 1.0)))
    elif kind == "mtp":
        kw.update(mtp_n_predict_depths=int(arm.get("mtp_depths", 1)))
    elif kind == "rope":
        kw.update(rope_scaling_method=str(arm.get("rope_method", "none")),
                  rope_scaling_factor=float(arm.get("rope_factor", 1.0)),
                  rope_original_max_position=8192)
    return QArch(**kw)


@dataclass
class PairResult:
    pair_id: str
    term: str
    source: str
    observed_pct: float
    predicted_pct: float
    term_predicted_pct: float
    residual_pct: float
    tolerance_pct: float
    within_tolerance: bool
    operating_point: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TermFit:
    term: str
    n_pairs: int
    scale: Optional[float]           # None when unidentifiable (|pred|~0)
    bias_pct: float                  # mean residual
    rms_pct: float
    covered_points: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def evaluate_pairs(corpus: Dict[str, Any]) -> List[PairResult]:
    out: List[PairResult] = []
    for p in corpus.get("pairs", []):
        tr = {"training_tokens": int(float(p["training_tokens_t"]) * 1e12)}
        w = {"context_length": int(p["context"])}
        qa = estimate_quality(_arm_to_arch(p["arm_a"]), tr, workload_spec=w)
        qb = estimate_quality(_arm_to_arch(p["arm_b"]), tr, workload_spec=w)
        la, lb = qa.predicted_loss, qb.predicted_loss
        pred_pct = 100.0 * (lb - la) / max(1e-9, la)
        model_term = _MODEL_TERM_ALIAS.get(p["term"], p["term"])
        ta = (qa.terms or {}).get(model_term)
        tb = (qb.terms or {}).get(model_term)
        term_pred_pct = 100.0 * (
            float(getattr(tb, "value", 0.0) or 0.0)
            - float(getattr(ta, "value", 0.0) or 0.0))
        obs = float(p["observed_delta_pct"])
        tol = float(p.get("tolerance_pct", 0.5))
        out.append(PairResult(
            pair_id=p["id"], term=p["term"], source=p.get("source", ""),
            observed_pct=obs, predicted_pct=round(pred_pct, 4),
            term_predicted_pct=round(term_pred_pct, 4),
            residual_pct=round(obs - pred_pct, 4),
            tolerance_pct=tol,
            within_tolerance=abs(obs - pred_pct) <= tol,
            operating_point=p.get("operating_point", {}),
        ))
    return out


def fit_terms(results: List[PairResult]) -> List[TermFit]:
    """Least-squares per-term scale: attribute each pair's residual to its
    targeted term. observed ≈ (predicted_total − term_pred) + scale × term_pred
    ⇒ scale* = Σ(adj·t)/Σ(t²) with adj = observed − (predicted − term_pred)."""
    by_term: Dict[str, List[PairResult]] = {}
    for r in results:
        by_term.setdefault(r.term, []).append(r)
    fits: List[TermFit] = []
    for term in KNOWN_TERMS:
        rs = by_term.get(term, [])
        if not rs:
            fits.append(TermFit(
                term=term, n_pairs=0, scale=None, bias_pct=0.0, rms_pct=0.0,
                warnings=[f"UNCOVERED: no published pair constrains "
                          f"'{term}' — its constants are hand priors; any "
                          f"decision it dominates is extrapolation."]))
            continue
        num = sum((r.observed_pct - (r.predicted_pct - r.term_predicted_pct))
                  * r.term_predicted_pct for r in rs)
        den = sum(r.term_predicted_pct ** 2 for r in rs)
        scale = (num / den) if den > 1e-8 else None
        bias = sum(r.residual_pct for r in rs) / len(rs)
        rms = math.sqrt(sum(r.residual_pct ** 2 for r in rs) / len(rs))
        warnings = []
        if scale is None:
            warnings.append(
                f"'{term}' pairs exist but the model predicts ~0 delta for "
                f"all of them — the term is UNIDENTIFIABLE from this corpus "
                f"(shape error, or the pairs sit where the term is flat).")
        elif not (0.5 <= scale <= 2.0):
            warnings.append(
                f"fitted scale {scale:.2f} is far from 1.0 — the term's "
                f"magnitude disagrees with published evidence; inspect "
                f"before trusting decisions dominated by '{term}'.")
        if len(rs) < 2:
            warnings.append(
                f"only {len(rs)} pair(s) — scale is a point estimate with "
                f"no cross-validation; treat as directional.")
        fits.append(TermFit(
            term=term, n_pairs=len(rs),
            scale=None if scale is None else round(scale, 3),
            bias_pct=round(bias, 3), rms_pct=round(rms, 3),
            covered_points=[r.operating_point for r in rs],
            warnings=warnings))
    return fits


def coverage_gaps(fits: List[TermFit]) -> List[str]:
    """Region-level audit for terms whose operating points carry a known
    sweep axis. Flags unanchored extrapolation regions explicitly."""
    gaps: List[str] = []
    for f in fits:
        if f.term == "state_residual" and f.n_pairs:
            p_attns = sorted(
                v for pt in f.covered_points
                for k, v in pt.items() if k.startswith("p_attn"))
            ctxs = sorted(pt.get("context", 0) for pt in f.covered_points)
            if p_attns:
                msg = (f"state_residual: anchored p_attn range "
                       f"[{min(p_attns):.2f}, {max(p_attns):.2f}], contexts "
                       f"{ctxs}.")
                if min(p_attns) > 0.0:
                    msg += (f" Behavior at p_attn < {min(p_attns):.2f} is "
                            f"extrapolation — this exact gap allowed the "
                            f"pre-18f pure-SSM winners.")
                if max(c for c in ctxs) < 1048576:
                    msg += (f" No anchor above ctx {max(ctxs):,} — the "
                            f"long-context benefit magnitude at 1M+ is "
                            f"extrapolation.")
                gaps.append(msg)
        if f.term == "attention_locality" and f.n_pairs:
            fracs = sorted(pt.get("local_fraction", 0.0) for pt in f.covered_points)
            windows = sorted(pt.get("window", 0) for pt in f.covered_points)
            gaps.append(
                f"attention_locality: anchored local fractions "
                f"{fracs}, windows {windows}; other regions are priors.")
        if f.term == "effective_capacity" and f.n_pairs:
            ratios = sorted(pt.get("ratio", 0.0) for pt in f.covered_points)
            gaps.append(
                f"effective_capacity: anchored sparsity ratios {ratios}; "
                f"ratios beyond this range (e.g. >8x) lean on the N_eff "
                f"functional form, not on paired evidence.")
    return gaps


def render_report(results: List[PairResult], fits: List[TermFit],
                  gaps: List[str]) -> str:
    lines = ["# Paired-ablation residual fit (Wave 18h)", ""]
    n_ok = sum(1 for r in results if r.within_tolerance)
    lines.append(f"Pairs: {len(results)} | within tolerance: {n_ok} | "
                 f"outside: {len(results) - n_ok}")
    lines.append("")
    lines.append("| pair | term | observed Δ% | predicted Δ% | residual | ok |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r.pair_id} | {r.term} | {r.observed_pct:+.2f} | "
            f"{r.predicted_pct:+.2f} | {r.residual_pct:+.2f} | "
            f"{'✓' if r.within_tolerance else '✗'} |")
    lines.append("")
    lines.append("## Per-term fit")
    lines.append("")
    lines.append("| term | pairs | scale | bias % | rms % |")
    lines.append("|---|---:|---:|---:|---:|")
    for f in fits:
        lines.append(
            f"| {f.term} | {f.n_pairs} | "
            f"{'—' if f.scale is None else f.scale} | "
            f"{f.bias_pct:+.2f} | {f.rms_pct:.2f} |")
    lines.append("")
    lines.append("## Coverage audit")
    lines.append("")
    for f in fits:
        for wmsg in f.warnings:
            lines.append(f"- **{f.term}**: {wmsg}")
    for g in gaps:
        lines.append(f"- {g}")
    lines.append("")
    lines.append(
        "Fitted scales are cross-paper (datamix/tokenizer confounded); use "
        "as priors and ingest lab pairs via the same format to sharpen.")
    return "\n".join(lines)


def run_fit_pairs(pairs_path: str, out_dir: str) -> Dict[str, Any]:
    with open(pairs_path) as f:
        corpus = json.load(f)
    results = evaluate_pairs(corpus)
    fits = fit_terms(results)
    gaps = coverage_gaps(fits)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "schema_version": "wave18h.pair_fit.v1",
        "pairs_path": os.path.abspath(pairs_path),
        "results": [vars(r) for r in results],
        "term_fits": [vars(f) for f in fits],
        "coverage_gaps": gaps,
    }
    with open(os.path.join(out_dir, "pair_fit.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "pair_fit_report.md"), "w") as f:
        f.write(render_report(results, fits, gaps))
    return payload
