"""Wave 18d — Decision confidence, abstention, and evidence provenance.

The decision layer turns an ``OptimizationResult`` into a ``DecisionAssessment``
that distinguishes:

* a true winner (one candidate dominates beyond the noise floor and is
  in-domain and stable);
* an unresolved outcome (multiple candidates within the noise floor);
* an out-of-domain extrapolation (we cannot confidently say anything);
* no physically feasible candidate (Wave 18c physical guards left nothing);
* a model-validation failure (Wave 18e anchor or registry check tripped).

The intent is to make AC abstain rather than over-promise in the regime where
modeled differences are smaller than modeled uncertainty.

Pre-calibration unique-winner rule (Wave 18d):

1. candidate is physically feasible;
2. candidate is not out-of-domain;
3. loss advantage over every other contender exceeds
   ``max(practical_effect_threshold_pct, combined_quality_uncertainty_pct)``;
4. when 18e stability data is available, winner stability fraction must be at
   least 0.80 and contender retention fraction must be at least 0.95.

Operational flags (Wave 18c) never force ``unresolved``; they determine which
Pareto frontier the user reads next.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Evidence provenance taxonomy
# ---------------------------------------------------------------------------

# Wave 18d enumerates the kinds of evidence that produce a quality or systems
# term. Calibration drives the difference between e.g. a heuristic prior and a
# calibrated measurement; reviewers need to see both kinds without conflating
# them in the displayed decision.
EVIDENCE_KINDS: Tuple[str, ...] = (
    "scaling_law",            # closed-form Chinchilla-class spine
    "analytical_hardware",    # arithmetic from hardware spec + roofline
    "measured_hardware",      # measured on real hardware (Wave 19 lab)
    "literature_prior",       # published ablation prior (small magnitude)
    "heuristic",              # internal-uncalibrated prior (must widen sigma)
    "calibrated_measurement", # hierarchical posterior (Wave 19 production)
)


# Decision statuses (Wave 18 shared contract).
DECISION_WINNER = "winner"
DECISION_UNRESOLVED = "unresolved"
DECISION_OUT_OF_DOMAIN = "out_of_domain"
DECISION_NO_PHYSICAL = "no_physically_feasible_candidate"
DECISION_MODEL_FAILURE = "model_validation_failure"

ALL_DECISION_STATUSES: Tuple[str, ...] = (
    DECISION_WINNER,
    DECISION_UNRESOLVED,
    DECISION_OUT_OF_DOMAIN,
    DECISION_NO_PHYSICAL,
    DECISION_MODEL_FAILURE,
)


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceProvenance:
    """Source attribution for a single quality / systems term.

    A non-zero term that lacks calibrated provenance MUST primarily contribute
    to widened uncertainty rather than a decisive shift in the mean. The
    rendering layer surfaces ``evidence_kind`` and ``source`` so a reviewer
    can tell physics, literature, heuristic, and calibrated measurement
    apart at a glance.
    """
    evidence_kind: str                    # one of EVIDENCE_KINDS
    source: str = ""                      # human-readable origin (citation, code path)
    calibration_pack_id: Optional[str] = None  # Wave 19 calibration manifest id
    domain_status: str = "in_domain"      # in_domain | near_domain_boundary | out_of_domain
    uncertainty: float = 0.0              # fractional one-sigma-ish

    def is_calibrated(self) -> bool:
        return self.evidence_kind in {"measured_hardware", "calibrated_measurement"}

    def widens_only(self) -> bool:
        """True for heuristic/literature priors: must not decisively shift the
        decision without showing up in the explanation."""
        return self.evidence_kind in {"heuristic", "literature_prior"}


def make_provenance(
    evidence_kind: str,
    source: str = "",
    calibration_pack_id: Optional[str] = None,
    domain_status: str = "in_domain",
    uncertainty: float = 0.0,
) -> EvidenceProvenance:
    """Construct a provenance record with validation."""
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError(
            f"unknown evidence_kind {evidence_kind!r}; "
            f"must be one of {EVIDENCE_KINDS}"
        )
    if domain_status not in {"in_domain", "near_domain_boundary", "out_of_domain"}:
        raise ValueError(
            f"unknown domain_status {domain_status!r}; "
            f"must be in_domain | near_domain_boundary | out_of_domain"
        )
    if uncertainty < 0:
        raise ValueError("uncertainty must be non-negative")
    return EvidenceProvenance(
        evidence_kind=evidence_kind,
        source=source,
        calibration_pack_id=calibration_pack_id,
        domain_status=domain_status,
        uncertainty=float(uncertainty),
    )


# ---------------------------------------------------------------------------
# Decision assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionAssessment:
    """Confidence-aware decision for one matrix cell.

    Replaces the previous "loss-min wins" pattern. A status field plus a
    contender list lets downstream UI render:

      * ``winner: <signature>``
      * ``unresolved: <signature A> / <signature B> [/ <C> ...]``
      * ``out-of-domain``
      * ``no physically feasible candidate``
      * ``model validation failure``

    Legacy ``optimal`` / family fields remain available on the upstream
    ``OptimizationResult`` for one transition version; consumers should
    migrate to read ``DecisionAssessment`` instead.
    """
    status: str
    selected_candidate_id: Optional[str]
    contender_ids: Tuple[str, ...]
    practical_effect_threshold_pct: float
    uncertainty_threshold_pct: float
    stability_fraction: Optional[float]
    domain_status: str
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    # Pareto coordinates of the selected candidate's loss vs the runner-up,
    # for explanation. Both in fractional loss units.
    selected_loss: Optional[float] = None
    runner_up_loss: Optional[float] = None
    runner_up_gap_pct: Optional[float] = None
    # Combined quality uncertainty actually used in the rule (max over each
    # term's `provenance.uncertainty`, or per-candidate aggregate).
    combined_quality_uncertainty_pct: Optional[float] = None

    def is_winner(self) -> bool:
        return self.status == DECISION_WINNER

    def to_display_line(self, signature_of) -> str:
        """One-line display per Wave 18d acceptance gate.

        ``signature_of`` is a callable that maps a candidate id to its
        printable signature (Wave 18a). Kept as an injection so this module
        does not import from ``ac.architecture`` (avoids a hard dependency
        while 18a lands).
        """
        if self.status == DECISION_WINNER and self.selected_candidate_id:
            return f"winner: {signature_of(self.selected_candidate_id)}"
        if self.status == DECISION_UNRESOLVED:
            sigs = " / ".join(signature_of(cid) for cid in self.contender_ids)
            return f"unresolved: {sigs}"
        if self.status == DECISION_OUT_OF_DOMAIN:
            return "out-of-domain"
        if self.status == DECISION_NO_PHYSICAL:
            return "no physically feasible candidate"
        if self.status == DECISION_MODEL_FAILURE:
            return "model validation failure"
        return f"unknown_status: {self.status}"


# ---------------------------------------------------------------------------
# Assessment logic
# ---------------------------------------------------------------------------


def _candidate_id(ev: Any) -> str:
    """Stable id for a candidate, derived from arch fields only.

    Independent of evaluation order so the same architecture in two runs
    receives the same id.
    """
    arch = getattr(ev, "arch", None) or ev
    parts = [
        f"d={int(getattr(arch, 'd_model', 0))}",
        f"L={int(getattr(arch, 'n_layers', 0))}",
        f"H={int(getattr(arch, 'n_heads', 0))}",
        f"dh={int(getattr(arch, 'd_head', 0))}",
        f"kv={int(getattr(arch, 'n_kv_heads', 0))}",
        f"ffn={int(getattr(arch, 'ffn_dim', 0))}",
        f"tp={int(getattr(arch, 'tp_degree', 1) or 1)}",
        f"pp={int(getattr(arch, 'pp_degree', 1) or 1)}",
        f"att={getattr(arch, 'attention_type', 'full')}",
        f"moe={'1' if getattr(arch, 'moe', None) else '0'}",
        f"state={'1' if getattr(arch, 'state_config', None) else '0'}",
    ]
    return "|".join(parts)


def _candidate_uncertainty_pct(ev: Any) -> float:
    """Combine the quality model's per-term uncertainties for one candidate.

    For two-candidate comparison we want the *joint* uncertainty band. We
    use the geometric quadrature sum of the per-term ``provenance.uncertainty``
    fields when present, falling back to ``QualityResult.uncertainty_total``
    or a conservative 8% prior when none is recorded.

    Returned as a percentage of the candidate's predicted loss.
    """
    q = getattr(ev, "quality", None)
    if q is None:
        return 8.0  # conservative pre-calibration prior

    # 1) Per-term provenance.uncertainty (preferred — Wave 18d source)
    terms = getattr(q, "terms", {}) or {}
    per_term: List[float] = []
    for t in terms.values():
        prov = getattr(t, "provenance", None)
        if prov is not None:
            per_term.append(float(prov.uncertainty))
        else:
            per_term.append(float(getattr(t, "uncertainty", 0.0)))
    if per_term:
        # quadrature on fractional uncertainties → fractional total
        total = (sum(u * u for u in per_term)) ** 0.5
        return 100.0 * total

    # 2) QualityResult.uncertainty_total
    ut = float(getattr(q, "uncertainty_total", 0.0))
    if ut > 0:
        return 100.0 * ut

    # 3) Spread from uncertainty_low_pct / uncertainty_high_pct
    lo = float(getattr(q, "uncertainty_low_pct", 0.0))
    hi = float(getattr(q, "uncertainty_high_pct", 0.0))
    if hi > lo > 0:
        return 0.5 * (hi - lo)

    return 8.0  # conservative pre-calibration prior


def _combined_uncertainty_pct(a: Any, b: Any) -> float:
    """Joint quality uncertainty between two candidates, as a percent.

    Wave 18h: when both candidates carry term-level quality breakdowns,
    use the correlated-error paired sigma — the two predictions share the
    scaling-law spine, data assumptions, and any residual evaluated at the
    same operating point, and that shared error cancels in the difference.
    The old independent-quadrature combine remains the fallback (and the
    conservative bound) when term breakdowns are unavailable.
    """
    qa = getattr(a, "quality", None)
    qb = getattr(b, "quality", None)
    if getattr(qa, "terms", None) and getattr(qb, "terms", None):
        try:
            from ac.quality_model import paired_loss_uncertainty
        except Exception:  # pragma: no cover - package layout fallback
            try:
                from quality_model import paired_loss_uncertainty
            except Exception:
                paired_loss_uncertainty = None
        if paired_loss_uncertainty is not None:
            try:
                p = paired_loss_uncertainty(qa, qb)
                if p.get("enabled", True):
                    return float(p["sigma_rel"]) * 100.0
            except Exception:
                pass
    ua = _candidate_uncertainty_pct(a)
    ub = _candidate_uncertainty_pct(b)
    return (ua * ua + ub * ub) ** 0.5


def _candidate_domain_status(ev: Any) -> str:
    """Worst-case domain status across all this candidate's terms.

    ``out_of_domain`` dominates ``near_domain_boundary`` dominates
    ``in_domain``. The candidate's overall domain status is the worst of
    any contributing term — a single out-of-domain residual taints the
    whole prediction.
    """
    q = getattr(ev, "quality", None)
    if q is None:
        return "in_domain"
    terms = getattr(q, "terms", {}) or {}
    worst = "in_domain"
    for t in terms.values():
        prov = getattr(t, "provenance", None)
        if prov is None:
            continue
        ds = getattr(prov, "domain_status", "in_domain")
        if ds == "out_of_domain":
            return "out_of_domain"
        if ds == "near_domain_boundary":
            worst = "near_domain_boundary"
    # Also fold in the QualityResult's chinchilla-regime check as a coarse
    # OOD signal — out-of-Chinchilla-regime is at least "near_domain_boundary".
    if not getattr(q, "in_chinchilla_regime", True):
        if worst == "in_domain":
            worst = "near_domain_boundary"
    return worst


_FAMILY_BIAS_CACHE: Optional[Dict[str, Any]] = None


def _load_family_bias() -> Dict[str, Any]:
    """Wave 19 (P1-5): per-family signed loss-bias table measured on the
    public-model anchors (ac/calibration/family_bias_v1.json).

    Pre-calibration, AC's loss bias is FAMILY-CORRELATED (dense mean +3.8%,
    MoE mean +8.1% on the v1 anchor set). Shared bias cancels in
    within-family comparisons, but the differential component corrupts
    cross-family ranking — exactly the dense-vs-MoE greenfield question.
    The table converts that from a silent corruption into an explicit
    higher bar / abstention.
    """
    global _FAMILY_BIAS_CACHE
    if _FAMILY_BIAS_CACHE is not None:
        return _FAMILY_BIAS_CACHE
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calibration", "family_bias_v1.json")
    try:
        with open(path) as f:
            _FAMILY_BIAS_CACHE = json.load(f).get("families", {})
    except Exception:
        _FAMILY_BIAS_CACHE = {}
    return _FAMILY_BIAS_CACHE


def _family_of_candidate(ev: Any) -> str:
    arch = getattr(ev, "arch", None)
    if arch is None:
        return "dense"
    if getattr(arch, "moe", None) or getattr(arch, "moe_config", None):
        return "moe"
    if int(getattr(arch, "n_state_layers", 0) or 0) > 0 or \
            getattr(arch, "state_config", None):
        return "hybrid"
    return "dense"


def _quality_pack_loaded() -> bool:
    p = os.environ.get("AC_QUALITY_DEFAULTS")
    return bool(p) and os.path.exists(p)


def _family_metric_stats(fam_entry: Dict[str, Any],
                         metric: str) -> Optional[Tuple[float, float]]:
    """Return (mean_signed_err_pct, stdev_err_pct) for a metric.

    Reads the Wave 20 v2 per-metric block first; falls back to the v1
    loss-only keys when metric == "loss" so older tables keep working.
    """
    metrics = fam_entry.get("metrics") or {}
    m = metrics.get(metric)
    if m is not None:
        try:
            return float(m["mean_signed_err_pct"]), float(m["stdev_err_pct"])
        except (KeyError, TypeError, ValueError):
            return None
    if metric == "loss" and "mean_signed_loss_err_pct" in fam_entry:
        try:
            return (float(fam_entry["mean_signed_loss_err_pct"]),
                    float(fam_entry["stdev_loss_err_pct"]))
        except (KeyError, TypeError, ValueError):
            return None
    return None


_METRIC_LABEL = {
    "loss": "loss", "tbt_ms": "decode TBT", "ttft_ms": "prefill TTFT",
    "mem_gb": "serving memory",
}


def family_metric_span_pct(fam: str, metric: str = "tbt_ms"
                           ) -> Optional[Tuple[float, float]]:
    """(min, max) signed anchor error %, for one family/metric (Wave 25).

    Reads the live family_bias_v1.json per-anchor entries so user-facing
    disclosure text tracks the current audit instead of a hard-coded span
    from whichever release last edited the string. Returns None when the
    table or the per-anchor block is unavailable.
    """
    fam_entry = _load_family_bias().get(fam) or {}
    m = (fam_entry.get("metrics") or {}).get(metric) or {}
    anchors = m.get("anchors") or {}
    try:
        vals = [float(v) for v in anchors.values()]
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    return min(vals), max(vals)


def family_metric_span_text(fam: str = "moe", metric: str = "tbt_ms",
                            fallback: str = "−94%…+93%") -> str:
    """Render the anchor-error span as e.g. '−32%…+62%' (Wave 25)."""
    span = family_metric_span_pct(fam, metric)
    if span is None:
        return fallback
    lo, hi = span

    def _fmt(v: float) -> str:
        return f"{'−' if v < 0 else '+'}{abs(v):.0f}%"

    return f"{_fmt(lo)}…{_fmt(hi)}"


def _cross_family_bias_bar_pct(a: Any, b: Any,
                               metric: str = "loss") -> Tuple[float, str]:
    """Extra winner-bar (in %) for a cross-family pair, uncalibrated.

    bar = |mean_bias_a - mean_bias_b| + 0.5 × (stdev_a + stdev_b).
    Returns (0.0, "") for same-family pairs, calibrated runs, or families
    absent from the table.

    Wave 20 (feedback #1): `metric` selects which anchor-audited metric's
    bias floor to use ("loss", "tbt_ms", "ttft_ms", "mem_gb"). Serving
    metrics carry larger cross-family bias than loss (see
    `family_metric_span_pct("moe", "tbt_ms")` for the current
    anchor-measured span), so serving-cost / latency comparisons must be
    floored with their own numbers, not the loss floor.
    """
    if _quality_pack_loaded():
        return 0.0, ""
    fam_a = _family_of_candidate(a)
    fam_b = _family_of_candidate(b)
    return cross_family_bias_bar_by_name(fam_a, fam_b, metric=metric)


def cross_family_bias_bar_by_name(fam_a: str, fam_b: str,
                                  metric: str = "loss"
                                  ) -> Tuple[float, str]:
    """Bias floor for a cross-family pair, by family NAME (Wave 20).

    Used by report/justification code that has family labels rather than
    candidate objects. Same formula as `_cross_family_bias_bar_pct`.
    Note: does NOT check `_quality_pack_loaded()`; callers that must skip
    the floor post-calibration do that check themselves.
    """
    if fam_a == fam_b:
        return 0.0, ""
    table = _load_family_bias()
    ta, tb = table.get(fam_a), table.get(fam_b)
    if not ta or not tb:
        return 0.0, ""
    sa = _family_metric_stats(ta, metric)
    sb = _family_metric_stats(tb, metric)
    if sa is None or sb is None:
        return 0.0, ""
    bar = abs(sa[0] - sb[0]) + 0.5 * (sa[1] + sb[1])
    label = _METRIC_LABEL.get(metric, metric)
    reason = (
        f"cross-family comparison ({fam_a} vs {fam_b}) under known "
        f"uncalibrated model bias: anchor-measured family {label} bias gap "
        f"plus scatter gives a {bar:.1f}% floor; {label} deltas below it are "
        "inside known model bias, not signal (fit a calibration pack to "
        "lower it)"
    )
    return bar, reason


def cross_family_serving_floor_pct(a: Any, b: Any,
                                   metric: str = "tbt_ms"
                                   ) -> Tuple[float, str]:
    """Public wrapper for report/justification code that needs the
    serving-metric bias floor for a cross-family pair (Wave 20)."""
    return _cross_family_bias_bar_pct(a, b, metric=metric)


def assess_decision(
    candidates: Sequence[Any],
    *,
    practical_effect_threshold_pct: float = 5.0,
    stability_fraction: Optional[float] = None,
    contender_retention_fraction: Optional[float] = None,
    require_stability_for_winner: bool = False,
    model_validation_failure: Optional[str] = None,
) -> DecisionAssessment:
    """Assess a list of evaluated candidates and return a DecisionAssessment.

    Parameters
    ----------
    candidates :
        Iterable of ``EvaluatedCandidate``-shaped objects. Each must expose
        ``predicted_loss``, ``meets_constraints``, ``arch``, and ``quality``.
    practical_effect_threshold_pct :
        Minimum loss gap (in %) over the runner-up required to declare a
        winner, regardless of how tight the uncertainty band is. Default 5%.
    stability_fraction :
        Wave 18e ``winner_stability_fraction``. If supplied, must be ≥ 0.80
        for the winner rule to fire. If None and
        ``require_stability_for_winner`` is True the cell is unresolved
        ("missing 18e stability data leaves the decision unresolved in
        trust mode").
    contender_retention_fraction :
        Wave 18e ``contender_retention_fraction``. If supplied, must be ≥ 0.95.
    require_stability_for_winner :
        If True, missing Wave 18e stability data prevents declaring a winner
        (the "trust-mode" gate from the spec). Default False so this module
        is useful in environments where 18e has not yet run.
    model_validation_failure :
        If supplied (non-None), short-circuits the decision to
        ``model_validation_failure`` with the provided reason. Used by Wave
        18e public-model anchor failures and similar registry checks.

    Returns
    -------
    DecisionAssessment
    """
    # 1) Hard short-circuits.
    if model_validation_failure:
        return DecisionAssessment(
            status=DECISION_MODEL_FAILURE,
            selected_candidate_id=None,
            contender_ids=tuple(),
            practical_effect_threshold_pct=practical_effect_threshold_pct,
            uncertainty_threshold_pct=0.0,
            stability_fraction=stability_fraction,
            domain_status="unknown",
            reasons=(model_validation_failure,),
        )

    feasible = [c for c in candidates if getattr(c, "meets_constraints", True)]
    if not feasible:
        return DecisionAssessment(
            status=DECISION_NO_PHYSICAL,
            selected_candidate_id=None,
            contender_ids=tuple(),
            practical_effect_threshold_pct=practical_effect_threshold_pct,
            uncertainty_threshold_pct=0.0,
            stability_fraction=stability_fraction,
            domain_status="unknown",
            reasons=("no candidate passed Wave 18c physical guards",),
        )

    # 2) Sort by predicted loss (ascending — lower is better).
    sorted_cands = sorted(feasible, key=lambda c: float(getattr(c, "predicted_loss", float("inf"))))
    best = sorted_cands[0]
    best_id = _candidate_id(best)
    best_loss = float(getattr(best, "predicted_loss", float("inf")))

    # 3) Out-of-domain top candidate cannot be a unique winner.
    best_domain = _candidate_domain_status(best)
    if best_domain == "out_of_domain":
        return DecisionAssessment(
            status=DECISION_OUT_OF_DOMAIN,
            selected_candidate_id=None,
            contender_ids=tuple(_candidate_id(c) for c in sorted_cands[:5]),
            practical_effect_threshold_pct=practical_effect_threshold_pct,
            uncertainty_threshold_pct=0.0,
            stability_fraction=stability_fraction,
            domain_status=best_domain,
            reasons=(
                "best candidate carries at least one out-of-domain term; "
                "AC abstains until calibration extends coverage",
            ),
            selected_loss=best_loss,
        )

    # 4) Single candidate ⇒ trivially the winner if feasible+in-domain, but
    # we still demand stability if trust-mode is on.
    if len(sorted_cands) == 1:
        if require_stability_for_winner and stability_fraction is None:
            return DecisionAssessment(
                status=DECISION_UNRESOLVED,
                selected_candidate_id=None,
                contender_ids=(best_id,),
                practical_effect_threshold_pct=practical_effect_threshold_pct,
                uncertainty_threshold_pct=0.0,
                stability_fraction=stability_fraction,
                domain_status=best_domain,
                reasons=("trust mode requires Wave 18e stability data; not supplied",),
                selected_loss=best_loss,
            )
        return DecisionAssessment(
            status=DECISION_WINNER,
            selected_candidate_id=best_id,
            contender_ids=(best_id,),
            practical_effect_threshold_pct=practical_effect_threshold_pct,
            uncertainty_threshold_pct=0.0,
            stability_fraction=stability_fraction,
            domain_status=best_domain,
            reasons=("only one feasible in-domain candidate",),
            selected_loss=best_loss,
        )

    # 5) Two-or-more candidates: apply the unique-winner rule.
    runner = sorted_cands[1]
    runner_loss = float(getattr(runner, "predicted_loss", float("inf")))
    if best_loss <= 0:
        gap_pct = 0.0
    else:
        gap_pct = 100.0 * (runner_loss - best_loss) / best_loss

    combined_unc_pct = _combined_uncertainty_pct(best, runner)
    # Wave 19 (P1-5): cross-family pairs carry an anchor-measured bias
    # floor on top of the modeled uncertainty when uncalibrated.
    _fam_bar_runner, _fam_reason = _cross_family_bias_bar_pct(best, runner)
    threshold_pct = max(practical_effect_threshold_pct, combined_unc_pct,
                        _fam_bar_runner)

    # Build the contender set: best + every other candidate whose gap is
    # within the threshold (the "indistinguishable from the winner" band).
    contenders = [best]
    _fam_reasons: List[str] = [r for r in (_fam_reason,) if r]
    for c in sorted_cands[1:]:
        cl = float(getattr(c, "predicted_loss", float("inf")))
        cg = 100.0 * (cl - best_loss) / max(best_loss, 1e-9)
        # Use the pairwise combined uncertainty for each contender vs best.
        pairwise_unc = _combined_uncertainty_pct(best, c)
        _fb, _fr = _cross_family_bias_bar_pct(best, c)
        pairwise_threshold = max(practical_effect_threshold_pct,
                                 pairwise_unc, _fb)
        if cg <= pairwise_threshold:
            contenders.append(c)
            if _fr and _fr not in _fam_reasons:
                _fam_reasons.append(_fr)
    contender_ids = tuple(_candidate_id(c) for c in contenders)

    # 6) Stability gate (Wave 18e).
    stability_ok = True
    stability_reasons: List[str] = []
    if stability_fraction is not None:
        if stability_fraction < 0.80:
            stability_ok = False
            stability_reasons.append(
                f"winner stability {stability_fraction:.2f} < 0.80 (Wave 18e gate)"
            )
    elif require_stability_for_winner:
        stability_ok = False
        stability_reasons.append("trust mode requires Wave 18e stability data; not supplied")
    if contender_retention_fraction is not None and contender_retention_fraction < 0.95:
        stability_ok = False
        stability_reasons.append(
            f"contender retention {contender_retention_fraction:.2f} < 0.95 (Wave 18e gate)"
        )

    # 7) Decide.
    if len(contenders) == 1 and stability_ok:
        return DecisionAssessment(
            status=DECISION_WINNER,
            selected_candidate_id=best_id,
            contender_ids=(best_id,),
            practical_effect_threshold_pct=practical_effect_threshold_pct,
            uncertainty_threshold_pct=combined_unc_pct,
            stability_fraction=stability_fraction,
            domain_status=best_domain,
            reasons=(
                f"loss gap {gap_pct:.2f}% over runner-up exceeds "
                f"max(practical={practical_effect_threshold_pct:.1f}%, "
                f"uncertainty={combined_unc_pct:.1f}%)",
            ),
            selected_loss=best_loss,
            runner_up_loss=runner_loss,
            runner_up_gap_pct=gap_pct,
            combined_quality_uncertainty_pct=combined_unc_pct,
        )

    # Unresolved: report the contender set.
    _core_reason = (
        f"runner-up within {threshold_pct:.1f}% effective threshold "
        f"(gap {gap_pct:.2f}%, combined uncertainty {combined_unc_pct:.1f}%)"
    )
    if not stability_ok:
        unresolved_reasons = tuple(
            stability_reasons + [_core_reason] + _fam_reasons)
    else:
        unresolved_reasons = tuple([_core_reason] + _fam_reasons)

    return DecisionAssessment(
        status=DECISION_UNRESOLVED,
        selected_candidate_id=None,
        contender_ids=contender_ids,
        practical_effect_threshold_pct=practical_effect_threshold_pct,
        uncertainty_threshold_pct=combined_unc_pct,
        stability_fraction=stability_fraction,
        domain_status=best_domain,
        reasons=unresolved_reasons,
        selected_loss=best_loss,
        runner_up_loss=runner_loss,
        runner_up_gap_pct=gap_pct,
        combined_quality_uncertainty_pct=combined_unc_pct,
    )


# ---------------------------------------------------------------------------
# Rendering adapters
# ---------------------------------------------------------------------------


def decision_to_json(d: DecisionAssessment) -> Dict[str, Any]:
    """Serialize a DecisionAssessment to a plain JSON-compatible dict."""
    return {
        "status": d.status,
        "selected_candidate_id": d.selected_candidate_id,
        "contender_ids": list(d.contender_ids),
        "practical_effect_threshold_pct": d.practical_effect_threshold_pct,
        "uncertainty_threshold_pct": d.uncertainty_threshold_pct,
        "stability_fraction": d.stability_fraction,
        "domain_status": d.domain_status,
        "reasons": list(d.reasons),
        "selected_loss": d.selected_loss,
        "runner_up_loss": d.runner_up_loss,
        "runner_up_gap_pct": d.runner_up_gap_pct,
        "combined_quality_uncertainty_pct": d.combined_quality_uncertainty_pct,
    }


def decision_to_markdown(d: DecisionAssessment, signature_of=lambda cid: cid) -> str:
    """Render a Markdown explanation block for one decision."""
    out: List[str] = []
    out.append(f"**Decision:** {d.to_display_line(signature_of)}")
    if d.selected_loss is not None:
        out.append(f"  - selected loss: {d.selected_loss:.4f}")
    if d.runner_up_loss is not None and d.runner_up_gap_pct is not None:
        out.append(
            f"  - runner-up loss: {d.runner_up_loss:.4f} "
            f"(gap {d.runner_up_gap_pct:.2f}%)"
        )
    out.append(
        f"  - practical-effect threshold: "
        f"{d.practical_effect_threshold_pct:.1f}%"
    )
    if d.combined_quality_uncertainty_pct is not None:
        out.append(
            f"  - combined quality uncertainty: "
            f"{d.combined_quality_uncertainty_pct:.1f}%"
        )
    if d.stability_fraction is not None:
        out.append(f"  - stability fraction (Wave 18e): {d.stability_fraction:.2f}")
    out.append(f"  - domain status: {d.domain_status}")
    if d.reasons:
        out.append("  - reasons:")
        for r in d.reasons:
            out.append(f"    - {r}")
    return "\n".join(out)


def evidence_breakdown(ev: Any) -> List[Dict[str, Any]]:
    """Return per-term provenance for an evaluated candidate, suitable for
    JSON/Markdown rendering by report.py.

    Each entry has ``name``, ``value``, ``evidence_kind``, ``source``,
    ``calibration_pack_id``, ``domain_status``, ``uncertainty``.
    """
    out: List[Dict[str, Any]] = []
    q = getattr(ev, "quality", None)
    if q is None:
        return out
    for name, term in (getattr(q, "terms", {}) or {}).items():
        prov = getattr(term, "provenance", None)
        if prov is None:
            out.append({
                "name": name,
                "value": float(getattr(term, "value", 0.0)),
                "evidence_kind": "heuristic",
                "source": str(getattr(term, "source", "compiler_default")),
                "calibration_pack_id": None,
                "domain_status": "in_domain",
                "uncertainty": float(getattr(term, "uncertainty", 0.0)),
                "widens_only": True,
                "is_calibrated": False,
            })
        else:
            out.append({
                "name": name,
                "value": float(getattr(term, "value", 0.0)),
                "evidence_kind": prov.evidence_kind,
                "source": prov.source,
                "calibration_pack_id": prov.calibration_pack_id,
                "domain_status": prov.domain_status,
                "uncertainty": prov.uncertainty,
                "widens_only": prov.widens_only(),
                "is_calibrated": prov.is_calibrated(),
            })
    return out
