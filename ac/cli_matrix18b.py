"""Wave 18b CLI — budget-matched matrix + scenario Pareto frontiers.

Per `plan/redesign/18b-budget-and-serving-pareto.md`. Runs the AC
optimizer over a small (hardware × context × reference-active-B) grid
and emits Wave 18b's five budget-projection views + four scenario
Pareto frontiers per cell, plus a matrix-level (hw × ctx) main view.

Not a replacement for `ac-compile`; this is the auditable comparison
matrix a reviewer runs before committing to an architecture. The main
matrix intentionally displays contender status (owned by Wave 18d);
until 18d lands, the fallback status is `<family> [loss-argmin]`.

Usage:

    ac-matrix-18b \\
        --hardware h100 \\
        --contexts 32768,131072 \\
        --reference-active-b 7,70 \\
        --training-tokens 20e12 \\
        --out matrix18b.json \\
        --md matrix18b.md
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ac.optimizer import DeploymentConstraints, optimize
from ac.budget_pareto import (
    BudgetMatrix,
    Matrix,
    MatrixKey,
    MatrixCell,
    extract_metrics,
    render_matrix_json,
    render_matrix_markdown,
    render_markdown,
    render_json,
)
# Wave 18d — contender-status labels for the main matrix. Falling back to
# `[loss-argmin]` before this was wired defeated 18d's whole point (turn
# sub-uncertainty differences into `unresolved`).
from ac.decision import assess_decision, DECISION_WINNER, DECISION_UNRESOLVED


def _parse_int_list(s: str) -> List[int]:
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def _parse_scientific(s: str) -> int:
    return int(float(s))


def _format_decision_status(decision, cell) -> str:
    """Wave 18d → single-string label rendered inside the main matrix cell.

    Chooses among:
      `<signature> [winner]` — one candidate clearly dominates
      `<sigA> / <sigB> [unresolved]` — several contenders within noise floor
      `[out_of_domain]` — top candidate outside calibration domain
      `[no physically feasible candidate]` — Wave 18c guard tripped
      `[model_validation_failure: <reason>]` — Wave 18e anchor tripped

    The signature is the compact `identity_label` from the winning
    candidate's `CandidateMetrics` when available; otherwise the raw
    `selected_candidate_id` from Wave 18d.
    """
    from ac.decision import (
        DECISION_WINNER, DECISION_UNRESOLVED, DECISION_OUT_OF_DOMAIN,
        DECISION_NO_PHYSICAL, DECISION_MODEL_FAILURE,
    )
    status = decision.status
    if status == DECISION_MODEL_FAILURE:
        reason = decision.reasons[0] if decision.reasons else "unknown"
        return f"[model_validation_failure: {reason}]"
    if status == DECISION_NO_PHYSICAL:
        return "[no physically feasible candidate]"
    if status == DECISION_OUT_OF_DOMAIN:
        return "[out_of_domain]"

    # Look up identity_label for a given candidate id by scanning
    # `cell.all_evaluated` (which holds `CandidateMetrics`).
    def label_of(cid: Optional[str]) -> str:
        if cid is None:
            return "?"
        for m in cell.all_evaluated:
            if getattr(m, "candidate_id", None) == cid:
                return m.identity_label
        return cid  # fall back to raw id

    if status == DECISION_WINNER:
        return f"{label_of(decision.selected_candidate_id)} [winner]"
    if status == DECISION_UNRESOLVED:
        top = list(decision.contender_ids)[:2]
        if len(top) == 0:
            return "[unresolved]"
        if len(top) == 1:
            return f"{label_of(top[0])} [unresolved]"
        return f"{label_of(top[0])} / {label_of(top[1])} [unresolved]"
    return "[loss-argmin]"  # unknown status → conservative fallback


def _run_cell(
    *,
    hardware: str,
    context_length: int,
    reference_active_b: float,
    training_tokens: int,
    max_candidates: Optional[int],
    max_full_evals: Optional[int],
    allow_moe: bool,
    allow_state: bool,
):
    """Run one cell end-to-end and return `(BudgetMatrix, feasible_evals)` or
    None on failure. `feasible_evals` is the raw EvaluatedCandidate list
    the caller hands to Wave 18d `assess_decision` for the main-matrix
    contender-status label. Kept separate from `BudgetMatrix` because
    18d reads `predicted_loss` and `quality.*` fields the metrics-view
    doesn't retain."""
    constraints_kw = {}
    if allow_moe:
        constraints_kw["allow_moe"] = True
        constraints_kw["max_total_params_b"] = reference_active_b * 8
        constraints_kw["moe_n_experts_options"] = [8]
        constraints_kw["moe_top_k_options"] = [2]
        constraints_kw["ep_options"] = [4]
        constraints_kw["moe_granularity_targets"] = [1.0]
        constraints_kw["dense_ffn_layer_options"] = [0, 1, 2, 3]
    if allow_state:
        constraints_kw["allow_state"] = True
        constraints_kw["state_type"] = "mamba2"
    c = DeploymentConstraints(
        target_params_b=reference_active_b,
        training_tokens=training_tokens,
        context_length=context_length,
        tp=8, pp=1, dp=8,
        serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=16,
        vocab_size=32000,
        allow_mla=True,
        mla_kv_latent_options=[512],
        param_tolerance=0.08,
        allow_quality_sentinel=True,
        max_candidates=max_candidates,
        max_full_evaluations=max_full_evals,
        **constraints_kw,
    )
    try:
        result = optimize(hardware, c)
    except Exception as exc:
        print(f"[matrix18b] ({hardware}, ctx={context_length}, "
              f"ref={reference_active_b}B) optimize failed: {exc}",
              file=sys.stderr)
        return None
    driver = BudgetMatrix(
        hardware=hardware, context_length=context_length,
        training_tokens=training_tokens,
    )
    feasible_evals = []
    for ev in result.all_evaluated:
        if not getattr(ev, "meets_constraints", True):
            continue
        driver.add(extract_metrics(ev, c))
        feasible_evals.append(ev)
    return driver, feasible_evals


def main() -> int:
    p = argparse.ArgumentParser(
        description="Wave 18b — budget-matched matrix + scenario Pareto frontiers")
    p.add_argument("--hardware", default="h100",
                   help="Comma-separated list of hardware targets.")
    p.add_argument("--contexts", default="32768,131072",
                   help="Comma-separated list of context lengths in tokens.")
    p.add_argument("--reference-active-b", default="7,70",
                   help="Comma-separated list of reference active-parameter "
                        "counts in billions (matrix columns).")
    p.add_argument("--training-tokens", default="20e12",
                   type=_parse_scientific,
                   help="Training tokens per cell (default 20T).")
    p.add_argument("--max-candidates", type=int, default=400,
                   help="Enumeration cap per generator (default 400).")
    p.add_argument("--max-full-evaluations", type=int, default=80,
                   help="Full-evaluation cap after cheap-rank prune "
                        "(default 80; see Wave 8b).")
    p.add_argument("--no-moe", action="store_true",
                   help="Skip allow_moe generator.")
    p.add_argument("--no-state", action="store_true",
                   help="Skip allow_state generator.")
    p.add_argument("--out", default=None,
                   help="Write JSON matrix to this path.")
    p.add_argument("--md", default=None,
                   help="Write Markdown matrix to this path.")
    p.add_argument("--per-cell-md", action="store_true",
                   help="Include per-cell detailed budget/frontier tables "
                        "in the Markdown output (default: matrix-level only).")
    args = p.parse_args()

    # Wave 28: fail fast on a broken AC_QUALITY_DEFAULTS /
    # AC_HARDWARE_SPEC_DIR before the (potentially multi-minute) matrix
    # search; per-candidate swallowing otherwise turns it into empty
    # cells / "no feasible candidate" noise.
    try:
        from quality_model import validate_calibration_environment
    except ImportError:
        from .quality_model import validate_calibration_environment
    try:
        validate_calibration_environment()
    except Exception as e:
        print(f"ERROR: invalid calibration environment: {e}", file=sys.stderr)
        return 2

    # Wave 18f fix: prepare output paths BEFORE the (potentially multi-minute)
    # matrix search. Previously a missing parent directory — or an --out
    # path that collided with a directory — only surfaced at the final
    # open(), throwing away the whole run. Fail fast instead, and create
    # parent directories the way ac-compile's output paths do.
    import os as _os
    for _label, _path in (("--out", args.out), ("--md", args.md)):
        if not _path:
            continue
        if _os.path.isdir(_path):
            p.error(f"{_label} expects a file path, got directory: {_path}")
        _parent = _os.path.dirname(_os.path.abspath(_path))
        try:
            _os.makedirs(_parent, exist_ok=True)
        except (OSError, NotADirectoryError) as e:
            p.error(f"cannot create parent directory for {_label}={_path}: {e}")

    hardware_list = [h.strip() for h in args.hardware.split(",") if h.strip()]
    contexts = _parse_int_list(args.contexts)
    ref_bs = _parse_float_list(args.reference_active_b)
    training_tokens = args.training_tokens

    matrix = Matrix(training_tokens=training_tokens)
    # Wave 18d — map MatrixKey → contender-status label from
    # `assess_decision`. Populated as each cell runs; consumed by the
    # `status_fn` closure passed to `render_matrix_markdown` below.
    decision_status_by_key: dict = {}

    for hw in hardware_list:
        for ctx in contexts:
            for ref_b in ref_bs:
                cell_out = _run_cell(
                    hardware=hw, context_length=ctx,
                    reference_active_b=ref_b,
                    training_tokens=training_tokens,
                    max_candidates=args.max_candidates,
                    max_full_evals=args.max_full_evaluations,
                    allow_moe=(not args.no_moe),
                    allow_state=(not args.no_state),
                )
                if cell_out is None:
                    continue
                driver, feasible_evals = cell_out
                # Anchor = the smallest dense candidate at this ref_b, or
                # if none exists, the first evaluated candidate.
                dense_candidates = [
                    m for m in driver._candidates
                    if m.identity_label == "dense"
                ]
                if not driver._candidates:
                    continue
                anchor = (dense_candidates[0] if dense_candidates
                          else driver._candidates[0])
                cell = driver.build_cell(anchor)
                key = MatrixKey(hw, ctx, ref_b)
                matrix.add_cell(key, cell)

                # Wave 18d — contender-status assessment. Passes raw
                # EvaluatedCandidate objects so `assess_decision` can read
                # `predicted_loss` and per-term uncertainty. Stability
                # inputs are left at their defaults (None) until Wave 18e
                # sensitivity data is wired in; per 18d that keeps
                # `require_stability_for_winner=False` so we still emit a
                # winner when the practical + uncertainty gates pass on
                # their own.
                # Wave 18h: with correlated-error paired sigma feeding the
                # uncertainty gate, the practical-effect floor no longer
                # needs to stand in for model error. 1% is the level at
                # which a loss gap is worth acting on at frontier scale
                # (0.5-1% loss ~ a hardware-generation of compute); the
                # old 5% floor was masking every decision the pairing
                # math can now actually resolve.
                decision = assess_decision(
                    feasible_evals,
                    practical_effect_threshold_pct=1.0,
                )
                decision_status_by_key[key.as_tuple()] = _format_decision_status(
                    decision, cell)

                print(
                    f"[matrix18b] {hw} ctx={ctx} ref={ref_b}B: "
                    f"{len(driver._candidates)} candidates → "
                    f"decision={decision.status} "
                    f"({decision.selected_candidate_id or '—'}) "
                    f"| interactive frontier "
                    f"{len(cell.scenario_frontiers['interactive_serving'].frontier)}",
                    file=sys.stderr,
                )

    # Wave 18d — status_fn closure. `render_matrix_markdown` calls it
    # per cell; we look up the pre-computed `assess_decision` result by
    # its MatrixKey. If a cell was skipped in the run loop above (e.g.
    # optimize failed) we fall back to the legacy loss-argmin label so
    # the matrix still renders.
    def _status_fn(cell: MatrixCell) -> str:
        for k, v in matrix.cells.items():
            if v is cell:
                return decision_status_by_key.get(k, "[loss-argmin]")
        return "[loss-argmin]"

    json_payload = render_matrix_json(matrix)
    # Attach per-cell decision status to the JSON payload so downstream
    # consumers (web app, CLI report, calibration audit) don't have to
    # re-run assess_decision.
    for cell_entry in json_payload.get("cells", []):
        k = cell_entry["key"]
        tup = (k["hardware"], k["context_length"], k["reference_active_b"])
        cell_entry["contender_status"] = decision_status_by_key.get(
            tup, "[loss-argmin]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(json_payload, f, indent=2, default=str)
        print(f"[matrix18b] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(json_payload, indent=2, default=str))
    if args.md:
        if args.per_cell_md:
            md = render_matrix_markdown(matrix, status_fn=_status_fn)
        else:
            # Matrix-level only: skip the "Detailed cells" section by
            # rendering each cell independently and only keeping the
            # main-matrix header lines.
            md = render_matrix_markdown(matrix, status_fn=_status_fn)
            head_end = md.find("## Detailed cells")
            if head_end > 0:
                md = md[:head_end].rstrip() + "\n"
        with open(args.md, "w") as f:
            f.write(md)
        print(f"[matrix18b] wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
