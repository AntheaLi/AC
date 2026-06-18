"""
Delta-eval CLI.

    delta-eval \\
        --baseline-config configs/mistral_7b_like.json \\
        --hardware h100 --tp 8 \\
        --apply swap_attention_to_gqa --apply-args n_kv_heads=8 \\
        --out outputs/mistral_gqa_eval/

Supports:
    --apply REPEAT      — multiple --apply NAME flags compose into a sequence
    --apply-args k=v    — args for the most recent --apply (repeatable)
    --workload PRESET   — chat | batched | long_context | training
    --serving-batch N   — explicit override (overrides preset)
    --context-length N  — explicit override
    --no-pareto         — skip the Pareto-position computation (cheap mode)
    --json              — dump JSON only (no Markdown)
    --stdout            — print Markdown to stdout instead of writing files

Outputs (under --out):
    evaluation.json     — full DeltaEvaluation serialized
    report.md           — one-screen Markdown report
    pareto.csv          — 3-row Pareto CSV (baseline / candidate / delta)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from optimizer import DeploymentConstraints  # noqa: E402
from baseline import load_baseline_model  # noqa: E402

from evaluator import (  # noqa: E402
    DeltaEvaluation,
    evaluate_delta,
    evaluate_delta_sequence,
)
from report import (  # noqa: E402
    render_markdown,
    render_markdown_multi,
    render_json,
    render_pareto_csv,
)


# =============================================================================
# Workload presets (multiplicative scaling of constraints)
# =============================================================================

_WORKLOAD_PRESETS = {
    "chat": {
        "serving_batch": 1,
        "context_length": 2048,
        "prompt_len": 2048,
        "serving_tbt_ms": 50.0,
    },
    "batched": {
        "serving_batch": 64,
        "context_length": 4096,
        "prompt_len": 4096,
        "serving_tbt_ms": 50.0,
    },
    "long_context": {
        "serving_batch": 8,
        "context_length": 32768,
        "prompt_len": 32768,
        "serving_tbt_ms": 80.0,
    },
    "training": {
        "serving_batch": 4,
        "context_length": 4096,
        "prompt_len": 4096,
        "serving_tbt_ms": 200.0,
    },
}

VALID_HARDWARE = ["h100", "b200", "tpu_v5p",
                  "trainium2", "trn2", "trainium3", "trn3"]


def _coerce_arg_value(v: str) -> Any:
    """Coerce a CLI arg string into int / float / bool / str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


# =============================================================================
# Custom argparse: collect --apply NAME [--apply-args k=v]... groups
# =============================================================================

def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="delta-eval",
        description="Evaluate the Pareto-front and metric influence of a "
                    "delta transformation against a baseline architecture.",
    )
    parser.add_argument("--baseline-config", required=True,
                        help="Path to a compiler-schema JSON baseline.")
    parser.add_argument("--hardware", default="h100",
                        choices=VALID_HARDWARE)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--workload", default="chat",
                        choices=list(_WORKLOAD_PRESETS.keys()))
    parser.add_argument("--serving-batch", type=int, default=None,
                        help="Override workload preset.")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Override workload preset.")
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--target-params-b", type=float, default=0.0,
                        help="0 means inferred from baseline.")
    parser.add_argument("--training-tokens", type=int,
                        default=2_000_000_000_000)
    parser.add_argument("--apply", action="append", required=True,
                        metavar="DELTA_NAME",
                        help="A transformation name (repeatable for "
                             "sequences). Use --apply-args after each "
                             "--apply for that delta's args.")
    parser.add_argument("--apply-args", action="append", default=[],
                        metavar="k=v",
                        help="Args for the most recent --apply. Repeat "
                             "to set multiple args for the same delta. "
                             "Separator between delta groups: --apply-args "
                             "is bound to the closest preceding --apply.")
    parser.add_argument("--no-pareto", action="store_true",
                        help="Skip Pareto-position classification.")
    parser.add_argument("--out", default=None,
                        help="Output directory. Default: "
                             "outputs/delta_eval_<baseline>_<delta>/")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON only (no Markdown / CSV).")
    parser.add_argument("--stdout", action="store_true",
                        help="Print Markdown to stdout instead of writing files.")
    return parser.parse_args(argv)


def _build_delta_groups(argv: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Walk argv to associate each --apply-args with its preceding --apply.

    argparse's `action="append"` puts all --apply names in one list and all
    --apply-args strings in another, losing the association. We re-walk argv
    here to restore it.
    """
    groups: List[Tuple[str, Dict[str, Any]]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--apply":
            name = argv[i + 1] if i + 1 < len(argv) else ""
            groups.append((name, {}))
            i += 2
            continue
        if tok == "--apply-args":
            if not groups:
                i += 2
                continue
            kv = argv[i + 1] if i + 1 < len(argv) else ""
            if "=" in kv:
                k, _, v = kv.partition("=")
                groups[-1][1][k] = _coerce_arg_value(v)
            i += 2
            continue
        i += 1
    return groups


def _validate_delta_groups(groups: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Validate --apply names + --apply-args keys against the delta REGISTRY
    *before* trying to evaluate. Emits a useful, actionable error instead of
    a late TypeError from the delta's `apply()` method.
    """
    import inspect
    # Lazy import — keeps the cli importable even when v1-stress isn't on path
    # at module import time.
    from deltas import REGISTRY
    errors: List[str] = []
    for name, args in groups:
        if name not in REGISTRY:
            errors.append(
                f"--apply {name!r} is not a known transformation. "
                f"Available: {', '.join(sorted(REGISTRY.keys()))}"
            )
            continue
        sig = inspect.signature(REGISTRY[name]().apply)
        # arg 0 is `arch` — strip it. Keep everything else as legal kwargs.
        legal = [p for p in sig.parameters.keys() if p != "arch"]
        legal_set = set(legal)
        for k in args.keys():
            if k not in legal_set:
                errors.append(
                    f"--apply {name}: unknown --apply-args key {k!r}. "
                    f"Legal keys for this delta: {', '.join(legal) or '(none)'}"
                )
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


# =============================================================================
# Main
# =============================================================================

def main(argv: List[str] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = _parse_args(argv)
    delta_groups = _build_delta_groups(argv)
    if not delta_groups:
        print("No --apply DELTA_NAME provided.", file=sys.stderr)
        return 2

    # Validate delta names + kwargs early so users get a useful error before
    # we load and evaluate anything.
    _validate_delta_groups(delta_groups)

    # 1) Load baseline
    bm = load_baseline_model(args.baseline_config)

    # 2) Build DeploymentConstraints, layering preset + explicit overrides
    preset = _WORKLOAD_PRESETS[args.workload]
    constraints = DeploymentConstraints(
        target_params_b=(args.target_params_b
                         if args.target_params_b > 0
                         else bm.candidate.total_params / 1e9),
        training_tokens=args.training_tokens,
        context_length=args.context_length or preset["context_length"],
        prompt_len=args.prompt_len or preset["prompt_len"],
        serving_tbt_ms=preset["serving_tbt_ms"],
        serving_ttft_ms=2000.0,
        serving_batch=args.serving_batch or preset["serving_batch"],
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
    )

    # 3) Evaluate. Single delta vs sequence.
    if len(delta_groups) == 1:
        name, dargs = delta_groups[0]
        ev = evaluate_delta(
            bm.candidate, args.hardware, constraints,
            delta_name=name, delta_args=dargs,
            baseline_name=bm.name,
            include_pareto=not args.no_pareto,
        )
        results = [ev]
    else:
        ev = evaluate_delta_sequence(
            bm.candidate, args.hardware, constraints,
            deltas=delta_groups,
            baseline_name=bm.name,
            include_pareto=not args.no_pareto,
        )
        results = [ev]

    # 4) Output
    if args.stdout:
        if args.json:
            for ev in results:
                print(render_json(ev))
        else:
            print(render_markdown_multi(results)
                   if len(results) > 1
                   else render_markdown(results[0]))
        return 0

    out_dir = args.out
    if not out_dir:
        slug = "+".join(g[0] for g in delta_groups)
        out_dir = os.path.join(
            "outputs", f"delta_eval_{bm.name}_{slug}")
    os.makedirs(out_dir, exist_ok=True)

    # evaluation.json
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        if len(results) > 1:
            f.write(json.dumps([ev.as_dict() for ev in results],
                                indent=2, default=str))
        else:
            f.write(render_json(results[0]))

    if not args.json:
        # report.md
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            if len(results) > 1:
                f.write(render_markdown_multi(results))
            else:
                f.write(render_markdown(results[0]))
        # pareto.csv
        with open(os.path.join(out_dir, "pareto.csv"), "w") as f:
            f.write(render_pareto_csv(results[0]))

    print(f"Wrote evaluation to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
