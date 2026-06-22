#!/usr/bin/env python3
"""
AC Architecture Compiler v0.3 — CLI

Batch command-line tool: hardware spec + deployment constraints in,
architecture config + justification + Pareto frontier out.

Usage:
    python cli.py \
        --hardware h100 \
        --params 7 \
        --tokens 2 \
        --context 8192 \
        --serving-tbt 50 \
        --serving-ttft 500 \
        --serving-batch 32 \
        --tp 8 --pp 1 --dp 8 \
        --output-config arch.json \
        --output-justification arch.md \
        --output-pareto pareto.csv

Not interactive. Inputs in, outputs out. Errors are CLI-grade messages.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Wire up paths
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from optimizer import optimize, result_to_config, result_to_pareto_csv, DeploymentConstraints
from cli_recipe import (
    expand_argv,
    render_group_help,
    snapshot_recipe,
    run_init as _recipe_run_init,
    run_config_show as _recipe_run_config_show,
)
from shadow_prices import compute_shadow_prices, shadow_prices_to_json
from justification import generate_justification, generate_assumptions, generate_model_card
from schema import save_config
from implementation_generator import save_pytorch_implementation
from baseline import BaselineUnsupportedError, load_baseline_model
from modifier import (
    modifier_pareto_to_csv,
    modifier_result_to_config,
    run_modifier_search,
)
from baseline_delta import (
    generate_baseline_delta_report,
    generate_modifier_justification,
    generate_modifier_shadow_report,
)


# Canonical hardware names shown in --help. Trainium short forms
# (`trn2`/`trn3`) are still parsed via `_normalize_hardware` below but
# hidden from --help so users only see one name per platform.
VALID_HARDWARE = [
    "h100", "b200", "tpu_v5p", "tpu_v5e",
    "trainium2", "trainium3",
]

_HARDWARE_ALIASES = {
    "trn2": "trainium2",
    "trn3": "trainium3",
}


def _normalize_hardware(value: str) -> str:
    v = (value or "").strip().lower()
    return _HARDWARE_ALIASES.get(v, v)


def _format_optimal_line(opt) -> str:
    """One-line summary of the picked candidate.

    The earlier template printed `kv=<n_kv_heads>` unconditionally, which
    was misleading on MLA runs (n_kv_heads is irrelevant under MLA, but
    the surrounding numerals looked normal). It also didn't surface MoE
    topology, MTP depth, or RoPE-scaling, so on the MAI-Thinking-1
    example the user couldn't tell from the log line that MLA+MoE+MTP
    were active. This helper makes the line arch-aware.
    """
    a = opt.arch
    parts = [f"d={a.d_model}", f"L={a.n_layers}"]
    # Attention block.
    attn = a.attention_type
    if attn == "mla":
        latent = a.mla_kv_latent_dim or 0
        q_latent = a.mla_q_latent_dim or 0
        parts.append(f"attn=mla(kv_latent={latent},q_latent={q_latent})")
    elif attn == "nsa":
        parts.append("attn=nsa")
    else:
        parts.append(f"attn=full h={a.n_heads} kv={a.n_kv_heads}")
    parts.append(f"ffn={a.ffn_dim}")
    # FFN family.
    if getattr(a, "moe_style", "dense") != "dense" and a.moe is not None:
        parts.append(
            f"moe={a.moe.get('n_experts')}x"
            f"top{a.moe.get('top_k')}(ep={a.ep_degree})"
        )
        if a.n_dense_ffn_layers:
            parts.append(f"dense_prefix={a.n_dense_ffn_layers}")
    # State / hybrid.
    if a.n_state_layers > 0:
        parts.append(
            f"state={a.n_state_layers}/{a.n_layers} "
            f"d_state={a.derived_d_state} placement={a.placement_strategy}"
        )
    # MTP.
    if a.mtp_n_predict_depths and a.mtp_n_predict_depths > 0:
        parts.append(f"mtp={a.mtp_n_predict_depths}")
    # RoPE scaling.
    if a.rope_scaling_method and a.rope_scaling_method != "none":
        parts.append(f"rope={a.rope_scaling_method}")
    # Precision.
    parts.append(f"prec={a.ffn_precision} kv_bits={a.kv_cache_bits}")
    return "[arch-compiler] Optimal: " + " ".join(parts)


def parse_billions(value: str) -> float:
    """Parse parameter count in billions, accepting '7' or '7B'."""
    s = str(value).strip().lower().replace("_", "")
    if s.endswith("b"):
        s = s[:-1]
    out = float(s)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return out


def parse_trillions(value: str) -> float:
    """Parse token count in trillions, accepting '2' or '2T'."""
    s = str(value).strip().lower().replace("_", "")
    if s.endswith("t"):
        s = s[:-1]
    out = float(s)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return out


def parse_positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return out


def parse_non_negative_int(value: str) -> int:
    out = int(value)
    if out < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return out


def parse_positive_float(value: str) -> float:
    out = float(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return out


def parse_non_negative_float(value: str) -> float:
    out = float(value)
    if out < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return out


def parse_int_list(value: str):
    """Parse comma-separated positive integers."""
    out = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    bad = [v for v in out if v <= 0]
    if bad:
        raise argparse.ArgumentTypeError(f"values must be > 0: {bad}")
    return out


def parse_kv_dtype_options(value: str):
    """Parse KV dtype option list into bit widths used by DeploymentConstraints."""
    mapping = {"bf16": 16, "fp16": 16, "int8": 8, "fp8": 8, "int4": 4, "fp4": 4}
    out = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise argparse.ArgumentTypeError(f"Unknown KV dtype option: {raw}")
        out.append(mapping[key])
    if not out:
        raise argparse.ArgumentTypeError("expected at least one KV dtype")
    return sorted(set(out), reverse=True)


def parse_precision_modes(value: str):
    """Parse precision mode aliases into optimizer precision config names."""
    mapping = {
        "bf16": "all_bf16",
        "all_bf16": "all_bf16",
        "fp8_ffn": "ffn_fp8",
        "ffn_fp8": "ffn_fp8",
        "fp8": "all_fp8",
        "all_fp8": "all_fp8",
        "fp4_ffn": "ffn_fp4",
        "ffn_fp4": "ffn_fp4",
        "fp4": "all_fp4",
        "all_fp4": "all_fp4",
        "mxfp4_ffn": "ffn_mxfp4",
        "ffn_mxfp4": "ffn_mxfp4",
        "mxfp4": "all_mxfp4",
        "all_mxfp4": "all_mxfp4",
        "mxfp6_ffn": "ffn_mxfp6",
        "ffn_mxfp6": "ffn_mxfp6",
        "mxfp6": "all_mxfp6",
        "all_mxfp6": "all_mxfp6",
    }
    out = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise argparse.ArgumentTypeError(f"Unknown precision mode: {raw}")
        out.append(mapping[key])
    if not out:
        raise argparse.ArgumentTypeError("expected at least one precision mode")
    return out


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def infer_output_paths(args) -> Dict[str, str]:
    """Populate sibling output paths on `args` from --output-config.

    Returns the resolved {field_name: path} map so `config show` can
    surface the same paths the real run will use. Idempotent: if a
    field is already set, we leave it alone and just record the path.

    Rule: derive a basename "stem" from --output-config and use
    "{stem}.md", "{stem}_pareto.csv", "{stem}_shadow_prices.json", etc.
    Special case: when the user accepts the default ("arch.json") the
    stem strips down to "arch" and the historical short names
    (pareto.csv, shadow_prices.json) are kept so existing scripts that
    globbed those names still work.
    """
    cfg_dir = os.path.dirname(os.path.abspath(args.output_config))
    cfg_stem = os.path.splitext(os.path.basename(args.output_config))[0]
    using_default_cfg = (os.path.abspath(args.output_config) ==
                         os.path.abspath("arch.json"))

    def _sibling(suffix: str, default_short: str) -> str:
        if using_default_cfg:
            return default_short
        # Strip a trailing "_arch" or "_config" so a stem like
        # "mistral_arch" produces "mistral_pareto.csv" rather than
        # "mistral_arch_pareto.csv". Documented in --help.
        stem = cfg_stem
        for tail in ("_arch", "_config"):
            if stem.endswith(tail) and len(stem) > len(tail):
                stem = stem[: -len(tail)]
                break
        return os.path.join(cfg_dir, f"{stem}{suffix}")

    if args.output_justification is None:
        # Justification inherits cfg_stem exactly (back-compat with the
        # README example: --output-config out/mistral_arch.json →
        # out/mistral_arch.md).
        args.output_justification = (
            "arch.md" if using_default_cfg
            else os.path.join(cfg_dir, f"{cfg_stem}.md"))
    if args.output_pareto is None:
        args.output_pareto = _sibling("_pareto.csv", "pareto.csv")
    if args.output_shadow_prices is None:
        args.output_shadow_prices = _sibling("_shadow_prices.json", "shadow_prices.json")
    if args.output_assumptions is None and getattr(args, "auto_emit_sidecars", False):
        args.output_assumptions = _sibling("_assumptions.md", "assumptions.md")
    if args.output_model_card is None and getattr(args, "auto_emit_sidecars", False):
        args.output_model_card = _sibling("_model_card.md", "model_card.md")

    # The contending-family sidecar is only written when the envelope
    # is non-robust, but show it in `config show` so users see the
    # full footprint a run may write.
    contending_sidecar = (
        f"{cfg_stem}_contending_family.json"
        if using_default_cfg
        else os.path.join(cfg_dir, f"{cfg_stem}_contending_family.json")
    )
    return {
        "output_config": args.output_config,
        "output_justification": args.output_justification,
        "output_pareto": args.output_pareto,
        "output_shadow_prices": args.output_shadow_prices,
        "output_assumptions": args.output_assumptions,
        "output_model_card": args.output_model_card,
        "output_implementation": getattr(args, "output_implementation", None),
        "contending_family_sidecar_if_non_robust": contending_sidecar,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Construct the ac-compile argparse parser.

    Split out from `parse_args` so subcommands (`ac-compile config show`)
    and the help-group filter can introspect groups without invoking
    `.parse_args()`. Argument GROUPS are named so `--help-group <name>`
    can filter to just one section.
    """
    p = argparse.ArgumentParser(
        prog="ac-compile",
        description=(
            "Hardware-aware architecture compiler v0.3. "
            "Takes hardware spec + deployment constraints, "
            "emits optimal architecture config + justification.\n\n"
            "Recipe-friendly: run `ac-compile --recipe configs/recipes/"
            "h100_dense_7b.yaml` and add `--override key=value` to tweak. "
            "Use `ac-compile --help-group <name>` to see just one flag "
            "group; available groups are listed at the bottom of --help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Meta flags. These are intercepted by `cli_recipe.expand_argv`
    # before the parser ever sees them, so the action here is just a
    # placeholder that surfaces them in --help. (action="store" with no
    # downstream consumer keeps argparse happy and lets users discover
    # the flag.)
    meta = p.add_argument_group("recipe & help")
    meta.add_argument("--recipe", metavar="PATH",
                      help="YAML/JSON recipe of flag values; see "
                           "configs/recipes/. Replaceable by `key=value` "
                           "pairs in --override.")
    meta.add_argument("--override", metavar="KEY=VALUE", action="append",
                      default=[],
                      help="Override a recipe field, e.g. "
                           "`--override params=70`. Repeatable.")
    meta.add_argument("--print-recipe", metavar="PATH", default=None,
                      help="After a successful run, snapshot the resolved "
                           "flags to a YAML recipe at PATH (replayable with "
                           "--recipe).")
    meta.add_argument("--help-group", metavar="NAME", default=None,
                      help="Print help only for the named argument group "
                           "and exit (e.g. --help-group moe).")

    # Required for greenfield mode; optional in baseline modifier mode.
    p.add_argument("--hardware", required=True, type=_normalize_hardware,
                   choices=VALID_HARDWARE,
                   help="Target hardware platform")
    p.add_argument("--params", default=None, type=parse_billions,
                   help="Target parameter count in billions (e.g., 7 or 7B)")
    p.add_argument("--tokens", default=None, type=parse_trillions,
                   help="Training token count in trillions (e.g., 2 or 2T)")

    # Architecture constraints
    p.add_argument("--context", type=parse_positive_int, default=8192,
                   help="Context length (default: 8192)")
    p.add_argument("--vocab-size", type=parse_positive_int, default=32000,
                   help="Vocabulary size (default: 32000)")
    p.add_argument("--param-tolerance", type=parse_non_negative_float, default=0.15,
                   help=("Parameter-count tolerance as a fraction of --params. "
                         "Default 0.15 (±15%%) — the search will accept any "
                         "shape whose total-param count lands in "
                         "[params*(1-tol), params*(1+tol)]. Tighten with "
                         "--param-tolerance 0.05 for a closer match."))
    # Deprecated alias — kept for back-compat, hidden from --help.
    p.add_argument("--param-band", dest="param_tolerance",
                   type=parse_non_negative_float, default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)

    # Serving constraints
    p.add_argument("--serving-tbt", type=parse_positive_float, default=None,
                   help="Serving time-between-tokens budget in ms")
    p.add_argument("--serving-ttft", type=parse_positive_float, default=None,
                   help="Serving time-to-first-token budget in ms")
    p.add_argument("--serving-batch", type=parse_positive_int, default=32,
                   help="Serving batch size (default: 32)")
    # Deprecated aliases — parsed for back-compat, hidden from --help.
    p.add_argument("--batch-size", dest="serving_batch",
                   type=parse_positive_int, default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--tbt-p95-ms", dest="serving_tbt",
                   type=parse_positive_float, default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--ttft-p95-ms", dest="serving_ttft",
                   type=parse_positive_float, default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)

    # Workload profile
    p.add_argument("--prompt-len", type=parse_positive_int, default=None,
                   help="Prompt length for serving (overrides --context for prefill)")
    p.add_argument("--output-len", type=parse_positive_int, default=512,
                   help="Expected generation length (default: 512)")
    p.add_argument("--concurrency", type=parse_positive_int, default=256,
                   help="Concurrent serving requests (default: 256)")
    p.add_argument("--scheduler", choices=["continuous", "static", "chunked"],
                   default="continuous", help="Serving scheduler type (default: continuous)")
    p.add_argument("--objective-profile",
                   choices=["balanced", "quality", "research_quality", "loss_only",
                            "latency", "serving_cost", "training_cost"],
                   default="research_quality",
                   help="Tradeoff preset for choosing the displayed optimal point from the Pareto frontier")

    # Parallelism
    p.add_argument("--tp", type=parse_positive_int, default=8,
                   help="Tensor parallelism degree (default: 8)")
    p.add_argument("--tp-options", default=None,
                   help="Comma-separated TP options for baseline modifier mode (e.g., 1,2,4,8)")
    p.add_argument("--pp", type=parse_positive_int, default=1,
                   help="Pipeline parallelism degree (default: 1)")
    p.add_argument("--dp", type=parse_positive_int, default=8,
                   help="Data parallelism degree (default: 8)")
    p.add_argument("--num-gpus", type=parse_positive_int, default=None,
                   help="Forward-compat shim. Ignored for modeling; --tp * --pp * --dp wins. "
                        "If passed and the product disagrees, AC prints a WARNING.")

    # Precision/KV search controls
    p.add_argument("--kv-dtypes", type=parse_kv_dtype_options, default=None,
                   help="Comma-separated KV dtype options, e.g. bf16,int8,int4")
    p.add_argument("--precision-modes", type=parse_precision_modes, default=None,
                   help="Comma-separated precision modes, e.g. bf16,fp8_ffn,fp4,mxfp4")

    # Output files
    p.add_argument("--output-config", default="arch.json",
                   help="Output JSON config path (default: arch.json). If "
                        "this is changed and the sibling outputs (justification, "
                        "pareto, shadow_prices) are left at their defaults, "
                        "they will follow --output-config into the same "
                        "directory rather than landing in the cwd.")
    p.add_argument("--output-justification", default=None,
                   help="Output markdown justification path (default: sibling "
                        "of --output-config, .md extension; or arch.md if "
                        "--output-config is at its default)")
    p.add_argument("--output-pareto", default=None,
                   help="Output Pareto frontier CSV path (default: sibling "
                        "of --output-config named pareto.csv; or pareto.csv "
                        "in cwd if --output-config is at its default)")
    p.add_argument("--output-shadow-prices", default=None,
                   help="Output shadow prices JSON path (default: sibling "
                        "of --output-config named shadow_prices.json; or in "
                        "cwd if --output-config is at its default)")
    p.add_argument("--output-assumptions", default=None,
                   help="Output assumptions markdown path (default: not written)")
    p.add_argument("--output-model-card", default=None,
                   help="Output model card markdown path (default: not written)")
    p.add_argument("--output-implementation", default=None,
                   help="Output generated PyTorch architecture implementation path (greenfield mode)")
    p.add_argument("--implementation-class-name", default="ACGeneratedModel",
                   help="Class name for --output-implementation (default: ACGeneratedModel)")

    # Baseline modifier mode
    p.add_argument("--baseline-config", default=None,
                   help="Compiler-schema baseline config to locally modify")
    p.add_argument("--out", default=None,
                   help="Output directory for baseline modifier mode")
    p.add_argument("--quality-risk-budget-pct", type=parse_non_negative_float, default=1.0,
                   help="Max relative loss-proxy increase when --allow-quality-spending is used (default: 1.0)")
    p.add_argument("--allow-quality-spending", action="store_true",
                   help="Allow modifier selection to spend quality risk; default selects only same-quality model-preserving moves")
    p.add_argument("--top-modifications", type=parse_positive_int, default=8,
                   help="Number of modifier candidates to show in reports (default: 8)")

    # v2 state/hybrid options
    p.add_argument("--allow-state", action="store_true",
                   help="Enable state/hybrid architecture search (state block + attention)")
    p.add_argument("--state-type", default="mamba2",
                   choices=["mamba2", "mamba", "gla", "kda", "gated_deltanet",
                            "deltanet", "rwkv7", "retnet", "swa",
                            "sliding_window", "linear_attention"],
                   help="State mechanism family (default: mamba2).")
    p.add_argument("--placement-strategy", default=None,
                   help="Comma-separated placement strategies for hybrid layers "
                        "(e.g., first_periodic_last,interleaved,periodic). "
                        "Default: search all three.")
    p.add_argument("--state-precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Precision for state parameters (default: bf16)")

    # v1 MoE
    p.add_argument("--allow-moe", action="store_true",
                   help="Enable MoE FFN candidates. --params is read as ACTIVE.")
    p.add_argument("--max-total-params-b", type=parse_positive_float, default=None,
                   help="MoE memory ceiling (total params billions).")
    p.add_argument("--moe-n-experts", default=None,
                   help="Comma-separated expert counts to sweep.")
    p.add_argument("--moe-top-k", default=None,
                   help="Comma-separated top-k options to sweep.")
    p.add_argument("--dense-ffn-layers", default=None,
                   help="Comma-separated first-K-dense layer counts to sweep.")
    p.add_argument("--ep-options", default=None,
                   help="Comma-separated expert-parallel degrees to sweep.")

    # MLA / MTP / CP / RoPE scaling
    p.add_argument("--allow-mla", action="store_true",
                   help="Enable MLA candidates (DeepSeek-V2/V3 style).")
    p.add_argument("--mla-kv-latent", default=None,
                   help="Comma-separated MLA c_kv options (default: 512).")
    p.add_argument("--mla-q-latent", default=None,
                   help="Comma-separated MLA c_q options (default: 1536).")
    p.add_argument("--allow-mtp", action="store_true",
                   help="Enable Multi-Token Prediction depth sweep.")
    p.add_argument("--mtp-depths", default=None,
                   help="Comma-separated MTP depths (e.g., 0,1,2).")
    p.add_argument("--cp", type=parse_positive_int, default=1,
                   help="Context parallelism degree (default: 1).")
    p.add_argument("--cp-method", default="ring", choices=["ring", "ulysses"],
                   help="CP method.")
    p.add_argument("--cp-options", default=None,
                   help="Comma-separated CP degrees to sweep.")
    p.add_argument("--nsa", action="store_true",
                   help="Emit NSA (Native Sparse Attention) block on optimal config.")
    p.add_argument("--nsa-compress-block-size", type=parse_positive_int, default=None)
    p.add_argument("--nsa-compress-block-stride", type=parse_positive_int, default=None)
    p.add_argument("--nsa-select-block-size", type=parse_positive_int, default=None)
    p.add_argument("--nsa-select-top-k", type=parse_positive_int, default=None)
    p.add_argument("--nsa-window-size", type=parse_positive_int, default=None)
    p.add_argument("--yoco", action="store_true",
                   help="Emit YOCO block on optimal config.")
    p.add_argument("--yoco-n-self-attn-layers", type=parse_positive_int, default=None)
    p.add_argument("--yoco-share-pattern", default=None,
                   choices=[None, "single_source", "block_shared"])
    p.add_argument("--allow-rope-scaling", action="store_true",
                   help="Enable RoPE extension method sweep.")
    p.add_argument("--rope-original-max-position", type=parse_positive_int, default=8192,
                   help="Pretrain context length (default: 8192).")
    p.add_argument("--rope-scaling-methods", default=None,
                   help="Comma-separated RoPE methods (yarn,ntk,longrope,pi,none).")

    # Options
    p.add_argument("--no-shadow-prices", action="store_true",
                   help="Skip shadow price computation (faster)")
    p.add_argument("--max-candidates", type=parse_positive_int, default=None,
                   help="Optional greenfield cap after deterministic candidate dedupe")
    p.add_argument("--progress-every", type=parse_non_negative_int, default=0,
                   help="Print greenfield evaluation progress every N candidates")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")

    p.epilog = (
        "logical groups for --help-group: hardware, workload, serving, "
        "parallelism, precision, state, moe, mla, mtp, rope, nsa, yoco, "
        "modifier, outputs, recipe."
    )
    return p


def parse_args(argv=None):
    """Recipe-aware wrapper around the argparse parser.

    Pipeline:
      1. Strip and apply --recipe / --override / --help-group /
         --print-recipe from argv (cli_recipe.expand_argv).
      2. If --help-group was passed, render only that group and exit.
      3. Otherwise, parse the expanded argv.
      4. Stash the print-recipe path on the namespace so main() can
         snapshot after a successful run.
    """
    if argv is None:
        argv = sys.argv[1:]
    expanded, print_recipe_path, help_group = expand_argv(argv)
    parser = _build_parser()
    if help_group:
        print(render_group_help(parser, help_group))
        sys.exit(0)
    # Capture the parser-default snapshot BEFORE parsing user input so
    # snapshot_recipe() can elide values the user didn't actually set.
    defaults_snapshot = {
        a.dest: a.default
        for a in parser._actions
        if a.dest != argparse.SUPPRESS and a.default is not argparse.SUPPRESS
    }
    args = parser.parse_args(expanded)
    setattr(args, "_print_recipe_path", print_recipe_path)
    setattr(args, "_defaults_snapshot", defaults_snapshot)
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # Subcommand dispatch — we keep argparse as the single source of
    # truth for the main flag surface and intercept only the two
    # recipe-related verbs at the front.
    if argv and argv[0] == "init":
        return _recipe_run_init("ac-compile init", argv[1:])
    if (
        len(argv) >= 2
        and argv[0] == "config"
        and argv[1] == "show"
    ):
        return _recipe_run_config_show(
            "ac-compile config show",
            _build_parser,
            argv[2:],
            infer_paths=infer_output_paths,
        )
    args = parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    # When the user points --output-config at a non-default location (e.g.
    # /tmp/run42/mistral_arch.json) and leaves the sibling outputs at
    # their defaults, route ALL siblings into the same directory AND have
    # them share the config's basename so a single `out/` directory can
    # hold runs from multiple recipes side by side. The prior behaviour
    # had .md inheriting the basename while pareto.csv and
    # shadow_prices.json hard-coded their names, which collided across
    # runs and was surprising to users who saw `mistral_arch.md` and
    # `pareto.csv` written together.
    #
    # Rule: derive a basename "stem" from --output-config and use
    # "{stem}.md", "{stem}_pareto.csv", "{stem}_shadow_prices.json", etc.
    # Special case: when the user accepts the default ("arch.json") the
    # stem strips down to "arch" and the historical short names
    # (pareto.csv, shadow_prices.json) are kept so existing scripts that
    # globbed those names still work.
    infer_output_paths(args)

    if args.baseline_config:
        return _run_modifier_mode(args, log)

    if args.params is None or args.tokens is None:
        print("ERROR: --params and --tokens are required when --baseline-config is not supplied.",
              file=sys.stderr)
        return 2

    # Fix #3: warn on unusual parallelism degrees. H100/B200 nodes have 8
    # GPUs and TPU pods are powers of 2; --tp / --pp / --dp values that are
    # neither 1 nor a power of 2 are almost always a typo (e.g. `--tp 3`
    # meant `--tp 8`). We don't reject — there's no hard rule — but we surface
    # the warning so it's not silent.
    def _is_pow2(n: int) -> bool:
        return n >= 1 and (n & (n - 1)) == 0
    for label, val in (("--tp", args.tp), ("--pp", args.pp), ("--dp", args.dp)):
        if val is not None and not _is_pow2(val):
            extra = ""
            if label == "--tp":
                extra = (
                    f" AC will restrict candidates to architectures whose "
                    f"n_heads (and n_kv_heads for non-MQA) divide evenly by "
                    f"{val}; common power-of-two head counts (e.g. 32, 64, 128) "
                    f"will be excluded."
                )
            print(
                f"WARNING: {label}={val} is not a power of two. Most accelerator "
                f"nodes use power-of-two parallelism; double-check this is "
                f"intended.{extra}",
                file=sys.stderr,
            )

    # --num-gpus is a forward-compat shim that other CLIs in the lab pass.
    # AC sizes the model from --tp / --pp / --dp directly, so --num-gpus is
    # ignored for modeling. Previously a mismatch between `--num-gpus` and
    # `tp*pp*dp` was silent, which let users believe their flag controlled
    # something it doesn't. Warn loudly so the user knows the number they
    # typed will be ignored.
    if getattr(args, "num_gpus", None) is not None:
        implied = (args.tp or 1) * (args.pp or 1) * (args.dp or 1)
        if int(args.num_gpus) != implied:
            print(
                f"WARNING: --num-gpus={args.num_gpus} does not match "
                f"--tp * --pp * --dp = {args.tp}*{args.pp}*{args.dp} = "
                f"{implied}. --num-gpus is accepted for command-line "
                "compatibility only; AC sizes the model from TP/PP/DP. "
                "Update --tp/--pp/--dp if you want a different total.",
                file=sys.stderr,
            )

    # Parse placement strategies if provided
    placement_strategies = None
    if getattr(args, "placement_strategy", None):
        placement_strategies = [s.strip() for s in args.placement_strategy.split(",")]

    def _split_int(v, label, minimum=1):
        if not v:
            return None
        out = [int(s.strip()) for s in v.split(",") if s.strip()]
        if not out:
            raise ValueError(f"{label} must contain at least one integer")
        bad = [x for x in out if x < minimum]
        if bad:
            op = ">=" if minimum == 0 else ">"
            limit = minimum if minimum == 0 else 0
            raise ValueError(f"{label} values must be {op} {limit}: {bad}")
        return out

    def _split_str(v):
        return [s.strip() for s in v.split(",") if s.strip()] if v else None

    try:
        moe_n_experts_options = _split_int(getattr(args, "moe_n_experts", None), "--moe-n-experts")
        moe_top_k_options     = _split_int(getattr(args, "moe_top_k", None), "--moe-top-k")
        dense_ffn_layer_opts  = _split_int(getattr(args, "dense_ffn_layers", None), "--dense-ffn-layers", minimum=0)
        ep_options            = _split_int(getattr(args, "ep_options", None), "--ep-options")
        mla_kv_latent_options = _split_int(getattr(args, "mla_kv_latent", None), "--mla-kv-latent")
        mla_q_latent_options  = _split_int(getattr(args, "mla_q_latent", None), "--mla-q-latent")
        mtp_depth_options     = _split_int(getattr(args, "mtp_depths", None), "--mtp-depths", minimum=0)
        cp_options            = _split_int(getattr(args, "cp_options", None), "--cp-options")
        rope_methods          = _split_str(getattr(args, "rope_scaling_methods", None))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Build constraints for greenfield compiler mode.
    try:
        constraints = DeploymentConstraints(
            target_params_b=args.params,
            param_tolerance=args.param_tolerance,
            training_tokens=int(args.tokens * 1e12),
            context_length=args.context,
            serving_tbt_ms=args.serving_tbt,
            serving_ttft_ms=args.serving_ttft,
            serving_batch=args.serving_batch,
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            vocab_size=args.vocab_size,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            concurrency=args.concurrency,
            scheduler=args.scheduler,
            objective_profile=args.objective_profile,
            precision_configs=args.precision_modes,
            kv_bits_options=args.kv_dtypes,
            # v2 state/hybrid
            allow_state=getattr(args, "allow_state", False),
            state_type=getattr(args, "state_type", "mamba2"),
            placement_strategies=placement_strategies,
            state_precision=getattr(args, "state_precision", "bf16"),
            # v1 MoE
            allow_moe=getattr(args, "allow_moe", False),
            max_total_params_b=getattr(args, "max_total_params_b", None),
            moe_n_experts_options=moe_n_experts_options,
            moe_top_k_options=moe_top_k_options,
            dense_ffn_layer_options=dense_ffn_layer_opts,
            ep_options=ep_options,
            # MLA
            allow_mla=getattr(args, "allow_mla", False),
            mla_kv_latent_options=mla_kv_latent_options,
            mla_q_latent_options=mla_q_latent_options,
            # MTP
            allow_mtp=getattr(args, "allow_mtp", False),
            mtp_depth_options=mtp_depth_options,
            # CP
            cp=getattr(args, "cp", 1),
            cp_method=getattr(args, "cp_method", "ring"),
            cp_options=cp_options,
            # RoPE scaling
            allow_rope_scaling=getattr(args, "allow_rope_scaling", False),
            rope_scaling_methods=rope_methods,
            rope_original_max_position=getattr(args, "rope_original_max_position", 8192),
            # Search ergonomics
            max_candidates=getattr(args, "max_candidates", None),
            progress_every=0 if args.quiet else getattr(args, "progress_every", 0),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Fix #8: long-context guardrails. When --context >> rope_original_max_position
    # and --allow-rope-scaling is off, the lattice will emit a config with no
    # RoPE scaling and the predicted loss will be unrealistically pessimistic.
    # Warn explicitly, and auto-cap candidates at very long context to avoid the
    # 45-second wall-clock cliff users were hitting.
    try:
        ctx = int(args.context)
    except Exception:
        ctx = 0
    pretrain_ctx_warn = int(getattr(args, "rope_original_max_position", 8192) or 8192)
    if ctx >= 4 * pretrain_ctx_warn and not getattr(args, "allow_rope_scaling", False):
        print(
            f"WARNING: --context={ctx} is {ctx // pretrain_ctx_warn}× the trained "
            f"range ({pretrain_ctx_warn}) and --allow-rope-scaling is not set. "
            "Predictions will assume vanilla RoPE and likely overstate quality "
            "loss. Pass `--allow-rope-scaling --rope-scaling-methods yarn,longrope` "
            "to enable the scaling sweep.",
            file=sys.stderr,
        )
    # When context is very long, the lattice × precision × KV-bits × rope-method
    # cross product grows quickly. Cap candidates so the CLI never hangs without
    # the user opting in.
    if ctx >= 262144 and getattr(args, "max_candidates", None) in (None, 0):
        default_cap = 500
        print(
            f"WARNING: --context={ctx} is very long; capping --max-candidates "
            f"at {default_cap} to keep search under a few seconds. Pass an "
            "explicit --max-candidates to override.",
            file=sys.stderr,
        )
        constraints.max_candidates = default_cap

    # Pre-flight: warn loudly if the user-requested precision modes are all
    # filtered out for this hardware. The optimizer silently falls back to
    # the hardware default, which previously made `--precision-modes fp4`
    # on H100 look like it had worked. Compute the intersection here so the
    # warning is visible before the search runs and is also recorded in the
    # output JSON for the calibration_warnings surface.
    precision_warnings: list = []
    if args.precision_modes:
        try:
            from optimizer import get_precision_configs_for_hardware  # type: ignore
            hw_supported = set(get_precision_configs_for_hardware(args.hardware))
        except Exception:
            hw_supported = set()
        requested = set(args.precision_modes or [])
        kept = requested & hw_supported
        if requested and not kept:
            msg = (
                f"--precision-modes={sorted(requested)} are all unsupported "
                f"on {args.hardware}; falling back to default "
                f"({sorted(hw_supported)[:3]})."
            )
            print(f"WARNING: {msg}", file=sys.stderr)
            precision_warnings.append(msg)
        elif requested - kept:
            dropped = sorted(requested - kept)
            msg = (
                f"--precision-modes dropped {dropped} (not supported on "
                f"{args.hardware}); kept {sorted(kept)}."
            )
            print(f"WARNING: {msg}", file=sys.stderr)
            precision_warnings.append(msg)

    # Warn once if the user is on a hardware target with no calibration
    # table on disk. The throughput model still runs (with default
    # efficiency multipliers) but the user should know the absolute
    # numbers are uncalibrated priors.
    try:
        from throughput_model import warn_if_uncalibrated
        warn_if_uncalibrated(args.hardware)
    except Exception:
        # Never let an advisory warning crash a real run.
        pass

    # Run optimizer
    log(f"[arch-compiler] Searching {args.params}B architectures on {args.hardware}...")
    t0 = time.time()
    try:
        result = optimize(args.hardware, constraints)
    except KeyboardInterrupt:
        print("Interrupted: architecture search cancelled by user.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    log(f"[arch-compiler] Search complete: {result.candidates_generated} candidates, "
        f"{result.candidates_feasible} feasible, {len(result.pareto_frontier)} Pareto, "
        f"{result.search_time_sec:.1f}s")

    if result.optimal is None:
        print(f"ERROR: No feasible architecture found for {args.params}B on {args.hardware}.",
              file=sys.stderr)
        # Targeted hint: if the param target is small relative to TP × tile
        # alignment, the most common cause is that no d_model on the lattice
        # is divisible by TP with enough heads to instantiate a transformer.
        tp = max(1, int(getattr(args, "tp", 1) or 1))
        params_b = float(getattr(args, "params", 0) or 0)
        if params_b > 0 and params_b < 1.0 and tp >= 4:
            print(
                f"Hint: --params={params_b}B with --tp={tp} leaves little room "
                "on the tile-aligned d_model lattice; try --tp 1 (or 2) for "
                "sub-1B models.",
                file=sys.stderr,
            )
        elif params_b >= 100 and not getattr(args, "pp", 0):
            print(
                f"Hint: --params={params_b}B usually needs --pp > 1 to fit on "
                f"{args.hardware}; the default pipeline degree is 1.",
                file=sys.stderr,
            )
        print("Try: relax serving constraints, widen param tolerance, lower --tp, "
              "raise --pp, or change hardware.",
              file=sys.stderr)

        # Still write justification explaining the failure
        md = generate_justification(result)
        ensure_parent_dir(args.output_justification)
        with open(args.output_justification, "w") as f:
            f.write(md)
        log(f"[arch-compiler] Wrote failure justification to {args.output_justification}")
        return 1

    opt = result.optimal
    log(_format_optimal_line(opt))
    # Decorate the loss numeral when no lab-calibration pack is loaded. The
    # README's "rank, don't predict" preamble says absolute loss is biased
    # priors as shipped; printing the bare number `Loss=2.0531` next to a
    # measured-looking `TPS=82127` invited users to quote it as if it were
    # measured. Star it until AC_QUALITY_DEFAULTS resolves to a pack.
    _q_override = os.environ.get("AC_QUALITY_DEFAULTS")
    _uncalibrated = not _q_override or not os.path.exists(_q_override)
    _loss_label = "Loss*" if _uncalibrated else "Loss"
    _loss_suffix = " (uncalibrated prior)" if _uncalibrated else ""
    log(f"[arch-compiler] {_loss_label}={opt.predicted_loss:.4f}{_loss_suffix} "
        f"TPS={opt.training_tps:.0f} TBT={opt.serving_tbt_ms:.1f}ms "
        f"Mem={opt.memory_per_gpu_gb:.1f}GB")

    # Confidence-envelope robustness. The optimizer computes
    # `robust_to_loss_uncertainty` (in `metadata.predicted.confidence_envelope`)
    # by counting candidates that overlap the picked candidate's loss CI band.
    # When false, the bare "Optimal:" banner above is over-confident: the top
    # several candidates are quality-equivalent within modeled uncertainty and
    # the picker tiebroke them on throughput/memory. Surface that so users do
    # not over-read the rank-1 row.
    contending_family_sidecar_path = None
    try:
        from optimizer import (
            compute_confidence_envelope,
            compute_contending_family_full,
        )
        envelope = compute_confidence_envelope(result, opt)
        if envelope and not envelope.get("robust_to_loss_uncertainty", True):
            n_contenders = int(envelope.get("contending_candidates", 0))
            if n_contenders >= 1:
                # Write the full top-32 family to a sidecar JSON. The
                # emitted config carries only the top-5 inline; users
                # who want the broader picture (auto-calibration runs,
                # dashboards) follow the path in the warning message.
                sidecar_dir = os.path.dirname(os.path.abspath(args.output_config))
                sidecar_stem = os.path.splitext(os.path.basename(args.output_config))[0]
                sidecar_path = os.path.join(
                    sidecar_dir, f"{sidecar_stem}_contending_family.json"
                )
                try:
                    full_family = compute_contending_family_full(result, opt, top_n=32)
                    ensure_parent_dir(sidecar_path)
                    with open(sidecar_path, "w") as _f:
                        json.dump(
                            {
                                "robust_to_loss_uncertainty": False,
                                "contending_candidates": n_contenders,
                                "loss_low": envelope.get("loss_low"),
                                "loss_high": envelope.get("loss_high"),
                                "uncertainty_total_pct": envelope.get(
                                    "uncertainty_total_pct"
                                ),
                                "contending_family": full_family,
                            },
                            _f,
                            indent=2,
                        )
                    contending_family_sidecar_path = sidecar_path
                except Exception:
                    contending_family_sidecar_path = None
                tail = (
                    f" Full top-32 family written to {contending_family_sidecar_path}."
                    if contending_family_sidecar_path
                    else ""
                )
                # Vary the attribution depending on whether a lab
                # calibration pack is loaded. Pre-fix, the message
                # always blamed "uncalibrated quality-model uncertainty"
                # even when AC_QUALITY_DEFAULTS resolved to a pack —
                # which is misleading, because a non-robust envelope
                # after calibration just means the lab's modeled
                # uncertainty is genuinely wide for this run, not that
                # calibration is missing.
                uncertainty_attribution = (
                    "uncalibrated quality-model uncertainty"
                    if _uncalibrated
                    else "the calibrated quality-model uncertainty band"
                )
                print(
                    f"WARNING: {n_contenders} contending candidate(s) sit "
                    "inside the loss CI band of the picked architecture "
                    f"(rank-1 is not robust to {uncertainty_attribution}). "
                    "The picker tiebroke them on "
                    "throughput/memory; inspect the pareto.csv before "
                    "treating the `Optimal:` row as unique." + tail,
                    file=sys.stderr,
                )
    except Exception:
        pass

    # MoE-fallback warning: the user opted into MoE search but the picked
    # config is dense, which usually means either no MoE candidate beat the
    # dense Pareto front under the chosen objective, or the requested expert
    # topology was filtered out by --max-total-params-b / tile alignment / TP
    # constraints. Make this visible so silent dense fallback is not mistaken
    # for "MoE didn't help."
    if getattr(args, "allow_moe", False) and getattr(opt.arch, "moe_style", "dense") == "dense":
        any_moe_in_frontier = any(
            getattr(c.arch, "moe_style", "dense") != "dense"
            for c in (result.pareto_frontier or [])
        )
        requested_experts = getattr(args, "moe_n_experts", None)
        if any_moe_in_frontier:
            msg = (
                "--allow-moe was set, but the picked architecture is dense. "
                "MoE candidates exist on the Pareto frontier but were dominated "
                "by dense under the current objective profile; inspect the CSV "
                "or rerun with --objective-profile=quality to prefer MoE."
            )
        else:
            extra = (
                f" Requested --moe-n-experts={requested_experts}"
                if requested_experts else ""
            )
            msg = (
                "--allow-moe was set, but no MoE candidate was feasible under "
                "the current constraints, so the picked architecture is dense."
                f"{extra} Common causes: --max-total-params-b too low, "
                "n_experts incompatible with --ep-options × tile alignment, "
                "or expert_dim too small for the model width."
            )
        print(f"WARNING: {msg}", file=sys.stderr)

    # MoE depth-floor warning: very shallow stacks at MoE scale are unusual
    # in the published literature. Mixtral 8×7B (47B total) uses 32 layers;
    # DeepSeek-V3 (671B total) uses 61; GPT-OSS-120B uses 36. The AC
    # picker tends to favour width over depth for MoE because the FFN-
    # attention ratio prior pulls toward wide FFNs. Surface this so the
    # user can decide whether the wide-and-shallow pick is intentional.
    if getattr(opt.arch, "moe_style", "dense") != "dense":
        total_b = float(opt.arch.total_params_b or 0.0)
        # Depth floor fitted to published MoE anchors:
        #   Mixtral 8×7B (47B total)     → 32 layers
        #   GPT-OSS-120B (~117B total)   → 36 layers
        #   DeepSeek-V3 (671B total)     → 61 layers
        # Power law: floor ≈ 24 * (total_b / 20)^0.2
        # Anything below 80% of this floor gets flagged.
        floor = max(20, int(round(24 * max(total_b / 20.0, 1.0) ** 0.2)))
        if opt.arch.n_layers < 0.8 * floor:
            print(
                f"WARNING: selected MoE has n_layers={opt.arch.n_layers}, "
                f"unusually shallow for {total_b:.0f}B total params "
                f"(published Mixtral / GPT-OSS / DeepSeek-V3 sit around "
                f"{floor}+ layers at this scale). Wide-and-shallow MoE "
                f"often trains but under-uses depth — verify against a "
                f"depth-anchored baseline before committing.",
                file=sys.stderr,
            )

    # Shadow prices
    shadow_report = None
    if not args.no_shadow_prices:
        log(f"[arch-compiler] Computing shadow prices...")
        shadow_report = compute_shadow_prices(args.hardware, constraints, result)
        log(f"[arch-compiler] Computed {len(shadow_report.prices)} shadow prices")

    # NSA / YOCO stamp blocks (optimizer doesn't sweep them yet).
    nsa_block = None
    if getattr(args, "nsa", False):
        nsa_block = {}
        for k in ("compress_block_size", "compress_block_stride",
                  "select_block_size", "select_top_k", "window_size"):
            v = getattr(args, f"nsa_{k}", None)
            if v is not None:
                nsa_block[k] = int(v)
    yoco_block = None
    if getattr(args, "yoco", False):
        yoco_block = {"enabled": True}
        if getattr(args, "yoco_n_self_attn_layers", None) is not None:
            yoco_block["n_self_attn_layers"] = int(args.yoco_n_self_attn_layers)
        if getattr(args, "yoco_share_pattern", None):
            yoco_block["share_pattern"] = str(args.yoco_share_pattern)

    # Write outputs
    # 1. JSON config
    config = result_to_config(result, nsa=nsa_block, yoco=yoco_block)
    ensure_parent_dir(args.output_config)
    save_config(config, args.output_config)
    log(f"[arch-compiler] Wrote {args.output_config}")
    if args.output_implementation:
        try:
            ensure_parent_dir(args.output_implementation)
            save_pytorch_implementation(
                config,
                args.output_implementation,
                class_name=args.implementation_class_name,
                source_name=args.output_config,
            )
        except ValueError as exc:
            print(f"ERROR: Could not generate implementation: {exc}", file=sys.stderr)
            return 2
        log(f"[arch-compiler] Wrote {args.output_implementation}")

    # 2. Justification
    md = generate_justification(result, shadow_report)
    # Diagnostics: when the user enabled MTP or RoPE-scaling sweeps but the
    # optimizer did not pick them, surface a one-line note so the report does
    # not silently look like those flags were no-ops.
    enabled_but_dropped = []
    arch_block = (config or {}).get("architecture", {}) or {}
    if getattr(args, "allow_mtp", False):
        mtp_cfg = arch_block.get("mtp")
        n_depths = int((mtp_cfg or {}).get("n_predict_depths", 0))
        if not mtp_cfg or n_depths == 0:
            enabled_but_dropped.append(
                "- MTP was enabled (`--allow-mtp`) but the selected candidate "
                "uses `mtp_depths=0`. The optimizer ranked MTP-off above MTP-on "
                "for this (params, tokens, hardware) point — typically because "
                "the MTP sample-efficiency bonus is smaller than competing "
                "shape moves at this scale."
            )
    if getattr(args, "allow_rope_scaling", False):
        pos = arch_block.get("positional_encoding") or {}
        scaling = pos.get("scaling")
        pretrain_ctx = int(getattr(args, "rope_original_max_position", 8192))
        target_ctx = int(getattr(args, "context", 8192))
        if (not scaling or scaling.get("method") in (None, "none")) and target_ctx > pretrain_ctx:
            enabled_but_dropped.append(
                f"- RoPE scaling was enabled (`--allow-rope-scaling`) but the "
                f"selected candidate uses no scaling (`method=none`) at "
                f"context={target_ctx} > pretrain_max={pretrain_ctx}. "
                f"Long-context degradation is not modeled in the chosen "
                f"candidate; if you intend to serve at the target context, "
                f"re-run with a narrower `--rope-scaling-methods` list to "
                f"force a scaling pick."
            )
    stamp_notes = []
    if nsa_block is not None:
        stamp_notes.append(
            "- NSA was applied as a post-search emission stamp. The optimizer "
            "selected the lattice point before this sparse-attention block was added."
        )
    if yoco_block is not None:
        stamp_notes.append(
            "- YOCO was applied as a post-search emission stamp. The optimizer "
            "selected the lattice point before this KV-sharing block was added."
        )
    if stamp_notes:
        md += "\n## Post-search Stamps\n" + "\n".join(stamp_notes) + "\n"
    if enabled_but_dropped:
        md += (
            "\n## Enabled-but-Dropped Sweeps\n"
            "These sweeps were enabled on the command line but did not appear "
            "in the selected candidate.\n\n"
            + "\n".join(enabled_but_dropped)
            + "\n"
        )
    # Surface precision-filter and parallelism-override warnings inline so the
    # user does not have to scroll back through stderr to find them.
    cli_warnings: list = list(precision_warnings)
    requested_cp = getattr(args, "cp", 1) or 1
    actual_cp = (config or {}).get("parallelism", {}).get(
        "context_parallel", requested_cp)
    if int(requested_cp) > 1 and int(actual_cp) != int(requested_cp):
        msg = (
            f"--cp={requested_cp} was requested but the selected candidate "
            f"uses context_parallel={actual_cp}. This usually means an EP or "
            f"PP sweep produced a winning candidate that rebalanced the "
            f"parallelism plan; re-run with --cp-options narrowed or without "
            f"--allow-moe to lock CP."
        )
        print(f"WARNING: {msg}", file=sys.stderr)
        cli_warnings.append(msg)
    if cli_warnings:
        md += (
            "\n## CLI Warnings\n"
            + "\n".join(f"- {w}" for w in cli_warnings)
            + "\n"
        )
        # Also pin them onto the JSON config metadata.predicted so downstream
        # tooling sees them.
        try:
            if config is not None:
                pred = config.setdefault("metadata", {}).setdefault(
                    "predicted", {})
                existing = list(pred.get("calibration_warnings") or [])
                existing.extend(cli_warnings)
                pred["calibration_warnings"] = existing
                save_config(config, args.output_config)
        except Exception:
            pass
    ensure_parent_dir(args.output_justification)
    with open(args.output_justification, "w") as f:
        f.write(md)
    log(f"[arch-compiler] Wrote {args.output_justification}")

    # 3. Pareto CSV
    csv = result_to_pareto_csv(result)
    ensure_parent_dir(args.output_pareto)
    with open(args.output_pareto, "w") as f:
        f.write(csv)
    log(f"[arch-compiler] Wrote {args.output_pareto} ({len(result.pareto_frontier)} points)")

    # 4. Shadow prices JSON
    if shadow_report:
        sp_json = shadow_prices_to_json(shadow_report)
        ensure_parent_dir(args.output_shadow_prices)
        with open(args.output_shadow_prices, "w") as f:
            json.dump(sp_json, f, indent=2)
        log(f"[arch-compiler] Wrote {args.output_shadow_prices}")

    # 5. Assumptions
    if args.output_assumptions:
        ensure_parent_dir(args.output_assumptions)
        with open(args.output_assumptions, "w") as f:
            f.write(generate_assumptions())
        log(f"[arch-compiler] Wrote {args.output_assumptions}")

    # 6. Model card
    if args.output_model_card:
        ensure_parent_dir(args.output_model_card)
        with open(args.output_model_card, "w") as f:
            f.write(generate_model_card())
        log(f"[arch-compiler] Wrote {args.output_model_card}")

    # Report binding constraints
    if result.binding_constraints:
        log(f"[arch-compiler] Binding constraints: {', '.join(result.binding_constraints)}")
    if result.optimal and result.optimal.binding_serving_regime:
        log(f"[arch-compiler] Serving regime: {result.optimal.binding_serving_regime}")

    elapsed = time.time() - t0
    log(f"[arch-compiler] Total time: {elapsed:.1f}s")
    # --print-recipe: snapshot the resolved flag set after a successful
    # run so it can be replayed exactly with --recipe. Only fires on
    # success; a failed/infeasible run shouldn't pollute a recipes/ dir.
    pr_path = getattr(args, "_print_recipe_path", None)
    if pr_path:
        try:
            snapshot_recipe(args, pr_path)
            log(f"[arch-compiler] Wrote replay recipe to {pr_path}")
        except Exception as e:
            log(f"[arch-compiler] WARNING: could not write --print-recipe: {e}")
    return 0


def _run_modifier_mode(args, log):
    """Run baseline-aware Pareto modifier mode."""

    try:
        baseline = load_baseline_model(args.baseline_config)
    except (BaselineUnsupportedError, ValueError) as exc:
        print(f"ERROR: Could not load baseline config: {exc}", file=sys.stderr)
        return 2

    params_b = args.params if args.params is not None else baseline.total_params_b
    tokens_t = args.tokens if args.tokens is not None else 2.0

    try:
        constraints = DeploymentConstraints(
            target_params_b=params_b,
            param_tolerance=args.param_tolerance,
            training_tokens=int(tokens_t * 1e12),
            context_length=args.context,
            serving_tbt_ms=args.serving_tbt,
            serving_ttft_ms=args.serving_ttft,
            serving_batch=args.serving_batch,
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            vocab_size=baseline.candidate.vocab_size,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            concurrency=args.concurrency,
            scheduler=args.scheduler,
            objective_profile=args.objective_profile,
            precision_configs=args.precision_modes,
            kv_bits_options=args.kv_dtypes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    tp_options = parse_int_list(args.tp_options) if args.tp_options else [args.tp]
    out_dir = args.out or os.path.join("outputs", f"{baseline.name}_modifier")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from throughput_model import warn_if_uncalibrated
        warn_if_uncalibrated(args.hardware)
    except Exception:
        pass

    log(f"[arch-compiler] Modifier mode: baseline={baseline.name} hardware={args.hardware}")
    log(f"[arch-compiler] Local TP options: {','.join(str(t) for t in tp_options)}")

    try:
        result = run_modifier_search(
            baseline,
            args.hardware,
            constraints,
            tp_options=tp_options,
            quality_risk_budget_pct=args.quality_risk_budget_pct,
            allow_quality_spending=args.allow_quality_spending,
            top_near_dominating=args.top_modifications,
        )
    except KeyboardInterrupt:
        print("Interrupted: modifier search cancelled by user.", file=sys.stderr)
        return 130

    selected = result.selected
    c = selected.evaluated.arch
    log(f"[arch-compiler] Modifier search complete: {result.candidates_evaluated} evaluated, "
        f"{len(result.feasible_records)} feasible, {len(result.pareto_frontier)} Pareto, "
        f"{result.search_time_sec:.1f}s")
    # If the local sweep found nothing feasible, surface that as a real
    # error rather than silently falling back to the baseline as "Selected".
    # Previously: a downstream pipeline reading the emitted config.json
    # would get the broken baseline back with exit code 0. Now we refuse
    # to write the bundle and exit non-zero so automation notices.
    if len(result.feasible_records) == 0:
        print(
            f"ERROR: Modifier search on baseline `{baseline.name}` produced "
            f"0 feasible candidates out of {result.candidates_evaluated} "
            f"evaluated. The baseline itself may be infeasible (e.g. MoE "
            f"config missing `parallelism.expert_parallel`), or the "
            f"workload / parallelism constraints rule out every local "
            f"modification. Re-check the baseline config and constraints, "
            f"then re-run.",
            file=sys.stderr,
        )
        return 2
    # Distinguish "modifier picked the baseline because nothing dominated it"
    # from "modifier picked a real modification". Downstream automation can
    # gate on the SELECTED-IS-BASELINE banner instead of diffing config.json.
    if getattr(selected, "is_baseline", False) or not selected.changes:
        n_eval = result.candidates_evaluated
        log(f"[arch-compiler] Selected: baseline (no Pareto-improving "
            f"modification found among {n_eval} candidate(s))")
        log(f"[arch-compiler] NOTE: emitted config.json is the baseline "
            f"itself; re-run with --allow-quality-spending or wider sweep "
            f"flags to explore quality-trading moves.")
    else:
        log(f"[arch-compiler] Selected: {selected.change_summary}")
    log(f"[arch-compiler] Config: d={c.d_model} L={c.n_layers} h={c.n_heads} "
        f"kv={c.n_kv_heads} ffn={c.ffn_dim} ffn_prec={c.ffn_precision} "
        f"kv_bits={c.kv_cache_bits} TP={selected.tp}")

    paths = {
        "config": os.path.join(out_dir, "config.json"),
        "pareto": os.path.join(out_dir, "pareto.csv"),
        "baseline_delta": os.path.join(out_dir, "baseline_delta.md"),
        "shadow": os.path.join(out_dir, "shadow_prices.md"),
        "justification": os.path.join(out_dir, "justification.md"),
        "assumptions": os.path.join(out_dir, "assumptions.md"),
        "model_card": os.path.join(out_dir, "model_card.md"),
    }

    save_config(modifier_result_to_config(result), paths["config"])
    with open(paths["pareto"], "w") as f:
        f.write(modifier_pareto_to_csv(result))
    with open(paths["baseline_delta"], "w") as f:
        f.write(generate_baseline_delta_report(result, args.top_modifications))
    with open(paths["shadow"], "w") as f:
        f.write(generate_modifier_shadow_report(result, args.top_modifications))
    with open(paths["justification"], "w") as f:
        f.write(generate_modifier_justification(result, args.top_modifications))
    with open(paths["assumptions"], "w") as f:
        f.write(generate_assumptions())
    with open(paths["model_card"], "w") as f:
        f.write(generate_model_card())

    for path in paths.values():
        log(f"[arch-compiler] Wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
