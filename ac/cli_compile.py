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

# Wire up paths
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from optimizer import optimize, result_to_config, result_to_pareto_csv, DeploymentConstraints
from shadow_prices import compute_shadow_prices, shadow_prices_to_json
from justification import generate_justification, generate_assumptions, generate_model_card
from schema import save_config
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


VALID_HARDWARE = [
    "h100", "b200", "tpu_v5p", "tpu_v5e",
    "trainium2", "trn2", "trainium3", "trn3",
]


def parse_billions(value: str) -> float:
    """Parse parameter count in billions, accepting '7' or '7B'."""
    s = str(value).strip().lower().replace("_", "")
    if s.endswith("b"):
        s = s[:-1]
    return float(s)


def parse_trillions(value: str) -> float:
    """Parse token count in trillions, accepting '2' or '2T'."""
    s = str(value).strip().lower().replace("_", "")
    if s.endswith("t"):
        s = s[:-1]
    return float(s)


def parse_int_list(value: str):
    """Parse comma-separated positive integers."""
    return [int(v.strip()) for v in value.split(",") if v.strip()]


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
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="ac-compile",
        description="Hardware-aware architecture compiler v0.3. "
                    "Takes hardware spec + deployment constraints, "
                    "emits optimal architecture config + justification.",
    )

    # Required for greenfield mode; optional in baseline modifier mode.
    p.add_argument("--hardware", required=True, choices=VALID_HARDWARE,
                   help="Target hardware platform")
    p.add_argument("--params", default=None, type=parse_billions,
                   help="Target parameter count in billions (e.g., 7 or 7B)")
    p.add_argument("--tokens", default=None, type=parse_trillions,
                   help="Training token count in trillions (e.g., 2 or 2T)")

    # Architecture constraints
    p.add_argument("--context", type=int, default=8192,
                   help="Context length (default: 8192)")
    p.add_argument("--vocab-size", type=int, default=32000,
                   help="Vocabulary size (default: 32000)")
    p.add_argument("--param-tolerance", type=float, default=0.15,
                   help="Parameter count tolerance fraction (default: 0.15)")
    p.add_argument("--param-band", dest="param_tolerance", type=float, default=argparse.SUPPRESS,
                   help="Alias for --param-tolerance")

    # Serving constraints
    p.add_argument("--serving-tbt", type=float, default=None,
                   help="Serving time-between-tokens budget in ms")
    p.add_argument("--serving-ttft", type=float, default=None,
                   help="Serving time-to-first-token budget in ms")
    p.add_argument("--serving-batch", type=int, default=32,
                   help="Serving batch size (default: 32)")
    p.add_argument("--batch-size", dest="serving_batch", type=int, default=argparse.SUPPRESS,
                   help="Alias for --serving-batch")
    p.add_argument("--tbt-p95-ms", dest="serving_tbt", type=float, default=argparse.SUPPRESS,
                   help="Alias for --serving-tbt")
    p.add_argument("--ttft-p95-ms", dest="serving_ttft", type=float, default=argparse.SUPPRESS,
                   help="Alias for --serving-ttft")

    # Workload profile
    p.add_argument("--prompt-len", type=int, default=None,
                   help="Prompt length for serving (overrides --context for prefill)")
    p.add_argument("--output-len", type=int, default=512,
                   help="Expected generation length (default: 512)")
    p.add_argument("--concurrency", type=int, default=256,
                   help="Concurrent serving requests (default: 256)")
    p.add_argument("--scheduler", choices=["continuous", "static", "chunked"],
                   default="continuous", help="Serving scheduler type (default: continuous)")

    # Parallelism
    p.add_argument("--tp", type=int, default=8,
                   help="Tensor parallelism degree (default: 8)")
    p.add_argument("--tp-options", default=None,
                   help="Comma-separated TP options for baseline modifier mode (e.g., 1,2,4,8)")
    p.add_argument("--pp", type=int, default=1,
                   help="Pipeline parallelism degree (default: 1)")
    p.add_argument("--dp", type=int, default=8,
                   help="Data parallelism degree (default: 8)")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Accepted for v0.5 command compatibility; TP/PP/DP still control modeling")

    # Precision/KV search controls
    p.add_argument("--kv-dtypes", type=parse_kv_dtype_options, default=None,
                   help="Comma-separated KV dtype options, e.g. bf16,int8,int4")
    p.add_argument("--precision-modes", type=parse_precision_modes, default=None,
                   help="Comma-separated precision modes, e.g. bf16,fp8_ffn,fp4,mxfp4")

    # Output files
    p.add_argument("--output-config", default="arch.json",
                   help="Output JSON config path (default: arch.json)")
    p.add_argument("--output-justification", default="arch.md",
                   help="Output markdown justification path (default: arch.md)")
    p.add_argument("--output-pareto", default="pareto.csv",
                   help="Output Pareto frontier CSV path (default: pareto.csv)")
    p.add_argument("--output-shadow-prices", default="shadow_prices.json",
                   help="Output shadow prices JSON path (default: shadow_prices.json)")
    p.add_argument("--output-assumptions", default=None,
                   help="Output assumptions markdown path (default: not written)")
    p.add_argument("--output-model-card", default=None,
                   help="Output model card markdown path (default: not written)")

    # Baseline modifier mode
    p.add_argument("--baseline-config", default=None,
                   help="Compiler-schema baseline config to locally modify")
    p.add_argument("--out", default=None,
                   help="Output directory for baseline modifier mode")
    p.add_argument("--quality-risk-budget-pct", type=float, default=1.0,
                   help="Max relative loss-proxy increase when --allow-quality-spending is used (default: 1.0)")
    p.add_argument("--allow-quality-spending", action="store_true",
                   help="Allow modifier selection to spend quality risk; default selects only same-quality model-preserving moves")
    p.add_argument("--top-modifications", type=int, default=8,
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
    p.add_argument("--max-total-params-b", type=float, default=None,
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
    p.add_argument("--cp", type=int, default=1,
                   help="Context parallelism degree (default: 1).")
    p.add_argument("--cp-method", default="ring", choices=["ring", "ulysses"],
                   help="CP method.")
    p.add_argument("--cp-options", default=None,
                   help="Comma-separated CP degrees to sweep.")
    p.add_argument("--nsa", action="store_true",
                   help="Emit NSA (Native Sparse Attention) block on optimal config.")
    p.add_argument("--nsa-compress-block-size", type=int, default=None)
    p.add_argument("--nsa-compress-block-stride", type=int, default=None)
    p.add_argument("--nsa-select-block-size", type=int, default=None)
    p.add_argument("--nsa-select-top-k", type=int, default=None)
    p.add_argument("--nsa-window-size", type=int, default=None)
    p.add_argument("--yoco", action="store_true",
                   help="Emit YOCO block on optimal config.")
    p.add_argument("--yoco-n-self-attn-layers", type=int, default=None)
    p.add_argument("--yoco-share-pattern", default=None,
                   choices=[None, "single_source", "block_shared"])
    p.add_argument("--allow-rope-scaling", action="store_true",
                   help="Enable RoPE extension method sweep.")
    p.add_argument("--rope-original-max-position", type=int, default=8192,
                   help="Pretrain context length (default: 8192).")
    p.add_argument("--rope-scaling-methods", default=None,
                   help="Comma-separated RoPE methods (yarn,ntk,longrope,pi,none).")

    # Options
    p.add_argument("--no-shadow-prices", action="store_true",
                   help="Skip shadow price computation (faster)")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Optional greenfield cap after deterministic candidate dedupe")
    p.add_argument("--progress-every", type=int, default=0,
                   help="Print greenfield evaluation progress every N candidates")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    if args.baseline_config:
        return _run_modifier_mode(args, log)

    if args.params is None or args.tokens is None:
        print("ERROR: --params and --tokens are required when --baseline-config is not supplied.",
              file=sys.stderr)
        return 2

    # Parse placement strategies if provided
    placement_strategies = None
    if getattr(args, "placement_strategy", None):
        placement_strategies = [s.strip() for s in args.placement_strategy.split(",")]

    def _split_int(v):
        return [int(s.strip()) for s in v.split(",") if s.strip()] if v else None

    def _split_str(v):
        return [s.strip() for s in v.split(",") if s.strip()] if v else None

    moe_n_experts_options = _split_int(getattr(args, "moe_n_experts", None))
    moe_top_k_options     = _split_int(getattr(args, "moe_top_k", None))
    dense_ffn_layer_opts  = _split_int(getattr(args, "dense_ffn_layers", None))
    ep_options            = _split_int(getattr(args, "ep_options", None))
    mla_kv_latent_options = _split_int(getattr(args, "mla_kv_latent", None))
    mla_q_latent_options  = _split_int(getattr(args, "mla_q_latent", None))
    mtp_depth_options     = _split_int(getattr(args, "mtp_depths", None))
    cp_options            = _split_int(getattr(args, "cp_options", None))
    rope_methods          = _split_str(getattr(args, "rope_scaling_methods", None))

    # Build constraints for greenfield compiler mode.
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

    # Run optimizer
    log(f"[arch-compiler] Searching {args.params}B architectures on {args.hardware}...")
    t0 = time.time()
    try:
        result = optimize(args.hardware, constraints)
    except KeyboardInterrupt:
        print("Interrupted: architecture search cancelled by user.", file=sys.stderr)
        return 130
    log(f"[arch-compiler] Search complete: {result.candidates_generated} candidates, "
        f"{result.candidates_feasible} feasible, {len(result.pareto_frontier)} Pareto, "
        f"{result.search_time_sec:.1f}s")

    if result.optimal is None:
        print(f"ERROR: No feasible architecture found for {args.params}B on {args.hardware}.",
              file=sys.stderr)
        print("Try: relax serving constraints, widen param tolerance, or change hardware.",
              file=sys.stderr)

        # Still write justification explaining the failure
        md = generate_justification(result)
        with open(args.output_justification, "w") as f:
            f.write(md)
        log(f"[arch-compiler] Wrote failure justification to {args.output_justification}")
        return 1

    opt = result.optimal
    base_log = (f"[arch-compiler] Optimal: d={opt.arch.d_model} L={opt.arch.n_layers} "
                f"h={opt.arch.n_heads} kv={opt.arch.n_kv_heads} ffn={opt.arch.ffn_dim} "
                f"prec={opt.arch.ffn_precision} kv_bits={opt.arch.kv_cache_bits}")
    if opt.arch.n_state_layers > 0:
        base_log += (f" state={opt.arch.n_state_layers}/{opt.arch.n_layers} "
                     f"d_state={opt.arch.derived_d_state} "
                     f"placement={opt.arch.placement_strategy}")
    log(base_log)
    log(f"[arch-compiler] Loss={opt.predicted_loss:.4f} "
        f"TPS={opt.training_tps:.0f} TBT={opt.serving_tbt_ms:.1f}ms "
        f"Mem={opt.memory_per_gpu_gb:.1f}GB")

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
    save_config(config, args.output_config)
    log(f"[arch-compiler] Wrote {args.output_config}")

    # 2. Justification
    md = generate_justification(result, shadow_report)
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
    with open(args.output_justification, "w") as f:
        f.write(md)
    log(f"[arch-compiler] Wrote {args.output_justification}")

    # 3. Pareto CSV
    csv = result_to_pareto_csv(result)
    with open(args.output_pareto, "w") as f:
        f.write(csv)
    log(f"[arch-compiler] Wrote {args.output_pareto} ({len(result.pareto_frontier)} points)")

    # 4. Shadow prices JSON
    if shadow_report:
        sp_json = shadow_prices_to_json(shadow_report)
        with open(args.output_shadow_prices, "w") as f:
            json.dump(sp_json, f, indent=2)
        log(f"[arch-compiler] Wrote {args.output_shadow_prices}")

    # 5. Assumptions
    if args.output_assumptions:
        with open(args.output_assumptions, "w") as f:
            f.write(generate_assumptions())
        log(f"[arch-compiler] Wrote {args.output_assumptions}")

    # 6. Model card
    if args.output_model_card:
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
        precision_configs=args.precision_modes,
        kv_bits_options=args.kv_dtypes,
    )

    tp_options = parse_int_list(args.tp_options) if args.tp_options else [args.tp]
    out_dir = args.out or os.path.join("outputs", f"{baseline.name}_modifier")
    os.makedirs(out_dir, exist_ok=True)

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
