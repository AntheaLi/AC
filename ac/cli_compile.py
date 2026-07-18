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

try:
    from .optimizer import (
        optimize, result_to_config, result_to_pareto_csv,
        DeploymentConstraints, _same_candidate,
    )
    from .cli_recipe import (
        expand_argv, render_group_help, snapshot_recipe,
        run_init as _recipe_run_init,
        run_config_show as _recipe_run_config_show,
    )
    from .shadow_prices import compute_shadow_prices, shadow_prices_to_json
    from .justification import (
        generate_justification, generate_assumptions, generate_model_card,
    )
    from .schema import save_config
    from .implementation_generator import save_pytorch_implementation
    from .baseline import BaselineUnsupportedError, load_baseline_model
    from .modifier import (
        modifier_pareto_to_csv, modifier_result_to_config,
        run_modifier_search,
    )
    from .baseline_delta import (
        generate_baseline_delta_report, generate_modifier_justification,
        generate_modifier_shadow_report,
    )
    from .pricing import attach_cost_block
except ImportError:
    from optimizer import (
        optimize, result_to_config, result_to_pareto_csv,
        DeploymentConstraints, _same_candidate,
    )
    from cli_recipe import (
        expand_argv, render_group_help, snapshot_recipe,
        run_init as _recipe_run_init,
        run_config_show as _recipe_run_config_show,
    )
    from shadow_prices import compute_shadow_prices, shadow_prices_to_json
    from justification import (
        generate_justification, generate_assumptions, generate_model_card,
    )
    from schema import save_config
    from implementation_generator import save_pytorch_implementation
    from baseline import BaselineUnsupportedError, load_baseline_model
    from modifier import (
        modifier_pareto_to_csv, modifier_result_to_config,
        run_modifier_search,
    )
    from baseline_delta import (
        generate_baseline_delta_report, generate_modifier_justification,
        generate_modifier_shadow_report,
    )
    from pricing import attach_cost_block


# Canonical hardware names shown in --help. Trainium short forms
# (`trn2`/`trn3`) are still parsed via `_normalize_hardware` below but
# hidden from --help so users only see one name per platform.
VALID_HARDWARE = [
    "h100", "b200", "gb200_nvl72", "h800", "tpu_v5p", "tpu_v5e",
    "trainium2", "trainium3",
]

_HARDWARE_ALIASES = {
    "trn2": "trainium2",
    "trn3": "trainium3",
}


def _normalize_hardware(value: str) -> str:
    v = (value or "").strip().lower()
    return _HARDWARE_ALIASES.get(v, v)


# State families accepted by the greenfield CLI. Keep this aligned with
# schema._SUPPORTED_STATE_TYPES and quality_model._resolve_hybrid_family.
STATE_TYPE_ALIASES = {
    "mamba1": "mamba",
    "delta_net": "deltanet",
    "gated_delta": "gated_deltanet",
    "gated_linear_attention": "gla",
    "rwkv": "rwkv7",
    "swa": "sliding_window",
    "local_recurrent": "sliding_window",
}
VALID_STATE_TYPES = [
    "mamba2", "mamba", "s4", "s5", "s6",
    "gla", "kda", "deltanet", "gated_deltanet",
    "rwkv7", "retnet", "linear_attention",
    "parallel_heads", "moh", "hydra",
    "sliding_window",
]


def _normalize_state_type(value: str) -> str:
    v = (value or "").strip().lower()
    return STATE_TYPE_ALIASES.get(v, v)


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
        parts.append(
            "attn=nsa("
            f"select_top_k={getattr(a, 'nsa_select_top_k', 0) or 0},"
            f"window={getattr(a, 'nsa_window_size', 0) or 0})"
        )
    elif attn == "csa":
        parts.append(
            "attn=csa("
            f"block={getattr(a, 'csa_block_size', 0) or 0},"
            f"top_k={getattr(a, 'csa_top_k_blocks', 0) or 0},"
            f"dim={getattr(a, 'csa_compression_dim', 0) or 0})"
        )
    elif attn == "indexshare":
        parts.append(
            "attn=indexshare("
            f"buckets={getattr(a, 'indexshare_num_buckets', 0) or 0},"
            f"top_k={getattr(a, 'indexshare_top_k_buckets', 0) or 0},"
            f"dim={getattr(a, 'indexshare_index_dim', 0) or 0})"
        )
    elif attn == "msa":
        parts.append(
            "attn=msa("
            f"window={getattr(a, 'msa_window_size', 0) or 0},"
            f"dilated_top_k={getattr(a, 'msa_dilated_top_k', 0) or 0},"
            f"global_top_k={getattr(a, 'msa_global_top_k', 0) or 0})"
        )
    else:
        parts.append(f"attn=full h={a.n_heads} kv={a.n_kv_heads}")
    # FFN family. Wave 19 (P2): on MoE rows, `ffn=<aggregate>` next to dense
    # lines was ambiguous (22016 looked like a dense width while the config's
    # per-expert dim was 2752). Name each number.
    if getattr(a, "moe_style", "dense") != "dense" and a.moe is not None:
        _n_exp = a.moe.get("n_experts")
        _tk = a.moe.get("top_k")
        _edim = a.moe.get("expert_dim")
        _shared = (a.moe.get("shared_expert") or {}).get("ffn_dim") \
            if isinstance(a.moe.get("shared_expert"), dict) else None
        _active_ffn = (_tk or 0) * (_edim or 0) + (_shared or 0)
        moe_desc = (
            f"ffn=moe(expert {_edim}x top{_tk}"
            + (f", shared {_shared}" if _shared else "")
            + f", active≈{_active_ffn}) "
            f"experts={_n_exp}(ep={a.ep_degree})"
        )
        parts.append(moe_desc)
        if a.n_dense_ffn_layers:
            parts.append(f"dense_prefix={a.n_dense_ffn_layers}")
    else:
        parts.append(f"ffn={a.ffn_dim}")
    # Vocab (Wave 19, P1-4): part of the searched identity when swept.
    if int(getattr(a, "vocab_size", 0) or 0) not in (0, 32000):
        parts.append(f"vocab={a.vocab_size}")
    # Local:global interleave (Wave 19, L1: was silently omitted — the log
    # line said `attn=full` for a 28/37-local pick).
    _n_local = int(getattr(a, "n_local_attn_layers", 0) or 0)
    _lg_win = int(getattr(a, "swa_window", 0) or 0)
    if _n_local > 0 and _lg_win > 0:
        parts.append(f"local={_n_local}/{a.n_layers}@w{_lg_win}")
    # State / hybrid.
    if a.n_state_layers > 0:
        parts.append(
            f"state={a.n_state_layers}/{a.n_layers} "
            f"d_state={a.derived_d_state} placement={a.placement_strategy}"
        )
    # YOCO (Wave 28: was silently omitted — a --yoco run printed a line
    # indistinguishable from plain full attention while the emitted
    # config carried architecture.yoco and the KV memory was halved).
    _yoco_k = int(getattr(a, "yoco_n_self_attn_layers", 0) or 0)
    if _yoco_k > 0:
        _yoco_pat = str(getattr(a, "yoco_share_pattern", "single_source"))
        parts.append(f"yoco(self_kv={_yoco_k},{_yoco_pat})")
    # MTP.
    if a.mtp_n_predict_depths and a.mtp_n_predict_depths > 0:
        parts.append(f"mtp={a.mtp_n_predict_depths}")
    # RoPE scaling.
    if a.rope_scaling_method and a.rope_scaling_method != "none":
        parts.append(f"rope={a.rope_scaling_method}")
    # Precision.
    parts.append(f"prec={a.ffn_precision} kv_bits={a.kv_cache_bits}")
    return "[arch-compiler] Optimal: " + " ".join(parts)


# =============================================================================
# Family-comparison rollup (Wave 2b Step 2b.2, Jun 2026)
# =============================================================================
#
# Lightweight rollup used by the CLI to print a per-arch-family comparison
# after the optimal config. Wave 2a will move this into optimizer.py as
# `family_rollup(pareto_4d)` and embed it in the cell output JSON. Until
# then the CLI computes it inline so 2b is independently shippable.

def _arch_mode_of(arch) -> str:
    """Determine arch_mode from arch flags. Matches the labelling used in
    v1-web/compiler-data.json so consumers see consistent names.

    Wave 18a: routes through the canonical ArchitectureSignature so this
    CLI can no longer disagree with optimizer / report / generator about
    what family a candidate belongs to. Falls back to the pre-Wave-18a
    inline classifier if the arch is missing minimal shape fields (test
    fixtures with partial mocks)."""
    try:
        from ac.architecture import architecture_signature
        return architecture_signature(arch).legacy_family
    except (ValueError, ImportError):
        has_moe = (getattr(arch, "moe_style", "dense") != "dense") and (
            getattr(arch, "moe", None) is not None
        )
        has_state = int(getattr(arch, "n_state_layers", 0) or 0) > 0
        if has_moe and has_state:
            return "moe_hybrid"
        if has_moe:
            return "moe"
        if has_state:
            return "hybrid"
        return "dense"


def _non_moe_family_label(arch) -> str:
    """Human-readable family label for a candidate with dense FFNs."""
    mode = _arch_mode_of(arch)
    if mode == "hybrid":
        n_attn = int(getattr(arch, "n_attention_layers", 0) or 0)
        return "state-attention hybrid" if n_attn > 0 else "state-space model"
    return "dense model"


def _display_family_label(arch, mode: Optional[str] = None) -> str:
    """Compact factorized label for the family comparison table."""
    mode = mode or _arch_mode_of(arch)
    label = {
        "dense": "dense",
        "hybrid": "hybrid",
        "moe": "MoE",
        "moe_hybrid": "MoE-hybrid",
    }.get(mode, mode)
    attention = str(getattr(arch, "attention_type", "full") or "full")
    if attention != "full":
        label += f"+{attention.upper()}"
    if int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0) > 0:
        label += "+YOCO"
    if int(getattr(arch, "n_local_attn_layers", 0) or 0) > 0:
        label += "+local/global"
    return label


def _state_type_of(arch):
    if int(getattr(arch, "n_state_layers", 0) or 0) <= 0:
        return None
    # The arch carries a state mechanism family in `state_mechanism` or
    # `state_layer_type`; fall back to a generic label.
    return (getattr(arch, "state_mechanism", None)
            or getattr(arch, "state_layer_type", None)
            or "mamba2")


def _rollup_families(result, opt) -> list:
    """Build a loss-sorted families list from result.all_evaluated.

    Picks the loss-minimum feasible candidate per arch_mode and annotates
    each row with the loss/TBT deltas vs the family-0 winner. Returns []
    when no feasible candidate exists.

    Wave 18h: each row now says explicitly whether it IS the picked config
    (`is_selected`). Previously the family table printed the per-family
    best-LOSS candidate while the `Optimal:`/`Loss*` lines printed the
    PICKED candidate — two different configs, three lines apart, with no
    labeling. When the picked config is not any family's best-loss row, an
    extra `picked` row is appended so both numbers are on screen with names.
    """
    def _row_for(ev, mode, is_selected, *, family_label=None):
        # Wave 26 fix #2: `arch_mode` is used both as a display label AND as
        # a bucket key for the cross-family caveat gate. When the picked
        # candidate is not any family's best-loss row we append a synthetic
        # row with `arch_mode="picked"` so the gate doesn't double-count
        # the family; but the reader still needs to see WHICH family the
        # pick sits in. `family_label` carries the printable family name
        # separately from the gate key. Defaults to `mode` for real
        # per-family rows so callers that don't set it keep the pre-fix
        # rendering.
        t = ev.throughput
        return {
            "arch_mode": mode,
            "family_label": family_label if family_label is not None else mode,
            "state_type": _state_type_of(ev.arch),
            "loss": float(ev.predicted_loss),
            "tbt_ms": float(ev.serving_tbt_ms),
            "ttft_ms": float(getattr(t, "prefill_time_ms", 0.0) or 0.0),
            "mem_gb": float(ev.memory_per_gpu_gb),
            "train_tps": int(ev.training_tps),
            "hbm_spill_gb": float(getattr(t, "hbm_spill_gb", 0.0) or 0.0),
            "spill_tier": getattr(t, "spill_tier", "fits"),
            "d_model": getattr(ev.arch, "d_model", None),
            "n_layers": getattr(ev.arch, "n_layers", None),
            "kv_bits": getattr(ev.arch, "kv_cache_bits", None),
            "is_selected": bool(is_selected),
        }

    def _same(a, b):
        return a is b or _same_candidate(a, b)

    eligible = [
        ev for ev in getattr(result, "all_evaluated", [])
        if getattr(ev, "meets_constraints", True)
    ]
    try:
        try:
            from .optimizer import _prefer_training_fit
        except ImportError:
            from optimizer import _prefer_training_fit
        eligible = _prefer_training_fit(eligible, result.hardware)
    except Exception:
        pass
    best_per_family = {}
    for ev in eligible:
        mode = _arch_mode_of(ev.arch)
        display_key = (
            mode,
            str(getattr(ev.arch, "attention_type", "full") or "full"),
            int(getattr(ev.arch, "yoco_n_self_attn_layers", 0) or 0) > 0,
            int(getattr(ev.arch, "n_local_attn_layers", 0) or 0) > 0,
        )
        prev = best_per_family.get(display_key)
        if prev is None or ev.predicted_loss < prev.predicted_loss:
            best_per_family[display_key] = ev
    if not best_per_family:
        return []
    rows = []
    opt_in_rows = False
    for display_key, ev in best_per_family.items():
        mode = display_key[0]
        is_sel = opt is not None and _same(ev, opt)
        opt_in_rows = opt_in_rows or is_sel
        rows.append(_row_for(
            ev, mode, is_sel,
            family_label=_display_family_label(ev.arch, mode)))
    if opt is not None and not opt_in_rows:
        # Preserve the picked config's real family in `family_label` so the
        # renderer can print "dense ←picked" instead of the anonymous
        # "picked". `arch_mode="picked"` is kept so the cross-family caveat
        # gate doesn't count the pick as a second family. See Wave 26 #2.
        rows.append(_row_for(
            opt, "picked", True,
            family_label=_display_family_label(opt.arch),
        ))
    rows.sort(key=lambda r: r["loss"])
    base_loss = rows[0]["loss"]
    base_tbt = rows[0]["tbt_ms"] or 1e-9
    for r in rows:
        r["loss_delta_pct"] = 100.0 * (r["loss"] - base_loss) / base_loss if base_loss else 0.0
        r["tbt_delta_pct"] = 100.0 * (r["tbt_ms"] - base_tbt) / base_tbt if base_tbt else 0.0
    return rows


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


def parse_vocab_options(value: str):
    """Parse --vocab-options: a comma int list, or 'none' to pin the
    vocabulary at --vocab-size (Wave 20, feedback #3)."""
    if str(value).strip().lower() in ("none", "off", "pin"):
        return "none"
    return parse_int_list(value)


# Wave 20 (feedback #3): the vocab sweep is ON by default. The previous
# opt-in default silently pinned every greenfield pick at vocab=32000 —
# off current frontier practice (Llama-3/Qwen3 ~128-152k, Gemma 256k) and
# inconsistent with the tool's own prior, which finds 128k interior at 7B.
# The ladder is lattice-friendly and deliberately short to bound the
# generator-pass multiplier; 256k joins at ≥30B where its embedding cost
# amortizes.
_DEFAULT_VOCAB_LADDER = [32000, 65536, 128256]
_DEFAULT_VOCAB_LADDER_LARGE = [32000, 65536, 128256, 256000]
_VOCAB_LADDER_LARGE_PARAMS_B = 30.0


def resolve_vocab_options(args) -> "Optional[List[int]]":
    """Resolve the vocab search axis from CLI args (Wave 20, feedback #3).

    Priority:
      1. --vocab-options none            -> pinned at --vocab-size
      2. --vocab-options a,b,c           -> explicit ladder
      3. (default)                       -> auto ladder including
                                            --vocab-size, scaled by --params
    """
    vo = getattr(args, "vocab_options", None)
    if vo == "none":
        return None
    if vo:
        return list(vo)
    params_b = float(getattr(args, "params", 0.0) or 0.0)
    ladder = (_DEFAULT_VOCAB_LADDER_LARGE
              if params_b >= _VOCAB_LADDER_LARGE_PARAMS_B
              else _DEFAULT_VOCAB_LADDER)
    pinned = int(getattr(args, "vocab_size", 32000) or 32000)
    return sorted(set(ladder) | {pinned})


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


def _route_out_dir(args) -> None:
    """Wave 21: `--out DIR` routes greenfield outputs into DIR when
    --output-config is left at (or near) its default. Shared by the real
    run (`main`) and the `config show` preview so the two can't
    disagree. Idempotent: after the first call the config path is no
    longer bare, so a second call is a no-op.
    """
    if args.out and not args.baseline_config:
        if os.path.abspath(args.output_config) == os.path.abspath("arch.json"):
            args.output_config = os.path.join(args.out, "arch.json")
        elif os.path.dirname(args.output_config) in ("", "."):
            args.output_config = os.path.join(
                args.out, os.path.basename(args.output_config))


def _modifier_output_paths(args) -> Dict[str, str]:
    """Preview the fixed-name outputs modifier mode writes into its out
    dir (see `_run_modifier_mode`). The default dir needs the baseline's
    metadata.model_name; fall back to the config file stem when the
    file can't be read (the preview must not crash on a bad path — the
    real run will surface that error)."""
    out_dir = args.out
    if not out_dir:
        name = None
        try:
            with open(args.baseline_config, "r") as f:
                meta = (json.load(f).get("metadata") or {})
            name = meta.get("model_name")
        except Exception:
            pass
        if not name:
            name = os.path.splitext(
                os.path.basename(str(args.baseline_config)))[0]
        out_dir = os.path.join("outputs", f"{name}_modifier")
    return {
        "mode": "modifier (--baseline-config set; fixed names in out dir)",
        "config": os.path.join(out_dir, "config.json"),
        "pareto": os.path.join(out_dir, "pareto.csv"),
        "baseline_delta": os.path.join(out_dir, "baseline_delta.md"),
        "shadow_prices": os.path.join(out_dir, "shadow_prices.md"),
        "justification": os.path.join(out_dir, "justification.md"),
        "assumptions": os.path.join(out_dir, "assumptions.md"),
        "model_card": os.path.join(out_dir, "model_card.md"),
    }


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

    Wave 22: applies the `--out DIR` routing itself (previously only
    `main()` did, so `ac-compile config show --out DIR` previewed cwd
    paths while the real run wrote into DIR), and previews the modifier
    output set when --baseline-config is given (modifier mode ignores
    --output-config entirely).
    """
    if getattr(args, "baseline_config", None):
        return _modifier_output_paths(args)
    _route_out_dir(args)
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
    p.add_argument("--hardware", "--hw", dest="hardware", required=True,
                   type=_normalize_hardware,
                   choices=VALID_HARDWARE,
                   help="Target hardware platform")
    p.add_argument("--params", default=None, type=parse_billions,
                   help="Target parameter count in billions (e.g., 7 or 7B)")
    p.add_argument("--tokens", default=20.0, type=parse_trillions,
                   help="Training token exposures in trillions (default: 20T)")
    p.add_argument("--unique-tokens", default=None, type=parse_trillions,
                   help=("Estimated unique training tokens in trillions. "
                         "Omit to assume all token exposures are unique."))
    p.add_argument("--pretrain-context", type=parse_positive_int, default=8192,
                   help="Pretraining context used by the quality model (default: 8192)")
    p.add_argument(
        "--quality-model",
        choices=["effective_capacity_v2", "legacy_residual_v1"],
        default="effective_capacity_v2",
        help=("Quality-model implementation. effective_capacity_v2 is the "
              "default; legacy_residual_v1 preserves the previous model."),
    )

    # Architecture constraints
    p.add_argument("--context", type=parse_positive_int, default=8192,
                   help="Context length (default: 8192)")
    p.add_argument("--vocab-size", type=parse_positive_int, default=32000,
                   help="Vocabulary size (default: 32000)")
    p.add_argument("--vocab-options", type=parse_vocab_options, default=None,
                   help=("Comma-list of vocabulary sizes to sweep as a search "
                         "axis (e.g. \"32000,128256,256000\"), or 'none' to "
                         "pin the vocabulary at --vocab-size. The quality "
                         "model applies a low-confidence undersized-vocab "
                         "prior so the axis has an interior optimum; "
                         "cross-vocab loss comparisons are per-token and "
                         "should be read as a structural prior, not a "
                         "measurement. Default (Wave 20): an automatic "
                         "ladder {32000, 65536, 128256} (+256000 at ≥30B) "
                         "including --vocab-size — the sweep is ON unless "
                         "you pass 'none'."))
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

    # Serving constraints.
    # v1-fix Wave 2a Step 2a.3 (Jun 2026): --serving-tbt and --serving-ttft
    # are now SOFT budgets — they no longer remove candidates from the
    # feasible set. The compiler reports continuous tbt_ms / ttft_ms /
    # hbm_spill_gb on each Pareto point and on each family in `families[]`;
    # the user picks the loss-vs-serving knee. Passing these flags emits a
    # deprecation note in the report but does not change the optimizer
    # behavior (other than tagging the warning).
    p.add_argument("--serving-tbt", type=parse_positive_float, default=None,
                   help="(soft) Time-between-tokens budget in ms — reported as a warning, no longer a feasibility cut.")
    p.add_argument("--serving-ttft", type=parse_positive_float, default=None,
                   help="(soft) Time-to-first-token budget in ms — reported as a warning, no longer a feasibility cut.")
    p.add_argument("--serving-batch", type=parse_positive_int, default=32,
                   help="Serving batch size (default: 32). Still affects decode latency and KV memory.")
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
    p.add_argument("--strict-quality", action="store_true", default=False,
                   help=("Rank/pick by point-estimate loss with NO "
                         "uncertainty noise-band bucketing. By default the "
                         "picker treats candidates within about a quarter of "
                         "the model's uncertainty as quality-equivalent and "
                         "tiebreaks them on memory/TBT/TPS, which can pick a "
                         "rank-1 a few percent worse in predicted loss than "
                         "the best-loss contender. This flag makes rank-1 "
                         "the argmin-loss candidate."))

    # Parallelism
    p.add_argument("--tp", type=parse_positive_int, default=8,
                   help="Tensor parallelism degree (default: 8)")
    p.add_argument("--tp-options", default=None,
                   help="Comma-separated TP search options, including cross-node degrees (e.g., 8,16,32)")
    p.add_argument("--pp", type=parse_positive_int, default=1,
                   help="Pipeline parallelism degree (default: 1)")
    p.add_argument("--pp-options", default=None,
                   help="Comma-separated pipeline-parallel search options (e.g., 1,2,4)")
    p.add_argument("--dp", type=parse_positive_int, default=8,
                   help="Data parallelism degree (default: 8)")
    p.add_argument(
        "--training-cluster-gpus",
        type=parse_positive_int,
        default=None,
        help=(
            "Minimum training-cluster size. When set, AC derives DP for "
            "each TP/PP/CP candidate and rounds it up to a legal EP "
            "multiple; this overrides the scalar --dp for greenfield "
            "evaluation."
        ),
    )
    p.add_argument("--training-micro-batch", type=parse_positive_int, default=None,
                   help="Training micro-batch per TP×PP×CP replica (default: 8)")
    p.add_argument("--pipeline-microbatches", type=parse_positive_int, default=1,
                   help="Pipeline microbatches / gradient-accumulation slots used for PP bubble modeling (default: 1)")
    p.add_argument("--num-gpus", type=parse_positive_int, default=None,
                   help="Forward-compat shim ignored for modeling. If passed and it disagrees "
                        "with --training-cluster-gpus or TP*PP*DP, AC prints a WARNING.")

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
    # Gate-2 Task E: USD cost block (pure-add; default off keeps output byte-identical)
    p.add_argument("--cost-usd", action="store_true",
                   help="Attach a cost_estimate_usd block (training_total / "
                        "serving_per_1m_tokens / annual_serving_at_load) to the "
                        "emitted config. Pure-add; list prices from "
                        "ac/pricing_specs/. Default off.")
    p.add_argument("--price-tier", default="on_demand",
                   choices=["on_demand", "reserved_1y", "spot"],
                   help="Price tier for --cost-usd (default: on_demand). "
                        "Falls back to on_demand, then reference_estimate, "
                        "when a tier is not published for the target.")

    # Baseline modifier mode
    p.add_argument("--baseline-config", default=None,
                   help="Compiler-schema baseline config to locally modify")
    p.add_argument("--out", default=None,
                   help="Output directory. Modifier mode writes its report "
                        "set here; greenfield mode routes arch.json and all "
                        "sibling outputs (justification, pareto, shadow "
                        "prices) into this directory (Wave 21).")
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
                   type=_normalize_state_type,
                   choices=VALID_STATE_TYPES,
                   metavar="STATE",
                   help="State mechanism family (default: mamba2). "
                        "Aliases: swa/local_recurrent -> sliding_window, "
                        "gated_delta -> gated_deltanet, delta_net -> deltanet.")
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
    p.add_argument("--moe-granularity", default=None,
                   help="Comma-separated expert-granularity targets "
                        "(expert_dim as a fraction of the dense ffn_dim / "
                        "top_k; e.g. '1.0,0.25'). Lower = finer-grained, "
                        "DeepSeek-V3-style experts. Reference granularity "
                        "8.0 in the effective-capacity model.")
    p.add_argument("--ep-topology", default=None,
                   choices=[None, "single_axis", "cross_axis"],
                   help="Expert-parallel all-to-all topology (default: "
                        "single_axis).")

    # MLA / MTP / CP / RoPE scaling
    p.add_argument("--allow-mla", action="store_true",
                   help="Enable MLA candidates (DeepSeek-V2/V3 style).")
    p.add_argument("--mla-kv-latent", default=None,
                   help="Comma-separated MLA c_kv options (default: 512).")
    p.add_argument("--mla-q-latent", default=None,
                   help="Comma-separated MLA c_q options (default: 1536).")
    p.add_argument("--mla-rope-head-dim", type=parse_positive_int, default=None,
                   help="MLA decoupled RoPE head dim (default: 64).")
    p.add_argument("--mla-nope-head-dim", type=parse_positive_int, default=None,
                   help="MLA NoPE head dim (default: 128).")
    # Wave 18g: per-layer attention heterogeneity — local:global interleave.
    p.add_argument("--allow-local-global", action="store_true",
                   help="Enable local:global attention interleave candidates "
                        "(GPT-OSS / Gemma-2 / Llama-4 pattern): a fraction of "
                        "layers use sliding-window attention, the rest stay "
                        "global (full/GQA, or MLA when --allow-mla is also "
                        "set). Sweeps --local-global-ratios x --local-windows.")
    p.add_argument("--local-windows", default=None,
                   help="Comma-separated sliding-window sizes for the local "
                        "layers (default: 1024,4096).")
    p.add_argument("--local-global-ratios", default=None,
                   help="Comma-separated local:global layer ratios "
                        "(default: 1:1,3:1,7:1). '3:1' = 3 local layers per "
                        "global layer.")
    p.add_argument("--allow-mtp", action="store_true",
                   help="Enable Multi-Token Prediction depth sweep.")
    p.add_argument("--mtp-depths", default=None,
                   help="Comma-separated MTP depths (e.g., 0,1,2).")
    p.add_argument("--mtp-depth-n-layers", type=parse_positive_int, default=None,
                   help="Transformer layers per MTP depth (default: 1).")
    p.add_argument("--mtp-train-loss-weight", type=float, default=None,
                   help="MTP auxiliary loss weight (default: 0.3).")
    p.add_argument("--cp", type=parse_positive_int, default=1,
                   help="Context parallelism degree (default: 1).")
    p.add_argument("--cp-method", default="ring", choices=["ring", "ulysses"],
                   help="CP method.")
    p.add_argument("--cp-options", default=None,
                   help="Comma-separated CP degrees to sweep.")
    # Wave 2b Step 2b.2 (Jun 2026): per-family comparison output.
    p.add_argument("--no-family-view", action="store_true",
                   help="Suppress the per-arch-family loss/TBT/TTFT/mem "
                        "comparison printed after the optimal config. "
                        "The comparison surfaces the loss-vs-serving "
                        "trade-off across {dense, hybrid, MoE, MoE-hybrid}.")
    p.add_argument("--nsa", action="store_true",
                   help="Evaluate and require NSA (Native Sparse Attention) on every candidate.")
    p.add_argument("--nsa-compress-block-size", type=parse_positive_int, default=None)
    p.add_argument("--nsa-compress-block-stride", type=parse_positive_int, default=None)
    p.add_argument("--nsa-select-block-size", type=parse_positive_int, default=None)
    p.add_argument("--nsa-select-top-k", type=parse_positive_int, default=None)
    p.add_argument("--nsa-window-size", type=parse_positive_int, default=None)
    p.add_argument("--yoco", action="store_true",
                   help="Evaluate and require YOCO KV sharing on every candidate.")
    p.add_argument("--yoco-n-self-attn-layers", type=parse_positive_int, default=None)
    p.add_argument("--yoco-share-pattern", default=None,
                   choices=[None, "single_source"],
                   help="YOCO's calibrated single shared-cache topology.")
    # Wave 32: compressed / indexer attention families (Wave 9 evaluator
    # support existed but was reachable only via the Python API — no CLI
    # flag could ever select them).
    p.add_argument("--allow-csa", action="store_true",
                   help="Enable Compressed Sparse Attention candidates "
                        "(block-compressed KV with top-k block selection).")
    p.add_argument("--csa-block-sizes", default=None,
                   help="Comma-separated CSA block sizes (default: 64,128).")
    p.add_argument("--csa-top-k-blocks", default=None,
                   help="Comma-separated CSA top-k block counts.")
    p.add_argument("--csa-compression-dim", type=parse_positive_int, default=None,
                   help="CSA compression dim (default: 64).")
    p.add_argument("--allow-indexshare", action="store_true",
                   help="Enable index-sharing attention candidates "
                        "(DSA-style bucketed lightning indexer; shared "
                        "top-k bucket selection across heads).")
    p.add_argument("--indexshare-buckets", default=None,
                   help="Comma-separated bucket counts (default: 64,128).")
    p.add_argument("--indexshare-top-k", default=None,
                   help="Comma-separated top-k bucket counts (default: 4,8).")
    p.add_argument("--indexshare-index-dim", type=parse_positive_int, default=None,
                   help="Indexer head dim (default: 64).")
    p.add_argument("--allow-msa", action="store_true",
                   help="Enable multi-scale attention candidates "
                        "(local window + dilated top-k + global top-k).")
    p.add_argument("--msa-windows", default=None,
                   help="Comma-separated MSA local windows (default: 512,1024).")
    p.add_argument("--msa-dilated-top-k", default=None,
                   help="Comma-separated MSA dilated top-k options.")
    p.add_argument("--msa-global-top-k", default=None,
                   help="Comma-separated MSA global top-k options.")
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
                   help="Print greenfield evaluation progress every N candidates "
                        "(default: auto — every 1000 candidates on large "
                        "searches; silenced by --quiet)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    p.add_argument("--max-full-evaluations", type=parse_positive_int, default=None,
                   help="Two-stage evaluation: cap the number of full "
                        "evaluations after cheap ranking (speed knob).")
    p.add_argument("--local-refine-budget", type=int, default=None,
                   help="Wave 34: extra full evaluations spent on lattice "
                        "neighbors of the per-class Pareto leaders when "
                        "--max-candidates dropped candidates (default 96; "
                        "0 disables).")
    p.add_argument("--allow-quality-sentinel", action="store_true",
                   help="Return a best-effort answer marked UNCOVERED "
                        "instead of failing when every candidate is outside "
                        "the quality model's covered envelope (extreme "
                        "params x context corners).")

    p.epilog = (
        "logical groups for --help-group: hardware, workload, serving, "
        "parallelism, precision, state, moe, mla, mtp, rope, nsa, yoco, "
        "compressed, modifier, outputs, recipe."
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
            resolve_args=lambda args: {
                # Show the axis the search will actually enumerate. The raw
                # argparse namespace carries None for the default auto ladder
                # and "none" for a pinned vocabulary, which made config show
                # claim vocab=32k while the real run swept/picked 128k.
                "vocab_options": (
                    resolve_vocab_options(args)
                    or [int(args.vocab_size)]
                )
            },
        )
    args = parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    # Wave 28: validate the calibration environment BEFORE searching. A
    # typo'd AC_QUALITY_DEFAULTS / AC_HARDWARE_SPEC_DIR used to be
    # swallowed by the evaluator's per-candidate exception handler and
    # re-surface as "No feasible architecture found ... relax serving
    # constraints" — a misdiagnosis contradicting the README contract
    # ("invalid pack path or contents: compilation fails").
    try:
        from .quality_model import validate_calibration_environment
    except ImportError:
        from quality_model import validate_calibration_environment
    try:
        validate_calibration_environment(getattr(args, "hardware", None))
    except Exception as e:
        print(
            f"ERROR: invalid calibration environment: {e}\n"
            "Fix or unset AC_QUALITY_DEFAULTS / AC_HARDWARE_SPEC_DIR and "
            "re-run. AC does not fall back to different constants.",
            file=sys.stderr,
        )
        return 2

    # Wave 28: modifier mode writes fixed file names into its --out
    # directory (see _modifier_output_paths). Explicit --output-* flags
    # used to be dropped SILENTLY — a user passing
    # `--baseline-config ... --output-config out/foo.json` got their
    # outputs in outputs/<model>_modifier/ with no hint why.
    if getattr(args, "baseline_config", None):
        _ignored = []
        if os.path.abspath(str(args.output_config)) != os.path.abspath("arch.json"):
            _ignored.append("--output-config")
        for _flag, _val in (
            ("--output-justification", args.output_justification),
            ("--output-pareto", args.output_pareto),
            ("--output-shadow-prices", args.output_shadow_prices),
            ("--output-assumptions", args.output_assumptions),
            ("--output-model-card", args.output_model_card),
        ):
            if _val:
                _ignored.append(_flag)
        if _ignored:
            log(
                "WARNING: modifier mode (--baseline-config) writes fixed "
                "file names (config.json, pareto.csv, baseline_delta.md, "
                "...) into its output directory; "
                + ", ".join(_ignored)
                + " will be ignored. Use --out DIR to choose the "
                "directory (default: outputs/<model>_modifier/)."
            )

        _ignored_parallelism = [
            flag for flag, value in (
                ("--pp-options", getattr(args, "pp_options", None)),
                ("--cp-options", getattr(args, "cp_options", None)),
                ("--ep-options", getattr(args, "ep_options", None)),
                ("--training-cluster-gpus", getattr(args, "training_cluster_gpus", None)),
            )
            if value is not None
        ]
        if _ignored_parallelism:
            log(
                "WARNING: modifier mode searches local shape, component "
                "precision, KV dtype, and TP only; "
                + ", ".join(_ignored_parallelism)
                + " do not change its candidate grid. Use `ac-delta-eval "
                "--apply change_parallelism` for PP/CP/EP topology "
                "experiments."
            )

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
    # Wave 21: `--out DIR` works in greenfield mode too (it previously was
    # silently ignored unless --baseline-config was given, which stranded
    # outputs in the cwd). Wave 22: the routing lives in _route_out_dir /
    # infer_output_paths so `config show` previews the same paths the
    # real run writes.
    infer_output_paths(args)

    if args.baseline_config:
        return _run_modifier_mode(args, log)

    if args.params is None:
        print("ERROR: --params is required when --baseline-config is not supplied.",
              file=sys.stderr)
        return 2

    # Wave 19 (P0-2): a prompt cannot exceed the context window.
    if args.prompt_len is not None and args.context is not None \
            and int(args.prompt_len) > int(args.context):
        print(
            f"ERROR: --prompt-len ({args.prompt_len}) exceeds --context "
            f"({args.context}). Pass a consistent pair.",
            file=sys.stderr,
        )
        return 2

    # Fix #3: warn on unusual parallelism degrees. H100/B200 nodes have 8
    # GPUs and TPU pods are powers of 2; --tp / --pp / --dp values that are
    # neither 1 nor a power of 2 are almost always a typo (e.g. `--tp 3`
    # meant `--tp 8`). We don't reject — there's no hard rule — but we surface
    # the warning so it's not silent.
    def _is_pow2(n: int) -> bool:
        return n >= 1 and (n & (n - 1)) == 0
    parallelism_values = [("--tp", args.tp), ("--pp", args.pp)]
    if getattr(args, "training_cluster_gpus", None) is None \
            or getattr(args, "baseline_config", None):
        parallelism_values.append(("--dp", args.dp))
    for label, val in parallelism_values:
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
        cluster_floor = getattr(args, "training_cluster_gpus", None)
        implied = (
            int(cluster_floor)
            if cluster_floor is not None and not getattr(args, "baseline_config", None)
            else (args.tp or 1) * (args.pp or 1) * (args.dp or 1)
        )
        if int(args.num_gpus) != implied:
            source = (
                f"--training-cluster-gpus = {implied}"
                if cluster_floor is not None and not getattr(args, "baseline_config", None)
                else (
                    f"--tp * --pp * --dp = {args.tp}*{args.pp}*{args.dp} "
                    f"= {implied}"
                )
            )
            print(
                f"WARNING: --num-gpus={args.num_gpus} does not match "
                f"{source}. --num-gpus is accepted for command-line "
                "compatibility only; update the modeled parallelism or "
                "cluster-floor flags if you want a different total.",
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
        tp_options            = _split_int(getattr(args, "tp_options", None), "--tp-options")
        pp_options            = _split_int(getattr(args, "pp_options", None), "--pp-options")
        rope_methods          = _split_str(getattr(args, "rope_scaling_methods", None))
        # Wave 32: compressed / indexer attention option lists
        csa_block_size_options    = _split_int(getattr(args, "csa_block_sizes", None), "--csa-block-sizes")
        csa_top_k_options         = _split_int(getattr(args, "csa_top_k_blocks", None), "--csa-top-k-blocks")
        indexshare_bucket_options = _split_int(getattr(args, "indexshare_buckets", None), "--indexshare-buckets")
        indexshare_top_k_options  = _split_int(getattr(args, "indexshare_top_k", None), "--indexshare-top-k")
        msa_window_options        = _split_int(getattr(args, "msa_windows", None), "--msa-windows")
        msa_dilated_top_k_options = _split_int(getattr(args, "msa_dilated_top_k", None), "--msa-dilated-top-k")
        msa_global_top_k_options  = _split_int(getattr(args, "msa_global_top_k", None), "--msa-global-top-k")
        moe_granularity_targets   = (
            [float(v) for v in str(args.moe_granularity).split(",") if v.strip()]
            if getattr(args, "moe_granularity", None) else None)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Build constraints for greenfield compiler mode.
    try:
        constraints = DeploymentConstraints(
            target_params_b=args.params,
            param_tolerance=args.param_tolerance,
            training_tokens=int(args.tokens * 1e12),
            unique_training_tokens=(
                int(args.unique_tokens * 1e12)
                if args.unique_tokens is not None else None
            ),
            pretraining_context_length=args.pretrain_context,
            quality_model_version=args.quality_model,
            context_length=args.context,
            serving_tbt_ms=args.serving_tbt,
            serving_ttft_ms=args.serving_ttft,
            serving_batch=args.serving_batch,
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            training_cluster_gpus=getattr(args, "training_cluster_gpus", None),
            tp_options=tp_options,
            pp_options=pp_options,
            vocab_size=args.vocab_size,
            vocab_options=resolve_vocab_options(args),
            strict_quality=getattr(args, "strict_quality", False),
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
            # Wave 18g: local:global attention interleave
            allow_local_global=getattr(args, "allow_local_global", False),
            local_window_options=(
                [int(v) for v in str(args.local_windows).split(",") if v.strip()]
                if getattr(args, "local_windows", None) else None),
            local_global_ratio_options=(
                [v.strip() for v in str(args.local_global_ratios).split(",") if v.strip()]
                if getattr(args, "local_global_ratios", None) else None),
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
            training_micro_batch=getattr(args, "training_micro_batch", None),
            pipeline_microbatches=getattr(args, "pipeline_microbatches", 1),
            # NSA / YOCO are evaluated transforms, never post-search stamps.
            force_nsa=bool(getattr(args, "nsa", False)),
            nsa_compress_block_size=int(getattr(args, "nsa_compress_block_size", None) or 64),
            nsa_compress_block_stride=int(getattr(args, "nsa_compress_block_stride", None) or 16),
            nsa_select_block_size=int(getattr(args, "nsa_select_block_size", None) or 64),
            nsa_select_top_k=int(getattr(args, "nsa_select_top_k", None) or 16),
            nsa_window_size=int(getattr(args, "nsa_window_size", None) or 512),
            force_yoco=bool(getattr(args, "yoco", False)),
            yoco_n_self_attn_layers=int(getattr(args, "yoco_n_self_attn_layers", None) or 1),
            yoco_share_pattern=str(getattr(args, "yoco_share_pattern", None) or "single_source"),
            # Wave 32: compressed / indexer attention families
            allow_csa=getattr(args, "allow_csa", False),
            csa_block_size_options=csa_block_size_options,
            csa_top_k_options=csa_top_k_options,
            csa_compression_dim=int(getattr(args, "csa_compression_dim", None) or 64),
            allow_indexshare=getattr(args, "allow_indexshare", False),
            indexshare_num_buckets_options=indexshare_bucket_options,
            indexshare_top_k_options=indexshare_top_k_options,
            indexshare_index_dim=int(getattr(args, "indexshare_index_dim", None) or 64),
            allow_msa=getattr(args, "allow_msa", False),
            msa_window_options=msa_window_options,
            msa_dilated_top_k_options=msa_dilated_top_k_options,
            msa_global_top_k_options=msa_global_top_k_options,
            # Wave 32: MoE granularity + EP topology
            moe_granularity_targets=moe_granularity_targets,
            ep_topology=str(getattr(args, "ep_topology", None) or "single_axis"),
            # Wave 32: MLA / MTP detail knobs
            mla_rope_head_dim=int(getattr(args, "mla_rope_head_dim", None) or 64),
            mla_nope_head_dim=int(getattr(args, "mla_nope_head_dim", None) or 128),
            mtp_depth_n_layers=int(getattr(args, "mtp_depth_n_layers", None) or 1),
            mtp_train_loss_weight=float(
                getattr(args, "mtp_train_loss_weight", None)
                if getattr(args, "mtp_train_loss_weight", None) is not None else 0.3),
            # Wave 32: search ergonomics
            max_full_evaluations=getattr(args, "max_full_evaluations", None),
            local_refine_budget=(
                int(args.local_refine_budget)
                if getattr(args, "local_refine_budget", None) is not None
                else 96),
            allow_quality_sentinel=getattr(args, "allow_quality_sentinel", False),
            # Search ergonomics
            max_candidates=getattr(args, "max_candidates", None),
            # Wave 47: auto-progress. An uncapped greenfield search can run
            # ~10^4 full evaluations (tens of seconds) — before this default,
            # the CLI printed one line and went silent unless the user knew
            # about --progress-every. 0 (unset) now means "auto": every 1000
            # candidates, which small searches never reach. --quiet still
            # disables entirely; an explicit --progress-every N is honored.
            progress_every=(
                0 if args.quiet
                else (getattr(args, "progress_every", 0) or 1000)),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Wave 20 (feedback #3): sleeping warning when the vocab axis is pinned.
    # The sweep is on by default now; if the user pinned it (--vocab-options
    # none, or an explicit single-value list), say what that forgoes instead
    # of silently shipping an off-frontier tokenizer.
    _resolved_vocabs = constraints.vocab_options or [constraints.vocab_size]
    if len(set(_resolved_vocabs)) == 1:
        print(
            f"NOTE: vocabulary pinned at {constraints.vocab_size:,} — the "
            "vocab search axis is disabled. AC's prior finds interior "
            "optima well above 32k at ≥7B (frontier practice: 128-256k). "
            "Drop `--vocab-options none` (or widen the list) to sweep.",
            file=sys.stderr,
        )

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
            try:
                from .optimizer import get_precision_configs_for_hardware
            except ImportError:
                from optimizer import get_precision_configs_for_hardware
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
        try:
            from .throughput_model import warn_if_uncalibrated
        except ImportError:
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
    log(f"[arch-compiler] Search complete: {result.candidates_evaluated} evaluated "
        f"({result.candidates_generated} initial after cap), "
        f"{result.candidates_feasible} feasible, {len(result.pareto_frontier)} Pareto, "
        f"{result.search_time_sec:.1f}s")
    if result.evaluation_failures:
        reasons = sorted(
            result.evaluation_failure_reasons.items(),
            key=lambda item: (-item[1], item[0]),
        )
        preview = "; ".join(
            f"{count}x {reason}" for reason, count in reasons[:3]
        )
        print(
            f"WARNING: {result.evaluation_failures} candidate evaluation(s) "
            f"failed and were excluded: {preview}",
            file=sys.stderr,
        )

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
    # Wave 20 (feedback #2): print the implied MFU next to the training
    # throughput, against the vendor DATASHEET peak for the training
    # precision. Without a stated basis, an fp8 TPS number divided by the
    # bf16 peak reads as ~68% MFU and the model looks flattering; against
    # the correct fp8 peak the same number is ~33%.
    _mfu_txt = ""
    try:
        try:
            from .throughput_model import load_hardware as _load_hw
        except ImportError:
            from throughput_model import load_hardware as _load_hw
        _hw = _load_hw(args.hardware)
        _prec = str(getattr(opt.arch, "ffn_precision", "bf16") or "bf16")
        _n_active = float(
            getattr(opt.arch, "active_params", 0)
            or getattr(opt.arch, "total_params", 0))
        # Wave 22: the training replica is TP × PP × CP — the same
        # convention `optimizer` uses to emit
        # `training_throughput_tokens_per_sec_per_gpu`. This line used
        # tp×pp only (and constraints.tp rather than the candidate's
        # tp_degree), so on a CP=2 run the console printed a per-GPU
        # TPS and MFU exactly 2× the number written into arch.json.
        _tp = int(getattr(opt.arch, "tp_degree", 0) or 0) \
            or int(getattr(constraints, "tp", 1) or 1)
        _pp = max(1, int(getattr(opt.arch, "pp_degree", 1) or 1))
        _cp = max(1, int(getattr(opt.arch, "cp_degree", 1) or 1))
        _gpus_per_replica = max(1, _tp * _pp * _cp)
        _tps_per_gpu = float(opt.training_tps) / _gpus_per_replica
        _peak = _hw.datasheet_peak_flops_s(_prec)
        if _n_active > 0 and _peak > 0 and _tps_per_gpu > 0:
            _mfu = 6.0 * _n_active * _tps_per_gpu / _peak
            _mfu_txt = (f" (≈{_mfu * 100:.0f}% MFU vs {_prec} datasheet "
                        f"peak, {_tps_per_gpu:.0f} tok/s/GPU)")
    except Exception:
        pass
    log(f"[arch-compiler] {_loss_label}={opt.predicted_loss:.4f}{_loss_suffix} "
        f"TPS={opt.training_tps:.0f}{_mfu_txt} TBT={opt.serving_tbt_ms:.1f}ms "
        f"Mem={opt.memory_per_gpu_gb:.1f}GB")

    # Wave 18h: make the picker's quality spend explicit. When the selected
    # candidate is NOT the best-loss candidate on the frontier, say so, say
    # by how much, and say why — the uncertainty-band tiebreak is a
    # deliberate design choice, but a silent one reads as a bug next to a
    # profile named "research_quality".
    try:
        _frontier = list(result.pareto_frontier or [])
        if _frontier:
            _best = min(_frontier, key=lambda ev: ev.predicted_loss)
            _gap_pct = (
                100.0 * (opt.predicted_loss - _best.predicted_loss)
                / max(1e-9, _best.predicted_loss)
            )
            if _gap_pct > 0.1:
                _unc_pct = float(
                    getattr(opt.quality, "uncertainty_total", 0.0) or 0.0
                ) * 100.0
                _b_arch = _best.arch
                _profile = str(getattr(
                    constraints, "objective_profile", "balanced"))
                _quality_first = _profile in ("research_quality", "loss_only")
                if _quality_first:
                    # Wave 19 (P1-4): quality-first profiles use the capped
                    # uncertainty bucket as the primary ordering.
                    if _uncalibrated:
                        _band_src = (
                            "the capped pre-calibration tiebreak band "
                            "(max ~0.5% loss spend; modeled uncertainty is "
                            f"±{_unc_pct:.1f}% but is not trusted for tiebreaks "
                            "until a pack is fitted)"
                        )
                    else:
                        _band_src = (
                            f"the calibrated ±{_unc_pct:.1f}% uncertainty band"
                        )
                    _why = (
                        f"The two are treated as quality-equivalent within "
                        f"{_band_src} and tiebroken on memory/TBT/TPS. "
                        "Use --strict-quality to rank by point-estimate loss."
                    )
                else:
                    # Balanced/latency/cost profiles rank the weighted
                    # multi-objective score first. The quality bucket is only
                    # a secondary tiebreak there, so calling a large gap a
                    # capped quality tie is false and dangerously reassuring.
                    _why = (
                        f"This is an explicit weighted tradeoff from the "
                        f"`{_profile}` objective, not a quality-equivalent "
                        "tie: serving/training/memory gains outweighed the "
                        "profile's loss term. Use --objective-profile "
                        "research_quality for a quality-first capped band, "
                        "or add --strict-quality for exact score/loss ordering."
                    )
                log(
                    "[arch-compiler] Pick rationale: selected loss "
                    f"{opt.predicted_loss:.4f} vs best-loss contender "
                    f"{_best.predicted_loss:.4f} "
                    f"(d={_b_arch.d_model} L={_b_arch.n_layers} "
                    f"kv_bits={getattr(_b_arch, 'kv_cache_bits', '?')}) — "
                    f"+{_gap_pct:.2f}%. {_why}"
                )
                # Wave 20 (feedback #5): cross-reference the plan-ladder
                # run-noise floor. When the quality gap is below what two
                # real paired runs could resolve, the quality ordering is
                # a coin flip — say so here instead of one CLI away.
                try:
                    try:
                        from .quality_model import run_noise_floor_pct
                    except ImportError:
                        from quality_model import run_noise_floor_pct
                    _floor = run_noise_floor_pct()
                    if 0.0 < _gap_pct < 2.0 * _floor:
                        log(
                            "[arch-compiler]   This margin is below the "
                            f"2× run-noise floor ({2.0 * _floor:.2f}%): a "
                            "real paired run could not resolve it. Treat "
                            "the quality ordering between these two as a "
                            "coin flip (plan-ladder reports the same pair "
                            "as UNRESOLVABLE) and decide on "
                            "throughput/memory/cost."
                        )
                except Exception:
                    pass
                # Wave 20 (feedback #1): if the tiebreak crossed
                # architecture families, its TBT/TPS justification leans
                # on serving predictions whose anchor-measured
                # cross-family bias is far larger than the loss bias.
                # Fire the serving-bias caveat that previously existed
                # only for loss.
                try:
                    try:
                        from .decision import (
                            cross_family_serving_floor_pct,
                            family_metric_span_text,
                            _family_of_candidate as _fam_of)
                    except ImportError:
                        from decision import (
                            cross_family_serving_floor_pct,
                            family_metric_span_text,
                            _family_of_candidate as _fam_of)
                    if _fam_of(opt) != _fam_of(_best):
                        _sbar, _ = cross_family_serving_floor_pct(
                            opt, _best, metric="tbt_ms")
                        if _sbar > 0.0:
                            # Wave 25: span read from the live bias table
                            # (was a hard-coded wave-19 snapshot that had
                            # already drifted from the shipped JSON).
                            _span = family_metric_span_text(
                                "moe", "tbt_ms")
                            log(
                                "[arch-compiler]   Cross-family serving "
                                "caveat: this tiebreak compares "
                                f"{_fam_of(opt)} vs {_fam_of(_best)} on "
                                "TBT/TPS, where anchor-measured model "
                                f"bias gives a ±{_sbar:.0f}% floor "
                                f"(MoE TBT anchor errors span {_span}). "
                                "Do not treat the serving margin as "
                                "measured until a serving calibration "
                                "pack is fitted (ac-auto-calibrate)."
                            )
                except Exception:
                    pass
    except Exception:
        pass

    # Wave 2b Step 2b.2 (Jun 2026): per-family loss/TBT/TTFT/mem comparison.
    # Surfaces the loss-vs-serving trade-off across {dense, hybrid, MoE,
    # MoE-hybrid} that the categorical serving regimes used to erase.
    # Default-on; suppress with --no-family-view.
    if not getattr(args, "no_family_view", False):
        try:
            try:
                from .report import render_family_comparison
            except ImportError:
                from report import render_family_comparison
            families = _rollup_families(result, opt)
            if families:
                # Params target and context from the input constraints; fall
                # back to the optimal arch if the constraints aren't on result.
                params_B = float(getattr(args, "params", 0.0) or 0.0)
                ctx = int(getattr(args, "context", 0) or
                          getattr(opt.arch, "seq_len", 0) or 0)
                log(render_family_comparison(families, params_B, ctx).rstrip("\n"))
                # Wave 19 (P1-5): cross-family loss deltas are inside known
                # model bias pre-calibration (anchor-measured dense-vs-MoE
                # bias spread ~10%); say so whenever the table mixes families.
                _fams = {f.get("arch_mode") for f in families
                         if f.get("arch_mode") not in (None, "picked")}
                if len(_fams) > 1 and _uncalibrated:
                    # Wave 25: read the spans from the live bias table
                    # instead of a hard-coded wave-19 snapshot.
                    try:
                        try:
                            from .decision import (
                                family_metric_span_text as _fspan,
                                _load_family_bias as _lfb)
                        except ImportError:
                            from decision import (
                                family_metric_span_text as _fspan,
                                _load_family_bias as _lfb)
                        _moe_span = _fspan("moe", "tbt_ms")
                        _dense_ttft = (
                            (_lfb().get("dense") or {}).get("metrics") or {}
                        ).get("ttft_ms", {}).get("mean_signed_err_pct")
                        _dense_txt = (
                            f"dense TTFT mean {_dense_ttft:+.0f}%"
                            if _dense_ttft is not None
                            else "dense TTFT mean −52%"
                        )
                    except Exception:
                        _moe_span = "−94%…+93%"
                        _dense_txt = "dense TTFT mean −52%"
                    log(
                        "  NOTE: rows span architecture families; "
                        "uncalibrated cross-family loss deltas below the "
                        "anchor-measured family-bias floor (~10%) are model "
                        "bias, not signal. The serving columns (TBT/TTFT/"
                        f"mem) carry LARGER cross-family bias than loss "
                        f"(anchor-measured MoE TBT errors span {_moe_span}; "
                        f"{_dense_txt}): cross-family serving deltas "
                        "below the printed † floor are also model bias. "
                        "Within-family ordering remains usable; fit a "
                        "serving calibration pack (ac-auto-calibrate) "
                        "before trusting cross-family serving-cost "
                        "rankings."
                    )
        except Exception as _fv_err:
            # Family view is informational only; never break the main flow.
            log(f"[arch-compiler] (family view skipped: {_fv_err})")

    # Confidence-envelope robustness. The optimizer computes
    # `robust_to_loss_uncertainty` (in `metadata.predicted.confidence_envelope`)
    # by counting candidates that overlap the picked candidate's loss CI band.
    # When false, the bare "Optimal:" banner above is over-confident: the top
    # several candidates are quality-equivalent within modeled uncertainty and
    # the picker tiebroke them on throughput/memory. Surface that so users do
    # not over-read the rank-1 row.
    contending_family_sidecar_path = None
    try:
        try:
            from .optimizer import (
                compute_confidence_envelope,
                compute_contending_family_full,
            )
        except ImportError:
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

    # Wave 19 (P0-1): the training throughput model assumes EP lays over the
    # DP dimension (every EP rank carries its own microbatch). If the run's
    # DP degree cannot host the picked EP group, the modeled training
    # numbers describe a layout the user cannot actually run.
    #
    # Wave 24 lifted the fix upstream — greenfield enumeration now filters
    # EP divides DP before candidates ever reach the picker (see
    # `_filter_ep_options_by_dp` in optimizer.py). This warning survives as
    # a defensive net for modifier-mode runs whose baseline config declares
    # EP > DP in `parallelism`, which the enumerator preserves rather than
    # rewrites.
    _picked_ep = int(getattr(opt.arch, "ep_degree", 1) or 1)
    _picked_dp = (
        int(getattr(opt.arch, "dp_degree", 0) or 0)
        or int(getattr(args, "dp", 1) or 1)
    )
    if (_picked_ep > 1 and _picked_dp > 1
            and (_picked_ep > _picked_dp or _picked_dp % _picked_ep != 0)):
        _layout_issue = (
            f"exceeds DP={_picked_dp}"
            if _picked_ep > _picked_dp
            else f"does not divide DP={_picked_dp}"
        )
        print(
            f"WARNING: picked EP={_picked_ep} {_layout_issue} "
            "(inherited from baseline `parallelism.expert_parallel`). "
            "Training throughput assumes EP-over-DP (each EP rank processes "
            "its own microbatch), which requires EP to divide DP. Serving "
            "predictions are unaffected (a serving instance spans "
            "TPxPPxCPxEP GPUs).",
            file=sys.stderr,
        )

    # Wave 45: surface HBM oversubscription of the PICKED architecture.
    # Serving spill is priced into TBT/TTFT and training memory overflow is
    # honest arithmetic, but neither was previously visible anywhere except
    # a family-table tag — a 546 GB/GPU pick on a 192 GB part sailed
    # through with zero warnings.
    _t = getattr(opt, "throughput", None)
    if _t is not None:
        try:
            try:
                from .optimizer import _get_hbm_gb
            except ImportError:
                from optimizer import _get_hbm_gb
            _hbm = float(_get_hbm_gb(args.hardware))
        except Exception:
            _hbm = 0.0
        if _hbm > 0:
            if getattr(_t, "spill_tier", "fits") != "fits":
                print(
                    f"WARNING: picked architecture needs "
                    f"{opt.memory_per_gpu_gb:.1f} GB/GPU to serve — exceeds "
                    f"{_hbm:.0f} GB HBM; {getattr(_t, 'hbm_spill_gb', 0.0):.1f} GB "
                    f"spills via {getattr(_t, 'spill_tier', '?')} (cost included "
                    f"in TBT/TTFT). Raise TP/PP, cut serving batch/context, or "
                    f"quantize the KV cache if spill is not deployable in your "
                    f"stack.", file=sys.stderr)
            _train_mem = float(getattr(_t, "training_memory_per_gpu_gb", 0.0) or 0.0)
            if _train_mem > _hbm:
                _any_train_fit = any(
                    float(getattr(
                        getattr(ev, "throughput", None),
                        "training_memory_per_gpu_gb", 0.0) or 0.0) <= _hbm
                    for ev in (getattr(result, "all_evaluated", None) or [])
                    if getattr(ev, "meets_constraints", False)
                )
                _fit_note = (
                    " A fitting candidate existed, so this indicates a "
                    "selection bug; please report the emitted bundle."
                    if _any_train_fit else
                    " No evaluated candidate fits training HBM; this is a "
                    "best-effort, physically incomplete plan."
                )
                print(
                    f"WARNING: picked architecture needs {_train_mem:.1f} GB/GPU "
                    f"to TRAIN at the declared TP×PP×DP — exceeds "
                    f"{_hbm:.0f} GB HBM and there is no training-spill "
                    f"mechanism. Training TPS assumes it fits; add CP/PP, "
                    f"reduce training micro-batch/context, or explicitly "
                    f"model offload before trusting the training numbers."
                    f"{_fit_note}",
                    file=sys.stderr)

    # MoE-fallback warning: the user opted into MoE search but the picked
    # config is dense, which usually means either no MoE candidate beat the
    # dense Pareto front under the chosen objective, or the requested expert
    # topology was filtered out by --max-total-params-b / tile alignment / TP
    # constraints. Make this visible so silent dense fallback is not mistaken
    # for "MoE didn't help."
    if getattr(args, "allow_moe", False) and getattr(opt.arch, "moe_style", "dense") == "dense":
        picked_family = _non_moe_family_label(opt.arch)
        any_moe_in_frontier = any(
            getattr(c.arch, "moe_style", "dense") != "dense"
            for c in (result.pareto_frontier or [])
        )
        # Wave 45: "dominated off the frontier" is NOT "infeasible". The old
        # branch keyed the infeasibility message on frontier membership, so a
        # run where hundreds of MoE candidates were evaluated, met every
        # constraint, and simply lost to dense on all axes told the user to
        # go hunting for --max-total-params-b / tile-alignment problems that
        # don't exist. Distinguish via all_evaluated.
        any_moe_feasible = any(
            getattr(c.arch, "moe_style", "dense") != "dense"
            and getattr(c, "meets_constraints", False)
            for c in (getattr(result, "all_evaluated", None) or [])
        )
        requested_experts = getattr(args, "moe_n_experts", None)
        if any_moe_in_frontier or any_moe_feasible:
            # Wave 34: with the refined two-stage search this is often the
            # honest answer (a well-shaped dense model can beat a coarse
            # MoE at small scale under the data-sufficiency gate), and the
            # old "rerun with --objective-profile=quality" hint fired even
            # when the run WAS quality-profile. State the fact; hint only
            # at knobs that actually change the MoE side.
            where = ("exist on the Pareto frontier but were dominated"
                     if any_moe_in_frontier else
                     "were evaluated and met every constraint but were "
                     "Pareto-dominated")
            msg = (
                "--allow-moe was set, but the picked architecture uses "
                f"dense FFNs ({picked_family}). "
                f"MoE candidates {where} "
                "by the selected non-MoE family under the current objective "
                "profile — often the honest answer at small scale / low "
                "tokens-per-param. Inspect "
                "the CSV; a finer expert axis (--moe-n-experts 64+ with "
                "--moe-granularity 0.25) is the usual way MoE wins back the "
                "loss axis."
            )
        else:
            extra = (
                f" Requested --moe-n-experts={requested_experts}"
                if requested_experts else ""
            )
            msg = (
                "--allow-moe was set, but no MoE candidate was feasible under "
                "the current constraints, so the picked architecture uses "
                f"dense FFNs ({picked_family})."
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

    # Write outputs
    # 1. JSON config
    config = result_to_config(result)
    # Gate-2 Task E: optional USD cost block (pure-add; off by default).
    if getattr(args, "cost_usd", False) and config is not None:
        config = attach_cost_block(
            config,
            args.hardware,
            {
                "training_tokens": constraints.training_tokens,
                "prompt_len": constraints.prompt_len,
                "output_len": constraints.output_len,
                "serving_batch": constraints.serving_batch,
            },
            price_tier=getattr(args, "price_tier", "on_demand"),
        )
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
    tokens_t = args.tokens if args.tokens is not None else 20.0

    try:
        constraints = DeploymentConstraints(
            target_params_b=params_b,
            param_tolerance=args.param_tolerance,
            training_tokens=int(tokens_t * 1e12),
            unique_training_tokens=(
                int(args.unique_tokens * 1e12)
                if args.unique_tokens is not None else None
            ),
            pretraining_context_length=args.pretrain_context,
            quality_model_version=args.quality_model,
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
        try:
            from .throughput_model import warn_if_uncalibrated
        except ImportError:
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
    width_label = (
        f"dense_ffn={c.ffn_dim} expert_dim={c.moe.get('expert_dim')}"
        if c.moe else f"ffn={c.ffn_dim}"
    )
    log(f"[arch-compiler] Config: d={c.d_model} L={c.n_layers} h={c.n_heads} "
        f"kv={c.n_kv_heads} {width_label} ffn_prec={c.ffn_precision} "
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

    # Gate-2 Task E: optional USD cost block (pure-add; off by default).
    modifier_config = modifier_result_to_config(result)
    if getattr(args, "cost_usd", False) and modifier_config is not None:
        mod_constraints = getattr(result, "constraints", None)
        workload = None
        if mod_constraints is not None:
            workload = {
                "training_tokens": getattr(mod_constraints, "training_tokens", None),
                "prompt_len": getattr(mod_constraints, "prompt_len", None),
                "output_len": getattr(mod_constraints, "output_len", None),
                "serving_batch": getattr(mod_constraints, "serving_batch", None),
            }
        modifier_config = attach_cost_block(
            modifier_config,
            args.hardware,
            workload,
            price_tier=getattr(args, "price_tier", "on_demand"),
        )
    save_config(modifier_config, paths["config"])
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
