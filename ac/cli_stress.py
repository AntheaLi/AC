"""
v1-stress CLI — `python -m v1_stress.cli stress …` and `… transition …`

Subcommands:

    stress        Print a StressVector for an architecture YAML / known model.
    quality       Print a QualityStressVector.
    transition    (Phase B — wired after delta_engine lands.)

Usage examples:

    python -m v1_stress.cli stress --known Llama-3-70B --hw h100 \\
        --batch 32 --decode-kv 4096 --tp 8

    python -m v1_stress.cli stress --arch arch.yaml --hw b200 \\
        --batch 8 --decode-kv 8192 --tp 4

    python -m v1_stress.cli quality --known Llama-3-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from throughput_model import ArchConfig  # noqa: E402
from quality_model import ArchConfig as QArchConfig, TrainingConfig  # noqa: E402
from lattice_engine import KNOWN_ARCHITECTURES  # noqa: E402

from stress import Workload, compute_throughput_stress  # noqa: E402
from quality_stress import compute_quality_stress  # noqa: E402
from delta_engine import apply_transitions, rank_transitions  # noqa: E402
from deltas import REGISTRY  # noqa: E402


VALID_HARDWARE = ["h100", "b200", "tpu_v5e", "tpu_v5p",
                  "trainium2", "trn2", "trainium3", "trn3"]

# Same map used in throughput_model.evaluate_known.
_GQA_KV_HEADS = {
    "Llama-2-7B": None, "Llama-2-13B": None,
    "Llama-2-70B": 8, "Llama-3-8B": 8, "Llama-3-70B": 8,
    "Mistral-7B": 8, "Gemma-2-9B": 8, "Qwen3-8B": 8, "Qwen3-32B": 8,
    "DeepSeek-V3": 128, "Kimi-K2.5": 64, "GLM-5.1": 64,
    "GPT-OSS-120B": 8, "MAI-Base-1": 8,
}


def _coerce(s: str) -> Any:
    """Coerce a CLI string into int / float / str (Fix #9: shared with delta-eval)."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _positive_int(v: str) -> int:
    out = int(v)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return out


def _non_negative_int(v: str) -> int:
    out = int(v)
    if out < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return out


# Accept snake_case file-name slugs and case-variant aliases for --known so
# users don't have to remember the exact PascalCase.
_KNOWN_ALIAS = {k.lower().replace("-", "_"): k for k in KNOWN_ARCHITECTURES}


def _resolve_known_name(name: str) -> str:
    if name in KNOWN_ARCHITECTURES:
        return name
    alias = _KNOWN_ALIAS.get(name.lower().replace("-", "_"))
    if alias is not None:
        return alias
    raise SystemExit(
        f"unknown architecture: {name!r} (available: "
        f"{sorted(KNOWN_ARCHITECTURES.keys())})"
    )


def _known_arch(name: str, batch: int, seq: int) -> ArchConfig:
    name = _resolve_known_name(name)
    ka = KNOWN_ARCHITECTURES[name]
    n_kv = _GQA_KV_HEADS.get(name) or ka["n_heads"]
    return ArchConfig(
        d_model=ka["d_model"], n_layers=ka["n_layers"],
        n_heads=ka["n_heads"], d_head=ka["d_head"],
        n_kv_heads=n_kv, ffn_dim=ka["ffn_dim"], ffn_type="swiglu",
        batch_size=batch, seq_len=seq, precision="bf16",
    )


def _archconfig_from_schema_v03(d: dict, batch: int, seq: int) -> ArchConfig:
    """Build an ArchConfig from a v0.3 schema JSON.

    Reads the first uniform layer band (or coalesces a first-K-dense MoE
    config) and the parallelism block; rest of the schema is ignored for
    stress purposes.
    """
    if not isinstance(d, dict):
        raise SystemExit("v0.3 schema must be a JSON object")
    arch = d.get("architecture")
    if not isinstance(arch, dict):
        raise SystemExit("v0.3 schema missing 'architecture' block")
    layers = arch.get("layer_configs") or []
    if not layers:
        raise SystemExit("v0.3 schema has no layer_configs")
    # Prefer the largest band (covers the MoE body, not just the dense prefix).
    def _band_size(lc):
        idx = lc.get("layer_idx", [])
        return len(idx) if isinstance(idx, list) else 1
    main_layer = max(layers, key=_band_size)
    attn = main_layer.get("attention") or {}
    ffn = main_layer.get("ffn") or {}
    par = d.get("parallelism") or {}

    d_model = int(arch.get("d_model"))
    n_layers = int(arch.get("n_layers"))
    n_heads = int(attn.get("n_heads", 0)) or 0
    d_head = int(attn.get("d_head", 0)) or (
        d_model // n_heads if n_heads else 0)
    n_kv_heads = int(attn.get("n_kv_heads", n_heads or 1))
    ffn_dim = int(ffn.get("ffn_dim", 0))
    if ffn.get("type") == "moe":
        ffn_dim = int(ffn.get("expert_dim", ffn_dim))
    ffn_type = "swiglu"
    precision = ffn.get("precision", "bf16")
    if isinstance(precision, dict):
        precision = precision.get("ffn", "bf16")
    moe_config = None
    if ffn.get("type") == "moe":
        moe_config = {
            "n_experts": int(ffn.get("n_experts", 1)),
            "top_k": int(ffn.get("top_k", 1)),
            "expert_dim": int(ffn.get("expert_dim", ffn_dim)),
        }
        shared = ffn.get("shared_expert")
        if isinstance(shared, dict):
            moe_config["shared_expert"] = dict(shared)
    return ArchConfig(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_head=d_head,
        n_kv_heads=n_kv_heads, ffn_dim=ffn_dim, ffn_type=ffn_type,
        batch_size=batch, seq_len=seq, precision=precision,
        moe_config=moe_config,
    )


def _load_arch_file(path: str, batch: int, seq: int) -> ArchConfig:
    """Load an arch description from JSON (v0.3 schema) or flat YAML.

    JSON path supports the schema 0.3 emitted by ac-compile / ac-delta-eval
    so users can pipe configs straight from one tool into another. YAML path
    keeps the legacy flat ArchConfig-field form for back-compat.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            d = json.load(f)
        # Detect v0.3 schema by the presence of 'architecture.layer_configs'
        if isinstance(d, dict) and isinstance(d.get("architecture"), dict) \
                and d["architecture"].get("layer_configs") is not None:
            return _archconfig_from_schema_v03(d, batch, seq)
        # Fall through to flat-field path
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            raise SystemExit("PyYAML is required to load --arch <yaml> files.")
        with open(path) as f:
            d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    if "batch_size" not in d:
        d["batch_size"] = batch
    if "seq_len" not in d:
        d["seq_len"] = seq
    # Drop fields not in ArchConfig to be tolerant of extended formats.
    fields = ArchConfig.__dataclass_fields__.keys()
    d = {k: v for k, v in d.items() if k in fields}
    return ArchConfig(**d)


# Kept under the old name so any external caller still finds it.
_load_arch_yaml = _load_arch_file


def _resolve_arch(args) -> tuple:
    if args.known:
        return _known_arch(args.known, args.batch, args.decode_kv), args.known
    if args.arch:
        return _load_arch_file(args.arch, args.batch, args.decode_kv), os.path.basename(args.arch)
    raise SystemExit("Provide --known <name> or --arch <yaml|json>.")


def cmd_stress(args) -> int:
    arch, name = _resolve_arch(args)
    wl = Workload(batch_size=args.batch, prefill_seq_len=args.prefill_seq,
                  decode_kv_len=args.decode_kv, phase=args.phase)
    sv = compute_throughput_stress(
        arch, args.hw, wl,
        tp_degree=args.tp, pp_degree=args.pp, ep_degree=args.ep,
        arch_name=name,
    )
    if args.json:
        print(json.dumps(sv.as_dict(), indent=2, default=float))
    else:
        print(sv.pretty())
    return 0


def cmd_quality(args) -> int:
    if args.known:
        name = _resolve_known_name(args.known)
        ka = KNOWN_ARCHITECTURES[name]
        n_kv = _GQA_KV_HEADS.get(name) or ka["n_heads"]
        arch = QArchConfig(
            d_model=ka["d_model"], n_layers=ka["n_layers"],
            n_heads=ka["n_heads"], d_head=ka["d_head"],
            n_kv_heads=n_kv, ffn_dim=ka["ffn_dim"],
            vocab_size=ka.get("vocab_size", 32000),
        )
    else:
        raise SystemExit("Provide --known <name>. (YAML loader for quality "
                         "ArchConfig pending.)")
    training = TrainingConfig(training_tokens=args.tokens,
                              hardware=args.hw,
                              sequence_length=args.prefill_seq)
    workload_spec = {"context_length": args.decode_kv}
    qsv = compute_quality_stress(arch, training, workload_spec, arch_name=name)
    if args.json:
        print(json.dumps(qsv.as_dict(), indent=2, default=float))
    else:
        print(qsv.pretty())
    return 0


def cmd_transition(args) -> int:
    arch, name = _resolve_arch(args)
    wl = Workload(batch_size=args.batch, prefill_seq_len=args.prefill_seq,
                  decode_kv_len=args.decode_kv, phase=args.phase)

    # Fix #9: support the same `--apply NAME --apply-args k=v` idiom that
    # `ac-delta-eval` uses. Legacy per-knob flags (`--gqa-group`, `--mla-latent`,
    # …) still work so existing scripts and the README's audit.sh keep running.
    # When both are present, `--apply-args` wins for keys it sets.
    apply_args_groups = list(getattr(args, "apply_args", None) or [])

    if args.apply:
        requested_names = [n.strip() for n in args.apply.split(",") if n.strip()]
    else:
        requested_names = list(REGISTRY.keys())

    # If we have --apply-args groups, attach them in order to the named
    # transformations. Otherwise, fall back to the legacy per-knob flags.
    def _legacy_params(tname: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if tname == "swap_attention_to_gqa":
            params = {"group_size": args.gqa_group}
        elif tname == "swap_attention_to_mla":
            params = {"latent_dim": args.mla_latent}
        elif tname == "swap_attention_to_swa":
            params = {"window_size": args.swa_window}
        elif tname == "add_state_layers":
            params = {"ratio": args.state_ratio}
        elif tname == "densify_first_k":
            params = {"k": args.dense_k}
        return params

    pairs = []
    used_groups = 0
    for tname in requested_names:
        params = _legacy_params(tname)
        # Walk --apply-args groups in declaration order, layering kv pairs onto
        # the corresponding --apply name. This matches ac-delta-eval semantics.
        if used_groups < len(apply_args_groups):
            for kv in apply_args_groups[used_groups]:
                if "=" not in kv:
                    print(f"ERROR: --apply-args expects k=v, got '{kv}'",
                          file=sys.stderr)
                    return 2
                k, v = kv.split("=", 1)
                params[k.strip()] = _coerce(v.strip())
            used_groups += 1
        pairs.append((tname, params))
    transitions = apply_transitions(
        arch, pairs,
        hardware=args.hw, workload=wl,
        tp_degree=args.tp, pp_degree=args.pp, ep_degree=args.ep,
        baseline_name=name,
    )
    ranked = rank_transitions(transitions)
    if args.json:
        print(json.dumps([t.as_dict() for t in transitions], indent=2, default=float))
        return 0
    print(f"Baseline: {name}   hw={args.hw}   workload=batch={args.batch} "
          f"prefill={args.prefill_seq} kv={args.decode_kv} phase={args.phase}")
    b = next((t.baseline_stress for t in transitions if t.baseline_stress), None)
    if b:
        binding = ", ".join(b.binding_axes) or "(none)"
        print(f"Binding stresses: {binding}")
        print()
    # Fix #10: spell out the strict definition of "relieved" — a baseline
    # binding axis is *relieved* only when the candidate band drops below
    # `pressured` (< 0.9). Axes that fall by ≥0.05 but stay ≥ pressured are
    # *softened* and now also reported, so a user reading `relief score=+0.099,
    # relieved=[(none)]` no longer thinks the ranker disagrees with itself.
    SOFTEN_THRESHOLD = 0.05
    print("Ranked transitions (most relief first):")
    print("  relieved = baseline-binding axis dropped below `pressured` "
          "(<0.9 stress)")
    print(f"  softened = baseline-binding axis fell ≥{SOFTEN_THRESHOLD:.2f} "
          "but stayed binding/pressured")
    for t in ranked:
        score = t.relief_score()
        relieved = ", ".join(t.relieved_binding_axes) or "(none)"
        newly = ", ".join(t.new_binding_axes) or "(none)"
        softened = []
        if t.baseline_stress and t.candidate_stress:
            for axis in t.baseline_stress.binding_axes:
                if axis in t.relieved_binding_axes:
                    continue
                drop = (getattr(t.baseline_stress, axis)
                        - getattr(t.candidate_stress, axis))
                if drop >= SOFTEN_THRESHOLD:
                    softened.append(f"{axis}(-{drop:.2f})")
        softened_str = ", ".join(softened) or "(none)"
        print(f"  - {t.transformation_name:32s}  "
              f"score={score: 6.3f}  relieved=[{relieved}]  "
              f"softened=[{softened_str}]  new=[{newly}]")
    infeasible = [t for t in transitions if not t.feasible]
    if infeasible:
        print()
        print("Infeasible:")
        for t in infeasible:
            print(f"  - {t.transformation_name:32s}  {t.reason_if_infeasible}")
    return 0


_CLI_STRESS_EPILOG = """\
examples:
  # 10-axis throughput stress vector for a known model on H100, decode phase
  ac-stress stress --known Mistral-7B --hw h100 --batch 32 --decode-kv 8192 --tp 8

  # quality stress vector (data risk, depth, KV pressure, etc.)
  ac-stress quality --known Llama-3-70B --tokens 2000000000000

  # apply named deltas and rank them by stress relief
  ac-stress transition --known Mistral-7B --hw h100 --tp 8 \\
      --apply swap_attention_to_gqa --apply-args group_size=8

run `ac-stress <subcommand> --help` for the full flag list of each subcommand.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ac-stress",
        description=(
            "AC stress-vector CLI — read the 10-axis throughput stress vector, "
            "the quality stress vector, or apply named deltas and rank the "
            "results. Each subcommand has its own flags; `ac-stress --help` "
            "lists the subcommands and `ac-stress <subcommand> --help` shows "
            "the flags for that subcommand."
        ),
        epilog=_CLI_STRESS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(
        dest="cmd",
        required=True,
        metavar="{stress,quality,transition}",
        title="subcommands",
        description=(
            "stress      — print the throughput StressVector for a (hw, arch, "
            "workload).\n"
            "quality     — print the QualityStressVector for a known arch.\n"
            "transition  — apply --apply <delta> [--apply-args k=v]... and "
            "rank the resulting candidates."
        ),
    )

    def _arch_args(sp):
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--known", help="known architecture name")
        g.add_argument("--arch", help="YAML architecture file")

    p_stress = sub.add_parser("stress", help="Print throughput StressVector")
    _arch_args(p_stress)
    p_stress.add_argument("--hw", default="h100",
                          choices=VALID_HARDWARE)
    p_stress.add_argument("--batch", type=_positive_int, default=1)
    p_stress.add_argument("--prefill-seq", type=_positive_int, default=2048)
    p_stress.add_argument("--decode-kv", type=_positive_int, default=2048)
    p_stress.add_argument("--phase", default="decode",
                          choices=["prefill", "decode", "training", "serving_mixed"])
    p_stress.add_argument("--tp", type=_positive_int, default=1)
    p_stress.add_argument("--pp", type=_positive_int, default=1)
    p_stress.add_argument("--ep", type=_positive_int, default=1)
    p_stress.add_argument("--json", action="store_true")
    p_stress.set_defaults(func=cmd_stress)

    p_q = sub.add_parser("quality", help="Print QualityStressVector")
    p_q.add_argument("--known", required=True)
    p_q.add_argument("--hw", default="h100")
    p_q.add_argument("--tokens", type=_positive_int, default=2_000_000_000_000)
    p_q.add_argument("--prefill-seq", type=_positive_int, default=4096)
    p_q.add_argument("--decode-kv", type=_positive_int, default=4096)
    p_q.add_argument("--json", action="store_true")
    p_q.set_defaults(func=cmd_quality)

    p_t = sub.add_parser("transition", help="apply transformations + rank")
    _arch_args(p_t)
    p_t.add_argument("--hw", default="h100",
                     choices=VALID_HARDWARE)
    p_t.add_argument("--batch", type=_positive_int, default=1)
    p_t.add_argument("--prefill-seq", type=_positive_int, default=2048)
    p_t.add_argument("--decode-kv", type=_positive_int, default=2048)
    p_t.add_argument("--phase", default="decode",
                     choices=["prefill", "decode", "training", "serving_mixed"])
    p_t.add_argument("--tp", type=_positive_int, default=1)
    p_t.add_argument("--pp", type=_positive_int, default=1)
    p_t.add_argument("--ep", type=_positive_int, default=1)
    p_t.add_argument("--apply", default="",
                     help="comma-separated transformation names (default: all)")
    # Fix #9: accept the same `--apply-args k=v` repeatable form as
    # `ac-delta-eval`. Groups are zipped with names in declaration order.
    p_t.add_argument("--apply-args", action="append", nargs="+", default=None,
                     help=("k=v args for the corresponding --apply transformation. "
                           "Repeatable; one group per --apply name. "
                           "Falls back to legacy per-knob flags when unset."))
    p_t.add_argument("--gqa-group", type=_positive_int, default=8)
    p_t.add_argument("--mla-latent", type=_positive_int, default=512)
    p_t.add_argument("--swa-window", type=_positive_int, default=4096)
    p_t.add_argument("--state-ratio", default="1:3")
    p_t.add_argument("--dense-k", type=_non_negative_int, default=3)
    p_t.add_argument("--json", action="store_true")
    p_t.set_defaults(func=cmd_transition)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
