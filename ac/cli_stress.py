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


def _known_arch(name: str, batch: int, seq: int) -> ArchConfig:
    if name not in KNOWN_ARCHITECTURES:
        raise SystemExit(f"unknown architecture: {name!r} (available: "
                         f"{sorted(KNOWN_ARCHITECTURES.keys())})")
    ka = KNOWN_ARCHITECTURES[name]
    n_kv = _GQA_KV_HEADS.get(name) or ka["n_heads"]
    return ArchConfig(
        d_model=ka["d_model"], n_layers=ka["n_layers"],
        n_heads=ka["n_heads"], d_head=ka["d_head"],
        n_kv_heads=n_kv, ffn_dim=ka["ffn_dim"], ffn_type="swiglu",
        batch_size=batch, seq_len=seq, precision="bf16",
    )


def _load_arch_yaml(path: str, batch: int, seq: int) -> ArchConfig:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise SystemExit("PyYAML is required to load --arch <yaml> files.")
    with open(path) as f:
        d = yaml.safe_load(f)
    if "batch_size" not in d:
        d["batch_size"] = batch
    if "seq_len" not in d:
        d["seq_len"] = seq
    # Drop fields not in ArchConfig to be tolerant of extended yaml.
    fields = ArchConfig.__dataclass_fields__.keys()
    d = {k: v for k, v in d.items() if k in fields}
    return ArchConfig(**d)


def _resolve_arch(args) -> tuple:
    if args.known:
        return _known_arch(args.known, args.batch, args.decode_kv), args.known
    if args.arch:
        return _load_arch_yaml(args.arch, args.batch, args.decode_kv), os.path.basename(args.arch)
    raise SystemExit("Provide --known <name> or --arch <yaml>.")


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
        ka = KNOWN_ARCHITECTURES[args.known]
        n_kv = _GQA_KV_HEADS.get(args.known) or ka["n_heads"]
        arch = QArchConfig(
            d_model=ka["d_model"], n_layers=ka["n_layers"],
            n_heads=ka["n_heads"], d_head=ka["d_head"],
            n_kv_heads=n_kv, ffn_dim=ka["ffn_dim"],
            vocab_size=ka.get("vocab_size", 32000),
        )
        name = args.known
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
    requested = args.apply.split(",") if args.apply else list(REGISTRY.keys())
    pairs = []
    for tname in requested:
        tname = tname.strip()
        if not tname:
            continue
        params = {}
        # Sensible defaults so the CLI is one-shot demoable.
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
    print("Ranked transitions (most relief first):")
    for t in ranked:
        score = t.relief_score()
        relieved = ", ".join(t.relieved_binding_axes) or "(none)"
        newly = ", ".join(t.new_binding_axes) or "(none)"
        print(f"  - {t.transformation_name:32s}  "
              f"score={score: 6.3f}  relieved=[{relieved}]  new=[{newly}]")
    infeasible = [t for t in transitions if not t.feasible]
    if infeasible:
        print()
        print("Infeasible:")
        for t in infeasible:
            print(f"  - {t.transformation_name:32s}  {t.reason_if_infeasible}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AC stress-vector CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _arch_args(sp):
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--known", help="known architecture name")
        g.add_argument("--arch", help="YAML architecture file")

    p_stress = sub.add_parser("stress", help="Print throughput StressVector")
    _arch_args(p_stress)
    p_stress.add_argument("--hw", default="h100",
                          choices=VALID_HARDWARE)
    p_stress.add_argument("--batch", type=int, default=1)
    p_stress.add_argument("--prefill-seq", type=int, default=2048)
    p_stress.add_argument("--decode-kv", type=int, default=2048)
    p_stress.add_argument("--phase", default="decode",
                          choices=["prefill", "decode", "training", "serving_mixed"])
    p_stress.add_argument("--tp", type=int, default=1)
    p_stress.add_argument("--pp", type=int, default=1)
    p_stress.add_argument("--ep", type=int, default=1)
    p_stress.add_argument("--json", action="store_true")
    p_stress.set_defaults(func=cmd_stress)

    p_q = sub.add_parser("quality", help="Print QualityStressVector")
    p_q.add_argument("--known", required=True)
    p_q.add_argument("--hw", default="h100")
    p_q.add_argument("--tokens", type=int, default=2_000_000_000_000)
    p_q.add_argument("--prefill-seq", type=int, default=4096)
    p_q.add_argument("--decode-kv", type=int, default=4096)
    p_q.add_argument("--json", action="store_true")
    p_q.set_defaults(func=cmd_quality)

    p_t = sub.add_parser("transition", help="apply transformations + rank")
    _arch_args(p_t)
    p_t.add_argument("--hw", default="h100",
                     choices=VALID_HARDWARE)
    p_t.add_argument("--batch", type=int, default=1)
    p_t.add_argument("--prefill-seq", type=int, default=2048)
    p_t.add_argument("--decode-kv", type=int, default=2048)
    p_t.add_argument("--phase", default="decode",
                     choices=["prefill", "decode", "training", "serving_mixed"])
    p_t.add_argument("--tp", type=int, default=1)
    p_t.add_argument("--pp", type=int, default=1)
    p_t.add_argument("--ep", type=int, default=1)
    p_t.add_argument("--apply", default="",
                     help="comma-separated transformation names (default: all)")
    p_t.add_argument("--gqa-group", type=int, default=8)
    p_t.add_argument("--mla-latent", type=int, default=512)
    p_t.add_argument("--swa-window", type=int, default=4096)
    p_t.add_argument("--state-ratio", default="1:3")
    p_t.add_argument("--dense-k", type=int, default=3)
    p_t.add_argument("--json", action="store_true")
    p_t.set_defaults(func=cmd_transition)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
