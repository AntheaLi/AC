#!/usr/bin/env python3
"""Generate baseline configs for TASK A-E3 (decision retrodiction, gate2 wave 1).

Builds JSON configs in the configs/ schema (schema_version "0.3"), runs each
through ac.baseline.load_baseline_model() as a smoke test, and back-fills
measured metadata.params from the model's own derived-parameter count.
"""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ac.baseline import load_baseline_model  # noqa: E402

OUT = REPO / "validation" / "e3_retrodiction" / "model_configs"

HEADER = {
    "compiler_version": "ac 0.4.0",
    "generated_at": "2026-07-17T00:00:00+00:00",
    "input_hardware": None,
    "ac_version": "0.4.0",
    "quality_model_version": "effective_capacity_v2",
    "git_commit": "c170cda",
    "experiment_date": "2026-07-17",
    "agent": "gate2-wave1",
    "task": "A-E3 preregistration",
}


def dense_ffn(ffn_dim, precision="bf16"):
    return {"type": "swiglu", "ffn_dim": ffn_dim, "precision": precision}


def moe_ffn(n_experts, top_k, expert_dim, shared_dim=None, aux=0.001,
            capacity=1.25):
    return {
        "type": "moe",
        "n_experts": n_experts,
        "top_k": top_k,
        "expert_dim": expert_dim,
        "shared_expert": (
            {"ffn_dim": shared_dim, "precision": "bf16"}
            if shared_dim else None
        ),
        "router": {"type": "topk_softmax", "aux_loss_coef": aux,
                   "z_loss_coef": 0.0},
        "capacity_factor": capacity,
        "precision": "bf16",
    }


def band(idxs, attn, ffn, eps=1e-6):
    return {
        "layer_idx": sorted(idxs),
        "type": "transformer_block",
        "attention": attn,
        "ffn": ffn,
        "normalization": {"type": "rmsnorm", "eps": eps, "precision": "bf16"},
        "residual_dtype": "bf16",
        "state": None,
    }


def _attn_common(attn_type, n_heads, n_kv, d_head):
    return {
        "type": attn_type, "n_heads": n_heads, "n_kv_heads": n_kv,
        "d_head": d_head, "rope": True, "kv_cache_bits": 16,
        "precision": {"qk": "bf16", "v": "bf16", "output": "bf16"},
    }


def mla_attn(n_heads, n_kv, d_head, kv_latent, q_latent, rope_dim, nope_dim):
    return dict(_attn_common("mla", n_heads, n_kv, d_head),
                kv_latent_dim=kv_latent, q_latent_dim=q_latent,
                rope_head_dim=rope_dim, nope_head_dim=nope_dim)


def full_attn(n_heads, n_kv, d_head):
    return _attn_common("full", n_heads, n_kv, d_head)


def banded_layers(dense_idx, dense_dim, moe_idx, moe_kwargs, attn, eps=1e-6):
    out = []
    dense_idx = sorted(dense_idx)
    if dense_idx:
        out.append(band(dense_idx, attn, dense_ffn(dense_dim), eps))
    moe_idx = sorted(moe_idx)
    if moe_idx:
        out.append(band(moe_idx, attn, moe_ffn(**moe_kwargs), eps))
    return out


def config(model_name, source_note, d_model, n_layers, vocab, rope_base,
           rope_scaling, layer_configs, parallelism, n_dense_ffn_layers,
           params_guess, eps=1e-6):
    arch = {
        "d_model": d_model,
        "n_layers": n_layers,
        "vocab_size": vocab,
        "tied_embeddings": False,
        "positional_encoding": {
            "type": "rope", "base": rope_base,
            "scaling": rope_scaling or {"method": "none", "factor": 1.0,
                                        "original_max_position": 32768},
        },
        "layer_configs": layer_configs,
    }
    if n_dense_ffn_layers is not None:
        arch["n_dense_ffn_layers"] = n_dense_ffn_layers
    return {
        "schema_version": "0.3",
        "metadata": dict(
            HEADER,
            model_name=model_name,
            source_note=source_note,
            params=dict(params_guess),
            input_constraints={"workload": "pretraining", "slo": {}},
            predicted={}, search_stats={},
        ),
        "parallelism": parallelism,
        "architecture": arch,
    }


P = lambda tp, pp, dp, ep=1, cp=1: {
    "tensor_parallel": tp, "pipeline_parallel": pp, "data_parallel": dp,
    "expert_parallel": ep, "context_parallel": cp, "cp_method": "ring"}

yarn = lambda factor, orig: {"method": "yarn", "factor": factor,
                             "original_max_position": orig}

CONFIGS = {}

# ---------------------------------------------------------------- 1. K2-like
CONFIGS["kimi_k2_like_1t_moe"] = config(
    "kimi-k2-like-1t-moe",
    "K2-like 1T MoE shape from Kimi K2 Tech Report (arXiv:2507.20534) + "
    "HF config moonshotai/Kimi-K2-Instruct; band 0 = layer 0 dense FFN "
    "(first_k_dense_replace=1), band 1 = layers 1..60 MoE. Stored with "
    "ep=8 (schema requires n_experts divisible by ep: 384%72!=0, so EP=72 "
    "is not expressible for 384 experts; wave-2 case-5 uses ep=64/96 "
    "nearest-legal and records the deviation).",
    d_model=7168, n_layers=61, vocab=163840, rope_base=50000,
    rope_scaling=yarn(32, 4096),
    layer_configs=banded_layers(
        dense_idx=[0], dense_dim=18432, moe_idx=range(1, 61),
        moe_kwargs=dict(n_experts=384, top_k=8, expert_dim=2048,
                        shared_dim=2048, aux=0.001),
        attn=mla_attn(64, 64, 128, 512, 1536, 64, 128)),
    parallelism=P(1, 1, 1, ep=8),
    n_dense_ffn_layers=1,
    params_guess={"total_b": 1040.0, "active_b": 32.6, "embed_b": 1.2,
                  "unembed_b": 1.2},
)

# ------------------------------------------------------- 2. V3 MLA k=0 base
V3_MOE = dict(n_experts=256, top_k=8, expert_dim=2048, shared_dim=2048,
              aux=0.001)
CONFIGS["deepseek_v3_mla_moe_k0"] = config(
    "deepseek-v3-mla-moe-k0",
    "DeepSeek-V3 (arXiv:2412.19437) flattened to all-MoE single band "
    "(first_k_dense_replace=0 counterfactual); wave-2 applies "
    "densify_first_k k=3 (actual V3) / k=1 (K2 choice).",
    d_model=7168, n_layers=61, vocab=129280, rope_base=10000,
    rope_scaling=yarn(40, 4096),
    layer_configs=[band(range(61), mla_attn(128, 128, 128, 512, 1536, 64,
                                            128), moe_ffn(**V3_MOE))],
    parallelism=P(8, 1, 8, ep=8),
    n_dense_ffn_layers=0,
    params_guess={"total_b": 671.0, "active_b": 37.0, "embed_b": 0.9,
                  "unembed_b": 0.9},
)

# ------------------------------------------------ 3/4. V3 GQA / MHA k=3 base
def v3_dense_moe_bands(attn):
    return banded_layers(dense_idx=[0, 1, 2], dense_dim=18432,
                         moe_idx=range(3, 61), moe_kwargs=V3_MOE,
                         attn=attn)

CONFIGS["deepseek_v3_gqa_moe_k3"] = config(
    "deepseek-v3-gqa-moe-k3",
    "DeepSeek-V3 counterfactual: GQA-8 attention (128 q heads / 8 KV), "
    "first-3-dense bands; wave-2 applies swap_attention_to_mla.",
    d_model=7168, n_layers=61, vocab=129280, rope_base=10000,
    rope_scaling=yarn(40, 4096),
    layer_configs=v3_dense_moe_bands(full_attn(128, 8, 128)),
    parallelism=P(8, 1, 8, ep=8),
    n_dense_ffn_layers=3,
    params_guess={"total_b": 676.0, "active_b": 41.0, "embed_b": 0.9,
                  "unembed_b": 0.9},
)
CONFIGS["deepseek_v3_mha_moe_k3"] = config(
    "deepseek-v3-mha-moe-k3",
    "DeepSeek-V3 counterfactual: full MHA (128 q / 128 KV heads), "
    "first-3-dense bands; wave-2 applies swap_attention_to_mla.",
    d_model=7168, n_layers=61, vocab=129280, rope_base=10000,
    rope_scaling=yarn(40, 4096),
    layer_configs=v3_dense_moe_bands(full_attn(128, 128, 128)),
    parallelism=P(8, 1, 8, ep=8),
    n_dense_ffn_layers=3,
    params_guess={"total_b": 676.0, "active_b": 41.0, "embed_b": 0.9,
                  "unembed_b": 0.9},
)

# --------------------------------------------------------------- 5. Llama-3
CONFIGS["llama3_70b_mha"] = config(
    "llama3-70b-mha",
    "Llama-3-70B shape (arXiv:2407.21783 Table 3) counterfactual MHA "
    "variant (64 KV heads); wave-2 applies swap_attention_to_gqa "
    "group_size in {1,8,64}.",
    d_model=8192, n_layers=80, vocab=128256, rope_base=500000,
    rope_scaling=None,
    layer_configs=[band(range(80), full_attn(64, 64, 128),
                        dense_ffn(28672), eps=1e-5)],
    parallelism=P(8, 1, 1),
    n_dense_ffn_layers=None,
    params_guess={"total_b": 72.0, "active_b": 72.0, "embed_b": 1.05,
                  "unembed_b": 1.05},
    eps=1e-5,
)

# ----------------------------------------------------------- 6. MHA-7B (N1)
CONFIGS["mha_7b_n1"] = config(
    "mha-7b-n1",
    "Mistral-7B shape (arXiv:2310.06825 Table 1) counterfactual pure-MHA "
    "variant (32 KV heads); wave-2 stress at 32k context.",
    d_model=4096, n_layers=32, vocab=32000, rope_base=10000,
    rope_scaling=None,
    layer_configs=[band(range(32), full_attn(32, 32, 128),
                        dense_ffn(14336))],
    parallelism=P(1, 1, 1),
    n_dense_ffn_layers=None,
    params_guess={"total_b": 7.3, "active_b": 7.3, "embed_b": 0.13,
                  "unembed_b": 0.13},
)

# ------------------------------------------------ 7. misaligned 7B (N2)
CONFIGS["dmodel_misaligned_7b_n2"] = config(
    "dmodel-misaligned-7b-n2",
    "Mistral-7B shape with d_model=4104 (aligned to 8 only, breaks "
    "256-element TC lattice); synthetic N2 control. Aligned twin is "
    "configs/mistral_7b.json (frozen, used read-only).",
    d_model=4104, n_layers=32, vocab=32000, rope_base=10000,
    rope_scaling=None,
    layer_configs=[band(range(32), full_attn(32, 8, 128),
                        dense_ffn(14336))],
    parallelism=P(1, 1, 1),
    n_dense_ffn_layers=None,
    params_guess={"total_b": 7.4, "active_b": 7.4, "embed_b": 0.13,
                  "unembed_b": 0.13},
)

# ------------------------------------------------------- 8/9. N3 MoE pair
CONFIGS["moe_7b_total_64x8_n3"] = config(
    "moe-7b-total-64x8-n3",
    "N3-literal: plan-specified 7B-total MoE (64 experts top-8); plan "
    "expects INFEASIBLE under expert spill, audit arithmetic predicts "
    "FEASIBLE (see prereg deviation note).",
    d_model=2048, n_layers=32, vocab=32000, rope_base=10000,
    rope_scaling=None,
    layer_configs=[band(range(32), full_attn(16, 4, 128),
                        moe_ffn(64, 8, 512, shared_dim=None))],
    parallelism=P(8, 1, 1),
    n_dense_ffn_layers=0,
    params_guess={"total_b": 6.9, "active_b": 1.3, "embed_b": 0.07,
                  "unembed_b": 0.07},
)
CONFIGS["moe_430b_total_64x8_n3"] = config(
    "moe-430b-total-64x8-n3",
    "N3-scaled: intent-preserving 430B-total MoE (64 experts top-8); "
    "bf16 ~848GB total -> TP8EP1 ~106GB/GPU (soft spill, 1.33x HBM80), "
    "TP1EP1 ~848GB > 10x HBM (hard sentinel).",
    d_model=8192, n_layers=64, vocab=128256, rope_base=500000,
    rope_scaling=None,
    layer_configs=[band(range(64), full_attn(64, 8, 128),
                        moe_ffn(64, 8, 4096, shared_dim=None), eps=1e-5)],
    parallelism=P(8, 1, 1),
    n_dense_ffn_layers=0,
    params_guess={"total_b": 424.0, "active_b": 63.0, "embed_b": 1.05,
                  "unembed_b": 1.05},
    eps=1e-5,
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cfg in CONFIGS.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        try:
            model = load_baseline_model(path)
            cfg["metadata"]["params"]["total_b"] = round(
                model.total_params_b, 3)
            cfg["metadata"]["params"]["active_b"] = round(
                model.candidate.active_params_b, 3)
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            results[name] = {
                "ok": True, "total_b": round(model.total_params_b, 3),
                "active_b": round(model.candidate.active_params_b, 3),
                "n_dense": model.candidate.n_dense_ffn_layers,
                "n_bands": len(cfg["architecture"]["layer_configs"]),
                "warnings": list(model.warnings),
            }
        except Exception as e:  # noqa: BLE001
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"[{name}] {results[name]}")
    bad = {k: v for k, v in results.items() if not v["ok"]}
    print(f"\n{len(results) - len(bad)}/{len(results)} configs load OK")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
