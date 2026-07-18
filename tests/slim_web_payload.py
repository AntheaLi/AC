"""Slim a compiler-data.json to the fields consumed by v1-web.

Drops generator-only cells, diagnostics, duplicate frontiers, calibration
internals, and family metadata that the browser never reads.  The allowlists
below are the serialization contract for index.html + app.js.

Usage: python3 scripts/slim_web_payload.py in.json out.json [--top-k 12]
"""

from __future__ import annotations
import argparse
import json
import os
import sys

TOP_KEEP = {"_regen20t", "hardware_info", "grid"}

HARDWARE_KEEP = {
    "label", "hbm_gb", "peak_bf16_tflops", "bf16_tflops",
    "peak_bf16_tf", "effective_baseline_bf16_tflops",
}

# Fields read from a grid row.  Candidate-specific topology lives on optimal
# and Pareto records; row topology remains as a legacy fallback only.
ROW_KEEP = {
    "hw", "params_B", "tokens_T", "serving", "context_length",
    "context_label", "arch_mode", "state_type", "tp", "pp", "dp",
    "nodes_per_replica", "training_cluster_gpus", "training_nodes",
    "candidates", "feasible", "pareto_size", "time_s", "optimal",
    "pareto", "justification",
    "shadow_prices", "arch_dim_prices", "families", "label", "provenance",
    "infeasible_reasons", "is_projection",
}

# Fields rendered or scored from optimal and Pareto candidates.
CAND_KEEP = {
    "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads", "ffn_dim",
    "weight_prec", "ffn_prec", "kv_bits", "params_B", "total_params_B",
    "param_cost_B", "active_params_B", "arch_family", "model_type",
    "attention_type", "moe_style", "state_type", "loss", "spine_loss",
    "chinchilla", "spine_active_params_B",
    "total_residual_pct", "architecture_residual_pct",
    "precision_residual_pct", "moe_residual_pct", "state_residual_pct",
    "penalty_pct", "confidence", "uncertainty_total_pct", "quality_terms",
    "train_tps", "train_tps_per_gpu", "train_tps_per_replica",
    "train_tps_aggregate", "train_tps_unit", "tbt_ms", "ttft_ms", "mem_gb",
    "hbm_spill_gb", "spill_tier", "prefill_model",
    "tp", "pp", "cp", "cp_degree", "cp_method", "dp", "ep",
    "training_replica_gpus", "serving_instance_gpus",
    "training_cluster_gpus", "serving_batch",
    "n_experts", "top_k", "expert_dim", "shared_expert",
    "n_dense_ffn_layers", "dense_prefix_bonus_pct",
    "mtp_n_predict_depths", "mtp_depth_n_layers", "mtp_train_loss_weight",
    "mtp_residual_pct", "nsa_compress_block_size",
    "nsa_compress_block_stride", "nsa_select_block_size",
    "nsa_select_top_k", "nsa_window_size", "yoco_n_self_attn_layers",
    "yoco_share_fraction", "sparsity_2_4", "rope_scaling_method",
    "rope_scaling_factor", "rope_original_max_position",
    "mla_kv_latent_dim", "mla_q_latent_dim", "mla_rope_head_dim",
    "mla_nope_head_dim", "mla_kv_bytes_per_token_per_layer",
    "mla_kv_reduction_vs_mha", "csa_block_size", "csa_top_k_blocks",
    "csa_compression_dim", "indexshare_num_buckets",
    "indexshare_top_k_buckets", "indexshare_index_dim", "msa_window_size",
    "msa_dilated_top_k", "msa_global_top_k", "n_attention_layers",
    "n_state_layers", "d_state", "placement_strategy", "hybrid_family",
    "hybrid_ratio", "p_attn", "in_band", "crossover_seq_len",
    "state_config",
}

FAMILY_KEEP = {
    "arch_mode", "state_type", "loss", "tbt_ms", "ttft_ms", "mem_gb",
    "spill_tier", "loss_delta_pct", "tbt_delta_pct",
}

SHADOW_KEEP = {"desc", "delta_pct", "interp"}
ARCH_DIM_KEEP = {
    "change", "delta_loss_pct", "delta_train_tps_pct", "delta_tbt_pct",
    "delta_mem_pct", "decision", "reason",
}

QUALITY_TERM_KEEP = {"value", "value_pct", "uncertainty", "uncertainty_pct"}
ARCH_QUALITY_FEATURE_KEEP = {
    "n_query_heads_default", "n_query_heads", "d_head", "n_kv_heads",
    "gqa_group_size", "kv_bytes_per_token_per_layer_bf16", "subterms",
}


def _slim_quality_terms(terms: dict) -> dict:
    out = {}
    for name, term in terms.items():
        slim = _keep_fields(term, QUALITY_TERM_KEEP)
        if name == "architecture_residual" and isinstance(
                term.get("features"), dict):
            slim["features"] = _keep_fields(
                term["features"], ARCH_QUALITY_FEATURE_KEEP)
        out[name] = slim
    return out


def _slim_candidate(c: dict) -> dict:
    out = {k: c[k] for k in c if k in CAND_KEEP}
    if isinstance(out.get("quality_terms"), dict):
        out["quality_terms"] = _slim_quality_terms(out["quality_terms"])
    return out


def _keep_fields(record: dict, fields: set[str]) -> dict:
    return {k: record[k] for k in record if k in fields}


def _slim_row(r: dict, top_k: int) -> dict:
    out = {k: r[k] for k in r if k in ROW_KEEP}
    if "optimal" in out and isinstance(out["optimal"], dict):
        out["optimal"] = _slim_candidate(out["optimal"])
    if "pareto" in out and isinstance(out["pareto"], list):
        out["pareto"] = [_slim_candidate(c) for c in out["pareto"][:top_k]]
    if "families" in out and isinstance(out["families"], list):
        out["families"] = [
            _keep_fields(family, FAMILY_KEEP) for family in out["families"]
        ]
    if "shadow_prices" in out and isinstance(out["shadow_prices"], list):
        out["shadow_prices"] = [
            _keep_fields(price, SHADOW_KEEP) for price in out["shadow_prices"]
        ]
    if "arch_dim_prices" in out and isinstance(out["arch_dim_prices"], list):
        out["arch_dim_prices"] = [
            _keep_fields(price, ARCH_DIM_KEEP) for price in out["arch_dim_prices"]
        ]
    return out


def slim_payload(data: dict, top_k: int = 12) -> dict:
    out = {k: data[k] for k in data if k in TOP_KEEP}
    hardware = out.get("hardware_info")
    if isinstance(hardware, dict):
        out["hardware_info"] = {
            name: _keep_fields(info, HARDWARE_KEEP)
            for name, info in hardware.items()
        }
    out["grid"] = [
        _slim_row(row, top_k) for row in data.get("grid", [])
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--top-k", type=int, default=12,
                    help="max pareto candidates kept per row (default 12)")
    args = ap.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    grid = data.get("grid", [])
    before_sz = os.path.getsize(args.in_path)
    data = slim_payload(data, args.top_k)
    with open(args.out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    after_sz = os.path.getsize(args.out_path)
    print(f"slim: {before_sz/1e6:.1f} MB -> {after_sz/1e6:.1f} MB "
          f"({(1 - after_sz/before_sz)*100:.1f}% reduction, "
          f"{len(grid)} rows, top_k={args.top_k})")


if __name__ == "__main__":
    main()
