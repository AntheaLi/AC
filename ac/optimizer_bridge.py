"""
Optimizer bridge — convert v1-ac-solver's CandidateArch + DeploymentConstraints
to a (throughput_model.ArchConfig, Workload) pair so we can compute a stress
vector for it.

The conversion is centralized here so v1-ac-solver/modifier.py can integrate
stress relief without growing extra surface area. Any future change to
CandidateArch (e.g. v2 MoE / state extensions) gets one corresponding edit
to this bridge, not to every caller.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

try:
    from .throughput_model import ArchConfig
    from .stress import StressVector, Workload, compute_throughput_stress
    from .architecture import compose_layer_type_list
except ImportError:
    from throughput_model import ArchConfig
    from stress import StressVector, Workload, compute_throughput_stress
    from architecture import compose_layer_type_list


_KV_BITS_TO_PRECISION = {16: "bf16", 8: "int8", 4: "fp4"}


def _kv_precision(kv_bits: int) -> str:
    return _KV_BITS_TO_PRECISION.get(int(kv_bits), "bf16")


def candidate_to_arch(candidate, *, batch_size: int = 1, seq_len: int = 2048) -> ArchConfig:
    """Coerce an optimizer.CandidateArch to throughput_model.ArchConfig.

    Maps:
        kv_cache_bits → kv_precision (16→bf16, 8→int8, 4→fp4)
        ffn_precision → precision (for the FFN tier; matmul cost path reads this)
        moe (dict)    → moe_config
        state_config  → state_config
        layer_type_list → layer_type_list

    Bug fix (Jul 2026): the bridge used to drop attention_type and every
    attention-variant field (MLA latents, NSA/CSA/IndexShare/MSA blocks,
    YOCO), plus MTP and CP. Any delta/stress round-trip through this
    function therefore priced an MLA/sparse baseline as full MHA — wrong
    decode KV bandwidth (TBT), wrong prefill cost (TTFT), and wrong KV
    memory. Thread all throughput-relevant fields through.
    """
    def _opt_int(name: str) -> Optional[int]:
        v = int(getattr(candidate, name, 0) or 0)
        return v if v > 0 else None

    arch = ArchConfig(
        d_model=candidate.d_model,
        n_layers=candidate.n_layers,
        n_heads=candidate.n_heads,
        d_head=candidate.d_head,
        n_kv_heads=candidate.n_kv_heads,
        ffn_dim=candidate.ffn_dim,
        ffn_type="swiglu",
        vocab_size=candidate.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        # Wave 35 fix: the canonical forward mapping
        # (optimizer.evaluate_candidate) uses ffn_precision as the
        # dominant throughput precision; this bridge passed
        # weight_precision, so any delta round-trip on an fp8-FFN
        # baseline silently flipped the candidate to bf16 FFN (phantom
        # `ffn_precision fp8→bf16` in field_changes, contaminated
        # TBT/TTFT deltas). Match the canonical convention, which the
        # reverse bridge (evaluator.arch_to_candidate) already assumes.
        precision=(getattr(candidate, "ffn_precision", None)
                   or getattr(candidate, "weight_precision", "bf16")
                   or "bf16"),
        weight_precision=str(
            getattr(candidate, "weight_precision", "bf16") or "bf16"),
        activation_precision=str(
            getattr(candidate, "activation_precision", "bf16") or "bf16"),
        attn_precision=copy.deepcopy(
            getattr(candidate, "attn_precision", None) or {
                "qk": "bf16", "v": "bf16", "output": "bf16",
            }),
        kv_precision=_kv_precision(getattr(candidate, "kv_cache_bits", 16)),
        moe_config=getattr(candidate, "moe", None),
        n_dense_ffn_layers=getattr(candidate, "n_dense_ffn_layers", 0),
        state_config=getattr(candidate, "state_config", None),
        layer_type_list=getattr(candidate, "layer_type_list", None),
        placement_strategy=str(
            getattr(candidate, "placement_strategy", "none") or "none"),
        # Attention variant (drives kv_bytes_per_token_per_layer and the
        # per-layer attention cost in the throughput model).
        attention_type=str(getattr(candidate, "attention_type", "full") or "full"),
        mla_kv_latent_dim=_opt_int("mla_kv_latent_dim"),
        mla_q_latent_dim=_opt_int("mla_q_latent_dim"),
        mla_rope_head_dim=_opt_int("mla_rope_head_dim"),
        mla_nope_head_dim=_opt_int("mla_nope_head_dim"),
        nsa_compress_block_size=_opt_int("nsa_compress_block_size"),
        nsa_compress_block_stride=_opt_int("nsa_compress_block_stride"),
        nsa_select_block_size=_opt_int("nsa_select_block_size"),
        nsa_select_top_k=_opt_int("nsa_select_top_k"),
        nsa_window_size=_opt_int("nsa_window_size"),
        csa_block_size=_opt_int("csa_block_size"),
        csa_top_k_blocks=_opt_int("csa_top_k_blocks"),
        csa_compression_dim=_opt_int("csa_compression_dim"),
        indexshare_num_buckets=_opt_int("indexshare_num_buckets"),
        indexshare_top_k_buckets=_opt_int("indexshare_top_k_buckets"),
        indexshare_index_dim=_opt_int("indexshare_index_dim"),
        msa_window_size=_opt_int("msa_window_size"),
        msa_dilated_top_k=_opt_int("msa_dilated_top_k"),
        msa_global_top_k=_opt_int("msa_global_top_k"),
        yoco_n_self_attn_layers=int(getattr(candidate, "yoco_n_self_attn_layers", 0) or 0),
        yoco_share_pattern=str(
            getattr(candidate, "yoco_share_pattern", "single_source")
            or "single_source"),
        mtp_n_predict_depths=int(getattr(candidate, "mtp_n_predict_depths", 0) or 0),
        mtp_depth_n_layers=int(getattr(candidate, "mtp_depth_n_layers", 1) or 1),
        mtp_train_loss_weight=float(
            getattr(candidate, "mtp_train_loss_weight", 0.3) or 0.3),
        rope_scaling_method=str(
            getattr(candidate, "rope_scaling_method", "none") or "none"),
        rope_scaling_factor=float(
            getattr(candidate, "rope_scaling_factor", 1.0) or 1.0),
        rope_original_max_position=int(
            getattr(candidate, "rope_original_max_position", 8192) or 8192),
        sparsity_2_4=copy.deepcopy(
            getattr(candidate, "sparsity_2_4", None)),
        cp_degree=int(getattr(candidate, "cp_degree", 1) or 1),
        cp_method=str(getattr(candidate, "cp_method", "ring") or "ring"),
    )
    # SWA is carried on CandidateArch as `swa_window`; the throughput model
    # reads the dynamic `local_window` attribute (same wiring as
    # evaluate_candidate).
    _swa = int(getattr(candidate, "swa_window", 0) or 0)
    _n_local = int(getattr(candidate, "n_local_attn_layers", 0) or 0)
    if _swa > 0 and _n_local > 0:
        # Wave 18g: local:global interleave — keep the global layers'
        # projection type and let the heterogeneous layer list carry the
        # per-layer window semantics.
        arch.local_window = _swa
        arch.n_local_attn_layers = _n_local
        arch.layer_type_list = compose_layer_type_list(
            getattr(candidate, "layer_type_list", None),
            candidate.n_layers,
            _n_local,
        )
    elif _swa > 0:
        arch.local_window = _swa  # type: ignore[attr-defined]
        arch.attention_type = "swa"
    return arch


def stress_for_candidate(
    candidate,
    *,
    hardware: str,
    tp_degree: int,
    pp_degree: int = 1,
    ep_degree: int = 1,
    dp_degree: int = 1,
    context_length: int,
    serving_batch: int = 1,
    prefill_seq_len: Optional[int] = None,
    phase: str = "decode",
    arch_name: str = "",
) -> StressVector:
    """Compute a StressVector for one optimizer candidate.

    `context_length` is the workload's decode KV length (what the user will
    be serving). `prefill_seq_len` defaults to the same value.
    """
    arch = candidate_to_arch(candidate, batch_size=serving_batch,
                              seq_len=context_length)
    wl = Workload(
        batch_size=serving_batch,
        prefill_seq_len=prefill_seq_len or context_length,
        decode_kv_len=context_length,
        phase=phase,
    )
    return compute_throughput_stress(
        arch, hardware, wl,
        tp_degree=max(1, int(tp_degree)),
        pp_degree=max(1, int(pp_degree)),
        ep_degree=max(1, int(ep_degree)),
        dp_degree=max(1, int(dp_degree)),
        cp_degree=max(1, int(getattr(arch, "cp_degree", 1) or 1)),
        arch_name=arch_name,
    )


# =============================================================================
# Helpers for modifier integration
# =============================================================================

def stress_relief_vs(baseline: StressVector, candidate: StressVector) -> dict:
    """Compute the bookkeeping that modifier.py needs for ranking.

    Returns a dict with:
        relieved_binding_axes  — axes that were pressured/binding/violated in
                                 baseline and dropped to loaded/relaxed in candidate
        new_binding_axes       — axes that weren't pressured in baseline but
                                 are now
        relief_score           — Σ relief on baseline-binding axes - 0.5 × new pressure
        severe_regression      — True if any axis jumped from <pressured to violated
    """
    from stress import PRESSURED_OR_WORSE, STRESS_AXES, severity_band

    relieved = []
    new_binding = []
    relief = 0.0
    severe = False
    for axis in STRESS_AXES:
        bv = getattr(baseline, axis)
        cv = getattr(candidate, axis)
        b_band = severity_band(bv)
        c_band = severity_band(cv)
        if b_band in PRESSURED_OR_WORSE and c_band not in PRESSURED_OR_WORSE:
            relieved.append(axis)
        if b_band not in PRESSURED_OR_WORSE and c_band in PRESSURED_OR_WORSE:
            new_binding.append(axis)
        if b_band in PRESSURED_OR_WORSE:
            relief += max(0.0, bv - cv)
        if b_band not in PRESSURED_OR_WORSE and c_band == "violated":
            severe = True
    # Penalty for new pressure (less than half-weight on the new pressure value).
    for axis in new_binding:
        relief -= 0.5 * getattr(candidate, axis)
    return {
        "relieved_binding_axes": relieved,
        "new_binding_axes": new_binding,
        "relief_score": relief,
        "severe_regression": severe,
    }
