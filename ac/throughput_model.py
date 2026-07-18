"""
Throughput Model v0 — T(A, H, lattice) -> ThroughputResult

Analytical end-to-end throughput model for dense transformers (with optional GQA)
on NVIDIA H100, NVIDIA B200, Google TPU v5e, and Google TPU v5p.

Sits on top of the tile-aligned lattice: queries tile efficiency per matmul,
never modifies the lattice. Designed for millisecond-scale evaluation so the
optimizer can call it thousands of times per second.

Extension hooks reserved for v1+: MoE, state layers, heterogeneous pipelines.
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Import lattice engine (sibling module in the same package)
#
# Audit follow-up: the previous version unconditionally inserted the
# package directory into sys.path before importing. That always works but
# pollutes sys.path even when this module is imported as part of the
# installed package. We now prefer a relative import and only fall back
# to the absolute path-hack form when the module is being run as a
# direct script (`python ac/throughput_model.py`) without a parent
# package — which is the case the path-hack was originally written for.
# ---------------------------------------------------------------------------
try:  # importable as part of the `ac` package (normal case)
    from .lattice_engine import (  # type: ignore[no-redef]
        HardwareSpec as LatticeHardwareSpec,
        TileSpec,
        HARDWARE as LATTICE_HARDWARE,
        matmul_tile_utilization,
        wave_efficiency,
        compute_lattice,
        compute_gqa_configs,
        estimate_params,
        KNOWN_ARCHITECTURES,
    )
except ImportError:  # direct-script invocation: synthesize sibling import
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from lattice_engine import (  # type: ignore[no-redef]
        HardwareSpec as LatticeHardwareSpec,
        TileSpec,
        HARDWARE as LATTICE_HARDWARE,
        matmul_tile_utilization,
        wave_efficiency,
        compute_lattice,
        compute_gqa_configs,
        estimate_params,
        KNOWN_ARCHITECTURES,
    )

try:
    from .architecture import (  # type: ignore[no-redef]
        parameter_byte_ledger, parameter_ledger,
        precision_bytes_per_element, training_parameter_byte_layout,
        training_parameter_layout,
    )
except ImportError:
    from architecture import (  # type: ignore[no-redef]
        parameter_byte_ledger, parameter_ledger,
        precision_bytes_per_element, training_parameter_byte_layout,
        training_parameter_layout,
    )


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ArchConfig:
    """Architecture configuration — input to the throughput model."""
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    n_kv_heads: int          # = n_heads for MHA, < n_heads for GQA/MQA
    ffn_dim: int
    ffn_type: str = "swiglu"  # "swiglu" | "dense"
    vocab_size: int = 32000
    batch_size: int = 1
    seq_len: int = 2048
    precision: str = "bf16"
    # Keep weight and activation precision explicit even though the current
    # matmul model uses `precision` as its dominant compute tier. Delta and
    # bridge paths need to preserve all three independently; activation
    # precision also controls activation/communication byte volume.
    weight_precision: str = "bf16"
    activation_precision: str = "bf16"
    attn_precision: Dict[str, str] = field(default_factory=lambda: {
        "qk": "bf16", "v": "bf16", "output": "bf16",
    })
    kv_precision: str = "bf16"  # KV cache precision (can differ for quantized KV)

    # v1 MoE hook (v0.1: always None; v0.2+: dict with the MoE FFN block).
    # Shape matches schema MoEFFNConfig:
    #   {n_experts, top_k, expert_dim, shared_expert, router,
    #    capacity_factor, precision}
    layer_types: Optional[List[str]] = None   # v0: always ['attention'] * n_layers
    moe_config: Optional[dict] = None

    # v1 MoE: worst-case routing imbalance multiplier. 1.0 = balanced (training
    # objective). Set > 1.0 to stress-test concentration during planning.
    worst_case_imbalance_factor: float = 1.0

    # v1-fix Part B: first-K-dense prefix when moe_config is set. Throughput
    # treats per-layer FFN cost as a weighted average:
    #   per-stack FFN time =
    #     n_dense_ffn_layers * dense_ffn_time +
    #     (n_layers - n_dense_ffn_layers) * moe_ffn_time
    # All-to-all volume also pro-rates by MoE-layer fraction.
    n_dense_ffn_layers: int = 0

    # v2 state/hybrid fields
    state_config: Optional[dict] = None
    # Keys: d_state, state_expansion, n_heads, d_head, state_precision
    layer_type_list: Optional[List[str]] = None
    # Per-layer type: "attention" or "state". When None, all attention.
    placement_strategy: str = "none"

    # v1-fix MLA: DeepSeek-V2/V3 Multi-head Latent Attention. When
    # `attention_type == "mla"`, the per-token KV cache stores ONE compressed
    # latent (c_kv) plus the RoPE'd key (d_rope), regardless of n_kv_heads.
    # This dramatically cuts decode KV bandwidth at long context. Quality
    # cost is captured by `attention_mla` in the architecture residual.
    attention_type: str = "full"          # "full" | "mla" | "nsa" | "csa" | "indexshare" | "msa"
    mla_kv_latent_dim: Optional[int] = None     # c_kv
    mla_q_latent_dim: Optional[int] = None      # c_q
    mla_rope_head_dim: Optional[int] = None     # d_rope
    mla_nope_head_dim: Optional[int] = None     # d_nope
    # v1-fix NSA: Native Sparse Attention (DeepSeek 2025). Three hierarchical
    # branches sum to per-token KV: B = ceil(L / stride) compressed blocks +
    # K × bs selected tokens + W sliding-window tokens.
    nsa_compress_block_size: Optional[int] = None
    nsa_compress_block_stride: Optional[int] = None
    nsa_select_block_size: Optional[int] = None
    nsa_select_top_k: Optional[int] = None
    nsa_window_size: Optional[int] = None
    # Wave 9 (Jun 2026): compressed-attention variants from 2025-2026 frontier.
    # See plan/redesign/09-compressed-attention-coverage.md for full scope.
    #
    # CSA (Compressed Sparse Attention) — DeepSeek-V4. KV stored per
    # compressed block; query attends top_k_blocks. KV reduction ≈
    # block_size / top_k_blocks at long ctx.
    csa_block_size: Optional[int] = None       # tokens per compressed block
    csa_top_k_blocks: Optional[int] = None     # blocks attended per query
    csa_compression_dim: Optional[int] = None  # per-block latent dim
    # IndexShare — GLM-5.2 / MiniMax M3. Bucket-routed KV cache; query
    # hashes to top_k_buckets. Constant-time decode at the cost of a
    # bucket-routing step.
    indexshare_num_buckets: Optional[int] = None
    indexshare_top_k_buckets: Optional[int] = None
    indexshare_index_dim: Optional[int] = None
    # MSA (Mixture of Sparse Attention) — sum of (window + dilated + global).
    # Each fraction is the share of attention heads using that pattern.
    msa_window_size: Optional[int] = None        # local window tokens
    msa_dilated_top_k: Optional[int] = None      # dilated heads top-k
    msa_global_top_k: Optional[int] = None       # global heads top-k

    # v1-fix MTP: Multi-Token Prediction training overhead. Inference path
    # is unchanged (heads dropped); training path pays per-depth FLOPs.
    mtp_n_predict_depths: int = 0
    mtp_depth_n_layers: int = 1
    mtp_inference_mode: str = "drop"      # "drop" | "speculative_decode"
    # Quality/config identity retained on this shared delta representation.
    mtp_train_loss_weight: float = 0.3

    # Positional identity is quality-only today, but delta transforms use
    # ArchConfig as their mutable architecture view and must preserve it.
    rope_scaling_method: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 8192

    # v1-fix CP: Context Parallelism. Splits the sequence axis across CP
    # ranks. At training time, attention compute and KV-cache memory both
    # divide by cp; the all-gather comm cost is roughly
    #   (cp-1)/cp × n_layers × B × S × d_model × bpe / NVLink_bw
    # per training step.
    cp_degree: int = 1                    # 1 = no CP
    cp_method: str = "ring"               # "ring" | "ulysses"

    # v1-fix Wave 1 Step 1.4/1.5 (Jun 2026): pipeline schedule.
    # "gpipe" — flush after every M_micro microbatches, queue depth = pp_degree.
    # "1f1b" — interleave 1F1B, queue depth ≈ (pp_degree + 1) / 2, bubble ≈ 0.5 × GPipe.
    # "interleaved" — Megatron interleaved 1F1B with `pp_virtual_stages` chunks,
    #                  bubble ≈ GPipe / pp_virtual_stages.
    pp_schedule: str = "1f1b"
    pp_virtual_stages: int = 2
    # Training execution plan. These are deliberately separate from serving
    # batch and from pipeline microbatch count.
    dp_degree: int = 1
    training_micro_batch: int = 8
    pipeline_microbatches: int = 1
    serving_scheduler: str = "continuous"
    serving_concurrency: int = 1
    serving_output_len: int = 1
    prefill_chunk_size: int = 65536

    # v1-fix 2:4 structured sparsity. Per-component flags driving the
    # tensor-core sparse path. NVIDIA H100/B200 give 2× matmul throughput
    # on 2:4-sparsified weights. Other vendors (TPU, Trainium) fall back
    # to dense, so the speedup is hardware-conditional.
    sparsity_2_4: Optional[Dict[str, bool]] = None

    # v1-fix YOCO: cross-layer KV sharing. Only the first K layers each
    # carry their own KV; the remaining N-K layers cross-attend to the
    # K-th layer's KV. KV cache shrinks to K/N of dense, and decode
    # bandwidth too (modulo the cross-attention read pattern).
    yoco_n_self_attn_layers: int = 0  # 0 = YOCO off
    yoco_share_pattern: str = "single_source"

    # Wave 18g (Jul 2026): per-layer attention heterogeneity — local:global
    # interleave (GPT-OSS / Gemma-2 / Llama-4 pattern). `n_local_attn_layers`
    # of the attention layers use sliding-window attention with
    # `local_window`; the remainder are global (full/GQA or MLA, per
    # `attention_type`). Local layers cap their KV cache and decode reads at
    # the window and their prefill attention at S x W. Globals are spread
    # evenly through the stack (periodic placement), matching published
    # interleaves. `local_window` alone (without n_local_attn_layers) keeps
    # its legacy meaning: every attention layer is windowed.
    # NOTE: local_window was previously only ever set dynamically by the
    # optimizer bridge (tput_arch.local_window = swa_window) — it was not a
    # declared field, so any code path that read it on a freshly-constructed
    # ArchConfig would AttributeError. Declared properly as of Wave 18g.
    local_window: Optional[int] = None
    n_local_attn_layers: int = 0

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["attention"] * self.n_layers
        if self.layer_type_list is None:
            n_local = int(self.n_local_attn_layers or 0)
            if 0 < n_local < self.n_layers:
                # Even periodic spread of global layers through the stack.
                n_global = self.n_layers - n_local
                lst = []
                acc = 0
                for i in range(self.n_layers):
                    nxt = (i + 1) * n_global // self.n_layers
                    if nxt > acc:
                        lst.append("attention")
                        acc = nxt
                    else:
                        lst.append("local_attention")
                self.layer_type_list = lst
            elif n_local >= self.n_layers and self.n_layers > 0:
                self.layer_type_list = ["local_attention"] * self.n_layers
            else:
                self.layer_type_list = ["attention"] * self.n_layers

    def local_kv_len(self, full_len: int) -> int:
        """Effective KV length of a local (sliding-window) attention layer."""
        w = int(self.local_window or 0)
        if w <= 0:
            return int(full_len)
        return int(min(full_len, w))

    def kv_bytes_per_token_per_layer(self, context_length: int = 0) -> int:
        """v1-fix MLA/NSA: per-token per-layer KV cache bytes.

        For MLA, KV is ONE compressed latent (c_kv) + RoPE'd K (d_rope)
        shared across all heads. For NSA, KV is the sum of three branches:
        compressed (ceil(L/stride) blocks × d_head), selected (top-k blocks ×
        block-size × d_head), and sliding window (W × d_head). The full
        L-dependent sum is folded into `kv_bytes_per_layer` below by callers
        that pass `context_length`; for the per-token figure, we report the
        amortized per-token cost using the effective NSA budget.
        For non-MLA/non-NSA, K and V are each stored per-kv-head:
        2 × n_kv_heads × d_head × bpe.
        """
        bpe = precision_bytes_per_element(self.kv_precision)
        if self.attention_type == "mla" and self.mla_kv_latent_dim:
            c_kv = int(self.mla_kv_latent_dim)
            d_rope = int(self.mla_rope_head_dim or 0)
            return int((c_kv + d_rope) * bpe)
        if self.attention_type == "nsa" and self.nsa_window_size:
            # NSA's *amortized* per-token KV: stored once but read multiple
            # times. For decode-time read bandwidth, the effective budget is
            # (compressed_blocks + selected_tokens + window) attended *per
            # query token*. We use a context-aware compact form here when
            # the caller supplies L; otherwise approximate by the steady-state
            # NSA budget at L=64k (DeepSeek paper).
            L = int(context_length or 65536)
            cbs = int(self.nsa_compress_block_size or 64)
            cbst = int(self.nsa_compress_block_stride or 16)
            sbs = int(self.nsa_select_block_size or 64)
            stk = int(self.nsa_select_top_k or 16)
            win = int(self.nsa_window_size or 512)
            # Total bytes read per query token (averaged over heads):
            n_compress = max(1, (L + cbst - 1) // cbst)
            n_select = stk * sbs
            effective_tokens = min(L, n_compress + n_select + win)
            # Per-token *cache footprint* (write-side) is still the full L
            # per kv-head; but bandwidth-relevant figure is effective_tokens
            # divided by L. To match the existing kv_per_layer math
            # (per_token × L = total), we shrink per-token bytes so that
            # per_token × L = effective_tokens × per_kv_head_bytes_per_tok.
            per_kv_head = 2 * self.n_kv_heads * self.d_head * bpe
            return int(per_kv_head * effective_tokens / max(1, L))
        if self.attention_type == "csa" and self.csa_top_k_blocks:
            # Wave 9 CSA: KV stored at per-block compressed representation
            # (compression_dim per block instead of block_size × d_head).
            # Each query reads top_k_blocks compressed blocks.
            # Per-token effective: (top_k_blocks × compression_dim) bytes,
            # write-side reduced to (compression_dim / block_size) of dense.
            block_size = int(self.csa_block_size or 64)
            top_k = int(self.csa_top_k_blocks or 16)
            comp_dim = int(self.csa_compression_dim or self.d_head)
            # Effective per-token: amortize compression over block_size tokens.
            # Read bandwidth = (top_k × comp_dim × 2) / block_size, weighted by
            # n_kv_heads for shape parity with full attention.
            per_kv_head_csa = 2 * self.n_kv_heads * comp_dim * bpe
            return int(per_kv_head_csa * top_k / max(1, block_size))
        if self.attention_type == "indexshare" and self.indexshare_top_k_buckets:
            # Wave 9 IndexShare: KV stored per bucket; query routes to
            # top_k_buckets. Effective per-token KV ≈ (top_k / num_buckets)
            # of dense, plus a small constant for the index dim.
            num_buckets = int(self.indexshare_num_buckets or 64)
            top_k = int(self.indexshare_top_k_buckets or 4)
            idx_dim = int(self.indexshare_index_dim or 64)
            dense_per_token = 2 * self.n_kv_heads * self.d_head * bpe
            # Top-K bucket coverage + per-token index lookup.
            return int(dense_per_token * top_k / max(1, num_buckets)
                       + 2 * idx_dim * bpe)
        if self.attention_type == "msa":
            # Wave 9 MSA: weighted sum of (window + dilated + global) sub-
            # patterns. Each sub-pattern reads its own share of KV per
            # query; we sum the per-token effective bytes across patterns.
            L = int(context_length or 65536)
            win = int(self.msa_window_size or 512)
            dilated_k = int(self.msa_dilated_top_k or 64)
            global_k = int(self.msa_global_top_k or 16)
            # Window: read `win` per query — local always-paid.
            window_share = win / max(1, L)
            # Dilated + global: read top_k per query (sampled across L).
            dilated_share = dilated_k / max(1, L)
            global_share = global_k / max(1, L)
            dense_per_token = 2 * self.n_kv_heads * self.d_head * bpe
            effective_share = min(1.0, window_share + dilated_share + global_share)
            return int(dense_per_token * effective_share)
        return int(2 * self.n_kv_heads * self.d_head * bpe)

    def kv_bytes_per_token_split(self, context_length: int = 0):
        """Wave 35: (tp_shardable, tp_replicated) per-token per-layer KV bytes.

        The per-kv-head portion of a KV cache shards across TP exactly like
        GQA heads do — this covers full/GQA/MQA, NSA, CSA, and MSA, whose
        caches are all stored per kv head. Two structures are genuinely
        replicated across TP ranks because they are shared across query
        heads: the MLA latent (+ RoPE key), and the IndexShare per-token
        index entry (the DSA-style lightning indexer is shared across
        heads). Previously the decode-bandwidth path treated ALL of
        NSA/CSA/IndexShare/MSA as replicated while the capacity path
        treated them (index included) as sharded — mutually inconsistent
        and both wrong for IndexShare. Sums to kv_bytes_per_token_per_layer.
        """
        total = self.kv_bytes_per_token_per_layer(context_length)
        if self.attention_type == "mla" and self.mla_kv_latent_dim:
            return 0, total
        if self.attention_type == "indexshare" and self.indexshare_top_k_buckets:
            bpe = precision_bytes_per_element(self.kv_precision)
            idx_dim = int(self.indexshare_index_dim or 64)
            replicated = int(2 * idx_dim * bpe)
            return max(0, total - replicated), replicated
        return total, 0


@dataclass
class HardwareConfig:
    """Hardware configuration loaded from JSON spec files."""
    vendor: str
    accelerator_family: str
    accelerator_name: str
    compute_units: int
    compute_unit_type: str
    hbm_capacity_gb: float
    hbm_bandwidth_tb_s: float
    peak_flops_tf: Dict[str, float]
    bytes_per_element: Dict[str, float]
    supported_precisions: List[str]
    fused_attention_efficiency: Dict[str, float]
    fused_attention_kernel: str
    interconnect: dict
    gpus_per_node: int = 8
    chips_per_host: int = 4

    # Fabric domain size: how many ranks share full-bandwidth NVLink (or single-axis ICI).
    # 8 for DGX-H100/HGX-H100/DGX-B200, 72 for NVL72, 16 for TPU v5p single torus axis,
    # 8 for TPU v5e. Optional in JSON — falls back to family-based inference when absent.
    nvlink_domain_size: Optional[int] = None

    # Gate-2 Task C (rack-scale system layer, pure-additive): optional
    # rack-level metadata for system targets such as GB200 NVL72. Both
    # default to None; single-node specs are completely unaffected.
    #   gpus_per_rack — accelerators inside one rack-scale NVLink domain.
    #   chip          — records chip-level parameter inheritance
    #                   (gb200_nvl72 reuses b200 single-chip params).
    gpus_per_rack: Optional[int] = None
    chip: Optional[str] = None

    # Wave 20 (feedback #2): vendor DATASHEET dense Tensor-Core peaks, for
    # implied-MFU reporting only. On NVIDIA parts `peak_flops_tf` is the
    # internal roofline baseline (~half of datasheet by convention); on
    # TPU/Trainium the two coincide. The roofline model must NOT use this.
    datasheet_peak_flops_tf: Optional[Dict[str, float]] = None

    # Calibration constants — account for real-world overheads not modeled analytically
    calibration: dict = field(default_factory=lambda: {
        "kernel_launch_overhead_us": 5.0,
        "kernels_per_layer": 12,
        "optimizer_bytes_per_param": 12,    # AdamW: 4 (fp32 master) + 4 (m) + 4 (v) = 12 bytes
        "training_system_efficiency": 0.55, # accounts for data loading, optimizer, mem mgmt
        "decode_system_efficiency": 0.42,   # accounts for kernel launch dominance at small batch
        "prefill_system_efficiency": 0.60,
    })

    @classmethod
    def from_json(cls, path: str) -> "HardwareConfig":
        with open(path) as f:
            d = json.load(f)
        calibration = d.get("calibration", {})
        interconnect = dict(d["interconnect"])

        # Gate-2 Task C: optional rack-scale `system` block. When present it
        # is the authoritative source for the fabric-domain layer and is
        # mapped onto the legacy fields below; when absent the loader keeps
        # single-node semantics exactly (nvlink_domain_size stays whatever
        # the top-level field says — default None → family inference in
        # `_nvlink_domain_size`, 8 for all shipping single-node specs — and
        # the interconnect dict passes through unmodified), so pre-existing
        # spec files behave byte-identically.
        system = d.get("system") or {}
        if not isinstance(system, dict):
            system = {}
        if system:
            if system.get("intra_domain_bandwidth_gbps") is not None:
                interconnect["intra_node_bw_gb_s"] = system["intra_domain_bandwidth_gbps"]
            if system.get("inter_domain_bandwidth_gbps") is not None:
                interconnect["inter_node_bw_gb_s"] = system["inter_domain_bandwidth_gbps"]

        nvlink_domain_size = d.get("nvlink_domain_size")
        if system.get("nvlink_domain_size") is not None:
            nvlink_domain_size = system["nvlink_domain_size"]
        gpus_per_rack = d.get("gpus_per_rack")
        if system.get("gpus_per_rack") is not None:
            gpus_per_rack = system["gpus_per_rack"]

        return cls(
            vendor=d["vendor"],
            accelerator_family=d["accelerator_family"],
            accelerator_name=d["accelerator_name"],
            compute_units=d["compute_units"],
            compute_unit_type=d["compute_unit_type"],
            hbm_capacity_gb=d["hbm_capacity_gb"],
            hbm_bandwidth_tb_s=d["hbm_bandwidth_tb_s"],
            peak_flops_tf=d["peak_flops_tf"],
            bytes_per_element=d["bytes_per_element"],
            supported_precisions=d["supported_precisions"],
            fused_attention_efficiency=d["fused_attention_efficiency"],
            fused_attention_kernel=d["fused_attention_kernel"],
            interconnect=interconnect,
            gpus_per_node=d.get("gpus_per_node", d.get("chips_per_host", 8)),
            chips_per_host=d.get("chips_per_host", d.get("gpus_per_node", 4)),
            nvlink_domain_size=nvlink_domain_size,
            gpus_per_rack=gpus_per_rack,
            chip=d.get("chip"),
            datasheet_peak_flops_tf=d.get("datasheet_peak_flops_tf"),
            calibration=calibration,
        )

    @property
    def hbm_bandwidth_bytes_s(self) -> float:
        return self.hbm_bandwidth_tb_s * 1e12

    def peak_flops_s(self, precision: str) -> float:
        """Peak FLOPS in raw FLOPS (not teraflops)."""
        return self.peak_flops_tf.get(precision, self.peak_flops_tf.get("bf16", 0)) * 1e12

    def datasheet_peak_flops_s(self, precision: str) -> float:
        """Vendor datasheet dense peak in raw FLOPS — for implied-MFU
        reporting only (Wave 20, feedback #2). Falls back to the internal
        roofline baseline when the spec file lacks the field."""
        table = self.datasheet_peak_flops_tf or self.peak_flops_tf
        return table.get(precision, table.get("bf16", 0)) * 1e12

    # Wave 21: canonical fallback byte widths. The per-spec JSON
    # `bytes_per_element` tables are incomplete (h100_sxm.json has no
    # mxfp4/mxfp6/int4 entries), and the old `.get(precision, 2)` fallback
    # silently priced any missing narrow format as bf16-sized — so the
    # SAME mxfp4 expert weights cost 0.53 B/elem in the memory estimator
    # (module-level map) but 2 B/elem in the matmul/decode roofline
    # (spec-table lookup). On the gpt-oss-120b anchor this inflated decode
    # TBT by ~2.2x while memory read ~4x small: two panels, two physics.
    # MX formats include the shared scale (1 byte / 32 elems).
    def bytes_per_elem(self, precision: str) -> float:
        v = self.bytes_per_element.get(precision)
        if v is not None:
            return v
        return precision_bytes_per_element(precision)

    def interconnect_bw_bytes_s(self, tp_degree: int) -> float:
        """Effective interconnect bandwidth for TP all-reduce."""
        if tp_degree <= 1:
            return float("inf")
        ic = self.interconnect
        local_domain = (
            self.nvlink_domain_size
            if self.vendor == "nvidia" and self.nvlink_domain_size
            else self.gpus_per_node
            if self.vendor == "nvidia"
            else self.chips_per_host
        )
        if tp_degree <= local_domain:
            return ic["intra_node_bw_gb_s"] * 1e9
        else:
            return ic["inter_node_bw_gb_s"] * 1e9


@dataclass
class LayerBreakdown:
    """Per-layer time breakdown in seconds.

    v0.2: extended with MoE-specific terms (alltoall_s, expert_load_s,
    load_balance_factor). Dense layers report zeros for these. The
    bottleneck enum now includes "alltoall" and "expert_load".
    """
    compute_s: float = 0.0
    memory_s: float = 0.0
    communication_s: float = 0.0
    total_s: float = 0.0
    bottleneck: str = "compute"

    # Sub-operation detail
    qkv_proj_s: float = 0.0
    attention_s: float = 0.0
    out_proj_s: float = 0.0
    ffn_up_s: float = 0.0
    ffn_down_s: float = 0.0
    membound_ops_s: float = 0.0
    allreduce_s: float = 0.0

    # v1+ MoE-specific terms (zero for dense layers)
    alltoall_s: float = 0.0           # expert dispatch + combine
    expert_load_s: float = 0.0        # decode-phase expert weight loading
    shared_expert_s: float = 0.0      # DeepSeek-style always-on expert compute
    load_balance_factor: float = 1.0  # routing imbalance multiplier applied to expert compute


@dataclass
class ThroughputResult:
    """Output of the throughput model."""
    # Training
    training_time_per_step_s: float = 0.0
    training_throughput_tokens_per_sec: float = 0.0
    # v1-fix Wave 1 Step 1.2 (Jun 2026): time spent in the DP gradient
    # reduce-scatter + weight all-gather under FSDP/ZeRO-3. Folded into
    # training_time_per_step_s; reported here so the optimizer / report
    # can flag DP-bandwidth-bound configs explicitly.
    dp_grad_allreduce_s: float = 0.0

    # Inference — prefill
    prefill_time_ms: float = 0.0
    # Wave 29: additive serving-stack TTFT floor included in
    # prefill_time_ms (tokenize + scheduler admission + sampler +
    # detokenize + transport; EXCLUDES load-dependent queueing).
    # Recorded separately so reports can attribute it.
    ttft_serving_overhead_ms: float = 0.0

    # Inference — decode (per generated token, at a given KV cache length)
    decode_time_per_token_ms: float = 0.0
    decode_kv_cache_length: int = 0
    serving_request_latency_ms: float = 0.0
    serving_scheduler: str = "continuous"
    effective_serving_batch: int = 1

    # v1-fix throughput uncertainty. One-sigma absolute uncertainty in
    # milliseconds for the matching phase, derived from the calibrated
    # (regime × arch_family) efficiency table's sigma. Downstream Pareto /
    # contender code reads these to treat near-ties as actual ties.
    training_throughput_sigma_tps: float = 0.0
    prefill_time_sigma_ms: float = 0.0
    decode_time_sigma_ms: float = 0.0

    # Which (regime, family) bucket was used to set efficiency. Useful in
    # reports so a researcher can see "ah, this is using moe_compute_bound,
    # which is calibrated from only 4 samples." Empty string when the
    # scalar fallback was used.
    efficiency_bucket: str = ""

    # Memory
    memory_footprint_per_gpu_gb: float = 0.0
    # v1-fix Wave 2a Step 2a.2 (Jun 2026): HBM-overflow as continuous cost.
    # When memory_footprint_per_gpu_gb exceeds HBM, the v0 model declared
    # infeasibility. In a 10k+ GPU deployment the model actually spills via
    # NVLink (intra-node, 3-5× slower than HBM) and then PCIe/IB (50-100×
    # slower). We now record the overflow and the tier, and adjust the
    # reported decode/prefill times to reflect the bandwidth penalty. The
    # serving-side hard cap moves out of optimizer.py:_check_feasibility
    # (Step 2a.3) so memory becomes a continuous Pareto axis.
    hbm_spill_gb: float = 0.0
    spill_tier: str = "fits"          # "fits" | "nvlink" | "pcie" | "mixed"
    tbt_ms_no_spill: float = 0.0       # what TBT would be if it fit in HBM
    ttft_ms_no_spill: float = 0.0      # same for prefill
    # v1-fix demo-audit-2 (Jun 2026): the field above is *inference* memory
    # (weights + KV cache + activations). The field below is the per-GPU
    # *training* memory under FSDP/ZeRO-3 sharding across DP replicas, which
    # is what actually has to fit in HBM during pretraining. Without this
    # the optimizer happily declared 7B BF16 feasible on a single H100 with
    # TP=1 PP=1 even though real training memory (weights + grads + AdamW
    # opt states + activations) is 90-150 GB before sharding.
    training_memory_per_gpu_gb: float = 0.0
    training_sequence_length: int = 0

    # Breakdown
    per_layer_breakdown: Optional[LayerBreakdown] = None  # training
    prefill_layer_breakdown: Optional[LayerBreakdown] = None
    decode_layer_breakdown: Optional[LayerBreakdown] = None
    bottleneck: str = "compute"

    # Parallelism
    tp_degree: int = 1
    pp_degree: int = 1
    dp_degree: int = 1
    ep_degree: int = 1
    cp_degree: int = 1
    pipeline_microbatches: int = 1
    bubble_fraction: float = 0.0
    pp_training_comm_s: float = 0.0
    pp_prefill_comm_s: float = 0.0
    pp_decode_comm_s: float = 0.0

    # Metadata
    hardware_name: str = ""
    precision: str = ""


# =============================================================================
# Hardware spec loader
# =============================================================================

_SPEC_DIR = os.environ.get(
    "AC_HARDWARE_SPEC_DIR",
    os.path.join(os.path.dirname(__file__), "hardware_specs"),
)

# Wave 29: default serving-stack TTFT floor. Published TTFTs include a
# stack floor AC's pure-compute prefill never modeled — every anchor's
# TTFT was under-predicted −30…−90% with the small-prompt anchors worst
# (predicted 6.8 ms vs published 90 ms on GPT-OSS-120B @ 1k prompt).
# The floor is tokenize (~1-2 µs/token BPE on a CPU core) + scheduler
# admission + sampling + detokenize + HTTP framing (~10-20 ms on
# vLLM/TRT-LLM-class stacks). It deliberately EXCLUDES queueing: p95
# endpoint numbers under load sit far above this floor, and charging
# load-dependent waiting to the architecture would be dishonest.
# Override per hardware target via the spec's calibration block:
#   "ttft_serving_overhead": {"fixed_ms": 15.0, "per_prompt_token_us": 1.5}
DEFAULT_TTFT_FIXED_OVERHEAD_MS = 15.0
DEFAULT_TTFT_PER_PROMPT_TOKEN_US = 1.5
_CALIBRATION_DIR = os.environ.get(
    "AC_CALIBRATION_DIR",
    os.path.join(os.path.dirname(__file__), "calibration"),
)


# =============================================================================
# Calibration layer — optional measured efficiency overrides
# =============================================================================

@dataclass
class CalibrationTable:
    """Measured kernel efficiencies that override analytic estimates."""
    gemm_efficiencies: Dict[str, float] = field(default_factory=dict)
    attention_latencies: Dict[str, float] = field(default_factory=dict)
    decode_kv_latencies: Dict[str, float] = field(default_factory=dict)
    allreduce_latencies: Dict[str, float] = field(default_factory=dict)
    # Wave 7b.4: per-hardware Wave 1-5 calibration fits. Each field is
    # optional — when None the throughput/quality models fall back to the
    # Python default the spec lists (dp_grad_overlap=0.7, tp_overlap=0.5,
    # state_long_context_weight=0.030, hbm_spill_factor=1.0). pp_queue is
    # a per-schedule dict, e.g. {"1f1b": 1.05, "gpipe": 0.98}.
    dp_grad_overlap_fraction: Optional[float] = None
    tp_allreduce_overlap_fraction: Optional[float] = None
    # Wave 19 (P0-1): fraction of MoE dispatch/combine all-to-all hidden
    # behind expert/shared compute in training & prefill (DeepEP-style
    # overlap). None → the throughput model's default (0.6).
    moe_alltoall_overlap_fraction: Optional[float] = None
    pp_queue_multipliers: Dict[str, float] = field(default_factory=dict)
    state_long_context_weight: Optional[float] = None
    hbm_spill_decode_factor: Optional[float] = None
    source: str = "analytic"
    _gemm_lookup_cache: Dict[Tuple[int, int, int, str], Optional[float]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_json(cls, path: str) -> "CalibrationTable":
        with open(path) as f:
            d = json.load(f)
        table = cls(source=d.get("source", "public_estimate"))
        for entry in d.get("gemm_shapes", []):
            key = f"{entry['m_bucket']}x{entry['n_bucket']}x{entry['k_bucket']}_{entry.get('precision','bf16')}"
            table.gemm_efficiencies[key] = entry["efficiency"]
        for entry in d.get("attention_prefill", []):
            key = f"b{entry['batch']}_s{entry['seq_len']}_h{entry['n_heads']}_d{entry['d_head']}"
            if entry.get("latency_ms", 0) > 0:
                table.attention_latencies[key] = entry["latency_ms"]
        for entry in d.get("decode_kv", []):
            key = f"b{entry['batch']}_c{entry['context']}_kv{entry['n_kv_heads']}_d{entry['d_head']}_{entry.get('kv_dtype','bf16')}"
            if entry.get("latency_ms", 0) > 0:
                table.decode_kv_latencies[key] = entry["latency_ms"]
        for entry in d.get("all_reduce", []):
            key = f"tp{entry['tp']}_msg{entry['message_size_mb']}mb"
            if entry.get("latency_ms", 0) > 0:
                table.allreduce_latencies[key] = entry["latency_ms"]
        # Wave 7b.4: read the new Wave 1-5 fits. Two shapes are accepted:
        #   (a) Top-level `dp_grad_overlap_fraction: 0.65` etc. — the
        #       simplest case where the calibration JSON has been hand-merged
        #       per-hw from the auto_calibrate output.
        #   (b) Nested `wave5: {dp_grad_overlap: 0.65, tp_overlap: 0.42,
        #       pp_queue: {"1f1b": 1.05}, state_long_context_weight: 0.028,
        #       hbm_spill_factor: 1.10}` — what a copy of the auto_calibrate
        #       pack's per-hw section looks like before flattening.
        w5 = d.get("wave5", {}) if isinstance(d.get("wave5"), dict) else {}
        def _pick(key_top, key_w5):
            v = d.get(key_top, w5.get(key_w5))
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        table.dp_grad_overlap_fraction = _pick(
            "dp_grad_overlap_fraction", "dp_grad_overlap")
        table.tp_allreduce_overlap_fraction = _pick(
            "tp_allreduce_overlap_fraction", "tp_overlap")
        table.moe_alltoall_overlap_fraction = _pick(
            "moe_alltoall_overlap_fraction", "moe_a2a_overlap")
        table.state_long_context_weight = _pick(
            "state_long_context_weight", "state_long_context_weight")
        table.hbm_spill_decode_factor = _pick(
            "hbm_spill_decode_factor", "hbm_spill_factor")
        pq_top = d.get("pp_queue_multipliers")
        pq_w5 = w5.get("pp_queue")
        pq = pq_top if isinstance(pq_top, dict) else (
            pq_w5 if isinstance(pq_w5, dict) else {})
        # `pq` may itself be {"1f1b": 1.05} or {"1f1b": {"multiplier": 1.05}}
        for sched, val in pq.items():
            if isinstance(val, dict):
                v = val.get("multiplier")
            else:
                v = val
            try:
                if v is not None:
                    table.pp_queue_multipliers[str(sched).lower()] = float(v)
            except (TypeError, ValueError):
                continue
        return table

    def lookup_gemm_efficiency(self, M: int, N: int, K: int, precision: str) -> Optional[float]:
        """Find closest matching GEMM efficiency from calibration data."""
        cache_key = (int(M), int(N), int(K), str(precision))
        if cache_key in self._gemm_lookup_cache:
            return self._gemm_lookup_cache[cache_key]
        best_key = None
        best_dist = float("inf")
        for key, eff in self.gemm_efficiencies.items():
            parts = key.split("_")
            dims = parts[0].split("x")
            prec = parts[1] if len(parts) > 1 else "bf16"
            if prec != precision:
                continue
            m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
            dist = abs(math.log(max(M,1)/max(m,1))) + abs(math.log(max(N,1)/max(n,1))) + abs(math.log(max(K,1)/max(k,1)))
            if dist < best_dist:
                best_dist = dist
                best_key = key
        if best_key is not None and best_dist < 2.0:
            result = self.gemm_efficiencies[best_key]
        else:
            result = None
        self._gemm_lookup_cache[cache_key] = result
        return result


_CALIBRATION_CACHE: Dict[str, Optional[CalibrationTable]] = {}

# Wave 7b.4: module-level holder used by helpers that don't currently take a
# cal_table argument (e.g. `_allreduce_cost`, `estimate_memory_per_gpu`). The
# top of `throughput()` sets this to the per-hw calibration; the helpers
# read it via `_current_calibration()`. We deliberately do NOT make this a
# contextvar — the throughput model is fully synchronous and the existing
# code path is already single-threaded per call. A try/finally inside
# `throughput()` clears it on exit so the next call sees a clean slate.
_CURRENT_CALIBRATION: Optional[CalibrationTable] = None


def _current_calibration() -> Optional[CalibrationTable]:
    return _CURRENT_CALIBRATION


# =============================================================================
# Regime × arch-family efficiency table (v1-fix #27)
# =============================================================================
#
# Replaces the three scalar `*_system_efficiency` knobs with a
# (phase, regime, arch_family) → {mu, sigma} table. The regime is gated on
# the dominant cost from the per-layer breakdown's `bottleneck` field
# ("compute", "memory", "communication", "expert_load", "alltoall"); the
# family is read from the candidate (`dense`, `dense_gqa`, `moe`, `moe_mla`,
# `hybrid_state`, `mla_dense`). Defaults below come from public reports and
# the ±20% caveat in the README; they should be overridden by a real
# auto-calibrate fit per cluster. The sigma column is the missing piece —
# we propagate it through `ThroughputResult.*_sigma` so the contender filter
# in optimizer._contending_family can treat near-ties as ties.

_DEFAULT_EFFICIENCY_TABLE = {
    "training": {
        "dense":         {"compute": (0.55, 0.07), "memory": (0.42, 0.08),
                          "communication": (0.40, 0.10)},
        "dense_gqa":     {"compute": (0.55, 0.07), "memory": (0.42, 0.08),
                          "communication": (0.40, 0.10)},
        "mla_dense":     {"compute": (0.50, 0.09), "memory": (0.40, 0.10),
                          "communication": (0.38, 0.11)},
        "moe":           {"compute": (0.45, 0.10), "memory": (0.38, 0.10),
                          "communication": (0.32, 0.12), "alltoall": (0.28, 0.14),
                          "expert_load": (0.30, 0.12)},
        "moe_mla":       {"compute": (0.42, 0.12), "memory": (0.36, 0.12),
                          "communication": (0.30, 0.13), "alltoall": (0.26, 0.15),
                          "expert_load": (0.28, 0.13)},
        "hybrid_state":  {"compute": (0.48, 0.10), "memory": (0.40, 0.10),
                          "communication": (0.36, 0.12)},
    },
    "prefill": {
        "dense":         {"compute": (0.60, 0.06), "memory": (0.45, 0.09),
                          "communication": (0.42, 0.10)},
        "dense_gqa":     {"compute": (0.60, 0.06), "memory": (0.45, 0.09),
                          "communication": (0.42, 0.10)},
        "mla_dense":     {"compute": (0.55, 0.08), "memory": (0.42, 0.10),
                          "communication": (0.40, 0.11)},
        "moe":           {"compute": (0.50, 0.09), "memory": (0.40, 0.10),
                          "communication": (0.32, 0.12), "alltoall": (0.26, 0.15),
                          "expert_load": (0.30, 0.13)},
        "moe_mla":       {"compute": (0.46, 0.11), "memory": (0.38, 0.12),
                          "communication": (0.30, 0.13), "alltoall": (0.24, 0.16),
                          "expert_load": (0.28, 0.14)},
        "hybrid_state":  {"compute": (0.52, 0.10), "memory": (0.42, 0.10),
                          "communication": (0.38, 0.11)},
    },
    "decode": {
        # Decode is overwhelmingly memory-bound at small batch on dense
        # models; compute-bound regimes only appear at large batch.
        "dense":         {"compute": (0.45, 0.10), "memory": (0.42, 0.08),
                          "communication": (0.32, 0.12)},
        "dense_gqa":     {"compute": (0.45, 0.10), "memory": (0.42, 0.08),
                          "communication": (0.32, 0.12)},
        "mla_dense":     {"compute": (0.42, 0.11), "memory": (0.40, 0.10),
                          "communication": (0.30, 0.13)},
        # Wave 25 (2026-07): expert_load raised 0.20/0.18 → 0.42/0.38.
        # The expert_load term is a pure HBM weight stream, already priced
        # at datasheet bandwidth by _moe_ffn_cost; the overheads the old
        # 0.20 was meant to cover (kernel launch, scheduler dispatch, TP
        # allreduce, a2a) are ALL priced separately in the layer breakdown,
        # so the low value double-counted them and put every large-MoE
        # decode ~5× over roofline. Anchor evidence: the implied streaming
        # efficiency of the five MoE anchors' PUBLISHED TBTs spans
        # 0.29 (Mixtral, 2024-era vLLM) … 0.68 (Qwen3-235B, Alibaba
        # stack) — the old default sat BELOW the worst published stack,
        # biasing the whole family (mean +66% signed TBT error, Qwen3
        # +194%). 0.42 matches the dense memory-bound decode μ (same
        # physics — contiguous weight streaming); moe_mla keeps the same
        # −0.04 discount that mla_dense carries vs dense for the extra
        # latent-projection work. σ widened vs dense to reflect the
        # published-stack spread. Regenerate family_bias_v1.json
        # (ac-trust-audit --out) whenever these move.
        "moe":           {"compute": (0.36, 0.13), "memory": (0.34, 0.12),
                          "communication": (0.26, 0.14), "alltoall": (0.22, 0.16),
                          "expert_load": (0.42, 0.14)},
        "moe_mla":       {"compute": (0.34, 0.14), "memory": (0.32, 0.13),
                          "communication": (0.24, 0.15), "alltoall": (0.20, 0.17),
                          "expert_load": (0.38, 0.15)},
        "hybrid_state":  {"compute": (0.44, 0.10), "memory": (0.40, 0.10),
                          "communication": (0.30, 0.13)},
    },
}

# Map LayerBreakdown.bottleneck values into the regime axis.
_REGIME_ALIASES = {
    "compute": "compute",
    "memory": "memory",
    "communication": "communication",
    "comm": "communication",
    "alltoall": "alltoall",
    "expert_load": "expert_load",
}


def _arch_family(arch: ArchConfig) -> str:
    """Classify an arch into one of the calibrated families.

    Wave 18a: derived from the canonical ArchitectureSignature so this
    calibration-key taxonomy stays consistent with the user-visible
    ``legacy_family``. The mapping projects the 4-axis signature into the
    5-bucket calibration space:
      hybrid_state   ← any state mixer
      moe_mla        ← ffn_mode=moe   AND kv_projection=mla
      moe            ← ffn_mode=moe
      mla_dense      ← kv_projection=mla
      dense_gqa      ← kv_projection=gqa/mqa (all grouped attention shares a bucket)
      dense          ← kv_projection=mha (full MHA)
    """
    try:
        from ac.architecture import architecture_signature
        sig = architecture_signature(arch)
    except (ValueError, ImportError):
        # Defensive fallback — pre-Wave-18a inline classifier so partial
        # test fixtures without minimal shape fields still get a bucket.
        is_moe = arch.moe_config is not None
        is_mla = (arch.attention_type == "mla"
                  and int(getattr(arch, "mla_kv_latent_dim", 0) or 0) > 0)
        is_hybrid = (arch.layer_type_list is not None
                     and any(lt == "state" for lt in arch.layer_type_list)
                     and arch.state_config is not None)
        if is_hybrid:
            return "hybrid_state"
        if is_moe and is_mla:
            return "moe_mla"
        if is_moe:
            return "moe"
        if is_mla:
            return "mla_dense"
        if arch.n_kv_heads < arch.n_heads:
            return "dense_gqa"
        return "dense"

    if sig.has_state_mixer:
        return "hybrid_state"
    if sig.is_moe and sig.uses_mla:
        return "moe_mla"
    if sig.is_moe:
        return "moe"
    if sig.uses_mla:
        return "mla_dense"
    if sig.kv_projection in ("gqa", "mqa"):
        return "dense_gqa"
    return "dense"


def _efficiency_for(
    phase: str,
    bottleneck: str,
    arch: ArchConfig,
    cal: dict,
) -> tuple:
    """Return (mu, sigma, bucket_label) for a phase/regime/family lookup.

    Resolution order:
      1. cal["efficiency_table"][phase][family][regime]            # full lab fit
      2. cal["efficiency_table"][phase][family]["overall"]          # family fit
      3. _DEFAULT_EFFICIENCY_TABLE[phase][family][regime]           # default
      4. _DEFAULT_EFFICIENCY_TABLE[phase][family]["compute"]        # family default
      5. cal[f"{phase}_system_efficiency"] (legacy scalar)          # back-compat

    A hardware-wide multiplicative override may also be applied. When a spec's
    `calibration.efficiency_multipliers[phase]` is present the resolved `mu` is
    scaled by it; this lets `ac-auto-calibrate` flow a cluster-wide
    observed/predicted ratio into every family/regime cell without having to
    populate a per-family lab table. Multiplier is clamped to [0.1, 1.5] and
    the bucket label is annotated with `*mult=<x>` so downstream reports can
    tell when calibration was applied.
    """
    regime = _REGIME_ALIASES.get(str(bottleneck or "compute"), "compute")
    family = _arch_family(arch)

    def _unpack(entry):
        if isinstance(entry, dict):
            mu = float(entry.get("mu", entry.get("efficiency", 0.0)))
            sigma = float(entry.get("sigma", 0.0))
            return mu, sigma
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return float(entry[0]), float(entry[1])
        if isinstance(entry, (int, float)):
            return float(entry), 0.0
        return None

    bucket = None
    mu = sigma = None

    lab_table = (cal or {}).get("efficiency_table") or {}
    lab_phase = lab_table.get(phase) or {}
    lab_family = lab_phase.get(family) or {}
    if regime in lab_family:
        mu_sigma = _unpack(lab_family[regime])
        if mu_sigma is not None:
            mu, sigma = mu_sigma
            bucket = f"lab:{family}/{regime}"
    if mu is None and "overall" in lab_family:
        mu_sigma = _unpack(lab_family["overall"])
        if mu_sigma is not None:
            mu, sigma = mu_sigma
            bucket = f"lab:{family}/overall"

    if mu is None:
        default_phase = _DEFAULT_EFFICIENCY_TABLE.get(phase, {})
        default_family = default_phase.get(family) or default_phase.get("dense_gqa", {})
        if regime in default_family:
            mu, sigma = default_family[regime]
            bucket = f"default:{family}/{regime}"
        elif "compute" in default_family:
            mu, sigma = default_family["compute"]
            bucket = f"default:{family}/compute"

    if mu is None:
        # Legacy scalar fallback so unmigrated hardware specs still work.
        legacy = float(cal.get(f"{phase}_system_efficiency", 0.5))
        mu, sigma, bucket = legacy, 0.15 * legacy, "legacy_scalar"

    # Hardware-wide multiplicative override (fix #1). When a spec carries a
    # calibration.efficiency_multipliers[phase] entry (written by
    # ac-auto-calibrate from the cluster's observed/predicted ratio), apply it
    # uniformly. This is the single most important path for lab calibration
    # because most sparse measurement files cannot populate a per-family table.
    # Two-layer multiplier composition.
    #
    # `efficiency_multipliers` is the *base* multiplier shipped with the
    # spec — it reflects the gap between the per-layer compute model (which
    # is intentionally ~1.5-2× conservative vs roofline) and what a
    # well-tuned production stack achieves. AC ships sensible defaults
    # (e.g. H100 BF16 training = 1.8×) so out-of-the-box predictions land
    # in the published-benchmark range.
    #
    # `calibration_efficiency_multipliers` is the *lab-specific* multiplier
    # written by `ac-auto-calibrate fit` from observed/predicted ratios.
    # It multiplies the base, rather than replacing it, so a lab's local
    # measurement gap (e.g. their cluster runs 0.9× the shipped default)
    # composes correctly with the modern-baseline default. Without this
    # split, auto-calibrate's small-sample fit silently overwrote the
    # modern baseline and pushed predictions back to a 2× pessimistic
    # regime.
    base_mults = (cal or {}).get("efficiency_multipliers") or {}
    cal_mults = (cal or {}).get("calibration_efficiency_multipliers") or {}

    def _coerce(v, fallback=1.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return fallback

    base_mult = _coerce(base_mults.get(phase, 1.0), 1.0)
    cal_mult = _coerce(cal_mults.get(phase, 1.0), 1.0)
    # Floor base at 0.1 (sanity); cap composed product at 3.0 so a tiny-
    # sample observed/predicted ratio can't pile a 2× on top of a 2× base.
    base_mult = max(0.1, min(2.5, base_mult))
    cal_mult = max(0.1, min(2.5, cal_mult))
    composed = max(0.05, min(3.0, base_mult * cal_mult))
    if abs(composed - 1.0) > 1e-6:
        mu = mu * composed
        sigma = sigma * composed
        if abs(base_mult - 1.0) > 1e-6 and abs(cal_mult - 1.0) > 1e-6:
            bucket = f"{bucket}*base={base_mult:.3f}*cal={cal_mult:.3f}"
        elif abs(base_mult - 1.0) > 1e-6:
            bucket = f"{bucket}*base={base_mult:.3f}"
        elif abs(cal_mult - 1.0) > 1e-6:
            bucket = f"{bucket}*cal={cal_mult:.3f}"

    return mu, sigma, bucket


def _mixed_efficiency_for(
    phase: str,
    breakdowns: List[LayerBreakdown],
    arch: ArchConfig,
    cal: dict,
) -> tuple:
    """Blend calibrated efficiency continuously across layer cost classes.

    A winner-take-all bottleneck lookup creates a discontinuity: shaving one
    microsecond from compute can relabel a layer ``communication`` and apply a
    lower efficiency to *all* work, including fixed launch overhead. The
    reported latency can then increase after a real speedup. Calibrate each
    additive cost class and combine their exposed times instead.
    """
    totals = {
        "compute": sum(max(0.0, bd.compute_s) for bd in breakdowns),
        "memory": sum(max(0.0, bd.memory_s) for bd in breakdowns),
        "communication": sum(
            max(0.0, bd.communication_s) for bd in breakdowns
        ),
    }
    raw_total = sum(totals.values())
    if raw_total <= 0:
        return _efficiency_for(phase, "compute", arch, cal)

    exposed_total = 0.0
    variance = 0.0
    labels = []
    for regime, raw_time in totals.items():
        if raw_time <= 0:
            continue
        mu, sigma, label = _efficiency_for(phase, regime, arch, cal)
        mu = max(mu, 1e-9)
        exposed = raw_time / mu
        exposed_total += exposed
        variance += (exposed * sigma / mu) ** 2
        short_label = {
            "compute": "compute",
            "memory": "memory",
            "communication": "comm",
        }[regime]
        labels.append(f"{short_label}={raw_time / raw_total:.2f}")

    effective_mu = raw_total / max(exposed_total, 1e-12)
    relative_sigma = math.sqrt(variance) / max(exposed_total, 1e-12)
    effective_sigma = effective_mu * relative_sigma
    return effective_mu, effective_sigma, "mix:" + ",".join(labels)

def load_calibration(hw_name: str) -> Optional[CalibrationTable]:
    """Load calibration table for a hardware target, if available.

    Gate-2 Task C: `_HARDWARE_CALIBRATION_ALIAS` is now honored here (not
    only in `warn_if_uncalibrated`) so same-silicon system targets
    (gb200_nvl72 → b200, h800 → h100) share the measured table of their
    chip. Existing targets are unaffected: every pre-existing alias
    (trn2/trn3) points at a canonical name with no calibration file on
    disk, which still returns None exactly as before.
    """
    canonical = _HARDWARE_CALIBRATION_ALIAS.get(hw_name, hw_name)
    if canonical in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[canonical]
    path = os.path.join(_CALIBRATION_DIR, f"{canonical}_calibration.json")
    if os.path.exists(path):
        table = CalibrationTable.from_json(path)
        _CALIBRATION_CACHE[canonical] = table
        return table
    _CALIBRATION_CACHE[canonical] = None
    return None


# Hardware aliases that share a calibration table (alias -> canonical name
# used for the calibration file lookup). Keep in sync with `load_hardware`.
_HARDWARE_CALIBRATION_ALIAS: Dict[str, str] = {
    "trn2": "trainium2",
    "trn3": "trainium3",
    # Gate-2 Task C: system targets share the measured calibration table of
    # their chip silicon (compute-side efficiency transfers; the bandwidth
    # differences live in the spec JSON interconnect fields).
    "gb200_nvl72": "b200",
    "h800": "h100",
}

# Track which hardware names we've already warned about in this process so
# the user sees at most one WARNING per (CLI invocation, hardware name).
_CALIBRATION_WARNED: set = set()


def warn_if_uncalibrated(hw_name: str, stream=None) -> bool:
    """Emit a one-shot WARNING if `hw_name` has no calibration table on disk.

    Returns True if a warning was emitted, False otherwise. The warning is
    de-duplicated per-process via `_CALIBRATION_WARNED`. The throughput
    model still runs (with default efficiency multipliers); this just
    surfaces the fact that the user is on uncalibrated priors so the
    headline TPS / TBT numbers can be read with appropriate skepticism.
    """
    import sys as _sys
    canonical = _HARDWARE_CALIBRATION_ALIAS.get(hw_name, hw_name)
    if canonical in _CALIBRATION_WARNED:
        return False
    if load_calibration(canonical) is not None:
        return False
    _CALIBRATION_WARNED.add(canonical)
    out = stream if stream is not None else _sys.stderr
    print(
        f"WARNING: hardware `{hw_name}` has no calibration table "
        f"(searched {os.path.join(_CALIBRATION_DIR, canonical + '_calibration.json')}). "
        f"Throughput predictions will use AC's default efficiency "
        f"multipliers. Run `ac-auto-calibrate fit --measurements "
        f"<traces>.jsonl` to fit lab-local efficiency. Treat absolute "
        f"TPS/TBT/loss numbers as uncalibrated priors until then.",
        file=out,
    )
    return True


def load_hardware(name: str) -> HardwareConfig:
    """Load a hardware config by short name.

    v1-fix Trainium: AWS Trainium 2 / Trainium 3 added as fourth-vendor
    hardware targets. Trn2 ships today (re:Invent 2024); Trn3 numbers are
    public-estimate from the AWS roadmap.
    """
    mapping = {
        "h100": "h100_sxm.json",
        "b200": "b200.json",
        "gb200_nvl72": "gb200_nvl72.json",  # Gate-2 Task C: rack-scale system target
        "h800": "h800.json",                # Gate-2 Task C: H100 export SKU (NVLink 400 GB/s)
        "tpu_v5e": "tpu_v5e.json",
        "tpu_v5p": "tpu_v5p.json",
        "trainium2": "trainium2.json",
        "trn2": "trainium2.json",          # alias
        "trainium3": "trainium3.json",
        "trn3": "trainium3.json",          # alias
    }
    filename = mapping.get(name)
    if filename is None:
        raise ValueError(f"Unknown hardware: {name}. Supported: {list(mapping.keys())}")
    return HardwareConfig.from_json(os.path.join(_SPEC_DIR, filename))


# =============================================================================
# Lattice integration — tile efficiency lookup
# =============================================================================

def get_tile_efficiency(
    M: int, N: int, K: int,
    precision: str,
    lattice_hw: LatticeHardwareSpec,
) -> float:
    """
    Query the lattice for tile efficiency of a matmul (M, N, K) at given precision.
    Returns combined tile utilization × wave efficiency as the effective efficiency.
    """
    if precision not in lattice_hw.tiles:
        # Fall back to bf16 if precision not in lattice
        precision = "bf16"
    if precision not in lattice_hw.tiles:
        return 0.8  # conservative fallback

    tile = lattice_hw.tiles[precision]
    tile_util = matmul_tile_utilization(M, N, K, tile)
    wave_eff = wave_efficiency(M, N, tile, lattice_hw.n_sms)
    return tile_util * wave_eff


_CALIBRATION_PRECISION_ALIASES = {
    # These recipes use the same Blackwell tensor-core kernel families; their
    # block-scale metadata changes storage/numerics, not the CTA geometry.
    "mxfp8": "fp8",
    "mxfp6": "fp8",
    "nvfp4": "fp4",
    "mxfp4": "fp4",
}


# =============================================================================
# Per-operation cost functions
# =============================================================================

def _matmul_cost(
    M: int, N: int, K: int,
    precision: str,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    calibration: Optional[CalibrationTable] = None,
) -> Tuple[float, float, str]:
    """
    Roofline cost for a single matmul.
    Returns (time_s, flops, bottleneck_type).

    If a calibration table is provided and has a matching GEMM shape,
    uses measured efficiency instead of analytic tile efficiency.
    """
    bpe = hw.bytes_per_elem(precision)
    flops = 2 * M * N * K

    # Try calibration first, fall back to lattice tile efficiency
    eff = None
    if calibration is not None:
        calibration_precision = precision
        if precision in hw.supported_precisions:
            calibration_precision = _CALIBRATION_PRECISION_ALIASES.get(
                precision, precision
            )
        eff = calibration.lookup_gemm_efficiency(
            M, N, K, calibration_precision
        )
    if eff is None:
        eff = get_tile_efficiency(M, N, K, precision, lattice_hw)
    eff = max(eff, 0.1)  # floor to avoid division by zero

    t_compute = flops / (hw.peak_flops_s(precision) * eff)

    # Memory traffic: weights + activations
    weight_bytes = N * K * bpe
    act_bytes = (M * K + M * N) * bpe
    t_memory = (weight_bytes + act_bytes) / hw.hbm_bandwidth_bytes_s

    t = max(t_compute, t_memory)
    bottleneck = "compute" if t_compute >= t_memory else "memory"
    return t, flops, bottleneck


def _attention_cost(
    B: int, S: int, n_heads: int, d_head: int, n_kv_heads: int,
    precision: str,
    hw: HardwareConfig,
    tp_degree: int = 1,
) -> float:
    """
    Fused attention cost (FlashAttention / Splash Attention).
    Uses analytical IO model from Dao 2022.
    """
    heads_per_gpu = n_heads // tp_degree
    kv_heads_per_gpu = max(1, math.ceil(n_kv_heads / max(1, tp_degree)))
    bpe = hw.bytes_per_elem(precision)

    # Compute: QK^T and softmax×V
    # With GQA, each KV head serves (n_heads / n_kv_heads) query heads
    flops_attn = 2 * B * heads_per_gpu * S * S * d_head * 2

    # HBM traffic under fusion: Q, K, V read + O written (no S×S materialization)
    hbm_bytes = B * S * d_head * bpe * (heads_per_gpu + 2 * kv_heads_per_gpu + heads_per_gpu)

    fused_eff = hw.fused_attention_efficiency.get(precision,
                hw.fused_attention_efficiency.get("bf16", 0.75))

    # v1-fix demo-audit: d_head-aware efficiency. Hopper/Blackwell tensor
    # cores ingest 128-element MMA tiles along the K dimension. d_head=128
    # fills exactly one tile and hits peak fused-attention efficiency. At
    # d_head=64, FlashAttention pads to a half-tile and incurs ~30%
    # throughput loss on H100 (measured: Dao et al. FA2 paper Table 5;
    # Llama2-vs-Llama3 head-dim ablations). d_head=256 is slightly less
    # efficient than 128 because of register pressure. We model the
    # multiplier as a piecewise factor anchored at d_head=128 = 1.0.
    if d_head <= 32:
        d_head_eff = 0.55
    elif d_head <= 48:
        d_head_eff = 0.62
    elif d_head <= 64:
        d_head_eff = 0.70
    elif d_head <= 96:
        d_head_eff = 0.88
    elif d_head <= 128:
        d_head_eff = 1.00
    elif d_head <= 192:
        d_head_eff = 0.92
    else:  # >=256, register-pressure regime
        d_head_eff = 0.82
    fused_eff = fused_eff * d_head_eff

    t_compute = flops_attn / (hw.peak_flops_s(precision) * fused_eff)
    t_memory = hbm_bytes / hw.hbm_bandwidth_bytes_s
    return max(t_compute, t_memory)


def _sparse_attention_cost(
    B: int,
    query_len: int,
    attended_len: int,
    n_heads: int,
    d_head: int,
    n_kv_heads: int,
    precision: str,
    hw: HardwareConfig,
    tp_degree: int = 1,
) -> float:
    """Fused rectangular attention cost for sparse/compressed patterns."""
    heads_per_gpu = n_heads // tp_degree
    kv_heads_per_gpu = max(
        1, math.ceil(n_kv_heads / max(1, tp_degree))
    )
    bpe = hw.bytes_per_elem(precision)
    q = max(1, int(query_len))
    k = max(1, min(q, int(attended_len)))
    flops = 4 * B * heads_per_gpu * q * k * d_head
    hbm_bytes = B * d_head * bpe * (
        2 * heads_per_gpu * q + 2 * kv_heads_per_gpu * k
    )
    fused_eff = hw.fused_attention_efficiency.get(
        precision, hw.fused_attention_efficiency.get("bf16", 0.75)
    )
    return max(
        flops / max(1.0, hw.peak_flops_s(precision) * fused_eff),
        hbm_bytes / max(1.0, hw.hbm_bandwidth_bytes_s),
    )


def _membound_ops_cost(
    B: int, S: int, d_model: int,
    precision: str,
    hw: HardwareConfig,
    n_norms: int = 3,  # pre-attn norm, pre-FFN norm, post-FFN norm/residual
) -> float:
    """
    Cost of memory-bound operations per layer: norms, residuals, activations.
    These are bandwidth-bound — just data movement.
    """
    bpe = hw.bytes_per_elem(precision)
    # Each norm: read + write = 2 × B × S × d_model
    # Residual connections: similar traffic
    # Activation functions: read + write
    total_bytes = n_norms * 2 * B * S * d_model * bpe
    return total_bytes / hw.hbm_bandwidth_bytes_s


def _allreduce_cost(
    B: int, S: int, d_model: int,
    precision: str,
    hw: HardwareConfig,
    tp_degree: int,
    n_allreduces: int = 2,  # one after attention, one after FFN
    overlap_fraction: float = 0.5,
    phase: str = "training",
) -> float:
    """
    TP all-reduce cost per layer.
    GPU: NVLink ring/tree. TPU: ICI.

    v1-fix Wave 1 Step 1.3 (Jun 2026): now accepts an overlap_fraction so
    Megatron-style async TP and cuBLAS-stream overlap can be modelled.
    The v0 cost assumed serial exposed comm (`overlap_fraction=0`), which
    over-counts TP comm by ~1.4-2× for matmul-bound layers. Public
    reports from Megatron-LM and DeepSpeed put overlap at 0.4-0.7 across
    common shapes; 0.5 is a conservative middle. Set to 0.0 to recover
    the v0 behavior. Caller can pass a smaller overlap when the
    collective follows another collective on the same fabric (e.g. EP
    alltoall → shared-expert allreduce; see Step 1.6).

    Wave 17 Part 2 fix (Jun 2026): two refinements that the original
    formula missed at decode:
      - `phase="decode"` uses overlap_fraction=0.1 (default 0.5 hides
        all-reduce behind matmul, which only applies during training/
        prefill; decode is memory-bound with nothing to hide behind).
        Explicit caller-supplied non-default overlap_fraction still wins.
      - Add per-collective launch-latency floor. NCCL/NVSHMEM round-trip
        latency is ~5-8 µs intra-NVLink, ~12-15 µs cross-IB. Without this
        floor, decode all-reduces at TP=32 with tiny (S=1) payloads round
        to near-zero in the bandwidth-only formula and the throughput
        model under-counts the per-token cost by 1-3 ms.

    See plan/redesign/17-pp-decode-serving-cost-fix.md Part 2 + the
    superseded 17-tp-amortization-cost-fix.md plan for full context.
    """
    if tp_degree <= 1:
        return 0.0

    bpe = hw.bytes_per_elem(precision)
    per_allreduce_bytes = 2 * B * S * d_model * bpe

    link_bw = hw.interconnect_bw_bytes_s(tp_degree)

    if hw.vendor == "nvidia":
        # Ring all-reduce: 2 × (P-1)/P × bytes / BW
        t_per_bw = 2 * (tp_degree - 1) / tp_degree * per_allreduce_bytes / link_bw
    else:
        # TPU ICI: simpler model — 2 × bytes / effective_BW
        t_per_bw = 2 * per_allreduce_bytes / link_bw

    # Wave 17 Part 2: per-collective launch latency floor. Inter-node hops
    # add ~12-15 µs round-trip; intra-NVLink adds ~5-8 µs. At decode where
    # payload (B × 1 × d_model) is tiny, this is the dominant cost.
    local_domain = (
        hw.nvlink_domain_size
        if hw.vendor == "nvidia" and hw.nvlink_domain_size
        else hw.gpus_per_node
        if hw.vendor == "nvidia"
        else hw.chips_per_host
    )
    if tp_degree <= local_domain:
        per_ar_latency_s = 6e-6   # intra-NVLink NCCL launch round-trip
    else:
        per_ar_latency_s = 14e-6  # cross-IB NCCL launch round-trip
    t_per = t_per_bw + per_ar_latency_s

    # Wave 7b.4: when a per-hw calibration provides a fitted TP overlap, use
    # it instead of the function's default. The override only fires if the
    # caller didn't pass an explicit non-default overlap (else the explicit
    # arg wins). `0.5` is the Wave 1 Step 1.3 default; we detect "user didn't
    # override" by comparing against it. This keeps callers that explicitly
    # tune overlap (e.g. the EP→AR chain in Step 1.6) unaffected.
    if abs(overlap_fraction - 0.5) < 1e-9:
        # Wave 17 Part 2: decode has no big matmul to overlap with —
        # bandwidth-bound autoregressive token gen. Drop overlap to 0.1.
        # Training/prefill keep the 0.5 default.
        if phase == "decode":
            overlap_fraction = 0.1
        _cal = _current_calibration()
        if _cal is not None and _cal.tp_allreduce_overlap_fraction is not None:
            overlap_fraction = float(_cal.tp_allreduce_overlap_fraction)

    exposed = max(0.0, 1.0 - overlap_fraction)
    return exposed * n_allreduces * t_per


def _pipeline_link_costs(
    payload_bytes: float,
    pp_degree: int,
    hw: HardwareConfig,
) -> Tuple[float, float]:
    """Return (one_boundary_worst_case, full_pipeline_path) transfer cost.

    Ranks are assumed contiguous within a local NVLink/ICI domain. Most stage
    boundaries are local; only boundaries crossing a domain use inter-node
    fabric. Training throughput pays the slowest boundary once per direction,
    while prefill/decode latency pays every boundary on the path.
    """
    if pp_degree <= 1:
        return 0.0, 0.0
    local_domain = int(
        hw.nvlink_domain_size
        if hw.vendor == "nvidia" and hw.nvlink_domain_size
        else hw.gpus_per_node
        if hw.vendor == "nvidia"
        else hw.chips_per_host
    )
    local_boundaries = pp_degree - 1
    cross_boundaries = 0
    if pp_degree > local_domain:
        cross_boundaries = (pp_degree - 1) // max(1, local_domain)
        local_boundaries -= cross_boundaries
    local_cost = (
        payload_bytes
        / max(1.0, hw.interconnect["intra_node_bw_gb_s"] * 1e9)
        + 5e-6
    )
    cross_cost = (
        payload_bytes
        / max(1.0, hw.interconnect["inter_node_bw_gb_s"] * 1e9)
        + 12e-6
    )
    worst = cross_cost if cross_boundaries else local_cost
    full_path = local_boundaries * local_cost + cross_boundaries * cross_cost
    return worst, full_path


# =============================================================================
# DP gradient allreduce (v1-fix Wave 1 Step 1.2, Jun 2026)
# =============================================================================

def _dp_grad_allreduce_cost(
    total_params: int,
    precision: str,
    hw: HardwareConfig,
    dp_degree: int,
    zero_stage: int = 3,
    overlap_fraction: float = 0.7,
) -> float:
    """Time for the DP gradient sync per training step.

    At dp_degree=1 the cost is zero (no sync). Beyond that:
      - ZeRO-0/1: full gradient allreduce, 2 × (dp-1)/dp × params × bpe.
      - ZeRO-2/3 + FSDP: reduce-scatter (grad) + all-gather (weight before
        next forward), same total byte volume on the wire because the
        reduce-scatter halves the bytes per rank but doubles the number of
        collectives in a step.

    Intra-node uses NVLink/ICI; once dp_degree exceeds gpus_per_node we
    saturate the inter-node fabric (IB/RoCE), which on most clusters is
    50-200 GB/s vs the 600-1800 GB/s intra-node link — i.e. the dp cost
    grows sharply at the node boundary.

    overlap_fraction (default 0.7): modern frameworks (PyTorch FSDP,
    DeepSpeed ZeRO-3, Megatron) launch the reduce-scatter for layer N
    while still doing the backward of layer N-1, hiding most of the sync
    behind compute. Empirical values from PyTorch FSDP blog posts and
    DeepSpeed ZeRO-3 reports are 65-90%; 0.7 is a conservative middle.
    The exposed cost is (1 - overlap_fraction) × raw_sync.

    This term is missing from the v0 throughput model, which silently
    assumed perfect DP scaling. Without it, `aggregate_tps = per_replica
    × dp_degree` overcounts large-cluster throughput by ~1.2-3× at
    dp ≥ 256, depending on params and fabric.
    """
    bpe = hw.bytes_per_elem(precision)
    return _dp_grad_allreduce_bytes_cost(
        total_params * bpe,
        hw,
        dp_degree,
        zero_stage=zero_stage,
        overlap_fraction=overlap_fraction,
    )


def _dp_grad_allreduce_bytes_cost(
    grad_bytes: float,
    hw: HardwareConfig,
    dp_degree: int,
    zero_stage: int = 3,
    overlap_fraction: float = 0.7,
) -> float:
    """DP gradient synchronization cost from already-typed tensor bytes."""
    if dp_degree <= 1:
        return 0.0
    intra_bw = hw.interconnect["intra_node_bw_gb_s"] * 1e9
    inter_bw = hw.interconnect["inter_node_bw_gb_s"] * 1e9
    per_node = hw.gpus_per_node if hw.vendor == "nvidia" else hw.chips_per_host
    bw = intra_bw if dp_degree <= per_node else inter_bw
    factor = 2.0 if zero_stage >= 1 else 1.0
    raw = factor * (dp_degree - 1) / dp_degree * grad_bytes / max(bw, 1.0)
    exposed = max(0.0, 1.0 - overlap_fraction)
    return exposed * raw


# =============================================================================
# v1 MoE cost helpers
# =============================================================================

def _nvlink_domain_size(hw: HardwareConfig) -> int:
    """How many ranks share full-bandwidth NVLink (or single-axis ICI).

    Prefers the explicit `nvlink_domain_size` field from the hardware spec JSON
    (added in v1-fix part G so DGX-B200 vs NVL72 can be distinguished without
    code edits). Falls back to family-based inference for older specs that
    haven't been migrated.
    """
    if hw.nvlink_domain_size is not None and hw.nvlink_domain_size > 0:
        return hw.nvlink_domain_size
    fam = hw.accelerator_family.lower()
    if "blackwell" in fam:
        return 72       # legacy fallback: assume NVL72; DGX-B200 should set spec field to 8
    if "hopper" in fam:
        return 8        # DGX/HGX H100
    if fam.startswith("tpu"):
        return 16       # single ICI torus axis
    return hw.gpus_per_node


def _moe_alltoall_cost(
    volume_bytes: float,
    ep_degree: int,
    hw: HardwareConfig,
    ep_topology: str = "single_axis",
) -> float:
    """Time for the two all-to-alls in one MoE layer (dispatch + combine).

    volume_bytes is the *total* per-layer volume (both all-to-alls included).

    NVLink (NVIDIA): ring all-to-all at ~67% of peak link BW within the NVLink
    domain (8 on H100, 72 on B200 NVL72). Beyond the domain, the inter-node
    fabric is used and is dramatically slower; the formula below uses the
    inter-node BW directly because it is already much smaller than NVLink,
    so the cross-domain penalty is captured by the BW switch.

    TPU ICI 3D torus: effective single-axis BW at ~80% of peak ICI BW. If
    ep_topology=='cross_axis', traversal across torus axes inflates cost by
    a factor of 2.5 (default). The optimizer should prefer single-axis EP.
    """
    if ep_degree <= 1:
        # EP=1 still pays a small dispatch/combine kernel cost even though
        # there's no cross-rank traffic. Approximate as 1 us per layer.
        return 1e-6

    if hw.vendor == "nvidia":
        nvlink_domain = _nvlink_domain_size(hw)
        off_rank = (ep_degree - 1) / ep_degree
        if ep_degree <= nvlink_domain:
            link_bw = hw.interconnect["intra_node_bw_gb_s"] * 1e9
            effective_bw = 0.67 * link_bw  # ring all-to-all efficiency
            return off_rank * volume_bytes / max(effective_bw, 1.0)
        # Wave 19 (P0-1): hierarchical all-to-all beyond the NVLink domain.
        # The old model switched ALL volume to inter-node bandwidth once
        # ep > domain, but with uniform routing only (ep - domain)/(ep - 1)
        # of the off-rank destinations are on other nodes; the intra-node
        # share still rides NVLink, and the two legs use different fabrics
        # concurrently (take the max, not the sum).
        #
        # Node-limited routing (DeepSeek-V3 §3.4 "restrict each token to at
        # most M nodes", Qwen3-MoE similar): the cross-node leg sends each
        # token ONCE per destination node (then fans out over NVLink), so
        # its volume scales with min(top_k, node_limit)/top_k. The volume
        # passed in here is top_k-proportional; apply the dedup as a
        # discount. node_limit=4 matches published practice and is a
        # calibration surface, not a claim.
        n_peers = ep_degree - 1
        f_intra = (nvlink_domain - 1) / n_peers
        f_inter = 1.0 - f_intra
        node_limit = 4.0
        # top_k is not visible here; the dedup ratio is folded in by the
        # caller via ep_topology="node_limited:<ratio>" — default assume
        # top_k=8-class routing → ratio 0.5 when unspecified.
        dedup = 0.5
        if isinstance(ep_topology, str) and ep_topology.startswith("node_limited:"):
            try:
                dedup = min(1.0, max(0.05, float(ep_topology.split(":", 1)[1])))
            except ValueError:
                pass
        intra_bw = 0.67 * hw.interconnect["intra_node_bw_gb_s"] * 1e9
        inter_bw = 0.67 * hw.interconnect["inter_node_bw_gb_s"] * 1e9
        t_intra = off_rank * volume_bytes * f_intra / max(intra_bw, 1.0)
        t_inter = off_rank * volume_bytes * f_inter * dedup / max(inter_bw, 1.0)
        return max(t_intra, t_inter)

    # TPU (or any non-NVIDIA): 3D torus / ICI
    link_bw = hw.interconnect["intra_node_bw_gb_s"] * 1e9  # v5p: intra == inter
    if ep_topology == "single_axis":
        effective_bw = 0.8 * link_bw
    else:
        effective_bw = 0.8 * link_bw / 2.5  # cross-axis penalty
    return (ep_degree - 1) / ep_degree * volume_bytes / max(effective_bw, 1.0)


def _moe_ffn_cost(
    B: int, S: int, d_model: int,
    moe_cfg: dict,
    ep_degree: int,
    tp_degree: int,
    activation_precision: str,
    expert_precision: str,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    phase: str,                          # "training" | "prefill" | "decode"
    calibration: Optional[CalibrationTable] = None,
    ep_topology: str = "single_axis",
    imbalance: float = 1.0,
) -> Dict[str, float]:
    """Compute the per-layer FFN time for an MoE layer.

    Returns a dict with keys:
      compute_s        — sum of expert and shared-expert matmul times
      shared_expert_s  — shared-expert portion (subset of compute_s)
      alltoall_s       — dispatch + combine communication time
      expert_load_s    — decode-phase HBM bandwidth to stream top_k experts
                          (= 0 for training/prefill where weights are reused)
      load_balance_factor — imbalance multiplier actually applied

    The dense-equivalent FFN matmul cost is *replaced* by this function when
    arch.moe_config is set; the caller should not add the dense ffn_up_s and
    ffn_down_s terms in that case.
    """
    n_experts = int(moe_cfg["n_experts"])
    top_k = int(moe_cfg["top_k"])
    expert_dim = int(moe_cfg["expert_dim"])
    capacity = float(moe_cfg.get("capacity_factor", 1.0))
    shared_block = moe_cfg.get("shared_expert")
    shared_dim = int(shared_block["ffn_dim"]) if shared_block else 0

    bpe_act = hw.bytes_per_elem(activation_precision)
    bpe_exp = hw.bytes_per_elem(expert_precision)

    expert_dim_per_rank = max(expert_dim // max(tp_degree, 1), 1)
    shared_dim_per_rank = max(shared_dim // max(tp_degree, 1), 1) if shared_dim else 0

    # --- Compute: per-rank tokens routed to local experts ---
    # Wave 19 (P0-1): EP semantics are PHASE-DEPENDENT.
    #
    # TRAINING — EP lays over the DP dimension (Megatron/DeepSpeed-MoE/
    # DeepSeek layout): every EP rank carries its OWN microbatch of M
    # tokens. Total (token, expert) assignments in the EP group are
    # M × ep × top_k; the rank owns n_experts/ep experts, so balanced
    # routing lands M × top_k assignments on it. The old formula
    # (M × top_k / ep) modeled EP ranks as pure parameter servers sharing
    # ONE microbatch — each rank looked ep× under-utilized, and per-GPU
    # MoE training throughput came out ~ep× too low (the dominant factor
    # in the "MoE 20× slower than dense" release-review finding).
    #
    # SERVING (prefill/decode) — an inference instance genuinely spans
    # tp × ep GPUs sharing one batch (vLLM/SGLang EP layout), so the
    # M × top_k / ep spread is correct there and is retained.
    S_eff = S if phase != "decode" else 1
    M = B * S_eff

    if phase == "training":
        tokens_per_rank = max(1, int(M * top_k * capacity))
    else:
        tokens_per_rank = max(1, int(M * top_k * capacity / max(ep_degree, 1)))
    tokens_per_rank = int(tokens_per_rank * imbalance)

    # SwiGLU expert: three matmuls (up, gate, down) per active expert per token.
    # In this analytic model we charge the *aggregate* matmul shape at the
    # tokens_per_rank batch; this implicitly assumes routed tokens are
    # contiguous-batched on each rank (the standard EP implementation).
    t_up, _, _ = _matmul_cost(
        tokens_per_rank, expert_dim_per_rank, d_model,
        expert_precision, hw, lattice_hw, tp_degree, calibration,
    )
    t_gate, _, _ = _matmul_cost(
        tokens_per_rank, expert_dim_per_rank, d_model,
        expert_precision, hw, lattice_hw, tp_degree, calibration,
    )
    t_down, _, _ = _matmul_cost(
        tokens_per_rank, d_model, expert_dim_per_rank,
        expert_precision, hw, lattice_hw, tp_degree, calibration,
    )
    expert_compute_s = t_up + t_gate + t_down

    # Shared expert (DeepSeek-style): always-on, sees every LOCAL token.
    # Replicated across EP; sharded across TP. Wave 19 (P0-1): the local
    # token count is phase-dependent (training: per-rank microbatch M;
    # serving: the instance batch is spread across EP ranks, M/ep each) —
    # consistent with the routed-token accounting above.
    shared_compute_s = 0.0
    if shared_dim_per_rank > 0:
        M_shared = max(1, int(M if phase == "training"
                              else M / max(ep_degree, 1)))
        t_su, _, _ = _matmul_cost(
            M_shared, shared_dim_per_rank, d_model,
            expert_precision, hw, lattice_hw, tp_degree, calibration,
        )
        t_sg, _, _ = _matmul_cost(
            M_shared, shared_dim_per_rank, d_model,
            expert_precision, hw, lattice_hw, tp_degree, calibration,
        )
        t_sd, _, _ = _matmul_cost(
            M_shared, d_model, shared_dim_per_rank,
            expert_precision, hw, lattice_hw, tp_degree, calibration,
        )
        shared_compute_s = t_su + t_sg + t_sd

    compute_s = expert_compute_s + shared_compute_s

    # --- Decode-phase expert weight loading ---
    # At decode, each token activates top_k experts. Their weights must be
    # streamed from HBM. This is the *bandwidth* term that makes MoE
    # economical for serving relative to a same-total-param dense.
    #
    # Wave 18e/19 fix (2026-07): the previous formula counted `B × top_k`
    # expert-slot loads independently, which missed the batch-level
    # weight-reuse: when multiple tokens in a decode batch route to the
    # SAME expert, that expert's weights are streamed once and reused
    # across all those tokens. The number of DISTINCT experts touched by
    # a decode batch of size B with top_k routing is bounded by
    # `min(B × top_k, n_experts)`. For B=16, top_k=4, n_experts=32 the
    # old formula counted 64 expert-slots; reality touches at most 32.
    # The 2× over-count was the dominant contributor to Qwen3-235B-A22B
    # TBT blowing up from +45% to +1076% error against published data.
    #
    # Per-rank version: each rank owns `n_experts / ep_degree` experts.
    # The batch stresses at most `min(B × top_k, n_experts) / ep_degree`
    # distinct experts on any given rank.
    expert_load_s = 0.0
    if phase == "decode":
        # Number of distinct experts touched across the whole batch.
        # Bounded above by n_experts (can't touch more than we have).
        distinct_experts_batch = min(B * top_k, n_experts)
        # Per-rank share (each rank owns n_experts / ep_degree of them).
        distinct_experts_per_rank = max(
            1, distinct_experts_batch // max(1, ep_degree)
        )
        # Bytes per expert (three SwiGLU matmuls, TP-sharded).
        bytes_per_expert = 3 * d_model * expert_dim_per_rank * bpe_exp
        total_expert_load_bytes = distinct_experts_per_rank * bytes_per_expert
        if shared_dim_per_rank > 0:
            # Shared expert is always-on: one load per rank per batch step
            # (weights re-used across every token in the batch, so no B
            # multiplier here either — this was correct in the old formula
            # by coincidence when B was small).
            total_expert_load_bytes += 3 * d_model * shared_dim_per_rank * bpe_exp
        expert_load_s = total_expert_load_bytes / hw.hbm_bandwidth_bytes_s
        # In decode the load typically *exceeds* the compute term — take the
        # max so the layer time reflects the bandwidth wall.
        compute_s = max(compute_s, expert_load_s)

    # --- All-to-all (dispatch + combine) ---
    # Two all-to-alls per MoE layer; each moves top_k × d_model bytes per
    # token (with capacity_factor inflation for dispatch overflow).
    #
    # Wave 19 (P0-1): per-rank egress volume, phase-consistent with the
    # token accounting above. Training: M is already per-rank (each EP rank
    # dispatches its own microbatch), so the volume stands. Serving: the
    # instance's batch of M tokens is spread over the ep ranks, so each
    # rank's egress is M/ep — the old formula charged the WHOLE batch's
    # bytes to every rank, over-pricing serving all-to-all by ep×.
    M_a2a = M if phase == "training" else M / max(ep_degree, 1)
    volume = 2 * M_a2a * top_k * d_model * bpe_act * capacity
    # Node-limited routing dedup for the cross-node leg (only used when
    # ep exceeds the NVLink domain — see _moe_alltoall_cost): each token
    # is sent once per destination node (≤4 nodes in published practice),
    # not once per expert.
    _a2a_topology = ep_topology
    if hw.vendor == "nvidia" and ep_topology == "single_axis":
        _a2a_topology = f"node_limited:{min(1.0, 4.0 / max(1, top_k)):.3f}"
    alltoall_s = _moe_alltoall_cost(volume, ep_degree, hw, _a2a_topology)

    # Wave 19 (P0-1): dispatch/combine overlap. Production MoE stacks hide
    # most all-to-all behind expert + shared-expert compute (DeepEP,
    # dual-batch / 1F1B-overlap schedules). Charging it fully serial made
    # per-replica MoE ~2-3× dense where measured runs sit at ~1.2-1.7×.
    # Exposed cost = (1 - overlap) × raw. Default 0.6; calibratable per
    # hardware via `moe_alltoall_overlap_fraction` (ac-auto-calibrate).
    # Decode is latency-bound small-message a2a with little compute to hide
    # behind — no overlap credit there.
    if phase in ("training", "prefill"):
        _overlap = getattr(calibration, "moe_alltoall_overlap_fraction", None) \
            if calibration is not None else None
        if _overlap is None:
            _overlap = 0.6
        alltoall_s = alltoall_s * max(0.0, 1.0 - float(_overlap))

    # --- Shared-expert allreduce: REMOVED (Wave 19, P0-1) ---
    # Wave 1 Step 1.6 charged a full-activation allreduce across the EP
    # group for shared-expert combination, on the premise that the shared
    # expert is sharded across EP ranks. That premise was wrong: in the
    # DeepSeek-V3 layout the shared expert is REPLICATED per rank and runs
    # on the rank's own tokens; its output is added locally to the routed
    # combine result. No EP collective exists for it. The phantom AR grew
    # with EP beyond the NVLink domain (12.8 ms/layer at EP=32 — 3× the
    # entire real layer compute) and was a major contributor to the "MoE
    # 20× slower than dense" release-review finding. TP sharding of the
    # shared expert is already priced inside _matmul_cost's TP handling.
    shared_expert_ar_s = 0.0

    return {
        "compute_s": compute_s,
        "shared_expert_s": shared_compute_s,
        "shared_expert_ar_s": shared_expert_ar_s,
        "alltoall_s": alltoall_s,
        "expert_load_s": expert_load_s,
        "load_balance_factor": imbalance,
    }


# =============================================================================
# v2 State layer cost
# =============================================================================

def _state_layer_cost(
    arch: ArchConfig,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    phase: str = "training",
    calibration: Optional[CalibrationTable] = None,
) -> LayerBreakdown:
    """
    Compute per-layer cost for a Mamba-2 state layer.

    State replaces attention (QKV proj + attention + output proj).
    FFN still runs normally (handled separately by caller).

    Key insight: at decode, state is SRAM-resident. No KV cache,
    no L-dependent term. Only pay weight loading + small compute.

    Training/prefill: matmul-based compute (structured SSM duality) + weight loading
    Decode: weight loading only (state is SRAM-resident)

    Returns a LayerBreakdown with the state-attention-replacement cost
    in the qkv_proj_s, attention_s, and out_proj_s fields.
    """
    sc = arch.state_config
    if sc is None:
        raise ValueError("state_config is required for state layer cost computation")

    B = arch.batch_size
    S = arch.seq_len if phase != "decode" else 1
    d = arch.d_model
    d_state = int(sc.get("d_state", 128))
    state_expansion = int(sc.get("state_expansion", 2))
    state_n_heads = int(sc.get("n_heads", arch.n_heads))
    state_d_head = int(sc.get("d_head", 64))
    state_prec = str(sc.get("state_precision", arch.weight_precision))
    prec = state_prec
    state_bpe = hw.bytes_per_elem(state_prec)

    heads_per_gpu = state_n_heads // tp_degree
    M = B * S

    breakdown = LayerBreakdown()

    # Mamba-2 structured SSM: the computation involves:
    # 1. Input projection: (d_model) -> (state_expansion * d_model)
    #    which includes B, C, delta projections
    # 2. SSM scan: structured state space computation
    # 3. Output projection: (d_model) -> (d_model)
    #
    # We model this as matmul-equivalent projections.

    # Input projection: (M, d) -> (M, state_expansion * d / tp)
    in_proj_N = state_expansion * d // tp_degree
    t_in_proj, _, _ = _matmul_cost(M, in_proj_N, d, prec, hw, lattice_hw, tp_degree, calibration)

    # SSM scan compute (structured state space duality):
    # In training/prefill, this is a matmul-like operation
    # Compute: 2 * B * S * heads_per_gpu * d_state * state_d_head FLOPs
    if phase != "decode":
        ssm_flops = 2 * M * heads_per_gpu * d_state * state_d_head
        eff = get_tile_efficiency(M, d_state, state_d_head, state_prec, lattice_hw)
        eff = max(eff, 0.1)
        t_ssm = ssm_flops / (hw.peak_flops_s(state_prec) * eff)
        # SSM also has memory traffic for the state matrices
        ssm_bytes = M * (heads_per_gpu * d_state * state_d_head) * state_bpe * 2
        t_ssm_mem = ssm_bytes / hw.hbm_bandwidth_bytes_s
        t_ssm = max(t_ssm, t_ssm_mem)
    else:
        # Decode: state is SRAM-resident, no KV cache, no L-dependent term
        # Only pay weight loading for the SSM parameters
        # State update is a small vector operation (d_state x d_head per head)
        ssm_weight_bytes = heads_per_gpu * d_state * state_d_head * state_bpe
        t_ssm = ssm_weight_bytes / hw.hbm_bandwidth_bytes_s
        # Plus a tiny compute term for the state update
        ssm_flops = 2 * B * heads_per_gpu * d_state * state_d_head
        t_ssm_compute = ssm_flops / max(hw.peak_flops_s(state_prec), 1.0)
        t_ssm = max(t_ssm, t_ssm_compute)

    # Output projection: (M, d / tp) -> (M, d)
    out_K = d // tp_degree
    t_out, _, _ = _matmul_cost(M, d, out_K, prec, hw, lattice_hw, tp_degree, calibration)

    if phase == "decode":
        # Decode: dominated by weight loading. No KV cache. No L-dependent term.
        # Input projection weight load
        in_proj_weight_bytes = d * in_proj_N * state_bpe
        t_in_proj_decode = in_proj_weight_bytes / hw.hbm_bandwidth_bytes_s
        # Small compute for the projection
        t_in_proj_decode = max(t_in_proj_decode, t_in_proj)

        # Output projection weight load
        out_weight_bytes = out_K * d * state_bpe
        t_out_decode = out_weight_bytes / hw.hbm_bandwidth_bytes_s
        t_out_decode = max(t_out_decode, t_out)

        breakdown.qkv_proj_s = t_in_proj_decode
        breakdown.attention_s = t_ssm
        breakdown.out_proj_s = t_out_decode
    else:
        breakdown.qkv_proj_s = t_in_proj
        breakdown.attention_s = t_ssm
        breakdown.out_proj_s = t_out

    return breakdown


def compute_crossover_seq_len(
    arch: ArchConfig,
    hw: HardwareConfig,
    tp_degree: int = 1,
) -> float:
    """
    Compute L* where state decode cost equals attention decode cost.

    At decode:
    - Attention: KV cache load = 2 * B * n_kv_heads_per_gpu * L * d_head * kv_bpe / HBM_BW
    - State: weight load (L-independent) = state_weight_bytes / HBM_BW

    L* = state_weight_bytes / (2 * B * n_kv_heads_per_gpu * d_head * kv_bpe)

    Above L*, state layers are cheaper at decode time.
    """
    sc = arch.state_config
    if sc is None:
        return float("inf")

    d = arch.d_model
    B = arch.batch_size
    kv_bpe = precision_bytes_per_element(arch.kv_precision)
    state_prec = sc.get("state_precision", arch.precision)
    state_bpe = precision_bytes_per_element(state_prec)

    d_state = int(sc.get("d_state", 128))
    state_expansion = int(sc.get("state_expansion", 2))
    state_n_heads = int(sc.get("n_heads", arch.n_heads))
    state_d_head = int(sc.get("d_head", 64))

    heads_per_gpu = state_n_heads // tp_degree
    kv_heads_per_gpu = max(1, math.ceil(arch.n_kv_heads / max(1, tp_degree)))

    # State weight bytes per layer (loaded at decode):
    # Input projection: d * state_expansion * d / tp
    # SSM parameters: heads_per_gpu * d_state * state_d_head
    # Output projection: d/tp * d
    in_proj_bytes = d * (state_expansion * d // tp_degree) * state_bpe
    ssm_bytes = heads_per_gpu * d_state * state_d_head * state_bpe
    out_proj_bytes = (d // tp_degree) * d * state_bpe
    total_state_bytes = in_proj_bytes + ssm_bytes + out_proj_bytes

    # Attention KV cache load per unit of L:
    # 2 * B * kv_heads_per_gpu * d_head * kv_bpe (per increment of L by 1)
    # v1-fix MLA: for MLA, the per-token KV is the latent + d_rope (not
    # 2× n_kv × d_head), and it's not sharded across TP because the latent
    # is shared across heads. Use the model's helper for accuracy.
    if arch.attention_type == "mla" and arch.mla_kv_latent_dim:
        kv_load_per_L = B * arch.kv_bytes_per_token_per_layer()
    else:
        kv_load_per_L = 2 * B * kv_heads_per_gpu * arch.d_head * kv_bpe

    if kv_load_per_L <= 0:
        return float("inf")

    # Also need attention weight bytes to compare fairly:
    # QKV proj weights: d * (heads_per_gpu + 2*kv_heads_per_gpu) * d_head * bpe
    # Output proj weights: heads_per_gpu * d_head * d * bpe
    attn_heads_per_gpu = arch.n_heads // tp_degree
    attn_bpe = precision_bytes_per_element(arch.precision)
    qkv_weight_bytes = d * (attn_heads_per_gpu + 2 * kv_heads_per_gpu) * arch.d_head * attn_bpe
    out_weight_bytes = attn_heads_per_gpu * arch.d_head * d * attn_bpe
    attn_fixed_bytes = qkv_weight_bytes + out_weight_bytes

    # L* where: attn_fixed_bytes + kv_load_per_L * L = total_state_bytes
    # So: L* = (total_state_bytes - attn_fixed_bytes) / kv_load_per_L
    # If total_state_bytes < attn_fixed_bytes, state is always cheaper (L*=0)
    if total_state_bytes <= attn_fixed_bytes:
        return 0.0

    return (total_state_bytes - attn_fixed_bytes) / kv_load_per_L


# =============================================================================
# Memory footprint estimator
# =============================================================================

def estimate_memory_per_gpu(
    arch: ArchConfig,
    tp_degree: int = 1,
    pp_degree: int = 1,
    include_kv_cache: bool = True,
    kv_cache_len: int = 2048,
    ep_degree: int = 1,
    cp_degree: int = 1,
) -> float:
    """Estimate memory footprint per GPU in bytes.

    Dense path matches v0 behavior.

    MoE path: subtracts the dense-FFN contribution implicit in the v0
    estimate_params call and replaces it with the MoE expert + shared
    contribution sharded by both TP and EP:

        expert_weights_per_gpu = n_experts * 3 * d_model * expert_dim
                                  * bpe_expert
                                  / (tp_degree * ep_degree)
        shared_weights_per_gpu = 3 * d_model * shared_dim
                                  * bpe_expert / tp_degree   (replicated over EP)
    """
    activation_bpe = precision_bytes_per_element(arch.activation_precision)

    layers_per_stage = math.ceil(arch.n_layers / max(pp_degree, 1))
    byte_ledger = parameter_byte_ledger(arch)
    shared_divisor = max(1, tp_degree) * max(1, pp_degree)
    expert_divisor = shared_divisor * max(1, ep_degree)
    model_bytes = (
        byte_ledger.shared_bytes / shared_divisor
        + byte_ledger.expert_total / expert_divisor
    )

    # KV cache
    kv_bytes = 0
    if include_kv_cache:
        # Per layer: 2 (K+V) × n_kv_heads × d_head × seq_len × batch × bytes
        # v1-fix MLA: when type=mla, KV is a single compressed latent + RoPE
        # key (not 2× n_kv_heads × d_head). kv_bytes_per_token_per_layer
        # returns the right per-token quantity.
        # Wave 35: split per-token KV into the TP-shardable per-kv-head
        # portion and the TP-replicated portion (MLA latent, IndexShare
        # index). Replaces the old kv_tp_shards special-case, which
        # wrongly TP-divided the IndexShare index.
        kv_shard_b, kv_repl_b = arch.kv_bytes_per_token_split(kv_cache_len)
        _kv_tp_shards = max(1, min(tp_degree, arch.n_kv_heads))
        kv_per_token_sharded = kv_shard_b / _kv_tp_shards + kv_repl_b
        kv_per_layer = kv_per_token_sharded * kv_cache_len * arch.batch_size
        # v1-fix YOCO: cross-layer KV sharing — only K layers carry their
        # own KV. Cuts kv_bytes by K/n_layers in the dense memory estimator.
        yoco_k = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
        if 0 < yoco_k < arch.n_layers:
            effective_kv_layers = max(1, int(yoco_k * layers_per_stage / arch.n_layers + 0.5))
        else:
            effective_kv_layers = layers_per_stage
        # Wave 18g: local:global interleave — local layers only store
        # min(L, window) KV entries. Weight the effective layer count by
        # the per-layer-type KV length ratio.
        n_local = int(getattr(arch, "n_local_attn_layers", 0) or 0)
        if 0 < n_local <= arch.n_layers and int(arch.local_window or 0) > 0:
            local_ratio = arch.local_kv_len(kv_cache_len) / max(1, kv_cache_len)
            local_frac = n_local / arch.n_layers
            effective_kv_layers = effective_kv_layers * (
                (1.0 - local_frac) + local_frac * local_ratio)
        # v1-fix long-ctx (June 2026): CP splits the sequence axis, so
        # each rank only holds 1/(cp_degree) of the KV cache for the full
        # sequence. Previously the memory estimator ignored cp_degree
        # entirely, which made every long-ctx cell stamp infeasible
        # even when CP would have made it fit.
        cp = max(1, int(cp_degree))
        # Wave 35: TP sharding is already folded into kv_per_layer via
        # kv_bytes_per_token_split (per-kv-head part / min(TP, n_kv_heads),
        # MLA latent + IndexShare index replicated). CP splits the
        # sequence axis for every cache type.
        kv_bytes = kv_per_layer * effective_kv_layers / cp

    # Activations.
    # v1-fix long-ctx (June 2026): the previous formula
    # (batch × seq × d_model × bpe × 10) modelled UNCHECKPOINTED training
    # peak — keeping every intermediate alive across all layers. At long
    # context that's astronomically larger than any real deployment.
    #
    # Real frameworks (Megatron, DeepSpeed, FSDP) on long context use:
    #   - Activation checkpointing: only sqrt(n_layers) boundaries kept
    #   - Megatron-style sequence-parallel: activations / TP
    #   - CP: activations / CP along seq axis
    #
    # Peak memory ≈ B × S × d × bpe × (4 + sqrt(L)) / (TP × CP). The
    # constant 4 covers one layer's working set (Q, K, V, O) during
    # recompute; sqrt(L) is the checkpoint boundary count.
    #
    # v1-fix Wave 1 Step 1.4 (Jun 2026): multiply activations by the PP
    # queue depth. GPipe holds `pp_degree` in-flight microbatches per
    # stage; 1F1B holds (pp_degree+1)/2 on average; interleaved 1F1B with
    # v virtual stages holds (pp_degree+1)/2 × v. The v0 model omitted
    # this entirely, under-reporting memory by `pp_degree×` on PP-heavy
    # configs.
    import math as _math
    cp = max(1, int(cp_degree))
    tp_act = max(1, int(tp_degree))
    n_layers = max(1, arch.n_layers)
    layer_factor = 4 + _math.sqrt(n_layers)
    pp = max(1, int(pp_degree))
    schedule = getattr(arch, "pp_schedule", "1f1b")
    if pp <= 1:
        pp_act_queue = 1.0
    elif schedule == "gpipe":
        pp_act_queue = float(pp)
    elif schedule == "interleaved":
        v_stages = max(1, int(getattr(arch, "pp_virtual_stages", 2)))
        pp_act_queue = ((pp + 1) / 2) * v_stages
    else:  # "1f1b" — modern default
        pp_act_queue = (pp + 1) / 2
    # Wave 18e post-audit fix (2026-07): activations are sized by the
    # actual prefill sequence length used during serving, not by the
    # model's supported context. `arch.seq_len` carries the model's
    # capacity (e.g. 8192 for Llama-3-8B); `kv_cache_len` carries the
    # workload's actual prefill length (e.g. 1024 for the vLLM benchmark
    # against which Llama-3-8B TTFT=55ms was measured). Using seq_len
    # over-allocated activations by up to 8× on the Llama-3-8B anchor
    # and explained the +48% memory error. Use max(kv_cache_len,
    # arch.seq_len when kv_cache_len is unset) so callers that didn't
    # thread kv_cache_len see the old behavior.
    act_seq_len = (
        int(kv_cache_len) if kv_cache_len and kv_cache_len > 0 else int(arch.seq_len)
    )
    # Guard: never larger than the model's supported context.
    act_seq_len = min(act_seq_len, int(arch.seq_len)) if arch.seq_len else act_seq_len
    act_bytes = (arch.batch_size * act_seq_len * arch.d_model * activation_bpe
                 * layer_factor * pp_act_queue / (tp_act * cp))

    return model_bytes + kv_bytes + act_bytes


def estimate_memory_per_gpu_hybrid(
    arch: ArchConfig,
    tp_degree: int = 1,
    pp_degree: int = 1,
    include_kv_cache: bool = True,
    kv_cache_len: int = 2048,
    ep_degree: int = 1,
    cp_degree: int = 1,
) -> float:
    """Estimate memory footprint for hybrid attention/state architectures.

    KV cache is only allocated for attention layers.
    State layers contribute SSM projection weights but no KV cache.
    """
    layer_types = arch.layer_type_list or (["attention"] * arch.n_layers)
    # Wave 18g: "local_attention" layers are attention layers for weight
    # and KV-cache purposes (GQA projection + windowed KV); they must NOT
    # fall into the state-layer bucket, which would price them as SSM
    # blocks and drop their KV entirely.
    n_global_attn_layers = sum(1 for lt in layer_types if lt == "attention")
    n_local_attn_layers = sum(1 for lt in layer_types if lt == "local_attention")
    n_attn_layers = n_global_attn_layers + n_local_attn_layers
    n_state_layers = sum(1 for lt in layer_types if lt == "state")

    activation_bpe = precision_bytes_per_element(arch.activation_precision)

    layers_per_stage = arch.n_layers // max(pp_degree, 1)

    # Count attention and state layers in this pipeline stage
    # (Simplified: assume uniform distribution across stages)
    attn_frac = n_attn_layers / max(1, arch.n_layers)
    attn_layers_this_stage = int(attn_frac * layers_per_stage + 0.5)

    # Use the shared parameter ledger so hybrid+MoE stores all experts rather
    # than one dense-FFN surrogate.
    byte_ledger = parameter_byte_ledger(arch)
    shared_divisor = max(1, tp_degree) * max(1, pp_degree)
    expert_divisor = shared_divisor * max(1, ep_degree)
    model_bytes = (
        byte_ledger.shared_bytes / shared_divisor
        + byte_ledger.expert_total / expert_divisor
    )

    # KV cache: only for attention layers
    kv_bytes = 0
    if include_kv_cache:
        # v1-fix MLA: MLA caches one compressed latent + d_rope key, not
        # 2× n_kv × d_head. The helper handles both attention types.
        # Wave 35: TP-shardable vs TP-replicated split (see
        # kv_bytes_per_token_split). Folds min(TP, n_kv_heads) sharding
        # into the per-token figure; MLA latent and IndexShare index stay
        # replicated across TP ranks.
        kv_shard_b, kv_repl_b = arch.kv_bytes_per_token_split(kv_cache_len)
        _kv_tp_shards = max(1, min(tp_degree, arch.n_kv_heads))
        kv_per_token_sharded = kv_shard_b / _kv_tp_shards + kv_repl_b
        kv_per_layer = kv_per_token_sharded * kv_cache_len * arch.batch_size
        # v1-fix YOCO: cross-layer KV sharing — only K layers keep their own
        # KV. Cuts kv_bytes by K/n_layers.
        yoco_k = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
        if 0 < yoco_k < arch.n_layers:
            effective_kv_layers = max(1, int(yoco_k * layers_per_stage / arch.n_layers + 0.5))
        else:
            effective_kv_layers = attn_layers_this_stage
        # Wave 18g: local:global interleave — local attention layers only
        # store min(L, window) KV entries. Weight by the local share of the
        # ATTENTION layer count (state layers already excluded above).
        if n_local_attn_layers > 0 and int(arch.local_window or 0) > 0 and n_attn_layers > 0:
            local_ratio = arch.local_kv_len(kv_cache_len) / max(1, kv_cache_len)
            local_frac = n_local_attn_layers / n_attn_layers
            effective_kv_layers = effective_kv_layers * (
                (1.0 - local_frac) + local_frac * local_ratio)
        cp = max(1, int(cp_degree))
        # v1-fix long-ctx (June 2026): CP splits the sequence axis, so KV
        # and seq-parallel activations are 1/cp per rank. Wave 35: TP
        # sharding already folded into kv_per_layer via the split helper.
        kv_bytes = kv_per_layer * effective_kv_layers / cp

    # Activations: TP-sequence-parallel + CP-sharded + sqrt(L) checkpoint
    # boundaries + PP queue depth. See estimate_memory_per_gpu for the
    # derivation. Wave 1 Step 1.4 added the PP queue multiplier.
    import math as _math
    cp = max(1, int(cp_degree))
    tp_act = max(1, int(tp_degree))
    n_layers = max(1, arch.n_layers)
    layer_factor = 4 + _math.sqrt(n_layers)
    pp = max(1, int(pp_degree))
    schedule = getattr(arch, "pp_schedule", "1f1b")
    if pp <= 1:
        pp_act_queue = 1.0
    elif schedule == "gpipe":
        pp_act_queue = float(pp)
    elif schedule == "interleaved":
        v_stages = max(1, int(getattr(arch, "pp_virtual_stages", 2)))
        pp_act_queue = ((pp + 1) / 2) * v_stages
    else:  # "1f1b"
        pp_act_queue = (pp + 1) / 2
    act_bytes = (arch.batch_size * arch.seq_len * arch.d_model * activation_bpe
                 * layer_factor * pp_act_queue / (tp_act * cp))

    return model_bytes + kv_bytes + act_bytes


# =============================================================================
# Core throughput function
# =============================================================================

def compute_layer_time(
    arch: ArchConfig,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    phase: str = "training",  # "training" | "prefill" | "decode"
    kv_cache_len: int = 0,    # for decode phase
    calibration: Optional[CalibrationTable] = None,
    ep_degree: int = 1,                # v1: expert-parallel degree (ignored if dense)
    ep_topology: str = "single_axis",  # v1: TPU torus axis layout
    layer_type: str = "attention",     # v2: "attention" | "state"
) -> LayerBreakdown:
    """
    Compute per-layer time for a given phase.

    Training / prefill: batch x seq matmuls, fused attention.
    Decode: batch x 1 matmuls, KV cache load dominates.

    v2: layer_type="state" branches to state layer cost (Mamba-2).
    State layers have NO KV cache, NO L-dependent decode term.
    """
    B = arch.batch_size
    S = arch.seq_len if phase != "decode" else 1
    d = arch.d_model
    dh = arch.d_head
    nh = arch.n_heads
    nkv = arch.n_kv_heads
    ffn = arch.ffn_dim
    ffn_prec = arch.precision
    attn_precisions = dict(getattr(arch, "attn_precision", {}) or {})
    projection_prec = str(
        attn_precisions.get("v", arch.weight_precision) or arch.weight_precision
    )
    attention_prec = str(
        attn_precisions.get(
            "qk", attn_precisions.get("q", arch.activation_precision)
        ) or arch.activation_precision
    )
    output_prec = str(
        attn_precisions.get(
            "output", attn_precisions.get("o", arch.weight_precision)
        ) or arch.weight_precision
    )
    activation_prec = str(arch.activation_precision or "bf16")

    # Wave 18g: local (sliding-window) attention layer. Same projections
    # and FFN as a global attention layer; the only differences are that
    # decode reads at most `local_window` KV entries and prefill attention
    # is S x min(S, W) instead of S x S. Handled here by capping the
    # phase-relevant lengths, then falling through the ordinary attention
    # path. Local layers always use the GQA projection (matches GPT-OSS /
    # Gemma-2 practice) even when the global layers are MLA — MLA's latent
    # brings nothing once the window already bounds the KV read.
    _is_local_layer = (layer_type == "local_attention"
                       and int(arch.local_window or 0) > 0)
    if _is_local_layer and phase == "decode":
        kv_cache_len = arch.local_kv_len(kv_cache_len)

    heads_per_gpu = nh // tp_degree
    kv_heads_per_gpu = max(1, math.ceil(nkv / max(1, tp_degree)))

    # v2: branch on layer type
    if layer_type == "state" and arch.state_config is not None:
        # State layer: compute state cost (replaces attention) + FFN
        state_bd = _state_layer_cost(arch, hw, lattice_hw, tp_degree, phase, calibration)
        breakdown = LayerBreakdown()
        breakdown.qkv_proj_s = state_bd.qkv_proj_s
        breakdown.attention_s = state_bd.attention_s
        breakdown.out_proj_s = state_bd.out_proj_s

        # FFN still runs (state replaces attention, not FFN)
        M = B * S
        if arch.moe_config is None:
            ffn_per_gpu = ffn // tp_degree
            if arch.ffn_type == "swiglu":
                t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
                t_gate, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
                breakdown.ffn_up_s = t_up + t_gate
            else:
                t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
                breakdown.ffn_up_s = t_up
            t_down, _, _ = _matmul_cost(M, d, ffn_per_gpu, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            breakdown.ffn_down_s = t_down
        else:
            expert_prec = arch.moe_config.get("precision", ffn_prec)
            moe_cost = _moe_ffn_cost(
                B=B, S=S, d_model=d,
                moe_cfg=arch.moe_config,
                ep_degree=ep_degree,
                tp_degree=tp_degree,
                activation_precision=arch.activation_precision,
                expert_precision=expert_prec,
                hw=hw, lattice_hw=lattice_hw,
                phase=phase,
                calibration=calibration,
                ep_topology=ep_topology,
                imbalance=getattr(arch, "worst_case_imbalance_factor", 1.0),
            )

            # v1-fix Part B: blend dense + MoE FFN costs by layer-mix
            # fraction when n_dense_ffn_layers > 0.
            n_dense = max(0, min(int(getattr(arch, "n_dense_ffn_layers", 0)), arch.n_layers))
            n_moe = max(0, arch.n_layers - n_dense)
            if n_dense > 0 and n_moe > 0:
                ffn_per_gpu = ffn // tp_degree
                t_up_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
                t_gate_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
                t_down_d, _, _ = _matmul_cost(M, d, ffn_per_gpu, ffn_prec, hw, lattice_hw, tp_degree, calibration)
                dense_layer_s = t_up_d + t_gate_d + t_down_d
                w_d = n_dense / arch.n_layers
                w_m = n_moe / arch.n_layers
                breakdown.ffn_up_s = w_m * moe_cost["compute_s"] + w_d * dense_layer_s
                breakdown.ffn_down_s = 0.0
                breakdown.shared_expert_s = w_m * moe_cost["shared_expert_s"]
                breakdown.alltoall_s = w_m * moe_cost["alltoall_s"]
                breakdown.expert_load_s = w_m * moe_cost["expert_load_s"]
                breakdown.load_balance_factor = moe_cost["load_balance_factor"]
            else:
                breakdown.ffn_up_s = moe_cost["compute_s"]
                breakdown.ffn_down_s = 0.0
                breakdown.shared_expert_s = moe_cost["shared_expert_s"]
                breakdown.alltoall_s = moe_cost["alltoall_s"]
                breakdown.expert_load_s = moe_cost["expert_load_s"]
                breakdown.load_balance_factor = moe_cost["load_balance_factor"]

        # Membound ops and allreduce
        breakdown.membound_ops_s = _membound_ops_cost(B, S if phase != "decode" else 1, d, activation_prec, hw)
        # Wave 17 Part 2: pass phase through so decode picks lower overlap +
        # picks up the per-collective launch-latency floor.
        breakdown.allreduce_s = _allreduce_cost(
            B, S if phase != "decode" else 1, d, activation_prec, hw, tp_degree,
            phase=phase,
        )

        # Aggregate
        compute_total = (breakdown.qkv_proj_s + breakdown.attention_s +
                        breakdown.out_proj_s + breakdown.ffn_up_s + breakdown.ffn_down_s)
        memory_total = breakdown.membound_ops_s
        comm_total = breakdown.allreduce_s + breakdown.alltoall_s

        breakdown.compute_s = compute_total
        breakdown.memory_s = memory_total
        breakdown.communication_s = comm_total
        breakdown.total_s = compute_total + memory_total + comm_total

        if compute_total >= max(memory_total, comm_total):
            breakdown.bottleneck = "compute"
        elif comm_total >= memory_total:
            breakdown.bottleneck = "communication"
        else:
            breakdown.bottleneck = "memory"

        return breakdown

    # --- Original attention layer path (v0 behavior, unchanged) ---
    breakdown = LayerBreakdown()

    # Effective M dimension for matmuls
    M = B * S  # tokens in this step

    # --- QKV projection ---
    # Wave 18g: local layers use the plain GQA projection even when the
    # global layers are MLA (mixed-projection stacks a la Kimi-linear).
    is_mla = (arch.attention_type == "mla"
              and int(getattr(arch, "mla_kv_latent_dim", 0) or 0) > 0
              and not _is_local_layer)
    if is_mla:
        # v1-fix MLA compute path (DeepSeek-V2/V3). Instead of one fused
        # (d → (nh+2nkv)*dh) projection, MLA has:
        #   - Q  down-project   : (d) → c_q          (per-token)
        #   - Q  up-project     : c_q → nh*dh        (per-token)
        #   - KV down-project   : (d) → c_kv + d_rope (per-token, shared)
        #   - KV up-project     : c_kv → nh*(dh-d_rope)  (folded into attention
        #     via matmul absorption at decode time)
        # In compute terms, the down/up factorization saves FLOPs whenever
        # c_q + c_kv < (nh + 2*nkv) * dh. At decode the absorption trick
        # collapses the KV up-projection into the attention dot product so
        # we only pay the small Q up-project per generated token.
        c_q = int(getattr(arch, "mla_q_latent_dim", 0) or 0)
        c_kv = int(getattr(arch, "mla_kv_latent_dim", 0) or 0)
        d_rope = int(getattr(arch, "mla_rope_head_dim", 0) or 0)
        # Q path: (M, d) × (d, c_q) + (M, c_q) × (c_q, nh*dh/tp)
        if c_q > 0:
            t_q_down, _, _ = _matmul_cost(M, c_q, d, projection_prec, hw, lattice_hw,
                                            tp_degree, calibration)
            t_q_up, _, _ = _matmul_cost(M, heads_per_gpu * dh, c_q, projection_prec,
                                          hw, lattice_hw, tp_degree, calibration)
        else:
            # Without an explicit q-latent, treat Q as a direct projection.
            t_q_down, _, _ = _matmul_cost(M, heads_per_gpu * dh, d, projection_prec,
                                            hw, lattice_hw, tp_degree, calibration)
            t_q_up = 0.0
        # KV down-projection: shared latent + RoPE'd key, NOT sharded by TP
        # (because the latent feeds every query head; replicating is cheaper
        # than the all-gather required to shard).
        kv_proj_N = c_kv + d_rope
        t_kv_down, _, _ = _matmul_cost(M, kv_proj_N, d, projection_prec, hw, lattice_hw,
                                         1, calibration)
        if phase == "decode":
            # Absorbed: KV up-projection is folded into the attention dot;
            # we only pay Q-up here.
            breakdown.qkv_proj_s = t_q_down + t_q_up + t_kv_down
        else:
            # Prefill / training: explicit KV up-projection per token.
            kv_up_N = heads_per_gpu * max(0, dh - d_rope)
            if kv_up_N > 0:
                t_kv_up, _, _ = _matmul_cost(M, kv_up_N, c_kv, projection_prec, hw,
                                               lattice_hw, tp_degree, calibration)
            else:
                t_kv_up = 0.0
            breakdown.qkv_proj_s = t_q_down + t_q_up + t_kv_down + t_kv_up
    else:
        # Q: (M, d) × (d, nh*dh/tp) -> three such matmuls (Q, K, V separately or fused)
        # Fused: (M, d) × (d, (nh + 2*nkv)*dh / tp)
        qkv_N = (heads_per_gpu + 2 * kv_heads_per_gpu) * dh
        t_qkv, _, _ = _matmul_cost(M, qkv_N, d, projection_prec, hw, lattice_hw, tp_degree, calibration)
        breakdown.qkv_proj_s = t_qkv

    # --- Attention ---
    if phase == "decode":
        # Decode attention: query (B,1,nh,dh) × keys (B,L,nkv,dh)
        # This is KV-cache-bandwidth-bound, not compute-bound
        kv_bpe = precision_bytes_per_element(arch.kv_precision)
        L = kv_cache_len

        # KV cache load per layer: K + V, each (B, nkv_per_gpu, L, dh).
        # v1-fix MLA: MLA caches a single shared latent (c_kv + d_rope) per
        # token, NOT 2 × n_kv × d_head. The latent is not sharded across TP
        # because it's shared across query heads.
        if _is_local_layer:
            # Wave 18g: local layer under any global projection — plain GQA
            # KV, already length-capped at the window above.
            kv_bytes_per_layer = 2 * B * kv_heads_per_gpu * L * dh * kv_bpe
        elif arch.attention_type == "mla" and arch.mla_kv_latent_dim:
            kv_bytes_per_layer = B * L * arch.kv_bytes_per_token_per_layer(L)
        elif arch.attention_type in ("nsa", "csa", "indexshare", "msa"):
            # Wave 35 fix: these caches are stored PER KV HEAD, so the
            # per-rank decode stream shards across TP exactly like GQA
            # (the capacity estimator already divided them by
            # min(TP, n_kv_heads); the bandwidth path here charged every
            # rank the full unsharded stream — an 8× TBT over-charge at
            # TP=8). Only the truly head-shared structures stay
            # replicated: the MLA latent (branch above) and the
            # IndexShare per-token index entry.
            shard_b, repl_b = arch.kv_bytes_per_token_split(L)
            kv_shards = max(1, min(tp_degree, arch.n_kv_heads))
            kv_bytes_per_layer = B * L * (shard_b / kv_shards + repl_b)
        else:
            kv_bytes_per_layer = 2 * B * kv_heads_per_gpu * L * dh * kv_bpe
        # Wave 35 YOCO decode fix: the previous K/N amortization claimed
        # decode TBT shrinks by K/N — physically wrong. Each of the N-K
        # cross-decoder layers still STREAMS the shared cache from HBM
        # every decode step (a multi-GB cache does not persist in ~50 MB
        # of L2 across layers), and the K self layers stream their own:
        # N cache-reads per step, same as a conventional stack. YOCO's
        # real serving wins are cache CAPACITY (K/N, modeled in the
        # memory estimators) and PREFILL early-exit (modeled in the
        # serving prefill path). No decode-bandwidth factor is applied.
        t_kv_load = kv_bytes_per_layer / hw.hbm_bandwidth_bytes_s

        # Small compute: (B, heads_per_gpu, 1, dh) × (B, heads_per_gpu, dh, L)
        # GQA: each kv head serves multiple q heads, but compute is still small
        attn_flops = 2 * B * heads_per_gpu * 1 * L * dh * 2
        fused_eff = hw.fused_attention_efficiency.get(attention_prec,
                    hw.fused_attention_efficiency.get("bf16", 0.75))
        t_attn_compute = attn_flops / (hw.peak_flops_s(attention_prec) * fused_eff)
        breakdown.attention_s = max(t_kv_load, t_attn_compute)
    else:
        if _is_local_layer:
            # Wave 18g: sliding-window prefill/training attention is
            # S x min(S, W) — same sparse-cost model NSA uses, with the
            # window as the attended span.
            attended = min(arch.seq_len, int(arch.local_window or 0))
            breakdown.attention_s = _sparse_attention_cost(
                B, arch.seq_len, attended, nh, dh, nkv, attention_prec, hw, tp_degree
            )
        elif arch.attention_type == "nsa" and arch.nsa_window_size:
            stride = int(arch.nsa_compress_block_stride or 16)
            selected = (
                int(arch.nsa_select_top_k or 16)
                * int(arch.nsa_select_block_size or 64)
            )
            attended = min(
                arch.seq_len,
                math.ceil(arch.seq_len / max(1, stride))
                + selected
                + int(arch.nsa_window_size or 512),
            )
            breakdown.attention_s = _sparse_attention_cost(
                B, arch.seq_len, attended, nh, dh, nkv, attention_prec, hw, tp_degree
            )
        elif arch.attention_type == "csa" and arch.csa_top_k_blocks:
            # Wave 32 fix: CSA/indexshare/MSA prefill used to fall through
            # to the full S x S branch below, pricing sparse-attention
            # prefill AT OR ABOVE dense — physically wrong. Their decode
            # KV bytes were already sparse (kv_bytes_per_token_per_layer),
            # so TBT was right while TTFT was dense-priced. Each family
            # now uses the same S x attended sparse-cost model NSA uses,
            # with `attended` mirroring its decode-side effective-KV
            # formula.
            #
            # CSA: score S/block_size compressed blocks (comp_dim wide,
            # scaled to d_head units), then attend top_k selected blocks
            # at full width.
            block_size = int(arch.csa_block_size or 64)
            top_k = int(arch.csa_top_k_blocks or 16)
            comp_dim = int(arch.csa_compression_dim or dh)
            attended = min(
                arch.seq_len,
                math.ceil(
                    math.ceil(arch.seq_len / max(1, block_size))
                    * (comp_dim / max(1, dh)))
                + top_k * block_size,
            )
            breakdown.attention_s = _sparse_attention_cost(
                B, arch.seq_len, attended, nh, dh, nkv, attention_prec, hw, tp_degree
            )
        elif arch.attention_type == "indexshare" and arch.indexshare_top_k_buckets:
            # IndexShare: indexer scores num_buckets centroids (idx_dim
            # wide, scaled to d_head units), then attends
            # top_k_buckets x (S / num_buckets) tokens at full width.
            num_buckets = int(arch.indexshare_num_buckets or 64)
            top_k = int(arch.indexshare_top_k_buckets or 4)
            idx_dim = int(arch.indexshare_index_dim or 64)
            attended = min(
                arch.seq_len,
                math.ceil(num_buckets * (idx_dim / max(1, dh)))
                + top_k * math.ceil(arch.seq_len / max(1, num_buckets)),
            )
            breakdown.attention_s = _sparse_attention_cost(
                B, arch.seq_len, attended, nh, dh, nkv, attention_prec, hw, tp_degree
            )
        elif arch.attention_type == "msa":
            # MSA: local window + dilated top-k + global top-k per query.
            win = int(arch.msa_window_size or 512)
            dilated_k = int(arch.msa_dilated_top_k or 64)
            global_k = int(arch.msa_global_top_k or 16)
            attended = min(arch.seq_len, win + dilated_k + global_k)
            breakdown.attention_s = _sparse_attention_cost(
                B, arch.seq_len, attended, nh, dh, nkv, attention_prec, hw, tp_degree
            )
        else:
            # Training / prefill: full S×S attention
            breakdown.attention_s = _attention_cost(
                B, arch.seq_len, nh, dh, nkv, attention_prec, hw, tp_degree
            )

    # --- Output projection ---
    # (M, heads_per_gpu * dh) × (heads_per_gpu * dh, d)
    out_K = heads_per_gpu * dh
    t_out, _, _ = _matmul_cost(M, d, out_K, output_prec, hw, lattice_hw, tp_degree, calibration)
    breakdown.out_proj_s = t_out

    # v1-fix 2:4 sparsity: per-component speedup factor. NVIDIA tensor cores
    # give 2× on 2:4-sparsified matmuls; TPU/Trainium have no native sparse
    # path so the speedup is gated by vendor.
    sparsity = getattr(arch, "sparsity_2_4", None) or {}
    sparse_speedup_nvidia = (hw.vendor == "nvidia")
    def _sparse_factor(component: str) -> float:
        if sparsity.get(component) and sparse_speedup_nvidia:
            return 0.5
        return 1.0
    breakdown.qkv_proj_s *= _sparse_factor("attn_qkv")
    breakdown.out_proj_s *= _sparse_factor("attn_o")

    # --- FFN: dense or MoE branch ---
    if arch.moe_config is None:
        # Dense path (v0 behavior, unchanged).
        ffn_per_gpu = ffn // tp_degree
        if arch.ffn_type == "swiglu":
            # SwiGLU: two parallel projections (up + gate), each (M, d) → (M, ffn/tp)
            t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            t_gate, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            breakdown.ffn_up_s = t_up * _sparse_factor("ffn_up") + t_gate * _sparse_factor("ffn_gate")
        else:
            t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            breakdown.ffn_up_s = t_up * _sparse_factor("ffn_up")

        # --- FFN down ---
        t_down, _, _ = _matmul_cost(M, d, ffn_per_gpu, ffn_prec, hw, lattice_hw, tp_degree, calibration)
        breakdown.ffn_down_s = t_down * _sparse_factor("ffn_down")
    else:
        # MoE path. ffn_up_s carries the expert-compute total; ffn_down_s is
        # left at 0 (the down matmul is rolled into compute_s by _moe_ffn_cost).
        # The shared-expert compute is recorded separately for inspection.
        expert_prec = arch.moe_config.get("precision", ffn_prec)
        moe_cost = _moe_ffn_cost(
            B=B, S=S, d_model=d,
            moe_cfg=arch.moe_config,
            ep_degree=ep_degree,
            tp_degree=tp_degree,
            activation_precision=arch.activation_precision,
            expert_precision=expert_prec,
            hw=hw, lattice_hw=lattice_hw,
            phase=phase,
            calibration=calibration,
            ep_topology=ep_topology,
            imbalance=getattr(arch, "worst_case_imbalance_factor", 1.0),
        )

        # v1-fix Part B: blend dense + MoE FFN costs by layer-mix fraction
        # when n_dense_ffn_layers > 0. The dense FFN compute is computed
        # here at the same B*S; the per-stack throughput multiplies by
        # n_layers downstream, so blending here gives the right total cost
        # for a mixed stack.
        n_dense = max(0, min(int(getattr(arch, "n_dense_ffn_layers", 0)), arch.n_layers))
        n_moe = max(0, arch.n_layers - n_dense)
        if n_dense > 0 and n_moe > 0:
            ffn_per_gpu = ffn // tp_degree
            t_up_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            t_gate_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            t_down_d, _, _ = _matmul_cost(M, d, ffn_per_gpu, ffn_prec, hw, lattice_hw, tp_degree, calibration)
            dense_layer_s = t_up_d + t_gate_d + t_down_d
            w_d = n_dense / arch.n_layers
            w_m = n_moe / arch.n_layers
            breakdown.ffn_up_s = w_m * moe_cost["compute_s"] + w_d * dense_layer_s
            breakdown.ffn_down_s = 0.0
            breakdown.shared_expert_s = w_m * moe_cost["shared_expert_s"]
            breakdown.alltoall_s = w_m * moe_cost["alltoall_s"]
            breakdown.expert_load_s = w_m * moe_cost["expert_load_s"]
            breakdown.load_balance_factor = moe_cost["load_balance_factor"]
        else:
            breakdown.ffn_up_s = moe_cost["compute_s"]
            breakdown.ffn_down_s = 0.0
            breakdown.shared_expert_s = moe_cost["shared_expert_s"]
            breakdown.alltoall_s = moe_cost["alltoall_s"]
            breakdown.expert_load_s = moe_cost["expert_load_s"]
            breakdown.load_balance_factor = moe_cost["load_balance_factor"]

    # --- Memory-bound ops ---
    breakdown.membound_ops_s = _membound_ops_cost(B, S if phase != "decode" else 1, d, activation_prec, hw)

    # --- Communication ---
    # Wave 17 Part 2: pass phase through so decode picks lower overlap +
    # picks up the per-collective launch-latency floor.
    breakdown.allreduce_s = _allreduce_cost(
        B, S if phase != "decode" else 1, d, activation_prec, hw, tp_degree,
        phase=phase,
    )

    # --- Aggregate ---
    compute_total = (breakdown.qkv_proj_s + breakdown.attention_s +
                     breakdown.out_proj_s + breakdown.ffn_up_s + breakdown.ffn_down_s)
    memory_total = breakdown.membound_ops_s
    # MoE all-to-all is part of communication; expert weight loading is already
    # folded into ffn_up_s (compute_s = max(compute, expert_load) in decode).
    comm_total = breakdown.allreduce_s + breakdown.alltoall_s

    breakdown.compute_s = compute_total
    breakdown.memory_s = memory_total
    breakdown.communication_s = comm_total

    # v0: no compute-communication overlap
    breakdown.total_s = compute_total + memory_total + comm_total

    # MoE-aware bottleneck identification.
    if arch.moe_config is not None and phase == "decode" and breakdown.expert_load_s > 0:
        # If expert loading is the dominant term inside ffn_up_s, surface that.
        if breakdown.expert_load_s >= 0.5 * breakdown.ffn_up_s and breakdown.ffn_up_s >= 0.5 * compute_total:
            breakdown.bottleneck = "expert_load"
        elif breakdown.alltoall_s > max(compute_total, memory_total) * 0.5:
            breakdown.bottleneck = "alltoall"
        elif compute_total >= max(memory_total, comm_total):
            breakdown.bottleneck = "compute"
        elif comm_total >= memory_total:
            breakdown.bottleneck = "communication"
        else:
            breakdown.bottleneck = "memory"
    else:
        if arch.moe_config is not None and breakdown.alltoall_s > max(compute_total, memory_total) * 0.5:
            breakdown.bottleneck = "alltoall"
        elif compute_total >= max(memory_total, comm_total):
            breakdown.bottleneck = "compute"
        elif comm_total >= memory_total:
            breakdown.bottleneck = "communication"
        else:
            breakdown.bottleneck = "memory"

    return breakdown


def compute_heterogeneous_layer_times(
    arch: ArchConfig,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    phase: str = "training",
    kv_cache_len: int = 0,
    calibration: Optional[CalibrationTable] = None,
    ep_degree: int = 1,
    ep_topology: str = "single_axis",
) -> List[LayerBreakdown]:
    """
    Compute per-layer costs for a heterogeneous (hybrid) architecture.

    Returns a list of LayerBreakdown, one per layer, respecting each layer's
    type (attention or state).
    """
    layer_types = arch.layer_type_list or (["attention"] * arch.n_layers)
    results = []
    for i in range(arch.n_layers):
        lt = layer_types[i] if i < len(layer_types) else "attention"
        bd = compute_layer_time(
            arch, hw, lattice_hw, tp_degree, phase,
            kv_cache_len=kv_cache_len,
            calibration=calibration,
            ep_degree=ep_degree,
            ep_topology=ep_topology,
            layer_type=lt,
        )
        results.append(bd)
    return results


def throughput(
    arch: ArchConfig,
    hardware: str,         # "h100", "b200", "tpu_v5e", "tpu_v5p"
    tp_degree: int = 1,
    pp_degree: int = 1,
    microbatches: Optional[int] = None,
    dp_degree: Optional[int] = None,
    decode_kv_len: int = 1024,
    prefill_seq_len: Optional[int] = None,
    training_seq_len: Optional[int] = None,
    lattice_hw_override: str = None,
    ep_degree: int = 1,                # v1 MoE: expert-parallel degree
    ep_topology: str = "single_axis",  # v1 MoE: TPU torus axis layout
) -> ThroughputResult:
    """
    T(A, H, lattice) -> ThroughputResult

    Main entry point. Computes training throughput, prefill time, and
    decode time per token for a given (architecture, hardware) pair.

    -----------------------------------------------------------------
    DEFERRED (Wave 1 Step 1.6.5): bandwidth-contention sharing
    -----------------------------------------------------------------
    The current critical path sums comm times for TP allreduce, DP grad
    sync, EP all-to-all, CP all-gather, and PP send/recv as if they
    each owned the fabric. In a 4D-parallel training deployment
    (TP×PP×DP×CP, with EP partitioning DP)
    several of these collectives can land in the same window on the
    same NVLink / IB fabric and should split the available bandwidth:

        effective_bw_for_collective_c = bw / N_overlapping_at_window

    Implementing this requires:
      1. Annotating each collective with a "window phase" (forward-attn,
         forward-FFN, backward-attn, backward-FFN, gradient-sync).
      2. Tracking which collectives co-occur per phase given the chosen
         schedule (1F1B reorders backward differently than GPipe).
      3. Reshaping the critical-path computation in this function to
         compute per-window exposed comm rather than summing terms.

    Deferred to v2 because it's a structural change. For 1-2 dimensional
    parallelism (TP-only, DP-only, TP+PP) the current model is within
    20% of reality; the gap is largest at 4D-parallel 1000+ GPU runs.
    See plan/redesign/01-parallelism-fixes.md Step 1.6.5.
    -----------------------------------------------------------------

    Args:
        arch: Architecture configuration.
        hardware: Hardware target name.
        tp_degree: Tensor parallelism degree.
        pp_degree: Pipeline parallelism degree.
        microbatches: Number of pipeline microbatches. This is independent of
            the data-parallel replica count.
        dp_degree: Data-parallel degree used for gradient synchronization and
            FSDP/ZeRO-3 memory sharding.
        decode_kv_len: KV cache length for decode phase estimate.
        prefill_seq_len: Prompt length for cold-prefill TTFT. Defaults to
            arch.seq_len for backward compatibility; callers should pass the
            serving prompt/context length explicitly for long-context studies.
        training_seq_len: Sequence length used for training throughput and
            activation memory. Defaults to arch.seq_len. Compiler callers
            pass the independently declared pretraining context.
        lattice_hw_override: Override lattice hardware name (if different from throughput hw).
    """
    hw = load_hardware(hardware)
    # Wave 21: EP=1 MoE is a legal execution plan, not an error. With TP>1
    # the experts are TP-sharded like any FFN (each rank holds a
    # 1/tp-slice of EVERY expert) — this is the standard vLLM/TRT-LLM
    # Mixtral TP8 deployment, and the parameter ledger already shards
    # expert params by tp×pp×ep. The old guard ("EP=1 would replicate all
    # experts") was wrong for tp>1, and its blast radius was large: the
    # trust audit fabricated EP=2 for MoE anchors whose published
    # benchmarks ran TP-only, inflating the modeled GPU count 2× and
    # manufacturing a uniform −50…−62% per-GPU memory "model bias" (plus
    # inflated TBT/TTFT errors) that got baked into family_bias_v1.json.
    # The MoE cost path needs no special-casing at ep=1: all-to-all
    # degenerates to a ~1us local dispatch, every routed token stays
    # local, and expert_dim is already TP-sharded.
    dp_degree = max(
        1, int(dp_degree if dp_degree is not None else getattr(arch, "dp_degree", 1))
    )
    if arch.moe_config and dp_degree > 1 and (
        ep_degree > dp_degree or dp_degree % max(1, ep_degree) != 0
    ):
        raise ValueError(
            f"EP={ep_degree} must divide DP={dp_degree} when EP overlays "
            "the DP dimension"
        )
    microbatches = max(
        1,
        int(
            microbatches
            if microbatches is not None
            else getattr(arch, "pipeline_microbatches", 1)
        ),
    )
    cal_table = load_calibration(hardware)
    # Wave 7b.4: publish the per-hw calibration on a module-level holder so
    # helpers that don't take a cal_table argument (e.g. `_allreduce_cost`)
    # can consult the fitted overrides. The throughput model is synchronous,
    # so each top-level `throughput()` call overwrites this before doing
    # work; we deliberately do not use try/finally because (a) the model
    # never holds external resources we'd need to release on exception,
    # (b) the next throughput() call always sets a fresh value, and
    # (c) wrapping the 450-line body in try/finally just to clear a
    # cache pointer is structural noise.
    global _CURRENT_CALIBRATION
    _CURRENT_CALIBRATION = cal_table

    # Map throughput hardware name to lattice hardware name
    lattice_hw_name = lattice_hw_override or hardware
    if lattice_hw_name not in LATTICE_HARDWARE:
        # Fallback: try closest match
        if "tpu" in lattice_hw_name:
            lattice_hw_name = "tpu_v5p" if "v5p" in lattice_hw_name else "tpu_v5e"
        else:
            lattice_hw_name = "h100"
    lattice_hw = LATTICE_HARDWARE[lattice_hw_name]

    result = ThroughputResult(
        hardware_name=hw.accelerator_name,
        precision=arch.precision,
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        dp_degree=dp_degree,
        ep_degree=ep_degree,
        cp_degree=max(1, int(getattr(arch, "cp_degree", 1) or 1)),
        pipeline_microbatches=microbatches,
        decode_kv_cache_length=decode_kv_len,
        serving_scheduler=str(getattr(arch, "serving_scheduler", "continuous")),
        effective_serving_batch=max(1, int(arch.batch_size)),
    )

    layers_per_stage = arch.n_layers // max(pp_degree, 1)
    cal = hw.calibration

    # Per-layer kernel launch overhead (real systems pay ~5-8μs per kernel launch)
    kernel_overhead_per_layer_s = (
        cal.get("kernel_launch_overhead_us", 5.0) *
        cal.get("kernels_per_layer", 12) * 1e-6
    )

    # Check if we have a heterogeneous (hybrid) architecture.
    # Wave 18g: a local:global attention interleave is also heterogeneous —
    # local layers cost differently in decode KV reads and prefill span.
    is_hybrid = (arch.layer_type_list is not None and
                 ((any(lt == "state" for lt in arch.layer_type_list) and
                   arch.state_config is not None) or
                  any(lt == "local_attention" for lt in arch.layer_type_list)))

    # Kernel launch overhead across all layers
    # Wave 17 Part 1 fix (Jun 2026): training/prefill see only this stage's
    # layers per step, but decode walks every layer for every new token
    # (autoregressive), so the decode kernel-overhead total must scale with
    # arch.n_layers, not layers_per_stage.
    kernel_overhead_total = kernel_overhead_per_layer_s * layers_per_stage
    kernel_overhead_decode_total = kernel_overhead_per_layer_s * arch.n_layers
    prefill_arch = arch
    if prefill_seq_len is not None and int(prefill_seq_len) != arch.seq_len:
        prefill_arch = replace(arch, seq_len=max(1, int(prefill_seq_len)))
    # v1-fix audit (TTFT-per-user): TTFT is a per-request SLO, not a
    # batched-throughput metric. Previously prefill_arch inherited
    # arch.batch_size = serving_batch (typically 32), so the reported
    # prefill_time_ms was the latency to GEMM all 32 prompts in one shot —
    # ~30× larger than the per-user first-token latency that ttft budgets
    # are written against. We force batch_size=1 for the prefill phase
    # so the reported TTFT is comparable to constraints.serving_ttft_ms.
    # Batched-prefill throughput (if needed) can still be derived from
    # decode/training metrics; the per-user latency is what matters for
    # SLO feasibility.
    prefill_arch = replace(prefill_arch, batch_size=1)

    # v1-fix audit (training-batch-per-replica): training throughput must
    # not depend on the serving SLO. Previously every phase inherited
    # `arch.batch_size = constraints.serving_batch`, so when the optimizer
    # explored a tight TBT cell (e.g. serving_batch=2) the training-step
    # matmul shapes shrank to B=2 and slid into the memory-bound regime —
    # making the reported `train_tps` 12-21% lower for *identical*
    # architectures whose only difference was the serving SLO knob.
    # In real training, per-replica micro-batch is set independently of
    # serving (4-32 is typical: Llama-2/3, DeepSeek-V3). We pin a sensible
    # default here; future work: thread an explicit
    # `constraints.training_micro_batch` through evaluate_candidate.
    TRAINING_MICRO_BATCH_PER_REPLICA = 8
    configured_training_mb = int(
        getattr(arch, "training_micro_batch", 0) or 0
    )
    train_arch = replace(
        arch,
        seq_len=max(1, int(training_seq_len or arch.seq_len)),
        batch_size=(
            configured_training_mb
            if configured_training_mb > 0
            else TRAINING_MICRO_BATCH_PER_REPLICA
        ),
    )
    result.training_sequence_length = int(train_arch.seq_len)

    if is_hybrid:
        # --- Heterogeneous path: sum per-layer costs ---
        train_layers = compute_heterogeneous_layer_times(
            train_arch, hw, lattice_hw, tp_degree, "training",
            calibration=cal_table, ep_degree=ep_degree, ep_topology=ep_topology,
        )
        # Sum costs for layers in this pipeline stage
        # Simplified: use first layers_per_stage layers
        raw_train_s = sum(bd.total_s for bd in train_layers[:layers_per_stage])
        layer_train = train_layers[0]  # Representative for breakdown
        train_efficiency_layers = train_layers[:layers_per_stage]

        prefill_layers = compute_heterogeneous_layer_times(
            prefill_arch, hw, lattice_hw, tp_degree, "prefill",
            calibration=cal_table, ep_degree=ep_degree, ep_topology=ep_topology,
        )
        raw_prefill_s = sum(bd.total_s for bd in prefill_layers[:layers_per_stage])
        layer_prefill = prefill_layers[0]
        prefill_efficiency_layers = prefill_layers[:layers_per_stage]

        decode_layers = compute_heterogeneous_layer_times(
            arch, hw, lattice_hw, tp_degree, "decode",
            kv_cache_len=decode_kv_len, calibration=cal_table,
            ep_degree=ep_degree, ep_topology=ep_topology,
        )
        # Wave 17 Part 1 fix (Jun 2026): decode is autoregressive — every
        # new token must traverse all `arch.n_layers` sequentially regardless
        # of PP. The prior code used `layers_per_stage = n_layers / pp_degree`
        # for raw_decode_s, which made PP look like a linear decode speedup
        # (1000B@TP=32,PP=8 reported 6.9ms TBT — *lower* than 7B@TP=4 at
        # 12.8ms — physically impossible). PP only helps TRAINING (microbatch
        # pipelining); at decode each token still walks the whole stack.
        # We use sum of ALL per-layer costs from the heterogeneous path; the
        # first `layers_per_stage` slice is a leftover from when this branch
        # mistakenly mirrored the training/prefill subsampling.
        # See plan/redesign/17-pp-decode-serving-cost-fix.md.
        raw_decode_s = sum(bd.total_s for bd in decode_layers[:arch.n_layers])
        layer_decode = decode_layers[0]
        decode_efficiency_layers = decode_layers[:arch.n_layers]
    else:
        # --- Uniform path (v0 behavior) ---
        layer_train = compute_layer_time(train_arch, hw, lattice_hw, tp_degree, "training",
                                         calibration=cal_table,
                                         ep_degree=ep_degree, ep_topology=ep_topology)
        raw_train_s = layer_train.total_s * layers_per_stage
        train_efficiency_layers = [layer_train]

        layer_prefill = compute_layer_time(prefill_arch, hw, lattice_hw, tp_degree, "prefill",
                                           calibration=cal_table,
                                           ep_degree=ep_degree, ep_topology=ep_topology)
        raw_prefill_s = layer_prefill.total_s * layers_per_stage
        prefill_efficiency_layers = [layer_prefill]

        layer_decode = compute_layer_time(
            arch, hw, lattice_hw, tp_degree, "decode",
            kv_cache_len=decode_kv_len, calibration=cal_table,
            ep_degree=ep_degree, ep_topology=ep_topology,
        )
        # Wave 17 Part 1 fix (Jun 2026): decode walks all `arch.n_layers`
        # regardless of PP — see the hybrid branch comment above for the
        # full reasoning. Replaces `layer_decode.total_s * layers_per_stage`
        # which made PP > 1 erroneously divide decode TBT linearly.
        raw_decode_s = layer_decode.total_s * arch.n_layers
        decode_efficiency_layers = [layer_decode]

    # Optimizer step: read params + gradients, write params + optimizer states
    # AdamW: ~12 bytes per param (fp32 master weights + m + v)
    ledger = parameter_ledger(arch)
    training_layout = training_parameter_layout(
        ledger, tp=tp_degree, pp=pp_degree, dp=dp_degree, ep=ep_degree,
    )
    training_byte_layout = training_parameter_byte_layout(
        arch, tp=tp_degree, pp=pp_degree, dp=dp_degree, ep=ep_degree,
    )
    params_per_gpu = training_layout.zero3_params
    opt_bytes = params_per_gpu * cal.get("optimizer_bytes_per_param", 12)
    optimizer_step_s = opt_bytes / hw.hbm_bandwidth_bytes_s

    # Pipeline bubble
    # v1-fix Wave 1 Step 1.5 (Jun 2026): the GPipe formula
    # (pp - 1) / (M_micro + pp - 1) was the only option in v0; 1F1B
    # cuts the bubble in half (backward of microbatch i overlaps with
    # forward of microbatch i + pp), and Megatron interleaved 1F1B with
    # v virtual stages cuts it by another factor of v.
    # Reference: Narayanan et al. (Megatron-LM v2, 2021), §3.
    if pp_degree > 1:
        M_micro = max(microbatches, 1)
        gpipe_bubble = (pp_degree - 1) / (M_micro + pp_degree - 1)
        schedule = getattr(arch, "pp_schedule", "1f1b")
        if schedule == "1f1b":
            bubble = 0.5 * gpipe_bubble
        elif schedule == "interleaved":
            v_stages = max(1, int(getattr(arch, "pp_virtual_stages", 2)))
            bubble = gpipe_bubble / v_stages
        else:  # "gpipe"
            bubble = gpipe_bubble
    else:
        bubble = 0.0
    result.bubble_fraction = bubble

    # Training: forward + backward ~ 3x forward compute
    sys_eff_train, sigma_train, bucket_train = _mixed_efficiency_for(
        "training", train_efficiency_layers, arch, cal)
    optimizer_eff, _, _ = _efficiency_for(
        "training", "memory", arch, cal)
    calibrated_optimizer_step_s = optimizer_step_s / max(optimizer_eff, 1e-9)
    train_step_s = (
        raw_train_s * 3.0 / max(sys_eff_train, 1e-9)
        + kernel_overhead_total * 3
        + calibrated_optimizer_step_s
    ) * (1 + bubble)
    optimizer_step_exposed_s = (
        calibrated_optimizer_step_s * (1 + bubble)
    )
    kernel_overhead_exposed_s = kernel_overhead_total * 3 * (1 + bubble)

    # Pipeline activation/gradient transfers. In steady-state training, each
    # stage sends activations forward and gradients backward once per
    # microbatch; the slowest boundary limits throughput. Bubble cost remains
    # separate above. Prefill/decode latency traverses every boundary.
    pp_prefill_path_s = 0.0
    if pp_degree > 1:
        train_payload = (
            train_arch.batch_size
            * train_arch.seq_len
            * train_arch.d_model
            * hw.bytes_per_elem(arch.activation_precision)
        )
        worst_boundary_s, _ = _pipeline_link_costs(
            train_payload, pp_degree, hw
        )
        result.pp_training_comm_s = 2.0 * worst_boundary_s
        train_step_s += result.pp_training_comm_s

        prefill_payload = (
            prefill_arch.batch_size
            * prefill_arch.seq_len
            * prefill_arch.d_model
            * hw.bytes_per_elem(arch.activation_precision)
        )
        _, pp_prefill_path_s = _pipeline_link_costs(
            prefill_payload, pp_degree, hw
        )
        result.pp_prefill_comm_s = pp_prefill_path_s

    # v1-fix MTP: extra training compute from MTP heads (DeepSeek-V3 §2.2).
    # Each MTP depth is a small transformer block run on the same batch,
    # roughly costing `(mtp_layers / n_layers)` of one forward+backward pass.
    # Conservative estimate: 8% per depth, capped at 20% total.
    mtp_depths = int(getattr(arch, "mtp_n_predict_depths", 0) or 0)
    mtp_layers = int(getattr(arch, "mtp_depth_n_layers", 1) or 1)
    if mtp_depths > 0:
        per_depth_overhead = min(0.20, mtp_layers / max(1, arch.n_layers))
        mtp_overhead = min(0.20, mtp_depths * per_depth_overhead)
        train_step_s *= (1 + mtp_overhead)

    # v1-fix CP: Context Parallelism — split sequence across `cp_degree` ranks.
    # Attention compute and KV memory both shrink by 1/cp; the all-gather/
    # ring-attention comm cost is (cp-1)/cp × n_attn_layers × B × S × d_model × bpe.
    # Ulysses is roughly 2× cheaper in comm than Ring (head scatter vs ring KV).
    #
    # v1-fix CP-hybrid (Jun 2026, redesign Wave 1 Step 1.1): state layers do
    # NOT need full-activation all-gather. Under chunk-parallel SSM (Mamba-2
    # production path), the comm cost across cp ranks is just the recurrent
    # SSM state per layer boundary: heads × d_state × d_head × bpe per chunk,
    # passed in log2(cp) rounds via parallel scan (Brent-Kung). For a striped
    # hybrid (e.g. 1:7 attention:state) this is ~3-5 orders of magnitude
    # smaller than the attention all-gather. Treating all layers as attention
    # over-counted CP comm by ~n_layers/n_attn_layers and made hybrids look
    # artificially expensive at long context.
    cp = max(1, int(getattr(arch, "cp_degree", 1) or 1))
    if cp > 1:
        bpe_act = hw.bytes_per_elem(arch.activation_precision)
        comm_factor = (cp - 1) / cp
        cp_method_factor = 0.5 if getattr(arch, "cp_method", "ring") == "ulysses" else 1.0

        # Split layers by type
        layer_types_for_cp = arch.layer_type_list or (["attention"] * arch.n_layers)
        n_attn_cp = sum(
            1 for lt in layer_types_for_cp
            if lt in ("attention", "local_attention")
        )
        n_state_cp = sum(1 for lt in layer_types_for_cp if lt == "state")

        # Attention layers: ring/ulysses on full activations
        attn_seq_bytes_per_layer = train_arch.batch_size * train_arch.seq_len * train_arch.d_model * bpe_act
        attn_cp_comm_bytes = comm_factor * n_attn_cp * attn_seq_bytes_per_layer * cp_method_factor

        # State layers: chunk-parallel SSM — pass the SSM state, log2(cp) rounds
        state_cp_comm_bytes = 0.0
        if n_state_cp > 0:
            sc = arch.state_config or {}
            state_n_heads = int(sc.get("n_heads", arch.n_heads))
            state_d_head = int(sc.get("d_head", 64))
            d_state = int(sc.get("d_state", 128))
            state_bytes_per_layer = (train_arch.batch_size * state_n_heads
                                     * d_state * state_d_head * bpe_act)
            state_cp_comm_bytes = (math.log2(max(2, cp)) * n_state_cp
                                   * state_bytes_per_layer)

        cp_comm_bytes = attn_cp_comm_bytes + state_cp_comm_bytes
        # All-gather over NVLink within the CP group
        cp_bw = hw.interconnect_bw_bytes_s(cp)
        cp_comm_s = cp_comm_bytes / max(1.0, cp_bw)
        # CP reduces the attention compute proportionally; sequence-parallel
        # FLOP savings are folded into raw_train_s here.
        # CP partitions token work, not optimizer-state traffic or pipeline
        # transfers. Preserve those non-shardable components rather than
        # granting them an artificial 1/cp speedup.
        non_cp_shardable_s = (
            optimizer_step_exposed_s
            + kernel_overhead_exposed_s
            + result.pp_training_comm_s
        )
        cp_shardable_s = max(0.0, train_step_s - non_cp_shardable_s)
        train_step_s = (
            cp_shardable_s / cp + non_cp_shardable_s + cp_comm_s
        )

    # v1-fix Wave 1 Step 1.2 (Jun 2026): DP gradient sync.
    # The v0 model assumed aggregate_tps = per_replica × dp_degree (perfect
    # scaling). At dp ≥ 64 the gradient reduce-scatter + weight all-gather
    # cost grows with params × bpe / inter_node_bw and becomes 10-50% of
    # the training step time. Folding it into train_step_s here makes
    # `training_throughput_tokens_per_sec × dp_degree` an honest
    # aggregate.
    dp_degree_for_grad = dp_degree
    if dp_degree_for_grad > 1:
        # Wave 7b.4: pull the DP overlap fraction from the calibration
        # table when fitted; fall back to the default in the function
        # signature otherwise. The pattern is to read the override into a
        # local then pass it explicitly so the function's own default
        # remains the source of truth for "no calibration available".
        _dp_overlap_override = (
            cal_table.dp_grad_overlap_fraction if cal_table is not None else None
        )
        _dp_overlap_kw = (
            {"overlap_fraction": _dp_overlap_override}
            if _dp_overlap_override is not None else {}
        )
        # Shared and expert parameters use different replica groups when EP
        # overlays DP. Expert gradients synchronize over EDP=DP/EP, not the
        # full DP group; pricing one combined tensor double-counted EP in the
        # parameter shard and disagreed with the memory model.
        dp_grad_s = _dp_grad_allreduce_bytes_cost(
            training_byte_layout.shared_resident_bytes, hw,
            dp_degree_for_grad, zero_stage=3, **_dp_overlap_kw,
        )
        if (training_layout.expert_resident_params > 0
                and training_layout.expert_data_parallel_degree > 1):
            dp_grad_s += _dp_grad_allreduce_bytes_cost(
                training_byte_layout.expert_resident_bytes, hw,
                training_layout.expert_data_parallel_degree,
                zero_stage=3, **_dp_overlap_kw,
            )
        train_step_s = train_step_s + dp_grad_s
        result.dp_grad_allreduce_s = dp_grad_s

    # Per-replica tokens per step uses the *training* batch, not serving.
    tokens_per_step = train_arch.batch_size * train_arch.seq_len
    result.training_time_per_step_s = train_step_s
    result.training_throughput_tokens_per_sec = tokens_per_step / train_step_s if train_step_s > 0 else 0

    # --- Prefill (inference) ---
    sys_eff_prefill, sigma_prefill, bucket_prefill = _mixed_efficiency_for(
        "prefill", prefill_efficiency_layers, arch, cal)
    # Wave 35 YOCO prefill early-exit (Sun et al. 2024 §3.3): the shared KV
    # cache is produced entirely by the K self-decoder layers, and a
    # cross-decoder layer's output at position i depends only on position
    # i's input — so cold prefill only needs all S positions through the K
    # self layers, plus the LAST position through the N-K cross layers
    # (≈ a 1/S share of a layer's prefill work). This is the paper's
    # headline TTFT win and was previously not modeled at all (all N
    # layers were priced at full S×S). Training is unaffected — the
    # training path above needs every position through every layer.
    _yoco_k_pf = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
    if 0 < _yoco_k_pf < arch.n_layers:
        _pf_S = max(1, int(prefill_seq_len or arch.seq_len))
        _self_frac = _yoco_k_pf / arch.n_layers
        _cross_frac = 1.0 - _self_frac
        raw_prefill_s = raw_prefill_s * (_self_frac + _cross_frac / _pf_S)
    calibrated_prefill_layers = raw_prefill_s / max(sys_eff_prefill, 1e-9)
    if cp > 1 and (prefill_seq_len or arch.seq_len) >= 32768:
        # Mirror of the training CP block: split comm by layer type so state
        # layers pay state-pass cost (log2(cp) × state_bytes), not full-
        # activation all-gather. See Wave 1 Step 1.1 in plan/redesign/.
        bpe_act = hw.bytes_per_elem(arch.activation_precision)
        prefill_s = max(1, int(prefill_seq_len or arch.seq_len))
        comm_factor = (cp - 1) / cp
        cp_method_factor = 0.5 if getattr(arch, "cp_method", "ring") == "ulysses" else 1.0

        layer_types_for_cp = arch.layer_type_list or (["attention"] * arch.n_layers)
        n_attn_cp_stage = int(
            sum(
                1 for lt in layer_types_for_cp
                if lt in ("attention", "local_attention")
            )
            * layers_per_stage / max(1, arch.n_layers) + 0.5
        )
        n_state_cp_stage = int(
            sum(1 for lt in layer_types_for_cp if lt == "state")
            * layers_per_stage / max(1, arch.n_layers) + 0.5
        )

        attn_seq_bytes_per_layer = (prefill_arch.batch_size * prefill_s
                                    * prefill_arch.d_model * bpe_act)
        attn_cp_comm_bytes = comm_factor * n_attn_cp_stage * attn_seq_bytes_per_layer * cp_method_factor

        state_cp_comm_bytes = 0.0
        if n_state_cp_stage > 0:
            sc = arch.state_config or {}
            state_n_heads = int(sc.get("n_heads", arch.n_heads))
            state_d_head = int(sc.get("d_head", 64))
            d_state = int(sc.get("d_state", 128))
            state_bytes_per_layer = (prefill_arch.batch_size * state_n_heads
                                     * d_state * state_d_head * bpe_act)
            state_cp_comm_bytes = (math.log2(max(2, cp)) * n_state_cp_stage
                                   * state_bytes_per_layer)

        cp_comm_bytes = attn_cp_comm_bytes + state_cp_comm_bytes
        cp_bw = hw.interconnect_bw_bytes_s(cp)
        cp_prefill_comm_s = cp_comm_bytes / max(1.0, cp_bw)
        calibrated_prefill_layers = (
            calibrated_prefill_layers / cp + cp_prefill_comm_s
        )
    raw_prefill = (
        calibrated_prefill_layers
        + kernel_overhead_total
        + pp_prefill_path_s
    )
    result.prefill_time_ms = raw_prefill * 1000

    # --- Decode ---
    sys_eff_decode, sigma_decode, bucket_decode = _mixed_efficiency_for(
        "decode", decode_efficiency_layers, arch, cal)
    # Wave 17 Part 1 fix (Jun 2026): use the decode-specific kernel-overhead
    # total (which scales with arch.n_layers, not layers_per_stage) — the
    # original used kernel_overhead_total which divided by PP and made the
    # serving cost look linearly cheaper as PP grew.
    raw_decode = (
        raw_decode_s / max(sys_eff_decode, 1e-9)
        + kernel_overhead_decode_total
    )
    # Wave 17 Part 1 fix (Jun 2026): PP decode bubble. At PP > 1, every
    # token crosses (pp_degree - 1) stage boundaries; each boundary needs
    # an activation send/recv between adjacent pipeline stages. For decode
    # the activation is small (B × 1 × d_model × bpe) so this is bandwidth-
    # dominated, but latency adds up at long pipelines. Round-trip across
    # NVLink is ~3-5 µs; over IB ~10-15 µs. We use a conservative 5 µs per
    # boundary intra-island and the inter-node BW transit time outside it.
    if pp_degree > 1:
        bpe_act = hw.bytes_per_elem(arch.activation_precision)
        decode_payload = max(1, arch.batch_size) * arch.d_model * bpe_act
        _, pp_decode_path_s = _pipeline_link_costs(
            decode_payload, pp_degree, hw
        )
        result.pp_decode_comm_s = pp_decode_path_s
        raw_decode += pp_decode_path_s
    result.decode_time_per_token_ms = raw_decode * 1000

    # Scheduler launch/dispatch overhead. Kernel service time is already
    # modeled above; this adds only the control-plane work that differs by
    # scheduler. Values are calibration knobs, not hidden changes to GEMM
    # efficiency.
    scheduler = str(getattr(arch, "serving_scheduler", "continuous"))
    scheduler_overheads = cal.get("scheduler_overhead_us", {}) or {}
    default_overheads = {"static": 0.0, "continuous": 2.0, "chunked": 4.0}
    scheduler_overhead_us = float(
        scheduler_overheads.get(
            scheduler, default_overheads.get(scheduler, 0.0)
        )
    )
    result.decode_time_per_token_ms += scheduler_overhead_us / 1000.0
    if scheduler == "chunked":
        chunk_size = max(1, int(getattr(arch, "prefill_chunk_size", 65536)))
        n_chunks = max(1, math.ceil(int(prefill_seq_len or arch.seq_len) / chunk_size))
        result.prefill_time_ms += (
            n_chunks * scheduler_overhead_us / 1000.0
        )

    # --- Sigma propagation (v1-fix #28) ---
    # Convert efficiency sigma into absolute uncertainty on each phase's
    # reported metric so downstream tooling (optimizer._contending_family,
    # report renderers) can read it directly. Sigma is propagated as
    #   sigma_metric = metric * (sigma_eff / mu_eff)
    # because metric ∝ 1/efficiency.
    if sys_eff_train > 0:
        train_rel_sigma = sigma_train / sys_eff_train
        result.training_throughput_sigma_tps = (
            result.training_throughput_tokens_per_sec * train_rel_sigma)
    if sys_eff_prefill > 0:
        result.prefill_time_sigma_ms = (
            result.prefill_time_ms * (sigma_prefill / sys_eff_prefill))
    if sys_eff_decode > 0:
        result.decode_time_sigma_ms = (
            result.decode_time_per_token_ms * (sigma_decode / sys_eff_decode))
    # Record the bucket used for the dominant inference phase so the report
    # can show which calibrated cell drove the prediction.
    result.efficiency_bucket = (
        f"decode={bucket_decode}; prefill={bucket_prefill}; "
        f"training={bucket_train}"
    )

    # --- Memory ---
    # v1-fix long-ctx (June 2026): pass cp through so the KV cache and
    # CP-sharded activations get their 1/cp credit; without this every
    # long-ctx cell was infeasible by KV alone regardless of cp_degree.
    if is_hybrid:
        mem_bytes = estimate_memory_per_gpu_hybrid(
            arch, tp_degree, pp_degree,
            include_kv_cache=True, kv_cache_len=decode_kv_len,
            ep_degree=ep_degree, cp_degree=cp,
        )
    else:
        mem_bytes = estimate_memory_per_gpu(
            arch, tp_degree, pp_degree,
            include_kv_cache=True, kv_cache_len=decode_kv_len,
            ep_degree=ep_degree, cp_degree=cp,
        )
    result.memory_footprint_per_gpu_gb = mem_bytes / (1024**3)

    # v1-fix demo-audit-2 (Jun 2026): training memory under FSDP/ZeRO-3.
    # Components per GPU after sharding across dp:
    #   - weights (bpe bytes/param), sharded across (TP * dp)
    #   - grads (bpe bytes/param), sharded across (TP * dp)
    #   - optimizer states (cal.optimizer_bytes_per_param, default 12 for
    #     AdamW = 4 fp32 master + 4 m + 4 v), sharded across (TP * dp)
    #   - activations: same per-replica cost as inference activations,
    #     scaled by training_micro_batch / serving_batch
    # We surface this so the optimizer / report can flag training-infeasible
    # configs even when inference-memory fits.
    try:
        opt_bpp = float(cal.get("optimizer_bytes_per_param", 12))
        zero3_params = training_layout.zero3_params
        weight_bytes = training_byte_layout.zero3_bytes
        grad_bytes = training_byte_layout.zero3_bytes
        opt_bytes_total = zero3_params * opt_bpp
        # Activations follow the same sequence-parallel/checkpoint/pipeline
        # accounting as estimate_memory_per_gpu. Previously this used the
        # serving context and replicated it on every TP/CP rank, making CP a
        # silent no-op for training memory.
        train_mb = max(1, int(train_arch.batch_size))
        layer_factor = 4.0 + math.sqrt(max(1, int(arch.n_layers)))
        schedule = getattr(arch, "pp_schedule", "1f1b")
        if pp_degree <= 1:
            pp_act_queue = 1.0
        elif schedule == "gpipe":
            pp_act_queue = float(pp_degree)
        elif schedule == "interleaved":
            pp_act_queue = (
                (pp_degree + 1) / 2
                * max(1, int(getattr(arch, "pp_virtual_stages", 2))))
        else:
            pp_act_queue = (pp_degree + 1) / 2
        bpe_activation = hw.bytes_per_elem(arch.activation_precision)
        act_bytes_train = (
            train_mb * train_arch.seq_len * arch.d_model * bpe_activation
            * layer_factor * pp_act_queue
            / (max(1, int(tp_degree)) * max(1, int(cp)))
        )
        train_total = weight_bytes + grad_bytes + opt_bytes_total + act_bytes_train
        result.training_memory_per_gpu_gb = train_total / (1024 ** 3)
    except Exception:
        # If anything is missing (e.g. estimate_params unavailable for
        # exotic state-space configs), leave at 0.0 and skip the check.
        result.training_memory_per_gpu_gb = 0.0

    # --- HBM-overflow → continuous spill penalty (Wave 2a Step 2a.2) ---
    # When memory_per_gpu exceeds HBM, replace the v0 hard-cap with a
    # smooth bandwidth-tiered penalty. Decode is HBM-bandwidth bound (weights
    # + KV stream), so the overflow fraction effectively reads from a slower
    # tier (NVLink first, then PCIe / IB). Prefill is more compute-bound,
    # so it's hit by only ~30% of the same multiplier.
    #
    # Tiers, per GPU:
    #   - HBM:    full hbm_bandwidth_bytes_s
    #   - NVLink: intra_node_bw_gb_s × 1e9 (3-5× slower than HBM)
    #   - PCIe:   ~64 GB/s default (PCIe 5) — 50-100× slower than HBM
    #   - IB:     inter_node_bw_gb_s — comparable to PCIe on most clusters
    #
    # NVLink pool ≈ (gpus_per_node - 1) × HBM (one rank's worth per peer).
    # Anything past that spills to PCIe (or remote node via IB; we pick min).
    result.tbt_ms_no_spill = result.decode_time_per_token_ms
    result.ttft_ms_no_spill = result.prefill_time_ms
    hbm_bytes_per_gpu = float(hw.hbm_capacity_gb) * (1024 ** 3)
    mem_bytes_for_spill = float(result.memory_footprint_per_gpu_gb) * (1024 ** 3)
    overflow_bytes = max(0.0, mem_bytes_for_spill - hbm_bytes_per_gpu)
    if overflow_bytes > 0:
        nvlink_pool_bytes = max(1, hw.gpus_per_node - 1) * hbm_bytes_per_gpu
        nvlink_bytes = min(overflow_bytes, nvlink_pool_bytes)
        pcie_bytes = max(0.0, overflow_bytes - nvlink_pool_bytes)

        hbm_bw = hw.hbm_bandwidth_bytes_s
        nvlink_bw = max(1.0, hw.interconnect["intra_node_bw_gb_s"] * 1e9)
        pcie_bw_default = hw.interconnect.get("pcie_bw_gb_s", 64) * 1e9
        inter_bw = max(1.0, hw.interconnect["inter_node_bw_gb_s"] * 1e9)
        # If the user has many remote nodes, IB might be faster than PCIe;
        # take the better of the two as the second-tier spill bandwidth.
        spill_far_bw = max(pcie_bw_default, inter_bw)

        # Effective bandwidth as a weighted harmonic mean across tiers.
        hbm_frac = (mem_bytes_for_spill - overflow_bytes) / mem_bytes_for_spill
        nvlink_frac = nvlink_bytes / mem_bytes_for_spill
        far_frac = pcie_bytes / mem_bytes_for_spill
        inv_bw = (hbm_frac / hbm_bw
                  + nvlink_frac / nvlink_bw
                  + far_frac / spill_far_bw)
        effective_bw = 1.0 / inv_bw if inv_bw > 0 else hbm_bw
        decode_spill_factor = hbm_bw / max(effective_bw, 1.0)
        # Prefill is partially compute-bound; apply only 30% of the
        # bandwidth penalty so we don't double-count the compute path.
        prefill_spill_factor = 1.0 + 0.3 * (decode_spill_factor - 1.0)

        result.decode_time_per_token_ms *= decode_spill_factor
        result.prefill_time_ms *= prefill_spill_factor
        result.hbm_spill_gb = overflow_bytes / (1024 ** 3)
        if pcie_bytes <= 0:
            result.spill_tier = "nvlink"
        elif nvlink_bytes <= 0:
            result.spill_tier = "pcie"
        else:
            result.spill_tier = "mixed"
    else:
        result.hbm_spill_gb = 0.0
        result.spill_tier = "fits"

    # Wave 29: additive serving-stack TTFT floor (tokenize + scheduler
    # admission + sampler + detokenize + transport). Applied AFTER the
    # spill adjustment because it is control-plane latency, not HBM
    # traffic. Excludes load-dependent queueing — see the module-level
    # note at DEFAULT_TTFT_FIXED_OVERHEAD_MS. Calibratable per hardware
    # target via calibration["ttft_serving_overhead"].
    _ttft_ovh_cfg = cal.get("ttft_serving_overhead", {}) or {}
    _ttft_fixed_ms = float(
        _ttft_ovh_cfg.get("fixed_ms", DEFAULT_TTFT_FIXED_OVERHEAD_MS))
    _ttft_per_tok_us = float(
        _ttft_ovh_cfg.get("per_prompt_token_us",
                          DEFAULT_TTFT_PER_PROMPT_TOKEN_US))
    _ttft_prompt_tokens = max(1, int(prefill_seq_len or arch.seq_len))
    result.ttft_serving_overhead_ms = (
        _ttft_fixed_ms + _ttft_per_tok_us * _ttft_prompt_tokens / 1000.0)
    result.prefill_time_ms += result.ttft_serving_overhead_ms
    result.ttft_ms_no_spill += result.ttft_serving_overhead_ms

    result.serving_request_latency_ms = (
        result.prefill_time_ms
        + max(1, int(getattr(arch, "serving_output_len", 1)))
        * result.decode_time_per_token_ms
    )

    # Per-layer breakdowns for all phases
    result.per_layer_breakdown = layer_train
    result.prefill_layer_breakdown = layer_prefill
    result.decode_layer_breakdown = layer_decode
    result.bottleneck = layer_train.bottleneck

    return result


# =============================================================================
# Convenience: evaluate a known architecture
# =============================================================================

def evaluate_known(
    arch_name: str,
    hardware: str = "h100",
    precision: str = "bf16",
    tp_degree: int = 1,
    pp_degree: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    decode_kv_len: int = 1024,
) -> ThroughputResult:
    """Evaluate a known architecture (e.g., 'Llama-2-7B') on a hardware target."""
    if arch_name not in KNOWN_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Known: {list(KNOWN_ARCHITECTURES.keys())}")

    ka = KNOWN_ARCHITECTURES[arch_name]

    # Determine n_kv_heads (GQA info for known architectures)
    gqa_map = {
        "Llama-2-7B": None,   # MHA
        "Llama-2-13B": None,  # MHA
        "Llama-2-70B": 8,     # GQA
        "Llama-3-8B": 8,      # GQA
        "Llama-3-70B": 8,     # GQA
        "Mistral-7B": 8,      # GQA
        "Gemma-2-9B": 8,      # GQA
        "Qwen3-8B": 8,        # GQA
        "Qwen3-32B": 8,       # GQA
        # MoE models (dense-equivalent n_kv_heads)
        "DeepSeek-V3": 128,   # MLA (all heads act as KV via latent decompression)
        "Kimi-K2.5": 64,      # MLA
        "GLM-5.1": 64,        # MLA
        "GPT-OSS-120B": 8,    # GQA
        "MAI-Base-1": 8,      # GQA
    }
    n_kv = gqa_map.get(arch_name) or ka["n_heads"]

    vocab_map = {
        "Llama-2-7B": 32000, "Llama-2-13B": 32000, "Llama-2-70B": 32000,
        "Llama-3-8B": 128256, "Llama-3-70B": 128256,
        "Mistral-7B": 32000,
        "Gemma-2-9B": 256000,
        "Qwen3-8B": 151936, "Qwen3-32B": 151936,
        "DeepSeek-V3": 129280, "Kimi-K2.5": 163840,
        "GLM-5.1": 154880, "GPT-OSS-120B": 201088,
        "MAI-Base-1": 141056,
    }

    arch = ArchConfig(
        d_model=ka["d_model"],
        n_layers=ka["n_layers"],
        n_heads=ka["n_heads"],
        d_head=ka["d_head"],
        n_kv_heads=n_kv,
        ffn_dim=ka["ffn_dim"],
        ffn_type="swiglu",
        vocab_size=vocab_map.get(arch_name, 32000),
        batch_size=batch_size,
        seq_len=seq_len,
        precision=precision,
        kv_precision=precision,
    )
    return throughput(arch, hardware, tp_degree, pp_degree, decode_kv_len=decode_kv_len)


# =============================================================================
# Validation harness
# =============================================================================

# Published reference throughput numbers (approximate, from public benchmarks)
# Format: (arch_name, hardware, tp, batch, seq, tokens/sec or tok/s/gpu)
REFERENCE_TRAINING = {
    # H100 training throughput — tokens/sec/GPU at given batch×seq
    ("Llama-2-7B", "h100", 1, 4, 2048):   {"tokens_per_sec_gpu": 3800, "source": "Meta training infra / MLPerf approx"},
    ("Llama-3-8B", "h100", 1, 4, 2048):   {"tokens_per_sec_gpu": 3500, "source": "Meta release / community benchmarks"},
    ("Mistral-7B", "h100", 1, 4, 2048):   {"tokens_per_sec_gpu": 3600, "source": "Community benchmarks / Mistral docs"},
}

REFERENCE_DECODE = {
    # H100 decode throughput — tokens/sec at batch=1, seq=1, kv_len=1024
    ("Llama-2-7B", "h100", 1, 1, 1024):    {"tokens_per_sec": 85, "source": "vLLM benchmarks H100"},
    ("Llama-3-8B", "h100", 1, 1, 1024):    {"tokens_per_sec": 75, "source": "vLLM / TensorRT-LLM benchmarks"},
    ("Mistral-7B", "h100", 1, 1, 1024):    {"tokens_per_sec": 80, "source": "vLLM benchmarks"},
}


def run_validation(verbose: bool = True) -> dict:
    """
    Run validation against reference throughput numbers.
    Returns dict of results with predicted/measured ratios.
    """
    results = {"training": [], "decode": []}

    # Training validation
    for (arch_name, hw, tp, batch, seq), ref in REFERENCE_TRAINING.items():
        try:
            r = evaluate_known(arch_name, hw, "bf16", tp, 1, batch, seq)
            predicted = r.training_throughput_tokens_per_sec
            measured = ref["tokens_per_sec_gpu"]
            ratio = predicted / measured if measured > 0 else 0
            error_pct = abs(ratio - 1.0) * 100
            status = "PASS" if error_pct <= 25 else "FAIL"

            entry = {
                "arch": arch_name, "hardware": hw, "tp": tp,
                "batch": batch, "seq": seq,
                "predicted": round(predicted, 1),
                "measured": measured,
                "ratio": round(ratio, 3),
                "error_pct": round(error_pct, 1),
                "status": status,
                "source": ref["source"],
            }
            results["training"].append(entry)

            if verbose:
                print(f"  [TRAIN {status}] {arch_name} on {hw} TP={tp} B={batch} S={seq}: "
                      f"predicted={predicted:.0f} measured={measured} ratio={ratio:.3f} error={error_pct:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {arch_name} on {hw}: {e}")

    # Decode validation
    for (arch_name, hw, tp, batch, kv_len), ref in REFERENCE_DECODE.items():
        try:
            r = evaluate_known(arch_name, hw, "bf16", tp, 1, batch, 1, decode_kv_len=kv_len)
            predicted_tbt = r.decode_time_per_token_ms
            predicted_tps = 1000 / predicted_tbt if predicted_tbt > 0 else 0
            measured = ref["tokens_per_sec"]
            ratio = predicted_tps / measured if measured > 0 else 0
            error_pct = abs(ratio - 1.0) * 100
            status = "PASS" if error_pct <= 25 else "FAIL"

            entry = {
                "arch": arch_name, "hardware": hw, "tp": tp,
                "batch": batch, "kv_len": kv_len,
                "predicted_tps": round(predicted_tps, 1),
                "predicted_tbt_ms": round(predicted_tbt, 2),
                "measured_tps": measured,
                "ratio": round(ratio, 3),
                "error_pct": round(error_pct, 1),
                "status": status,
                "source": ref["source"],
            }
            results["decode"].append(entry)

            if verbose:
                print(f"  [DECODE {status}] {arch_name} on {hw} TP={tp} B={batch} KV={kv_len}: "
                      f"predicted={predicted_tps:.0f} tok/s measured={measured} ratio={ratio:.3f} error={error_pct:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {arch_name} on {hw}: {e}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    print("=" * 70)
    print("Throughput Model v0 — Validation Run")
    print("=" * 70)

    # Run all known architectures across all hardware
    for hw_name in ["h100", "b200", "tpu_v5e", "tpu_v5p"]:
        print(f"\n{'='*70}")
        print(f"Hardware: {hw_name}")
        print(f"{'='*70}")

        for arch_name in KNOWN_ARCHITECTURES:
            try:
                r = evaluate_known(arch_name, hw_name, "bf16", tp_degree=1, batch_size=1, seq_len=2048)
                decode_tps = 1000 / r.decode_time_per_token_ms if r.decode_time_per_token_ms > 0 else 0
                print(f"  {arch_name:20s} | train={r.training_throughput_tokens_per_sec:8.0f} tok/s | "
                      f"prefill={r.prefill_time_ms:7.1f}ms | decode={decode_tps:6.0f} tok/s | "
                      f"mem={r.memory_footprint_per_gpu_gb:5.1f}GB | bottleneck={r.bottleneck}")
            except Exception as e:
                print(f"  {arch_name:20s} | ERROR: {e}")

    print(f"\n{'='*70}")
    print("Validation against reference numbers (H100):")
    print(f"{'='*70}")
    run_validation(verbose=True)
