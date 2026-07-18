"""Canonical architecture utilities shared by search, cost, and reporting.

AC still has phase-specific input dataclasses for the optimizer, throughput
model, and quality model.  This module is the single place where architecture
identity and parameter accounting are defined so those views cannot silently
disagree about the model being evaluated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _shared_expert_dim(moe: Dict[str, Any]) -> int:
    shared = moe.get("shared_expert")
    if isinstance(shared, dict):
        return int(shared.get("ffn_dim", 0) or 0)
    return int(moe.get("shared_dim", 0) or 0)


@dataclass(frozen=True)
class ParameterLedger:
    """Named parameter categories for one architecture.

    ``shared_params`` are sharded by TP/PP. ``expert_*`` are additionally
    sharded by EP. Active expert parameters count routed experts; total expert
    parameters count all stored experts.
    """

    embeddings: int
    norms: int
    attention: int
    dense_ffn: int
    state: int
    expert_active: int
    expert_total: int
    shared_expert: int
    mtp: int

    @property
    def shared_params(self) -> int:
        return (
            self.embeddings
            + self.norms
            + self.attention
            + self.dense_ffn
            + self.state
            + self.shared_expert
            + self.mtp
        )

    @property
    def active_params(self) -> int:
        return self.shared_params + self.expert_active

    @property
    def total_params(self) -> int:
        return self.shared_params + self.expert_total

    def local_total_params(
        self,
        *,
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
    ) -> float:
        """Parameters resident on one pipeline/expert/tensor rank."""
        shared_divisor = max(1, int(tp)) * max(1, int(pp))
        expert_divisor = shared_divisor * max(1, int(ep))
        return (
            self.shared_params / shared_divisor
            + self.expert_total / expert_divisor
        )

    def local_active_params(
        self,
        *,
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
    ) -> float:
        shared_divisor = max(1, int(tp)) * max(1, int(pp))
        expert_divisor = shared_divisor * max(1, int(ep))
        return (
            self.shared_params / shared_divisor
            + self.expert_active / expert_divisor
        )

    def as_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ParameterByteLedger:
    """Physical stored bytes for the categories in :class:`ParameterLedger`.

    Component precision is part of architecture identity. In particular,
    an ``ffn_fp8`` candidate keeps embeddings and attention weights in BF16;
    charging the whole model at the FFN precision silently turns that mode
    into ``all_fp8`` in every memory and communication estimate.
    """

    embeddings: float
    norms: float
    attention: float
    dense_ffn: float
    state: float
    expert_active: float
    expert_total: float
    shared_expert: float
    mtp: float

    @property
    def shared_bytes(self) -> float:
        return (
            self.embeddings
            + self.norms
            + self.attention
            + self.dense_ffn
            + self.state
            + self.shared_expert
            + self.mtp
        )

    @property
    def active_bytes(self) -> float:
        return self.shared_bytes + self.expert_active

    @property
    def total_bytes(self) -> float:
        return self.shared_bytes + self.expert_total

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TrainingParameterLayout:
    """Resident and ZeRO-3-owned parameters for one training rank.

    EP partitions the DP dimension. Shared parameters have a DP group of
    size ``dp``; expert parameters are resident on one EP shard and have an
    expert-data-parallel group of size ``dp / ep``. Consequently the total
    ZeRO-3-owned parameter count is independent of EP at fixed DP.
    """

    shared_resident_params: float
    expert_resident_params: float
    shared_zero3_params: float
    expert_zero3_params: float
    expert_data_parallel_degree: int

    @property
    def resident_params(self) -> float:
        return self.shared_resident_params + self.expert_resident_params

    @property
    def zero3_params(self) -> float:
        return self.shared_zero3_params + self.expert_zero3_params


def training_parameter_layout(
    arch_or_ledger: Any,
    *,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    ep: int = 1,
) -> TrainingParameterLayout:
    """Return canonical EP-over-DP training parameter ownership.

    ``dp <= 1`` retains the historical single-cell probe behavior where EP
    can be supplied without a complete training world. Real distributed
    layouts (``dp > 1``) require EP to divide DP exactly.
    """
    ledger = (
        arch_or_ledger
        if isinstance(arch_or_ledger, ParameterLedger)
        else parameter_ledger(arch_or_ledger)
    )
    tp_i = max(1, int(tp))
    pp_i = max(1, int(pp))
    dp_i = max(1, int(dp))
    ep_i = max(1, int(ep))
    if ledger.expert_total and dp_i > 1 and (
        ep_i > dp_i or dp_i % ep_i != 0
    ):
        raise ValueError(
            f"EP={ep_i} must divide DP={dp_i} when EP overlays the DP dimension"
        )

    shared_resident = ledger.shared_params / (tp_i * pp_i)
    expert_resident = ledger.expert_total / (tp_i * pp_i * ep_i)
    expert_dp = max(1, dp_i // ep_i) if ledger.expert_total else dp_i
    return TrainingParameterLayout(
        shared_resident_params=shared_resident,
        expert_resident_params=expert_resident,
        shared_zero3_params=shared_resident / dp_i,
        expert_zero3_params=expert_resident / expert_dp,
        expert_data_parallel_degree=expert_dp,
    )


def format_state_attention_ratio(n_state: int, n_attention: int) -> str:
    """Return the exact reduced ``state:attention`` layer-count ratio."""
    state = max(0, int(n_state))
    attention = max(0, int(n_attention))
    if state == 0:
        return "pure_attention"
    if attention == 0:
        return "pure_state"
    divisor = math.gcd(state, attention)
    return f"{state // divisor}:{attention // divisor}"


def parameter_ledger(arch: Any) -> ParameterLedger:
    """Return canonical active/total parameter accounting for ``arch``.

    The function intentionally accepts the optimizer, throughput, or quality
    view via attribute access.  Adding a new architecture field now requires
    updating this one ledger instead of several independent dense estimates.
    """
    d = int(_get(arch, "d_model", 0) or 0)
    n_layers = int(_get(arch, "n_layers", 0) or 0)
    n_heads = int(_get(arch, "n_heads", 0) or 0)
    d_head = int(_get(arch, "d_head", 0) or 0)
    n_kv = int(_get(arch, "n_kv_heads", n_heads) or n_heads)
    ffn_dim = int(_get(arch, "ffn_dim", 0) or 0)
    vocab = int(_get(arch, "vocab_size", 32000) or 32000)
    attention_type = str(_get(arch, "attention_type", "full") or "full")

    embeddings = 2 * vocab * d
    norms = 2 * d * n_layers

    if attention_type == "mla" and int(
        _get(arch, "mla_kv_latent_dim", _get(arch, "mla_latent_dim", 0)) or 0
    ) > 0:
        c_kv = int(
            _get(arch, "mla_kv_latent_dim", _get(arch, "mla_latent_dim", 0))
            or 0
        )
        c_q = int(_get(arch, "mla_q_latent_dim", 0) or 0)
        d_rope = int(_get(arch, "mla_rope_head_dim", 0) or 0)
        d_nope = int(_get(arch, "mla_nope_head_dim", d_head - d_rope) or 0)
        query_projection = (
            d * c_q + c_q * n_heads * (d_nope + d_rope)
            if c_q > 0
            # c_q=0 means an ordinary direct Q projection, not no Q path.
            else d * n_heads * d_head
        )
        per_layer_attention = (
            d * c_kv
            + d * d_rope
            + 2 * c_kv * n_heads * d_nope
            + query_projection
            + n_heads * d_nope * d
        )
    else:
        per_layer_attention = (
            d * d_head * n_heads
            + 2 * d * d_head * n_kv
            + d_head * n_heads * d
        )

    layer_types = _get(arch, "layer_type_list", None) or _get(
        arch, "layer_types", None
    )
    state_cfg = _get(arch, "state_config", None) or {}
    configured_state_layers = int(state_cfg.get("state_layers", 0) or 0)
    if configured_state_layers > 0:
        n_state = configured_state_layers
    elif layer_types:
        n_state = sum(1 for value in layer_types if value == "state")
    else:
        n_state = 0
    n_state = max(0, min(n_layers, n_state))
    n_attention = n_layers - n_state

    # Attention params only exist on attention layers.  State layers replace
    # attention with input projection + SSM state + output projection.
    attention = per_layer_attention * n_attention
    state = 0
    if n_state > 0:
        expansion = int(state_cfg.get("state_expansion", 2) or 2)
        state_heads = int(state_cfg.get("n_heads", n_heads) or n_heads)
        state_d_head = int(state_cfg.get("d_head", 64) or 64)
        d_state = int(state_cfg.get("d_state", 128) or 128)
        state_per_layer = (
            d * expansion * d
            + state_heads * d_state * state_d_head
            + d * d
        )
        state = state_per_layer * n_state

    moe = _get(arch, "moe", None) or _get(arch, "moe_config", None)
    n_dense_prefix = max(
        0, min(n_layers, int(_get(arch, "n_dense_ffn_layers", 0) or 0))
    )
    if moe:
        n_moe_layers = n_layers - n_dense_prefix
        dense_ffn = 3 * d * ffn_dim * n_dense_prefix
        n_experts = max(1, int(moe.get("n_experts", 1) or 1))
        top_k = max(1, min(n_experts, int(moe.get("top_k", 1) or 1)))
        expert_dim = int(moe.get("expert_dim", ffn_dim) or ffn_dim)
        expert_per_layer = 3 * d * expert_dim
        expert_active = expert_per_layer * top_k * n_moe_layers
        expert_total = expert_per_layer * n_experts * n_moe_layers
        shared_expert = (
            3 * d * _shared_expert_dim(moe) * n_moe_layers
        )
    else:
        dense_ffn = 3 * d * ffn_dim * n_layers
        expert_active = 0
        expert_total = 0
        shared_expert = 0

    mtp_depths = int(_get(arch, "mtp_n_predict_depths", 0) or 0)
    mtp_layers = int(_get(arch, "mtp_depth_n_layers", 1) or 1)
    # MTP depth blocks share embeddings/head. Approximate each block with one
    # ordinary attention + dense FFN layer, matching the throughput model.
    mtp = mtp_depths * mtp_layers * (
        per_layer_attention + 3 * d * ffn_dim + 2 * d
    )

    return ParameterLedger(
        embeddings=embeddings,
        norms=norms,
        attention=attention,
        dense_ffn=dense_ffn,
        state=state,
        expert_active=expert_active,
        expert_total=expert_total,
        shared_expert=shared_expert,
        mtp=mtp,
    )


_PRECISION_STORAGE_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "tf32": 4.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
    # OCP MX formats include one shared scale byte per 32 elements.
    "mxfp4": 0.53,
    "mxfp6": 0.78,
    # MXFP8: FP8 payload + one E8M0 scale byte per 32 values.
    "mxfp8": 1.03125,
    # NVFP4: E2M1 payload + one E4M3 scale byte per 16 values. The
    # per-tensor FP32 scale is negligible for architecture-level accounting.
    "nvfp4": 0.5625,
}


def precision_bytes_per_element(precision: str) -> float:
    """Canonical physical storage width for one precision element."""
    return _PRECISION_STORAGE_BYTES.get(str(precision or "bf16"), 2.0)


def _attention_projection_params_per_layer(arch: Any) -> Dict[str, float]:
    """Split the canonical attention count into QKV and output weights."""
    d = int(_get(arch, "d_model", 0) or 0)
    n_heads = int(_get(arch, "n_heads", 0) or 0)
    d_head = int(_get(arch, "d_head", 0) or 0)
    n_kv = int(_get(arch, "n_kv_heads", n_heads) or n_heads)
    attention_type = str(_get(arch, "attention_type", "full") or "full")
    if attention_type == "mla" and int(
        _get(arch, "mla_kv_latent_dim", _get(arch, "mla_latent_dim", 0)) or 0
    ) > 0:
        c_kv = int(
            _get(arch, "mla_kv_latent_dim", _get(arch, "mla_latent_dim", 0))
            or 0
        )
        c_q = int(_get(arch, "mla_q_latent_dim", 0) or 0)
        d_rope = int(_get(arch, "mla_rope_head_dim", 0) or 0)
        d_nope = int(_get(arch, "mla_nope_head_dim", d_head - d_rope) or 0)
        query = (
            d * c_q + c_q * n_heads * (d_nope + d_rope)
            if c_q > 0 else d * n_heads * d_head
        )
        shared_kv_down = d * c_kv
        key = d * d_rope + c_kv * n_heads * d_nope + shared_kv_down / 2
        value = c_kv * n_heads * d_nope + shared_kv_down / 2
        output = n_heads * d_nope * d
        return {"q": query, "k": key, "v": value, "output": output}
    return {
        "q": d * d_head * n_heads,
        "k": d * d_head * n_kv,
        "v": d * d_head * n_kv,
        "output": d_head * n_heads * d,
    }


def parameter_byte_ledger(arch: Any) -> ParameterByteLedger:
    """Return component-precision-aware physical parameter bytes."""
    ledger = parameter_ledger(arch)
    weight_precision = str(_get(arch, "weight_precision", "bf16") or "bf16")
    ffn_precision = str(
        _get(arch, "ffn_precision", _get(arch, "precision", weight_precision))
        or weight_precision
    )
    attn_precision = dict(_get(arch, "attn_precision", {}) or {})
    # ``v`` is AC's QKV projection storage/compute precision. ``qk`` is the
    # accumulator/logit precision and therefore does not change weight bytes.
    qkv_precision = str(attn_precision.get("v", weight_precision) or weight_precision)
    output_precision = str(
        attn_precision.get("output", attn_precision.get("o", weight_precision))
        or weight_precision
    )
    weight_bpe = precision_bytes_per_element(weight_precision)
    ffn_bpe = precision_bytes_per_element(ffn_precision)

    projection_params = _attention_projection_params_per_layer(arch)
    projection_total = sum(projection_params.values())
    projection_bytes = (
        (projection_params["q"] + projection_params["k"]
         + projection_params["v"])
        * precision_bytes_per_element(qkv_precision)
        + projection_params["output"]
        * precision_bytes_per_element(output_precision)
    )
    attention_bytes = (
        ledger.attention * projection_bytes / projection_total
        if projection_total > 0 else 0.0
    )

    state_cfg = _get(arch, "state_config", None) or {}
    state_precision = str(
        state_cfg.get("state_precision", weight_precision) or weight_precision
    )
    moe = _get(arch, "moe", None) or _get(arch, "moe_config", None) or {}
    expert_precision = str(
        moe.get("expert_precision", moe.get("precision", ffn_precision))
        or ffn_precision
    )
    shared = moe.get("shared_expert")
    shared_precision = (
        str(shared.get("precision", expert_precision) or expert_precision)
        if isinstance(shared, dict) else expert_precision
    )

    d = int(_get(arch, "d_model", 0) or 0)
    ffn_dim = int(_get(arch, "ffn_dim", 0) or 0)
    mtp_depths = int(_get(arch, "mtp_n_predict_depths", 0) or 0)
    mtp_layers = int(_get(arch, "mtp_depth_n_layers", 1) or 1)
    mtp_bytes = mtp_depths * mtp_layers * (
        projection_bytes + 3 * d * ffn_dim * ffn_bpe + 2 * d * weight_bpe
    )

    return ParameterByteLedger(
        embeddings=ledger.embeddings * weight_bpe,
        norms=ledger.norms * weight_bpe,
        attention=attention_bytes,
        dense_ffn=ledger.dense_ffn * ffn_bpe,
        state=ledger.state * precision_bytes_per_element(state_precision),
        expert_active=(
            ledger.expert_active * precision_bytes_per_element(expert_precision)
        ),
        expert_total=(
            ledger.expert_total * precision_bytes_per_element(expert_precision)
        ),
        shared_expert=(
            ledger.shared_expert * precision_bytes_per_element(shared_precision)
        ),
        mtp=mtp_bytes,
    )


@dataclass(frozen=True)
class TrainingParameterByteLayout:
    """Resident and ZeRO-3-owned parameter bytes for one training rank."""

    shared_resident_bytes: float
    expert_resident_bytes: float
    shared_zero3_bytes: float
    expert_zero3_bytes: float
    expert_data_parallel_degree: int

    @property
    def resident_bytes(self) -> float:
        return self.shared_resident_bytes + self.expert_resident_bytes

    @property
    def zero3_bytes(self) -> float:
        return self.shared_zero3_bytes + self.expert_zero3_bytes


def training_parameter_byte_layout(
    arch: Any,
    *,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    ep: int = 1,
) -> TrainingParameterByteLayout:
    """Return component-aware byte ownership for the canonical EP/DP layout."""
    params = training_parameter_layout(arch, tp=tp, pp=pp, dp=dp, ep=ep)
    byte_ledger = parameter_byte_ledger(arch)
    tp_i = max(1, int(tp))
    pp_i = max(1, int(pp))
    dp_i = max(1, int(dp))
    ep_i = max(1, int(ep))
    shared_resident = byte_ledger.shared_bytes / (tp_i * pp_i)
    expert_resident = byte_ledger.expert_total / (tp_i * pp_i * ep_i)
    return TrainingParameterByteLayout(
        shared_resident_bytes=shared_resident,
        expert_resident_bytes=expert_resident,
        shared_zero3_bytes=shared_resident / dp_i,
        expert_zero3_bytes=(
            expert_resident / params.expert_data_parallel_degree
        ),
        expert_data_parallel_degree=params.expert_data_parallel_degree,
    )


def compose_layer_type_list(
    layer_types: Any,
    n_layers: int,
    n_local_attention_layers: int = 0,
) -> list[str]:
    """Return one coherent attention/local-attention/state layer layout.

    ``CandidateArch`` stores state placement in ``layer_type_list`` and the
    local-attention axis as a count.  Those axes can coexist (for example,
    applying ``add_state_layers`` to GPT-OSS' alternating local/global
    baseline).  Rebuilding the local interleave from scratch used to erase
    state layers; preserving the state list without materialising local
    layers made the count and list describe different models.

    This helper keeps state positions fixed and distributes the requested
    number of local-attention layers only over the remaining attention slots.
    An already-coherent combined layout is returned unchanged.
    """
    n = max(0, int(n_layers))
    requested_local = max(0, int(n_local_attention_layers or 0))
    values = list(layer_types or [])
    if len(values) != n:
        values = ["attention"] * n
    else:
        values = [
            value if value in {"attention", "local_attention", "state"}
            else "attention"
            for value in values
        ]

    attention_slots = [
        idx for idx, value in enumerate(values) if value != "state"
    ]
    target_local = min(requested_local, len(attention_slots))
    existing_local = sum(
        1 for idx in attention_slots if values[idx] == "local_attention"
    )
    if existing_local == target_local:
        return values

    for idx in attention_slots:
        values[idx] = "attention"
    if target_local == 0:
        return values

    # Mirror ArchConfig.__post_init__'s even interleave, but operate only on
    # the attention-capable slots so state placement remains untouched.
    n_global = len(attention_slots) - target_local
    emitted_global = 0
    for ordinal, idx in enumerate(attention_slots):
        next_global = (ordinal + 1) * n_global // len(attention_slots)
        if next_global > emitted_global:
            emitted_global = next_global
        else:
            values[idx] = "local_attention"
    return values


# ---------------------------------------------------------------------------
# Wave 18a — Factorized architecture signature
# ---------------------------------------------------------------------------
#
# The historical family labels (`dense`, `moe`, `hybrid`, `moe_hybrid`)
# conflate independent choices.  In particular `hybrid` has been used as a
# proxy for state-space mixing, MLA, sparse/local attention, and long-context
# serving efficiency.  That is scientifically ambiguous for public-model
# comparisons.
#
# ``ArchitectureSignature`` factors identity into independent axes and lets
# downstream consumers (Pareto records, decision diagnostics, calibration
# ingestion) compare candidates on those axes rather than on a coarse family
# label.  The old family label is retained as ``legacy_family`` for
# compatibility with existing UIs and one schema-migration cycle.


FFN_MODES = ("dense", "moe")
ATTENTION_PATTERNS = ("global", "local", "block_sparse", "compressed")
KV_PROJECTIONS = ("mha", "gqa", "mqa", "mla")
SEQUENCE_MIXERS = (
    "attention",
    "mamba2",
    "mamba",
    "delta",
    "gated_delta",
    "kda",
    "gla",
    "sliding_window",
    "linear_attention",
)
CONTEXT_EXTENSIONS = ("none", "pi", "ntk", "yarn", "longrope")

_COMPRESSED_ATTN_TYPES = frozenset({"csa", "indexshare", "msa"})
_LOCAL_ATTN_TYPES = frozenset({"swa"})
_BLOCK_SPARSE_ATTN_TYPES = frozenset({"nsa"})
# Anything not in the above lists (full/gqa/mha/mqa/mla) uses the "global"
# attention pattern.  MLA is a KV projection axis; the attention *pattern* it
# runs is still global unless combined with one of the sparse types above.


@dataclass(frozen=True)
class ArchitectureSignature:
    """Factorized architecture identity — Wave 18a.

    The signature is deliberately independent of *requested* search modes.
    It is derived exclusively from the architecture object actually
    evaluated by throughput/quality/cost.  A candidate that was requested as
    ``allow_moe=True`` but returned a dense FFN is classified ``dense``.

    ``legacy_family`` is derived last and is compatibility-only.  New code
    should compare by the factorized axes; the old label is present so a
    single UI/JSON schema migration can happen atomically without breaking
    every consumer at once.
    """

    ffn_mode: str                       # dense | moe
    attention_pattern: str              # global | local | block_sparse | compressed
    kv_projection: str                  # mha | gqa | mqa | mla
    sequence_mixer: str                 # attention | mamba2 | mamba | delta | ...
    mixer_fraction: float               # fraction of non-attention mixer layers
    context_extension: str              # none | pi | ntk | yarn | longrope
    modifiers: Tuple[str, ...]          # nsa, yoco, mtp, swa_when_hybrid, etc.
    active_params: int
    total_params: int
    legacy_family: str                  # dense | moe | hybrid | moe_hybrid

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def display(self) -> str:
        """Compact human-readable form, e.g. ``moe + mla + attention``."""
        parts = [self.ffn_mode]
        if self.kv_projection != "gqa":
            parts.append(self.kv_projection)
        if self.attention_pattern != "global":
            parts.append(self.attention_pattern)
        if self.sequence_mixer != "attention":
            parts.append(f"{self.sequence_mixer}@{self.mixer_fraction:.2f}")
        if self.context_extension != "none":
            parts.append(self.context_extension)
        if self.modifiers:
            parts.append("+" + "+".join(self.modifiers))
        return " / ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ffn_mode": self.ffn_mode,
            "attention_pattern": self.attention_pattern,
            "kv_projection": self.kv_projection,
            "sequence_mixer": self.sequence_mixer,
            "mixer_fraction": float(self.mixer_fraction),
            "context_extension": self.context_extension,
            "modifiers": list(self.modifiers),
            "active_params": int(self.active_params),
            "total_params": int(self.total_params),
            "legacy_family": self.legacy_family,
        }

    def to_json(self) -> str:
        """Stable serialization for artifacts, fingerprints, and audit rows."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArchitectureSignature":
        return cls(
            ffn_mode=str(payload["ffn_mode"]),
            attention_pattern=str(payload["attention_pattern"]),
            kv_projection=str(payload["kv_projection"]),
            sequence_mixer=str(payload["sequence_mixer"]),
            mixer_fraction=float(payload.get("mixer_fraction", 0.0) or 0.0),
            context_extension=str(payload.get("context_extension", "none")),
            modifiers=tuple(payload.get("modifiers", []) or []),
            active_params=int(payload.get("active_params", 0) or 0),
            total_params=int(payload.get("total_params", 0) or 0),
            legacy_family=str(payload.get("legacy_family", "dense")),
        )

    # ------------------------------------------------------------------
    # Structural predicates (avoid re-deriving family from strings)
    # ------------------------------------------------------------------

    @property
    def is_moe(self) -> bool:
        return self.ffn_mode == "moe"

    @property
    def has_state_mixer(self) -> bool:
        return self.sequence_mixer != "attention" and self.mixer_fraction > 0.0

    @property
    def uses_mla(self) -> bool:
        return self.kv_projection == "mla"

    @property
    def uses_compressed_attention(self) -> bool:
        return self.attention_pattern in {"compressed", "block_sparse"}

    def sparsity_ratio(self) -> float:
        """total / active — 1.0 for dense, >1 for MoE."""
        if self.active_params <= 0:
            return 1.0
        return self.total_params / self.active_params


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _classify_kv_projection(arch: Any) -> str:
    """Derive kv_projection from the concrete attention shape actually used.

    The optimizer's ``CandidateArch.attention_type`` uses ``"mla"`` when the
    latent projection is active; otherwise the projection is determined by
    the ratio of query heads to KV heads.  Requested MoE/GQA/etc. names are
    irrelevant here — we look at the numbers that were actually built.
    """
    attention_type = str(_get(arch, "attention_type", "full") or "full").lower()
    if attention_type == "mla":
        c_kv = int(
            _get(arch, "mla_kv_latent_dim", _get(arch, "mla_latent_dim", 0)) or 0
        )
        if c_kv > 0:
            return "mla"
        # Fall through — labeled MLA but no latent dim = not actually MLA.

    n_heads = int(_get(arch, "n_heads", 0) or 0)
    n_kv = int(_get(arch, "n_kv_heads", n_heads) or n_heads)
    if n_heads <= 0:
        return "gqa"
    if n_kv == 1:
        return "mqa"
    if n_kv == n_heads:
        return "mha"
    return "gqa"


def _classify_attention_pattern(arch: Any) -> str:
    """Derive attention_pattern from concrete attention type / SWA window.

    NSA → block_sparse.  SWA (nonzero window) → local.  CSA/IndexShare/MSA →
    compressed.  Everything else (full / mha / gqa / mqa / mla) → global.
    """
    attention_type = str(_get(arch, "attention_type", "full") or "full").lower()
    if attention_type in _COMPRESSED_ATTN_TYPES:
        return "compressed"
    if attention_type in _BLOCK_SPARSE_ATTN_TYPES:
        return "block_sparse"
    if attention_type in _LOCAL_ATTN_TYPES:
        return "local"
    if int(_get(arch, "swa_window", 0) or 0) > 0:
        # Wave 18g: a local:global interleave (some layers windowed, some
        # global) is its own pattern — GPT-OSS / Gemma-2 / Llama-4 style.
        n_local = int(_get(arch, "n_local_attn_layers", 0) or 0)
        n_layers = int(_get(arch, "n_layers", 0) or 0)
        if 0 < n_local < n_layers:
            return "local_global"
        # SWA can be applied on top of another attention_type (e.g.
        # Mistral-7B is dense-GQA with a 4096-token sliding window).  When
        # the window is nonzero, the pattern is local.
        return "local"
    return "global"


def _classify_sequence_mixer(arch: Any) -> Tuple[str, float]:
    """Return (mixer_family, fraction).

    The mixer family names the *state* mechanism if any state layers exist.
    ``fraction`` is the share of layers that use the state mixer; 0.0 means
    pure attention.  Values live in [0, 1].
    """
    n_layers = int(_get(arch, "n_layers", 0) or 0)
    if n_layers <= 0:
        return "attention", 0.0

    state_cfg = _get(arch, "state_config", None) or {}
    layer_types = _get(arch, "layer_type_list", None) or _get(
        arch, "layer_types", None
    )
    configured_state_layers = int(state_cfg.get("state_layers", 0) or 0)
    if configured_state_layers > 0:
        n_state = configured_state_layers
    elif layer_types:
        n_state = sum(1 for value in layer_types if value == "state")
    else:
        n_state = int(_get(arch, "n_state_layers", 0) or 0)

    n_state = max(0, min(n_layers, n_state))
    if n_state == 0:
        return "attention", 0.0

    family = str(
        state_cfg.get("state_type")
        or _get(arch, "state_type", None)
        or "mamba2"
    ).lower()
    if family not in SEQUENCE_MIXERS:
        # Preserve unrecognized family verbatim so calibration ingestion can
        # see it, but the taxonomy doesn't attempt to normalize it.
        pass
    return family, n_state / n_layers


def _classify_context_extension(arch: Any) -> str:
    """Derive context_extension from RoPE scaling method actually applied."""
    method = str(_get(arch, "rope_scaling_method", "none") or "none").lower()
    if method in CONTEXT_EXTENSIONS:
        return method
    # Some code paths use the shorthand "linear" or "extended"; normalize.
    if method in {"linear"}:
        return "pi"
    return "none"


def _classify_modifiers(arch: Any) -> Tuple[str, ...]:
    """Collect orthogonal modifier flags (NSA, YOCO, MTP, SWA-when-hybrid).

    Modifiers are structural features that don't fit any of the four primary
    axes.  Kept as an ordered tuple so tests can assert stable set semantics
    without sensitivity to iteration order.
    """
    mods = []
    if int(_get(arch, "yoco_n_self_attn_layers", 0) or 0) > 0:
        mods.append("yoco")
    if int(_get(arch, "mtp_n_predict_depths", 0) or 0) > 0:
        mods.append("mtp")
    # NSA/CSA/etc. are already reflected in attention_pattern, but we also
    # record them as modifiers so calibration ingestion can query by
    # concrete mechanism name.
    attention_type = str(_get(arch, "attention_type", "full") or "full").lower()
    if attention_type in _COMPRESSED_ATTN_TYPES:
        mods.append(attention_type)
    elif attention_type in _BLOCK_SPARSE_ATTN_TYPES:
        mods.append(attention_type)
    # SWA on top of another attention type (Mistral-style hybrid attention).
    if attention_type not in _LOCAL_ATTN_TYPES and int(
        _get(arch, "swa_window", 0) or 0
    ) > 0:
        mods.append("swa")
    return tuple(sorted(set(mods)))


def _is_moe(arch: Any) -> bool:
    """True iff the architecture actually carries an enabled MoE config.

    A requested ``allow_moe=True`` that returned a dense FFN reports False —
    classification tracks the *built* architecture, not the search request.
    """
    moe = _get(arch, "moe", None) or _get(arch, "moe_config", None)
    if not moe:
        return False
    # Empty dicts / disabled placeholders don't count.
    if isinstance(moe, dict):
        if not moe:
            return False
        if moe.get("enabled") is False:
            return False
        # A single expert with top_k=1 is dense in disguise.
        n_experts = int(moe.get("n_experts", 0) or 0)
        top_k = int(moe.get("top_k", 0) or 0)
        if n_experts <= 1 and top_k <= 1:
            return False
    return True


def _derive_legacy_family(ffn_mode: str, has_state: bool) -> str:
    """Legacy family label — compatibility only, per Wave 18a §Classification."""
    if ffn_mode == "moe" and has_state:
        return "moe_hybrid"
    if ffn_mode == "moe":
        return "moe"
    if has_state:
        return "hybrid"
    return "dense"


def architecture_signature(arch: Any) -> ArchitectureSignature:
    """Derive the canonical Wave 18a signature for one architecture.

    The signature is a pure function of the concrete architecture — no
    search-request flags are read.  Active/total parameters come exclusively
    from :func:`parameter_ledger` so the four axes and the parameter counts
    cannot disagree about the model being evaluated.

    Raises
    ------
    ValueError
        If the architecture is missing the minimal fields needed to
        classify (``d_model``, ``n_layers``, ``n_heads``).  Callers that
        pass partial fixtures should populate at least those before asking
        for a signature.
    """
    if arch is None:
        raise ValueError("architecture_signature: arch is None")

    d_model = int(_get(arch, "d_model", 0) or 0)
    n_layers = int(_get(arch, "n_layers", 0) or 0)
    n_heads = int(_get(arch, "n_heads", 0) or 0)
    if d_model <= 0 or n_layers <= 0 or n_heads <= 0:
        raise ValueError(
            "architecture_signature: arch missing required shape fields "
            f"(d_model={d_model}, n_layers={n_layers}, n_heads={n_heads})"
        )

    ffn_mode = "moe" if _is_moe(arch) else "dense"
    attention_pattern = _classify_attention_pattern(arch)
    kv_projection = _classify_kv_projection(arch)
    sequence_mixer, mixer_fraction = _classify_sequence_mixer(arch)
    context_extension = _classify_context_extension(arch)
    modifiers = _classify_modifiers(arch)

    ledger = parameter_ledger(arch)
    active_params = int(ledger.active_params)
    total_params = int(ledger.total_params)

    has_state = mixer_fraction > 0.0 and sequence_mixer != "attention"
    legacy_family = _derive_legacy_family(ffn_mode, has_state)

    return ArchitectureSignature(
        ffn_mode=ffn_mode,
        attention_pattern=attention_pattern,
        kv_projection=kv_projection,
        sequence_mixer=sequence_mixer,
        mixer_fraction=float(mixer_fraction),
        context_extension=context_extension,
        modifiers=modifiers,
        active_params=active_params,
        total_params=total_params,
        legacy_family=legacy_family,
    )


def signature_fingerprint(sig: ArchitectureSignature) -> str:
    """Short, stable hex identifier for the factorized signature.

    Distinct from :func:`architecture_fingerprint`, which hashes the whole
    arch object.  ``signature_fingerprint`` is deliberately coarser: two
    architectures with the same signature (same axes + same active/total
    params) share a fingerprint so downstream code can group like-for-like
    candidates without depending on private shape fields.
    """
    encoded = sig.to_json().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def architecture_fingerprint(arch: Any) -> str:
    """Stable content identity used by selection and report joins."""
    if dataclasses.is_dataclass(arch):
        payload = dataclasses.asdict(arch)
    elif isinstance(arch, dict):
        payload = arch
    else:
        payload = {
            key: value
            for key, value in vars(arch).items()
            if not key.startswith("_")
        }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attention_family(value: Any) -> str:
    value = str(value or "full").lower()
    # CandidateArch calls ordinary grouped attention "full"; the quality
    # model calls the same implementation "gqa".
    return "gqa" if value in {"full", "mha", "gqa", "mqa"} else value


def validate_architecture_views(
    source: Any,
    throughput_view: Any,
    quality_view: Optional[Any] = None,
) -> None:
    """Fail when phase-specific architecture views describe different models.

    CandidateArch remains the optimizer-facing source of truth, while the
    cost and quality models retain purpose-built dataclasses. This contract
    makes their overlap explicit and turns forgotten wiring into an immediate
    error rather than attaching predictions to a different architecture.
    """
    mismatches = []

    def check(label: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append(f"{label}: source={expected!r}, view={actual!r}")

    for field_name in (
        "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads",
        "ffn_dim", "vocab_size", "n_dense_ffn_layers",
        "mtp_n_predict_depths", "mtp_depth_n_layers",
        "yoco_n_self_attn_layers",
    ):
        check(
            f"throughput.{field_name}",
            _get(source, field_name, 0),
            _get(throughput_view, field_name, 0),
        )

    check(
        "throughput.attention_type",
        _attention_family(_get(source, "attention_type", "full")),
        _attention_family(_get(throughput_view, "attention_type", "full")),
    )
    check(
        "throughput.weight_precision",
        _get(source, "weight_precision", "bf16"),
        _get(throughput_view, "weight_precision", "bf16"),
    )
    check(
        "throughput.ffn_precision",
        _get(source, "ffn_precision", "bf16"),
        _get(throughput_view, "precision", "bf16"),
    )
    check(
        "throughput.activation_precision",
        _get(source, "activation_precision", "bf16"),
        _get(throughput_view, "activation_precision", "bf16"),
    )
    check(
        "throughput.attn_precision",
        dict(_get(source, "attn_precision", {}) or {}),
        dict(_get(throughput_view, "attn_precision", {}) or {}),
    )

    source_moe = _get(source, "moe", None) or {}
    tput_moe = _get(throughput_view, "moe_config", None) or {}
    for field_name in ("n_experts", "top_k", "expert_dim"):
        check(
            f"throughput.moe.{field_name}",
            source_moe.get(field_name),
            tput_moe.get(field_name),
        )

    if _attention_family(_get(source, "attention_type")) == "nsa":
        for field_name in (
            "nsa_compress_block_size", "nsa_compress_block_stride",
            "nsa_select_block_size", "nsa_select_top_k", "nsa_window_size",
        ):
            check(
                f"throughput.{field_name}",
                int(_get(source, field_name, 0) or 0),
                int(_get(throughput_view, field_name, 0) or 0),
            )

    if quality_view is not None:
        for field_name in (
            "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads",
            "ffn_dim", "vocab_size", "n_dense_ffn_layers",
            "mtp_n_predict_depths", "mtp_depth_n_layers",
            "yoco_n_self_attn_layers",
        ):
            check(
                f"quality.{field_name}",
                _get(source, field_name, 0),
                _get(quality_view, field_name, 0),
            )
        check(
            "quality.attention_type",
            _attention_family(_get(source, "attention_type", "full")),
            _attention_family(_get(quality_view, "attention_type", "full")),
        )
        check(
            "quality.weight_precision",
            _get(source, "weight_precision", "bf16"),
            _get(quality_view, "weight_precision", "bf16"),
        )
        check(
            "quality.activation_precision",
            _get(source, "activation_precision", "bf16"),
            _get(quality_view, "activation_precision", "bf16"),
        )
        quality_moe = _get(quality_view, "moe_config", None) or {}
        for field_name in ("n_experts", "top_k", "expert_dim"):
            check(
                f"quality.moe.{field_name}",
                source_moe.get(field_name),
                quality_moe.get(field_name),
            )
        if _attention_family(_get(source, "attention_type")) == "nsa":
            for field_name in (
                "nsa_compress_block_size", "nsa_compress_block_stride",
                "nsa_select_block_size", "nsa_select_top_k", "nsa_window_size",
            ):
                check(
                    f"quality.{field_name}",
                    int(_get(source, field_name, 0) or 0),
                    int(_get(quality_view, field_name, 0) or 0),
                )

    source_ledger = parameter_ledger(source).as_dict()
    for view_name, view in (
        ("throughput", throughput_view),
        ("quality", quality_view),
    ):
        if view is None:
            continue
        view_ledger = parameter_ledger(view).as_dict()
        for category, expected in source_ledger.items():
            check(
                f"{view_name}.parameter_ledger.{category}",
                expected,
                view_ledger[category],
            )

    if mismatches:
        raise ValueError(
            "Architecture view wiring mismatch: " + "; ".join(mismatches)
        )
