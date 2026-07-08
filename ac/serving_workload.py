"""Wave 18c — Serving workload specification and canonical scenarios.

Per ``plan/redesign/18c-context-workloads-and-operational-flags.md``. Provides:

    * `ServingWorkloadSpec` — immutable, validated workload description that
      separates the four length axes AC previously conflated into one
      ``context_length`` field: max supported, fresh prompt, cached prefix,
      and decode KV length. Also carries topology and available-GPU limits so
      operational-flag evaluation (18c) is workload-scoped.
    * `WorkloadScenario` — enum of the five canonical scenarios spec §Canonical
      scenarios enumerates.
    * `canonical_workloads(max_context_length)` — build all five scenarios
      for one context row using the spec's default policy. Each field is
      overridable through the CLI/config layer.
    * `WorkloadRegistry` — small helper that stores a set of workloads by
      name so 18b's `BudgetMatrix` can iterate them.

Design principles:

    1. Caching does NOT eliminate decode access to the long KV state. Cached
       prefixes only avoid repeated prefill work; `decode_kv_length` is the
       length decode actually pays. This is asserted by the tests.
    2. Changing ``max_context_length`` alone must not create fresh prefill
       work. The interactive scenarios keep the fresh prompt at
       ``min(C, 8192)`` (interactive_ordinary/cached) or ``min(C, 32768)``
       (incremental_long_session); only ``cold_full_context_ingestion``
       treats the row as a cold ingestion request.
    3. Disaggregated serving is representable at the workload layer: prefill
       and decode topologies are distinct fields so downstream cost models
       can charge each phase with its own parallelism.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Enums and constants
# =============================================================================


class WorkloadScenario(str, enum.Enum):
    """Five canonical scenarios per 18c spec §Canonical scenarios."""
    INTERACTIVE_ORDINARY = "interactive_ordinary"
    INTERACTIVE_CACHED_LONG_CONTEXT = "interactive_cached_long_context"
    INCREMENTAL_LONG_SESSION = "incremental_long_session"
    COLD_FULL_CONTEXT_INGESTION = "cold_full_context_ingestion"
    UNCONSTRAINED_DIAGNOSTIC = "unconstrained_diagnostic"


# Interactive fresh-prompt cap for ordinary/cached scenarios.
INTERACTIVE_FRESH_PROMPT = 8192

# Incremental-session fresh-prompt cap.
INCREMENTAL_SESSION_FRESH_PROMPT = 32768

# Default cache-hit rates per scenario.
CACHE_HIT_RATE_CACHED = 0.90
CACHE_HIT_RATE_INCREMENTAL = 0.75


# Memory tier names. `hbm` = per-GPU HBM only. `hbm+spill` = HBM plus
# an offload/spill tier (CPU or NVMe). Passing `hbm+spill` in
# `allowed_memory_tiers` turns "spill needed" from a physical
# infeasibility into an operational warning. See 18c §Feasibility.
MEMORY_TIER_HBM = "hbm"
MEMORY_TIER_HBM_SPILL = "hbm+spill"


# Topology tokens for prefill/decode. "colocated" = prefill and decode run
# on the same replica with the same parallelism (the current default).
# "disaggregated" = prefill and decode run on separate replicas, allowing
# 18c to charge prefill and decode with independent topologies.
TOPOLOGY_COLOCATED = "colocated"
TOPOLOGY_DISAGGREGATED = "disaggregated"

_VALID_TOPOLOGIES = {TOPOLOGY_COLOCATED, TOPOLOGY_DISAGGREGATED}


# =============================================================================
# ServingWorkloadSpec
# =============================================================================


@dataclass(frozen=True)
class ServingWorkloadSpec:
    """Immutable, validated workload description per spec §Workload interface.

    All values are token counts unless otherwise noted. Validation happens in
    ``__post_init__`` — attempting to build a workload whose fresh + cached
    prompt exceeds ``max_context_length`` raises ``ValueError``.
    """
    name: str
    max_context_length: int
    fresh_prompt_tokens: int
    cached_prefix_tokens: int
    decode_kv_length: int
    output_tokens: int
    cache_hit_rate: float
    serving_batch: int
    concurrency: int
    available_gpus: int
    allowed_memory_tiers: Tuple[str, ...]
    prefill_topology: str
    decode_topology: str

    def __post_init__(self) -> None:
        # Spec validation rules.
        if self.max_context_length <= 0:
            raise ValueError(
                f"max_context_length must be positive; got {self.max_context_length}")
        if self.fresh_prompt_tokens < 0 or self.cached_prefix_tokens < 0:
            raise ValueError(
                f"fresh_prompt_tokens and cached_prefix_tokens must be "
                f"non-negative; got fresh={self.fresh_prompt_tokens}, "
                f"cached={self.cached_prefix_tokens}")
        if self.fresh_prompt_tokens + self.cached_prefix_tokens > self.max_context_length:
            raise ValueError(
                f"fresh + cached prompt ({self.fresh_prompt_tokens} + "
                f"{self.cached_prefix_tokens} = "
                f"{self.fresh_prompt_tokens + self.cached_prefix_tokens}) "
                f"must be <= max_context_length ({self.max_context_length})")
        if self.decode_kv_length < 0 or self.decode_kv_length > self.max_context_length:
            raise ValueError(
                f"decode_kv_length must be in [0, max_context_length]; "
                f"got {self.decode_kv_length} vs {self.max_context_length}")
        if not (0.0 <= self.cache_hit_rate <= 1.0):
            raise ValueError(
                f"cache_hit_rate must be in [0, 1]; got {self.cache_hit_rate}")
        if self.serving_batch <= 0:
            raise ValueError(f"serving_batch must be positive; got {self.serving_batch}")
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be positive; got {self.concurrency}")
        if self.available_gpus <= 0:
            raise ValueError(f"available_gpus must be positive; got {self.available_gpus}")
        if not self.allowed_memory_tiers:
            raise ValueError("allowed_memory_tiers must not be empty")
        if self.prefill_topology not in _VALID_TOPOLOGIES:
            raise ValueError(
                f"prefill_topology must be one of {_VALID_TOPOLOGIES}; "
                f"got {self.prefill_topology!r}")
        if self.decode_topology not in _VALID_TOPOLOGIES:
            raise ValueError(
                f"decode_topology must be one of {_VALID_TOPOLOGIES}; "
                f"got {self.decode_topology!r}")

    # ---- Convenience accessors ---------------------------------------

    @property
    def spill_permitted(self) -> bool:
        return MEMORY_TIER_HBM_SPILL in self.allowed_memory_tiers

    @property
    def is_disaggregated(self) -> bool:
        return (self.prefill_topology == TOPOLOGY_DISAGGREGATED
                or self.decode_topology == TOPOLOGY_DISAGGREGATED)

    @property
    def is_cold_ingestion(self) -> bool:
        return self.name == WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value

    @property
    def effective_prefill_tokens(self) -> int:
        """Prefill tokens the cost model should charge.

        Cached prefill is amortized by the hit rate; only the miss fraction
        pays the full prefill cost. Fresh prompt tokens always pay full cost.
        """
        miss_frac = max(0.0, 1.0 - self.cache_hit_rate)
        return int(self.fresh_prompt_tokens
                   + round(miss_frac * self.cached_prefix_tokens))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_context_length": self.max_context_length,
            "fresh_prompt_tokens": self.fresh_prompt_tokens,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "decode_kv_length": self.decode_kv_length,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "serving_batch": self.serving_batch,
            "concurrency": self.concurrency,
            "available_gpus": self.available_gpus,
            "allowed_memory_tiers": list(self.allowed_memory_tiers),
            "prefill_topology": self.prefill_topology,
            "decode_topology": self.decode_topology,
        }


# =============================================================================
# Canonical scenario builders
# =============================================================================


def _base_kwargs(
    *,
    output_tokens: int,
    serving_batch: int,
    concurrency: int,
    available_gpus: int,
    allowed_memory_tiers: Tuple[str, ...],
    prefill_topology: str,
    decode_topology: str,
) -> Dict[str, Any]:
    return {
        "output_tokens": output_tokens,
        "serving_batch": serving_batch,
        "concurrency": concurrency,
        "available_gpus": available_gpus,
        "allowed_memory_tiers": tuple(allowed_memory_tiers),
        "prefill_topology": prefill_topology,
        "decode_topology": decode_topology,
    }


def canonical_workloads(
    max_context_length: int,
    *,
    output_tokens: int = 512,
    serving_batch: int = 16,
    concurrency: int = 256,
    available_gpus: int = 8192,
    allowed_memory_tiers: Tuple[str, ...] = (MEMORY_TIER_HBM,),
    prefill_topology: str = TOPOLOGY_COLOCATED,
    decode_topology: str = TOPOLOGY_COLOCATED,
) -> Dict[str, ServingWorkloadSpec]:
    """Return the five canonical workloads for one context row.

    Per spec §Canonical scenarios. Every value here is a reference default;
    callers override any field via kwargs. Each scenario is a full
    `ServingWorkloadSpec` — the mapping key is the scenario name.
    """
    if max_context_length <= 0:
        raise ValueError(
            f"max_context_length must be positive; got {max_context_length}")

    C = int(max_context_length)
    base = _base_kwargs(
        output_tokens=output_tokens, serving_batch=serving_batch,
        concurrency=concurrency, available_gpus=available_gpus,
        allowed_memory_tiers=allowed_memory_tiers,
        prefill_topology=prefill_topology, decode_topology=decode_topology,
    )

    # 1) interactive_ordinary: fresh prompt = min(C, 8192), no cached prefix,
    #    decode KV = live context after prefill.
    fresh_ord = min(C, INTERACTIVE_FRESH_PROMPT)
    ordinary = ServingWorkloadSpec(
        name=WorkloadScenario.INTERACTIVE_ORDINARY.value,
        max_context_length=C,
        fresh_prompt_tokens=fresh_ord,
        cached_prefix_tokens=0,
        decode_kv_length=fresh_ord,
        cache_hit_rate=0.0,
        **base,
    )

    # 2) interactive_cached_long_context: fresh = min(C, 8192),
    #    cached = max(0, C - fresh), decode KV = C, cache_hit=0.90.
    fresh_cached = min(C, INTERACTIVE_FRESH_PROMPT)
    cached_prefix = max(0, C - fresh_cached)
    cached = ServingWorkloadSpec(
        name=WorkloadScenario.INTERACTIVE_CACHED_LONG_CONTEXT.value,
        max_context_length=C,
        fresh_prompt_tokens=fresh_cached,
        cached_prefix_tokens=cached_prefix,
        decode_kv_length=C,
        cache_hit_rate=CACHE_HIT_RATE_CACHED,
        **base,
    )

    # 3) incremental_long_session: fresh = min(C, 32768), cached fills the
    #    remainder, decode KV = C, cache_hit=0.75.
    fresh_inc = min(C, INCREMENTAL_SESSION_FRESH_PROMPT)
    cached_inc = max(0, C - fresh_inc)
    incremental = ServingWorkloadSpec(
        name=WorkloadScenario.INCREMENTAL_LONG_SESSION.value,
        max_context_length=C,
        fresh_prompt_tokens=fresh_inc,
        cached_prefix_tokens=cached_inc,
        decode_kv_length=C,
        cache_hit_rate=CACHE_HIT_RATE_INCREMENTAL,
        **base,
    )

    # 4) cold_full_context_ingestion: fresh = C, no cached, decode KV = C.
    cold = ServingWorkloadSpec(
        name=WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value,
        max_context_length=C,
        fresh_prompt_tokens=C,
        cached_prefix_tokens=0,
        decode_kv_length=C,
        cache_hit_rate=0.0,
        **base,
    )

    # 5) unconstrained_diagnostic: same shape as ordinary but no operational
    #    SLA filtering downstream. The workload spec itself is identical to
    #    ordinary; the diagnostic flag lives in the scenario name.
    diagnostic = ServingWorkloadSpec(
        name=WorkloadScenario.UNCONSTRAINED_DIAGNOSTIC.value,
        max_context_length=C,
        fresh_prompt_tokens=fresh_ord,
        cached_prefix_tokens=0,
        decode_kv_length=fresh_ord,
        cache_hit_rate=0.0,
        **base,
    )

    return {
        ordinary.name: ordinary,
        cached.name: cached,
        incremental.name: incremental,
        cold.name: cold,
        diagnostic.name: diagnostic,
    }


# =============================================================================
# WorkloadRegistry
# =============================================================================


@dataclass
class WorkloadRegistry:
    """Tiny helper that stores workloads by name.

    Consumers (BudgetMatrix / matrix driver) can iterate ``.items()`` and run
    a per-workload evaluation loop without needing to hand-thread the five
    scenario specs.
    """
    workloads: Dict[str, ServingWorkloadSpec] = field(default_factory=dict)

    def register(self, workload: ServingWorkloadSpec) -> None:
        self.workloads[workload.name] = workload

    def register_canonical(self, max_context_length: int, **kwargs: Any) -> None:
        for w in canonical_workloads(max_context_length, **kwargs).values():
            self.register(w)

    def get(self, name: str) -> Optional[ServingWorkloadSpec]:
        return self.workloads.get(name)

    def items(self) -> List[Tuple[str, ServingWorkloadSpec]]:
        return list(self.workloads.items())

    def as_dict(self) -> Dict[str, Any]:
        return {name: w.as_dict() for name, w in self.workloads.items()}
