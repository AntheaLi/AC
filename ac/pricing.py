"""USD cost layer for AC compile outputs (Gate-2 Task E, pure-add).

Attaches a ``cost_estimate_usd`` block to an already-emitted AC config dict,
computed exclusively from the existing predicted TPS / TBT / TTFT / memory /
parallelism fields. This module:

- never mutates the input dict (a deep copy is returned),
- never changes any existing numeric field,
- degrades gracefully: a missing/unreadable pricing spec (or a spec with no
  usable price) emits a ``warnings.warn`` and returns the dict unchanged
  instead of raising,
- reads per-target price books from ``ac/pricing_specs/<target>.json``
  (override the directory with ``AC_PRICING_SPEC_DIR``). Each spec is a
  single hot-updatable JSON file with a ``schema_version`` field.

Public API
----------
attach_cost_block(result_dict, hardware_target, workload_dict=None, *,
                  price_tier="on_demand") -> dict
load_pricing_spec(hardware_target) -> Optional[dict]
available_pricing_targets() -> List[str]

Cost model (deliberately simple, assumptions are stated in the block)
---------------------------------------------------------------------
training_total
    total training tokens / aggregate training TPS x training-GPU count x
    ($/accelerator-hour + energy $/accelerator-hour).
serving_per_1m_tokens
    per-instance cost rate / per-instance token throughput x 1e6 tokens,
    where one request is modeled as occupying one continuous-batching slot
    for ``ttft + output_len x tbt`` seconds and producing
    ``prompt_len + output_len`` tokens.
annual_serving_at_load
    instances-needed x per-instance hourly cost x 8760 h. When the workload
    carries no explicit average token rate, one instance serving 24/7 for a
    year is reported.

The energy sub-term is ``tdp x utilization x PUE x electricity_price`` and is
omitted (null) for targets whose vendor does not publish TDP (TPU v5e/v5p).
"""

from __future__ import annotations

import copy
import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

PRICING_SCHEMA_VERSION = "pricing_v1"

_DEFAULT_SPEC_DIR = os.path.join(os.path.dirname(__file__), "pricing_specs")


def _spec_dir() -> str:
    """Pricing-spec directory; ``AC_PRICING_SPEC_DIR`` wins when set.

    Read at call time (not import time) so tests and hot-update tooling can
    point the loader at a fresh directory without re-importing the module.
    """
    return os.environ.get("AC_PRICING_SPEC_DIR", _DEFAULT_SPEC_DIR)

# Canonical hardware-target names used by ac-compile (see
# throughput_model.load_hardware); aliases mirror its mapping.
_TARGET_ALIASES = {
    "trn2": "trainium2",
    "trn3": "trainium3",
    "h100_sxm": "h100",
}

_PRICE_TIERS = ("on_demand", "reserved_1y", "spot")

_HOURS_PER_YEAR = 8760.0

_SUFFIX_MULTIPLIERS = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


def _canonical_target(hardware_target: str) -> str:
    v = (hardware_target or "").strip().lower()
    return _TARGET_ALIASES.get(v, v)


def load_pricing_spec(hardware_target: str) -> Optional[Dict[str, Any]]:
    """Load the pricing spec for a hardware target.

    Returns None (never raises) when the spec file is missing, unreadable,
    or fails a minimal shape check.
    """
    target = _canonical_target(hardware_target)
    path = os.path.join(_spec_dir(), f"{target}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            spec = json.load(f)
    except Exception:
        return None
    if not isinstance(spec, dict):
        return None
    if spec.get("schema_version") != PRICING_SCHEMA_VERSION:
        return None
    if not isinstance(spec.get("usd_per_accelerator_hour"), dict):
        return None
    return spec


def available_pricing_targets() -> List[str]:
    """List hardware targets with a loadable pricing spec."""
    out = []
    spec_dir = _spec_dir()
    if os.path.isdir(spec_dir):
        for name in sorted(os.listdir(spec_dir)):
            if name.endswith(".json") and load_pricing_spec(name[:-5]) is not None:
                out.append(name[:-5])
    return out


def _resolve_price(
    spec: Dict[str, Any], price_tier: str
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Resolve $/accelerator-hour for the requested tier.

    Fallback order: requested tier -> on_demand -> reference_estimate.
    Returns (price, source_label, note). price is None when nothing usable.
    """
    prices = spec.get("usd_per_accelerator_hour") or {}
    requested = (price_tier or "on_demand").strip().lower()
    order: List[Tuple[str, str]] = [(requested, requested)]
    if requested != "on_demand":
        order.append(("on_demand", "on_demand"))
    order.append(("reference_estimate", "reference_estimate"))

    for key, label in order:
        if key == "reference_estimate":
            value = spec.get("reference_estimate_usd_per_accelerator_hour")
        else:
            value = prices.get(key)
        if isinstance(value, (int, float)) and value > 0:
            note = None
            if label != requested:
                note = (
                    f"requested price tier '{requested}' has no published "
                    f"price for this target; fell back to '{label}'"
                )
            return float(value), label, note
    return None, None, (
        f"no usable price in spec (tier '{requested}', on_demand, or "
        "reference_estimate all missing)"
    )


def _energy_rate_usd_per_accel_hour(spec: Dict[str, Any]) -> Optional[float]:
    """Electricity cost per accelerator-hour, or None when TDP is unknown."""
    tdp = spec.get("tdp_watts")
    if not isinstance(tdp, (int, float)) or tdp <= 0:
        return None
    util = float(spec.get("assumed_avg_utilization", 0.65))
    pue = float(spec.get("assumed_pue", 1.2))
    kwh_price = float(spec.get("electricity_usd_per_kwh", 0.10))
    kw = tdp * util * pue / 1000.0
    return kw * kwh_price


def _parse_token_count(value: Any) -> Optional[float]:
    """Parse token counts like "2.0T", "15B", 2e12, or "2000000000000"."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    s = str(value).strip().lower().replace(",", "")
    if not s:
        return None
    mult = 1.0
    if s[-1] in _SUFFIX_MULTIPLIERS:
        mult = _SUFFIX_MULTIPLIERS[s[-1]]
        s = s[:-1]
    try:
        v = float(s) * mult
    except ValueError:
        return None
    return v if v > 0 else None


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    v = _parse_number(value)
    if v is None or v <= 0:
        return None
    return max(1, int(round(v)))


def _blocks(result_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Locate (predicted, input_constraints, parallelism) sub-dicts.

    Accepts either a full emitted AC config (with a ``metadata`` block) or a
    bare dict carrying the three sub-dicts at top level.
    """
    meta = result_dict.get("metadata")
    if isinstance(meta, dict):
        predicted = meta.get("predicted") or {}
        constraints = meta.get("input_constraints") or {}
    else:
        predicted = result_dict.get("predicted") or {}
        constraints = result_dict.get("input_constraints") or {}
    parallelism = result_dict.get("parallelism") or {}
    return predicted, constraints, parallelism


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


def attach_cost_block(
    result_dict: Dict[str, Any],
    hardware_target: str,
    workload_dict: Optional[Dict[str, Any]] = None,
    *,
    price_tier: str = "on_demand",
) -> Dict[str, Any]:
    """Return a copy of ``result_dict`` with a ``cost_estimate_usd`` block.

    The block is added under ``metadata.predicted`` (the same block that
    carries the TPS/TBT fields it derives from), or at top level when the
    dict is not an emitted AC config. Existing fields are never mutated.

    ``workload_dict`` (all keys optional) overrides config-derived inputs:
        training_tokens             total pretraining tokens (float or "2.0T")
        prompt_len / output_len     serving request shape (tokens)
        serving_batch               continuous-batching slots per instance
        avg_serving_tokens_per_sec  average offered load for the annual figure
        price_tier                  "on_demand" | "reserved_1y" | "spot"
    """
    workload = dict(workload_dict or {})
    tier = str(workload.pop("price_tier", None) or price_tier or "on_demand")
    out = copy.deepcopy(result_dict)

    target = _canonical_target(hardware_target)
    spec = load_pricing_spec(target)
    if spec is None:
        warnings.warn(
            f"[ac.pricing] no pricing spec for hardware target "
            f"'{hardware_target}' (looked for {target}.json in {_spec_dir()}); "
            "cost_estimate_usd block omitted",
            stacklevel=2,
        )
        return out

    price, price_source, price_note = _resolve_price(spec, tier)
    if price is None:
        warnings.warn(
            f"[ac.pricing] pricing spec for '{target}' has no usable price "
            f"({price_note}); cost_estimate_usd block omitted",
            stacklevel=2,
        )
        return out

    energy_rate = _energy_rate_usd_per_accel_hour(spec)
    combined_rate = price + (energy_rate or 0.0)

    predicted, constraints, parallelism = _blocks(out)

    # ------------------------------------------------------------------
    # Training cost
    # ------------------------------------------------------------------
    training_tokens = _parse_token_count(
        workload.get("training_tokens", constraints.get("training_tokens"))
    )
    agg_tps = _parse_number(
        predicted.get("aggregate_training_throughput_tokens_per_sec")
    )
    if agg_tps is None:
        # Fall back to per-replica TPS x DP when the aggregate is absent.
        per_replica = _parse_number(
            predicted.get("training_throughput_tokens_per_sec")
        )
        dp_for_tps = _parse_int(parallelism.get("data_parallel")) or 1
        agg_tps = per_replica * dp_for_tps if per_replica else None

    tp_d = _parse_int(parallelism.get("tensor_parallel")) or 1
    pp_d = _parse_int(parallelism.get("pipeline_parallel")) or 1
    dp_d = _parse_int(parallelism.get("data_parallel")) or 1
    cp_d = _parse_int(parallelism.get("context_parallel")) or 1
    ep_d = _parse_int(parallelism.get("expert_parallel")) or 1
    training_gpus = tp_d * pp_d * dp_d * cp_d

    training_total = None
    training_breakdown: Dict[str, Any] = {
        "training_tokens": training_tokens,
        "aggregate_training_tps": agg_tps,
        "training_gpu_count": training_gpus,
    }
    if training_tokens and agg_tps and agg_tps > 0:
        train_hours = training_tokens / (agg_tps * 3600.0)
        accel_hours = train_hours * training_gpus
        instance_usd = accel_hours * price
        energy_usd = (
            accel_hours * energy_rate if energy_rate is not None else None
        )
        training_total = instance_usd + (energy_usd or 0.0)
        training_breakdown.update({
            "training_hours": round(train_hours, 2),
            "accelerator_hours": round(accel_hours, 2),
            "instance_usd": round(instance_usd, 2),
            "energy_usd": _round_or_none(energy_usd, 2),
        })
    else:
        training_breakdown["note"] = (
            "training cost not computed: missing training_tokens or "
            "aggregate training TPS in the compile output"
        )

    # ------------------------------------------------------------------
    # Serving cost
    # ------------------------------------------------------------------
    serving_gpus = _parse_int(predicted.get("serving_instance_gpus"))
    if serving_gpus is None:
        serving_gpus = tp_d * pp_d * cp_d * ep_d
    ttft_s = (_parse_number(predicted.get("serving_ttft_ms")) or 0.0) / 1000.0
    tbt_ms = _parse_number(predicted.get("serving_tbt_ms"))
    tbt_s = (tbt_ms or 0.0) / 1000.0
    batch = _parse_int(workload.get("serving_batch")) \
        or _parse_int(constraints.get("serving_batch")) or 1
    prompt_len = _parse_int(workload.get("prompt_len")) \
        or _parse_int(constraints.get("prompt_len")) \
        or _parse_int((predicted.get("prefill_model") or {}).get("prompt_len")) \
        or _parse_int(constraints.get("context_length")) or 0
    output_len = _parse_int(workload.get("output_len")) \
        or _parse_int(constraints.get("output_len")) or 256

    serving_per_1m = None
    annual_serving = None
    hourly_instance_cost = serving_gpus * combined_rate
    tokens_per_request = prompt_len + output_len
    seconds_per_request = ttft_s + output_len * tbt_s
    instance_tps = None
    # Require an explicit TBT: without it the decode share of the slot time
    # is unmodeled and any throughput figure would be silently optimistic.
    if tbt_ms is not None and tokens_per_request > 0 and seconds_per_request > 0:
        instance_tps = batch * tokens_per_request / seconds_per_request
    serving_breakdown: Dict[str, Any] = {
        "serving_instance_gpus": serving_gpus,
        "serving_batch": batch,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "tokens_per_request": tokens_per_request,
        "seconds_per_request_slot": round(seconds_per_request, 4),
        "instance_throughput_tokens_per_sec": (
            round(instance_tps, 1) if instance_tps else None
        ),
        "instance_cost_usd_per_hour": round(hourly_instance_cost, 4),
    }
    if instance_tps and instance_tps > 0:
        serving_per_1m = hourly_instance_cost * (1e6 / instance_tps) / 3600.0
        offered_rate = _parse_number(workload.get("avg_serving_tokens_per_sec"))
        if offered_rate and offered_rate > 0:
            instances_needed = max(1, math.ceil(offered_rate / instance_tps))
            load_note = (
                f"workload-specified average load {offered_rate} tok/s -> "
                f"{instances_needed} instance(s)"
            )
        else:
            instances_needed = 1
            load_note = (
                "no avg_serving_tokens_per_sec in workload; one instance "
                "serving 24/7 for a year"
            )
        annual_serving = instances_needed * hourly_instance_cost * _HOURS_PER_YEAR
        serving_breakdown.update({
            "instances_for_annual_figure": instances_needed,
            "annual_load_note": load_note,
        })
    else:
        serving_breakdown["note"] = (
            "serving cost not computed: missing serving_ttft_ms / "
            "serving_tbt_ms in the compile output"
        )

    assumptions = [
        "cost = instance_price + electricity; all figures are LIST-price "
        "estimates, not quotes (see spec price_note)",
        "serving throughput model: each request occupies one batching slot "
        "for (ttft + output_len x tbt) and yields (prompt_len + output_len) "
        "tokens; queueing, preemption, and cache-hit effects excluded",
        "training cost uses aggregate (cluster-wide) training TPS x "
        "TPxPPxDPxCP GPU count from the compile output",
        f"annual serving figure spans {_HOURS_PER_YEAR:.0f} h",
    ]
    if energy_rate is None:
        assumptions.append(
            "electricity sub-term NOT modeled for this target (vendor does "
            "not publish TDP); instance price only"
        )
    if price_note:
        assumptions.append(price_note)

    block = {
        "schema_version": PRICING_SCHEMA_VERSION,
        "currency": "USD",
        "hardware_target": target,
        "price_tier_requested": tier,
        "price_source": price_source,
        "usd_per_accelerator_hour": {
            "instance_price": round(price, 4),
            "energy_price": _round_or_none(energy_rate, 6),
            "combined": round(combined_rate, 4),
        },
        "training_total": _round_or_none(training_total, 2),
        "serving_per_1m_tokens": _round_or_none(serving_per_1m, 4),
        "annual_serving_at_load": _round_or_none(annual_serving, 2),
        "breakdown": {
            "training": training_breakdown,
            "serving": serving_breakdown,
        },
        "assumptions": assumptions,
        "provenance": spec.get("provenance"),
        "disclaimer": (
            "List-price estimate from ac/pricing_specs/"
            f"{target}.json (schema {PRICING_SCHEMA_VERSION}); not a quote. "
            "Hot-update the spec file when prices change."
        ),
    }

    meta = out.get("metadata")
    if isinstance(meta, dict):
        meta.setdefault("predicted", {})["cost_estimate_usd"] = block
    else:
        out["cost_estimate_usd"] = block
    return out


__all__ = [
    "PRICING_SCHEMA_VERSION",
    "attach_cost_block",
    "available_pricing_targets",
    "load_pricing_spec",
]
