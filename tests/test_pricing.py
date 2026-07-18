"""Unit tests for ac.pricing (Gate-2 Task E, USD TCO layer).

Covers:
  - hand-computed block math for a fully-specified fake compile output
  - pricing-spec loader (all six real targets + missing/broken specs)
  - no-mutation guarantee for the input dict
  - graceful degradation when the spec is missing (warning, no crash)
  - price-tier fallback (reserved_1y -> on_demand -> reference_estimate)
"""

import copy
import json
import os
import warnings

import pytest

from ac import pricing
from ac.pricing import (
    PRICING_SCHEMA_VERSION,
    attach_cost_block,
    available_pricing_targets,
    load_pricing_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_SPEC = {
    "schema_version": PRICING_SCHEMA_VERSION,
    "hardware_target": "testchip",
    "cloud_vendor": "testcloud",
    "usd_per_accelerator_hour": {
        "on_demand": 2.0,
        "reserved_1y": None,
        "spot": 1.0,
    },
    "reference_estimate_usd_per_accelerator_hour": None,
    "tdp_watts": 400,
    "assumed_avg_utilization": 0.5,
    "assumed_pue": 2.0,
    "electricity_usd_per_kwh": 0.10,
    "provenance": {"prices": [], "note": "synthetic test spec"},
}


def _fake_compile_result():
    """Minimal fake of an emitted AC config with known, round numbers."""
    return {
        "schema_version": "0.3",
        "metadata": {
            "input_hardware": "testchip",
            "input_constraints": {
                "hardware": "testchip",
                "training_tokens": "2.0T",
                "context_length": 8192,
                "prompt_len": 1000,
                "output_len": 100,
                "serving_batch": 32,
            },
            "predicted": {
                "training_throughput_tokens_per_sec": 100000,
                "aggregate_training_throughput_tokens_per_sec": 800000,
                "serving_ttft_ms": 1000.0,
                "serving_tbt_ms": 100.0,
                "serving_instance_gpus": 8,
                "memory_per_gpu_gb": 42.0,
            },
        },
        "parallelism": {
            "tensor_parallel": 8,
            "pipeline_parallel": 1,
            "data_parallel": 1,
            "expert_parallel": 1,
            "context_parallel": 1,
        },
    }


@pytest.fixture()
def spec_dir(tmp_path, monkeypatch):
    d = tmp_path / "pricing_specs"
    d.mkdir()
    (d / "testchip.json").write_text(json.dumps(TEST_SPEC))
    monkeypatch.setenv("AC_PRICING_SPEC_DIR", str(d))
    yield str(d)
    monkeypatch.delenv("AC_PRICING_SPEC_DIR", raising=False)


# ---------------------------------------------------------------------------
# Hand-computed block math
# ---------------------------------------------------------------------------

def test_cost_block_hand_computed(spec_dir):
    """Verify every number in the block against a hand derivation.

    Test spec: instance $2.0/accel-h; energy = 400W x 0.5 util x 2.0 PUE
    = 0.4 kW x $0.10/kWh = $0.04/accel-h. Combined = $2.04/accel-h.

    Training: 2e12 tokens / (800000 tok/s x 3600) = 694.4444 h; GPUs =
    8x1x1x1 = 8 -> 5555.5556 accel-h. instance = 5555.5556 x 2.0 =
    11111.1111; energy = 5555.5556 x 0.04 = 222.2222; total = 11333.3333.

    Serving: slot time = 1.0s ttft + 100 x 0.1s = 11.0 s; tokens/request =
    1000 + 100 = 1100; instance tps = 32 x 1100 / 11 = 3200 tok/s; hourly
    cost = 8 GPUs x 2.04 = 16.32 $/h; per 1M tokens = 16.32 x (1e6/3200)
    / 3600 = 1.4167.

    Annual: 1 instance x 16.32 x 8760 = 142963.20.
    """
    result = attach_cost_block(_fake_compile_result(), "testchip")
    block = result["metadata"]["predicted"]["cost_estimate_usd"]

    assert block["training_total"] == pytest.approx(11333.33, abs=0.01)
    assert block["serving_per_1m_tokens"] == pytest.approx(1.4167, abs=1e-4)
    assert block["annual_serving_at_load"] == pytest.approx(142963.20, abs=0.01)

    rates = block["usd_per_accelerator_hour"]
    assert rates["instance_price"] == pytest.approx(2.0)
    assert rates["energy_price"] == pytest.approx(0.04)
    assert rates["combined"] == pytest.approx(2.04)

    tb = block["breakdown"]["training"]
    assert tb["training_gpu_count"] == 8
    assert tb["training_hours"] == pytest.approx(694.44, abs=0.01)
    assert tb["accelerator_hours"] == pytest.approx(5555.56, abs=0.01)

    sb = block["breakdown"]["serving"]
    assert sb["instance_throughput_tokens_per_sec"] == pytest.approx(3200.0)
    assert sb["instance_cost_usd_per_hour"] == pytest.approx(16.32)


def test_cost_block_annual_with_offered_load(spec_dir):
    """avg_serving_tokens_per_sec drives the instance count for the year."""
    result = attach_cost_block(
        _fake_compile_result(), "testchip",
        {"avg_serving_tokens_per_sec": 10000.0},
    )
    block = result["metadata"]["predicted"]["cost_estimate_usd"]
    # 10000 tok/s offered / 3200 tok/s per instance -> ceil(3.125) = 4.
    assert block["breakdown"]["serving"]["instances_for_annual_figure"] == 4
    assert block["annual_serving_at_load"] == pytest.approx(
        4 * 16.32 * 8760, abs=0.01)


def test_price_tier_fallbacks(spec_dir):
    """reserved_1y is null in the test spec -> falls back to on_demand."""
    result = attach_cost_block(
        _fake_compile_result(), "testchip", price_tier="reserved_1y")
    block = result["metadata"]["predicted"]["cost_estimate_usd"]
    assert block["price_tier_requested"] == "reserved_1y"
    assert block["price_source"] == "on_demand"
    assert block["usd_per_accelerator_hour"]["instance_price"] == 2.0

    # spot tier is present and used directly.
    result = attach_cost_block(
        _fake_compile_result(), "testchip", price_tier="spot")
    block = result["metadata"]["predicted"]["cost_estimate_usd"]
    assert block["price_source"] == "spot"
    assert block["usd_per_accelerator_hour"]["instance_price"] == 1.0


def test_reference_estimate_fallback(spec_dir, tmp_path, monkeypatch):
    """A spec with all tiers null falls back to reference_estimate."""
    spec = copy.deepcopy(TEST_SPEC)
    spec["hardware_target"] = "estchip"
    spec["usd_per_accelerator_hour"] = {
        "on_demand": None, "reserved_1y": None, "spot": None}
    spec["reference_estimate_usd_per_accelerator_hour"] = 1.8
    spec["tdp_watts"] = None  # energy not modeled
    (tmp_path / "pricing_specs" / "estchip.json").write_text(json.dumps(spec))

    result = attach_cost_block(_fake_compile_result(), "estchip")
    block = result["metadata"]["predicted"]["cost_estimate_usd"]
    assert block["price_source"] == "reference_estimate"
    assert block["usd_per_accelerator_hour"]["instance_price"] == 1.8
    assert block["usd_per_accelerator_hour"]["energy_price"] is None
    # training: 5555.5556 accel-h x 1.8 = 10000.0
    assert block["training_total"] == pytest.approx(10000.0, abs=0.01)


# ---------------------------------------------------------------------------
# Spec loader
# ---------------------------------------------------------------------------

def test_load_real_specs_all_targets():
    """Every shipped hardware target spec loads and has a usable price."""
    targets = available_pricing_targets()
    for expected in ("h100", "b200", "tpu_v5p", "tpu_v5e",
                     "trainium2", "trainium3"):
        assert expected in targets, f"missing pricing spec for {expected}"
        spec = load_pricing_spec(expected)
        assert spec["schema_version"] == PRICING_SCHEMA_VERSION
        prices = spec["usd_per_accelerator_hour"]
        has_price = any(
            isinstance(v, (int, float)) and v > 0 for v in prices.values()
        ) or isinstance(
            spec.get("reference_estimate_usd_per_accelerator_hour"),
            (int, float),
        )
        assert has_price, f"{expected} spec carries no usable price"


def test_spec_loader_alias_and_missing():
    assert load_pricing_spec("trn2") is not None  # alias -> trainium2
    assert load_pricing_spec("H100") is not None  # case-insensitive
    assert load_pricing_spec("no_such_chip") is None


def test_spec_loader_rejects_bad_schema(tmp_path, monkeypatch):
    d = tmp_path / "specs"
    d.mkdir()
    (d / "badchip.json").write_text(json.dumps({"schema_version": "v0"}))
    monkeypatch.setenv("AC_PRICING_SPEC_DIR", str(d))
    assert load_pricing_spec("badchip") is None


# ---------------------------------------------------------------------------
# No-mutation guarantee
# ---------------------------------------------------------------------------

def test_input_dict_not_mutated(spec_dir):
    src = _fake_compile_result()
    snapshot = copy.deepcopy(src)
    out = attach_cost_block(src, "testchip")
    assert src == snapshot, "attach_cost_block mutated its input dict"
    assert out is not src
    # The output differs only by the added block.
    assert "cost_estimate_usd" not in src["metadata"]["predicted"]
    assert "cost_estimate_usd" in out["metadata"]["predicted"]
    # Existing numeric fields identical.
    assert (out["metadata"]["predicted"]["serving_tbt_ms"]
            == snapshot["metadata"]["predicted"]["serving_tbt_ms"])


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_missing_spec_warns_and_omits_block(spec_dir):
    src = _fake_compile_result()
    with pytest.warns(UserWarning, match="no pricing spec"):
        out = attach_cost_block(src, "no_such_chip")
    assert "cost_estimate_usd" not in out["metadata"]["predicted"]
    assert "cost_estimate_usd" not in out


def test_spec_without_any_price_warns(spec_dir, tmp_path):
    spec = copy.deepcopy(TEST_SPEC)
    spec["hardware_target"] = "brokechip"
    spec["usd_per_accelerator_hour"] = {
        "on_demand": None, "reserved_1y": None, "spot": None}
    spec["reference_estimate_usd_per_accelerator_hour"] = None
    (tmp_path / "pricing_specs" / "brokechip.json").write_text(json.dumps(spec))

    with pytest.warns(UserWarning, match="no usable price"):
        out = attach_cost_block(_fake_compile_result(), "brokechip")
    assert "cost_estimate_usd" not in out.get("metadata", {}).get(
        "predicted", {})


def test_missing_predicted_fields_partial_block(spec_dir):
    """Missing TBT -> serving figures null with a note; training still works."""
    src = _fake_compile_result()
    del src["metadata"]["predicted"]["serving_tbt_ms"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must not warn for partial blocks
        out = attach_cost_block(src, "testchip")
    block = out["metadata"]["predicted"]["cost_estimate_usd"]
    assert block["serving_per_1m_tokens"] is None
    assert block["annual_serving_at_load"] is None
    assert block["training_total"] == pytest.approx(11333.33, abs=0.01)
    assert "note" in block["breakdown"]["serving"]


def test_bare_dict_without_metadata(spec_dir):
    """Non-config dicts get the block at top level instead of crashing."""
    src = {
        "predicted": _fake_compile_result()["metadata"]["predicted"],
        "input_constraints": {},
        "parallelism": {"tensor_parallel": 8, "pipeline_parallel": 1,
                        "data_parallel": 1},
    }
    out = attach_cost_block(src, "testchip", {"training_tokens": 2e12})
    assert "cost_estimate_usd" in out
    assert out["cost_estimate_usd"]["training_total"] == pytest.approx(
        11333.33, abs=0.01)
