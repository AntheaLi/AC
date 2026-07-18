"""End-to-end contracts for the Blackwell NVFP4 and MXFP8 recipes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ac.architecture import precision_bytes_per_element
from ac.cli_compile import parse_precision_modes
from ac.cli_recipe import _PRECISION_MODE_CANONICAL_TO_INPUT
from ac.delta_engine import apply_transition
from ac.deltas import get as get_transformation
from ac.lattice_engine import B200
from ac.optimizer import PRECISION_CONFIGS, get_precision_configs_for_hardware
from ac.penalties import precision_supported, weight_storage_supported
from ac.quality_model import DEFAULT_QUALITY_CONSTANTS
from ac.stress import _bpe
from ac.stress import Workload
from ac.throughput_model import (
    ArchConfig,
    _matmul_cost,
    get_tile_efficiency,
    load_calibration,
    load_hardware,
)


ROOT = Path(__file__).resolve().parents[1]
PRECISION_ORDER = (
    "bf16", "mxfp8", "fp8", "mxfp6", "nvfp4", "mxfp4", "fp4",
)


def test_cli_aliases_and_recipe_round_trip():
    parsed = parse_precision_modes(
        "mxfp8,mxfp8_ffn,nvfp4,nvfp4_ffn"
    )
    assert parsed == [
        "all_mxfp8", "ffn_mxfp8", "all_nvfp4", "ffn_nvfp4",
    ]
    assert [_PRECISION_MODE_CANONICAL_TO_INPUT[item] for item in parsed] == [
        "mxfp8", "mxfp8_ffn", "nvfp4", "nvfp4_ffn",
    ]

    assert PRECISION_CONFIGS["ffn_mxfp8"] == {
        "weight_precision": "bf16",
        "ffn_precision": "mxfp8",
        "attn_precision": {"qk": "bf16", "v": "bf16", "output": "bf16"},
    }
    assert PRECISION_CONFIGS["ffn_nvfp4"] == {
        "weight_precision": "fp8",
        "ffn_precision": "nvfp4",
        "attn_precision": {"qk": "bf16", "v": "fp8", "output": "fp8"},
    }


def test_native_compute_is_blackwell_only_but_weight_storage_is_portable():
    for precision in ("mxfp8", "nvfp4"):
        assert precision_supported(precision, "b200")
        assert precision_supported(precision, "gb200_nvl72")
        assert not precision_supported(precision, "h100")
        assert not precision_supported(precision, "trainium3")
        assert weight_storage_supported(precision, "h100")

    b200_modes = set(get_precision_configs_for_hardware("b200"))
    assert {
        "ffn_mxfp8", "all_mxfp8", "ffn_nvfp4", "all_nvfp4",
    } <= b200_modes
    for hardware in ("h100", "h800", "trainium2", "trainium3", "tpu_v5p"):
        modes = set(get_precision_configs_for_hardware(hardware))
        assert not modes & {
            "ffn_mxfp8", "all_mxfp8", "ffn_nvfp4", "all_nvfp4",
        }


@pytest.mark.parametrize(
    ("precision", "expected_bpe"),
    (("mxfp8", 1.03125), ("nvfp4", 0.5625)),
)
def test_storage_width_is_identical_in_every_arithmetic_path(
    precision, expected_bpe
):
    assert precision_bytes_per_element(precision) == pytest.approx(expected_bpe)
    assert _bpe(precision) == pytest.approx(expected_bpe)
    for hardware in ("h100", "b200", "gb200_nvl72"):
        assert load_hardware(hardware).bytes_per_elem(precision) == pytest.approx(
            expected_bpe
        )

    arch = ArchConfig(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=14336,
        kv_precision=precision,
    )
    assert arch.kv_bytes_per_token_per_layer() == pytest.approx(
        2 * 8 * 128 * expected_bpe
    )


def test_blackwell_peaks_tiles_and_attention_efficiencies_are_explicit():
    hw = load_hardware("b200")
    assert hw.peak_flops_s("mxfp8") == hw.peak_flops_s("fp8")
    assert hw.peak_flops_s("nvfp4") == hw.peak_flops_s("fp4")
    assert hw.datasheet_peak_flops_s("mxfp8") == 4500e12
    assert hw.datasheet_peak_flops_s("nvfp4") == 9000e12
    assert hw.fused_attention_efficiency["mxfp8"] == pytest.approx(0.78)
    assert hw.fused_attention_efficiency["nvfp4"] == pytest.approx(0.72)

    assert B200.tiles["mxfp8"].cta_k == B200.tiles["fp8"].cta_k
    assert B200.tiles["nvfp4"].cta_k == B200.tiles["fp4"].cta_k
    shape = (384, 12288, 4096)
    assert get_tile_efficiency(*shape, "mxfp8", B200) == pytest.approx(
        get_tile_efficiency(*shape, "fp8", B200)
    )
    assert get_tile_efficiency(*shape, "nvfp4", B200) == pytest.approx(
        get_tile_efficiency(*shape, "fp4", B200)
    )


def test_microscaled_formats_reuse_the_matching_blackwell_kernel_calibration():
    hw = load_hardware("b200")
    calibration = load_calibration("b200")
    shape = (4096, 11008, 4096)

    def cost(precision):
        return _matmul_cost(
            *shape,
            precision,
            hw,
            B200,
            calibration=calibration,
        )[0]

    assert cost("mxfp8") == pytest.approx(cost("fp8"))
    assert cost("nvfp4") == pytest.approx(cost("fp4"))
    assert cost("mxfp6") > cost("fp4")
    assert cost("mxfp6") < cost("fp8")


def test_quality_priors_are_complete_and_monotone():
    table = DEFAULT_QUALITY_CONSTANTS["precision_sensitivity"]
    for component in ("ffn", "attention_qkv", "attention_o", "lm_head"):
        rows = table[component]
        assert all(precision in rows or precision == "bf16"
                   for precision in PRECISION_ORDER)
        values = [0.0 if p == "bf16" else rows[p]["delta"]
                  for p in PRECISION_ORDER]
        assert values == sorted(values), (component, values)

    qk = table["qk_logits"]
    qk_values = [0.0 if p == "bf16" else qk[p]["delta"]
                 for p in PRECISION_ORDER]
    assert qk_values == sorted(qk_values)


@pytest.mark.parametrize("precision", ("mxfp8", "nvfp4"))
def test_greenfield_cli_emits_requested_blackwell_precision(tmp_path, precision):
    config_path = tmp_path / f"{precision}.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ac.cli_compile",
            "--hardware",
            "b200",
            "--params",
            "1",
            "--tokens",
            "0.2",
            "--context",
            "2048",
            "--serving-tbt",
            "100",
            "--serving-batch",
            "4",
            "--tp",
            "1",
            "--pp",
            "1",
            "--dp",
            "1",
            "--precision-modes",
            precision,
            "--max-candidates",
            "30",
            "--output-config",
            str(config_path),
            "--output-justification",
            str(tmp_path / f"{precision}.md"),
            "--output-pareto",
            str(tmp_path / f"{precision}.csv"),
            "--no-shadow-prices",
            "--quiet",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(config_path.read_text())
    layer = config["architecture"]["layer_configs"][0]
    assert layer["ffn"]["precision"] == precision
    assert layer["attention"]["precision"]["qk"] == "bf16"


def test_h100_greenfield_request_is_loudly_filtered(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ac.cli_compile",
            "--hardware",
            "h100",
            "--params",
            "1",
            "--tokens",
            "0.2",
            "--context",
            "2048",
            "--precision-modes",
            "nvfp4,mxfp8",
            "--max-candidates",
            "10",
            "--out",
            str(tmp_path),
            "--no-shadow-prices",
            "--quiet",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "all unsupported on h100" in result.stderr
    config = json.loads((tmp_path / "arch.json").read_text())
    layer = config["architecture"]["layer_configs"][0]
    assert layer["ffn"]["precision"] not in {"nvfp4", "mxfp8"}


def test_delta_uses_target_hardware_and_fails_closed_on_unsupported_activation():
    arch = ArchConfig(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=14336,
        batch_size=1,
        seq_len=2048,
    )
    workload = Workload(
        batch_size=1,
        prefill_seq_len=2048,
        decode_kv_len=2048,
        phase="decode",
    )
    transform = get_transformation("change_precision_per_component")

    b200 = apply_transition(
        arch,
        transform,
        {"activation": "mxfp8"},
        hardware="b200",
        workload=workload,
        tp_degree=8,
        dp_degree=8,
    )
    assert b200.feasible, b200.reason_if_infeasible
    assert b200.candidate_quality.precision_loss == pytest.approx(0.00225)

    h100 = apply_transition(
        arch,
        transform,
        {"activation": "nvfp4"},
        hardware="h100",
        workload=workload,
        tp_degree=8,
        dp_degree=8,
    )
    assert not h100.feasible
    assert "quality_validation_failed" in h100.reason_if_infeasible


def test_delta_cli_reports_unsupported_h100_activation_as_infeasible():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ac.cli_delta_eval",
            "--baseline-config",
            "configs/mistral_7b.json",
            "--hardware",
            "h100",
            "--apply",
            "change_precision_per_component:activation=nvfp4",
            "--no-pareto",
            "--json",
            "--stdout",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert not payload["feasible"]
    assert payload["metrics"] == {}
    assert "quality_validation_failed" in payload["reason_if_infeasible"]
