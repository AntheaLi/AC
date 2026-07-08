"""Regression tests for the effective-capacity/effective-data quality model."""

import math

from ac.auto_calibrate import _fit_effective_quality_components
from ac.optimizer import DeploymentConstraints
from ac.quality_model import (
    ArchConfig,
    TrainingConfig,
    estimate_quality,
)


def _dense() -> ArchConfig:
    return ArchConfig(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=14336,
    )


def _moe() -> ArchConfig:
    return ArchConfig(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=14336,
        model_type="moe",
        moe_config={
            "enabled": True,
            "n_experts": 8,
            "top_k": 2,
            "expert_dim": 7168,
        },
    )


def _hybrid() -> ArchConfig:
    return ArchConfig(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=14336,
        model_type="hybrid",
        state_config={
            "enabled": True,
            "state_type": "mamba2",
            "state_layers": 28,
            "attention_layers": 4,
            "d_state": 192,
        },
    )


def test_central_scenario_is_20t_and_effective_v2():
    constraints = DeploymentConstraints()
    assert constraints.training_tokens == 20_000_000_000_000
    assert constraints.quality_model_version == "effective_capacity_v2"


def test_legacy_fallback_preserves_dense_result():
    arch = _dense()
    common = dict(training_tokens=20_000_000_000_000, sequence_length=8192)
    effective = estimate_quality(
        arch,
        TrainingConfig(**common, quality_model_version="effective_capacity_v2"),
        {"context_length": 8192},
    )
    legacy = estimate_quality(
        arch,
        TrainingConfig(**common, quality_model_version="legacy_residual_v1"),
        {"context_length": 8192},
    )
    assert effective.quality_model_version == "effective_capacity_v2"
    assert legacy.quality_model_version == "legacy_residual_v1"
    assert math.isclose(
        effective.predicted_loss, legacy.predicted_loss, rel_tol=0, abs_tol=1e-12
    )


def test_moe_capacity_moves_into_effective_spine():
    result = estimate_quality(
        _moe(),
        TrainingConfig(
            training_tokens=20_000_000_000_000,
            sequence_length=8192,
        ),
        {"context_length": 8192},
    )
    assert result.spine_effective_params > result.spine_active_params
    assert result.spine_effective_params < result.n_total_params
    assert result.terms["effective_capacity"].delta < 0
    assert (
        result.terms["moe_residual"].features["subterms"]["capacity_bonus"]
        == 0.0
    )


def test_sparse_capacity_is_more_data_limited_at_2t_than_20t():
    arch = _moe()
    low = estimate_quality(
        arch,
        TrainingConfig(training_tokens=2_000_000_000_000),
        {"context_length": 8192},
    )
    central = estimate_quality(
        arch,
        TrainingConfig(training_tokens=20_000_000_000_000),
        {"context_length": 8192},
    )
    low_gain = low.spine_effective_params / low.spine_active_params
    central_gain = central.spine_effective_params / central.spine_active_params
    assert central_gain > low_gain > 1.0


def test_repeated_tokens_are_discounted():
    arch = _dense()
    result = estimate_quality(
        arch,
        TrainingConfig(
            training_tokens=20_000_000_000_000,
            unique_tokens=5_000_000_000_000,
        ),
        {"context_length": 8192},
    )
    assert 5_000_000_000_000 < result.training_tokens < 20_000_000_000_000
    assert result.unique_training_tokens == 5_000_000_000_000
    assert math.isclose(result.data_repetitions, 3.0)
    assert result.terms["effective_data"].delta > 0


def test_context_utility_is_zero_at_pretraining_context():
    arch = _hybrid()
    training = TrainingConfig(
        training_tokens=20_000_000_000_000,
        sequence_length=8192,
    )
    at_reference = estimate_quality(
        arch, training, {"context_length": 8192}
    )
    long_context = estimate_quality(
        arch, training, {"context_length": 1_000_000}
    )
    assert at_reference.terms["context_utility"].value == 0.0
    assert math.isclose(
        at_reference.pretraining_loss_proxy,
        at_reference.predicted_loss,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        at_reference.pretraining_loss_proxy,
        long_context.pretraining_loss_proxy,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert long_context.terms["context_utility"].value != 0.0


def test_short_chat_traffic_does_not_receive_long_context_bonus():
    result = estimate_quality(
        _hybrid(),
        TrainingConfig(
            training_tokens=20_000_000_000_000,
            sequence_length=8192,
        ),
        {
            "context_length": 1_000_000,
            "traffic_mix": {"short_chat": 1.0},
        },
    )
    assert math.isclose(
        result.terms["context_utility"].value,
        0.0,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_auto_calibration_can_fit_effective_component_multipliers():
    rows = [
        {
            "predicted_loss": 2.0,
            "observed_loss": 1.98,
            "effective_capacity_delta": -0.02,
            "effective_data_delta": 0.0,
        },
        {
            "predicted_loss": 2.0,
            "observed_loss": 2.03,
            "effective_capacity_delta": 0.0,
            "effective_data_delta": 0.03,
        },
        {
            "predicted_loss": 2.0,
            "observed_loss": 2.01,
            "effective_capacity_delta": -0.02,
            "effective_data_delta": 0.03,
        },
    ]
    fit = _fit_effective_quality_components(rows)
    assert fit["n"] == 3
    assert math.isclose(fit["capacity_multiplier"], 2.0, rel_tol=1e-4)
    assert math.isclose(fit["data_multiplier"], 2.0, rel_tol=1e-4)
