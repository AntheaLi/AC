"""Cross-entry-point contracts for every public delta transformation."""

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac"))


def _baseline(name):
    from ac.baseline import load_baseline_model

    return load_baseline_model(str(ROOT / "configs" / name)).candidate


def _constraints(candidate, *, dp=8):
    from ac.optimizer import DeploymentConstraints

    return DeploymentConstraints(
        target_params_b=candidate.active_params_b or candidate.total_params_b,
        param_tolerance=0.75,
        training_tokens=int(2e12),
        pretraining_context_length=8192,
        context_length=131072,
        prompt_len=8192,
        output_len=1024,
        serving_batch=4,
        tp=8,
        pp=1,
        dp=dp,
        cp=1,
        max_candidates=16,
    )


DENSE_CASES = [
    ("interleave_local_attention", {"ratio": "1:1", "window": 4096}),
    ("swap_attention_to_gqa", {"group_size": 8}),
    ("swap_attention_to_mla", {"latent_dim": 512, "d_rope": 64}),
    ("swap_attention_to_swa", {"window_size": 4096}),
    ("add_state_layers", {"state_fraction": 0.75, "state_type": "mamba2"}),
    ("scale_d_model", {"delta": 1024}),
    ("scale_n_layers", {"delta": 8}),
    ("change_precision_per_component", {
        "weight": "fp8", "activation": "fp8", "kv": "int8",
    }),
    ("change_parallelism", {"tp": 4, "dp": 4}),
]


@pytest.mark.parametrize("name,args", DENSE_CASES)
def test_dense_registry_delta_is_visible_finite_and_decomposed(name, args):
    from ac.evaluator import evaluate_delta

    baseline = _baseline("mistral_7b.json")
    evaluation = evaluate_delta(
        baseline, "h100", _constraints(baseline), name, args,
        include_pareto=False)
    assert evaluation.feasible, evaluation.reason_if_infeasible
    substantive = [
        change for change in evaluation.field_changes
        if change["field"] != "applied_deltas"
    ]
    assert substantive, f"{name} became a field-level no-op"
    assert evaluation.stress_baseline is not None
    assert evaluation.stress_candidate is not None
    assert evaluation.quality_delta, f"{name} lost quality decomposition"
    assert "training_memory_per_gpu_gb" in evaluation.metrics
    for metric in evaluation.metrics.values():
        assert math.isfinite(metric.baseline)
        assert math.isfinite(metric.candidate)
        assert math.isfinite(metric.delta)


@pytest.mark.parametrize("name,args", [
    ("densify_first_k", {"k": 3}),
    ("change_moe_topology", {
        "n_experts": 64, "top_k": 2, "expert_dim": 3200,
    }),
    ("scale_d_model", {"delta": 704}),
])
def test_moe_registry_delta_is_visible_and_finite(name, args):
    from ac.evaluator import evaluate_delta

    baseline = _baseline("gpt_oss_120b.json")
    evaluation = evaluate_delta(
        baseline, "h100", _constraints(baseline, dp=16), name, args,
        include_pareto=False)
    assert evaluation.feasible, evaluation.reason_if_infeasible
    assert any(change["field"] != "applied_deltas"
               for change in evaluation.field_changes)
    assert evaluation.quality_delta
    for metric in evaluation.metrics.values():
        assert math.isfinite(metric.candidate)


def test_composed_projection_pattern_and_mixer_paths_are_feasible():
    from ac.evaluator import evaluate_delta_sequence

    dense = _baseline("mistral_7b.json")
    gqa_swa = evaluate_delta_sequence(
        dense, "h100", _constraints(dense), [
            ("swap_attention_to_gqa", {"group_size": 8}),
            ("swap_attention_to_swa", {"window_size": 4096}),
        ], include_pareto=False)
    assert gqa_swa.feasible, gqa_swa.reason_if_infeasible
    changed = {change["field"] for change in gqa_swa.field_changes}
    assert "n_kv_heads" in changed
    assert "attention.sliding_window" in changed

    moe = _baseline("gpt_oss_120b.json")
    mla_state = evaluate_delta_sequence(
        moe, "h100", _constraints(moe, dp=16), [
            ("swap_attention_to_mla", {"latent_dim": 256, "d_rope": 64}),
            ("add_state_layers", {"state_fraction": 0.25}),
        ], include_pareto=False)
    assert mla_state.feasible, mla_state.reason_if_infeasible
    assert mla_state.quality_delta
    changed = {change["field"] for change in mla_state.field_changes}
    assert "attention.type" in changed
    assert "state_enabled" in changed

