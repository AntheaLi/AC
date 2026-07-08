"""Quality-model regression pins.

(Pins from Waves 18h and 20.)

  * MoE data-sufficiency flip: dense wins at 2T, MoE wins at 20T
    (34B-active / ~174B-total operating point named in the README), and
    N_eff drops below N_active when tokens/total-param is under parity.
  * Locality gate is a slope, not a plateau: an in-parity-band interleave
    carries a small nonzero locality cost that grows with local fraction.
  * Vocab residual: undersized vocab at 7B carries a penalty; 128k does
    not; oversized is not penalized here (the spine prices it); the
    penalty respects the configured cap.
  * One run-noise floor (`quality_model.run_noise_floor_pct`) shared by
    plan-ladder and the picker.
  * The cross-mixer bias floor constant ships in the quality constants.
"""

import os
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
AC_DIR = os.path.join(REPO, "ac")
if AC_DIR not in sys.path:
    sys.path.insert(0, AC_DIR)

from ac.quality_model import (  # noqa: E402
    ArchConfig as QArch, TrainingConfig, estimate_quality,
)


def _dense_34b():
    return QArch(d_model=7680, n_layers=56, n_heads=120, d_head=64,
                 n_kv_heads=8, ffn_dim=20480, vocab_size=32000)


def _moe_34b():
    return QArch(d_model=7680, n_layers=56, n_heads=120, d_head=64,
                 n_kv_heads=8, ffn_dim=20480, vocab_size=32000,
                 model_type="moe",
                 moe_config={"enabled": True, "n_experts": 64, "top_k": 8,
                             "expert_dim": 2560})


def test_moe_flip_2t_dense_20t_moe():
    """README's flagship effective_capacity_v2 behavior, now pinned."""
    at_2t_dense = estimate_quality(_dense_34b(), TrainingConfig(training_tokens=int(2e12)))
    at_2t_moe = estimate_quality(_moe_34b(), TrainingConfig(training_tokens=int(2e12)))
    at_20t_dense = estimate_quality(_dense_34b(), TrainingConfig(training_tokens=int(20e12)))
    at_20t_moe = estimate_quality(_moe_34b(), TrainingConfig(training_tokens=int(20e12)))
    assert at_2t_dense.loss_proxy < at_2t_moe.loss_proxy, (
        "at 2T (11.5 tokens/total-param) the under-trained MoE must LOSE to "
        "equal-active dense")
    assert at_20t_moe.loss_proxy < at_20t_dense.loss_proxy, (
        "at 20T (115 tokens/total-param) the MoE must WIN vs equal-active dense")


def test_effective_capacity_below_active_when_underfed():
    q = estimate_quality(_moe_34b(), TrainingConfig(training_tokens=int(2e12)))
    assert q.spine_effective_params < q.spine_active_params, (
        "N_eff must drop below N_active when tokens/total-param is under parity")


def test_locality_gate_slope_not_plateau():
    """In-parity interleaves carry a small, monotone locality cost."""
    def loss_for(frac):
        a = QArch(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, vocab_size=128256,
                  local_window=4096, local_attention_fraction=frac)
        return estimate_quality(
            a, TrainingConfig(training_tokens=int(20e12)),
            workload_spec={"context_length": 32768},
        ).loss_proxy

    full_global = loss_for(0.0)
    at_1_1 = loss_for(0.5)     # global_frac 0.5 — deep in parity band
    at_3_1 = loss_for(0.75)    # global_frac 0.25 — parity band
    at_7_1 = loss_for(0.875)   # global_frac 0.125 — at the floor
    # Nonzero cost inside the parity band (the old gate returned exactly 0).
    assert at_1_1 > full_global, "1:1 interleave must not be exactly free"
    # Monotone in local fraction.
    assert at_3_1 >= at_1_1
    assert at_7_1 >= at_3_1
    # But shallow: parity-band cost stays well under the whole-model-SWA cost.
    whole_swa = loss_for(1.0)
    assert (at_7_1 - full_global) < 0.5 * (whole_swa - full_global)


def test_vocab_residual_one_sided_and_capped():
    def term_for(vocab, d=4096, L=32):
        a = QArch(d_model=d, n_layers=L, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, vocab_size=vocab)
        q = estimate_quality(a, TrainingConfig(training_tokens=int(20e12)))
        return q.terms["vocab_residual"].value

    at_32k = term_for(32000)
    at_128k = term_for(128256)
    at_256k = term_for(256000)
    assert at_32k > 0, "32k vocab at 7B-scale must carry an undersized penalty"
    assert at_128k == 0.0, "128k vocab at 7B-scale is at/above the prior optimum"
    assert at_256k == 0.0, "oversized vocab is NOT penalized here (spine prices it)"
    # Wave 21: read the cap from the constants instead of hard-coding the
    # pre-recalibration value (0.015). The invariant under test is
    # "penalty respects the configured cap", not the cap's magnitude.
    from ac.quality_model import DEFAULT_QUALITY_CONSTANTS
    cap = float(DEFAULT_QUALITY_CONSTANTS["vocab_residual"]["cap"])
    assert at_32k <= cap + 1e-9, "penalty must respect the cap"


def test_run_noise_floor_shared():
    from quality_model import run_noise_floor_pct, load_quality_constants
    floor = run_noise_floor_pct()
    assert floor == pytest.approx(
        load_quality_constants()["paired_decision"]["run_noise_floor_pct"])
    # ladder_plan must consume the same accessor (no private re-derivation).
    import inspect
    import ladder_plan
    src = inspect.getsource(ladder_plan.plan_ladder)
    assert "run_noise_floor_pct()" in src


def test_cross_mixer_floor_constant_exists():
    from quality_model import load_quality_constants
    c = load_quality_constants()
    assert c["state_residual"]["cross_mixer_bias_floor_pct"] == \
        pytest.approx(2.8)
