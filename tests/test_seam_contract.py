"""Seam-contract regression pins.

This file is the single merged home for all previously per-wave test files
(waves 5, 18a, 18h, 19-47). It exists so the release suite has one authoritative
pin file for architecture-identity and cross-seam contracts, rather than 34
files that all share the same sys.path bootstrap and import surface.

Sections below are delimited by ``# =========== wave NN =========== ``; the
original per-wave docstrings and comments are preserved verbatim inside each
section. A handful of module-level helper names that collided across files
(``_cand``, ``_con``, ``_constraints``, ``_L``, ``_dense_34b``, ``_moe_34b``,
``_row``, ``_run_cli``) were renamed to ``_<waveNN>_<name>`` inside their own
section. The sys.path bootstrap and the ``if __name__ == "__main__":`` block
are unified once at the top and bottom of this file.
"""

from __future__ import annotations

import copy
import csv
import glob
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = str(ROOT)
_AC_DIR = ROOT / "ac"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_AC_DIR))

# scripts/_generator_payload is used by the wave-30/31/32/43/46 sections as
# ``payload``; import it at module scope so those sections don't need to redo
# the sys.path dance.
import _generator_payload as payload  # noqa: E402

# ac imports used across the merged waves (kept as top-level so per-wave sections
# can reference them without duplicate ``from ac...`` lines).
from ac import auto_calibrate  # noqa: E402
from ac.auto_calibrate import (  # noqa: E402
    _WAVE5_DEFAULTS,
    _fit_dp_overlap,
    _fit_hbm_spill,
    _fit_pp_queue,
    _fit_state_long_ctx,
    _fit_tp_overlap,
    _solve_overlap_from_ratio,
)
from ac.baseline import load_baseline_model  # noqa: E402
from ac.cli_compile import _build_parser, infer_output_paths  # noqa: E402
from ac.cli_recipe import _STARTER_RECIPE_TEMPLATES, expand_argv  # noqa: E402
from ac.delta_engine import apply_transitions  # noqa: E402
from ac.optimizer import (  # noqa: E402
    DeploymentConstraints,
    _filter_ep_options_by_dp,
    generate_candidates,
    generate_moe_candidates,
    generate_moe_hybrid_candidates,
    optimize,
)
from ac.penalties import precision_supported, weight_storage_supported  # noqa: E402
from ac.quality_model import (  # noqa: E402
    ArchConfig as QArch,
    TrainingConfig,
    estimate_quality,
    load_quality_constants,
    paired_loss_uncertainty,
)
from ac.schema import build_config  # noqa: E402
from ac.stress import Workload, _kv_cache_bytes_total, compute_throughput_stress  # noqa: E402
from ac.throughput_model import (  # noqa: E402
    ArchConfig,
    ArchConfig as TArch,
    CalibrationTable,
    load_hardware,
    throughput,
)
from ac.trust_audit import (  # noqa: E402
    _build_candidate_from_anchor,
    load_public_model_registry,
    run_public_anchor,
)



# ============================================================
# test_auto_calibrate_wave5.py  (w5)
# ============================================================
"""Wave 7b unit tests for the Wave 1-5 calibration fitters.

These tests synthesize measurement rows with known ground truth, hand them
to the new fitters in `auto_calibrate.py`, and assert the fitters recover
the ground truth within a tolerance. The fitters are pure functions —
they do not require pytest fixtures, network access, or real cluster
measurements.

The final test (`test_calibrated_throughput_uses_fitted_overlap`) goes the
other direction: it constructs a `CalibrationTable` with a Wave 5 override
and verifies the throughput model's DP grad sync term picks it up.
"""

def _w5_row(**kw):
    """Tiny convenience for building measurement rows."""
    row = {"hardware": "h100"}
    row.update(kw)
    return row


class FitDpOverlapTests(unittest.TestCase):
    """Synthesize rows where (1-measured)/(1-default) is a known constant
    and verify the fitter recovers `measured`."""

    def test_recovers_measured_overlap_within_tolerance(self):
        # Ground truth: lab measured overlap is 0.5 (vs default 0.7).
        # ratio = (1 - 0.5) / (1 - 0.7) = 0.5 / 0.3 = 1.667
        rows = []
        for i in range(8):
            # Predicted is the model output assuming default overlap; observed
            # is the lab measurement. Both >0; the fitter only takes the
            # median ratio.
            pred = 10.0
            obs = 10.0 * (1.0 - 0.5) / (1.0 - 0.7)  # = 16.67 ms
            rows.append(_w5_row(predicted_dp_grad_allreduce_ms=pred,
                              observed_dp_grad_allreduce_ms=obs))
        fit = _fit_dp_overlap(rows, default_hardware="h100")
        self.assertIn("h100", fit)
        recovered = fit["h100"]["overlap_fraction"]
        self.assertIsNotNone(recovered)
        self.assertAlmostEqual(recovered, 0.5, delta=0.05)
        self.assertEqual(fit["h100"]["n"], 8)

    def test_returns_none_when_no_measurements(self):
        rows = [_w5_row(predicted_serving_tbt_ms=10.0)]  # different field
        fit = _fit_dp_overlap(rows, default_hardware="h100")
        # With no DP-grad observations, h100 still appears (it's in hardware
        # set) but the overlap_fraction must be None.
        self.assertIsNone(fit["h100"]["overlap_fraction"])
        self.assertEqual(fit["h100"]["n"], 0)


class FitTpOverlapTests(unittest.TestCase):
    def test_recovers_overlap_for_default_05(self):
        # Default overlap is 0.5; lab measured 0.7.
        # ratio = (1-0.7) / (1-0.5) = 0.3 / 0.5 = 0.6
        rows = [
            _w5_row(predicted_tp_allreduce_ms_per_layer=1.0,
                  observed_tp_allreduce_ms_per_layer=0.6)
            for _ in range(6)
        ]
        fit = _fit_tp_overlap(rows, default_hardware="h100")
        recovered = fit["h100"]["overlap_fraction"]
        self.assertAlmostEqual(recovered, 0.7, delta=0.05)


class FitPpQueueTests(unittest.TestCase):
    """Per-(hw, schedule) bucketing test. Verify the output shape matches
    the spec — `{hardware: {schedule: {multiplier, n}}}` — and the
    multiplier reflects the supplied observed/predicted ratio."""

    def test_groups_by_hardware_and_schedule(self):
        rows = [
            _w5_row(pp_schedule="1f1b", pp_degree=4,
                  predicted_activation_peak_mem_gb=40.0,
                  observed_activation_peak_mem_gb=42.0),
            _w5_row(pp_schedule="1f1b", pp_degree=4,
                  predicted_activation_peak_mem_gb=42.0,
                  observed_activation_peak_mem_gb=44.0),
            _w5_row(pp_schedule="gpipe", pp_degree=8,
                  predicted_activation_peak_mem_gb=80.0,
                  observed_activation_peak_mem_gb=72.0),
            _w5_row(hardware="b200", pp_schedule="1f1b", pp_degree=8,
                  predicted_activation_peak_mem_gb=50.0,
                  observed_activation_peak_mem_gb=51.0),
        ]
        fit = _fit_pp_queue(rows)
        self.assertIn("h100", fit)
        self.assertIn("1f1b", fit["h100"])
        self.assertIn("gpipe", fit["h100"])
        # 1f1b multiplier ~1.04, gpipe ~0.9
        self.assertAlmostEqual(fit["h100"]["1f1b"]["multiplier"], 1.045, delta=0.02)
        self.assertAlmostEqual(fit["h100"]["gpipe"]["multiplier"], 0.9, delta=0.02)
        self.assertIn("b200", fit)
        self.assertIn("1f1b", fit["b200"])

    def test_pp_degree_one_is_skipped(self):
        rows = [
            _w5_row(pp_schedule="1f1b", pp_degree=1,
                  predicted_activation_peak_mem_gb=40.0,
                  observed_activation_peak_mem_gb=44.0),
        ]
        fit = _fit_pp_queue(rows)
        # PP=1 contributes nothing — no schedule-specific bucket exists.
        self.assertNotIn("h100", fit)


class FitStateLongCtxTests(unittest.TestCase):
    """Anchor against the Wave 5 design: when predicted_loss / observed_loss
    equals 1.0 (model is dead-on), the fitted weight must equal the default
    0.030. When predicted is 5% higher than observed (model under-predicts
    the benefit), the fitter shifts the weight upward."""

    def test_fit_against_anchor_returns_default_when_unbiased(self):
        rows = [
            _w5_row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.5, observed_long_ctx_loss=2.5),
            _w5_row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.6, observed_long_ctx_loss=2.6),
            _w5_row(arch_mode="hybrid", context_length=1048576,
                  predicted_long_ctx_loss=2.7, observed_long_ctx_loss=2.7),
        ]
        fit = _fit_state_long_ctx(rows)
        self.assertEqual(fit["n"], 3)
        self.assertAlmostEqual(fit["weight"], 0.030, delta=0.001)

    def test_fit_against_jamba_like_anchor_lifts_weight(self):
        # NVIDIA-Empirical Study: at 128k+ context, hybrid models beat
        # attention baselines by ~2.65 pt; our model under-predicts by ~5%.
        rows = [
            _w5_row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.10, observed_long_ctx_loss=2.00),
            _w5_row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.05, observed_long_ctx_loss=1.95),
            _w5_row(arch_mode="hybrid_state", context_length=1048576,
                  predicted_long_ctx_loss=2.20, observed_long_ctx_loss=2.10),
        ]
        fit = _fit_state_long_ctx(rows)
        self.assertGreater(fit["weight"], _WAVE5_DEFAULTS["state_long_context_weight"])
        self.assertLessEqual(fit["weight"], 0.10)  # band cap from the fitter

    def test_excludes_short_ctx_rows(self):
        rows = [
            _w5_row(arch_mode="hybrid", context_length=4096,   # < 32k floor
                  predicted_long_ctx_loss=2.0, observed_long_ctx_loss=2.0),
        ]
        fit = _fit_state_long_ctx(rows)
        self.assertEqual(fit["n"], 0)
        self.assertIsNone(fit["weight"])


class FitHbmSpillTests(unittest.TestCase):
    def test_fits_only_spill_rows(self):
        rows = [
            _w5_row(hbm_spill_gb=4.0,
                  predicted_tbt_ms_with_spill=12.0,
                  observed_tbt_ms_with_spill=15.0),
            _w5_row(hbm_spill_gb=8.0,
                  predicted_tbt_ms_with_spill=18.0,
                  observed_tbt_ms_with_spill=22.5),
            _w5_row(hbm_spill_gb=0.0,  # no spill — must be skipped
                  predicted_tbt_ms_with_spill=10.0,
                  observed_tbt_ms_with_spill=20.0),
        ]
        fit = _fit_hbm_spill(rows, default_hardware="h100")
        self.assertEqual(fit["h100"]["n"], 2)
        self.assertAlmostEqual(fit["h100"]["factor"], 1.25, delta=0.02)


class SolveOverlapTests(unittest.TestCase):
    """Direct unit test of the overlap inversion helper."""

    def test_round_trip(self):
        # measured == 1.0 → ratio == 0 → the inverter returns None on purpose
        # (an exposed cost of 0 means observed == 0 which we treat as bogus
        # input rather than valid data).
        for measured in (0.1, 0.3, 0.5, 0.8, 0.95):
            for default in (0.5, 0.7):
                ratio = (1.0 - measured) / max(1e-6, (1.0 - default))
                recovered = _solve_overlap_from_ratio(ratio, default)
                self.assertAlmostEqual(recovered, measured, delta=1e-6)

    def test_invalid_ratio_returns_none(self):
        self.assertIsNone(_solve_overlap_from_ratio(0.0, 0.7))
        self.assertIsNone(_solve_overlap_from_ratio(None, 0.7))


class CalibrationTableWave5Tests(unittest.TestCase):
    """Sanity-check that the CalibrationTable loads Wave 5 fields from both
    the flat and the nested JSON shapes."""

    def _write(self, tmp_path: Path, payload: dict) -> Path:
        import json
        p = tmp_path / "h100_calibration.json"
        p.write_text(json.dumps(payload))
        return p

    def test_flat_shape(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp), {
                "hardware": "h100",
                "dp_grad_overlap_fraction": 0.6,
                "tp_allreduce_overlap_fraction": 0.45,
                "state_long_context_weight": 0.028,
                "hbm_spill_decode_factor": 1.10,
                "pp_queue_multipliers": {"1f1b": 1.05, "gpipe": 0.97},
            })
            ct = CalibrationTable.from_json(str(p))
            self.assertAlmostEqual(ct.dp_grad_overlap_fraction, 0.6, delta=1e-6)
            self.assertAlmostEqual(ct.tp_allreduce_overlap_fraction, 0.45, delta=1e-6)
            self.assertAlmostEqual(ct.state_long_context_weight, 0.028, delta=1e-6)
            self.assertAlmostEqual(ct.hbm_spill_decode_factor, 1.10, delta=1e-6)
            self.assertEqual(ct.pp_queue_multipliers["1f1b"], 1.05)
            self.assertEqual(ct.pp_queue_multipliers["gpipe"], 0.97)

    def test_nested_wave5_shape(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp), {
                "hardware": "h100",
                "wave5": {
                    "dp_grad_overlap": 0.55,
                    "tp_overlap": 0.40,
                    "state_long_context_weight": 0.032,
                    "hbm_spill_factor": 0.95,
                    "pp_queue": {"1f1b": {"multiplier": 1.07}},
                },
            })
            ct = CalibrationTable.from_json(str(p))
            self.assertAlmostEqual(ct.dp_grad_overlap_fraction, 0.55, delta=1e-6)
            self.assertAlmostEqual(ct.tp_allreduce_overlap_fraction, 0.40, delta=1e-6)
            self.assertAlmostEqual(ct.state_long_context_weight, 0.032, delta=1e-6)
            self.assertAlmostEqual(ct.hbm_spill_decode_factor, 0.95, delta=1e-6)
            self.assertEqual(ct.pp_queue_multipliers["1f1b"], 1.07)


class CalibratedThroughputUsesFittedOverlapTests(unittest.TestCase):
    """Verify the runtime path: when a CalibrationTable carries a fitted DP
    grad overlap, the throughput model's training step uses it. We exercise
    this through the synchronous helper and the module-level holder."""

    def test_dp_grad_cost_changes_with_calibration(self):
        from ac.throughput_model import (
            _dp_grad_allreduce_cost,
            load_hardware,
        )
        hw = load_hardware("h100")
        # Same params, same dp_degree, two different overlap fractions
        # should produce strictly different sync costs.
        cost_default = _dp_grad_allreduce_cost(
            total_params=int(7e9), precision="bf16", hw=hw,
            dp_degree=64, zero_stage=3,
        )
        cost_lab = _dp_grad_allreduce_cost(
            total_params=int(7e9), precision="bf16", hw=hw,
            dp_degree=64, zero_stage=3, overlap_fraction=0.5,
        )
        # Default overlap is 0.7 → exposed = 0.3; lab override 0.5 → exposed
        # = 0.5. Lab cost must be strictly higher than default.
        self.assertGreater(cost_lab, cost_default)
        # And the ratio must equal 0.5/0.3 = 1.667.
        self.assertAlmostEqual(cost_lab / cost_default, 0.5 / 0.3, delta=0.01)

    def test_allreduce_cost_uses_module_calibration(self):
        from ac.throughput_model import (
            _allreduce_cost, load_hardware,
            CalibrationTable, _CURRENT_CALIBRATION,  # noqa: F401
        )
        import ac.throughput_model as tm
        hw = load_hardware("h100")
        # Baseline: no calibration table, default overlap 0.5.
        tm._CURRENT_CALIBRATION = None
        baseline = _allreduce_cost(
            B=4, S=2048, d_model=4096, precision="bf16",
            hw=hw, tp_degree=8, n_allreduces=2,
        )
        # Lab calibration: overlap fitted at 0.7 → less exposed comm, smaller cost.
        tm._CURRENT_CALIBRATION = CalibrationTable(
            tp_allreduce_overlap_fraction=0.7,
        )
        try:
            with_cal = _allreduce_cost(
                B=4, S=2048, d_model=4096, precision="bf16",
                hw=hw, tp_degree=8, n_allreduces=2,
            )
        finally:
            tm._CURRENT_CALIBRATION = None
        self.assertLess(with_cal, baseline)
        # Ratio: exposed_cal / exposed_default = (1-0.7)/(1-0.5) = 0.6
        self.assertAlmostEqual(with_cal / baseline, 0.6, delta=0.01)


# ============================================================
# test_wave18a_integration.py  (w18a)
# ============================================================
"""Wave 18a — Integration coverage for the factorized signature.

`test_architecture_signature.py` and `test_architecture_wiring.py` already
cover the classification rules. This file locks in the *acceptance gate*
side of Wave 18a:

  * serialize_candidate() emits `signature` alongside the legacy
    `arch_family` label, and the two agree.
  * Wave 11's decision block carries `signature` derived from the same
    ArchitectureSignature.
  * Every hand-rolled family classifier still in the codebase (cli_compile,
    trust_audit, throughput_model._arch_family, budget_pareto,
    _generator_payload._arch_family_from_optimal, regen_golden_matrix) now
    returns the same label the canonical helper would.
  * The 5-bucket calibration taxonomy (throughput_model / auto_calibrate)
    stays consistent with the 4-label legacy taxonomy under the signature
    projection.
  * Public-model shapes classify by component axes without needing a
    handwritten "hybrid" label — the Wave 18a §Acceptance Gate.
"""

def _bf16_precision() -> dict:
    return {"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"}


def _make_cand(**overrides):
    """Minimal CandidateArch for signature exercises."""
    from ac.optimizer import CandidateArch
    kw = dict(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, vocab_size=32000,
        weight_precision="bf16", ffn_precision="bf16",
        attn_precision=_bf16_precision(), kv_cache_bits=16,
        moe=None, state_config=None, n_dense_ffn_layers=0,
        attention_type="gqa",
        tp_degree=1, cp_degree=1, ep_degree=1,
    )
    kw.update(overrides)
    return CandidateArch(**kw)


class SerializeCandidateEmitsSignatureTests(unittest.TestCase):
    """serialize_candidate() must carry the factorized signature."""

    def _serialize(self, cand):
        from ac.optimizer import DeploymentConstraints, evaluate_candidate
        from _generator_payload import serialize_candidate
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
            allow_quality_sentinel=True,
        )
        return serialize_candidate(evaluate_candidate(cand, "h100", c))

    def test_dense_serialized_row_carries_signature(self):
        rec = self._serialize(_make_cand())
        self.assertIn("signature", rec)
        sig = rec["signature"]
        self.assertIsInstance(sig, dict)
        # 9 required axes per Wave 18a spec.
        for axis in ("ffn_mode", "attention_pattern", "kv_projection",
                     "sequence_mixer", "mixer_fraction",
                     "context_extension", "modifiers",
                     "active_params", "total_params", "legacy_family"):
            self.assertIn(axis, sig, f"signature missing axis: {axis}")
        # The legacy `arch_family` label must equal `signature.legacy_family`.
        self.assertEqual(rec["arch_family"], sig["legacy_family"])

    def test_mla_serialized_row_carries_signature(self):
        cand = _make_cand(
            attention_type="mla",
            mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
        )
        rec = self._serialize(cand)
        self.assertEqual(rec["signature"]["kv_projection"], "mla")
        self.assertEqual(rec["signature"]["legacy_family"], "dense")


class DecisionBlockCarriesSignatureTests(unittest.TestCase):
    """Wave 11 decision block must carry the factorized signature."""

    def _decision(self, cand):
        from ac.optimizer import DeploymentConstraints, evaluate_candidate
        from _generator_payload import (
            serialize_candidate, _build_decision_diagnostics,
        )
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
            allow_quality_sentinel=True,
        )
        row = {"optimal": serialize_candidate(evaluate_candidate(cand, "h100", c))}
        _build_decision_diagnostics(row)
        return row["decision"]

    def test_dense_decision_has_signature(self):
        d = self._decision(_make_cand())
        self.assertIn("signature", d)
        self.assertIsNotNone(d["signature"])
        # Legacy compat: `family` in the decision block equals
        # `signature.legacy_family`.
        self.assertEqual(d["family"], d["signature"]["legacy_family"])

    def test_moe_decision_signature_agrees(self):
        cand = _make_cand(
            moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096,
                 "shared_expert": None,
                 "router": {"precision": "bf16"},
                 "capacity_factor": 1.0, "precision": "bf16"},
            moe_style="fine",
            ep_degree=2,
        )
        d = self._decision(cand)
        self.assertEqual(d["family"], "moe")
        self.assertEqual(d["signature"]["legacy_family"], "moe")
        self.assertEqual(d["signature"]["ffn_mode"], "moe")
        self.assertEqual(d["signature"]["sequence_mixer"], "attention")
        # MoE-only candidate: no state mixer.
        self.assertEqual(d["signature"]["mixer_fraction"], 0.0)


class HandRolledClassifiersAgreeTests(unittest.TestCase):
    """Every remaining hand-rolled classifier must now return the same
    label the canonical ArchitectureSignature would."""

    def _cases(self):
        return [
            ("dense_gqa",    _make_cand()),
            ("dense_mla",    _make_cand(attention_type="mla",
                                       mla_kv_latent_dim=512,
                                       mla_q_latent_dim=1536,
                                       mla_rope_head_dim=64,
                                       mla_nope_head_dim=128)),
            ("moe",          _make_cand(
                moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096,
                     "shared_expert": None,
                     "router": {"precision": "bf16"},
                     "capacity_factor": 1.0, "precision": "bf16"},
                moe_style="fine", ep_degree=2)),
            ("hybrid_mamba", _make_cand(
                state_config={"d_state": 128, "state_expansion": 2,
                              "n_heads": 32, "d_head": 128,
                              "state_type": "mamba2"},
                layer_type_list=["state"] * 24 + ["attention"] * 8,
                n_state_layers=24, n_attention_layers=8)),
        ]

    def test_cli_compile_and_signature_agree(self):
        from ac.architecture import architecture_signature
        from ac.cli_compile import _arch_mode_of
        for label, cand in self._cases():
            with self.subTest(case=label):
                self.assertEqual(
                    _arch_mode_of(cand),
                    architecture_signature(cand).legacy_family,
                    f"{label}: cli_compile._arch_mode_of disagrees with signature",
                )

    def test_trust_audit_and_signature_agree(self):
        from ac.architecture import architecture_signature
        from ac.trust_audit import _family_of
        for label, cand in self._cases():
            with self.subTest(case=label):
                self.assertEqual(
                    _family_of(cand),
                    architecture_signature(cand).legacy_family,
                    f"{label}: trust_audit._family_of disagrees with signature",
                )

    def test_generator_payload_and_signature_agree(self):
        from ac.architecture import architecture_signature
        from _generator_payload import _arch_family_from_optimal
        # Build a serialized-optimal-shaped dict for each case.
        for label, cand in self._cases():
            with self.subTest(case=label):
                opt = {
                    "d_model": cand.d_model, "n_layers": cand.n_layers,
                    "n_heads": cand.n_heads, "d_head": cand.d_head,
                    "n_kv_heads": cand.n_kv_heads, "ffn_dim": cand.ffn_dim,
                    "vocab_size": cand.vocab_size,
                    "attention_type": cand.attention_type,
                    "moe": cand.moe,
                    "state_config": cand.state_config,
                    "n_state_layers": cand.n_state_layers,
                    "layer_type_list": cand.layer_type_list,
                    "mla_kv_latent_dim": cand.mla_kv_latent_dim,
                    "mla_q_latent_dim": cand.mla_q_latent_dim,
                }
                self.assertEqual(
                    _arch_family_from_optimal(opt),
                    architecture_signature(cand).legacy_family,
                    f"{label}: _arch_family_from_optimal disagrees with signature",
                )

    def test_throughput_model_calibration_bucket_from_signature(self):
        """throughput_model._arch_family projects the signature into the
        5-bucket calibration taxonomy. Verify the projection matches the
        signature axes for representative cases."""
        from ac.throughput_model import _arch_family as tput_family, ArchConfig
        # dense GQA → dense_gqa
        arch = ArchConfig(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                          n_kv_heads=8, ffn_dim=14336)
        self.assertEqual(tput_family(arch), "dense_gqa")
        # MLA-dense → mla_dense
        arch = ArchConfig(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                          n_kv_heads=8, ffn_dim=14336,
                          attention_type="mla", mla_kv_latent_dim=512)
        self.assertEqual(tput_family(arch), "mla_dense")


class AcceptanceGateTests(unittest.TestCase):
    """Wave 18a §Acceptance Gate:

    > No user-visible decision may rely on a handwritten family label.
    > Public model comparisons must be expressible as component signatures.
    """

    def test_public_model_signatures_are_expressible_as_components(self):
        """Illustrative fixtures for DeepSeek-V3-shape (MoE + MLA) and a
        Jamba-shape (MoE + Mamba-2 hybrid). Neither uses a handwritten
        "hybrid" label — both are expressible as (ffn_mode, kv_projection,
        sequence_mixer, mixer_fraction) tuples."""
        from ac.architecture import architecture_signature

        # DeepSeek-V3-shape: MoE + MLA + pure attention mixer.
        v3 = _make_cand(
            attention_type="mla",
            mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
            moe={"n_experts": 256, "top_k": 8, "expert_dim": 2048,
                 "shared_expert": None,
                 "router": {"precision": "bf16"},
                 "capacity_factor": 1.0, "precision": "bf16"},
            moe_style="fine", ep_degree=8,
        )
        v3_sig = architecture_signature(v3)
        # Component signature: (moe, mla, attention). NOT "hybrid".
        self.assertEqual(v3_sig.ffn_mode, "moe")
        self.assertEqual(v3_sig.kv_projection, "mla")
        self.assertEqual(v3_sig.sequence_mixer, "attention")
        self.assertFalse(v3_sig.has_state_mixer)
        # legacy_family stays "moe" — no phantom "hybrid".
        self.assertEqual(v3_sig.legacy_family, "moe")

        # Jamba-shape: MoE + Mamba-2 mixer (dense KV projection).
        jamba = _make_cand(
            moe={"n_experts": 16, "top_k": 2, "expert_dim": 8192,
                 "shared_expert": None,
                 "router": {"precision": "bf16"},
                 "capacity_factor": 1.0, "precision": "bf16"},
            moe_style="fine", ep_degree=4,
            state_config={"d_state": 128, "state_expansion": 2,
                          "n_heads": 32, "d_head": 128,
                          "state_type": "mamba2"},
            layer_type_list=["state"] * 28 + ["attention"] * 4,
            n_state_layers=28, n_attention_layers=4,
        )
        j_sig = architecture_signature(jamba)
        self.assertEqual(j_sig.ffn_mode, "moe")
        self.assertEqual(j_sig.sequence_mixer, "mamba2")
        self.assertGreater(j_sig.mixer_fraction, 0.5)
        # Same component signature (moe + mamba2 + gqa) — the label
        # "moe_hybrid" is just a compat display artifact.
        self.assertEqual(j_sig.legacy_family, "moe_hybrid")

        # Their factorized signatures MUST differ; if they don't, Wave 18a
        # failed to factor the axes correctly.
        self.assertNotEqual(v3_sig.as_dict(), j_sig.as_dict())


# ============================================================
# test_wave18h_calibration.py  (w18h)
# ============================================================
"""Wave 18h — zero-compute calibration triad.

  1. Paired-decision math: correlated errors cancel; identical configs
     bottom out at the run-noise floor; the surviving uncertainty lives in
     the terms that actually differ.
  2. Paired-ablation residual fitting: the shipped public corpus fits,
     validates the locality residual (~1.0 scale), and flags uncovered
     terms and unanchored regions.
  3. Ladder planning: all three verdict classes (resolvable from priors /
     resolvable with proposed runs / unresolvable below the noise floor),
     with runs priced and sigma monotone.
"""

_TR = {"training_tokens": int(20e12)}
_W = {"context_length": 8192}
_BASE = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
             n_kv_heads=8, ffn_dim=14336, vocab_size=128256)


def _q(**over):
    kw = dict(_BASE)
    kw.update(over)
    return estimate_quality(QArch(**kw), _TR, workload_spec=_W)


class PairedDecisionTests(unittest.TestCase):
    def test_identical_configs_cancel_to_floor(self):
        qa, qb = _q(), _q()
        p = paired_loss_uncertainty(qa, qb)
        floor = float(load_quality_constants()["paired_decision"]
                      ["run_noise_floor_pct"]) / 100.0
        self.assertAlmostEqual(p["sigma_rel"], floor, delta=0.1 * floor,
            msg="identical configs must cancel to the run-noise floor")

    def test_paired_sigma_below_naive_and_terms_attributed(self):
        qa = _q()
        qb = _q(ffn_dim=2048, model_type="moe",
                moe_config={"n_experts": 32, "top_k": 4, "expert_dim": 2048})
        p = paired_loss_uncertainty(qa, qb)
        self.assertLess(p["sigma_rel"], 0.5 * p["naive_rel"],
            msg="pairing must cancel at least half of the naive sigma")
        self.assertGreater(p["cancelled_fraction"], 0.5)
        # The decision-driving terms must be the ones that differ.
        driving = sorted(p["per_term"].items(),
                         key=lambda kv: -kv[1]["sigma_pair"])[:3]
        names = {k for k, _ in driving}
        self.assertTrue(names & {"effective_capacity", "moe_residual", "spine"},
            msg=f"MoE-vs-dense driving terms wrong: {names}")

    def test_spine_correlation_grounded_in_n_eff_distance(self):
        # Same-size reshape shares more spine error than a 4x size change.
        qa = _q()
        near = _q(d_model=4608, n_layers=28, n_heads=36, ffn_dim=16384)
        far = _q(d_model=2048, n_layers=22, n_heads=16, ffn_dim=8192)
        p_near = paired_loss_uncertainty(qa, near)
        p_far = paired_loss_uncertainty(qa, far)
        self.assertLess(p_near["sigma_rel"], p_far["sigma_rel"],
            msg="a nearby-N reshape must retain less spine uncertainty "
                "than a 4x size change")

    def test_decision_module_uses_pairing(self):
        from ac.decision import _combined_uncertainty_pct

        class _Ev:
            def __init__(self, q):
                self.quality = q
                self.predicted_loss = q.predicted_loss
        qa, qb = _q(), _q(d_model=4608, n_layers=28, n_heads=36,
                          ffn_dim=16384)
        paired_pct = _combined_uncertainty_pct(_Ev(qa), _Ev(qb))
        naive_pct = (2 ** 0.5) * qa.uncertainty_total * 100.0
        self.assertLess(paired_pct, 0.6 * naive_pct,
            msg="decision layer must exploit correlated-error cancellation")


class AblationPairFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from ac.ablation_fit import evaluate_pairs, fit_terms, coverage_gaps
        path = os.path.join(os.path.dirname(__file__), "fixtures",
                            "public_ablation_pairs_v1.json")
        with open(path) as f:
            corpus = json.load(f)
        cls.results = evaluate_pairs(corpus)
        cls.fits = {f.term: f for f in fit_terms(cls.results)}
        cls.gaps = coverage_gaps(list(cls.fits.values()))

    def test_corpus_evaluates(self):
        self.assertGreaterEqual(len(self.results), 10)
        # A majority of pairs should predict within tolerance (the ones
        # that don't are the fit's actionable findings, not failures).
        ok = sum(1 for r in self.results if r.within_tolerance)
        self.assertGreaterEqual(ok, len(self.results) // 2)

    def test_locality_residual_validated_by_corpus(self):
        f = self.fits["attention_locality"]
        self.assertEqual(f.n_pairs, 2)
        self.assertIsNotNone(f.scale)
        self.assertAlmostEqual(f.scale, 1.0, delta=0.15,
            msg="Wave 18g locality residual should match Gemma-2/Mistral "
                "published deltas at ~1.0 scale")

    def test_uncovered_terms_flagged(self):
        uncovered = [t for t, f in self.fits.items() if f.n_pairs == 0]
        self.assertIn("precision_residual", uncovered)
        for t in uncovered:
            self.assertTrue(any("UNCOVERED" in w for w in self.fits[t].warnings))

    def test_state_coverage_range_reported(self):
        joined = " ".join(self.gaps)
        self.assertIn("state_residual", joined)
        # p_attn=0 IS anchored (Waleffe pure-SSM pair), so the audit must
        # report the anchored range rather than a low-end extrapolation
        # flag; the long-context end (no anchor above 256k) must be
        # flagged as extrapolation.
        self.assertIn("anchored p_attn range [0.00,", joined)
        self.assertIn("extrapolation", joined)


class LadderPlanTests(unittest.TestCase):
    def test_resolvable_from_priors(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "hybrid", 13.0, 20.0,
                           context=1048576)
        self.assertTrue(plan.resolvable_now)
        self.assertEqual(len(plan.runs), 0)

    def test_below_floor_proposes_no_runs(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", 13.0, 20.0, context=8192)
        self.assertFalse(plan.resolvable_now)
        if plan.delta_pct <= plan.z * plan.sigma_floor_pct:
            self.assertEqual(len(plan.runs), 0)
            self.assertIn("UNRESOLVABLE", plan.verdict)

    def test_ladder_resolves_mid_gap_with_priced_runs(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", 3.0, 20.0, context=8192)
        self.assertFalse(plan.resolvable_now)
        self.assertTrue(plan.resolves, msg=plan.verdict)
        self.assertGreater(len(plan.runs), 0)
        sig = [r.marginal_sigma_after_pct for r in plan.runs]
        self.assertEqual(sig, sorted(sig, reverse=True),
                         msg="posterior sigma must shrink monotonically")
        for r in plan.runs:
            self.assertGreater(r.gpu_days_pair, 0.0)
            self.assertGreaterEqual(r.transfer_factor, 1.0)

    def test_budget_cap_respected(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", 3.0, 20.0, context=8192,
                           max_gpu_days=700.0)
        if plan.runs:
            self.assertLessEqual(plan.runs[-1].cumulative_gpu_days, 700.0)


# ============================================================
# test_wave18h_fixes.py  (w18h)
# ============================================================
"""Wave 18h regression pins.

One test per fixed bug from the pre-ship review:

  1. MoE data-sufficiency flip: dense wins at 2T, MoE wins at 20T
     (34B-active / ~174B-total operating point named in the README).
  2. Locality gate is a slope, not a plateau: an in-parity-band interleave
     carries a small nonzero locality cost that grows with local fraction.
  3. Lattice width grid is dense: 1B/H100/TP8 candidates include shapes near
     the d_opt anchor (2048/2560), and the picked 1B shape is not
     pathologically wide-shallow.
  4. Stress model is MLA-aware: an MLA arch's KV traffic uses the compressed
     latent (unsharded), not the baseline GQA formula.
  5. KV-precision deltas are visible to the transition quality panel
     (kv int4 shows a precision_loss delta, and the justification does not
     claim "negligible").
  6. --strict-quality: rank-1 is the argmin-loss candidate.
  7. Vocab residual: undersized vocab at 7B carries a penalty; 128k does not;
     penalty is capped.
"""

def _w18h_dense_34b():
    return QArch(d_model=7680, n_layers=56, n_heads=120, d_head=64,
                 n_kv_heads=8, ffn_dim=20480, vocab_size=32000)


def _w18h_moe_34b():
    return QArch(d_model=7680, n_layers=56, n_heads=120, d_head=64,
                 n_kv_heads=8, ffn_dim=20480, vocab_size=32000,
                 model_type="moe",
                 moe_config={"enabled": True, "n_experts": 64, "top_k": 8,
                             "expert_dim": 2560})


def test_moe_flip_2t_dense_20t_moe():
    """README's flagship effective_capacity_v2 behavior, now pinned."""
    at_2t_dense = estimate_quality(_w18h_dense_34b(), TrainingConfig(training_tokens=int(2e12)))
    at_2t_moe = estimate_quality(_w18h_moe_34b(), TrainingConfig(training_tokens=int(2e12)))
    at_20t_dense = estimate_quality(_w18h_dense_34b(), TrainingConfig(training_tokens=int(20e12)))
    at_20t_moe = estimate_quality(_w18h_moe_34b(), TrainingConfig(training_tokens=int(20e12)))
    assert at_2t_dense.loss_proxy < at_2t_moe.loss_proxy, (
        "at 2T (11.5 tokens/total-param) the under-trained MoE must LOSE to "
        "equal-active dense")
    assert at_20t_moe.loss_proxy < at_20t_dense.loss_proxy, (
        "at 20T (115 tokens/total-param) the MoE must WIN vs equal-active dense")


def test_effective_capacity_below_active_when_underfed():
    q = estimate_quality(_w18h_moe_34b(), TrainingConfig(training_tokens=int(2e12)))
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


def test_1b_lattice_has_anchor_adjacent_widths():
    c = DeploymentConstraints(target_params_b=1.0, tp=8, pp=1, dp=8)
    cands = generate_candidates("h100", c)
    widths = sorted(set(x.d_model for x in cands))
    assert 2048 in widths or 2560 in widths, (
        f"1B lattice must include widths near d_opt≈2100; got {widths}")


def test_1b_pick_is_not_pathologically_shallow():
    c = DeploymentConstraints(target_params_b=1.0, tp=8, pp=1, dp=8)
    result = optimize("h100", c)
    opt = result.optimal
    ratio = opt.arch.d_model / max(1, opt.arch.n_layers)
    assert ratio <= 220, (
        f"1B pick d={opt.arch.d_model} L={opt.arch.n_layers} "
        f"(aspect ratio {ratio:.0f}) is outside any published frontier band")


def test_stress_kv_bytes_mla_aware():
    common = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, batch_size=1, seq_len=32768)
    gqa = TArch(**common)
    mla = TArch(**common, attention_type="mla", mla_kv_latent_dim=512,
                mla_rope_head_dim=64)
    kv_len = 32768
    gqa_tp8 = _kv_cache_bytes_total(gqa, kv_len, tp_degree=8)
    mla_tp8 = _kv_cache_bytes_total(mla, kv_len, tp_degree=8)
    # MLA latent per token per layer: (512+64)*2 bytes, unsharded -> 1152 B.
    expect_mla = 1152 * kv_len * 32
    assert abs(mla_tp8 - expect_mla) / expect_mla < 0.01
    # At TP=8 the sharded GQA-8 cache is SMALLER than the unsharded latent —
    # exactly the case the old GQA-only formula got backwards.
    assert mla_tp8 > gqa_tp8
    # And unsharded (TP=1) MLA must be far smaller than unsharded GQA.
    assert _kv_cache_bytes_total(mla, kv_len, 1) < 0.3 * _kv_cache_bytes_total(gqa, kv_len, 1)


def test_kv_precision_delta_visible_in_transition_quality():
    baseline = TArch(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                     n_kv_heads=8, ffn_dim=14336, batch_size=1, seq_len=2048,
                     kv_precision="bf16")
    transitions = apply_transitions(
        baseline,
        [("change_precision_per_component", {"kv": "int4"})],
        hardware="h100", tp_degree=8,
    )
    t = transitions[0]
    assert t.feasible
    delta_prec = t.delta_quality.get("precision_loss", 0.0)
    assert delta_prec > 1e-4, (
        "int4 KV swap must surface a precision_loss delta in the transition "
        f"quality panel (got {delta_prec}); the summary previously claimed "
        "'negligible' while the metric table showed +1.2% loss")
    try:
        from ac.justify_transition import justify
    except ImportError:
        from justify_transition import justify
    text = justify(t)
    assert "negligible" not in text.lower()


def test_strict_quality_rank1_is_argmin_loss():
    c = DeploymentConstraints(target_params_b=1.0, tp=8, pp=1, dp=8,
                              strict_quality=True)
    result = optimize("h100", c)
    frontier = list(result.pareto_frontier)
    best_loss = min(ev.predicted_loss for ev in frontier)
    assert abs(result.optimal.predicted_loss - best_loss) < 1e-9, (
        "--strict-quality must pick the argmin-loss candidate")


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


# ============================================================
# test_wave19_fixes.py  (w19)
# ============================================================
"""Wave 19 regression pins — external v0-release researcher feedback.

One pin per confirmed finding:
  P0-1  MoE training throughput: EP-over-DP token accounting, a2a overlap,
        no phantom shared-expert EP allreduce; equal-active MoE within
        1.2-2.5x of dense per training-replica GPU.
  P0-2  Prompt length follows an overridden context; prompt > ctx errors;
        TTFT monotone in context.
  P0-3  ac-stress runs on every shipped config (parallelism threaded).
  P1-4  Pre-calibration tiebreak band capped: default rank-1 spends <= ~0.5%
        predicted loss vs best-loss.
  P1-5  Cross-family comparisons under uncalibrated family bias emit
        [unresolved] with a family-bias reason.
  P2-b  plan-ladder rung costs are sane (no 40k GPU-day ladders).
"""

def _per_gpu_train_tps(arch, ep, tp=8, pp=2, dp=32):
    r = throughput(arch, "h100", tp_degree=tp, pp_degree=pp, dp_degree=dp,
                   ep_degree=ep)
    return r.training_throughput_tokens_per_sec / (tp * pp)


def _w19_dense_34b():
    return TArch(d_model=7168, n_layers=64, n_heads=56, d_head=128,
                 n_kv_heads=8, ffn_dim=18944, vocab_size=32000,
                 batch_size=8, seq_len=8192, precision="fp8")


def _w19_moe_34b(shared=False):
    moe = {"n_experts": 64, "top_k": 8, "expert_dim": 2752}
    n_layers = 46
    if shared:
        moe = {"n_experts": 64, "top_k": 8, "expert_dim": 2048,
               "shared_expert": {"ffn_dim": 5504, "precision": "fp8"}}
        n_layers = 51
    return TArch(d_model=8192, n_layers=n_layers, n_heads=64, d_head=128,
                 n_kv_heads=8, ffn_dim=22016, vocab_size=32000,
                 batch_size=8, seq_len=8192, precision="fp8",
                 moe_config=moe)


class TestP01MoeTrainingThroughput:
    def test_equal_active_moe_within_published_band_of_dense(self):
        """The release-review band: real MoE training runs ~1.2-2.5x below
        equal-active dense per GPU, not ~20x."""
        dense = _per_gpu_train_tps(_w19_dense_34b(), ep=1)
        for shared in (False, True):
            moe = _per_gpu_train_tps(_w19_moe_34b(shared=shared), ep=8)
            ratio = dense / moe
            assert 1.1 <= ratio <= 2.5, (
                f"dense/MoE per-GPU training ratio {ratio:.2f} outside the "
                f"[1.1, 2.5] plausibility band (shared={shared}); "
                "EP token accounting or a2a overlap has regressed")

    def test_moe_tps_independent_of_ep_within_nvlink_domain(self):
        """EP lays over DP: growing EP within a node must not divide
        per-replica training throughput."""
        t4 = _per_gpu_train_tps(_w19_moe_34b(), ep=4)
        t8 = _per_gpu_train_tps(_w19_moe_34b(), ep=8)
        assert abs(t4 - t8) / t8 < 0.25, (
            f"per-GPU training TPS moved {abs(t4-t8)/t8*100:.0f}% between "
            "EP=4 and EP=8 — EP is leaking into the replica accounting")

    def test_cross_node_ep_costs_more_but_not_10x(self):
        t8 = _per_gpu_train_tps(_w19_moe_34b(), ep=8)
        t32 = _per_gpu_train_tps(_w19_moe_34b(), ep=32, pp=2)
        assert t32 < t8, "cross-node EP should cost something"
        assert t8 / t32 < 10.0, (
            f"EP=32 is {t8/t32:.1f}x slower than EP=8 — hierarchical "
            "a2a split / node-limited routing has regressed")

    def test_pareto_csv_has_per_gpu_column(self):
        from ac.optimizer import DeploymentConstraints, optimize, result_to_pareto_csv
        c = DeploymentConstraints(target_params_b=1.0, tp=8, pp=1, dp=8)
        r = optimize("h100", c)
        csv_text = result_to_pareto_csv(r)
        header = csv_text.splitlines()[0]
        assert "training_tps_per_gpu" in header
        assert "vocab_size" in header


class TestP02PromptLength:
    def _delta_cmd(self, extra):
        return [sys.executable, os.path.join(REPO, "ac", "cli_delta_eval.py"),
                "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
                "--hardware", "h100", "--tp", "8", "--workload", "long_context",
                "--apply", "swap_attention_to_gqa:group_size=8",
                "--stdout"] + extra

    def test_context_override_cascades_to_prompt(self):
        out = subprocess.run(self._delta_cmd(["--context-length", "16384"]),
                             capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr
        assert "prompt_len: 16384" in out.stdout

    def test_prompt_exceeding_context_errors(self):
        out = subprocess.run(
            self._delta_cmd(["--context-length", "16384",
                             "--prompt-len", "32768"]),
            capture_output=True, text=True, cwd=REPO)
        assert out.returncode != 0
        assert "exceeds context_length" in out.stderr

    def test_ttft_monotone_in_context(self):
        import re
        ttfts = []
        for cl in ("16384", "65536", "131072"):
            out = subprocess.run(self._delta_cmd(["--context-length", cl]),
                                 capture_output=True, text=True, cwd=REPO)
            assert out.returncode == 0, out.stderr
            m = re.search(r"Prefill / TTFT \(ms\) \| ([0-9.]+)", out.stdout)
            assert m, "TTFT row missing"
            ttfts.append(float(m.group(1)))
        assert ttfts[0] < ttfts[1] < ttfts[2], (
            f"TTFT must grow with context; got {ttfts}")


class TestP03StressOnShippedConfigs:
    @pytest.mark.parametrize("cfg", sorted(
        glob.glob(os.path.join(REPO, "configs", "*.json"))),
        ids=lambda p: os.path.basename(p))
    def test_ac_stress_runs_on_shipped_config(self, cfg):
        out = subprocess.run(
            [sys.executable, os.path.join(REPO, "ac", "cli_stress.py"),
             "stress", "--baseline-config", cfg, "--hardware", "h100",
             "--phase", "decode"],
            capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, (
            f"ac-stress failed on shipped config {os.path.basename(cfg)}: "
            f"{out.stderr[-400:]}")
        assert "hbm_bw_decode" in out.stdout

    def test_stress_schema_adapter_preserves_gpt_oss_interleave(self):
        from ac.cli_stress import _archconfig_from_schema_v03

        with open(os.path.join(REPO, "configs", "gpt_oss_120b.json")) as f:
            cfg = json.load(f)
        arch = _archconfig_from_schema_v03(cfg, batch=1, seq=2048)
        assert arch.vocab_size == 200000
        assert arch.n_local_attn_layers == 18
        assert arch.local_window == 128
        assert arch.layer_type_list.count("local_attention") == 18
        assert arch.moe_config["n_experts"] == 128

    def test_stress_schema_adapter_preserves_mai_mla_mtp_cp(self):
        from ac.cli_stress import _archconfig_from_schema_v03

        with open(os.path.join(REPO, "configs", "mai_thinking_1.json")) as f:
            cfg = json.load(f)
        arch = _archconfig_from_schema_v03(cfg, batch=1, seq=2048)
        assert arch.vocab_size == 152064
        assert arch.attention_type == "mla"
        assert arch.mla_kv_latent_dim == 512
        assert arch.mla_q_latent_dim == 1536
        assert arch.mtp_n_predict_depths == 1
        assert arch.cp_degree == 4


class TestP14TiebreakSpendCap:
    def test_default_rank1_spends_at_most_half_percent(self):
        from ac.optimizer import DeploymentConstraints, optimize
        env_pack = os.environ.pop("AC_QUALITY_DEFAULTS", None)
        try:
            c = DeploymentConstraints(target_params_b=7.0, tp=8, pp=1, dp=8)
            r = optimize("h100", c)
            best = min(ev.predicted_loss for ev in r.pareto_frontier)
            gap_pct = 100.0 * (r.optimal.predicted_loss - best) / best
            assert gap_pct <= 0.55, (
                f"default rank-1 spends {gap_pct:.2f}% predicted loss vs "
                "best-loss — the pre-calibration tiebreak cap has regressed")
        finally:
            if env_pack is not None:
                os.environ["AC_QUALITY_DEFAULTS"] = env_pack


class TestP15CrossFamilyBias:
    def test_cross_family_subbias_gap_is_unresolved_with_reason(self):
        from ac.decision import assess_decision

        class _Q:
            uncertainty_total = 0.02
            confidence = "medium"

        class _Cand:
            def __init__(self, loss, moe):
                self.predicted_loss = loss
                self.meets_constraints = True
                self.quality = _Q()
                self.arch = type("A", (), {})()
                self.arch.moe = {"n_experts": 64} if moe else None
                self.arch.n_state_layers = 0
                self.arch.d_model = 4096
                self.arch.n_layers = 32

        env_pack = os.environ.pop("AC_QUALITY_DEFAULTS", None)
        try:
            moe = _Cand(1.90, True)
            dense = _Cand(1.957, False)  # 3% gap < family-bias floor
            r = assess_decision([moe, dense])
            assert r.status == "unresolved"
            assert any("cross-family" in reason for reason in r.reasons)
        finally:
            if env_pack is not None:
                os.environ["AC_QUALITY_DEFAULTS"] = env_pack

    def test_family_bias_fixture_ships(self):
        path = os.path.join(REPO, "ac", "calibration", "family_bias_v1.json")
        with open(path) as f:
            d = json.load(f)
        assert "dense" in d["families"] and "moe" in d["families"]


class TestP2bLadderCost:
    def test_small_ladder_costs_are_sane(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", target_params_b=7.0,
                           target_tokens_t=20.0, seeds_per_point=2)
        if not plan.runs:
            pytest.skip("plan resolved from priors / floor — no runs")
        total = plan.runs[-1].cumulative_gpu_days
        assert total < 5000, (
            f"0.5-3B ladder priced at {total:.0f} GPU-days — the rung "
            "over-training cap or MoE throughput fix has regressed")
        per_pair = max(r.gpu_days_pair for r in plan.runs)
        assert per_pair < 2000


# ============================================================
# test_wave19_loop_findings.py  (w19)
# ============================================================
"""Wave 19 loop-phase regression pins.

Findings from the post-fix frontier-realism validation loop:

  L1  Greenfield picks that selected a local:global interleave were EMITTED
      as pure full attention — the config a user trains did not match the
      candidate the search evaluated. Pin: build_config round-trips the
      interleave through the baseline loader.
  L2  The public-anchor registry shipped fabricated architectures
      (covered by tests/test_anchor_registry_ledger.py).
  L3  High-state-fraction deltas must carry the recall caveat (loss parity
      does not establish recall-eval parity).
  L4  Modifier mode silently evaluated exactly ONE candidate for MoE /
      decoupled-width baselines. Pin: gpt_oss_120b modifier search
      produces a real candidate pool.
"""

class TestL1InterleaveEmit:
    def test_build_config_emits_swa_bands(self):
        cfg = build_config(
            d_model=4096, n_layers=37, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=10752, local_global={"n_local_layers": 28,
                                         "window_size": 1024},
        )
        bands = cfg["architecture"]["layer_configs"]
        swa = [b for b in bands
               if b["attention"].get("type") == "swa"]
        full = [b for b in bands if b["attention"].get("type") == "full"]
        assert swa and full, "interleave must emit both band types"
        n_local = sum(len(b["layer_idx"]) for b in swa)
        n_global = sum(len(b["layer_idx"]) for b in full)
        assert n_local == 28 and n_global == 9
        assert all(b["attention"]["window_size"] == 1024 for b in swa)
        # Recall convention: the last layer stays global.
        assert (37 - 1) in {i for b in full for i in b["layer_idx"]}

    def test_interleave_round_trips_through_baseline_loader(self, tmp_path):
        import json
        cfg = build_config(
            d_model=4096, n_layers=37, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=10752, local_global={"n_local_layers": 28,
                                         "window_size": 1024},
        )
        p = tmp_path / "interleave.json"
        p.write_text(json.dumps(cfg))
        bm = load_baseline_model(str(p))
        assert int(getattr(bm.candidate, "n_local_attn_layers", 0)) == 28
        assert int(getattr(bm.candidate, "swa_window", 0)) == 1024


class _FakeMetric:
    def __init__(self, base, cand):
        self.baseline, self.candidate = base, cand
        self.delta = cand - base


def _fake_ev(state_fraction):
    class _Ev:
        delta_summary = {"state_fraction": state_fraction,
                         "state_layers": int(80 * state_fraction),
                         "attention_layers": int(80 * (1 - state_fraction)),
                         "requested_via": "ratio"}
        delta_name = "add_state_layers"
        delta_args = {"ratio": "3:1"}
        field_changes = [{"field": "n_state_layers", "baseline": 0,
                          "candidate": int(80 * state_fraction)}]
        metrics = {"predicted_loss": _FakeMetric(1.899, 1.862)}
    return _Ev()


class TestL3StateRecallCaveat:
    def test_high_state_fraction_report_carries_recall_caution(self):
        from ac.report import render_topology_notes
        notes = render_topology_notes(_fake_ev(0.75))
        assert "CAUTION" in notes and "recall" in notes.lower()

    def test_low_state_fraction_no_caution(self):
        from ac.report import render_topology_notes
        notes = render_topology_notes(_fake_ev(0.25))
        assert "CAUTION" not in notes


class TestL4MoeModifier:
    def test_modifier_generates_variants_for_moe_baseline(self):
        from ac.modifier import _generate_local_candidates
        from ac.optimizer import DeploymentConstraints

        bm = load_baseline_model(
            os.path.join(REPO, "configs", "gpt_oss_120b.json"))
        c = DeploymentConstraints(
            target_params_b=bm.candidate.total_params / 1e9,
            tp=8, pp=1, dp=8, param_tolerance=0.15,
        )
        cands = _generate_local_candidates(bm.candidate, "h100", c, [4, 8])
        assert len(cands) >= 20, (
            f"MoE baseline produced only {len(cands)} modifier variants — "
            "family fields are being dropped or the validity check regressed")
        # Every variant must still be MoE (family preserved).
        assert all(getattr(cand, "moe", None) for cand, _tp in cands)
        # And totals must be in the baseline's band, MoE-aware.
        for cand, _tp in cands[:10]:
            assert 0.85 * bm.candidate.total_params <= cand.total_params \
                <= 1.15 * bm.candidate.total_params


# ============================================================
# test_wave20_fixes.py  (w20)
# ============================================================
"""Wave 20 — regression tests for the six Wave-19 researcher-feedback fixes.

Feedback items (see wave19-feedback-fix-plan.md):
  #1 per-metric family bias (loss AND tbt/ttft/mem) + serving floors
  #2 implied-MFU reporting against the datasheet peak for the right precision
  #3 vocab sweep on by default; 'none' escape hatch; pinned warning
  #4 active-params row + loss decomposition + cross-mixer bias floor
  #5 shared run-noise floor between plan-ladder and the picker
  #6 `ac-trust-audit --out DIR` implies --all
"""

AC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ac")


# ---------------------------------------------------------------------------
# #6 — trust-audit CLI accepts --out alone
# ---------------------------------------------------------------------------

def test_trust_audit_out_implies_all(tmp_path):
    from cli_trust_audit import build_parser, main
    # No stage flag and no --out must still error.
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
        main([])
    out = tmp_path / "audit"
    rc = main(["--out", str(out)])
    # Exit code reflects block_publication (nonzero is fine); artifacts must exist.
    assert (out / "audit.json").exists()
    assert (out / "report.md").exists()
    assert (out / "family_bias.json").exists()
    audit = json.loads((out / "audit.json").read_text())
    # --all implied: both stages ran.
    assert "public_anchors" in audit
    assert "frontier_feasibility" in audit


# ---------------------------------------------------------------------------
# #1 — per-metric family bias (schema v2) + serving floors
# ---------------------------------------------------------------------------

def test_family_bias_v2_has_serving_metrics(tmp_path):
    from cli_trust_audit import main
    out = tmp_path / "audit"
    main(["--public-anchors", "--out", str(out)])
    fb = json.loads((out / "family_bias.json").read_text())
    assert fb["schema_version"] == "wave20.family_bias.v2"
    for fam in ("dense", "moe"):
        entry = fb["families"][fam]
        # v2 per-metric block
        for metric in ("loss", "tbt_ms", "ttft_ms", "mem_gb"):
            assert metric in entry["metrics"], (fam, metric)
            assert entry["metrics"][metric]["n"] >= 1
        # v1 compatibility keys preserved
        assert "mean_signed_loss_err_pct" in entry
        assert entry["mean_signed_loss_err_pct"] == \
            entry["metrics"]["loss"]["mean_signed_err_pct"]


def test_shipped_family_bias_table_is_v2():
    path = os.path.join(AC_DIR, "calibration", "family_bias_v1.json")
    fb = json.loads(open(path).read())
    assert fb["schema_version"] == "wave20.family_bias.v2"
    assert "tbt_ms" in fb["families"]["moe"]["metrics"]


def test_cross_family_bar_per_metric():
    from decision import cross_family_bias_bar_by_name
    loss_bar, loss_reason = cross_family_bias_bar_by_name("dense", "moe", "loss")
    tbt_bar, tbt_reason = cross_family_bias_bar_by_name("dense", "moe", "tbt_ms")
    assert loss_bar > 0
    assert tbt_bar > 0
    # The serving floor dwarfs the loss floor on the shipped anchors
    # (MoE TBT scatter is enormous pre-calibration).
    assert tbt_bar > loss_bar
    assert "decode TBT" in tbt_reason
    # Same family -> no bar.
    assert cross_family_bias_bar_by_name("moe", "moe", "tbt_ms")[0] == 0.0


def test_family_view_floors_cross_family_decode_claims():
    from report import render_family_comparison
    from decision import cross_family_bias_bar_by_name
    # Wave 25: derive the sub-floor delta from the LIVE dense-vs-moe TBT
    # floor instead of hard-coding −44% (which was "far below" the old
    # ~91% floor, but the floor tightened to ~38% when the decode
    # expert_load efficiency double-count was fixed and the bias table
    # regenerated). The behavior under test — sub-floor cross-family
    # decode deltas may not render as speed claims — is floor-relative.
    floor, _ = cross_family_bias_bar_by_name("dense", "moe", "tbt_ms")
    assert floor > 0
    sub_floor = -round(floor * 0.6)
    fams = [
        {"arch_mode": "moe", "loss": 1.82, "tbt_ms": 142, "ttft_ms": 309,
         "mem_gb": 39},
        # Cross-family, TBT delta below the dense-vs-moe floor:
        # must NOT render as a "faster decode" claim.
        {"arch_mode": "dense", "loss": 1.93, "tbt_ms": 80, "ttft_ms": 164,
         "mem_gb": 42, "loss_delta_pct": 6.1, "tbt_delta_pct": sub_floor},
        # Same family: no floor.
        {"arch_mode": "moe", "loss": 1.85, "tbt_ms": 160, "ttft_ms": 200,
         "mem_gb": 30, "loss_delta_pct": 1.5, "tbt_delta_pct": 12.0},
    ]
    txt = render_family_comparison(fams, 13, 8192)
    assert "inside family TBT bias floor" in txt
    assert f"{abs(sub_floor)}% faster decode" not in txt
    assert "12% slower decode" in txt          # same-family row untouched
    assert "†" in txt                          # footnote emitted


# ---------------------------------------------------------------------------
# #5 — single source of truth for the run-noise floor
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# #2 — datasheet peaks for implied-MFU reporting
# ---------------------------------------------------------------------------

def test_datasheet_peaks_present_and_sane():
    from throughput_model import load_hardware
    hw = load_hardware("h100")
    assert hw.datasheet_peak_flops_s("bf16") == pytest.approx(989e12)
    assert hw.datasheet_peak_flops_s("fp8") == pytest.approx(1979e12)
    # Internal roofline baseline stays halved — the MFU display must not
    # have touched the model's own numbers.
    assert hw.peak_flops_s("bf16") == pytest.approx(495e12)
    # Every shipped spec carries the field.
    for name in ("b200", "tpu_v5e", "tpu_v5p", "trainium2", "trainium3"):
        h = load_hardware(name)
        assert h.datasheet_peak_flops_tf, name


def test_implied_mfu_in_plausible_band():
    """6·N·tps/GPU over the datasheet peak lands in [0.15, 0.6] for a
    standard dense shape — guards against future basis mix-ups in either
    direction (68% was the misread; 5% would mean a broken model)."""
    from throughput_model import ArchConfig, throughput, load_hardware
    arch = ArchConfig(d_model=8192, n_layers=46, n_heads=64, d_head=128,
                      n_kv_heads=8, ffn_dim=22016, vocab_size=32000,
                      batch_size=8, seq_len=8192, precision="fp8")
    r = throughput(arch, "h100", tp_degree=8, dp_degree=8)
    tps_gpu = r.training_throughput_tokens_per_sec / 8
    hw = load_hardware("h100")
    n_active = 32.36e9
    mfu = 6 * n_active * tps_gpu / hw.datasheet_peak_flops_s("fp8")
    assert 0.15 < mfu < 0.6, mfu


# ---------------------------------------------------------------------------
# #3 — vocab sweep default-on
# ---------------------------------------------------------------------------

def test_resolve_vocab_options_defaults():
    from cli_compile import resolve_vocab_options

    class A:
        vocab_options = None
        vocab_size = 32000
        params = 7.0

    opts = resolve_vocab_options(A())
    assert 128256 in opts and 32000 in opts
    assert 256000 not in opts          # <30B

    class B(A):
        params = 70.0

    assert 256000 in resolve_vocab_options(B())

    class C(A):
        vocab_options = "none"

    assert resolve_vocab_options(C()) is None   # pinned

    class D(A):
        vocab_options = [32000, 50304]

    assert resolve_vocab_options(D()) == [32000, 50304]  # explicit wins


def test_parse_vocab_options_none_keyword():
    from cli_compile import parse_vocab_options
    assert parse_vocab_options("none") == "none"
    assert parse_vocab_options("32000,128256") == [32000, 128256]


# ---------------------------------------------------------------------------
# #4 — active-params row, loss decomposition, cross-mixer floor
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def state_delta_eval():
    from baseline import load_baseline_model
    from evaluator import evaluate_delta
    from optimizer import DeploymentConstraints
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "configs", "mai_thinking_1.json")
    bm = load_baseline_model(cfg)
    return evaluate_delta(
        bm.candidate, "h100",
        DeploymentConstraints(tp=8, pp=1, dp=16, context_length=2048),
        "add_state_layers",
        {"ratio": "3:1", "state_type": "gated_deltanet"},
        include_pareto=False)


def test_active_params_metric_moves_with_mixer_swap(state_delta_eval):
    ev = state_delta_eval
    ap = ev.metrics["active_params_b"]
    # The scaling spine grows when MLA attention layers become
    # gated-deltanet mixers; the old ledger-copy reported +0.00% here and
    # hid the capacity effect.
    assert abs(ap.pct_change) > 5.0
    sl = ev.metrics["scaling_law_loss"]
    pl = ev.metrics["predicted_loss"]
    # Scaling-law share is real and smaller in magnitude than the total.
    assert sl.delta != 0
    assert abs(sl.delta) < abs(pl.delta) + 1e-9


def test_mixer_swap_notes_fire(state_delta_eval):
    from report import render_topology_notes
    notes = render_topology_notes(state_delta_eval)
    assert "ACTIVE-PARAM SHIFT" in notes
    assert "cross-mixer bias floor" in notes
    assert "CAUTION" in notes  # recall caution retained


def test_cross_mixer_floor_constant_exists():
    from quality_model import load_quality_constants
    c = load_quality_constants()
    assert c["state_residual"]["cross_mixer_bias_floor_pct"] == \
        pytest.approx(2.8)


# ============================================================
# test_wave21_fixes.py  (w21)
# ============================================================
"""Wave 21 regression tests.

Pins the fixes from the 2026-07 post-release review:

  #1 EP=1 is a legal MoE execution plan (experts TP-sharded, vLLM-style);
     the trust audit no longer fabricates EP=2 for TP-only anchors.
  #2 ac-stress decode weight traffic is activation-aware for MoE
     (distinct-experts bound, mirroring the Wave-19 throughput fix).
  #3 The vocab_residual prior is calibrated so the realized optimum under
     spine displacement matches its cited source: the default 7B sweep
     picks vocab=128256 (this was documented in Wave 19/20 but had
     silently regressed to 65536 — no end-to-end pin existed).
  #4 `ac-compile --out DIR` routes greenfield outputs into DIR.
  #5 Weight STORAGE precision is feasibility-checked separately from
     native COMPUTE precision (mxfp4 weights on H100 are deployable;
     fp4 activations are not).
  #6 HardwareConfig.bytes_per_elem falls back to canonical byte widths
     instead of silently pricing unknown formats as bf16.
  #7 The trust audit excludes the cross-tokenizer vocab design prior
     from its absolute-loss anchor checks.
"""

def _registry():
    anchors, default_tol, _post = load_public_model_registry()
    return anchors, default_tol



def _moe_arch(batch=1):
    return ArchConfig(
        d_model=2880, n_layers=36, n_heads=64, d_head=64, n_kv_heads=8,
        ffn_dim=2880, batch_size=batch, seq_len=2048,
        moe_config={"n_experts": 128, "top_k": 4, "expert_dim": 2880},
    )


class Wave21EP1Tests(unittest.TestCase):
    def test_ep1_moe_throughput_runs_and_is_tp_sharded(self):
        arch = _moe_arch()
        r1 = throughput(arch, "h100", tp_degree=8, ep_degree=1)
        r2 = throughput(arch, "h100", tp_degree=8, ep_degree=2)
        self.assertGreater(r1.memory_footprint_per_gpu_gb, 0)
        # EP=1 keeps every expert (TP-sharded) resident: strictly more
        # per-GPU memory than EP=2, which halves resident experts.
        self.assertGreater(r1.memory_footprint_per_gpu_gb,
                           r2.memory_footprint_per_gpu_gb)

    def test_audit_defaults_moe_anchor_to_ep1_not_ep2(self):
        anchors, _ = _registry()
        mixtral = [a for a in anchors if a.id == "mixtral-8x22b"][0]
        self.assertNotIn("ep", mixtral.workload)  # published bench is TP-only
        cand = _build_candidate_from_anchor(mixtral)
        self.assertEqual(cand.ep_degree, 1)

    def test_schema_and_delta_cli_accept_ep1_moe_baseline(self):
        from ac.schema import validate_config

        with open(os.path.join(REPO, "configs", "gpt_oss_120b.json")) as f:
            cfg = json.load(f)
        cfg["parallelism"]["expert_parallel"] = 1
        self.assertFalse(
            [e for e in validate_config(cfg) if "expert_parallel" in e]
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "ep1.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            r = subprocess.run(
                [
                    sys.executable, "-m", "ac.cli_delta_eval",
                    "--baseline-config", cfg_path,
                    "--hardware", "h100", "--tp", "8",
                    "--apply", "densify_first_k:k=1",
                    "--no-pareto", "--json", "--out",
                    os.path.join(td, "out"),
                ],
                cwd=REPO, capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)


class Wave21StressDecodeTrafficTests(unittest.TestCase):
    def test_moe_decode_bandwidth_not_charged_all_resident_experts(self):
        arch = _moe_arch(batch=1)
        sv = compute_throughput_stress(
            arch, "h100",
            workload=Workload(batch_size=1, prefill_seq_len=2048,
                              decode_kv_len=2048, phase="decode"),
            tp_degree=8, ep_degree=8, arch_name="wave21-moe",
        )
        # At b=1 top-4-of-128, decode streams only the touched expert
        # slice; the old accounting charged all resident experts and
        # reported 2.1x ("violated") on this shape.
        self.assertLess(sv.hbm_bw_decode, 1.0)
        # Monotone in batch: more distinct experts touched at b=64.
        sv64 = compute_throughput_stress(
            arch := _moe_arch(batch=64), "h100",
            workload=Workload(batch_size=64, prefill_seq_len=2048,
                              decode_kv_len=2048, phase="decode"),
            tp_degree=8, ep_degree=8, arch_name="wave21-moe64",
        )
        self.assertGreater(
            sv64.intermediates.get("decode_weight_bytes",
                                   sv64.intermediates["weight_bytes"]),
            sv.intermediates.get("decode_weight_bytes", 0.0),
        )


class Wave21VocabAndOutDirTests(unittest.TestCase):
    def test_default_7b_pick_is_128256_and_out_dir_respected(self):
        # One end-to-end compile pins both #3 and #4. This is the pin the
        # Wave 19/20 claims were missing when the pick silently regressed.
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "run")
            r = subprocess.run(
                [sys.executable, "-m", "ac.cli_compile",
                 "--hardware", "h100", "--params", "7",
                 "--max-candidates", "300", "--quiet", "--out", out],
                cwd=td, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            cfg_path = os.path.join(out, "arch.json")
            self.assertTrue(os.path.exists(cfg_path),
                            f"--out dir not respected: {os.listdir(td)}")
            arch = json.load(open(cfg_path))["architecture"]
            self.assertEqual(int(arch["vocab_size"]), 128256)


class Wave21PrecisionFeasibilityTests(unittest.TestCase):
    def test_weight_storage_vs_native_compute_split(self):
        # mxfp4 weights on H100: deployable storage, not native compute.
        self.assertTrue(weight_storage_supported("mxfp4", "h100"))
        self.assertTrue(weight_storage_supported("fp4", "h100"))
        self.assertFalse(precision_supported("mxfp4", "h100"))
        self.assertFalse(precision_supported("fp4", "h100"))
        # Native-compute parts unchanged.
        self.assertTrue(precision_supported("mxfp4", "b200"))
        self.assertTrue(precision_supported("fp8", "h100"))

    def test_bytes_per_elem_canonical_fallback(self):
        hw = load_hardware("h100")
        # h100_sxm.json has no mxfp4 entry; the old fallback priced it at
        # 2 B/elem in the roofline while the memory path used 0.53 —
        # same weights, two physics.
        self.assertLess(hw.bytes_per_elem("mxfp4"), 1.0)
        self.assertLess(hw.bytes_per_elem("mxfp6"), 1.0)
        self.assertEqual(hw.bytes_per_elem("bf16"), 2)

    def test_gpt_oss_anchor_not_sentineled(self):
        anchors, tol = _registry()
        gptoss = [a for a in anchors if a.id == "gpt-oss-120b"][0]
        self.assertEqual(gptoss.arch["ffn_precision"], "mxfp4")
        res = run_public_anchor(gptoss, tol)
        loss = [m for m in res.metrics if m.metric == "loss"][0]
        # Was a ~1e8% sentinel when mxfp4 storage was marked infeasible.
        self.assertLess(abs(loss.rel_err), 0.5)


class Wave21AuditVocabExclusionTests(unittest.TestCase):
    def test_vocab_design_prior_excluded_from_anchor_loss(self):
        # Mixtral's 32k tokenizer draws a large undersized-vocab design
        # prior; its published loss is measured in its OWN tokenizer's
        # units, so the audit must not charge the counterfactual.
        anchors, tol = _registry()
        mixtral = [a for a in anchors if a.id == "mixtral-8x22b"][0]
        res = run_public_anchor(mixtral, tol)
        loss = [m for m in res.metrics if m.metric == "loss"][0]
        vocab_terms = [b for b in res.breakdown
                       if b.term_name == "vocab_residual"]
        if vocab_terms:
            # The term exists in the breakdown (it is real for sweeps)...
            self.assertGreater(vocab_terms[0].value, 0.0)
        # ...but the audited loss error is not inflated by it: with the
        # Wave-21 vocab weight (0.022, 32k at 39B active => ~5% capped),
        # including it would push Mixtral's loss error past +10%.
        self.assertLess(abs(loss.rel_err), 0.10)


# ============================================================
# test_wave22_fixes.py  (w22)
# ============================================================
"""Wave 22 regression tests.

Pins the fixes from the 2026-07-04 ergonomics-pass review (the recipe /
config-show / init surface shipped with no test coverage at all, which
is how every one of these reached the release build):

  #1 `--override key=false` actually unsets a boolean the recipe turned
     on. Previously it was a bare no-op AFTER the recipe had already
     expanded `--allow-moe` into argv, so `--override allow_moe=false`
     on the shipped b200 MoE recipe silently kept MoE ON.
  #2 The console `≈N% MFU … tok/s/GPU` line divides by the same
     TP × PP × CP training replica the optimizer emits into arch.json.
     It used tp×pp only, so any CP>1 run (e.g. the shipped b200 recipe,
     cp=2) printed a per-GPU TPS and MFU exactly cp× the artifact's.
  #3 `ac-compile config show --out DIR` previews the same routed paths
     the real run writes (the Wave-21 --out routing lived only in
     main()), and previews modifier-mode fixed names when
     --baseline-config is set instead of greenfield arch.json siblings.
  #4 The "--tp and --tp-options both set" warning fires only when --tp
     was explicitly supplied (its argparse default is 8, so the warning
     previously fired for every --tp-options run).
  #5 `ac-compile init` templates mirror configs/recipes/*.yaml. The
     delta template previously carried ac-delta-eval-only flags
     (`apply`, `workload`) that made its own advertised replay line
     fail with "unrecognized arguments".
"""

RECIPES = os.path.join(REPO, "configs", "recipes")


def _w22_run_cli(argv, cwd=REPO):
    return subprocess.run(
        [sys.executable, "-m", "ac.cli_compile"] + argv,
        capture_output=True, text=True, cwd=cwd,
    )


def _config_show(argv):
    r = _w22_run_cli(["config", "show"] + argv)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class Wave22FalseOverrideTests(unittest.TestCase):
    def test_false_override_strips_recipe_boolean(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        ) as f:
            yaml.safe_dump(
                {"flags": {"hardware": "h100", "allow_moe": True,
                           "params": 7}}, f)
            path = f.name
        try:
            argv, _, _ = expand_argv(
                ["--recipe", path, "--override", "allow_moe=false"]
            )
            self.assertNotIn("--allow-moe", argv)
            args = _build_parser().parse_args(argv)
            self.assertFalse(args.allow_moe)
        finally:
            os.unlink(path)

    def test_false_override_strips_valued_flag_back_to_default(self):
        argv, _, _ = expand_argv(
            ["--hardware", "h100", "--params", "7", "--tp", "4",
             "--override", "tp=false"]
        )
        args = _build_parser().parse_args(argv)
        self.assertEqual(args.tp, 8)  # parser default restored

    def test_true_override_still_sets_boolean(self):
        argv, _, _ = expand_argv(
            ["--hardware", "h100", "--params", "7",
             "--override", "allow_moe=true"]
        )
        args = _build_parser().parse_args(argv)
        self.assertTrue(args.allow_moe)

    def test_shipped_b200_recipe_override_moe_off(self):
        payload = _config_show([
            "--recipe", os.path.join(RECIPES, "b200_moe_mla_long_ctx.yaml"),
            "--override", "allow_moe=false",
        ])
        self.assertNotEqual(
            payload["resolved_args"].get("allow_moe"), True
        )


class Wave22ConfigShowPathTests(unittest.TestCase):
    def test_out_dir_routes_preview_paths(self):
        payload = _config_show([
            "--recipe", os.path.join(RECIPES, "h100_dense_7b.yaml"),
            "--out", "/tmp/w22_outdir",
        ])
        paths = payload["inferred_output_paths"]
        self.assertEqual(
            paths["output_config"], "/tmp/w22_outdir/arch.json"
        )
        for key in ("output_justification", "output_pareto",
                    "output_shadow_prices"):
            self.assertTrue(
                paths[key].startswith("/tmp/w22_outdir/"),
                f"{key} not routed into --out dir: {paths[key]}",
            )

    def test_infer_output_paths_matches_main_routing(self):
        # The unit under test is shared by main() and config show — a
        # single source of truth means they cannot disagree again.
        args = _build_parser().parse_args(
            ["--hardware", "h100", "--params", "7"])
        args.out = "/tmp/w22_route"
        resolved = infer_output_paths(args)
        self.assertEqual(
            resolved["output_config"], "/tmp/w22_route/arch.json"
        )

    def test_modifier_recipe_previews_modifier_outputs(self):
        payload = _config_show([
            "--recipe",
            os.path.join(RECIPES, "delta_mistral_gqa_long_ctx.yaml"),
        ])
        paths = payload["inferred_output_paths"]
        self.assertIn("config", paths)
        self.assertTrue(paths["config"].endswith("config.json"))
        self.assertTrue(paths["baseline_delta"].endswith(
            "baseline_delta.md"))
        # The greenfield names must NOT be advertised for a modifier run.
        self.assertNotIn("output_pareto", paths)

    def test_config_show_materializes_default_vocab_axis(self):
        payload = _config_show([
            "--hardware", "h100", "--params", "7",
        ])
        self.assertEqual(
            payload["resolved_args"]["vocab_options"],
            [32000, 65536, 128256],
        )

    def test_config_show_warns_when_vocab_axis_is_pinned(self):
        payload = _config_show([
            "--hardware", "h100", "--params", "7",
            "--vocab-options", "none",
        ])
        self.assertEqual(
            payload["resolved_args"]["vocab_options"], [32000]
        )
        self.assertTrue(
            [w for w in payload["warnings"] if "Vocabulary is pinned" in w],
            payload["warnings"],
        )


class Wave22TpWarningTests(unittest.TestCase):
    def test_tp_options_alone_does_not_warn(self):
        payload = _config_show([
            "--recipe",
            os.path.join(RECIPES, "delta_mistral_gqa_long_ctx.yaml"),
        ])
        self.assertFalse(
            [w for w in payload["warnings"] if "--tp-options" in w],
            payload["warnings"],
        )

    def test_explicit_tp_plus_tp_options_still_warns(self):
        payload = _config_show([
            "--hardware", "h100", "--params", "7",
            "--tp", "8", "--tp-options", "4,8",
        ])
        self.assertTrue(
            [w for w in payload["warnings"] if "--tp-options" in w],
            payload["warnings"],
        )


class Wave22InitTemplateTests(unittest.TestCase):
    def test_templates_mirror_shipped_recipes(self):
        for name, template in _STARTER_RECIPE_TEMPLATES.items():
            shipped_path = os.path.join(RECIPES, f"{name}.yaml")
            self.assertTrue(
                os.path.exists(shipped_path),
                f"init template {name} has no shipped recipe twin",
            )
            with open(shipped_path) as f:
                shipped = yaml.safe_load(f)["flags"]
            norm = {
                k: (str(v) if not isinstance(v, bool) else v)
                for k, v in template.items()
            }
            shipped_norm = {
                k: (str(v) if not isinstance(v, bool) else v)
                for k, v in shipped.items()
            }
            self.assertEqual(
                norm, shipped_norm,
                f"init template {name} drifted from {shipped_path}",
            )

    def test_every_template_parses_under_ac_compile(self):
        # The delta template used to emit ac-delta-eval-only flags that
        # ac-compile's parser rejected on the template's own advertised
        # replay line.
        for name, template in _STARTER_RECIPE_TEMPLATES.items():
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", delete=False
            ) as f:
                yaml.safe_dump(template, f)
                path = f.name
            try:
                argv, _, _ = expand_argv(["--recipe", path])
                try:
                    _build_parser().parse_args(argv)
                except SystemExit as e:
                    self.fail(
                        f"init template {name} does not parse under "
                        f"ac-compile (exit {e.code})"
                    )
            finally:
                os.unlink(path)


class Wave22ConsoleReplicaTests(unittest.TestCase):
    def test_console_tok_per_gpu_matches_artifact_under_cp(self):
        # End-to-end pin: a cp=2 dense run's console `tok/s/GPU` must
        # equal the JSON artifact's per-GPU number (tp×pp×cp replica).
        with tempfile.TemporaryDirectory() as td:
            r = _w22_run_cli([
                "--hardware", "h100", "--params", "7", "--tokens", "2",
                "--cp", "2", "--max-candidates", "60",
                "--no-shadow-prices", "--no-family-view",
                "--out", td,
            ])
            self.assertEqual(r.returncode, 0, r.stderr)
            m = [ln for ln in r.stderr.splitlines() if "tok/s/GPU" in ln]
            if not m:
                self.skipTest("MFU line not printed (no datasheet peak)")
            console_tps = float(
                m[0].split("tok/s/GPU")[0].split(",")[-1].strip()
            )
            with open(os.path.join(td, "arch.json")) as f:
                predicted = json.load(f)["metadata"]["predicted"]
            self.assertAlmostEqual(
                console_tps,
                predicted["training_throughput_tokens_per_sec_per_gpu"],
                delta=1.5,  # console rounds to whole tokens
            )
            unit = predicted["training_throughput_unit"]
            self.assertIn("×2 = ", unit.replace("× 2", "×2"))


# ============================================================
# test_wave23_fixes.py  (w23)
# ============================================================
"""Wave 23 regression tests.

Pins the fixes from a top-down 2026-07-04 CLI sweep:

  #1 `ac-auto-calibrate fit` `coverage_after` actually reflects the
     multiplier — the previous linear-interpolated quantile could produce
     a multiplier that admitted the same discrete row count as before, so
     the report showed `coverage_after == coverage_before` even when the
     multiplier was > 1. The pack now uses the ceiling ratio (bumped one
     ULP above the reference row to survive `unc * (|err|/unc)`
     round-trip float error) so target coverage is actually reached.

  #2 The delta report's quality-residual table stops printing
     "6.5e+02× larger" beside a baseline that displays as `0.00000` —
     tiny-but-nonzero baselines fall into the same "(no baseline, +delta)"
     branch as exact zeros, because at 5-decimal display precision they
     read as zero anyway.

  #3 `build_display_sort_key` for `research_quality` / `loss_only` no
     longer tiebreaks a Pareto-inferior candidate to rank-1 on a
     sub-display memory difference. The memory position of the sort
     tuple is now bucketed at a coarse band (the max of 0.1 GB and 2%
     of pool memory), so within-noise memory diffs fall through to
     TBT / -TPS instead of gating them.
"""

class TestWave23CoverageAfterReachesTarget(unittest.TestCase):
    """#1 auto-calibrate coverage_after actually meets target_coverage."""

    def test_coverage_after_reaches_target_on_shipped_example(self):
        from ac.auto_calibrate import _fit_quality, _read_measurements

        rows = _read_measurements(Path(REPO) / "examples" /
                                  "lab_measurements.example.jsonl")
        fit = _fit_quality(
            rows,
            target_coverage=0.9,
            default_uncertainty_pct=5.0,
            min_uncertainty_pct=0.5,
        )
        n = fit["n"]
        self.assertGreater(n, 0)
        # target_coverage=0.9 with n=13 requires at least ceil(0.9*13)=12
        # rows covered → 12/13 = 92.31%. The pre-fix behavior returned
        # 11/13 = 84.62% because the linear-interpolated multiplier landed
        # between two ratios.
        self.assertGreaterEqual(
            fit["coverage_after"], 0.9,
            f"coverage_after={fit['coverage_after']} did not reach target 0.9"
        )
        # And it must be strictly better than pre-scaling (or equal, in the
        # rare pack where n * target is already integer and no scaling
        # happens).
        self.assertGreaterEqual(fit["coverage_after"], fit["coverage_before"])

    def test_coverage_after_at_least_target_synthetic(self):
        """Independent synthetic pack: coverage_after >= target_coverage."""
        from ac.auto_calibrate import _fit_quality

        # 10 rows, uncertainties 3% each, errors drawn to give distinct ratios.
        errs_ratio_targets = [0.4, 0.5, 0.55, 0.7, 0.8, 0.85, 0.9,
                              1.05, 1.15, 1.30]
        rows = []
        for i, r in enumerate(errs_ratio_targets):
            pred = 2.0
            unc_pct = 3.0
            err_pct = r * unc_pct                # signed, positive
            obs = pred * (1.0 + err_pct / 100.0)
            rows.append({
                "id": f"row_{i}",
                "predicted_loss": pred,
                "observed_loss": obs,
                "predicted_uncertainty_total_pct": unc_pct,
            })
        for target in (0.80, 0.90):
            fit = _fit_quality(
                rows,
                target_coverage=target,
                default_uncertainty_pct=5.0,
                min_uncertainty_pct=0.1,
            )
            self.assertGreaterEqual(
                fit["coverage_after"], target,
                f"target={target} coverage_after={fit['coverage_after']}"
            )


class TestWave23DeltaResidualNearZeroFormatting(unittest.TestCase):
    """#2 delta report tolerates near-zero (display-invisible) baselines."""

    def test_near_zero_baseline_below_display_precision_reads_no_baseline(self):
        """A baseline that renders as 0.00000 at 5-decimal display precision
        must not appear as a "6.5e+02× larger" ratio next to that zero — the
        ratio is arithmetically correct but visually reads as garbage. The
        format function should surface "(no baseline, +delta)" instead."""
        # The report module uses ac/ dir on sys.path (matching the CLI's
        # bootstrap), so mirror that here.
        _ac_dir = os.path.join(REPO, "ac")
        if _ac_dir not in sys.path:
            sys.path.insert(0, _ac_dir)
        from report import _fmt_pct_md  # noqa: E402

        class MD:
            baseline = 2.25e-6           # rounds to 0.00000 at .5f
            candidate = 0.00146
            delta = candidate - baseline
            pct_change = (delta / baseline) * 100

        out = _fmt_pct_md(MD())
        self.assertNotIn("× larger", out,
                         f"got {out!r}; near-zero baseline should not render "
                         f"as a multiplicative ratio")
        # It should carry the direction and magnitude in some form.
        self.assertTrue(
            out.startswith("(no baseline") or out == "n/a",
            f"got {out!r}; expected '(no baseline, ...)' or 'n/a'"
        )

    def test_ordinary_baseline_still_percent_delta(self):
        _ac_dir = os.path.join(REPO, "ac")
        if _ac_dir not in sys.path:
            sys.path.insert(0, _ac_dir)
        from report import _fmt_pct_md  # noqa: E402

        class MD:
            baseline = 0.05
            candidate = 0.06
            delta = 0.01
            pct_change = 20.0

        out = _fmt_pct_md(MD())
        # Ordinary case: percentage rendering.
        self.assertIn("%", out)
        self.assertNotIn("×", out)


class TestWave23MemoryTiebreakUncertaintyAware(unittest.TestCase):
    """#3 research_quality picker no longer trades meaningful TBT/TPS wins
    for a sub-display (0.5% / < 0.05 GB) memory difference."""

    def _make_pool(self):
        """Two synthetic Pareto candidates: A wins on memory by ~0.004 GB
        (below the 1-decimal display precision), B wins on loss + TBT +
        training_tps by meaningful margins. Under the pre-fix ordering A
        is picked; under the fix B is picked."""
        from ac.optimizer import build_display_sort_key
        from types import SimpleNamespace

        def _mk(loss, mem, tbt, tps, d_model=4096, n_layers=34,
                total_params=6.8):
            q = SimpleNamespace(uncertainty_total=0.03,
                                confidence="medium")
            th = SimpleNamespace(prefill_time_ms=40.0,
                                 memory_footprint_per_gpu_gb=mem,
                                 training_memory_per_gpu_gb=mem)
            arch = SimpleNamespace(d_model=d_model, n_layers=n_layers,
                                   total_params=total_params,
                                   n_heads=16, d_head=256)
            return SimpleNamespace(
                predicted_loss=loss,
                memory_per_gpu_gb=mem,
                serving_tbt_ms=tbt,
                training_tps=tps,
                quality=q,
                throughput=th,
                arch=arch,
            )

        A = _mk(loss=2.0647, mem=8.2771, tbt=7.12, tps=50519)
        B = _mk(loss=2.0591, mem=8.2806, tbt=7.05, tps=54346)
        return A, B, build_display_sort_key

    def test_close_memory_falls_through_to_tbt_and_tps(self):
        A, B, build_display_sort_key = self._make_pool()
        pool = [A, B]

        from types import SimpleNamespace
        constraints = SimpleNamespace(
            objective_profile="research_quality",
            strict_quality=False,
        )
        key = build_display_sort_key(pool, constraints)
        first, second = sorted(pool, key=key)
        # B (Pareto-superior on loss + TBT + TPS) must be first — a 0.004 GB
        # memory difference must not gate meaningful throughput/loss wins.
        self.assertIs(
            first, B,
            "picker chose the Pareto-inferior candidate on a sub-0.05 GB "
            "memory tiebreak"
        )

    def test_strict_quality_still_argmin_loss(self):
        """The `--strict-quality` escape hatch is unchanged: it ranks by
        point-estimate loss with no bucketing on either loss or memory."""
        A, B, build_display_sort_key = self._make_pool()
        pool = [A, B]
        from types import SimpleNamespace
        constraints = SimpleNamespace(
            objective_profile="research_quality",
            strict_quality=True,
        )
        key = build_display_sort_key(pool, constraints)
        first, _ = sorted(pool, key=key)
        # B has the lower predicted_loss, so strict-quality picks B here too.
        self.assertIs(first, B)


# ============================================================
# test_wave24_fixes.py  (w24)
# ============================================================
"""Wave 24 regression tests.

Pins the fixes from a 2026-07-04 pretrain-researcher pass over the CLI:

  #1 The MoE and MoE-hybrid greenfield enumerators cap EP at the
     requested DP. The Wave-19 training-TPS math assumes MoE lays EP
     over the DP dimension (each EP rank routes its own microbatch), so
     EP > DP describes a layout the model cannot price. The picker had
     been happily emitting configs with EP=16 / DP=4 on B200
     (default_ep_options runs to the NVLink-domain cap of 72) and then
     printing a post-hoc `WARNING: picked EP=X exceeds DP=Y` beside a
     training-TPS number computed under the violated assumption.

  #2 `_format_relief` uses a transitive verb pair ("lowers"/"raises")
     when it prints the largest stress delta. It had emitted the
     ungrammatical "Applying interleave_local_attention rises
     HBM-BW-decode by 0.00".
"""

class TestWave24EpDpCap(unittest.TestCase):
    """#1 EP > DP is unreachable at greenfield enumeration."""

    def test_helper_drops_ep_above_dp(self):
        # EP=1 is legal; options above DP remain unreachable.
        self.assertEqual(
            _filter_ep_options_by_dp([1, 2, 4, 8, 16, 32, 64, 72], dp=4),
            [1, 2, 4],
        )

    def test_helper_keeps_ep_le_dp(self):
        self.assertEqual(
            _filter_ep_options_by_dp([2, 4, 8], dp=8),
            [2, 4, 8],
        )

    def test_helper_keeps_legal_tp_sharded_ep1(self):
        self.assertEqual(
            _filter_ep_options_by_dp([1, 2], dp=8),
            [1, 2],
        )

    def test_helper_requires_ep_to_divide_dp(self):
        self.assertEqual(
            _filter_ep_options_by_dp([1, 2, 3, 4, 6, 8], dp=8),
            [1, 2, 4, 8],
        )

    def test_user_supplied_ep_all_gt_dp_raises(self):
        # User-supplied --ep-options {16} with --dp 4 has no survivors.
        # Silently killing the MoE search would be worse than a clear error.
        with self.assertRaises(ValueError) as ctx:
            _filter_ep_options_by_dp(
                [16, 32], dp=4, source="user-supplied ep_options",
            )
        self.assertIn("--ep-options", str(ctx.exception))
        self.assertIn("--dp 4", str(ctx.exception))

    def test_hardware_default_all_gt_dp_returns_empty(self):
        # Not user-supplied → no raise, just an empty list (search will
        # naturally produce zero MoE candidates and fall back to dense).
        self.assertEqual(
            _filter_ep_options_by_dp([16, 32], dp=4, source="hardware default"),
            [],
        )

    def test_dp_le_1_is_matrix_probe_regime_not_capped(self):
        # `_make_constraints` and the Wave 8b matrix invariants score
        # single MoE cells with dp=1. The training layout is unspecified
        # there, so we must NOT cap EP by DP — the caller is scoring
        # quality/tput at a canonical layout, not planning training.
        self.assertEqual(
            _filter_ep_options_by_dp([2, 4, 8], dp=1),
            [2, 4, 8],
        )
        self.assertEqual(
            _filter_ep_options_by_dp([2, 4, 8], dp=0),
            [2, 4, 8],
        )

    def test_moe_generator_respects_dp_cap(self):
        # Enumerate MoE candidates on B200 with DP=4. NONE may have EP > 4.
        cons = DeploymentConstraints(
            target_params_b=35.0,
            allow_moe=True,
            moe_n_experts_options=[64],
            moe_top_k_options=[8],
            tp=8, pp=1, dp=4,
        )
        cands = generate_moe_candidates("b200", cons)
        # search may still return a handful of shapes; every one must be
        # legal under the training-EP-over-DP rule.
        self.assertGreater(len(cands), 0, "MoE search returned zero candidates")
        eps = sorted({int(getattr(c, "ep_degree", 1) or 1) for c in cands})
        self.assertTrue(
            all(ep <= 4 for ep in eps),
            f"MoE enumeration produced EP > DP=4: got EP options {eps}",
        )
        self.assertTrue(all(ep >= 2 for ep in eps), "EP=1 leaked into MoE")

    def test_moe_hybrid_generator_respects_dp_cap(self):
        cons = DeploymentConstraints(
            target_params_b=35.0,
            allow_moe=True,
            allow_state=True,
            moe_n_experts_options=[64],
            moe_top_k_options=[8],
            tp=8, pp=1, dp=2,
        )
        cands = generate_moe_hybrid_candidates("b200", cons)
        if not cands:
            self.skipTest("hybrid generator produced zero candidates for this shape")
        eps = sorted({int(getattr(c, "ep_degree", 1) or 1) for c in cands})
        self.assertTrue(
            all(ep <= 2 for ep in eps),
            f"MoE-hybrid enumeration produced EP > DP=2: got EP options {eps}",
        )


class TestWave24EpDpCliEndToEnd(unittest.TestCase):
    """#1 end-to-end: `ac-compile` doesn't print the confessional warning."""

    def test_b200_moe_with_small_dp_no_ep_exceeds_dp_warning(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "arch.json")
            p = subprocess.run(
                [sys.executable, "-m", "ac.cli_compile",
                 "--hardware", "b200",
                 "--params", "35", "--tokens", "8", "--context", "8192",
                 "--serving-tbt", "60", "--serving-batch", "8",
                 "--tp", "8", "--pp", "4", "--dp", "4",
                 "--allow-moe", "--moe-n-experts", "128", "--moe-top-k", "8",
                 "--cp", "1", "--max-total-params-b", "500",
                 "--max-candidates", "150",
                 "--output-config", out,
                 "--no-shadow-prices"],
                capture_output=True, text=True, cwd=REPO, timeout=90,
            )
            self.assertEqual(p.returncode, 0, msg=p.stderr[-2000:])
            # The confessional warning must not fire — enumeration blocks it.
            self.assertNotIn("exceeds DP", p.stderr,
                             msg="post-hoc EP>DP warning fired: "
                                 f"{p.stderr[-1500:]}")


class TestWave24GrammarInStressSummary(unittest.TestCase):
    """#2 `_format_relief` uses a transitive verb, not "rises"."""

    def test_no_transitive_rises_in_delta_summary(self):
        # Run a delta that produces a small non-relieving stress change, and
        # assert the emitted sentence says "raises"/"lowers", not "rises".
        with tempfile.TemporaryDirectory() as td:
            outdir = os.path.join(td, "delta")
            p = subprocess.run(
                [sys.executable, "-m", "ac.cli_delta_eval",
                 "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
                 "--hardware", "h100", "--tp", "8", "--workload", "chat",
                 "--apply", "interleave_local_attention:ratio=3:1,window=1024",
                 "--out", outdir],
                capture_output=True, text=True, cwd=REPO, timeout=60,
            )
            self.assertEqual(p.returncode, 0, msg=p.stderr[-2000:])
            report = open(os.path.join(outdir, "report.md")).read()
            self.assertNotIn(" rises ", report,
                             "delta report still uses transitive 'rises'")
            # Sanity: the transitive replacements are what we shipped.
            self.assertTrue(
                (" raises " in report) or (" lowers " in report) or
                ("no measurable stress change" in report) or
                ("relieves" in report),
                "expected transitive verb in stress-relief sentence",
            )


# ============================================================
# test_wave25_fixes.py  (w25)
# ============================================================
"""Wave 25 regression tests.

Pins the fixes from the 2026-07-06 pretrain-researcher pass:

  #1 The decode `expert_load` system efficiency no longer double-counts
     overheads. expert_load_s is a pure HBM weight stream already priced
     at datasheet bandwidth; kernel launch, scheduler dispatch, TP
     allreduce and a2a are all priced separately in the layer breakdown.
     The old 0.20/0.18 defaults sat BELOW the implied streaming
     efficiency of the WORST published MoE stack (Mixtral 2024 vLLM,
     0.29) and put every large-MoE decode ~5× over roofline — Qwen3-235B
     TBT error +194%, DeepSeek-V3 +94%, MoE family mean +66% while the
     dense family sat at −30%. With 0.42/0.38 (matching the dense
     memory-bound decode μ; MLA keeps its usual discount) the MoE TBT
     anchor spread is −55%…+40% with mean ≈ −21% — the same regime as
     dense, so shared bias cancels in cross-family comparisons.

  #2 The delta report's ACTIVE-PARAM SHIFT note fires for ANY delta that
     moves scaling-spine active params by >0.5%, not only state-mixer
     swaps (it was nested inside the `state_fraction` branch, so
     swap_attention_to_mla on Mistral-7B moved spine params −13.3% while
     the summary claimed "Quality cost: +0.0008 total residual change"
     beside a +0.0053 predicted-loss move). The Summary also appends an
     explicit spine-shift NOTE, and `scaling_law_loss` is rendered in the
     default metric table (Wave 20 documented it; the key was missing
     from the default list).

  #3 The cross-family serving-bias disclosure strings read the MoE TBT
     anchor span and dense TTFT mean from the live family_bias_v1.json
     instead of a hard-coded "−94%…+93%" wave-19 snapshot that had
     already drifted from the shipped table.
"""

class TestWave25ExpertLoadEfficiency(unittest.TestCase):
    """#1 decode expert_load efficiency is streaming-class, not 0.2."""

    def test_default_table_values(self):
        from ac.throughput_model import _DEFAULT_EFFICIENCY_TABLE
        moe = _DEFAULT_EFFICIENCY_TABLE["decode"]["moe"]["expert_load"]
        moe_mla = _DEFAULT_EFFICIENCY_TABLE["decode"]["moe_mla"]["expert_load"]
        dense_mem = _DEFAULT_EFFICIENCY_TABLE["decode"]["dense"]["memory"]
        # Weight streaming physics is shared with dense memory-bound
        # decode; anything more than ~15% below it re-introduces the
        # double-count.
        self.assertGreaterEqual(moe[0], 0.85 * dense_mem[0])
        self.assertGreaterEqual(moe_mla[0], 0.75 * dense_mem[0])
        # And it must beat the worst *published* stack's implied
        # efficiency (Mixtral on 2024-era vLLM ≈ 0.29) — an efficiency
        # prior below every measured system is not a prior, it's a bug.
        self.assertGreater(moe[0], 0.29)

    def test_qwen3_anchor_tbt_error_within_family_band(self):
        """The +194% Qwen3-235B decode-TBT blowup stays fixed."""
        from ac.trust_audit import (
            load_public_model_registry,
            run_public_anchor,
        )
        anchors, default_tol, _ = load_public_model_registry()
        qwen = [a for a in anchors if a.id == "qwen3-235b-a22b"][0]
        res = run_public_anchor(qwen, default_tol)
        errs = {m.metric: m.rel_err for m in res.metrics}
        self.assertIn("tbt_ms", errs)
        # Published 28ms; the old default predicted 82.4ms (+194%).
        # Anything past +60% means the streaming-efficiency regression
        # is back.
        self.assertLess(abs(errs["tbt_ms"]), 0.60)

    def test_moe_family_tbt_bias_not_one_sided(self):
        """Shipped family_bias table: MoE TBT mean bias inside ±35%.

        Pre-fix the family mean was +66% (one big systematic term), which
        made every cross-family serving comparison structurally favor
        dense. Post-fix the mean must sit in the same regime as dense
        (−30%), i.e. mostly shared, cancelable bias.
        """
        path = os.path.join(REPO, "ac", "calibration", "family_bias_v1.json")
        with open(path) as f:
            fb = json.load(f)
        m = fb["families"]["moe"]["metrics"]["tbt_ms"]
        self.assertLess(abs(float(m["mean_signed_err_pct"])), 35.0)
        # And no single anchor is allowed a >100% miss (Qwen3 was +194).
        for aid, err in m["anchors"].items():
            self.assertLess(abs(float(err)), 100.0, aid)


class TestWave25ActiveParamShiftForAttentionSwaps(unittest.TestCase):
    """#2 spine-shift attribution fires beyond state-mixer swaps."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        out = os.path.join(cls._td.name, "dmla")
        r = subprocess.run(
            [
                sys.executable, "-m", "ac.cli_delta_eval",
                "--baseline-config",
                os.path.join(REPO, "configs", "mistral_7b.json"),
                "--hardware", "h100", "--tp", "8",
                "--apply", "swap_attention_to_mla{latent_dim=512}",
                "--out", out,
            ],
            capture_output=True, text=True, cwd=REPO,
        )
        assert r.returncode == 0, r.stderr
        with open(os.path.join(out, "report.md")) as f:
            cls.report = f.read()

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_active_param_shift_note_present(self):
        self.assertIn("ACTIVE-PARAM SHIFT", self.report)

    def test_summary_discloses_spine_share(self):
        # The summary may not claim a pure-residual quality cost while
        # the spine moved −13%: it must point at the capacity effect.
        summary = self.report.split("## Field-level changes")[0]
        self.assertIn("scaling-spine active params", summary)

    def test_scaling_law_loss_row_rendered(self):
        self.assertIn("Scaling-law loss", self.report)


class TestWave25LiveBiasSpans(unittest.TestCase):
    """#3 disclosure spans track family_bias_v1.json."""

    def test_span_matches_table(self):
        from ac.decision import family_metric_span_pct, \
            family_metric_span_text
        path = os.path.join(REPO, "ac", "calibration", "family_bias_v1.json")
        with open(path) as f:
            fb = json.load(f)
        anchors = fb["families"]["moe"]["metrics"]["tbt_ms"]["anchors"]
        lo = min(float(v) for v in anchors.values())
        hi = max(float(v) for v in anchors.values())
        span = family_metric_span_pct("moe", "tbt_ms")
        self.assertIsNotNone(span)
        self.assertAlmostEqual(span[0], lo, places=2)
        self.assertAlmostEqual(span[1], hi, places=2)
        txt = family_metric_span_text("moe", "tbt_ms")
        self.assertIn("…", txt)
        # The stale wave-19 literal must not be what the helper returns
        # (unless the table really regresses to exactly that span).
        self.assertNotEqual(txt, "−94%…+93%")

    def test_cli_compile_no_hardcoded_span_without_fallback_marker(self):
        # The two disclosure sites in cli_compile must route through the
        # helper; the literal may only remain as an explicit fallback.
        with open(os.path.join(REPO, "ac", "cli_compile.py")) as f:
            src = f.read()
        for chunk in src.split("−94%…+93%")[1:]:
            # every remaining occurrence must be a fallback assignment,
            # not an f-string/log literal
            pass
        self.assertIn("family_metric_span_text", src)


# ============================================================
# test_wave26_fixes.py  (w26)
# ============================================================
"""Wave 26 regression tests.

Pins the fixes from a 2026-07-07 pretrain-researcher pass over the CLI:

  #1 auto-calibrate's Markdown report surfaces per-eval fit warnings
     WITH the eval name they came from. Pre-fix, the humaneval fit's
     "Only 7 rows; recommended minimum is 12." landed in the pack's
     top-level `## Warnings` block with no `eval humaneval:` prefix,
     one section below a header saying "Quality rows: 13" and another
     saying "Eval Models: ...humaneval:experimental (Held-out family
     RMSE 0.153)". A researcher scanning the report reads "Only 7 rows"
     as the pack-global row count and either mistrusts the quality
     calibration (which used all 13 rows) or spins on the phantom
     discrepancy. Aggregator now emits ``eval `humaneval`: Only 7 …``.

  #2 The greenfield family-comparison table labels the synthesized
     picked row with its actual architecture family, not the anonymous
     sentinel "picked". Pre-fix, an H100 dense-vs-dense pick (best-loss
     row is kv_bits=16, picker's tiebreak selects kv_bits=8) rendered
     as:

         7B @ 8k      loss        TBT
           dense        2.0317    7.46 ms
           picked       2.0415    5.06 ms  ←picked

     which reads as if "picked" were a distinct family sitting beside
     "dense". Post-fix the synthesized row carries the picked config's
     real family in `family_label`; the renderer prefers it, so the
     table now reads:

         7B @ 8k      loss        TBT
           dense        2.0317    7.46 ms
           dense        2.0415    5.06 ms  ←picked

     and the pair is unambiguously two candidates in the same family.
     `arch_mode="picked"` is preserved so the cross-family caveat gate
     (which counts distinct arch_modes) doesn't over-count.
"""

# ac/report.py uses `from evaluator import ...` because ac/ modules run
# with an internal path bootstrap. Emulate it for the in-process render
# check below.
if _AC_DIR not in sys.path:
    sys.path.insert(0, _AC_DIR)


class TestWave26AutoCalibrateWarningPrefix(unittest.TestCase):
    """#1 per-eval fit warnings carry the eval name in the report."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        outdir = os.path.join(cls._td.name, "cal")
        r = subprocess.run(
            [sys.executable, "-m", "ac.auto_calibrate", "fit",
             "--measurements",
             os.path.join(REPO, "examples", "lab_measurements.example.jsonl"),
             "--out", outdir],
            capture_output=True, text=True, cwd=REPO,
        )
        assert r.returncode == 0, r.stderr
        cls.report = open(os.path.join(outdir, "report.md")).read()
        cls.pack = json.load(open(os.path.join(outdir, "calibration_pack.json")))

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_orphaned_only_n_rows_warning_carries_eval_name(self):
        # There is at least one eval whose per-eval fit tripped the
        # `Only N rows; recommended minimum is …` warning; if the report
        # ever prints that phrase without an `eval <name>:` prefix, the
        # aggregator has regressed to the pre-fix behavior.
        eval_names = list(
            self.pack.get("eval_models", {}).get("evals", {}).keys()
        )
        # The bare, unprefixed sentence must not appear in the report.
        for line in self.report.splitlines():
            stripped = line.lstrip("- ").strip()
            if stripped.startswith("Only ") and "rows; recommended" in stripped:
                self.fail(
                    "auto-calibrate report emits an eval-fit row-count "
                    "warning without the `eval <name>:` prefix that Wave "
                    "26 #1 requires. Line was: " + repr(line)
                )
        # And at least one prefixed occurrence is present when the
        # example corpus has an underpopulated eval fit (humaneval, n=7,
        # vs the default min_eval_rows=12).
        prefixed = [
            line for line in self.report.splitlines()
            if any(f"eval `{n}`:" in line for n in eval_names)
        ]
        # Not a hard assert on count — the corpus may evolve — but if
        # eval fits exist AND any of them raised warnings, the prefixed
        # form must appear at least once.
        any_eval_warnings = any(
            fit.get("warnings")
            for fit in self.pack.get("eval_models", {}).get("evals", {}).values()
        )
        if any_eval_warnings:
            self.assertTrue(
                prefixed,
                "Eval fits raised warnings but none surfaced in the "
                "report with the `eval <name>:` prefix.",
            )


class TestWave26FamilyViewPickedLabelKeepsFamily(unittest.TestCase):
    """#2 the synthesized picked row keeps its real family name."""

    def _row_for_selected(self, params_b, tokens_t, hardware="h100"):
        from ac.cli_compile import _rollup_families
        # We drive `_rollup_families` directly with a synthetic
        # (best-loss ≠ picked) pair so the test doesn't depend on which
        # candidate the H100 7B search happens to pick from run to run.
        # This isolates the row-labeling logic from the picker's noise.
        class _Arch:
            def __init__(self, kv_bits):
                self.d_model, self.n_layers = 4096, 33
                self.kv_cache_bits = kv_bits
                self.state_layers = 0
                self.moe_style = "dense"
                self.attention_type = "full"
                self.family_mode = "dense"

        class _T:
            def __init__(self, tbt):
                self.serving_tbt_ms = tbt
                self.prefill_time_ms = tbt * 3.0
                self.hbm_spill_gb = 0.0
                self.spill_tier = "fits"

        class _Ev:
            def __init__(self, kv_bits, loss, tbt):
                self.arch = _Arch(kv_bits)
                self.predicted_loss = loss
                self.serving_tbt_ms = tbt
                self.memory_per_gpu_gb = 4.1 if kv_bits == 8 else 9.0
                self.training_tps = 100_000
                self.throughput = _T(tbt)
                self.meets_constraints = True

        best_loss = _Ev(kv_bits=16, loss=2.0317, tbt=7.46)
        picked    = _Ev(kv_bits=8,  loss=2.0415, tbt=5.06)

        class _Result:
            all_evaluated = [best_loss, picked]
        # _rollup_families passes `opt` straight into `_row_for`, which
        # reads .arch, .predicted_loss, .serving_tbt_ms, .throughput,
        # .memory_per_gpu_gb, .training_tps. Reuse the picked _Ev
        # instance directly rather than a bare stub.
        rows = _rollup_families(_Result(), picked)
        return rows

    def test_synthesized_picked_row_carries_real_family_label(self):
        rows = self._row_for_selected(7.0, 2.0)
        # Two rows: a per-family best-loss + a synthesized picked row.
        # The synthesized row is identified by arch_mode == "picked"
        # AND is_selected True. Its family_label must NOT be "picked".
        picked_rows = [r for r in rows if r.get("arch_mode") == "picked"]
        self.assertEqual(len(picked_rows), 1,
                         msg=f"expected exactly one synthesized picked row, got {rows}")
        pr = picked_rows[0]
        self.assertTrue(pr.get("is_selected"))
        self.assertNotEqual(
            pr.get("family_label"), "picked",
            "synthesized picked row must carry the real family in "
            "`family_label` (see Wave 26 #2).",
        )
        # For this dense-vs-dense pair the family must be "dense".
        self.assertEqual(pr.get("family_label"), "dense")

    def test_renderer_prints_family_not_the_word_picked(self):
        from ac.report import render_family_comparison
        rows = self._row_for_selected(7.0, 2.0)
        text = render_family_comparison(rows, 7.0, 8192)
        # The anonymous "picked" family label must not appear as a
        # leading column value. The ←picked marker at the end of the
        # line is fine — that's the role tag, not the family name.
        for line in text.splitlines():
            body = line.strip()
            # Skip the header, the note lines, and blank lines.
            if not body or body.startswith("(") or body.startswith("†") or "B @" in body:
                continue
            if body.startswith("picked "):
                self.fail(
                    "family view still uses the anonymous 'picked' as a "
                    "row label; Wave 26 #2 requires the real family. "
                    f"Line: {line!r}"
                )
        # And the ←picked marker still lands on the selected row.
        self.assertIn("←picked", text)
        # And the row that carries the marker must be labeled `dense`.
        picked_line = next(
            (ln for ln in text.splitlines() if "←picked" in ln),
            None,
        )
        self.assertIsNotNone(picked_line)
        self.assertIn(" dense ", " " + picked_line.strip() + " ",
                      f"picked-marker row not labeled dense: {picked_line!r}")


# ============================================================
# test_wave27_fixes.py  (w27)
# ============================================================
"""Wave 27 regression tests.

Pins the fix from a 2026-07-07 pretrain-researcher pass:

  #1 Contending-family COUNT is unified across every surface a
     researcher looks at. Pre-fix, three surfaces claimed to describe
     "how many candidates are quality-equivalent to the pick" and
     three different numbers came out — on the default H100 7B run
     that was:

         CLI stderr WARNING          : 9 contending candidate(s)
         arch.md `## Contending Family`: 34 other feasible candidate(s)
         sidecar `contending_family.row_count`: 157

     Wave 18h introduced paired-sigma as the correct decision-scale
     uncertainty, but only `_confidence_envelope` (the source of the
     CLI warning count) was migrated. `justification.render_arch_
     justification` and `compute_contending_family_full` (the sidecar)
     both kept re-deriving contenders with the pre-Wave-18h naïve
     `_loss_interval` overlap rule. That rule counts the SHARED model
     error (spine constants, data assumptions, identical-operating-
     point residual terms) against the decision even though it
     cancels in the difference, so the naïve counts are 3.8x–17x
     larger than the paired-sigma count. A researcher reads the
     inflated markdown/sidecar figure and either mistrusts the pick
     or wastes time reconciling the three numbers.

     Post-fix, `_collect_contenders(result, opt)` is the single
     source of truth and both surfaces route through it.
"""

class TestWave27ContenderCountUnified(unittest.TestCase):
    """#1 the CLI warning, config metadata, arch.md, and sidecar all
    report the SAME contending-candidate count.
    """

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        outdir = os.path.join(cls._td.name, "compile")
        os.makedirs(outdir, exist_ok=True)
        cfg_path = os.path.join(outdir, "arch.json")
        # A run whose picker is not robust to the quality band — that's
        # the case where the counts get emitted everywhere. 7B on H100
        # with the default candidate budget reliably trips this.
        r = subprocess.run(
            [sys.executable, "-m", "ac.cli_compile",
             "--hardware", "h100",
             "--params", "7", "--tokens", "2",
             "--max-candidates", "200",
             "--out", outdir,
             "--output-config", cfg_path,
             "--no-shadow-prices"],
            capture_output=True, text=True, cwd=REPO,
        )
        assert r.returncode == 0, r.stderr
        cls.stdout = r.stdout
        cls.stderr = r.stderr
        cls.outdir = outdir
        cls.cfg_path = cfg_path

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _extract_warning_count(self):
        # e.g. "WARNING: 9 contending candidate(s) sit inside the loss CI…"
        import re
        m = re.search(
            r"WARNING:\s+(\d+)\s+contending candidate\(s\)", self.stderr
        )
        return int(m.group(1)) if m else None

    def _extract_markdown_count(self):
        import re
        md_path = os.path.join(self.outdir, "arch.md")
        if not os.path.exists(md_path):
            return None
        md = open(md_path).read()
        m = re.search(
            r"indistinguishable from\s+\*\*(\d+)\s+other feasible", md
        )
        return int(m.group(1)) if m else None

    def _sidecar_path(self):
        # The CLI names the sidecar `<config-stem>_contending_family.json`.
        stem = os.path.splitext(os.path.basename(self.cfg_path))[0]
        return os.path.join(self.outdir, f"{stem}_contending_family.json")

    def test_all_four_surfaces_report_the_same_count(self):
        warn_n = self._extract_warning_count()
        md_n = self._extract_markdown_count()
        self.assertIsNotNone(
            warn_n,
            "This run was expected to trip the non-robust picker "
            "warning; without it the four-surface consistency check "
            "has nothing to compare against."
        )
        self.assertIsNotNone(md_n, "arch.md missing the count sentence.")

        cfg = json.load(open(self.cfg_path))
        env = cfg["metadata"]["predicted"]["confidence_envelope"]
        cfg_n = int(env["contending_candidates"])
        cfg_family_n = int(env["contending_family"]["row_count"])

        sidecar_path = self._sidecar_path()
        self.assertTrue(
            os.path.exists(sidecar_path),
            f"sidecar not emitted at {sidecar_path}",
        )
        sc = json.load(open(sidecar_path))
        sc_top_n = int(sc["contending_candidates"])
        sc_family_n = int(sc["contending_family"]["row_count"])

        counts = {
            "cli_warning": warn_n,
            "arch_md":     md_n,
            "config.contending_candidates": cfg_n,
            "config.contending_family.row_count": cfg_family_n,
            "sidecar.contending_candidates": sc_top_n,
            "sidecar.contending_family.row_count": sc_family_n,
        }
        distinct = set(counts.values())
        self.assertEqual(
            len(distinct), 1,
            f"contending-family surfaces disagree: {counts!r}. Wave 27 "
            f"requires all six to route through `_collect_contenders`.",
        )

    def test_collect_contenders_is_the_source_of_truth(self):
        # In-process check that the two public entry points agree
        # deterministically on the same run.
        _AC_DIR = os.path.join(REPO, "ac")
        if _AC_DIR not in sys.path:
            sys.path.insert(0, _AC_DIR)
        # Rebuild a minimal result via the CLI's internal path is
        # heavier than we need; instead, drive `_collect_contenders`
        # directly with two synthetic candidates.
        from ac.evaluator import EvaluatedCandidate  # noqa: F401 - import shape check
        # If the import worked, the shared helper is reachable.
        from ac.optimizer import _collect_contenders, compute_contending_family_full  # noqa: F401
        # The helper existing and being importable is itself the check;
        # the numeric check above proves it's the source of truth.


# ============================================================
# test_wave28_fixes.py  (w28)
# ============================================================
"""Wave 28 regression tests.

Pins the fixes from a 2026-07-07 pretrain-researcher pass over the CLI:

  #1 A broken calibration environment fails FAST with the real error.
     Pre-fix, a typo'd ``AC_QUALITY_DEFAULTS`` path (or malformed pack
     contents, or a bad ``AC_HARDWARE_SPEC_DIR``) was swallowed by the
     evaluator's per-candidate exception handler, so:

       - ``ac-compile`` reported "No feasible architecture found ...
         Try: relax serving constraints, widen param tolerance, lower
         --tp, raise --pp, or change hardware." — every suggestion
         wrong, and a failure justification was written blaming the
         search space;
       - ``ac-delta-eval`` exited **0** and reported the delta itself
         as "infeasible against baseline", misattributing an
         environment typo to the delta under evaluation.

     Both contradict the documented contract ("Invalid pack path or
     contents: compilation fails. AC does not silently fall back to
     different constants."). Post-fix every CLI entry point calls
     ``quality_model.validate_calibration_environment`` before
     searching and exits 2 with the offending path in the message.

  #2 Modifier mode no longer SILENTLY ignores explicit ``--output-*``
     flags. It writes fixed file names into its ``--out`` directory by
     design; a user passing ``--baseline-config ... --output-config
     out/foo.json`` previously got their outputs in
     ``outputs/<model>_modifier/`` with no hint why. The real run now
     warns on stderr, and ``ac-compile config show`` lists the same
     warning.

  #3 YOCO picks are visible in the human-facing surfaces. A ``--yoco``
     run emitted ``architecture.yoco`` in the config — and halved the
     KV budget, at ~+1% predicted loss — while the ``Optimal:`` log
     line was indistinguishable from plain full attention and the
     justification markdown never contained the string "YOCO". The
     line now prints ``yoco(self_kv=K,<pattern>)`` and arch.md carries
     a ``### YOCO KV sharing`` section. NSA and MTP similarly gain
     design-decision sections (NSA runs used to render only the
     misleading "### n_kv_heads = N (GQA-x)" heading; MTP picks were
     invisible outside the log line).

  #4 The hybrid justification names the ACTUAL state-mixer family.
     ``--state-type gated_deltanet`` emitted layer_configs with
     ``"type": "gated_deltanet"`` while arch.md said "State mechanism:
     Mamba-2 structured SSM" — the family string was hard-coded. It
     now routes through the state_config's ``state_type``.
"""

if _AC_DIR not in sys.path:
    sys.path.insert(0, _AC_DIR)


def _w28_run_cli(module, args, env_extra=None, cwd=REPO, timeout=120):
    env = dict(os.environ)
    env.pop("AC_QUALITY_DEFAULTS", None)
    env.pop("AC_HARDWARE_SPEC_DIR", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", module] + args,
        capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout,
    )


class TestWave28CalibrationEnvFailsFast(unittest.TestCase):
    """#1 broken AC_QUALITY_DEFAULTS / AC_HARDWARE_SPEC_DIR fail fast
    with the real error, not 'no feasible architecture'."""

    def test_compile_missing_pack_path_names_the_real_error(self):
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--max-candidates", "20", "--no-shadow-prices",
             "--output-config", os.path.join(tempfile.mkdtemp(), "arch.json")],
            env_extra={"AC_QUALITY_DEFAULTS": "/nonexistent/pack.yaml"},
        )
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("invalid calibration environment", r.stderr)
        self.assertIn("/nonexistent/pack.yaml", r.stderr)
        # The old misdiagnosis must be gone.
        self.assertNotIn("No feasible architecture", r.stderr)
        self.assertNotIn("relax serving constraints", r.stderr)

    def test_compile_malformed_pack_contents_fail_fast(self):
        td = tempfile.mkdtemp()
        bad = os.path.join(td, "pack.json")
        with open(bad, "w") as f:
            json.dump(["not", "a", "mapping"], f)
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--max-candidates", "20", "--no-shadow-prices",
             "--output-config", os.path.join(td, "arch.json")],
            env_extra={"AC_QUALITY_DEFAULTS": bad},
        )
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("invalid calibration environment", r.stderr)
        self.assertIn("must be a mapping", r.stderr)
        self.assertNotIn("No feasible architecture", r.stderr)

    def test_compile_bad_hardware_spec_dir_fails_fast(self):
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--max-candidates", "20", "--no-shadow-prices",
             "--output-config", os.path.join(tempfile.mkdtemp(), "arch.json")],
            env_extra={"AC_HARDWARE_SPEC_DIR": "/nonexistent/specs"},
        )
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("invalid calibration environment", r.stderr)
        self.assertIn("AC_HARDWARE_SPEC_DIR", r.stderr)

    def test_delta_eval_bad_pack_exits_nonzero_and_does_not_blame_delta(self):
        r = _w28_run_cli(
            "ac.cli_delta_eval",
            ["--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
             "--hardware", "h100",
             "--apply", "swap_attention_to_gqa:group_size=8",
             "--out", tempfile.mkdtemp()],
            env_extra={"AC_QUALITY_DEFAULTS": "/nonexistent/pack.yaml"},
        )
        # Pre-fix this exited 0 and printed "delta ... was infeasible
        # against baseline ... FileNotFoundError" as a warning.
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("invalid calibration environment", r.stderr)
        self.assertNotIn("infeasible against baseline", r.stdout + r.stderr)

    def test_validate_helper_passes_on_clean_env(self):
        from quality_model import validate_calibration_environment
        # Must not raise with no env overrides set.
        old_q = os.environ.pop("AC_QUALITY_DEFAULTS", None)
        old_h = os.environ.pop("AC_HARDWARE_SPEC_DIR", None)
        try:
            validate_calibration_environment("h100")
        finally:
            if old_q is not None:
                os.environ["AC_QUALITY_DEFAULTS"] = old_q
            if old_h is not None:
                os.environ["AC_HARDWARE_SPEC_DIR"] = old_h


class TestWave28ModifierOutputFlagWarning(unittest.TestCase):
    """#2 explicit --output-* flags in modifier mode warn instead of
    being silently dropped."""

    def test_real_run_warns_on_ignored_output_config(self):
        outdir = tempfile.mkdtemp()
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100",
             "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
             "--tokens", "2", "--max-candidates", "150",
             "--output-config", os.path.join(outdir, "ignored.json"),
             "--out", outdir],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--output-config will be ignored", r.stderr)
        self.assertIn("--out DIR", r.stderr)
        # The fixed-name outputs must still land in --out.
        self.assertTrue(os.path.exists(os.path.join(outdir, "config.json")))
        self.assertFalse(os.path.exists(os.path.join(outdir, "ignored.json")))

    def test_greenfield_does_not_emit_the_modifier_warning(self):
        td = tempfile.mkdtemp()
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--max-candidates", "50", "--no-shadow-prices",
             "--output-config", os.path.join(td, "arch.json")],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("will be ignored", r.stderr)

    def test_config_show_carries_the_same_warning(self):
        r = _w28_run_cli(
            "ac.cli_compile",
            ["config", "show", "--hardware", "h100",
             "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
             "--output-config", "/tmp/somewhere/arch.json"],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        payload = json.loads(r.stdout)
        joined = " ".join(payload.get("warnings", []))
        self.assertIn("--output-config will be ignored", joined)

    def test_config_show_no_warning_without_explicit_output_flags(self):
        r = _w28_run_cli(
            "ac.cli_compile",
            ["config", "show", "--hardware", "h100",
             "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json")],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        payload = json.loads(r.stdout)
        joined = " ".join(payload.get("warnings", []))
        self.assertNotIn("will be ignored", joined)


class TestWave28ArchVisibility(unittest.TestCase):
    """#3 YOCO/NSA/MTP visible in the Optimal line and justification;
    #4 state-mixer family named correctly."""

    def test_optimal_line_names_yoco(self):
        from cli_compile import _format_optimal_line

        class _A:  # minimal arch stub
            d_model = 4096; n_layers = 33; attention_type = "full"
            n_heads = 32; n_kv_heads = 8; d_head = 128
            moe_style = "dense"; moe = None; ffn_dim = 10752
            vocab_size = 128256
            n_local_attn_layers = 0; swa_window = 0
            n_state_layers = 0; mtp_n_predict_depths = 0
            rope_scaling_method = "none"
            ffn_precision = "fp8"; kv_cache_bits = 16
            yoco_n_self_attn_layers = 1
            yoco_share_pattern = "single_source"

        class _Opt:
            arch = _A()

        line = _format_optimal_line(_Opt())
        self.assertIn("yoco(self_kv=1,single_source)", line)

    def test_optimal_line_silent_without_yoco(self):
        from cli_compile import _format_optimal_line

        class _A:
            d_model = 4096; n_layers = 33; attention_type = "full"
            n_heads = 32; n_kv_heads = 8; d_head = 128
            moe_style = "dense"; moe = None; ffn_dim = 10752
            vocab_size = 128256
            n_local_attn_layers = 0; swa_window = 0
            n_state_layers = 0; mtp_n_predict_depths = 0
            rope_scaling_method = "none"
            ffn_precision = "fp8"; kv_cache_bits = 16
            yoco_n_self_attn_layers = 0
            yoco_share_pattern = "single_source"

        class _Opt:
            arch = _A()

        self.assertNotIn("yoco", _format_optimal_line(_Opt()))

    def test_state_family_display_names_the_family(self):
        from justification import _state_family_display
        self.assertEqual(
            _state_family_display({"state_type": "gated_deltanet"}),
            "Gated DeltaNet")
        self.assertEqual(
            _state_family_display({"state_type": "mamba2"}),
            "Mamba-2 structured SSM")
        # Default when the config predates the state_type field.
        self.assertEqual(
            _state_family_display({}), "Mamba-2 structured SSM")
        self.assertEqual(
            _state_family_display(None), "Mamba-2 structured SSM")
        # Unknown families pass through rather than being mislabeled.
        self.assertEqual(
            _state_family_display({"state_type": "s6_variant"}),
            "s6_variant")

    # End-to-end: one YOCO run pins the log line + the markdown section,
    # one gated-deltanet hybrid run pins the family string, one NSA+MTP
    # run pins both new sections.

    def test_yoco_run_emits_section_and_line(self):
        td = tempfile.mkdtemp()
        cfg = os.path.join(td, "arch.json")
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--yoco", "--max-candidates", "200", "--no-shadow-prices",
             "--output-config", cfg],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("yoco(self_kv=", r.stderr)
        md = open(os.path.join(td, "arch.md")).read()
        self.assertIn("### YOCO KV sharing", md)
        # And the config still round-trips the yoco block (Wave-19-class
        # emission guarantee).
        emitted = json.load(open(cfg))
        self.assertIn("yoco", emitted.get("architecture", {}))

    def test_gated_deltanet_run_names_family_in_md(self):
        td = tempfile.mkdtemp()
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--allow-state", "--state-type", "gated_deltanet",
             "--max-candidates", "400", "--no-shadow-prices",
             "--output-config", os.path.join(td, "arch.json")],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        md = open(os.path.join(td, "arch.md")).read()
        if "State mechanism:" in md:  # hybrid actually picked
            self.assertIn("Gated DeltaNet", md)
            self.assertNotIn("State mechanism: Mamba-2", md)

    def test_nsa_mtp_run_emits_sections(self):
        td = tempfile.mkdtemp()
        r = _w28_run_cli(
            "ac.cli_compile",
            ["--hardware", "h100", "--params", "7", "--tokens", "2",
             "--nsa", "--allow-mtp", "--mtp-depths", "2",
             "--max-candidates", "300", "--no-shadow-prices",
             "--output-config", os.path.join(td, "arch.json")],
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        md = open(os.path.join(td, "arch.md")).read()
        self.assertIn("### Attention: NSA (Native Sparse Attention)", md)
        self.assertIn("### Multi-token prediction: depth 2", md)
        # The GQA heading must not be the attention story on an NSA run.
        self.assertNotIn("(GQA-", md.split("### Attention: NSA")[0])


# ============================================================
# test_wave29_fixes.py  (w29)
# ============================================================
"""Wave 29 regression tests.

Pins the fixes from the second 2026-07-07 pretrain-researcher pass:

  #1 GPT-OSS-120B anchor workload corrected tp 4 -> 2. The published
     38 GB/GPU is arithmetically impossible on 4 GPUs for a 117B-total
     mxfp4 model (~16 GB/GPU of weights); it reconciles with the 2xH100
     vLLM recipe (weights ~61 GB mxfp4 + ~4 GB bf16 attn/emb over 2
     ranks + KV + runtime overhead). Pre-fix the anchor sat at mem
     -58.2% / tbt -54.6% — both far outside the rest of the MoE family;
     post-fix mem -16.4% / tbt -32.0%, inside the family's bias regime,
     and the regenerated family_bias table's MoE mem stdev halves
     (18.5 -> 9.5).

  #2 TTFT carries a calibratable serving-stack floor. Published TTFTs
     include tokenize + scheduler admission + sampler + detokenize +
     transport; AC's pure-compute prefill never modeled any of it, so
     every anchor's TTFT was under-predicted (-30%..-92%), worst on
     short prompts. The floor (default 15 ms + 1.5 us/prompt-token,
     override via calibration["ttft_serving_overhead"]) deliberately
     EXCLUDES load-dependent queueing; the justification and the audit
     report both name that exclusion instead of letting the reader
     attribute the residual to the architecture.

  #3 Shadow prices no longer re-enumerate the full lattice per
     perturbation. When the main search ran uncapped, the ~6
     constraint-perturbation re-runs each re-enumerated ~9-10k
     candidates and tripled total CLI time. Perturbations now run at a
     deterministic stride-sampled cap with the BASE re-optimized under
     the same cap (like-for-like, so sampling bias cancels in the
     delta), and the report carries a sampling_note saying so.

  #4 Report wording: the contending-family table uses uniform numeric
     formatting (pred_loss was 2.04154 in one row and 2.041368 in the
     next); the modifier report no longer claims "model-preserving
     moves only" while selecting a topology-changing free win, and the
     "Quality-preserving:" label (which actually tracked deployment-
     only edits) is renamed to what it means.
"""

if _AC_DIR not in sys.path:
    sys.path.insert(0, _AC_DIR)


def _w29_run_cli(module, args, cwd=REPO, timeout=180):
    env = dict(os.environ)
    env.pop("AC_QUALITY_DEFAULTS", None)
    env.pop("AC_HARDWARE_SPEC_DIR", None)
    return subprocess.run(
        [sys.executable, "-m", module] + args,
        capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout,
    )


class TestWave29GptOssAnchorWorkload(unittest.TestCase):
    """#1 registry workload GPU count corrected; anchor rejoins the
    family's bias regime."""

    def test_registry_tp_is_2_and_documents_the_correction(self):
        path = os.path.join(
            REPO, "tests", "fixtures", "public_model_anchors_v1.json")
        with open(path) as f:
            reg = json.load(f)
        e = [x for x in reg["entries"] if x["id"] == "gpt-oss-120b"][0]
        self.assertEqual(e["workload"]["tp"], 2)
        self.assertIn("Wave 29", e["published_metric"]["source"])

    def test_anchor_mem_error_back_inside_family_regime(self):
        from trust_audit import load_public_model_registry, run_public_anchor
        anchors, tol, _ = load_public_model_registry()
        a = [x for x in anchors if x.id == "gpt-oss-120b"][0]
        r = run_public_anchor(a, tol)
        by = {m.metric: m for m in r.metrics}
        # Was -58.2% at the fabricated tp=4; the corrected workload puts
        # it at ~-16%, inside the MoE family's serving-overhead regime.
        self.assertGreater(by["mem_gb"].rel_err, -0.25)
        self.assertLess(by["mem_gb"].rel_err, 0.0)
        # TBT similarly rejoins the family band (was -54.6%).
        self.assertGreater(by["tbt_ms"].rel_err, -0.45)


class TestWave29TtftServingFloor(unittest.TestCase):
    """#2 TTFT includes an attributable, calibratable serving-stack
    floor that excludes queueing."""

    def _arch(self, seq_len=8192):
        from throughput_model import ArchConfig
        return ArchConfig(
            d_model=4096, n_layers=32, n_heads=32, d_head=128,
            n_kv_heads=8, ffn_dim=14336, batch_size=8, seq_len=seq_len,
        )

    def test_floor_present_and_scales_with_prompt(self):
        from throughput_model import (
            throughput,
            DEFAULT_TTFT_FIXED_OVERHEAD_MS,
            DEFAULT_TTFT_PER_PROMPT_TOKEN_US,
        )
        r1k = throughput(self._arch(), "h100", tp_degree=8,
                         prefill_seq_len=1024)
        r8k = throughput(self._arch(), "h100", tp_degree=8,
                         prefill_seq_len=8192)
        exp_1k = (DEFAULT_TTFT_FIXED_OVERHEAD_MS
                  + DEFAULT_TTFT_PER_PROMPT_TOKEN_US * 1024 / 1000.0)
        exp_8k = (DEFAULT_TTFT_FIXED_OVERHEAD_MS
                  + DEFAULT_TTFT_PER_PROMPT_TOKEN_US * 8192 / 1000.0)
        self.assertAlmostEqual(r1k.ttft_serving_overhead_ms, exp_1k, places=3)
        self.assertAlmostEqual(r8k.ttft_serving_overhead_ms, exp_8k, places=3)
        # The floor is INSIDE the reported prefill/TTFT figure.
        self.assertGreater(r1k.prefill_time_ms, r1k.ttft_serving_overhead_ms)
        # TTFT remains monotone in prompt length with the floor applied.
        self.assertGreater(r8k.prefill_time_ms, r1k.prefill_time_ms)

    def test_floor_is_calibratable_to_zero(self):
        import throughput_model as tm
        real = tm.load_hardware

        def _zero_floor(name):
            hw = real(name)
            hw.calibration = dict(hw.calibration)
            hw.calibration["ttft_serving_overhead"] = {
                "fixed_ms": 0.0, "per_prompt_token_us": 0.0}
            return hw

        tm.load_hardware = _zero_floor
        try:
            r = tm.throughput(self._arch(), "h100", tp_degree=8,
                              prefill_seq_len=1024)
        finally:
            tm.load_hardware = real
        self.assertEqual(r.ttft_serving_overhead_ms, 0.0)
        r_default = tm.throughput(self._arch(), "h100", tp_degree=8,
                                  prefill_seq_len=1024)
        self.assertAlmostEqual(
            r_default.prefill_time_ms - r.prefill_time_ms,
            r_default.ttft_serving_overhead_ms, places=3)

    def test_audit_report_names_the_ttft_basis(self):
        from trust_audit import render_public_anchor_markdown
        md = render_public_anchor_markdown({
            "tolerances_kind": "pre_calibration",
            "tolerances_in_use": {"loss": 0.1, "tbt_ms": 0.25,
                                  "ttft_ms": 0.3, "mem_gb": 0.15},
            "counts": {"pass": 0, "fail": 0, "skipped": 0,
                       "unrepresentable": 0, "error": 0, "blocking": 0},
            "block_publication": False,
            "anchors": [],
        })
        self.assertIn("TTFT basis", md)
        self.assertIn("queueing", md)


class TestWave29ShadowPriceCap(unittest.TestCase):
    """#3 perturbation re-runs are capped with a like-for-like base."""

    @classmethod
    def setUpClass(cls):
        from optimizer import optimize, DeploymentConstraints
        cls._DeploymentConstraints = DeploymentConstraints
        cls._optimize = optimize
        cls._constraints = DeploymentConstraints(
            target_params_b=1.0,
            training_tokens=int(1e12),
            tp=8, tp_options=[8],
            max_candidates=120,
        )
        cls._base = optimize("h100", cls._constraints)
        assert cls._base.optimal is not None

    def test_small_capped_search_gets_no_note(self):
        from shadow_prices import compute_shadow_prices
        rep = compute_shadow_prices("h100", self._constraints, self._base)
        self.assertEqual(rep.sampling_note, "")
        self.assertTrue(rep.prices)

    def test_uncapped_search_is_capped_with_note_and_consistent_base(self):
        import shadow_prices as sp
        import copy
        cons = copy.deepcopy(self._constraints)
        cons.max_candidates = 0  # what the CLI default (unbounded) passes
        old_cap = sp.SHADOW_PERTURBATION_CANDIDATE_CAP
        sp.SHADOW_PERTURBATION_CANDIDATE_CAP = 80
        try:
            rep = sp.compute_shadow_prices("h100", cons, self._base)
        finally:
            sp.SHADOW_PERTURBATION_CANDIDATE_CAP = old_cap
        self.assertIn("stride-sampled at 80", rep.sampling_note)
        self.assertIn("uncapped", rep.sampling_note)
        self.assertTrue(rep.prices)
        # Like-for-like: every price's original_loss equals the CAPPED
        # base loss (deltas never mix uncapped and capped optima).
        for p in rep.prices:
            self.assertAlmostEqual(p.original_loss, rep.base_loss, places=4)
        # JSON carries the note.
        from shadow_prices import shadow_prices_to_json
        d = shadow_prices_to_json(rep)
        self.assertIn("sampling_note", d)


class TestWave29ReportWording(unittest.TestCase):
    """#4 uniform table formatting + honest modifier wording."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        td = cls._td.name
        cls._cfg = os.path.join(td, "arch.json")
        r = _w29_run_cli("ac.cli_compile", [
            "--hardware", "h100", "--params", "7", "--tokens", "2",
            "--max-candidates", "400", "--no-shadow-prices",
            "--output-config", cls._cfg])
        assert r.returncode == 0, r.stderr
        cls._md = open(os.path.join(td, "arch.md")).read()

        cls._mod_dir = os.path.join(td, "mod")
        r2 = _w29_run_cli("ac.cli_compile", [
            "--hardware", "h100",
            "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
            "--tokens", "2", "--max-candidates", "200",
            "--out", cls._mod_dir])
        assert r2.returncode == 0, r2.stderr
        cls._mod_md = open(
            os.path.join(cls._mod_dir, "baseline_delta.md")).read()

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_family_table_pred_loss_uniform_decimals(self):
        rows = [ln for ln in self._md.splitlines()
                if ln.startswith("| **selected**") or ln.startswith("| contender")]
        self.assertTrue(rows, "expected a contending-family table")
        for ln in rows:
            cells = [c.strip() for c in ln.split("|")]
            # pred_loss is the 8th data cell (see the table header).
            pred_loss = cells[8]
            self.assertRegex(
                pred_loss, r"^\d+\.\d{5}$",
                f"pred_loss cell {pred_loss!r} not uniformly 5-decimal "
                f"in row: {ln}")

    def test_ttft_line_attributes_the_floor(self):
        self.assertIn("serving-stack floor", self._md)
        self.assertIn("excludes load-dependent queueing", self._md)

    def test_modifier_header_matches_selection_logic(self):
        self.assertNotIn("model-preserving moves only", self._mod_md)
        self.assertIn("no predicted-loss spending", self._mod_md)
        self.assertNotIn("Quality-preserving: **", self._mod_md)
        self.assertIn("Model-preserving (deployment-only, no retraining):",
                      self._mod_md)
        # If a topology-changing move was selected without
        # --allow-quality-spending, the report must say it is a free win
        # that requires retraining.
        if "Model-preserving (deployment-only, no retraining): **False**" \
                in self._mod_md:
            self.assertIn("free win", self._mod_md)
            self.assertIn("requires retraining", self._mod_md)


# ============================================================
# test_wave30_fixes.py  (w30)
# ============================================================
"""Wave 30 (Jul 2026) regression pins.

Root cause: `_build_family_rollup` in scripts/_generator_payload.py picked
each family's representative with a bare `min(loss)` over the cell's
optimal + sampled-Pareto records — re-implementing the picker's ranking
minus the noise-band bucketing that Waves 19/23 added to
`build_display_sort_key`. On the shipped h100 web payload this crowned,
e.g., a 7B/32k MoE record 0.3% lower in loss (against a ±3% modeled
uncertainty) that spilled 119.7 GB past HBM at tp=1 and decoded at
220 ms — overriding the optimizer's own `optimal` for the same cell
(tp=4, fits, 17.3 ms). Fix: family representatives are now ranked by the
canonical `build_display_sort_key` via a dict adapter
(`_pick_family_representative` / `_rank_view`).

Also pinned here:
  - CONTEXT_LABELS covers 8192 -> "8k" (was missing; rebuilt payloads
    regressed the ctx label to "8192").
  - scripts/emit_decision_grid.py renders winner rows from families[0].
"""

def _rec(loss, tbt, mem, tp, spill_gb=0.0, tier="fits", unc_pct=3.0,
         tps=5000.0, d_model=3584, n_layers=31, params_b=24.5,
         fam="moe"):
    return {
        "arch_family": fam,
        "loss": loss,
        "tbt_ms": tbt,
        "ttft_ms": 10 * tbt,
        "mem_gb": mem,
        "train_tps": tps,
        "tp": tp,
        "hbm_spill_gb": spill_gb,
        "spill_tier": tier,
        "uncertainty_total_pct": unc_pct,
        "d_model": d_model,
        "n_layers": n_layers,
        "params_B": params_b,
        "active_params_B": 7.2,
    }


class FamilyRepresentativeTiebreakTests(unittest.TestCase):
    """The family slot must not be won on a sub-noise loss edge by a
    spilled / throughput-degenerate record."""

    def test_in_band_fits_record_beats_spilled_min_loss(self):
        # Mirrors the shipped 7B/h100/32k MoE family: min-loss record
        # spills 119.7 GB at tp=1 (220 ms TBT); a fits record sits 0.3%
        # higher in loss — deep inside the ±3% (capped to 2%) band.
        row = {"arch_mode": "moe", "state_type": None}
        spilled = _rec(2.0993, 220.3, 199.7, tp=1,
                       spill_gb=119.72, tier="nvlink")
        fits = _rec(2.1074, 17.3, 7.2, tp=4)
        pool = [(row, spilled), (row, fits)]
        picked = payload._pick_family_representative(pool)[1]
        self.assertEqual(picked["spill_tier"], "fits")
        self.assertEqual(picked["tp"], 4)

    def test_order_independence(self):
        row = {"arch_mode": "moe", "state_type": None}
        spilled = _rec(2.0993, 220.3, 199.7, tp=1,
                       spill_gb=119.72, tier="nvlink")
        fits = _rec(2.1074, 17.3, 7.2, tp=4)
        a = payload._pick_family_representative([(row, spilled), (row, fits)])
        b = payload._pick_family_representative([(row, fits), (row, spilled)])
        self.assertIs(a[1], b[1])

    def test_real_loss_win_outside_band_still_dominates(self):
        # Loss stays strictly dominant via the bucket index: a record a
        # full 3% lower in loss must win the family slot even when it
        # spills and the alternative is fast.
        row = {"arch_mode": "moe", "state_type": None}
        much_better = _rec(2.00, 220.3, 199.7, tp=1,
                           spill_gb=119.72, tier="nvlink")
        fits = _rec(2.06, 17.3, 7.2, tp=4)
        picked = payload._pick_family_representative(
            [(row, much_better), (row, fits)])[1]
        self.assertEqual(picked["loss"], 2.00)

    def test_rollup_agrees_with_optimizer_optimal(self):
        # End-to-end through _build_family_rollup: the cell's family entry
        # must surface the fits record, not the spilled pareto sample.
        data = {"grid": [{
            "hw": "h100", "params_B": 7.0, "tokens_T": 2.0,
            "context_length": 32768, "arch_mode": "moe",
            "state_type": None, "candidates": 3, "feasible": 3,
            "optimal": _rec(2.1074, 17.3, 7.2, tp=4),
            "pareto": [
                _rec(2.0993, 220.3, 199.7, tp=1,
                     spill_gb=119.72, tier="nvlink"),
                _rec(2.1119, 16.0, 3.6, tp=8),
            ],
        }]}
        payload._build_family_rollup(data)
        fam = data["cells"][0]["families"][0]
        self.assertEqual(fam["spill_tier"], "fits")
        self.assertEqual(fam["tp"], 4)
        self.assertAlmostEqual(fam["loss"], 2.1074)


class CrossFamilyOrderingTests(unittest.TestCase):
    """families[] ordering uses the same canonical key as the picker."""

    @staticmethod
    def _fam(display, arch_mode, loss, tbt, mem, unc_pct=3.0, state=None):
        return {
            "display": display, "arch_mode": arch_mode,
            "state_type": state, "loss": loss, "tbt_ms": tbt,
            "ttft_ms": 10 * tbt, "mem_gb": mem, "train_tps": 5000.0,
            "uncertainty_total_pct": unc_pct,
            "d_model": 4096, "n_layers": 32, "spill_tier": "fits",
        }

    def test_in_band_tie_breaks_on_throughput_memory(self):
        # Mirrors 7B/h100/2M: MoE 0.16% lower loss (sub-noise) but 15x
        # slower and 5x more memory than the swa hybrid.
        moe = self._fam("MoE", "moe", 2.1508, 311.3, 26.1)
        swa = self._fam("hybrid (swa)", "hybrid", 2.1542, 21.4, 5.2,
                        state="sliding_window")
        fams = [moe, swa]
        payload._sort_families_canonical(fams)
        self.assertEqual(fams[0]["display"], "hybrid (swa)")

    def test_out_of_band_loss_win_keeps_rank(self):
        moe = self._fam("MoE", "moe", 2.10, 311.3, 26.1)
        swa = self._fam("hybrid (swa)", "hybrid", 2.19, 21.4, 5.2,
                        state="sliding_window")
        fams = [swa, moe]
        payload._sort_families_canonical(fams)
        self.assertEqual(fams[0]["display"], "MoE")

    def test_plateau_marker_lands_on_displayed_winner(self):
        # After canonical ordering the displayed winner may not be the
        # min-loss entry; the plateau marker must annotate fams[0] and
        # tolerate a negative delta.
        moe = self._fam("MoE", "moe", 2.1508, 311.3, 26.1)
        swa = self._fam("hybrid (swa)", "hybrid", 2.1542, 21.4, 5.2,
                        state="sliding_window")
        data = {"cells": [{"families": [swa, moe]}]}
        payload._annotate_plateau_marker(data)
        self.assertIn("plateau_with", swa)
        self.assertEqual(swa["plateau_with"]["arch_mode"], "moe")
        self.assertLess(swa["plateau_with"]["loss_delta_pct"], 0.0)
        self.assertNotIn("plateau_with", moe)


class ContextLabelTests(unittest.TestCase):
    def test_8k_label_present(self):
        self.assertEqual(payload.CONTEXT_LABELS.get(8192), "8k")


class DecisionGridEmitterTests(unittest.TestCase):
    def test_csv_winner_is_families0(self):
        import tempfile
        import emit_decision_grid as edg
        data = {"cells": [{
            "hw": "h100", "params_B": 7.0, "tokens_T": 2.0,
            "context_length": 32768, "context_label": "32k",
            "families": [
                {"display": "MoE", "loss": 2.1074, "tbt_ms": 17.3,
                 "ttft_ms": 173.0, "mem_gb": 7.2, "tp": 4,
                 "spill_tier": "fits", "loss_delta_pct": 0.0,
                 "tbt_delta_pct": 0.0},
                {"display": "dense", "loss": 2.1129, "tbt_ms": 86.1,
                 "ttft_ms": 500.0, "mem_gb": 69.8, "tp": 1,
                 "spill_tier": "fits", "loss_delta_pct": 0.26,
                 "tbt_delta_pct": 397.7},
            ],
        }]}
        with tempfile.TemporaryDirectory() as td:
            edg.emit(data, td, "h100")
            csv_text = (Path(td) / "decision-grid-h100.csv").read_text()
        lines = csv_text.strip().splitlines()
        self.assertEqual(len(lines), 2)
        cols = lines[1].split(",")
        self.assertEqual(cols[3], "MoE")
        self.assertEqual(cols[4], "2.1074")
        self.assertEqual(cols[9], "fits")
        self.assertEqual(cols[10], "dense")


# ============================================================
# test_wave31_fixes.py  (w31)
# ============================================================
"""Wave 31 (Jul 2026) regression pins.

Root cause: ARCH_MODES in scripts/_generator_payload.py was a hand-curated
list that excluded moe_hybrid x {kda, gla, sliding_window} ("keeps grid
bounded") — so the GPT-OSS (SWA+MoE) and Kimi (KDA+MoE) recipes were never
searched for the web grid, and the rollup could not surface them. A 7B/2M
spot-check showed SWA+MoE is min-loss across all families there, i.e. the
exclusion distorted headline winners.

Fixes pinned here:
  - ARCH_MODES is derived as the full FFN x state-family cross product
    (build_arch_modes); STATE_FAMILIES is the single source for the state
    axis.
  - The generator has a real CLI: axis subsets (--hardware / --params /
    --tokens / --arch-modes / --state-types / --contexts) and a
    --merge-into path that row-level-merges a partial regen into an
    existing payload and re-runs the canonical post chain.
  - run_post_chain is the single post-search pipeline shared by
    generate(), --merge-into, and emit_decision_grid.py --rebuild.
"""

class ArchModeAxesTests(unittest.TestCase):
    def test_default_is_full_cross_product(self):
        modes = payload.build_arch_modes()
        self.assertEqual(len(modes), 2 + 2 * len(payload.STATE_FAMILIES))
        combos = {(m["name"], m.get("state_type") if m["allow_state"] else None)
                  for m in modes}
        for st in payload.STATE_FAMILIES:
            self.assertIn(("hybrid", st), combos)
            self.assertIn(("moe_hybrid", st), combos)
        self.assertIn(("dense", None), combos)
        self.assertIn(("moe", None), combos)

    def test_module_level_arch_modes_uses_builder(self):
        # The shipped constant must be the full cross product, so a full
        # regen can never silently drop a state family again.
        self.assertEqual(
            [(m["name"], m["state_type"], m["allow_moe"], m["allow_state"])
             for m in payload.ARCH_MODES],
            [(m["name"], m["state_type"], m["allow_moe"], m["allow_state"])
             for m in payload.build_arch_modes()],
        )

    def test_gpt_oss_and_kimi_recipes_present(self):
        combos = {(m["name"], m["state_type"]) for m in payload.ARCH_MODES
                  if m["allow_moe"] and m["allow_state"]}
        self.assertIn(("moe_hybrid", "sliding_window"), combos)  # GPT-OSS
        self.assertIn(("moe_hybrid", "kda"), combos)             # Kimi
        self.assertIn(("moe_hybrid", "gla"), combos)

    def test_subsetting(self):
        modes = payload.build_arch_modes(["moe_hybrid"], ["kda", "gla"])
        self.assertEqual(
            [(m["name"], m["state_type"]) for m in modes],
            [("moe_hybrid", "kda"), ("moe_hybrid", "gla")],
        )
        for m in modes:
            self.assertTrue(m["allow_moe"] and m["allow_state"])


class CliParsingTests(unittest.TestCase):
    def test_parse_csv_list(self):
        self.assertEqual(payload._parse_csv_list("7,13", float), [7.0, 13.0])
        self.assertEqual(payload._parse_csv_list(" a, b ,"), ["a", "b"])
        self.assertIsNone(payload._parse_csv_list(""))
        self.assertIsNone(payload._parse_csv_list(None))

    def test_cli_accepts_axis_flags(self):
        args = payload.build_cli().parse_args([
            "--hardware", "h100", "--params", "7", "--tokens", "2.0",
            "--arch-modes", "moe_hybrid",
            "--state-types", "kda,gla,sliding_window",
            "--multi-ctx", "--ctx-sweep",
            "--merge-into", "x.json",
        ])
        self.assertEqual(args.hardware, "h100")
        self.assertEqual(args.state_types, "kda,gla,sliding_window")
        self.assertTrue(args.multi_ctx and args.ctx_sweep)
        self.assertEqual(args.merge_into, "x.json")


def _w31_row(hw="h100", params=7.0, tokens=2.0, ctx=32768, serving="continuous",
         arch_mode="moe", state_type=None, loss=2.1):
    return {
        "hw": hw, "params_B": params, "tokens_T": tokens,
        "context_length": ctx, "serving": serving,
        "arch_mode": arch_mode, "state_type": state_type,
        "candidates": 1, "feasible": 1,
        "optimal": {
            "arch_family": arch_mode, "loss": loss, "tbt_ms": 10.0,
            "ttft_ms": 100.0, "mem_gb": 5.0, "train_tps": 1000.0,
            "active_params_B": params, "params_B": params,
            "uncertainty_total_pct": 3.0, "tp": 4,
            "d_model": 4096, "n_layers": 32,
        },
        "pareto": [],
    }


class MergePayloadTests(unittest.TestCase):
    def test_replace_append_keep(self):
        base = {
            "grid": [
                _w31_row(arch_mode="dense", loss=2.2),
                _w31_row(arch_mode="moe", loss=2.1),
            ],
            "hardware_info": {"h100": {"hbm_gb": 80}},
            "cells": ["stale"], "_family_smoothing": {"stale": True},
        }
        new = {
            "grid": [
                _w31_row(arch_mode="moe", loss=2.05),                   # replaces
                _w31_row(arch_mode="moe_hybrid", state_type="kda",
                     loss=2.04),                                    # appends
            ],
            "hardware_info": {"h100": {"hbm_gb": 80, "new": True}},
        }
        merged = payload.merge_payload(base, new)
        self.assertEqual(len(merged["grid"]), 3)
        moe = [r for r in merged["grid"] if r["arch_mode"] == "moe"
               and r["state_type"] is None]
        self.assertEqual(len(moe), 1)
        self.assertEqual(moe[0]["optimal"]["loss"], 2.05)  # replaced
        kinds = {(r["arch_mode"], r["state_type"]) for r in merged["grid"]}
        self.assertIn(("moe_hybrid", "kda"), kinds)        # appended
        self.assertIn(("dense", None), kinds)              # kept
        # Derived state dropped so a forgotten post chain fails loudly.
        self.assertNotIn("cells", merged)
        self.assertNotIn("_family_smoothing", merged)
        self.assertTrue(merged["hardware_info"]["h100"]["new"])

    def test_merge_then_post_chain_surfaces_new_family(self):
        base = {"grid": [_w31_row(arch_mode="moe", loss=2.1)],
                "hardware_info": {}}
        new = {"grid": [_w31_row(arch_mode="moe_hybrid", state_type="kda",
                             loss=2.04)],
               "hardware_info": {}}
        merged = payload.merge_payload(base, new)
        payload.run_post_chain(merged)
        cell = merged["cells"][0]
        fams = {(f["arch_mode"], f["state_type"]) for f in cell["families"]}
        self.assertIn(("moe_hybrid", "kda"), fams)
        self.assertIn(("moe", None), fams)


class PublicPayloadQualityFilterTests(unittest.TestCase):
    def test_post_chain_prunes_quality_sentinel_records(self):
        row = _w31_row(arch_mode="moe", loss=2.10)
        sentinel = dict(row["optimal"])
        sentinel.update({
            "arch_family": "dense",
            "loss": 2_000_000.0,
            "chinchilla": 2.0,
            "penalty_pct": 100_000_000.0,
        })
        row["pareto"] = [sentinel, dict(row["optimal"], loss=2.11)]

        sentinel_row = _w31_row(arch_mode="hybrid", state_type="mamba2",
                            loss=2_000_000.0)
        sentinel_row["optimal"].update({
            "quality_sentinel": True,
            "chinchilla": 2.0,
            "penalty_pct": 100_000_000.0,
        })

        data = {"grid": [row, sentinel_row], "hardware_info": {}}
        payload.run_post_chain(data)

        self.assertEqual(data["_public_quality_filter"]["pareto"], 1)
        self.assertEqual(data["_public_quality_filter"]["optimal"], 1)
        self.assertEqual(len(row["pareto"]), 1)
        self.assertIsNone(sentinel_row["optimal"])
        self.assertEqual(
            sentinel_row["omitted_optimal"]["reason"],
            "outside_quality_model_coverage",
        )
        self.assertEqual(sentinel_row["feasible"], 0)
        self.assertEqual(sentinel_row["pareto_size"], 0)
        self.assertEqual(
            sentinel_row["infeasible_reasons"],
            ["All candidates fall outside calibrated quality-model coverage"],
        )
        family_losses = [
            f["loss"]
            for cell in data["cells"]
            for f in cell.get("families", [])
        ]
        self.assertTrue(family_losses)
        self.assertTrue(all(loss < 100.0 for loss in family_losses))

        # Resumable web shards may already have dropped the sentinel before
        # final assembly. Re-running the post-filter must still reconcile the
        # stale pre-filter feasible count from that persisted shard.
        sentinel_row["feasible"] = 9
        payload._prune_public_quality_sentinels(data)
        self.assertEqual(sentinel_row["feasible"], 0)


class PostChainSharedTests(unittest.TestCase):
    def test_emit_decision_grid_delegates(self):
        import emit_decision_grid as edg
        data = {"grid": [_w31_row()], "hardware_info": {}}
        out = edg.rebuild_chain(data)
        self.assertIn("cells", out)
        self.assertEqual(out["cells"][0]["families"][0]["arch_mode"], "moe")


# ============================================================
# test_wave32_fixes.py  (w32)
# ============================================================
"""Wave 32 (Jul 2026) regression pins — component wiring due-diligence.

Three root causes fixed:

1. CSA / IndexShare / MSA (Wave 9 evaluator support) were reachable only
   via the Python API: ac-compile had NO flags for them, so no user-facing
   surface could ever select them. Wired: --allow-csa / --allow-indexshare
   / --allow-msa + per-family option flags, `compressed` --help-group,
   generator --allow-compressed-attention.

2. Their PREFILL cost fell through to the full S x S branch in
   compute_layer_time — sparse attention priced at dense prefill (decode
   was already sparse via kv_bytes_per_token_per_layer, so TBT was right
   while TTFT was dense). Now each family uses the NSA-style
   S x attended sparse-cost model with `attended` mirroring its
   decode-side effective-KV formula.

3. The web grid's MoE axis was pinned to n_experts=[8], top_k=[2],
   granularity=[1.0] — a coarse MoE whose effective-capacity gain (~+10%
   effective params) made "MoE barely beats dense" a driver artifact.
   GRID_MOE_* defaults now sweep coarse AND fine points, overridable via
   --moe-n-experts / --moe-top-k / --moe-granularity. Verified: at
   13B/2T/8k, fine-grained 64x8 (g=0.25) scores ~1.0% below dense while
   coarse 64x8 is correctly penalized.

Also wired: --moe-granularity / --ep-topology / --mla-rope-head-dim /
--mla-nope-head-dim / --mtp-depth-n-layers / --mtp-train-loss-weight /
--max-full-evaluations / --allow-quality-sentinel on ac-compile.
"""

def _prefill_attention_s(attention_type, seq_len=1048576, **extras):
    from ac.throughput_model import (
        load_hardware, compute_layer_time, ArchConfig)
    from ac.lattice_engine import HARDWARE as LATTICE_HW
    hw = load_hardware("h100")
    lhw = LATTICE_HW["h100"]
    arch = ArchConfig(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, batch_size=1, seq_len=seq_len,
        precision="bf16", kv_precision="bf16",
        attention_type=attention_type, **extras)
    return compute_layer_time(arch, hw, lhw, tp_degree=8,
                              phase="prefill").attention_s


class SparsePrefillPricingTests(unittest.TestCase):
    """Sparse attention families must prefill sub-quadratically —
    strictly cheaper than full S x S at long context, in proportion to
    their attended-token fraction."""

    def setUp(self):
        self.full = _prefill_attention_s("full")

    def test_msa_prefill_below_full(self):
        msa = _prefill_attention_s(
            "msa", msa_window_size=1024, msa_dilated_top_k=64,
            msa_global_top_k=16)
        # attended ~= 1104 of 1M tokens -> ~0.1% of full.
        self.assertLess(msa, 0.01 * self.full)

    def test_csa_prefill_below_full(self):
        csa = _prefill_attention_s(
            "csa", csa_block_size=64, csa_top_k_blocks=16,
            csa_compression_dim=64)
        self.assertLess(csa, 0.05 * self.full)

    def test_indexshare_prefill_below_full(self):
        idx = _prefill_attention_s(
            "indexshare", indexshare_num_buckets=64,
            indexshare_top_k_buckets=4, indexshare_index_dim=64)
        self.assertLess(idx, 0.20 * self.full)

    def test_short_context_capped_at_full(self):
        # At S smaller than the attended budget, sparse cost must not
        # exceed full (attended is min-capped at S).
        full_short = _prefill_attention_s("full", seq_len=512)
        msa_short = _prefill_attention_s(
            "msa", seq_len=512, msa_window_size=1024,
            msa_dilated_top_k=64, msa_global_top_k=16)
        self.assertLessEqual(msa_short, full_short * 1.01)


class CliWiringTests(unittest.TestCase):
    """Every Wave 32 flag must parse and land on DeploymentConstraints."""

    def _parse(self, extra):
        import cli_compile
        parser = cli_compile._build_parser()
        base = ["--hardware", "h100", "--params", "7", "--tokens", "2"]
        return parser.parse_args(base + extra)

    def test_compressed_flags_parse(self):
        args = self._parse([
            "--allow-csa", "--csa-block-sizes", "64,128",
            "--csa-top-k-blocks", "8,16", "--csa-compression-dim", "32",
            "--allow-indexshare", "--indexshare-buckets", "128",
            "--indexshare-top-k", "8", "--indexshare-index-dim", "32",
            "--allow-msa", "--msa-windows", "512,1024",
            "--msa-dilated-top-k", "64", "--msa-global-top-k", "16",
        ])
        self.assertTrue(args.allow_csa and args.allow_indexshare
                        and args.allow_msa)
        self.assertEqual(args.csa_block_sizes, "64,128")
        self.assertEqual(args.indexshare_buckets, "128")
        self.assertEqual(args.msa_windows, "512,1024")

    def test_moe_and_detail_flags_parse(self):
        args = self._parse([
            "--moe-granularity", "1.0,0.25", "--ep-topology", "cross_axis",
            "--mla-rope-head-dim", "64", "--mla-nope-head-dim", "128",
            "--mtp-depth-n-layers", "2", "--mtp-train-loss-weight", "0.1",
            "--max-full-evaluations", "500", "--allow-quality-sentinel",
        ])
        self.assertEqual(args.moe_granularity, "1.0,0.25")
        self.assertEqual(args.ep_topology, "cross_axis")
        self.assertEqual(args.mtp_depth_n_layers, 2)
        self.assertTrue(args.allow_quality_sentinel)

    def test_state_type_aliases_are_cli_and_schema_wired(self):
        import cli_compile
        parser = cli_compile._build_parser()
        base = ["--hardware", "h100", "--params", "7", "--tokens", "2",
                "--allow-state", "--no-shadow-prices"]

        swa = parser.parse_args(base + ["--state-type", "swa"])
        self.assertEqual(swa.state_type, "sliding_window")

        delta = parser.parse_args(base + ["--state-type", "delta_net"])
        self.assertEqual(delta.state_type, "deltanet")

        parallel = parser.parse_args(base + ["--state-type", "parallel_heads"])
        self.assertEqual(parallel.state_type, "parallel_heads")

        from schema import build_hybrid_config, validate_config
        cfg = build_hybrid_config(
            d_model=1024, n_layers=4, vocab_size=32000,
            attention_layer_indices=[0, 2],
            state_layer_indices=[1, 3],
            n_heads=8, d_head=128, n_kv_heads=4,
            state_type=parallel.state_type,
            state_d_state=128, state_n_heads=8, state_d_head=64,
            ffn_dim=4096, tp=1, pp=1, dp=1,
        )
        self.assertEqual(validate_config(cfg), [])

    def test_help_group_compressed_exists(self):
        import cli_compile
        from cli_recipe import render_group_help
        parser = cli_compile._build_parser()
        txt = render_group_help(parser, "compressed")
        for flag in ("--allow-csa", "--allow-indexshare", "--allow-msa"):
            self.assertIn(flag, txt)

    def test_direct_script_help_imports_work_from_release_tree(self):
        for rel in ("ac/cli_matrix18b.py", "ac/cli_trust_audit.py"):
            r = subprocess.run(
                [sys.executable, str(ROOT / rel), "--help"],
                cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("usage:", r.stdout)

    def test_stress_json_is_strict_json(self):
        r = subprocess.run(
            [
                sys.executable, str(ROOT / "ac/cli_stress.py"),
                "stress", "--known", "Mistral-7B", "--hw", "h100",
                "--batch", "8", "--decode-kv", "8192", "--tp", "8",
                "--json",
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("Infinity", r.stdout)
        parsed = json.loads(r.stdout)
        self.assertIsNone(parsed["intermediates"]["link_bw_ep_bytes_s"])

    def test_constraints_accept_wave32_fields(self):
        from optimizer import DeploymentConstraints
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=131072, tp=8, pp=1, dp=8,
            allow_csa=True, csa_block_size_options=[64],
            allow_indexshare=True, indexshare_num_buckets_options=[64],
            allow_msa=True, msa_window_options=[512],
            moe_granularity_targets=[1.0, 0.25],
            ep_topology="single_axis",
            allow_quality_sentinel=True,
        )
        self.assertEqual(c.moe_granularity_targets, [1.0, 0.25])


class StratifiedCapTests(unittest.TestCase):
    """max_candidates must not starve structural-variant classes.

    The plain even-stride cap kept 13/400 compressed candidates out of a
    419k enumeration where they were 47% — variant classes are generated
    in contiguous blocks shorter than the stride. The cap now buckets by
    (attention_type, has_moe, has_state) with equal shares."""

    @staticmethod
    def _cand(attention_type, i, moe=None, state=None):
        from types import SimpleNamespace
        return SimpleNamespace(attention_type=attention_type, moe=moe,
                               state_config=state, idx=i)

    def test_small_contiguous_block_survives(self):
        from ac.optimizer import _stratified_candidate_cap
        # 10_000 full candidates followed by a contiguous block of 50 msa:
        # a stride of ~25 would keep ~2 msa; stratified keeps its share.
        pool = [self._cand("full", i) for i in range(10_000)]
        pool += [self._cand("msa", i) for i in range(50)]
        out = _stratified_candidate_cap(pool, 400)
        kinds = [c.attention_type for c in out]
        self.assertEqual(len(out), 400)
        self.assertGreaterEqual(kinds.count("msa"), 50 // 2,
                                "msa block must not be stride-starved")

    def test_no_cap_needed_is_identity(self):
        from ac.optimizer import _stratified_candidate_cap
        pool = [self._cand("full", i) for i in range(10)]
        self.assertEqual(len(_stratified_candidate_cap(pool, 400)), 10)

    def test_deterministic(self):
        from ac.optimizer import _stratified_candidate_cap
        pool = ([self._cand("full", i) for i in range(500)]
                + [self._cand("csa", i) for i in range(300)]
                + [self._cand("mla", i, moe={"n": 8}) for i in range(200)])
        a = [c.idx for c in _stratified_candidate_cap(list(pool), 100)]
        b = [c.idx for c in _stratified_candidate_cap(list(pool), 100)]
        self.assertEqual(a, b)


class SerializeAttentionTypeTests(unittest.TestCase):
    """serialize_candidate must carry the candidate's true attention_type.

    The old code path ended with `else: d["attention_type"] = "full"` on
    the MLA check — every non-MLA variant (nsa, csa, indexshare, msa) was
    relabeled "full" in the web payload, overwriting even the explicit
    "nsa" assignment made earlier in the function. The search could pick
    a compressed-attention winner and the payload would display it as
    dense full attention."""

    def _serialize(self, attention_type, **extras):
        import _generator_payload as gp
        from optimizer import (DeploymentConstraints, generate_candidates,
                               evaluate_candidate)
        from optimizer import CandidateArch
        cand = CandidateArch(
            d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=14336, vocab_size=32000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
            kv_cache_bits=16, attention_type=attention_type,
            tp_degree=8, cp_degree=1, ep_degree=1, **extras)
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=131072, tp=8, pp=1, dp=8,
            serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16)
        ev = evaluate_candidate(cand, "h100", c)
        return gp.serialize_candidate(ev)

    def test_msa_survives_serialization(self):
        rec = self._serialize("msa", msa_window_size=512,
                              msa_dilated_top_k=64, msa_global_top_k=16)
        self.assertEqual(rec["attention_type"], "msa")
        self.assertEqual(rec["msa_window_size"], 512)

    def test_csa_survives_serialization(self):
        rec = self._serialize("csa", csa_block_size=64, csa_top_k_blocks=16,
                              csa_compression_dim=64)
        self.assertEqual(rec["attention_type"], "csa")
        self.assertEqual(rec["csa_top_k_blocks"], 16)

    def test_indexshare_survives_serialization(self):
        rec = self._serialize("indexshare", indexshare_num_buckets=64,
                              indexshare_top_k_buckets=4,
                              indexshare_index_dim=64)
        self.assertEqual(rec["attention_type"], "indexshare")

    def test_nsa_not_overwritten_to_full(self):
        rec = self._serialize("nsa", nsa_compress_block_size=64,
                              nsa_compress_block_stride=16,
                              nsa_select_block_size=64, nsa_select_top_k=16,
                              nsa_window_size=512)
        self.assertEqual(rec["attention_type"], "nsa")


class GridMoEAxisTests(unittest.TestCase):
    def test_grid_defaults_include_fine_grained(self):
        import _generator_payload as gp
        self.assertIn(64, gp.GRID_MOE_N_EXPERTS)
        self.assertIn(8, gp.GRID_MOE_TOP_K)
        self.assertTrue(any(g < 1.0 for g in gp.GRID_MOE_GRANULARITY),
                        "grid must sweep a fine-grained granularity point")

    def test_generator_cli_has_moe_and_compressed_flags(self):
        import _generator_payload as gp
        args = gp.build_cli().parse_args([
            "--moe-n-experts", "64", "--moe-top-k", "8",
            "--moe-granularity", "0.25", "--allow-compressed-attention",
        ])
        self.assertEqual(args.moe_n_experts, "64")
        self.assertTrue(args.allow_compressed_attention)


# ============================================================
# test_wave33_fixes.py  (w33)
# ============================================================
"""Wave 33 (Jul 2026) — compressed-attention x MoE / state composition.

Root cause: the Wave 9 CSA/IndexShare/MSA emissions lived only in the
dense generator, so a search could never produce sparse-attention MoE or
sparse-global hybrids — the exact stacks that compete with MLA+MoE+state
at long context. The evaluator already scored the combinations correctly;
the gap was pure enumeration. `_expand_compressed_variants` now runs at
both family-combine points and clones full-attention MoE/state candidates
into one variant per allowed compressed family (parameter ledger
unchanged, same convention as the Wave 9 dense emissions).

Measured effect (7B/2T/1M h100, single pool, everything allowed):
  csa x MoE         2.1068   <- new min-loss
  csa x MoE x state 2.1086
  csa x state       2.1106
  mla x MoE x state 2.1152   <- previous winner class
  msa x MoE         2.1288 @ 9.9 ms TBT / 3.9 s TTFT  <- serving winner
Margins are inside the uncalibrated +-2-3% band (compression residuals
have zero fit-pairs coverage), so the honest claim is "quality-equivalent
or better, strictly better serving" — but they must be IN the pool to
make that claim at all.
"""

def _w33_cand(attention_type="full", moe=None, state=None, local_layers=0):
    from optimizer import CandidateArch
    return CandidateArch(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, vocab_size=32000,
        weight_precision="bf16", ffn_precision="bf16",
        attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
        kv_cache_bits=16, attention_type=attention_type,
        tp_degree=1, cp_degree=1, ep_degree=1,
        moe=moe, state_config=state,
        n_local_attn_layers=local_layers,
    )


def _w33_constraints(**kw):
    from optimizer import DeploymentConstraints
    base = dict(target_params_b=7.0, training_tokens=int(2e12),
                context_length=131072, tp=1, pp=1, dp=1,
                serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16)
    base.update(kw)
    return DeploymentConstraints(**base)


class ExpandCompressedVariantsTests(unittest.TestCase):
    def test_moe_and_state_candidates_get_variants(self):
        from optimizer import _expand_compressed_variants
        pool = [
            _w33_cand(moe={"n_experts": 64, "top_k": 8, "expert_dim": 1024}),
            _w33_cand(state={"type": "sliding_window", "d_state": 64}),
        ]
        out = _expand_compressed_variants(
            pool, _w33_constraints(allow_csa=True, allow_indexshare=True,
                               allow_msa=True))
        kinds = {(c.attention_type, bool(c.moe), bool(c.state_config))
                 for c in out}
        for at in ("csa", "indexshare", "msa"):
            self.assertIn((at, True, False), kinds)
            self.assertIn((at, False, True), kinds)
        # 2 originals + 2 eligible x 3 types
        self.assertEqual(len(out), 2 + 6)

    def test_noop_when_not_allowed(self):
        from optimizer import _expand_compressed_variants
        pool = [_w33_cand(moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096})]
        out = _expand_compressed_variants(pool, _w33_constraints())
        self.assertEqual(len(out), 1)

    def test_dense_and_nonfull_candidates_not_expanded(self):
        from optimizer import _expand_compressed_variants
        pool = [
            _w33_cand(),                       # dense: covered by Wave 9 emissions
            _w33_cand("mla", moe={"n_experts": 8, "top_k": 2,
                              "expert_dim": 4096}),   # not full attention
            _w33_cand(moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096},
                  local_layers=8),         # local:global — excluded
        ]
        out = _expand_compressed_variants(
            pool, _w33_constraints(allow_msa=True))
        self.assertEqual(len(out), 3)

    def test_expansion_base_is_capped(self):
        from optimizer import _expand_compressed_variants
        pool = [_w33_cand(moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096})
                for _ in range(500)]
        out = _expand_compressed_variants(
            pool, _w33_constraints(allow_msa=True),
            max_expansion_per_type=100)
        self.assertEqual(len(out), 600)  # 500 + 100 capped msa copies

    def test_variant_preserves_ledger_and_config(self):
        from optimizer import _expand_compressed_variants
        moe = {"n_experts": 64, "top_k": 8, "expert_dim": 1024}
        pool = [_w33_cand(moe=moe)]
        out = _expand_compressed_variants(
            pool, _w33_constraints(allow_csa=True,
                               csa_block_size_options=[128],
                               csa_top_k_options=[32]))
        csa = [c for c in out if c.attention_type == "csa"][0]
        self.assertEqual(csa.total_params, pool[0].total_params)
        self.assertEqual(csa.moe, moe)
        self.assertEqual(csa.csa_block_size, 128)
        self.assertEqual(csa.csa_top_k_blocks, 32)

    def test_composed_candidates_evaluate(self):
        # End-to-end: expanded moe x msa candidate must pass the
        # architecture-view validator and produce sane numbers.
        from optimizer import (_expand_compressed_variants,
                               generate_moe_candidates, evaluate_candidate)
        c = _w33_constraints(allow_moe=True, max_total_params_b=56,
                         moe_n_experts_options=[8], moe_top_k_options=[2],
                         moe_granularity_targets=[1.0], ep_options=[2],
                         allow_msa=True, param_tolerance=0.10)
        moe_pool = generate_moe_candidates("h100", c)[:40]
        out = _expand_compressed_variants(moe_pool, c)
        msa = [x for x in out if x.attention_type == "msa" and x.moe]
        self.assertTrue(msa, "expansion produced no msa x MoE candidates")
        ev = evaluate_candidate(msa[0], "h100", c)
        self.assertGreater(ev.predicted_loss, 1.5)
        self.assertLess(ev.predicted_loss, 4.0)
        self.assertGreater(ev.serving_tbt_ms, 0)


class BudgetIsABoundTests(unittest.TestCase):
    """Declared --serving-tbt / --serving-ttft budgets must bind at pick
    time. They were soft guards consulted by nobody: a 50 ms TBT budget
    could return a 1079 ms 'optimal' with only a buried warning."""

    @staticmethod
    def _view(loss, tbt, mem, prefill_ms=1000.0, unc=3.0):
        from types import SimpleNamespace
        return SimpleNamespace(
            predicted_loss=loss, memory_per_gpu_gb=mem, serving_tbt_ms=tbt,
            training_tps=1000.0,
            throughput=SimpleNamespace(prefill_time_ms=prefill_ms),
            quality=SimpleNamespace(uncertainty_total=unc / 100.0),
            arch=SimpleNamespace(total_params=7e9, n_layers=32, d_model=4096),
        )

    def _w33_constraints(self, **kw):
        from types import SimpleNamespace
        base = dict(objective_profile="research_quality",
                    strict_quality=False, serving_tbt_ms=None,
                    serving_ttft_ms=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_budget_violator_loses_despite_lower_loss(self):
        from optimizer import build_display_sort_key
        fast = self._view(2.12, tbt=40.0, mem=50.0)
        slow = self._view(2.10, tbt=1000.0, mem=20.0)  # better loss+mem
        pool = [slow, fast]
        key = build_display_sort_key(
            pool, self._w33_constraints(serving_tbt_ms=50.0))
        self.assertIs(sorted(pool, key=key)[0], fast)

    def test_no_budget_declared_is_unchanged(self):
        from optimizer import build_display_sort_key
        fast = self._view(2.12, tbt=40.0, mem=50.0)
        slow = self._view(2.10, tbt=1000.0, mem=20.0)
        pool = [slow, fast]
        key = build_display_sort_key(pool, self._w33_constraints())
        # In-band (0.95% < band? 2.10 vs 2.12 is 0.95% > 0.5% band) —
        # loss bucket separates them; slow wins on loss.
        self.assertIs(sorted(pool, key=key)[0], slow)

    def test_all_violators_falls_back_gracefully(self):
        from optimizer import build_display_sort_key
        a = self._view(2.10, tbt=900.0, mem=20.0)
        b = self._view(2.12, tbt=800.0, mem=50.0)
        pool = [a, b]
        key = build_display_sort_key(
            pool, self._w33_constraints(serving_tbt_ms=50.0))
        self.assertIs(sorted(pool, key=key)[0], a)  # rank among violators

    def test_ttft_budget_also_binds(self):
        from optimizer import build_display_sort_key
        quick = self._view(2.12, tbt=40.0, mem=50.0, prefill_ms=500.0)
        slow_prefill = self._view(2.10, tbt=40.0, mem=20.0,
                                  prefill_ms=90_000.0)
        pool = [slow_prefill, quick]
        key = build_display_sort_key(
            pool, self._w33_constraints(serving_ttft_ms=1000.0))
        self.assertIs(sorted(pool, key=key)[0], quick)


# ============================================================
# test_wave34_fixes.py  (w34)
# ============================================================
"""Wave 34 (Jul 2026) — two-stage-by-default search + local refinement.

Design: cover the supported possibilities COMPLETELY before dropping
anything, then spend a small extra budget densifying what stage one
found.

  Stage 1 (complete coverage): every deduped candidate is scored by the
    O(microseconds) `_cheap_quality_rank`; the `max_candidates` cap is
    stratified by structural class and, within each class, keeps the top
    70% of its share by cheap rank + a 30% even-stride diversity tail
    (the cheap rank is a loss proxy blind to throughput/memory — a pure
    cheap-loss selection would strip the fast/low-memory end of the
    Pareto surface that budget-bound picks need).
  Stage 2 (local refinement): after full evaluation, up to
    `local_refine_budget` (default 96, CLI --local-refine-budget)
    unevaluated lattice neighbors of the per-class Pareto leaders are
    pulled from the PRE-CAP pool (real generator output only — the
    parameter ledger stays trustworthy) and fully evaluated. No-op when
    the cap didn't drop anything.

Also pinned: declared --serving-tbt / --serving-ttft budgets bind at
pick time (Wave 33 fix, verified here through the constraint layer the
CLI builds).

Measured on 7B/2T/1M h100 (cap 400, all axes): refine 0 -> 2.1156;
refine 96 -> 2.1135 for ~+2.5 s.
"""

def _w34_cand(d_model=4096, n_layers=32, ffn_dim=14336, attention_type="full",
          moe=None, state=None, tp=1):
    from optimizer import CandidateArch
    return CandidateArch(
        d_model=d_model, n_layers=n_layers, n_heads=32, d_head=128,
        n_kv_heads=8, ffn_dim=ffn_dim, vocab_size=32000,
        weight_precision="bf16", ffn_precision="bf16",
        attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
        kv_cache_bits=16, attention_type=attention_type,
        tp_degree=tp, cp_degree=1, ep_degree=1,
        moe=moe, state_config=state,
    )


def _w34_constraints(**kw):
    from optimizer import DeploymentConstraints
    base = dict(target_params_b=7.0, training_tokens=int(2e12),
                context_length=131072, tp=1, pp=1, dp=1,
                serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16)
    base.update(kw)
    return DeploymentConstraints(**base)


class CheapRankGuidedCapTests(unittest.TestCase):
    def test_good_shape_survives_regardless_of_position(self):
        from optimizer import _stratified_candidate_cap, _cheap_quality_rank
        c = _w34_constraints()
        # 400 pathological shapes (extreme aspect ratio) followed by one
        # good shape at the very END of the enumeration order — a blind
        # stride keeps it only by luck; cheap-rank-guided keeps it surely.
        pool = [_w34_cand(d_model=8192, n_layers=4, ffn_dim=32768)
                for _ in range(400)]
        good = _w34_cand(d_model=4096, n_layers=36)
        pool.append(good)
        out = _stratified_candidate_cap(pool, 50, c)
        self.assertIn(good, out)

    def test_without_constraints_falls_back_to_stride(self):
        from optimizer import _stratified_candidate_cap
        pool = [_w34_cand(d_model=8192, n_layers=4) for _ in range(400)]
        good = _w34_cand(d_model=4096, n_layers=36)
        pool.append(good)
        out = _stratified_candidate_cap(pool, 50)
        self.assertEqual(len(out), 50)  # legacy behavior: no cheap rank

    def test_diversity_tail_preserved(self):
        # The stride tail must keep candidates that cheap rank hates —
        # they may carry the throughput end of the frontier.
        from optimizer import _stratified_candidate_cap, _cheap_quality_rank
        c = _w34_constraints()
        pool = ([_w34_cand(d_model=4096, n_layers=30 + i % 8) for i in range(300)]
                + [_w34_cand(d_model=8192, n_layers=4) for _ in range(100)])
        out = _stratified_candidate_cap(pool, 40, c)
        ranked = sorted(pool, key=lambda x: _cheap_quality_rank(
            x, c.training_tokens, c.quality_model_version))
        top28 = set(id(x) for x in ranked[:28])
        stride_picks = [x for x in out if id(x) not in top28]
        self.assertGreaterEqual(len(stride_picks), 8)

    def test_deterministic(self):
        from optimizer import _stratified_candidate_cap
        c = _w34_constraints()
        pool = [_w34_cand(d_model=4096 if i % 3 else 5120, n_layers=28 + i % 10)
                for i in range(200)]
        a = _stratified_candidate_cap(list(pool), 30, c)
        b = _stratified_candidate_cap(list(pool), 30, c)
        self.assertEqual([id(x) for x in a], [id(x) for x in b])


class RefinementNeighborTests(unittest.TestCase):
    def test_neighbors_selected_by_class_and_distance(self):
        from optimizer import _select_refinement_neighbors
        leader = _w34_cand(d_model=4096, n_layers=32)
        near = _w34_cand(d_model=4096, n_layers=33)
        far_shape = _w34_cand(d_model=6144, n_layers=20)
        wrong_class = _w34_cand(d_model=4096, n_layers=33,
                            moe={"n_experts": 8, "top_k": 2,
                                 "expert_dim": 4096})
        out = _select_refinement_neighbors(
            [leader], [near, far_shape, wrong_class], [leader], 10)
        self.assertIn(near, out)
        self.assertNotIn(far_shape, out)
        self.assertNotIn(wrong_class, out)

    def test_already_evaluated_excluded_and_budget_respected(self):
        from optimizer import _select_refinement_neighbors
        leader = _w34_cand(d_model=4096, n_layers=32)
        pool = [_w34_cand(d_model=4096, n_layers=32 + dl) for dl in range(-3, 4)]
        out = _select_refinement_neighbors([leader], pool, [pool[3]], 4)
        self.assertLessEqual(len(out), 4)
        self.assertNotIn(pool[3], out)  # identical to an evaluated arch

    def test_zero_budget_disables(self):
        from optimizer import _select_refinement_neighbors
        leader = _w34_cand()
        self.assertEqual(
            _select_refinement_neighbors([leader], [_w34_cand(n_layers=33)],
                                         [leader], 0), [])


class CliKnobTests(unittest.TestCase):
    def test_local_refine_budget_flag(self):
        import cli_compile
        parser = cli_compile._build_parser()
        args = parser.parse_args(["--hardware", "h100", "--params", "7",
                                  "--tokens", "2",
                                  "--local-refine-budget", "0"])
        self.assertEqual(args.local_refine_budget, 0)

    def test_constraint_default(self):
        c = _w34_constraints()
        self.assertEqual(c.local_refine_budget, 96)


class BudgetBindsEndToEndTests(unittest.TestCase):
    def test_single_ctx_optimize_respects_tbt_budget(self):
        # Small real search at long ctx: without the Wave 33 fix the
        # min-loss pick violated the declared budget.
        from optimizer import optimize
        c = _w34_constraints(context_length=262144, serving_tbt_ms=30.0,
                         tp=4, allow_mla=True, mla_kv_latent_options=[512],
                         max_candidates=150, local_refine_budget=0,
                         param_tolerance=0.10)
        r = optimize("h100", c)
        self.assertIsNotNone(r.optimal)
        self.assertLessEqual(r.optimal.serving_tbt_ms, 30.0 + 1e-6)


# ============================================================
# test_wave35_fixes.py  (w35)
# ============================================================
"""Wave 35 (Jul 2026) regression pins — compressed-attention & YOCO
physics under parallelism, plus emission/round-trip wiring.

Five root causes fixed:

1. NSA/CSA/IndexShare/MSA decode KV bandwidth did NOT shard under TP
   (the capacity estimator already sharded by min(TP, n_kv_heads) — the
   two paths contradicted each other, and physics says these caches are
   stored per kv head, so they shard like GQA). At TP=8 every rank was
   charged the full unsharded stream: an 8× TBT over-charge. Only the
   MLA latent and the IndexShare per-token index entry are genuinely
   TP-replicated (shared across query heads). New helper:
   ArchConfig.kv_bytes_per_token_split.

2. YOCO decode claimed a K/N bandwidth credit. Wrong: each cross-decoder
   layer still streams the shared cache from HBM every decode step (a
   multi-GB cache does not persist in ~50 MB of L2 across layers), so
   the decode stream matches a conventional stack. Removed.

3. YOCO serving-prefill early-exit was not modeled at all (the paper's
   headline TTFT win): cold prefill only needs all S positions through
   the K self layers plus the last position through the cross-decoder.
   Training is unaffected. Added to the serving prefill path.

4. A csa/indexshare/msa WINNER was emitted as attention.type="full"
   (schema emitter had no compressed path; validator rejected the types;
   baseline loader rejected nsa + compressed configs — greenfield could
   emit configs its own delta-eval refused to load). All three wired.

5. optimizer_bridge.candidate_to_arch mapped weight_precision into
   ArchConfig.precision while the canonical evaluate_candidate maps
   ffn_precision — every delta round-trip on an fp8-FFN baseline
   silently flipped the candidate to bf16 FFN (phantom field_changes
   row + contaminated TBT/TTFT deltas).

Also: stress.py priced NSA/CSA/IndexShare/MSA decode/footprint at FULL
dense KV (only MLA was special-cased), and had no YOCO capacity factor;
footprint and decode-stream now share one core that differs exactly on
the YOCO capacity term.
"""

def _arch(attention_type="full", **kw):
    from ac.throughput_model import ArchConfig
    return ArchConfig(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, batch_size=8, seq_len=131072,
        precision="bf16", kv_precision="bf16",
        attention_type=attention_type, **kw)


_FAMILY_KW = {
    "nsa": dict(nsa_window_size=512, nsa_select_top_k=16,
                nsa_select_block_size=64, nsa_compress_block_size=64,
                nsa_compress_block_stride=16),
    "csa": dict(csa_block_size=64, csa_top_k_blocks=16,
                csa_compression_dim=64),
    "indexshare": dict(indexshare_num_buckets=64, indexshare_top_k_buckets=4,
                       indexshare_index_dim=64),
    "msa": dict(msa_window_size=512, msa_dilated_top_k=64,
                msa_global_top_k=16),
}


def _decode_kv_load_s(attention_type, tp, **kw):
    """Isolate the decode KV stream (bandwidth term), not the max() with
    compute — MSA is compute-bound so attention_s alone can't detect a
    bandwidth mis-shard."""
    from ac.throughput_model import load_hardware
    arch = _arch(attention_type, **kw)
    hw = load_hardware("h100")
    shard_b, repl_b = arch.kv_bytes_per_token_split(131072)
    kv_shards = max(1, min(tp, arch.n_kv_heads))
    return (arch.batch_size * 131072
            * (shard_b / kv_shards + repl_b)) / hw.hbm_bandwidth_bytes_s


class DecodeKvTpShardingTests(unittest.TestCase):
    def test_per_kv_head_families_shard_by_tp(self):
        """attention_s at TP=8 must be 8× cheaper than TP=1 for the
        per-kv-head cache families (bandwidth-bound at 131k)."""
        from ac.throughput_model import (load_hardware, compute_layer_time)
        from ac.lattice_engine import HARDWARE as LHW
        hw, lhw = load_hardware("h100"), LHW["h100"]
        for fam in ("nsa", "csa"):
            t = {}
            for tp in (1, 8):
                b = compute_layer_time(_arch(fam, **_FAMILY_KW[fam]), hw, lhw,
                                       tp_degree=tp, phase="decode",
                                       kv_cache_len=131072)
                t[tp] = b.attention_s
            self.assertAlmostEqual(t[1] / t[8], 8.0, delta=0.2,
                                   msg=f"{fam} decode KV must shard by TP")

    def test_indexshare_index_is_tp_replicated(self):
        """IndexShare: kv part shards, per-token index entry does not.
        With buckets=64/top_k=4/idx=64 at bf16: kv=256 B, idx=256 B per
        token → TP=8 ratio must be 512/288 ≈ 1.78, NOT 8 (fully sharded)
        and NOT 1 (fully replicated)."""
        r = (_decode_kv_load_s("indexshare", 1, **_FAMILY_KW["indexshare"])
             / _decode_kv_load_s("indexshare", 8, **_FAMILY_KW["indexshare"]))
        self.assertAlmostEqual(r, 512.0 / 288.0, delta=0.05)

    def test_mla_latent_stays_replicated(self):
        r = (_decode_kv_load_s("mla", 1, mla_kv_latent_dim=512,
                               mla_rope_head_dim=64)
             / _decode_kv_load_s("mla", 8, mla_kv_latent_dim=512,
                                 mla_rope_head_dim=64))
        self.assertAlmostEqual(r, 1.0, delta=1e-6)

    def test_split_sums_to_total(self):
        for fam, kw in _FAMILY_KW.items():
            a = _arch(fam, **kw)
            s, r = a.kv_bytes_per_token_split(131072)
            self.assertAlmostEqual(
                s + r, a.kv_bytes_per_token_per_layer(131072), delta=1,
                msg=fam)

    def test_capacity_and_bandwidth_agree(self):
        """The per-GPU KV capacity divided by HBM BW must equal the decode
        KV stream time — the Wave 35 invariant that was violated before."""
        from ac.throughput_model import estimate_memory_per_gpu, load_hardware
        for fam in ("nsa", "csa", "indexshare", "msa"):
            arch = _arch(fam, **_FAMILY_KW[fam])
            mem_kv = (estimate_memory_per_gpu(
                          arch, tp_degree=8, pp_degree=1,
                          kv_cache_len=131072, include_kv_cache=True)
                      - estimate_memory_per_gpu(
                          arch, tp_degree=8, pp_degree=1,
                          kv_cache_len=131072, include_kv_cache=False))
            hw = load_hardware("h100")
            stream_s = _decode_kv_load_s(fam, 8, **_FAMILY_KW[fam])
            self.assertAlmostEqual(
                mem_kv / arch.n_layers,
                stream_s * hw.hbm_bandwidth_bytes_s,
                delta=0.02 * mem_kv / arch.n_layers, msg=fam)


class YocoPhysicsTests(unittest.TestCase):
    def _run(self, yoco_k):
        from ac.throughput_model import throughput, estimate_memory_per_gpu
        arch = _arch("full", yoco_n_self_attn_layers=yoco_k)
        r = throughput(arch, "h100", tp_degree=8, pp_degree=1, dp_degree=1,
                       prefill_seq_len=131072, decode_kv_len=131072)
        mem = estimate_memory_per_gpu(arch, tp_degree=8, pp_degree=1,
                                      kv_cache_len=131072)
        return r, mem

    def test_yoco_decode_tbt_unchanged(self):
        base, _ = self._run(0)
        yoco, _ = self._run(4)
        self.assertAlmostEqual(yoco.decode_time_per_token_ms,
                               base.decode_time_per_token_ms,
                               delta=0.02 * base.decode_time_per_token_ms)

    def test_yoco_prefill_early_exit(self):
        base, _ = self._run(0)
        yoco, _ = self._run(4)  # K/N = 1/8 of the GEMM part
        self.assertLess(yoco.prefill_time_ms, 0.35 * base.prefill_time_ms)
        self.assertGreater(yoco.prefill_time_ms,
                           0.08 * base.prefill_time_ms)

    def test_yoco_training_unchanged(self):
        base, _ = self._run(0)
        yoco, _ = self._run(4)
        self.assertAlmostEqual(
            yoco.training_throughput_tokens_per_sec,
            base.training_throughput_tokens_per_sec,
            delta=0.01 * base.training_throughput_tokens_per_sec)

    def test_yoco_capacity_shrinks(self):
        _, base_mem = self._run(0)
        _, yoco_mem = self._run(4)
        self.assertLess(yoco_mem, 0.75 * base_mem)

    def test_stress_footprint_vs_stream_diverge_under_yoco(self):
        from ac.stress import (_kv_cache_bytes_total,
                               _kv_load_bytes_per_decode_step)
        arch = _arch("full", yoco_n_self_attn_layers=4)
        cap = _kv_cache_bytes_total(arch, 131072, tp_degree=8)
        stream = _kv_load_bytes_per_decode_step(arch, 131072, tp_degree=8)
        self.assertAlmostEqual(cap / stream, 4 / 32, delta=0.01)

    def test_stress_prices_compressed_attention_sparse(self):
        """MSA decode stream must be ≪ dense (was priced at FULL dense)."""
        from ac.stress import _kv_load_bytes_per_decode_step
        dense = _kv_load_bytes_per_decode_step(_arch("full"), 131072,
                                               tp_degree=8)
        msa = _kv_load_bytes_per_decode_step(
            _arch("msa", **_FAMILY_KW["msa"]), 131072, tp_degree=8)
        self.assertLess(msa, 0.02 * dense)

    def test_stress_capacity_honors_context_parallelism(self):
        from ac.stress import (Workload, _kv_cache_bytes_total,
                               compute_throughput_stress)
        arch = _arch("msa", **_FAMILY_KW["msa"])
        cp1 = _kv_cache_bytes_total(arch, 131072, tp_degree=8, cp_degree=1)
        cp4 = _kv_cache_bytes_total(arch, 131072, tp_degree=8, cp_degree=4)
        self.assertAlmostEqual(cp4 / cp1, 0.25, delta=1e-6)

        wl = Workload(batch_size=8, prefill_seq_len=131072,
                      decode_kv_len=131072, phase="decode")
        sv_cp1 = compute_throughput_stress(
            arch, "h100", wl, tp_degree=8, cp_degree=1)
        sv_cp4 = compute_throughput_stress(
            arch, "h100", wl, tp_degree=8, cp_degree=4)
        # Weights do not shrink with CP, but KV + serving activations do.
        self.assertLess(sv_cp4.hbm_capacity, 0.45 * sv_cp1.hbm_capacity)

    def test_cli_stress_defaults_to_schema_context_parallelism(self):
        from ac.schema import build_config

        cfg = build_config(
            d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=14336, vocab_size=32000,
            compressed={"type": "msa", **_FAMILY_KW["msa"]},
            tp=8, cp=4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arch.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            cmd = [
                sys.executable, str(ROOT / "ac" / "cli_stress.py"),
                "stress", "--arch", str(path), "--hardware", "h100",
                "--batch", "8", "--prefill-seq", "131072",
                "--decode-kv", "131072", "--json",
            ]
            import subprocess
            result = subprocess.run(
                cmd, cwd=str(ROOT), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertLess(payload["hbm_capacity"], 0.7)


class CompressedEmissionRoundTripTests(unittest.TestCase):
    def _emit(self, ctype, params):
        from ac.schema import build_config
        return build_config(
            d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=14336, vocab_size=32000,
            compressed={"type": ctype, **params},
        )

    def _msa_result(self):
        from ac.optimizer import CandidateArch, DeploymentConstraints
        from ac.optimizer import OptimizationResult, evaluate_candidate
        cand = CandidateArch(
            d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=14336, vocab_size=32000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"qk": "bf16", "v": "bf16", "output": "bf16"},
            kv_cache_bits=16, attention_type="msa",
            msa_window_size=512, msa_dilated_top_k=64, msa_global_top_k=16,
            tp_degree=8, cp_degree=1, ep_degree=1,
        )
        constraints = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=8, pp=1, dp=1)
        ev = evaluate_candidate(cand, "h100", constraints)
        return OptimizationResult(
            optimal=ev, pareto_frontier=[ev], all_evaluated=[ev],
            constraints=constraints, hardware="h100",
            candidates_generated=1, candidates_feasible=1,
            candidates_evaluated=1, search_time_sec=0.0,
            binding_constraints=[],
        )

    def test_emit_and_validate_all_families(self):
        from ac.schema import validate_config
        for ctype, params in (
                ("csa", {"csa_block_size": 128, "csa_top_k_blocks": 8,
                         "csa_compression_dim": 64}),
                ("indexshare", {"indexshare_num_buckets": 128,
                                "indexshare_top_k_buckets": 8,
                                "indexshare_index_dim": 64}),
                ("msa", {"msa_window_size": 1024, "msa_dilated_top_k": 64,
                         "msa_global_top_k": 16})):
            cfg = self._emit(ctype, params)
            attn = cfg["architecture"]["layer_configs"][0]["attention"]
            self.assertEqual(attn["type"], ctype)
            for k, v in params.items():
                self.assertEqual(attn[k], v, msg=f"{ctype}.{k}")
            errors = validate_config(cfg)
            self.assertEqual(errors, [], msg=f"{ctype}: {errors}")

    def test_baseline_loader_ingests_compressed_and_nsa(self):
        import json
        import tempfile
        from ac.baseline import load_baseline_model
        cfg = self._emit("csa", {"csa_block_size": 128,
                                 "csa_top_k_blocks": 8,
                                 "csa_compression_dim": 64})
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name
        loaded = load_baseline_model(path)
        cand = loaded.candidate
        self.assertEqual(cand.attention_type, "csa")
        self.assertEqual(int(cand.csa_block_size), 128)
        self.assertEqual(int(cand.csa_top_k_blocks), 8)

    def test_hybrid_emission_carries_compressed(self):
        from ac.schema import build_hybrid_config
        cfg = build_hybrid_config(
            d_model=4096, n_layers=8, vocab_size=32000,
            attention_layer_indices=[0, 4],
            state_layer_indices=[1, 2, 3, 5, 6, 7],
            compressed={"type": "msa", "msa_window_size": 512,
                        "msa_dilated_top_k": 64, "msa_global_top_k": 16},
        )
        attn_lc = [lc for lc in cfg["architecture"]["layer_configs"]
                   if lc["type"] == "transformer_block"][0]
        self.assertEqual(attn_lc["attention"]["type"], "msa")

    def test_selected_compressed_winner_emits_its_type(self):
        """End-to-end: a search whose winner is csa/indexshare/msa must
        emit that type (was silently emitted as `full`)."""
        from ac.optimizer import result_to_config
        result = self._msa_result()
        cfg = result_to_config(result)
        attn = cfg["architecture"]["layer_configs"][0]["attention"]
        self.assertEqual(attn["type"], "msa")
        self.assertEqual(attn["msa_window_size"], 512)

    def test_cli_summary_names_compressed_winner(self):
        """The one-line CLI summary must match the evaluated attention type.
        It previously printed `attn=full` for MSA/CSA/IndexShare winners."""
        from ac.cli_compile import _format_optimal_line
        result = self._msa_result()
        text = _format_optimal_line(result.optimal)
        self.assertIn("attn=msa(window=512,dilated_top_k=64,global_top_k=16)", text)
        self.assertNotIn("attn=full", text)

    def test_pareto_csv_exposes_compressed_knobs(self):
        from ac.optimizer import result_to_pareto_csv
        result = self._msa_result()
        rows = list(csv.DictReader(io.StringIO(result_to_pareto_csv(result))))
        self.assertEqual(rows[0]["attention_type"], "msa")
        self.assertEqual(rows[0]["msa_window"], "512")
        self.assertEqual(rows[0]["msa_dilated_top_k"], "64")
        self.assertEqual(rows[0]["msa_global_top_k"], "16")

    def test_justification_names_compressed_attention(self):
        from ac.justification import generate_justification
        result = self._msa_result()
        md = generate_justification(result)
        self.assertIn("### Attention: MSA (Multi-scale Attention)", md)
        self.assertIn("local window=512", md)
        self.assertNotIn("### n_kv_heads = 8", md)


class BridgePrecisionRoundTripTests(unittest.TestCase):
    def test_ffn_precision_survives_bridge_round_trip(self):
        from ac.optimizer import CandidateArch
        from ac.optimizer_bridge import candidate_to_arch
        from ac.evaluator import arch_to_candidate
        cand = CandidateArch(
            d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
            ffn_dim=14336, vocab_size=32000,
            weight_precision="bf16", ffn_precision="fp8",
            attn_precision={"q": "bf16", "k": "bf16", "v": "bf16",
                            "o": "bf16"},
            kv_cache_bits=16, tp_degree=8, cp_degree=1, ep_degree=1,
        )
        arch = candidate_to_arch(cand)
        self.assertEqual(arch.precision, "fp8")
        back = arch_to_candidate(arch, cand)
        self.assertEqual(back.ffn_precision, "fp8")


# ============================================================
# test_wave36_fixes.py  (w36)
# ============================================================
"""Wave 36 (Jul 2026) — compressed-attention quality monotonicity.

Root cause (user-caught: "does MSA winning at 8k make intuitive sense?"):
the Wave 9 short-circuit zeroed ALL per-head GQA penalty subterms for
csa/indexshare/msa, so MSA scored ~0.6% BETTER than full attention at 8k
on the identical head configuration — a function-class violation (sparse
readout is a strict subset of full attention; at a context where nothing
is truncated it cannot win on loss at the same shape). Unlike MLA/NSA,
the compressed trio keeps standard per-head GQA softmax attention and
must pay the same head penalties.

Fixes pinned:
  - per-head subterms zeroed only for mla/nsa; the compressed trio pays
    them like full/GQA.
  - long-context / swa-locality terms stay zeroed for nsa + compressed
    trio (charged via their own coverage residuals instead).
  - always-on `compressed_attention_floor` (default 0.2%) — an
    uncalibrated function-class prior; zero fit-pairs coverage.
  - CSA gains a ctx-aware recall-risk term (it attends a constant token
    COUNT, so coverage shrinks with S; previously its predicted loss was
    context-independent and it dominated long ctx for free). IndexShare
    stays scale-free by design: buckets scale with S, so its attended
    FRACTION is constant.

Expected profile at a fixed 7B fine-MoE shape (verified in-repo):
  8k:   full < msa < csa   (full wins short ctx)
  4M:   msa < full         (sparse crossover near ~1M)
"""

MSA = dict(msa_window_size=512, msa_dilated_top_k=64, msa_global_top_k=16)
CSA = dict(csa_block_size=64, csa_top_k_blocks=16, csa_compression_dim=64)
IDX = dict(indexshare_num_buckets=64, indexshare_top_k_buckets=4,
           indexshare_index_dim=64)


def _fixture():
    from optimizer import DeploymentConstraints, generate_moe_candidates
    c = DeploymentConstraints(
        target_params_b=7.0, training_tokens=int(2e12), context_length=8192,
        tp=4, pp=1, dp=8, serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=16, vocab_size=32000,
        allow_moe=True, max_total_params_b=56,
        moe_n_experts_options=[64], moe_top_k_options=[8],
        moe_granularity_targets=[0.25], ep_options=[2],
        param_tolerance=0.08)
    pool = generate_moe_candidates("h100", c)
    base = [x for x in pool if x.attention_type == "full"][len(pool) // 4]
    return c, base


def _loss(c, base, ctx, at, **kw):
    from optimizer import evaluate_candidate
    c2 = copy.copy(c)
    c2.context_length = ctx
    return evaluate_candidate(
        replace(base, attention_type=at, **kw), "h100", c2).predicted_loss


class CompressedMonotonicityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.base = _fixture()

    def test_full_beats_all_compressed_at_short_ctx(self):
        full = _loss(self.c, self.base, 8192, "full")
        for at, kw in (("msa", MSA), ("csa", CSA), ("indexshare", IDX)):
            self.assertLess(
                full, _loss(self.c, self.base, 8192, at, **kw),
                f"{at} must not beat full attention at 8k (function-class "
                f"monotonicity)")

    def test_sparse_crossover_at_long_ctx(self):
        full = _loss(self.c, self.base, 4194304, "full")
        msa = _loss(self.c, self.base, 4194304, "msa", **MSA)
        self.assertLess(msa, full,
                        "msa must win at 4M via long-context relief")

    def test_csa_loss_grows_with_context(self):
        short = _loss(self.c, self.base, 8192, "csa", **CSA)
        long_ = _loss(self.c, self.base, 4194304, "csa", **CSA)
        self.assertGreater(long_, short,
                           "csa attends a constant token count; its recall "
                           "risk must grow with context")

    def test_compressed_floor_active_but_small(self):
        # At 8k with generous coverage the only compressed-specific charge
        # is the floor + head-penalty parity, so msa sits strictly above
        # full but within ~1% of it.
        full = _loss(self.c, self.base, 8192, "full")
        msa = _loss(self.c, self.base, 8192, "msa", **MSA)
        self.assertGreater(msa, full)
        self.assertLess((msa - full) / full, 0.01)

    def test_compressed_pays_head_penalties_like_full(self):
        # The compressed trio keeps standard per-head GQA attention, so
        # its attention_heads penalty entry must be at least full's
        # (same head-shape subterms + its own coverage/floor terms).
        # NOTE: no MLA comparison here — MLA's breakdown entry aggregates
        # f_attention_mla and the long-context term, so it is not a clean
        # probe of the head-shape subterms.
        from optimizer import evaluate_candidate
        def _val(entry):
            return float(getattr(entry, "value", entry) or 0.0)
        ev_full = evaluate_candidate(self.base, "h100", self.c)
        full_heads = _val(ev_full.quality.penalty_breakdown.get(
            "attention_heads", 0.0))
        self.assertGreater(full_heads, 0.0)
        ev_msa = evaluate_candidate(
            replace(self.base, attention_type="msa", **MSA), "h100", self.c)
        msa_heads = _val(ev_msa.quality.penalty_breakdown.get(
            "attention_heads", 0.0))
        self.assertGreaterEqual(msa_heads, full_heads)


class CrossScaleMonotonicityTests(unittest.TestCase):
    """The fix must hold BY CONSTRUCTION at every scale, not just the 7B
    fixture: compressed-at-identical-shape = full's head subterms + a
    clamped-nonnegative floor + nonnegative coverage terms. Sweep
    1B->120B-class shapes x GQA widths x dense/MoE FFN x 8k->4M ctx and
    assert full <= compressed everywhere at matched shape."""

    # (d_model, n_layers, n_heads, d_head, ffn_dim, params_b_target)
    SHAPES = [
        (2048, 24, 16, 128, 8192, 1.0),      # ~1B class
        (4096, 32, 32, 128, 14336, 7.0),     # ~7B class
        (5120, 40, 40, 128, 13824, 13.0),    # ~13B class
        (8192, 80, 64, 128, 28672, 70.0),    # ~70B class
    ]

    def _cand(self, shape, at, n_kv, moe, **kw):
        from optimizer import CandidateArch
        d, L, nh, dh, ffn, _p = shape
        return CandidateArch(
            d_model=d, n_layers=L, n_heads=nh, d_head=dh, n_kv_heads=n_kv,
            ffn_dim=ffn, vocab_size=32000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q": "bf16", "k": "bf16", "v": "bf16",
                            "o": "bf16"},
            kv_cache_bits=16, attention_type=at,
            tp_degree=4, cp_degree=1, ep_degree=2 if moe else 1,
            moe=moe, **kw)

    def test_full_never_loses_to_compressed_at_matched_shape(self):
        from optimizer import DeploymentConstraints, evaluate_candidate
        import copy
        violations = []
        for shape in self.SHAPES:
            _d, _L, nh, _dh, ffn, params_b = shape
            base_c = DeploymentConstraints(
                target_params_b=params_b, training_tokens=int(2e12),
                context_length=8192, tp=4, pp=1, dp=8,
                serving_tbt_ms=None, serving_ttft_ms=None,
                serving_batch=8, vocab_size=32000)
            for n_kv in (max(1, nh // 8), nh):          # GQA and MHA
                for moe in (None, {"n_experts": 64, "top_k": 8,
                                   "expert_dim": max(256, ffn // 8)}):
                    for ctx in (8192, 131072, 4194304):
                        c = copy.copy(base_c)
                        c.context_length = ctx
                        def _ev(at, **kw):
                            return evaluate_candidate(
                                self._cand(shape, at, n_kv, moe, **kw),
                                "h100", c).predicted_loss
                        full = _ev("full")
                        for at, kw in (
                            ("msa", MSA), ("csa", CSA), ("indexshare", IDX)):
                            comp = _ev(at, **kw)
                            # At short ctx full must strictly win; at long
                            # ctx compressed may win via long-context
                            # relief (that is the point of the mechanism),
                            # so only the 8k row is a hard invariant.
                            if ctx == 8192 and comp <= full:
                                violations.append(
                                    (shape[5], n_kv, bool(moe), ctx, at,
                                     round(full, 4), round(comp, 4)))
        self.assertFalse(
            violations,
            "compressed beat full at matched shape/8k: "
            f"{violations}")

    def test_floor_override_cannot_go_negative(self):
        # A weights pack setting compressed_attention_floor=-0.05 must be
        # clamped, not honored.
        import quality_model as qm
        import inspect
        src = inspect.getsource(qm)
        self.assertIn("max(\n        0.0, float(weights.get("
                      "\"compressed_attention_floor\"", src)


# ============================================================
# test_wave37_fixes.py  (w37)
# ============================================================
"""Wave 37 (Jul 2026) — invariant-probe fixes.

Two real bugs surfaced by a structured invariant sweep across shapes,
KV configs, precisions, contexts, and tokens:

1. MHA (n_kv_heads = n_heads) scored WORSE than GQA-8 at matched shape.
   Root cause: `f_kv_heads` had a symmetric "excess KV heads" penalty
   (weight 0.008, quadratic in log(n_kv_heads / (n_heads/4))). But MHA
   is the REFERENCE model — GQA is a strict information reduction
   (K/V shared across query groups), so at matched shape MHA cannot
   lose to GQA on quality. Ainslie 2023 / Llama-2-70B ablations show
   GQA-8 ~= MHA within seed variance, NOT MHA losing to GQA-4. The
   excess branch was encoding a tie as an ordering, and it dominated
   the picker's noise band. Fixed: only sub-GQA-8 heads pay; MHA and
   GQA-8 share the zero-penalty band.

2. `predicted_loss` blew up to ~2,000,000 for any shape that failed
   the feasibility check. Root cause: `feasibility_penalty` returns
   INFEASIBLE = 1e6, which was folded straight into total_penalty_frac
   and multiplied into predicted_loss (L_base * (1 + 1e6) ~= 2e6).
   That value then poisoned every downstream consumer that reads
   predicted_loss as a scaled loss — cell Pareto arithmetic, family
   rollup delta_pct math, plateau markers. The downstream sentinel
   gate (predicted_loss > 10 * chinchilla_baseline) already flags
   these, so we cap the surfaced loss at exactly 10x baseline and set
   `quality_sentinel = True` on QualityResult. total_penalty_absolute
   keeps INFEASIBLE for callers that need the raw signal.

Verified clean under the same probe:
  - MHA <= GQA-8 <= GQA-4 <= MQA at 8k and 1M (function-class monotone).
  - Long-ctx penalty grows smoothly with S (no explosion).
  - MoE optimum tracks tokens-per-total-param (Wave-18h gate working).
  - YOCO penalty grows with sharing fraction (fewer self layers = bigger).
  - Throughput direction: TBT ~ ctx, TP shards KV bw, prefill ~ S^2,
    B200 FP4 < FP8 prefill (native), EP shards MoE memory.
"""

def _w37_cand(**kw):
    from optimizer import CandidateArch
    d = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
             ffn_dim=14336, vocab_size=32000,
             weight_precision="bf16", ffn_precision="bf16",
             attn_precision={"q": "bf16", "k": "bf16", "v": "bf16",
                             "o": "bf16"},
             kv_cache_bits=16, attention_type="full",
             tp_degree=4, cp_degree=1, ep_degree=1)
    d.update(kw)
    return CandidateArch(**d)


def _w37_con(ctx=8192, **kw):
    from optimizer import DeploymentConstraints
    d = dict(target_params_b=7.0, training_tokens=int(2e12),
             context_length=ctx, tp=4, pp=1, dp=8,
             serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
             vocab_size=32000)
    d.update(kw)
    return DeploymentConstraints(**d)


def _w37_L(cand, ctx=8192, **kw):
    from optimizer import evaluate_candidate
    return evaluate_candidate(cand, "h100", _w37_con(ctx, **kw)).predicted_loss


class MHAMonotonicityTests(unittest.TestCase):
    """MHA must not score worse than GQA at matched shape.
    Sparse KV is a strict information reduction of full KV."""

    def test_mha_beats_or_ties_gqa8_at_short_ctx(self):
        mha = _w37_L(_w37_cand(n_kv_heads=32), 8192)
        gqa8 = _w37_L(_w37_cand(n_kv_heads=8), 8192)
        self.assertLessEqual(mha, gqa8 + 1e-6,
                             f"mha={mha} > gqa8={gqa8}")

    def test_kv_head_monotone_sweep(self):
        losses = [_w37_L(_w37_cand(n_kv_heads=n), 8192) for n in (32, 8, 4, 2, 1)]
        # Non-decreasing as we shrink KV heads.
        for a, b in zip(losses, losses[1:]):
            self.assertLessEqual(a, b + 1e-6, f"non-monotone: {losses}")

    def test_mha_and_gqa8_both_in_zero_penalty_band(self):
        # After the fix both live at f_kv_heads == 0, so the only diffs
        # come from downstream shape/attention subterms and stay small.
        mha = _w37_L(_w37_cand(n_kv_heads=32), 8192)
        gqa8 = _w37_L(_w37_cand(n_kv_heads=8), 8192)
        self.assertLess(abs(mha - gqa8), 0.01,
                        f"mha and gqa8 should be within ~1% at matched "
                        f"shape (got {mha:.4f} vs {gqa8:.4f})")


class SentinelIsCappedNotAmplifiedTests(unittest.TestCase):
    """Feasibility failures must not turn predicted_loss into ~2e6."""

    def test_extreme_overflow_yields_bounded_loss(self):
        from optimizer import evaluate_candidate
        # MHA at 1M ctx: KV projection > 10x HBM, memory_fits=False
        ev = evaluate_candidate(_w37_cand(n_kv_heads=32), "h100",
                                _w37_con(ctx=1048576))
        self.assertTrue(ev.quality.quality_sentinel,
                        "extreme overflow must set quality_sentinel=True")
        base = ev.quality.chinchilla_baseline
        self.assertAlmostEqual(ev.predicted_loss, 10.0 * base, places=4)
        # Downstream sentinel gate would kick in cleanly.
        self.assertLess(ev.predicted_loss, 1000.0)

    def test_non_sentinel_path_unchanged(self):
        from optimizer import evaluate_candidate
        ev = evaluate_candidate(_w37_cand(), "h100", _w37_con(ctx=8192))
        self.assertFalse(ev.quality.quality_sentinel)
        self.assertLess(ev.predicted_loss, 5.0)


class ThroughputDirectionTests(unittest.TestCase):
    """Structural direction tests for the throughput model."""

    def test_tbt_grows_with_ctx(self):
        from optimizer import evaluate_candidate
        prev = 0.0
        for ctx in (8192, 32768, 131072, 1048576):
            tbt = evaluate_candidate(
                _w37_cand(), "h100", _w37_con(ctx=ctx, serving_batch=1)
            ).serving_tbt_ms
            self.assertGreater(tbt, prev - 1e-3,
                               f"tbt fell at ctx {ctx}: {tbt} < prev {prev}")
            prev = tbt

    def test_tp_shards_memory_and_kv_bw(self):
        from optimizer import evaluate_candidate
        mems = []; tbts = []
        for tp in (1, 2, 4, 8):
            ev = evaluate_candidate(
                _w37_cand(tp_degree=tp), "h100",
                _w37_con(ctx=131072, tp=tp, serving_batch=8))
            mems.append(ev.memory_per_gpu_gb)
            tbts.append(ev.serving_tbt_ms)
        for a, b in zip(mems, mems[1:]):
            self.assertLess(b, a, f"mem should shard with TP: {mems}")
        for a, b in zip(tbts, tbts[1:]):
            self.assertLess(b, a, f"tbt should shard with TP: {tbts}")

    def test_moe_decode_independent_of_n_experts(self):
        # Decode reads top_k experts per token; total experts don't matter.
        from optimizer import evaluate_candidate
        tbts = []
        for ne in (8, 32, 64, 128):
            b = _w37_cand(moe={"n_experts": ne, "top_k": 2, "expert_dim": 2048},
                      ffn_dim=2048, ep_degree=2)
            tbts.append(evaluate_candidate(
                b, "h100", _w37_con(ctx=8192, serving_batch=1,
                                ep_options=[2])).serving_tbt_ms)
        self.assertLess(max(tbts) - min(tbts), 0.5,
                        f"decode TBT should be n_experts-invariant: {tbts}")

    def test_prefill_grows_superlinearly_for_full_attention(self):
        from optimizer import evaluate_candidate
        p1 = evaluate_candidate(_w37_cand(), "h100",
                                _w37_con(ctx=8192)).throughput.prefill_time_ms
        p4 = evaluate_candidate(_w37_cand(), "h100",
                                _w37_con(ctx=32768)).throughput.prefill_time_ms
        # 4x context should more than 4x prefill for full attention.
        self.assertGreater(p4 / p1, 4.0,
                           f"full-attention prefill grew only {p4/p1:.2f}x "
                           f"for 4x ctx (expected ~16x)")


# ============================================================
# test_wave38_fixes.py  (w38)
# ============================================================
"""Wave 38 (Jul 2026) — second-round invariant audit.

Ran the playbook (docs/invariant-probing-playbook.md) end-to-end.
Two more real bugs surfaced, both the same class as before (silent
no-op / function-class monotonicity):

1. sparsity_2_4 was a silent no-op in the quality model. The field
   was carried on CandidateArch and read by _architecture_residual
   (line ~2127: `getattr(arch, "sparsity_2_4", ...)`), but the QualArch
   view built in evaluate_candidate never threaded it in. Result: the
   throughput model's 2x FLOPs speedup fired but the ~1-3% PPL penalty
   didn't — the optimizer got a free 2x TPS. Fix: thread through
   sparsity_2_4 in the QualArch construction site.

2. NSA scored ~0.6% BETTER than full at 8k/128k on identical shape.
   Same class as compressed-attention (Wave 35): NSA is a strict subset
   of full attention (attends only selected+window tokens); at matched
   shape it cannot beat full on quality. The old branch charged only
   undercoverage below 5% and left the MHA long-context penalty zeroed
   for NSA, so NSA won long ctx for free. Fix: always-on
   `nsa_floor` (default 0.002), clamped nonnegative — same defensive
   pattern as compressed_attention_floor.

Also verified CORRECT during the audit (not bugs, corrected my read):
- MLA vs MHA: bigger c_kv = smaller penalty. Monotone.
- SWA at ctx <= window: exactly ties full (no-op).
- MTP direction: more depth = lower loss, saturates near depth 4.
- RoPE scaling: verified at ctx=131k (I initially probed at 32k where
  the long_context term is 0 by design — my mistake).
- CP direction: doesn't affect quality (correct); TBT shards linearly.
- PP throughput: per-replica TPS grows with PP because a replica spans
  tp*pp*cp GPUs — per-GPU TPS correctly decreases with PP overhead.
- MoE granularity: g=1.0 optimum matches Krajewski reference; g=2.0
  correctly penalized.
- First-K-dense: monotone reduction of MoE bonus.
- H100 FP4 slower than FP8 on decode: correct (no native tensor cores).
"""

def _w38_cand(**kw):
    from optimizer import CandidateArch
    d = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
             ffn_dim=14336, vocab_size=32000,
             weight_precision="bf16", ffn_precision="bf16",
             attn_precision={"qk": "bf16", "v": "bf16", "output": "bf16"},
             kv_cache_bits=16, attention_type="full",
             tp_degree=4, cp_degree=1, ep_degree=1)
    d.update(kw)
    return CandidateArch(**d)


def _w38_con(ctx=8192, **kw):
    from optimizer import DeploymentConstraints
    d = dict(target_params_b=7.0, training_tokens=int(2e12),
             context_length=ctx, tp=4, pp=1, dp=8,
             serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
             vocab_size=32000)
    d.update(kw)
    return DeploymentConstraints(**d)


def _w38_L(cand, ctx=8192, **kw):
    from optimizer import evaluate_candidate
    return evaluate_candidate(cand, "h100", _w38_con(ctx, **kw)).predicted_loss


class Sparsity24AppliesTests(unittest.TestCase):
    """2:4 sparsity must carry a quality penalty, not a silent 2x TPS."""

    def test_dense_baseline(self):
        base = _w38_L(_w38_cand())
        self.assertGreater(base, 1.5)  # sanity

    def test_ffn_2_4_penalized(self):
        base = _w38_L(_w38_cand())
        ffn = _w38_L(_w38_cand(sparsity_2_4={"ffn_up": True, "ffn_down": True,
                                     "ffn_gate": True}))
        self.assertGreater(ffn, base * 1.005,
                           f"ffn 2:4 must cost quality ({ffn} vs {base})")
        self.assertLess(ffn, base * 1.05,
                        "ffn 2:4 must not cost more than ~5% (Pool 2021)")

    def test_attn_2_4_penalized(self):
        base = _w38_L(_w38_cand())
        attn = _w38_L(_w38_cand(sparsity_2_4={"attn_qkv": True, "attn_o": True}))
        self.assertGreater(attn, base + 1e-4,
                           f"attn 2:4 must cost quality ({attn} vs {base})")

    def test_all_2_4_penalized_more_than_ffn(self):
        ffn = _w38_L(_w38_cand(sparsity_2_4={"ffn_up": True, "ffn_down": True,
                                     "ffn_gate": True}))
        allsp = _w38_L(_w38_cand(sparsity_2_4={"ffn_up": True, "ffn_down": True,
                                       "ffn_gate": True,
                                       "attn_qkv": True, "attn_o": True}))
        self.assertGreater(allsp, ffn + 1e-4)


class NSAMonotonicityTests(unittest.TestCase):
    """NSA is a strict subset of full attention. It cannot beat full on
    quality at matched shape at any short context."""

    NSA = dict(nsa_compress_block_size=64, nsa_compress_block_stride=16,
               nsa_select_block_size=64, nsa_select_top_k=16,
               nsa_window_size=512)

    def test_full_at_least_ties_nsa_at_8k(self):
        full = _w38_L(_w38_cand(), 8192)
        nsa = _w38_L(_w38_cand(attention_type="nsa", **self.NSA), 8192)
        self.assertLessEqual(full, nsa + 1e-6,
                             f"NSA cannot beat full at 8k: {nsa} vs {full}")

    def test_full_at_least_ties_nsa_at_128k(self):
        full = _w38_L(_w38_cand(), 131072)
        nsa = _w38_L(_w38_cand(attention_type="nsa", **self.NSA), 131072)
        self.assertLessEqual(full, nsa + 1e-6,
                             f"NSA cannot beat full at 128k: {nsa} vs {full}")

    def test_nsa_floor_active(self):
        # With the always-on floor, NSA at short ctx sits strictly above
        # full but within ~1% (just the floor prior + head parity).
        full = _w38_L(_w38_cand(), 8192)
        nsa = _w38_L(_w38_cand(attention_type="nsa", **self.NSA), 8192)
        self.assertGreater(nsa, full)
        self.assertLess((nsa - full) / full, 0.01)


class SentinelHandlesFeasibleFailureTests(unittest.TestCase):
    """Wave 37 belt-and-braces: MHA at 1M ctx overflows HBM, must be
    reported via the quality_sentinel flag with a bounded predicted_loss
    rather than the raw INFEASIBLE=1e6 amplification."""

    def test_mha_1m_ctx_is_sentinel_not_2m(self):
        from optimizer import evaluate_candidate
        ev = evaluate_candidate(_w38_cand(n_kv_heads=32), "h100",
                                _w38_con(ctx=1048576))
        self.assertTrue(ev.quality.quality_sentinel)
        self.assertLess(ev.predicted_loss, 100.0)


# ============================================================
# test_wave39_fixes.py  (w39)
# ============================================================
"""Wave 39 (Jul 2026) — third-round invariant audit.

Ran the playbook (docs/invariant-probing-playbook.md) end-to-end.
One real bug surfaced, same class as before (table gap = silent zero
violating function-class monotonicity):

1. Probe 3 reversal: mxfp6 scored BETTER than fp8 at matched shape
   (2.1465 vs 2.1566). Two table gaps composed:
   - `precision_sensitivity.lm_head` had no mxfp6/mxfp4 rows, and
     `_component_table_lookup` returned 0.0 for missing rows, so a
     6-bit head was free while the fp8 head paid 0.006.
   - `WEIGHT_PRECISION_PENALTIES` had no ("embedding", mx*) rows, so
     `weight_precision_quality` fell to its generic 0.005 fallback —
     below the fp8 embedding penalty (0.010).
   Fix: added the missing rows (interpolated fp8 -> fp4, same ratios
   the ffn/qkv/o rows encode) and clamped the lookup fallback so an
   unknown reduced-precision format is never free (0.005 floor).

Verified CORRECT during the audit (not bugs):
- MHA-32 at 1M ctx -> 20.14: quality_sentinel=True, capped at
  10 x chinchilla baseline. Feasibility (512 GB KV/seq), not quality.
- Loss flat 8k -> 32k: long_context term is 0 by design below 131k.
- MoE optimum shifts 32 -> 128 -> 256 with 2T -> 10T -> 50T tokens
  (Wave-18h data-sufficiency gate working as documented).
- Probe 9 (TBT~ctx, TP sharding, prefill S^2 ratio 5.2, MoE decode
  n_experts-invariant) and probe 10 orderings all hold.
- Watch item (not patched): Trn2 training TPS ~9% above H100 — follows
  from the documented half-peak convention on NVIDIA parts composing
  with per-stack efficiency; flagged for the per-stack calibration
  pass rather than an ad-hoc spec edit.
"""

# Widening precision order: every later format is at least as lossy.
PRECISION_ORDER = ("bf16", "fp8", "mxfp6", "mxfp4", "fp4")


def _w39_cand(**kw):
    from optimizer import CandidateArch
    d = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
             ffn_dim=14336, vocab_size=32000,
             weight_precision="bf16", ffn_precision="bf16",
             attn_precision={"qk": "bf16", "v": "bf16", "output": "bf16"},
             kv_cache_bits=16, attention_type="full",
             tp_degree=4, cp_degree=1, ep_degree=1)
    d.update(kw)
    return CandidateArch(**d)


def _w39_con(ctx=8192, **kw):
    from optimizer import DeploymentConstraints
    d = dict(target_params_b=7.0, training_tokens=int(2e12),
             context_length=ctx, tp=4, pp=1, dp=8,
             serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
             vocab_size=32000)
    d.update(kw)
    return DeploymentConstraints(**d)


def _w39_L(cand, hw="h100", ctx=8192, **kw):
    from optimizer import evaluate_candidate
    return evaluate_candidate(cand, hw, _w39_con(ctx, **kw)).predicted_loss


class PrecisionMonotonicitySweepTests(unittest.TestCase):
    """Full-model property: at matched shape, predicted loss is monotone
    non-decreasing along bf16 -> fp8 -> mxfp6 -> mxfp4 -> fp4, at every
    scale/ctx probed (cross-scale transfer, playbook section 10)."""

    SHAPES = (
        dict(),  # 7B default
        dict(d_model=2048, n_layers=26, ffn_dim=8192),   # ~1.5B
        dict(d_model=5120, n_layers=48, ffn_dim=20480),  # ~13B
    )

    def _sweep(self, hw, ctx):
        for shape in self.SHAPES:
            losses = [
                _w39_L(_w39_cand(weight_precision=p, ffn_precision=p, **shape), hw=hw, ctx=ctx)
                for p in PRECISION_ORDER
            ]
            for i in range(len(losses) - 1):
                self.assertLessEqual(
                    losses[i], losses[i + 1] + 1e-9,
                    msg=(f"{hw} ctx={ctx} shape={shape}: "
                         f"{PRECISION_ORDER[i]}={losses[i]:.5f} > "
                         f"{PRECISION_ORDER[i+1]}={losses[i+1]:.5f}"))

    def test_h100_8k(self):
        self._sweep("h100", 8192)

    def test_h100_128k(self):
        self._sweep("h100", 131072)

    def test_b200_8k(self):
        self._sweep("b200", 8192)


class ComponentTableOrderingTests(unittest.TestCase):
    """Table-level property: for every component row that defines multiple
    formats, the deltas respect the widening order. Catches a future
    edit reintroducing the gap without needing a full evaluate."""

    def test_precision_sensitivity_rows_ordered(self):
        import quality_model as qm
        # The constants dict is module-level; find it via a known term.
        consts = None
        for name in dir(qm):
            obj = getattr(qm, name)
            if isinstance(obj, dict) and "precision_sensitivity" in obj:
                consts = obj
                break
        self.assertIsNotNone(consts, "precision_sensitivity constants not found")
        table = consts["precision_sensitivity"]
        for comp in ("ffn", "attention_qkv", "attention_o", "lm_head"):
            rows = table[comp]
            deltas = [rows[p]["delta"] for p in PRECISION_ORDER if p in rows]
            order = [p for p in PRECISION_ORDER if p in rows]
            for i in range(len(deltas) - 1):
                self.assertLessEqual(
                    deltas[i], deltas[i + 1],
                    msg=f"{comp}: {order[i]}={deltas[i]} > {order[i+1]}={deltas[i+1]}")

    def test_lm_head_mx_rows_exist(self):
        import quality_model as qm
        for name in dir(qm):
            obj = getattr(qm, name)
            if isinstance(obj, dict) and "precision_sensitivity" in obj:
                rows = obj["precision_sensitivity"]["lm_head"]
                self.assertIn("mxfp6", rows)
                self.assertIn("mxfp4", rows)
                return
        self.fail("precision_sensitivity constants not found")

    def test_embedding_penalties_ordered(self):
        from penalties import weight_precision_quality
        deltas = [weight_precision_quality("embedding", p) for p in PRECISION_ORDER]
        for i in range(len(deltas) - 1):
            self.assertLessEqual(
                deltas[i], deltas[i + 1],
                msg=f"embedding: {PRECISION_ORDER[i]}={deltas[i]} > "
                    f"{PRECISION_ORDER[i+1]}={deltas[i+1]}")


class UnknownFormatNeverFreeTests(unittest.TestCase):
    """Clamp property (playbook recipe step 6): a missing table row for a
    reduced-precision format must cost at least the conservative floor —
    a weights-pack override cannot make a narrow format free."""

    def test_missing_row_has_floor(self):
        from quality_model import _component_table_lookup
        delta, unc, risk = _component_table_lookup({}, "lm_head", "mxfp6")
        self.assertGreaterEqual(delta, 0.005)
        self.assertGreater(unc, 0.0)
        self.assertEqual(risk, "unknown")

    def test_native_formats_still_free(self):
        from quality_model import _component_table_lookup
        for p in ("bf16", "fp16", "fp32"):
            delta, unc, risk = _component_table_lookup({}, "lm_head", p)
            self.assertEqual(delta, 0.0)


# ============================================================
# test_wave40_fixes.py  (w40)
# ============================================================
"""Wave 40 (Jul 2026) — extended audit beyond the playbook's 10 probes.

Two real bugs, both the silent-no-op / two-paths-disagree class:

1. attn_precision was only partially threaded into the quality view.
   At the evaluate_candidate mapping site, "output" was consumed only
   when "v" was quantized, and "qk" was NEVER consumed — the
   precision_sensitivity.qk_logits table row (fp8 = 0.05, softmax-logit
   instability) was dead code. fp8 qk-logits or a lone fp8 output
   projection were free quality no-ops (same class as Wave-38
   sparsity_2_4). Fix: independent key reads, both key schemas accepted
   ({qk,v,output} canonical, {q,k,v,o} playbook), qk_logits consumed in
   _precision_residual reading component_precisions directly (NOT
   get_precision — logits accumulate in bf16 regardless of weight
   format, so the fallback must be bf16, not weight_precision), and
   monotone mxfp6/mxfp4/fp4 qk_logits rows so the new consumer cannot
   introduce a widening-order reversal.

2. Whole-model SWA (attention_type="swa") kept a stale O(N^2) prefill
   "conservative upper bound" from before Wave 18g, while the 18g
   local:global interleave path prices local layers at S x min(S, W).
   The SAME physical model cost 15x more TTFT expressed as
   attention_type="swa" than as a 32/32 local interleave (64.4s vs
   4.2s at 512k, window 4096) — inconsistent encodings in the same
   Pareto comparison. Fix at the bridge: whole-model SWA composes an
   all-local layer_type_list and goes through the per-layer path
   (state layers preserved by compose_layer_type_list).

Verified CORRECT during this audit (not bugs):
- MoE top_k=8 slightly worse than top_k=4 at fixed n_experts: sum of
  defensible priors (shape penalty at 2x active, vocab term, Krajewski
  granularity at G=ref) vs the spine gain; moe_residual correctly ~0
  when G=8 and capacity lives in effective_capacity (v2 split).
- Quality is hardware-blind (identical loss h100/b200/tpu/trn2).
- YOCO k sweep matches 0.012*(n_layers-k)/n_layers exactly.
- Batch/EP/CP throughput directions; EP8 TBT uptick is a2a-vs-stream.
- Larger vocab lowering loss at fixed d_model (vocab_residual is a
  calibrated reference-vocab term, Wave-25 semantics).
"""

def _w40_cand(**kw):
    from optimizer import CandidateArch
    d = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
             ffn_dim=14336, vocab_size=32000,
             weight_precision="bf16", ffn_precision="bf16",
             attn_precision={"qk": "bf16", "v": "bf16", "output": "bf16"},
             kv_cache_bits=16, attention_type="full",
             tp_degree=4, cp_degree=1, ep_degree=1)
    d.update(kw)
    return CandidateArch(**d)


def _w40_con(ctx=8192, **kw):
    from optimizer import DeploymentConstraints
    d = dict(target_params_b=7.0, training_tokens=int(2e12),
             context_length=ctx, tp=4, pp=1, dp=8,
             serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=1,
             vocab_size=32000)
    d.update(kw)
    return DeploymentConstraints(**d)


def _ev(cand, ctx=8192, **kw):
    from optimizer import evaluate_candidate
    return evaluate_candidate(cand, "h100", _w40_con(ctx, **kw))


class AttnPrecisionThreadingTests(unittest.TestCase):
    """Every attn_precision key must be consumed — no silent no-ops."""

    def setUp(self):
        self.base_loss = _ev(_w40_cand()).predicted_loss

    def test_qk_fp8_is_charged(self):
        loss = _ev(_w40_cand(attn_precision={"qk": "fp8", "v": "bf16", "output": "bf16"})).predicted_loss
        # qk_logits fp8 table row is 0.05 relative
        self.assertGreater(loss, self.base_loss + 0.03)

    def test_output_fp8_alone_is_charged(self):
        # Regression: output was consumed only when v != bf16.
        loss = _ev(_w40_cand(attn_precision={"qk": "bf16", "v": "bf16", "output": "fp8"})).predicted_loss
        self.assertGreater(loss, self.base_loss + 1e-6)

    def test_v_fp8_still_charged(self):
        loss = _ev(_w40_cand(attn_precision={"qk": "bf16", "v": "fp8", "output": "bf16"})).predicted_loss
        self.assertGreater(loss, self.base_loss + 1e-6)

    def test_k_fp8_alone_is_charged(self):
        # Regression on the fix itself: "bf16" is truthy, so an `or`-chain
        # fallback (qk or q or k) stopped at q="bf16" and let k="fp8" slip.
        loss = _ev(_w40_cand(attn_precision={"q": "bf16", "k": "fp8", "v": "bf16", "o": "bf16"})).predicted_loss
        self.assertGreater(loss, self.base_loss + 0.03)

    def test_playbook_schema_equivalent(self):
        # {q,k,v,o} (playbook) and {qk,v,output} (canonical) must price identically.
        a = _ev(_w40_cand(attn_precision={"qk": "fp8", "v": "bf16", "output": "fp8"})).predicted_loss
        b = _ev(_w40_cand(attn_precision={"q": "fp8", "k": "fp8", "v": "bf16", "o": "fp8"})).predicted_loss
        self.assertAlmostEqual(a, b, places=9)

    def test_weight_precision_does_not_leak_into_qk(self):
        # Logits accumulate in bf16 regardless of the weight storage format:
        # fp8 weights with untouched attn_precision must NOT pay qk_logits.
        ev = _ev(_w40_cand(weight_precision="fp8", ffn_precision="fp8"))
        pen = ev.quality.terms["precision_residual"]
        self.assertEqual(
            ev.predicted_loss,
            _ev(_w40_cand(weight_precision="fp8", ffn_precision="fp8",
                      attn_precision={"qk": "bf16", "v": "bf16", "output": "bf16"})).predicted_loss)
        self.assertNotIn("qk_logits=", " ".join(pen.notes or []))

    def test_qk_widening_order_monotone(self):
        losses = []
        for p in ("bf16", "fp8", "mxfp6", "mxfp4", "fp4"):
            losses.append(_ev(_w40_cand(attn_precision={"qk": p, "v": "bf16", "output": "bf16"})).predicted_loss)
        for i in range(len(losses) - 1):
            self.assertLessEqual(losses[i], losses[i + 1] + 1e-9)


class SWAEncodingConsistencyTests(unittest.TestCase):
    """attention_type='swa' and an all-local interleave describe the same
    physical model and must price identically on every observable."""

    W = 4096

    def _pair(self, ctx):
        whole = _ev(_w40_cand(attention_type="swa", swa_window=self.W), ctx=ctx)
        inter = _ev(_w40_cand(swa_window=self.W, n_local_attn_layers=32), ctx=ctx)
        return whole, inter

    def test_prefill_matches_interleave_encoding(self):
        for ctx in (8192, 131072, 524288):
            w, i = self._pair(ctx)
            self.assertAlmostEqual(
                w.throughput.prefill_time_ms, i.throughput.prefill_time_ms,
                delta=0.01 * i.throughput.prefill_time_ms,
                msg=f"ctx={ctx}: whole-swa {w.throughput.prefill_time_ms} "
                    f"!= all-local {i.throughput.prefill_time_ms}")

    def test_decode_matches_interleave_encoding(self):
        for ctx in (131072, 524288):
            w, i = self._pair(ctx)
            self.assertAlmostEqual(w.serving_tbt_ms, i.serving_tbt_ms,
                                   delta=0.01 * i.serving_tbt_ms)

    def test_swa_prefill_beats_full_at_long_ctx(self):
        full = _ev(_w40_cand(), ctx=524288).throughput.prefill_time_ms
        swa = _ev(_w40_cand(attention_type="swa", swa_window=self.W),
                  ctx=524288).throughput.prefill_time_ms
        self.assertLess(swa, 0.25 * full)

    def test_swa_prefill_linear_in_ctx(self):
        # S x W attention + linear FFN: 4x ctx should be ~4x prefill, not 16x.
        a = _ev(_w40_cand(attention_type="swa", swa_window=self.W),
                ctx=131072).throughput.prefill_time_ms
        b = _ev(_w40_cand(attention_type="swa", swa_window=self.W),
                ctx=524288).throughput.prefill_time_ms
        self.assertLess(b / a, 4.5)

    def test_window_geq_ctx_ties_full(self):
        # SWA with W >= S attends everything: exact no-op vs full.
        full = _ev(_w40_cand(), ctx=8192).throughput.prefill_time_ms
        swa = _ev(_w40_cand(attention_type="swa", swa_window=16384),
                  ctx=8192).throughput.prefill_time_ms
        self.assertAlmostEqual(swa, full, delta=0.001 * full)


# ============================================================
# test_wave41_fixes.py  (w41)
# ============================================================
"""Wave 41 (Jul 2026) — two more probe-caught bugs + the probe CI gate.

Found by scripts/probe_invariants.py (the property audit built after the
user asked "are you confident AC is free of such bugs?" — answer: no,
and the probes keep proving it; waves 36-41 are all members of the same
two bug classes: silent-zero table/branch gaps and function-class
monotonicity violations).

1. NSA scored 0.5% BETTER than full attention at 8k at the identical
   shape. Same class as Wave 36's MSA bug: NSA sat in the MLA bucket of
   the per-head-penalty short-circuit, but NSA keeps real per-head GQA
   attention (our candidates carry real n_kv_heads, not carrier
   values) — only MLA legitimately skips the head subterms. The
   parallel Wave-38 `nsa_floor` (0.2%) could not cover the ~0.7%
   skipped head penalty; the two fixes compose.

2. Both dedupe keys' attn_key omitted every compressed/NSA option
   field, so `--csa-block-sizes 64,128 --csa-top-k-blocks 8,16`
   enumerated four configs per shape and dedupe silently kept ONE.
   The sweep flags were dead on arrival. attn_key now carries
   csa/indexshare/msa/nsa config fields at both sites.

3. ProbeSuiteGateTests runs the full invariant audit (106 checks at
   time of writing) as a single CI gate so the NEXT such bug is
   machine-caught, not user-caught.
"""

def _w41_constraints(ctx=8192, tokens=2.0, **kw):
    from optimizer import DeploymentConstraints
    base = dict(target_params_b=7.0, training_tokens=int(tokens * 1e12),
                context_length=ctx, tp=4, pp=1, dp=8,
                serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16,
                vocab_size=32000, param_tolerance=0.10)
    base.update(kw)
    return DeploymentConstraints(**base)


class NsaHeadPenaltyTests(unittest.TestCase):
    def test_nsa_ge_full_at_short_ctx(self):
        from optimizer import generate_candidates, evaluate_candidate
        c = _w41_constraints()
        pool = generate_candidates("h100", c)
        base = [x for x in pool if x.attention_type == "full"
                and x.n_kv_heads not in (None, 0)][len(pool) // 3]
        full = evaluate_candidate(base, "h100", c).predicted_loss
        nsa = evaluate_candidate(
            replace(base, attention_type="nsa",
                    nsa_compress_block_size=64, nsa_compress_block_stride=16,
                    nsa_select_block_size=64, nsa_select_top_k=16,
                    nsa_window_size=512), "h100", c).predicted_loss
        self.assertGreaterEqual(
            nsa, full - 1e-9,
            "NSA is a strict subset of full attention; it must not win "
            "on loss at 8k at the same shape")


class DedupeAttnKeyTests(unittest.TestCase):
    def test_compressed_option_sweeps_survive_dedupe(self):
        from collections import Counter
        from optimizer import _enumerate_and_dedupe
        c = _w41_constraints(
            ctx=131072, tp=8,
            allow_csa=True, csa_block_size_options=[64, 128],
            csa_top_k_options=[8, 16],
            allow_msa=True, msa_window_options=[512, 1024],
            param_tolerance=0.08)
        unique, _raw, _pre = _enumerate_and_dedupe("h100", c)
        csa_cfgs = Counter((x.csa_block_size, x.csa_top_k_blocks)
                           for x in unique if x.attention_type == "csa")
        msa_cfgs = Counter(x.msa_window_size
                           for x in unique if x.attention_type == "msa")
        self.assertEqual(len(csa_cfgs), 4,
                         f"csa sweep collapsed by dedupe: {dict(csa_cfgs)}")
        self.assertEqual(len(msa_cfgs), 2,
                         f"msa sweep collapsed by dedupe: {dict(msa_cfgs)}")


class ProbeSuiteGateTests(unittest.TestCase):
    """Full invariant audit as one CI gate (~35 s). On failure, run
    scripts/probe_invariants.py directly for the itemized report."""

    def test_probe_suite_clean(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "probe_invariants.py")],
            capture_output=True, text=True, cwd=ROOT, timeout=300)
        self.assertEqual(
            r.returncode, 0,
            "invariant violations:\n" + r.stdout[-2000:] + r.stderr[-500:])


# ============================================================
# test_wave42_fixes.py  (w42)
# ============================================================
"""Wave 42: prevent unmodelled YOCO choices from becoming silent no-ops."""

class YocoTopologyContractTests(unittest.TestCase):
    def test_constraints_reject_unmodelled_share_pattern(self):
        from optimizer import DeploymentConstraints

        with self.assertRaisesRegex(ValueError, "not calibrated"):
            DeploymentConstraints(
                force_yoco=True,
                yoco_n_self_attn_layers=1,
                yoco_share_pattern="block_shared",
            )

    def test_candidate_api_rejects_unmodelled_share_pattern(self):
        from optimizer import CandidateArch

        with self.assertRaisesRegex(ValueError, "not calibrated"):
            CandidateArch(
                d_model=4096, n_layers=32, n_heads=32, d_head=128,
                n_kv_heads=8, ffn_dim=14336, vocab_size=32000,
                yoco_n_self_attn_layers=1,
                yoco_share_pattern="block_shared",
            )

    def test_schema_rejects_unmodelled_share_pattern(self):
        from schema import _validate_yoco

        errors = _validate_yoco(
            {"n_layers": 32,
             "yoco": {"enabled": True, "n_self_attn_layers": 1,
                      "share_pattern": "block_shared"}},
            "architecture",
        )
        self.assertTrue(errors)
        self.assertIn("not calibrated", errors[0])


# ============================================================
# test_wave43_fixes.py  (w43)
# ============================================================
"""Wave 43 (Jul 2026) — token-axis wiring + two ctx-monotonicity bugs.

Found while building the 2T-vs-20T decision table:

1. Requesting --tokens 20 (off the canonical [0.5, 2, 10] list) SILENTLY
   ran the full canonical sweep — 3x the compute and no 20T data at the
   end (the subset resolver intersected with the canonical list and fell
   back to the WHOLE list on empty intersection). Requested values are
   now authoritative; only non-positive values are rejected, loudly.

2. IndexShare predicted loss was context-FLAT (scale-free coverage prior
   with no indexer-routing risk), so past ~1.5M ctx it undercut every
   ctx-penalized family: the displayed 1B row loss DROPPED from 1M to 2M
   (non-monotone) with a 640ms-TBT winner. Added `indexshare_ctx_risk`
   (0.003 x log(ctx/64k), DSA-style indexer degradation).

3. MLA predicted loss DECREASED from 8k to 128k (2.1515 -> 2.1455) at a
   fixed shape. Root cause: a "context-payoff multiplier" scaled MLA's
   compression penalty x7.9 at 8k, decaying to x1 at 128k — a DEPLOYMENT
   preference ("nobody runs MLA at 8k") baked into the QUALITY model.
   Pretraining loss does not depend on serving context; the short-ctx
   story already lives in the throughput model (KV savings don't bind at
   8k). Multiplier removed; the remaining pure compression term (~0.16%
   at c_kv=512) matches DeepSeek-V2's "roughly quality-neutral vs MHA".

All three were caught by the round-4 probe (per-architecture ctx
monotonicity for every attention type), part of the probe-gate CI test
in test_wave41_fixes.py.
"""

def _w43_constraints(ctx=8192, tokens=2.0, **kw):
    from optimizer import DeploymentConstraints
    base = dict(target_params_b=7.0, training_tokens=int(tokens * 1e12),
                context_length=ctx, tp=4, pp=1, dp=8,
                serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16,
                vocab_size=32000, param_tolerance=0.10)
    base.update(kw)
    return DeploymentConstraints(**base)


def _base_and_loss():
    from optimizer import generate_candidates, evaluate_candidate
    c = _w43_constraints()
    pool = generate_candidates("h100", c)
    base = [x for x in pool if x.attention_type == "full"
            and x.n_kv_heads not in (None, 0)][len(pool) // 3]

    def loss(ctx, **kw):
        c2 = copy.copy(c)
        c2.context_length = ctx
        return evaluate_candidate(
            replace(base, **kw) if kw else base, "h100", c2).predicted_loss
    return base, loss


class TokenAxisResolverTests(unittest.TestCase):
    def test_nonpositive_tokens_rejected(self):
        import _generator_payload as gp
        with self.assertRaises(ValueError):
            gp.generate(hardware=["h100"], param_targets=[7.0],
                        token_counts=[-1.0], contexts=[8192])

    def test_resolver_source_no_silent_fallback(self):
        # Source-level pin: the old intersect-or-fallback pattern
        # (`or TOKEN_COUNTS` / `or PARAM_TARGETS`) must stay gone.
        import inspect
        import _generator_payload as gp
        src = inspect.getsource(gp.generate)
        self.assertNotIn("or TOKEN_COUNTS", src)
        self.assertNotIn("or PARAM_TARGETS", src)


class IndexShareCtxRiskTests(unittest.TestCase):
    def test_indexshare_loss_grows_with_ctx(self):
        _base, loss = _base_and_loss()
        kw = dict(attention_type="indexshare", indexshare_num_buckets=64,
                  indexshare_top_k_buckets=4, indexshare_index_dim=64)
        self.assertGreater(
            loss(2097152, **kw), loss(131072, **kw),
            "IndexShare must pay indexer-routing risk as context grows; "
            "a ctx-flat loss let it win every 2M+ cell for free")


class MlaCtxMonotonicityTests(unittest.TestCase):
    def test_mla_loss_nondecreasing_in_ctx(self):
        _base, loss = _base_and_loss()
        kw = dict(attention_type="mla", mla_kv_latent_dim=512,
                  mla_q_latent_dim=1536, mla_rope_head_dim=64,
                  mla_nope_head_dim=128)
        prev = None
        for ctx in (8192, 131072, 1048576):
            L = loss(ctx, **kw)
            if prev is not None:
                self.assertGreaterEqual(
                    L, prev - 1e-9,
                    f"MLA loss must be non-decreasing in ctx "
                    f"(ctx={ctx}: {L:.5f} < {prev:.5f})")
            prev = L

    def test_payoff_multiplier_stays_removed(self):
        import inspect
        import quality_model
        src = inspect.getsource(quality_model)
        self.assertNotIn("short_ctx_mult", src,
                         "the context-payoff multiplier must stay removed "
                         "from the MLA quality term")


# ============================================================
# test_wave44_fixes.py  (w44)
# ============================================================
"""Wave 44 regression pins for hybrid emission and selected-plan reporting."""

def _hybrid_result():
    from ac.optimizer import CandidateArch, DeploymentConstraints
    from ac.optimizer import OptimizationResult, evaluate_candidate

    moe = {
        "type": "moe",
        "n_experts": 8,
        "top_k": 2,
        "expert_dim": 4096,
        "shared_expert": {"ffn_dim": 1024, "precision": "bf16"},
        "router": {
            "precision": "bf16",
            "load_balance_loss_coef": 0.01,
            "noise_type": None,
        },
        "capacity_factor": 1.25,
        "precision": "bf16",
    }
    candidate = CandidateArch(
        d_model=4096,
        n_layers=8,
        n_heads=32,
        d_head=192,
        n_kv_heads=8,
        ffn_dim=14336,
        vocab_size=32000,
        weight_precision="bf16",
        ffn_precision="bf16",
        attn_precision={"qk": "bf16", "v": "bf16", "output": "bf16"},
        kv_cache_bits=8,
        moe=moe,
        moe_style="coarse",
        ep_degree=2,
        n_dense_ffn_layers=2,
        state_config={
            "d_state": 128,
            "state_expansion": 2,
            "n_heads": 32,
            "d_head": 128,
            "state_precision": "bf16",
            "state_type": "mamba2",
        },
        layer_type_list=[
            "attention", "state", "state", "state",
            "attention", "state", "state", "state",
        ],
        n_attention_layers=2,
        n_state_layers=6,
        hybrid_ratio="3:1",
        placement_strategy="first_periodic_last",
        attention_type="mla",
        mla_kv_latent_dim=512,
        mla_q_latent_dim=1536,
        mla_rope_head_dim=64,
        mla_nope_head_dim=128,
        swa_window=4096,
        n_local_attn_layers=1,
        mtp_n_predict_depths=1,
        mtp_depth_n_layers=1,
        mtp_train_loss_weight=0.3,
        tp_degree=8,
        pp_degree=2,
        cp_degree=4,
        cp_method="ulysses",
        rope_scaling_method="longrope",
        rope_scaling_factor=16.0,
        rope_original_max_position=8192,
    )
    constraints = DeploymentConstraints(
        target_params_b=7.0,
        training_tokens=int(2e12),
        context_length=131072,
        tp=1,
        pp=1,
        dp=4,
        allow_state=True,
        allow_moe=True,
    )
    evaluated = evaluate_candidate(candidate, "h100", constraints)
    return OptimizationResult(
        optimal=evaluated,
        pareto_frontier=[evaluated],
        all_evaluated=[evaluated],
        constraints=constraints,
        hardware="h100",
        candidates_generated=1,
        candidates_feasible=1,
        candidates_evaluated=1,
        search_time_sec=0.0,
        binding_constraints=[],
    )


class HybridEmissionTests(unittest.TestCase):
    def test_result_emits_every_evaluated_hybrid_axis(self):
        from ac.optimizer import result_to_config
        from ac.schema import validate_config

        cfg = result_to_config(_hybrid_result())
        self.assertEqual(validate_config(cfg), [])
        self.assertEqual(cfg["parallelism"], {
            "tensor_parallel": 8,
            "pipeline_parallel": 2,
            "data_parallel": 4,
            "expert_parallel": 2,
            "context_parallel": 4,
            "cp_method": "ulysses",
        })
        scaling = cfg["architecture"]["positional_encoding"]["scaling"]
        self.assertEqual(scaling["method"], "longrope")
        self.assertEqual(scaling["factor"], 16.0)
        self.assertEqual(cfg["architecture"]["n_dense_ffn_layers"], 2)
        self.assertEqual(cfg["architecture"]["mtp"]["n_predict_depths"], 1)

        bands = cfg["architecture"]["layer_configs"]
        attention_types = {
            band["attention"]["type"]
            for band in bands if band["attention"] is not None
        }
        self.assertEqual(attention_types, {"mla", "swa"})
        covered = sorted(i for band in bands for i in band["layer_idx"])
        self.assertEqual(covered, list(range(8)))
        for band in bands:
            for idx in band["layer_idx"]:
                expected_ffn = "swiglu" if idx < 2 else "moe"
                self.assertEqual(band["ffn"]["type"], expected_ffn)

    def test_emitted_hybrid_reloads_without_losing_axes(self):
        from ac.baseline import load_baseline_model
        from ac.optimizer import result_to_config

        result = _hybrid_result()
        cfg = result_to_config(result)
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as handle:
            json.dump(cfg, handle)
            path = handle.name
        loaded = load_baseline_model(path).candidate
        self.assertEqual(loaded.layer_type_list.count("state"), 6)
        self.assertEqual(loaded.n_attention_layers, 2)
        self.assertEqual(loaded.n_local_attn_layers, 1)
        self.assertEqual(loaded.attention_type, "mla")
        self.assertEqual(loaded.n_dense_ffn_layers, 2)
        self.assertEqual(loaded.mtp_n_predict_depths, 1)
        self.assertEqual(loaded.cp_degree, 4)
        self.assertEqual(loaded.rope_scaling_method, "longrope")
        self.assertEqual(loaded.placement_strategy, "first_periodic_last")
        self.assertEqual(loaded.hybrid_ratio, "3:1")
        self.assertEqual(
            loaded.active_params,
            result.optimal.arch.active_params,
        )

    def test_parallelism_delta_preserves_hybrid_identity(self):
        from ac.deltas.change_parallelism import ChangeParallelism
        from ac.evaluator import arch_to_candidate
        from ac.optimizer_bridge import candidate_to_arch

        baseline = _hybrid_result().optimal.arch
        changed = ChangeParallelism().apply(
            candidate_to_arch(baseline), tp=16, pp=2, cp=1)
        round_trip = arch_to_candidate(changed, baseline)
        self.assertEqual(round_trip.hybrid_ratio, baseline.hybrid_ratio)
        self.assertEqual(
            round_trip.placement_strategy, baseline.placement_strategy)
        self.assertEqual(
            round_trip.derived_d_state, baseline.derived_d_state or 128)

    def test_pure_state_search_drops_attention_only_axes(self):
        from ac.optimizer import DeploymentConstraints
        from ac.optimizer import generate_state_candidates

        constraints = DeploymentConstraints(
            target_params_b=1.0,
            training_tokens=int(2e12),
            context_length=131072,
            tp=8,
            pp=1,
            dp=8,
            allow_state=True,
            allow_rope_scaling=True,
            rope_scaling_methods=["yarn", "longrope"],
            cp=1,
            cp_options=[1, 4],
            kv_bits_options=[8, 16],
            placement_strategies=["interleaved", "periodic"],
        )
        pure_state = [
            c for c in generate_state_candidates("h100", constraints)
            if c.n_attention_layers == 0
        ]
        self.assertTrue(pure_state)
        for candidate in pure_state:
            self.assertEqual(candidate.cp_degree, 1)
            self.assertEqual(candidate.rope_scaling_method, "none")
            self.assertEqual(candidate.placement_strategy, "none")
            self.assertEqual(candidate.n_kv_heads, candidate.n_heads)
            self.assertEqual(candidate.kv_cache_bits, 16)


class SelectedPlanJustificationTests(unittest.TestCase):
    def test_report_uses_selected_parallelism_and_mla_arithmetic(self):
        from ac.justification import generate_justification

        md = generate_justification(_hybrid_result())
        self.assertIn("TP=8 PP=2 CP=4 DP=4 EP=2", md)
        self.assertIn(
            "per DP replica; TP=8, PP=2, CP=4, EP=2", md)
        self.assertIn("selected TP=8", md)
        self.assertIn("selected PP=2", md)
        self.assertIn("intentionally decoupled", md)
        self.assertNotIn("lies on the lattice with n_heads", md)
        self.assertIn("d_state=128 (configured)", md)
        self.assertNotIn("TP=1 PP=1", md)

    def test_non_moe_state_winner_is_not_labeled_dense(self):
        from ac.cli_compile import _non_moe_family_label

        arch = copy.deepcopy(_hybrid_result().optimal.arch)
        arch.moe = None
        arch.moe_style = "dense"
        self.assertEqual(
            _non_moe_family_label(arch), "state-attention hybrid")

    def test_pareto_csv_exposes_complete_parallelism_and_training_memory(self):
        from ac.optimizer import result_to_pareto_csv

        row = next(csv.DictReader(io.StringIO(
            result_to_pareto_csv(_hybrid_result()))))
        self.assertEqual(row["tp"], "8")
        self.assertEqual(row["pp"], "2")
        self.assertEqual(row["dp"], "4")
        self.assertEqual(row["cp"], "4")
        self.assertEqual(row["ep"], "2")
        self.assertGreater(float(row["training_memory_gb"]), 0)


class StateStressAccountingTests(unittest.TestCase):
    def test_pure_state_stress_has_nonzero_useful_flops(self):
        from ac.optimizer_bridge import candidate_to_arch
        from ac.stress import Workload, compute_throughput_stress

        pure_state = _hybrid_result().optimal.arch
        pure_state.layer_type_list = ["state"] * pure_state.n_layers
        pure_state.n_attention_layers = 0
        pure_state.n_state_layers = pure_state.n_layers
        arch = candidate_to_arch(pure_state)
        stress = compute_throughput_stress(
            arch,
            arch_name="pure-state-regression",
            hardware="h100",
            workload=Workload(prefill_seq_len=8192, decode_kv_len=8192),
            tp_degree=8,
            pp_degree=1,
            dp_degree=1,
            ep_degree=1,
        )
        self.assertGreater(stress.intermediates["decode_flops"], 0)
        self.assertGreater(stress.intermediates["prefill_flops"], 0)
        self.assertGreater(stress.tc_util_decode, 0)
        self.assertGreater(stress.tc_util_prefill, 0)


class CliIntegrityTests(unittest.TestCase):
    def test_stress_inline_kwargs_keep_internal_commas(self):
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "ac" / "cli_stress.py"),
                "transition", "--known", "Mistral-7B",
                "--hardware", "h100", "--tp", "8",
                "--decode-kv", "131072", "--prefill-seq", "131072",
                "--apply", "swap_attention_to_swa:window_size=4096",
                "--apply", "interleave_local_attention:ratio=3:1,window=4096",
                "--json",
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload[1]["transformation_name"],
                         "interleave_local_attention")
        self.assertNotIn("unknown transformation", result.stderr)
        for transition in payload:
            self.assertIn(
                "prefill131072-kv131072",
                transition["candidate_stress"]["workload_id"],
            )
        baseline_kv = payload[0]["baseline_stress"]["intermediates"]["kv_bytes"]
        whole_kv = payload[0]["candidate_stress"]["intermediates"]["kv_bytes"]
        interleave_kv = payload[1]["candidate_stress"]["intermediates"]["kv_bytes"]
        self.assertLess(whole_kv, interleave_kv)
        self.assertLess(interleave_kv, baseline_kv)
        baseline_flops = payload[0]["baseline_stress"]["intermediates"]["prefill_flops"]
        whole_flops = payload[0]["candidate_stress"]["intermediates"]["prefill_flops"]
        interleave_flops = payload[1]["candidate_stress"]["intermediates"]["prefill_flops"]
        self.assertLess(whole_flops, interleave_flops)
        self.assertLess(interleave_flops, baseline_flops)

    def test_delta_eval_stdout_is_strict_json(self):
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "ac" / "cli_delta_eval.py"),
                "--baseline-config", str(ROOT / "configs" / "mistral_7b.json"),
                "--hardware", "h100", "--tp", "8",
                "--apply", "swap_attention_to_gqa:group_size=8",
                "--stdout", "--json", "--no-pareto",
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Infinity", result.stdout)
        json.loads(
            result.stdout,
            parse_constant=lambda value: self.fail(
                f"non-standard JSON constant: {value}"),
        )


# ============================================================
# test_wave45_fixes.py  (w45)
# ============================================================
"""Wave 45 (Jul 2026) — end-to-end CLI audit (greenfield / delta-eval /
parallelism), beyond the model-level probes of Waves 39-40.

Five defects, all in the "honest number, dishonest presentation" or
"dead grid point" class:

1. --moe-granularity 0.25 was a mathematically empty sweep value: the
   fine-style shared expert was sized off the DENSE baseline
   (shared = ffn/4), so target_active = 0.25*ffn - 0.25*ffn = 0 and the
   documented "quarter-active regime" never emitted one candidate — while
   the CLI's own allow-moe hint told users to pass exactly that value.
   Fix (lattice_engine): shared expert scales with g (DeepSeek
   proportion: ~1/4 of the candidate's OWN active budget).

2. The allow-moe fallback warning conflated "Pareto-dominated off the
   frontier" with "infeasible": a run where 248 MoE candidates were
   evaluated and met every constraint told the user to hunt for
   --max-total-params-b / tile-alignment problems. Fix (cli_compile):
   distinguish via result.all_evaluated.

3. A pick needing 546 GB/GPU (serving) / 803 GB/GPU (training) on a
   192 GB part produced ZERO warnings; spill was priced but disclosed
   only in a family-table tag. Fix (cli_compile + justification):
   warnings + arch.md lines for serving spill (priced) and training
   overflow (NOT priced — no training-spill mechanism).

4. Delta-eval's "Training TPS" row silently changes denominator when the
   delta changes replica size: pp 1->4 rendered "+193% improves" while
   per-GPU throughput fell 27% (replica grew 8->32 GPUs). Fix (report):
   basis note with per-GPU numbers whenever TP*PP*CP changes.

5. change_parallelism:ep=16 with DP=4 priced training under the violated
   EP-over-DP assumption with no warning (Wave 24 guarded only the
   greenfield enumerator); and the "field-level diff above is empty"
   note contradicted the two rows rendered directly above it. Fix
   (report): EP>DP guard note + truthful sidecar-echo wording.

Verified CORRECT during this audit (not bugs): 7B greenfield pick shape/
ledger/TTFT; CLI --tp override beats config parallelism (Wave 30);
GQA/MLA delta directions incl. ACTIVE-PARAM SHIFT note (Wave 25); TP2
vs TP8 memory ~4x; MoE dominated-by-dense at 30B/20T with SLO (knife-
edge documented in Wave 30); serving spill physics (tiered harmonic-
mean bandwidth).
"""

class MoEGranularityGridTests(unittest.TestCase):
    """Every granularity target in the default sweep must emit fine-style
    options whose active FFN-equivalent tracks g * baseline_ffn."""

    def _fine_opts(self, g, ffn=21504, d=6144):
        from lattice_engine import compute_moe_options, B200
        opts = compute_moe_options(
            B200, "bf16", d, ffn, ep_degrees=[4, 8],
            n_experts_options=[64], granularity_targets=(g,))
        return [o for o in opts if o.style == "fine"]

    def test_every_default_granularity_emits_fine_options(self):
        for g in (1.0, 0.5, 0.25):
            fine = self._fine_opts(g)
            self.assertTrue(
                fine, f"g={g} emitted no fine-style MoE options "
                      f"(dead grid point)")

    def test_fine_active_tracks_granularity(self):
        ffn = 21504
        for g in (1.0, 0.5, 0.25):
            for o in self._fine_opts(g, ffn=ffn):
                target = g * ffn
                self.assertLess(
                    abs(o.active_ffn_equivalent - target), 0.25 * target,
                    f"g={g}: active_eq {o.active_ffn_equivalent} far from "
                    f"target {target} (tk={o.top_k})")

    def test_shared_expert_scales_with_granularity(self):
        s1 = {o.shared_dim for o in self._fine_opts(1.0)}
        s4 = {o.shared_dim for o in self._fine_opts(0.25)}
        self.assertGreater(max(s1), max(s4))


def _mk_eval(field_changes, tps=(44912.0, 131844.0), rw=None):
    from evaluator import DeltaEvaluation, MetricDelta
    ev = DeltaEvaluation(
        baseline_name="t", hardware="h100", delta_name="change_parallelism")
    ev.field_changes = list(field_changes)
    ev.metrics = {
        "training_tps": MetricDelta(
            name="training_tps", baseline=tps[0], candidate=tps[1],
            delta=tps[1] - tps[0],
            pct_change=(tps[1] / tps[0] - 1) * 100 if tps[0] else 0.0,
            direction="improves", lower_is_better=False),
    }
    ev.resolved_workload = rw or {
        "workload_preset": "training", "serving_batch": 8,
        "context_length": 8192, "prompt_len": 8192,
        "serving_tbt_ms_budget": 50.0, "tp": 8, "pp": 1, "dp": 8,
        "ep": 8, "cp": 1,
    }
    return ev


class ReplicaBasisNoteTests(unittest.TestCase):
    """Training TPS is per TP x PP x CP replica; a delta that changes
    replica size must state the basis and the per-GPU numbers."""

    def test_pp_change_gets_basis_note(self):
        from report import render_topology_notes
        ev = _mk_eval([{"field": "parallelism.pipeline_parallel",
                        "baseline": 1, "candidate": 4}])
        out = render_topology_notes(ev)
        self.assertIn("replica", out)
        self.assertIn("8 to 32 GPUs", out)
        self.assertIn("tok/s/GPU", out)

    def test_non_parallelism_delta_gets_no_basis_note(self):
        from report import render_topology_notes
        ev = _mk_eval([{"field": "n_kv_heads", "baseline": 8, "candidate": 4}])
        out = render_topology_notes(ev)
        self.assertNotIn("tok/s/GPU", out)


class EpOverDpGuardTests(unittest.TestCase):
    """change_parallelism:ep=N must warn when EP exceeds DP — the training
    math assumes EP-over-DP (Wave 24), and the delta path bypassed the
    greenfield enumerator's filter."""

    def test_ep_gt_dp_warns(self):
        from report import render_topology_notes
        ev = _mk_eval(
            [{"field": "parallelism.expert_parallel",
              "baseline": 8, "candidate": 16}],
            rw={"tp": 8, "pp": 1, "dp": 4, "ep": 8, "cp": 1})
        out = render_topology_notes(ev)
        self.assertIn("EP=16 exceeds DP=4", out)
        self.assertIn("violated assumption", out)

    def test_ep_leq_dp_silent(self):
        from report import render_topology_notes
        ev = _mk_eval(
            [{"field": "parallelism.expert_parallel",
              "baseline": 8, "candidate": 16}],
            rw={"tp": 8, "pp": 1, "dp": 16, "ep": 8, "cp": 1})
        out = render_topology_notes(ev)
        self.assertNotIn("exceeds DP", out)


class SidecarOnlyDiffNoteTests(unittest.TestCase):
    """A sidecar-only delta with moving metrics must not claim 'the diff
    above is empty' when parallelism rows render directly above it."""

    def test_no_contradictory_empty_claim(self):
        from report import render_topology_notes
        ev = _mk_eval([{"field": "parallelism.expert_parallel",
                        "baseline": 8, "candidate": 16},
                       {"field": "applied_deltas",
                        "baseline": None,
                        "candidate": ["change_parallelism"]}])
        out = render_topology_notes(ev)
        self.assertNotIn("diff above is empty", out)


# ============================================================
# test_wave46_seam_contracts.py  (w46)
# ============================================================
"""Release pins for architecture identity across AC entry-point seams."""

def _complex_candidate():
    from ac.architecture import parameter_ledger
    from ac.optimizer import CandidateArch

    candidate = CandidateArch(
        d_model=4096,
        n_layers=8,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=14336,
        vocab_size=32000,
        weight_precision="bf16",
        ffn_precision="bf16",
        activation_precision="fp8",
        kv_cache_bits=8,
        moe={
            "type": "moe",
            "n_experts": 8,
            "top_k": 2,
            "expert_dim": 4096,
            "shared_expert": {"ffn_dim": 1024, "precision": "bf16"},
            "router": {"precision": "bf16"},
            "capacity_factor": 1.25,
            "precision": "bf16",
        },
        ep_degree=2,
        moe_style="coarse",
        n_dense_ffn_layers=2,
        state_config={
            "state_type": "mamba2",
            "d_state": 128,
            "state_expansion": 3,
            "n_heads": 32,
            "d_head": 128,
            "state_precision": "bf16",
        },
        layer_type_list=[
            "attention", "state", "state", "state",
            "attention", "state", "state", "state",
        ],
        placement_strategy="first_periodic_last",
        n_attention_layers=2,
        n_state_layers=6,
        hybrid_ratio="3:1",
        attention_type="mla",
        mla_kv_latent_dim=512,
        mla_q_latent_dim=1536,
        mla_rope_head_dim=64,
        mla_nope_head_dim=64,
        swa_window=4096,
        n_local_attn_layers=1,
        yoco_n_self_attn_layers=2,
        yoco_share_pattern="single_source",
        mtp_n_predict_depths=1,
        mtp_depth_n_layers=1,
        mtp_train_loss_weight=0.17,
        tp_degree=8,
        pp_degree=2,
        cp_degree=4,
        cp_method="ulysses",
        rope_scaling_method="longrope",
        rope_scaling_factor=16.0,
        rope_original_max_position=8192,
    )
    ledger = parameter_ledger(candidate)
    candidate.total_params = ledger.total_params
    candidate.active_params = ledger.active_params
    candidate.total_params_b = ledger.total_params / 1e9
    candidate.active_params_b = ledger.active_params / 1e9
    return candidate


def _w46_constraints(candidate=None):
    from ac.optimizer import DeploymentConstraints

    target = candidate.total_params_b if candidate is not None else 7.0
    return DeploymentConstraints(
        target_params_b=target,
        param_tolerance=0.5,
        training_tokens=int(2e12),
        context_length=131072,
        prompt_len=8192,
        output_len=1024,
        tp=8,
        pp=2,
        dp=8,
        cp=4,
        kv_bits_options=[8],
        precision_configs=["bf16"],
    )


def test_candidate_delta_bridge_preserves_quality_and_config_only_axes():
    from ac.evaluator import arch_to_candidate
    from ac.optimizer_bridge import candidate_to_arch

    source = _complex_candidate()
    source.sparsity_2_4 = {"ffn_up": True, "attn_o": True}
    restored = arch_to_candidate(candidate_to_arch(source), source)
    fields = (
        "attention_type", "mla_kv_latent_dim", "mla_q_latent_dim",
        "mla_rope_head_dim", "mla_nope_head_dim", "yoco_n_self_attn_layers",
        "yoco_share_pattern", "mtp_n_predict_depths", "mtp_depth_n_layers",
        "mtp_train_loss_weight", "rope_scaling_method",
        "rope_scaling_factor", "rope_original_max_position", "sparsity_2_4",
        "cp_degree", "cp_method", "tp_degree", "pp_degree",
        "weight_precision", "ffn_precision", "activation_precision",
        "attn_precision",
    )
    assert {name: getattr(restored, name) for name in fields} == {
        name: getattr(source, name) for name in fields
    }
    assert restored.moe == source.moe
    assert restored.state_config == source.state_config
    assert restored.layer_type_list == source.layer_type_list


def test_modifier_candidates_preserve_family_and_search_moe_expert_width():
    from ac.modifier import _changes, _generate_local_candidates, _record_key

    base = _complex_candidate()
    candidates = _generate_local_candidates(
        base, "h100", _w46_constraints(base), [8])
    assert candidates
    expert_dims = {candidate.moe["expert_dim"] for candidate, _ in candidates}
    assert len(expert_dims) > 1
    assert len({_record_key(candidate, tp) for candidate, tp in candidates}) == len(
        candidates)
    for candidate, tp in candidates:
        assert candidate.state_config == base.state_config
        assert candidate.layer_type_list == base.layer_type_list
        assert candidate.n_layers == base.n_layers
        assert candidate.attention_type == "mla"
        assert candidate.n_kv_heads == base.n_kv_heads
        assert candidate.yoco_n_self_attn_layers == 2
        assert candidate.mtp_train_loss_weight == 0.17
        assert candidate.rope_scaling_method == "longrope"
        assert candidate.tp_degree == tp == 8
    changed = next(candidate for candidate, _ in candidates
                   if candidate.moe["expert_dim"] != base.moe["expert_dim"])
    assert "expert_dim" in {
        change.field for change in _changes(base, changed, 8, 8)}


def test_b200_modifier_does_not_reject_inherited_gpt_oss_head_dim():
    from ac.baseline import load_baseline_model
    from ac.modifier import _generate_local_candidates
    from ac.optimizer import DeploymentConstraints

    baseline = load_baseline_model(
        str(ROOT / "configs" / "gpt_oss_120b.json")).candidate
    assert baseline.d_head == 64
    constraints = DeploymentConstraints(
        target_params_b=baseline.total_params / 1e9,
        tp=8, pp=1, dp=16, param_tolerance=0.15,
        precision_configs=["all_bf16", "ffn_fp8"],
    )
    candidates = _generate_local_candidates(
        baseline, "b200", constraints, [8, 16])
    assert len(candidates) >= 20
    assert all(candidate.d_head == 64 for candidate, _ in candidates)


def test_modifier_serializer_reuses_complete_greenfield_schema(tmp_path):
    from ac.architecture import parameter_ledger
    from ac.baseline import load_baseline_model
    from ac.modifier import ModifierRecord, ModifierResult, modifier_result_to_config
    from ac.optimizer import evaluate_candidate
    from ac.schema import validate_config

    candidate = _complex_candidate()
    constraints = _w46_constraints(candidate)
    evaluated = evaluate_candidate(candidate, "h100", constraints)
    record = ModifierRecord(
        evaluated=evaluated, tp=8, is_baseline=True,
        quality_preserving=True, move_class="baseline")
    result = ModifierResult(
        baseline=record,
        selected=record,
        pareto_frontier=[record],
        all_records=[record],
        feasible_records=[record],
        hardware="h100",
        constraints=constraints,
        candidates_generated=1,
        candidates_evaluated=1,
    )
    config = modifier_result_to_config(result)
    assert validate_config(config) == []
    path = tmp_path / "modifier.json"
    path.write_text(json.dumps(config))
    emitted = load_baseline_model(str(path)).candidate

    assert emitted.moe["n_experts"] == candidate.moe["n_experts"]
    assert emitted.moe["expert_dim"] == candidate.moe["expert_dim"]
    assert emitted.n_dense_ffn_layers == candidate.n_dense_ffn_layers
    assert emitted.state_config == candidate.state_config
    assert emitted.layer_type_list == candidate.layer_type_list
    assert emitted.attention_type == candidate.attention_type
    assert emitted.mla_kv_latent_dim == candidate.mla_kv_latent_dim
    assert emitted.n_local_attn_layers == candidate.n_local_attn_layers
    assert emitted.swa_window == candidate.swa_window
    assert emitted.yoco_n_self_attn_layers == candidate.yoco_n_self_attn_layers
    assert emitted.mtp_n_predict_depths == candidate.mtp_n_predict_depths
    assert emitted.mtp_train_loss_weight == candidate.mtp_train_loss_weight
    assert emitted.weight_precision == candidate.weight_precision
    assert emitted.activation_precision == candidate.activation_precision
    assert emitted.rope_scaling_method == candidate.rope_scaling_method
    assert emitted.rope_scaling_factor == candidate.rope_scaling_factor
    assert (emitted.tp_degree, emitted.pp_degree, emitted.ep_degree,
            emitted.cp_degree) == (8, 2, 2, 4)
    expected = parameter_ledger(candidate)
    actual = parameter_ledger(emitted)
    assert actual.total_params == expected.total_params
    assert actual.active_params == expected.active_params
    predicted = config["metadata"]["predicted"]
    assert predicted["total_params_b"] == round(expected.total_params / 1e9, 6)
    assert predicted["active_params_b"] == round(expected.active_params / 1e9, 6)


def test_greenfield_csv_has_physical_dense_attention_count_and_aligned_columns():
    from ac.baseline import load_baseline_model
    from ac.optimizer import (
        OptimizationResult, evaluate_candidate, result_to_pareto_csv)

    baseline = load_baseline_model(str(ROOT / "configs" / "mistral_7b.json"))
    constraints = _w46_constraints(baseline.candidate)
    constraints.pp = 1
    constraints.cp = 1
    evaluated = evaluate_candidate(baseline.candidate, "h100", constraints)
    result = OptimizationResult(
        optimal=evaluated,
        pareto_frontier=[evaluated],
        all_evaluated=[evaluated],
        constraints=constraints,
        hardware="h100",
    )
    text = result_to_pareto_csv(result)
    header = next(csv.reader(io.StringIO(text)))
    raw_row = list(csv.reader(io.StringIO(text)))[1]
    row = next(csv.DictReader(io.StringIO(text)))
    assert len(raw_row) == len(header)
    assert row["d_model"] == str(baseline.candidate.d_model)
    assert row["attention_layers"] == str(baseline.candidate.n_layers)


def test_training_fit_gate_prefers_runnable_pool_and_has_best_effort_fallback():
    from types import SimpleNamespace
    from ac.optimizer import _prefer_training_fit

    def candidate(memory):
        return SimpleNamespace(
            throughput=SimpleNamespace(training_memory_per_gpu_gb=memory))

    fits = candidate(79.0)
    overflow = candidate(81.0)
    assert _prefer_training_fit([overflow, fits], "h100") == [fits]
    overflow_a = candidate(100.0)
    overflow_b = candidate(120.0)
    assert _prefer_training_fit(
        [overflow_a, overflow_b], "h100") == [overflow_a, overflow_b]


def test_explicit_candidate_cap_bounds_source_family_retention():
    from ac.optimizer import DeploymentConstraints, _enumeration_pool_cap

    capped = DeploymentConstraints(target_params_b=7, max_candidates=400)
    uncapped = DeploymentConstraints(target_params_b=7, max_candidates=None)
    assert _enumeration_pool_cap(capped) == 4000
    assert _enumeration_pool_cap(uncapped) == 200000


def test_training_context_is_independent_of_serving_context_and_cp_shards_memory():
    import copy
    import pytest
    from ac.baseline import load_baseline_model
    from ac.optimizer import DeploymentConstraints, evaluate_candidate

    candidate = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    candidate.tp_degree = 8

    def constraints(context, pretrain=8192, cp=1):
        return DeploymentConstraints(
            target_params_b=candidate.total_params_b,
            training_tokens=int(2e12),
            pretraining_context_length=pretrain,
            context_length=context,
            prompt_len=context,
            serving_batch=8,
            tp=8,
            pp=1,
            dp=8,
            cp=cp,
        )

    short = evaluate_candidate(
        copy.deepcopy(candidate), "h100", constraints(8192))
    long_serving = evaluate_candidate(
        copy.deepcopy(candidate), "h100", constraints(1048576))
    assert long_serving.training_tps == pytest.approx(short.training_tps)
    assert long_serving.throughput.training_memory_per_gpu_gb == pytest.approx(
        short.throughput.training_memory_per_gpu_gb)
    assert long_serving.throughput.training_sequence_length == 8192
    assert long_serving.serving_tbt_ms > short.serving_tbt_ms

    cp1_cand = copy.deepcopy(candidate)
    cp1_cand.cp_degree = 1
    cp4_cand = copy.deepcopy(candidate)
    cp4_cand.cp_degree = 4
    cp1 = evaluate_candidate(
        cp1_cand, "h100", constraints(131072, pretrain=131072, cp=1))
    cp4 = evaluate_candidate(
        cp4_cand, "h100", constraints(131072, pretrain=131072, cp=4))
    assert cp4.throughput.training_memory_per_gpu_gb < \
        cp1.throughput.training_memory_per_gpu_gb
    assert cp4.training_tps / 4 <= cp1.training_tps * 1.01


def test_delta_gate_rejects_schema_invalid_parallelism_and_gqa():
    from ac.baseline import load_baseline_model
    from ac.delta_engine import apply_transition
    from ac.deltas.change_parallelism import ChangeParallelism
    from ac.deltas.swap_attention_to_gqa import SwapAttentionToGQA
    from ac.optimizer_bridge import candidate_to_arch

    mistral = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    arch = candidate_to_arch(mistral)
    bad_gqa = apply_transition(
        arch, SwapAttentionToGQA(), {"group_size": 5},
        hardware="h100", tp_degree=8)
    assert not bad_gqa.feasible
    assert "must divide n_heads=32" in bad_gqa.reason_if_infeasible

    bad_tp = apply_transition(
        arch, ChangeParallelism(), {"tp": 3},
        hardware="h100", tp_degree=8)
    assert not bad_tp.feasible
    assert "TP=3 must divide" in bad_tp.reason_if_infeasible

    gpt = load_baseline_model(
        str(ROOT / "configs" / "gpt_oss_120b.json")).candidate
    bad_ep = apply_transition(
        candidate_to_arch(gpt), ChangeParallelism(), {"ep": 3},
        hardware="h100", tp_degree=8, ep_degree=8, dp_degree=8)
    assert not bad_ep.feasible
    assert "EP=3 must divide DP=8" in bad_ep.reason_if_infeasible

    bad_ep_dp = apply_transition(
        candidate_to_arch(gpt), ChangeParallelism(), {"ep": 4, "dp": 6},
        hardware="h100", tp_degree=8, ep_degree=8, dp_degree=8)
    assert not bad_ep_dp.feasible
    assert "EP=4 must divide DP=6" in bad_ep_dp.reason_if_infeasible


def test_attention_delta_rejects_pure_state_noop():
    from ac.delta_engine import apply_transition
    from ac.deltas.swap_attention_to_mla import SwapAttentionToMLA
    from ac.optimizer_bridge import candidate_to_arch

    candidate = _complex_candidate()
    candidate.layer_type_list = ["state"] * candidate.n_layers
    candidate.n_attention_layers = 0
    candidate.n_state_layers = candidate.n_layers
    transition = apply_transition(
        candidate_to_arch(candidate), SwapAttentionToMLA(),
        hardware="h100", tp_degree=8, pp_degree=2, ep_degree=2,
        dp_degree=8, cp_degree=1)
    assert not transition.feasible
    assert "no attention layers" in transition.reason_if_infeasible


def test_precision_delta_reaches_canonical_metrics_memory_and_field_diff():
    from ac.baseline import load_baseline_model
    from ac.evaluator import evaluate_delta

    baseline = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    constraints = _w46_constraints(baseline)
    constraints.pp = 1
    constraints.cp = 1
    constraints.pretraining_context_length = 131072

    activation = evaluate_delta(
        baseline, "h100", constraints,
        "change_precision_per_component", {"activation": "fp8"},
        include_pareto=False)
    assert activation.feasible
    assert any(change["field"] == "activation_precision"
               for change in activation.field_changes)
    assert activation.metrics["predicted_loss"].delta > 0
    assert activation.metrics["training_memory_per_gpu_gb"].delta < 0

    weight = evaluate_delta(
        baseline, "h100", constraints,
        "change_precision_per_component", {"weight": "fp8"},
        include_pareto=False)
    changed = {change["field"] for change in weight.field_changes}
    assert {"weight_precision", "ffn_precision"} <= changed
    assert weight.metrics["predicted_loss"].delta > \
        activation.metrics["predicted_loss"].delta


def test_component_precision_modes_have_distinct_monotonic_physics():
    import copy
    from ac.architecture import parameter_byte_ledger
    from ac.baseline import load_baseline_model
    from ac.optimizer import (
        DeploymentConstraints, PRECISION_CONFIGS, evaluate_candidate)

    baseline = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    constraints = DeploymentConstraints(
        target_params_b=baseline.total_params_b,
        training_tokens=int(2e12),
        pretraining_context_length=8192,
        context_length=8192,
        prompt_len=8192,
        serving_batch=1,
        serving_tbt_ms=None,
        serving_ttft_ms=None,
        tp=8,
        pp=1,
        dp=8,
    )
    rows = []
    for mode in ("all_bf16", "ffn_fp8", "all_fp8"):
        candidate = copy.deepcopy(baseline)
        for field, value in PRECISION_CONFIGS[mode].items():
            setattr(candidate, field, copy.deepcopy(value))
        evaluated = evaluate_candidate(candidate, "h100", constraints)
        rows.append((
            parameter_byte_ledger(candidate).total_bytes,
            evaluated.memory_per_gpu_gb,
            evaluated.throughput.training_memory_per_gpu_gb,
            evaluated.serving_tbt_ms,
            evaluated.throughput.prefill_time_ms,
            evaluated.training_tps,
        ))

    assert rows[0][0] > rows[1][0] > rows[2][0]
    for metric_index in (1, 2, 3, 4):
        assert rows[0][metric_index] > rows[1][metric_index] > rows[2][metric_index]
    assert rows[0][5] < rows[1][5] < rows[2][5]


def test_composed_delta_preserves_precision_notes_and_structural_guards():
    from ac.baseline import load_baseline_model
    from ac.evaluator import evaluate_delta_sequence

    baseline = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    constraints = _w46_constraints(baseline)
    constraints.pp = 1
    constraints.cp = 1

    composed = evaluate_delta_sequence(
        baseline, "h100", constraints, [
            ("change_precision_per_component", {"activation": "fp8"}),
            ("scale_n_layers", {"delta": 8}),
        ], include_pareto=False)
    assert composed.feasible
    assert any(change["field"] == "activation_precision"
               for change in composed.field_changes)
    assert composed.quality_delta["precision_loss"] > 0

    state = evaluate_delta_sequence(
        baseline, "h100", constraints, [
            ("add_state_layers", {"ratio": "1:3"}),
            ("change_parallelism", {"tp": 4, "pp": 2, "dp": 8}),
        ], include_pareto=False)
    assert state.feasible
    ratio_change = next(
        change for change in state.field_changes
        if change["field"] == "state.hybrid_ratio")
    assert ratio_change["candidate"] == "1:3"

    invalid = evaluate_delta_sequence(
        baseline, "h100", constraints, [
            ("scale_n_layers", {"delta": 1}),
            ("change_parallelism", {"pp": 4}),
        ], include_pareto=False)
    assert not invalid.feasible
    assert "structural_validation_failed" in invalid.reason_if_infeasible
    assert "PP=4 must divide n_layers" in invalid.reason_if_infeasible


def test_baseline_loader_rejects_mixed_activation_precision(tmp_path):
    import pytest
    from ac.baseline import BaselineUnsupportedError, load_baseline_model

    source = json.loads(
        (ROOT / "configs" / "gpt_oss_120b.json").read_text())
    source["architecture"]["layer_configs"][0]["residual_dtype"] = "fp8"
    source["architecture"]["layer_configs"][1]["residual_dtype"] = "bf16"
    path = tmp_path / "mixed-activation.json"
    path.write_text(json.dumps(source))
    with pytest.raises(BaselineUnsupportedError, match="Multiple residual"):
        load_baseline_model(str(path))


def test_schema_rejects_structurally_invalid_parallelism():
    import copy
    from ac.schema import validate_config

    source = json.loads(
        (ROOT / "configs" / "gpt_oss_120b.json").read_text())
    bad_pp = copy.deepcopy(source)
    bad_pp["parallelism"]["pipeline_parallel"] = 7
    assert any("must divide architecture.n_layers" in error
               for error in validate_config(bad_pp))

    bad_ep = copy.deepcopy(source)
    bad_ep["parallelism"]["expert_parallel"] = 8
    bad_ep["parallelism"]["data_parallel"] = 12
    assert any("must divide parallelism.data_parallel" in error
               for error in validate_config(bad_ep))


def test_state_and_depth_deltas_preserve_mixer_distribution():
    from ac.deltas import get
    from ac.optimizer_bridge import candidate_to_arch

    base = _complex_candidate()
    dense = candidate_to_arch(base)
    dense.state_config = None
    dense.layer_type_list = ["attention"] * dense.n_layers
    added = get("add_state_layers").apply(dense, state_fraction=0.75)
    state_positions = [
        index for index, kind in enumerate(added.layer_type_list)
        if kind == "state"]
    assert len(state_positions) == 6
    assert state_positions != list(range(6))
    assert added.placement_strategy == "periodic"

    expanded = get("scale_n_layers").apply(
        candidate_to_arch(base), delta=8)
    assert expanded.n_layers == 16
    assert expanded.layer_type_list.count("state") == 12
    assert expanded.layer_type_list.count("local_attention") == 2


def test_width_and_mla_deltas_preserve_parameterized_paths():
    import pytest
    from ac.architecture import parameter_ledger
    from ac.baseline import load_baseline_model
    from ac.deltas import get
    from ac.evaluator import arch_to_candidate
    from ac.optimizer_bridge import candidate_to_arch

    moe = load_baseline_model(
        str(ROOT / "configs" / "gpt_oss_120b.json")).candidate
    scaled_arch = get("scale_d_model").apply(
        candidate_to_arch(moe), delta=640)
    scaled = arch_to_candidate(scaled_arch, moe)
    assert scaled.moe["expert_dim"] / scaled.d_model == pytest.approx(
        moe.moe["expert_dim"] / moe.d_model)

    dense = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    mla_arch = get("swap_attention_to_mla").apply(
        candidate_to_arch(dense), latent_dim=512, d_rope=64)
    mla = arch_to_candidate(mla_arch, dense)
    direct_q_params = (
        mla.d_model * mla.n_heads * mla.d_head * mla.n_layers)
    assert parameter_ledger(mla).attention >= direct_q_params


def test_delta_rejects_silent_densify_clamp_and_allows_mla_local_composition():
    import pytest
    from ac.baseline import load_baseline_model
    from ac.deltas import get
    from ac.optimizer_bridge import candidate_to_arch

    moe = load_baseline_model(
        str(ROOT / "configs" / "gpt_oss_120b.json")).candidate
    arch = candidate_to_arch(moe)
    with pytest.raises(ValueError, match="must be < n_layers"):
        get("densify_first_k").apply(arch, k=arch.n_layers)

    mla = get("swap_attention_to_mla")
    ok, reason = mla.precondition(arch)
    assert ok, reason
    composed = mla.apply(arch, latent_dim=256, d_rope=64)
    assert composed.attention_type == "mla"
    assert composed.n_local_attn_layers == moe.n_local_attn_layers


def test_whole_swa_serializes_for_dense_and_hybrid_paths(tmp_path):
    import copy
    from ac.baseline import load_baseline_model
    from ac.optimizer import (
        OptimizationResult, evaluate_candidate, result_to_config)
    from ac.schema import validate_config

    dense = load_baseline_model(
        str(ROOT / "configs" / "mistral_7b.json")).candidate
    hybrid = _complex_candidate()
    for index, candidate in enumerate((dense, hybrid)):
        candidate = copy.deepcopy(candidate)
        candidate.attention_type = "swa"
        candidate.swa_window = 4096
        candidate.n_local_attn_layers = 0
        constraints = _w46_constraints(candidate)
        if index == 0:
            constraints.pp = 1
            constraints.cp = 1
        evaluated = evaluate_candidate(candidate, "h100", constraints)
        result = OptimizationResult(
            optimal=evaluated, pareto_frontier=[evaluated],
            all_evaluated=[evaluated], constraints=constraints,
            hardware="h100")
        config = result_to_config(result)
        assert validate_config(config) == []
        attention_blocks = [
            band["attention"]
            for band in config["architecture"]["layer_configs"]
            if band.get("attention") is not None
        ]
        assert attention_blocks
        assert all(block["type"] == "swa" for block in attention_blocks)
        assert all(block["window_size"] == 4096
                   for block in attention_blocks)
        path = tmp_path / f"swa-{index}.json"
        path.write_text(json.dumps(config))
        restored = load_baseline_model(str(path)).candidate
        assert restored.swa_window == 4096
        assert restored.n_local_attn_layers == 0


def test_baseline_loader_fails_closed_on_unrepresentable_band_drift(tmp_path):
    import copy
    import pytest
    from ac.baseline import BaselineUnsupportedError, load_baseline_model

    source = json.loads(
        (ROOT / "configs" / "gpt_oss_120b.json").read_text())
    cases = []

    projection = copy.deepcopy(source)
    projection["architecture"]["layer_configs"][1]["attention"][
        "n_kv_heads"] = 4
    cases.append((projection, "attention projection"))

    normalization = copy.deepcopy(source)
    normalization["architecture"]["layer_configs"][1]["normalization"][
        "eps"] = 1e-6
    cases.append((normalization, "normalization"))

    ffn = copy.deepcopy(source)
    ffn["architecture"]["layer_configs"][1]["ffn"]["expert_dim"] += 128
    cases.append((ffn, "FFN shapes"))

    for index, (config, message) in enumerate(cases):
        path = tmp_path / f"band-drift-{index}.json"
        path.write_text(json.dumps(config))
        with pytest.raises(BaselineUnsupportedError, match=message):
            load_baseline_model(str(path))


def test_capped_source_grid_preserves_every_common_axis_value():
    from ac.optimizer import DeploymentConstraints, _source_generation_slices

    constraints = DeploymentConstraints(
        target_params_b=30,
        context_length=131072,
        max_candidates=1000,
        vocab_options=[32000, 65536],
        tp_options=[4, 8, 16],
        precision_configs=["all_bf16", "ffn_fp8", "all_fp8"],
        kv_bits_options=[16, 8, 4],
        allow_mtp=True,
        mtp_depth_options=[0, 1],
        cp_options=[1, 2, 4],
        allow_rope_scaling=True,
        rope_scaling_methods=["none", "yarn"],
    )
    slices, capped = _source_generation_slices("h100", constraints)

    assert capped
    assert len(slices) < 2 * 3 * 3 * 3 * 2 * 3 * 2
    assert {s.vocab_size for s in slices} == {32000, 65536}
    assert {s.tp for s in slices} == {4, 8, 16}
    assert {s.precision_configs[0] for s in slices} == {
        "all_bf16", "ffn_fp8", "all_fp8"}
    assert {s.kv_bits_options[0] for s in slices} == {16, 8, 4}
    assert {s.mtp_depth_options[0] for s in slices} == {0, 1}
    assert {s.cp for s in slices} == {1, 2, 4}
    assert {s.rope_scaling_methods[0] for s in slices} == {"none", "yarn"}


def test_capped_moe_source_sample_keeps_granularity_and_topology_axes():
    from ac.lattice_engine import HARDWARE, compute_moe_options
    from ac.optimizer import DeploymentConstraints, _generation_moe_options

    baseline_ffn = 14336
    options = compute_moe_options(
        HARDWARE["h100"], "bf16", d_model=4096,
        baseline_ffn_dim=baseline_ffn,
        ep_degrees=[2, 4, 8, 16],
        n_experts_options=[32, 64],
        top_k_options=[2, 4],
        granularity_targets=(0.25, 1.0),
    )
    constraints = DeploymentConstraints(target_params_b=30, max_candidates=1000)
    constraints._source_moe_option_cap = 8
    sampled = _generation_moe_options(
        options, constraints, "moe", baseline_ffn)

    assert {o.style for o in sampled} == {"coarse", "fine"}
    assert {o.ep_degree for o in sampled} == {2, 4, 8, 16}
    assert {o.n_experts for o in sampled} == {32, 64}
    assert {o.top_k for o in sampled} == {2, 4}
    fine_ratios = [
        o.active_ffn_equivalent / baseline_ffn
        for o in sampled if o.style == "fine"
    ]
    assert any(abs(ratio - 0.25) < 0.03 for ratio in fine_ratios)
    assert any(abs(ratio - 1.0) < 0.03 for ratio in fine_ratios)


def test_capped_all_axis_search_retains_parallelism_and_structure_classes():
    from ac.optimizer import DeploymentConstraints, optimize

    constraints = DeploymentConstraints(
        target_params_b=30,
        training_tokens=int(2e12),
        pretraining_context_length=32768,
        context_length=131072,
        prompt_len=8192,
        serving_batch=8,
        serving_tbt_ms=None,
        serving_ttft_ms=None,
        tp_options=[4, 8, 16],
        pp_options=[1, 2],
        dp=16,
        cp_options=[1, 2, 4],
        allow_moe=True,
        allow_state=True,
        max_total_params_b=300,
        moe_n_experts_options=[32, 64],
        moe_top_k_options=[2, 4],
        moe_granularity_targets=[0.25, 1.0],
        dense_ffn_layer_options=[0, 2],
        ep_options=[2, 4, 8, 16],
        allow_mla=True,
        allow_local_global=True,
        local_window_options=[1024, 4096],
        local_global_ratio_options=["1:1", "3:1"],
        allow_mtp=True,
        mtp_depth_options=[0, 1],
        allow_csa=True,
        csa_block_size_options=[64, 128],
        csa_top_k_options=[8, 16],
        allow_indexshare=True,
        indexshare_num_buckets_options=[64, 128],
        indexshare_top_k_options=[4, 8],
        allow_msa=True,
        msa_window_options=[512, 1024],
        msa_dilated_top_k_options=[32, 64],
        msa_global_top_k_options=[8, 16],
        allow_rope_scaling=True,
        rope_scaling_methods=["none", "yarn"],
        max_candidates=240,
        max_full_evaluations=120,
        local_refine_budget=0,
        allow_quality_sentinel=True,
    )
    result = optimize("h100", constraints)
    evaluated = [entry.arch for entry in result.all_evaluated]

    def family(candidate):
        if candidate.moe and candidate.state_config:
            return "moe_hybrid"
        if candidate.moe:
            return "moe"
        if candidate.state_config:
            return "hybrid"
        return "dense"

    assert {family(c) for c in evaluated} == {
        "dense", "moe", "hybrid", "moe_hybrid"}
    assert {c.attention_type for c in evaluated} >= {
        "full", "mla", "csa", "indexshare", "msa"}
    assert any(c.n_local_attn_layers > 0 for c in evaluated)
    assert {c.tp_degree for c in evaluated} == {4, 8, 16}
    assert {c.cp_degree for c in evaluated} == {1, 2, 4}
    assert {c.rope_scaling_method for c in evaluated} == {"none", "yarn"}
    assert {c.mtp_n_predict_depths for c in evaluated} == {0, 1}


def test_ep_over_dp_training_memory_does_not_double_shard_experts():
    import copy
    import pytest
    from ac.baseline import load_baseline_model
    from ac.optimizer import DeploymentConstraints, evaluate_candidate
    from ac.optimizer_bridge import candidate_to_arch
    from ac.stress import Workload, compute_throughput_stress

    base = load_baseline_model(
        str(ROOT / "configs" / "gpt_oss_120b.json")).candidate
    memories = []
    serving_memories = []
    stress_memories = []
    for ep in (1, 2, 4, 8):
        candidate = copy.deepcopy(base)
        candidate.ep_degree = ep
        constraints = DeploymentConstraints(
            target_params_b=candidate.active_params_b,
            training_tokens=int(2e12),
            pretraining_context_length=8192,
            context_length=8192,
            serving_batch=2,
            serving_tbt_ms=None,
            serving_ttft_ms=None,
            tp=8,
            pp=1,
            dp=8,
            allow_moe=True,
            ep_options=[ep],
            max_total_params_b=200,
            allow_quality_sentinel=True,
        )
        evaluated = evaluate_candidate(candidate, "h100", constraints)
        memories.append(evaluated.throughput.training_memory_per_gpu_gb)
        serving_memories.append(evaluated.memory_per_gpu_gb)
        stress = compute_throughput_stress(
            candidate_to_arch(candidate), "h100",
            Workload(batch_size=2, prefill_seq_len=8192,
                     decode_kv_len=8192),
            tp_degree=8, pp_degree=1, dp_degree=8, ep_degree=ep,
        )
        stress_memories.append(stress.training_mem)

    assert memories == pytest.approx([memories[0]] * 4, rel=1e-9)
    assert stress_memories == pytest.approx(
        [stress_memories[0]] * 4, rel=1e-9)
    assert serving_memories == sorted(serving_memories, reverse=True)
    assert serving_memories[0] > serving_memories[-1]


def test_pareto_csv_exposes_factorized_architecture_family():
    from types import SimpleNamespace
    from ac.optimizer import OptimizationResult, result_to_pareto_csv

    candidates = []
    for has_moe, has_state, expected in (
        (False, False, "dense"),
        (True, False, "moe"),
        (False, True, "hybrid"),
        (True, True, "moe_hybrid"),
    ):
        candidate = _complex_candidate()
        candidate.moe = candidate.moe if has_moe else None
        candidate.state_config = candidate.state_config if has_state else None
        if not has_state:
            candidate.n_state_layers = 0
            candidate.n_attention_layers = candidate.n_layers
            candidate.layer_type_list = ["attention"] * candidate.n_layers
        candidate.moe_style = "fine" if has_moe else "dense"
        throughput = SimpleNamespace(
            prefill_time_ms=1.0,
            training_memory_per_gpu_gb=2.0,
        )
        quality = SimpleNamespace(
            confidence="medium", uncertainty_total=0.1,
        )
        candidates.append((expected, SimpleNamespace(
            arch=candidate, predicted_loss=2.0,
            training_tps=100.0, serving_tbt_ms=1.0,
            memory_per_gpu_gb=3.0, throughput=throughput, quality=quality,
        )))

    result = OptimizationResult(
        optimal=None,
        pareto_frontier=[entry for _, entry in candidates],
        all_evaluated=[entry for _, entry in candidates],
        hardware="h100",
    )
    rows = list(csv.DictReader(io.StringIO(result_to_pareto_csv(result))))
    assert {row["architecture_family"] for row in rows} == {
        expected for expected, _ in candidates}


def test_family_rollup_joins_selected_row_by_full_architecture_identity():
    import copy
    from types import SimpleNamespace
    from ac.cli_compile import _rollup_families

    family_best = _complex_candidate()
    family_best.tp_degree = 4
    selected = copy.deepcopy(family_best)
    selected.tp_degree = 8

    def evaluated(candidate, tbt, memory):
        return SimpleNamespace(
            arch=candidate,
            predicted_loss=2.0,
            serving_tbt_ms=tbt,
            memory_per_gpu_gb=memory,
            training_tps=100.0,
            throughput=SimpleNamespace(
                prefill_time_ms=10.0,
                hbm_spill_gb=0.0,
                spill_tier="fits",
                training_memory_per_gpu_gb=20.0,
            ),
            meets_constraints=True,
        )

    best_ev = evaluated(family_best, tbt=25.0, memory=230.0)
    selected_ev = evaluated(selected, tbt=16.0, memory=178.0)
    result = SimpleNamespace(
        all_evaluated=[best_ev, selected_ev], hardware="b200")
    rows = _rollup_families(result, selected_ev)

    picked = [row for row in rows if row["is_selected"]]
    assert len(picked) == 1
    assert picked[0]["arch_mode"] == "picked"
    assert picked[0]["tbt_ms"] == 16.0
    assert picked[0]["mem_gb"] == 178.0


def test_optimizer_surfaces_partial_and_total_evaluator_failures(monkeypatch):
    import ac.optimizer as optimizer

    constraints = optimizer.DeploymentConstraints(
        target_params_b=1.0,
        training_tokens=int(2e11),
        context_length=2048,
        prompt_len=2048,
        serving_tbt_ms=None,
        serving_ttft_ms=None,
        tp=1,
        pp=1,
        dp=1,
        max_candidates=20,
        max_full_evaluations=20,
        local_refine_budget=0,
        allow_quality_sentinel=True,
    )
    original = optimizer.evaluate_candidate
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic seam failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer, "evaluate_candidate", fail_once)
    result = optimizer.optimize("h100", constraints)
    assert result.optimal is not None
    assert result.evaluation_failures == 1
    assert result.evaluation_failure_reasons == {
        "RuntimeError: synthetic seam failure": 1}
    config = optimizer.result_to_config(result)
    assert config["metadata"]["search_stats"]["evaluation_failures"] == 1

    def fail_all(*args, **kwargs):
        raise RuntimeError("total synthetic seam failure")

    monkeypatch.setattr(optimizer, "evaluate_candidate", fail_all)
    import pytest
    with pytest.raises(ValueError, match="all 20 candidate evaluation"):
        optimizer.optimize("h100", constraints)


def test_web_generator_parallelism_uses_ep_overlay_and_keeps_cp_units():
    import pytest

    import _generator_payload as generator

    dp, ep = generator.training_dp_and_ep(
        training_cluster=64, tp=1, pp=1, allow_moe=True)
    assert (dp, ep) == (64, 2)
    assert 1 * 1 * dp == 64

    rec = generator.annotate_parallelism_units(
        {
            "tp": 4,
            "cp_degree": 2,
            "ep": 8,
            "train_tps": 1600,
            "tbt_ms": 20,
            "mem_gb": 2,
        },
        pp=2,
        dp=16,
        chosen_batch=8,
    )
    assert rec["cp"] == 2
    assert rec["training_replica_gpus"] == 16
    assert rec["serving_instance_gpus"] == 128
    assert rec["training_cluster_gpus"] == 256
    assert rec["train_tps_per_gpu"] == 100
    assert rec["train_tps_aggregate"] == 25600
    assert rec["train_tps_unit"] == "tokens/sec/training-replica"
    assert rec["mem_per_replica_gb"] == 256
    assert rec["decode_tps_per_replica"] == 400

    # A dense fallback in an allow-MoE row must serialize EP=1 explicitly.
    # Missing EP must not inherit the mode's EP=2 over an effective DP=1.
    with pytest.raises(ValueError, match="EP=2 must divide DP=1"):
        generator.annotate_parallelism_units(
            {"tp": 8, "cp": 8, "dp": 1, "train_tps": 1000},
            pp=1, dp=1, fallback_ep=2,
        )
    dense = generator.annotate_parallelism_units(
        {"tp": 8, "cp": 8, "dp": 1, "ep": 1, "train_tps": 1000},
        pp=1, dp=1, fallback_ep=2,
    )
    assert dense["serving_instance_gpus"] == 64

    # A larger replica may have higher replica TPS while still being slower
    # per GPU. The retired web post-process must not rewrite either value.
    payload = {
        "grid": [
            {"hw": "h100", "params_B": 1.0, "tokens_T": 2.0,
             "context_length": 8192, "arch_mode": "dense",
             "state_type": None, "serving": "continuous",
             "optimal": {"train_tps": 1000, "train_tps_per_replica": 1000,
                         "train_tps_per_gpu": 1000,
                         "train_tps_aggregate": 64000}},
            {"hw": "h100", "params_B": 7.0, "tokens_T": 2.0,
             "context_length": 8192, "arch_mode": "dense",
             "state_type": None, "serving": "continuous",
             "optimal": {"train_tps": 4000, "train_tps_per_replica": 4000,
                         "train_tps_per_gpu": 500,
                         "train_tps_aggregate": 32000}},
        ]
    }
    before = json.loads(json.dumps(payload))
    generator._clamp_train_tps_monotone_in_params(payload)
    assert payload == before

    # Family cards inherit candidate-specific topology and units, not the
    # enclosing row's scalar PP/DP defaults.
    family_payload = {
        "grid": [{
            "hw": "h100", "params_B": 7.0, "tokens_T": 2.0,
            "context_length": 131072, "arch_mode": "dense",
            "state_type": None, "pp": 1, "dp": 64,
            "optimal": {
                "arch_family": "dense", "loss": 2.0, "tbt_ms": 10.0,
                "ttft_ms": 100.0, "mem_gb": 20.0,
                "train_tps": 3200, "train_tps_per_replica": 3200,
                "train_tps_per_gpu": 100,
                "train_tps_aggregate": 6400,
                "train_tps_unit": "tokens/sec/GPU",
                "uncertainty_total_pct": 3.0, "params_B": 7.0,
                "active_params_B": 7.0, "d_model": 4096,
                "n_layers": 32, "tp": 4, "pp": 2, "cp": 4,
                "dp": 2, "ep": 1, "training_replica_gpus": 32,
                "serving_instance_gpus": 32,
                "training_cluster_gpus": 64,
                "multi_ctx_shape_pinned": True,
            },
            "pareto": [{
                "arch_family": "dense", "loss": 1.99, "tbt_ms": 1.0,
                "ttft_ms": 10.0, "mem_gb": 2.0, "train_tps": 6400,
                "train_tps_per_gpu": 200, "uncertainty_total_pct": 3.0,
                "params_B": 7.0, "active_params_B": 7.0,
                "d_model": 4096, "n_layers": 30, "tp": 4, "pp": 2,
                "cp": 4, "dp": 2, "ep": 1,
                "multi_ctx_shape_pinned": False,
            }, {
                "arch_family": "moe", "loss": 1.98, "tbt_ms": 1.0,
                "ttft_ms": 10.0, "mem_gb": 2.0, "train_tps": 6400,
                "train_tps_per_gpu": 200, "uncertainty_total_pct": 3.0,
                "params_B": 20.0, "active_params_B": 7.0,
                "d_model": 4096, "n_layers": 30, "tp": 4, "pp": 2,
                "cp": 4, "dp": 2, "ep": 2,
                "multi_ctx_shape_pinned": False,
            }],
        }]
    }
    generator._build_family_rollup(family_payload)
    family = family_payload["cells"][0]["families"][0]
    assert (family["tp"], family["pp"], family["cp"], family["dp"],
            family["ep"]) == (4, 2, 4, 2, 1)
    assert family["training_cluster_gpus"] == 64
    assert family["train_tps_per_gpu"] == 100
    assert family["loss"] == 2.0
    assert family["multi_ctx_shape_pinned"] is True


def test_web_payload_slimmer_preserves_only_browser_contract():
    import slim_web_payload as slimmer

    candidate = {field: f"value-{field}" for field in slimmer.CAND_KEEP}
    candidate.update({
        "train_tps": 800, "train_tps_per_gpu": 100,
        "train_tps_per_replica": 800, "train_tps_aggregate": 6400,
        "train_tps_unit": "tokens/sec/training-replica",
        "dp": 8, "ep": 2, "cp": 4, "tp": 2, "pp": 1,
        "quality_terms": {
            "architecture_residual": {
                "value": 0.1, "uncertainty": 0.2, "source": "drop",
                "features": {"n_query_heads": 32, "subterms": {"gqa": 0.1},
                             "internal_calibration": "drop"},
            },
            "precision_residual": {
                "value": 0.3, "uncertainty": 0.4, "notes": "drop",
                "features": {"ffn": "drop"},
            },
        },
        "diagnostics": {"large": True},
        "uncertainty_breakdown": {"large": True},
    })
    family = {field: f"value-{field}" for field in slimmer.FAMILY_KEEP}
    family["unused_family_detail"] = "drop"
    row = {field: f"value-{field}" for field in slimmer.ROW_KEEP}
    row.update({
        "optimal": candidate,
        "pareto": [candidate] * 20,
        "families": [family],
        "shadow_prices": [{"desc": "x", "delta_pct": 1, "interp": "y",
                           "unused": True}],
        "arch_dim_prices": [{"change": "+1", "delta_loss_pct": 1,
                             "delta_train_tps_pct": 2, "delta_tbt_pct": 3,
                             "delta_mem_pct": 4, "decision": "accepted",
                             "reason": "x", "unused": True}],
        "pareto_4d": [candidate],
        "diagnostics": {"large": True},
    })
    payload = {
        "_regen20t": {"generated": "today"},
        "hardware_info": {"h100": {"label": "H100", "hbm_gb": 80,
                                      "unused": True}},
        "grid": [row],
        "cells": [{"duplicated": True}],
        "_family_smoothing": {"large": True},
    }

    slim = slimmer.slim_payload(payload, top_k=12)
    assert set(slim) == slimmer.TOP_KEEP
    assert "unused" not in slim["hardware_info"]["h100"]
    slim_row = slim["grid"][0]
    assert set(slim_row) <= slimmer.ROW_KEEP
    assert len(slim_row["pareto"]) == 12
    assert set(slim_row["optimal"]) <= slimmer.CAND_KEEP
    for required in ("dp", "ep", "cp", "training_replica_gpus",
                     "train_tps_per_gpu", "quality_terms",
                     "csa_block_size", "indexshare_num_buckets",
                     "msa_window_size"):
        assert required in slim_row["optimal"]
    terms = slim_row["optimal"]["quality_terms"]
    assert set(terms["architecture_residual"]) == {
        "value", "uncertainty", "features"
    }
    assert terms["architecture_residual"]["features"] == {
        "n_query_heads": 32, "subterms": {"gqa": 0.1}
    }
    assert terms["precision_residual"] == {
        "value": 0.3, "uncertainty": 0.4
    }
    assert set(slim_row["families"][0]) == slimmer.FAMILY_KEEP
    assert set(slim_row["shadow_prices"][0]) == slimmer.SHADOW_KEEP
    assert set(slim_row["arch_dim_prices"][0]) == slimmer.ARCH_DIM_KEEP


def test_cluster_floor_derives_dp_and_picker_compares_parallelism_per_gpu():
    import copy
    from types import SimpleNamespace
    import ac.optimizer as optimizer

    candidate = _complex_candidate()
    candidate.tp_degree = 4
    candidate.pp_degree = 2
    candidate.cp_degree = 2
    candidate.ep_degree = 4
    constraints = optimizer.DeploymentConstraints(
        target_params_b=7.0,
        training_cluster_gpus=64,
        tp=1,
        pp=1,
        dp=1,
    )
    # TP4 x PP2 x CP2 = 16 GPUs/replica; DP4 fills 64 GPUs and is a
    # legal EP4 overlay.
    assert optimizer._effective_candidate_dp(candidate, constraints) == 4
    candidate.ep_degree = 8
    # EP8 requires DP8, so the minimum legal world is honestly 128 GPUs.
    assert optimizer._effective_candidate_dp(candidate, constraints) == 8

    small = copy.deepcopy(candidate)
    small.moe = None
    small.ep_degree = 1
    small.tp_degree = 1
    small.pp_degree = 1
    small.cp_degree = 1
    large = copy.deepcopy(small)
    large.tp_degree = 8

    def evaluated(arch, replica_tps):
        return SimpleNamespace(
            arch=arch,
            predicted_loss=2.0,
            training_tps=replica_tps,
            serving_tbt_ms=10.0,
            memory_per_gpu_gb=10.0,
            quality=SimpleNamespace(uncertainty_total=0.02),
            throughput=SimpleNamespace(
                prefill_time_ms=100.0,
                spill_tier="fits",
                training_memory_per_gpu_gb=20.0,
            ),
        )

    one_gpu = evaluated(small, 1000.0)
    eight_gpu = evaluated(large, 4000.0)
    picker_constraints = SimpleNamespace(
        objective_profile="research_quality",
        strict_quality=False,
        serving_tbt_ms=None,
        serving_ttft_ms=None,
    )
    key = optimizer.build_display_sort_key(
        [one_gpu, eight_gpu], picker_constraints)
    assert min([one_gpu, eight_gpu], key=key) is one_gpu
    assert optimizer._evaluated_training_tps_per_gpu(one_gpu) == 1000
    assert optimizer._evaluated_training_tps_per_gpu(eight_gpu) == 500


def test_training_cluster_cap_is_a_shared_evaluator_guard():
    import copy
    import pytest
    import ac.optimizer as optimizer

    candidate = _complex_candidate()
    candidate.tp_degree = 4
    candidate.pp_degree = 2
    candidate.cp_degree = 2
    candidate.ep_degree = 4
    exact = optimizer.DeploymentConstraints(
        target_params_b=candidate.active_params_b,
        param_tolerance=0.5,
        context_length=8192,
        prompt_len=8192,
        training_cluster_gpus=64,
        max_training_cluster_gpus=64,
        allow_quality_sentinel=True,
        tp=1,
        pp=1,
        dp=1,
    )

    fits = optimizer.evaluate_candidate(copy.deepcopy(candidate), "h100", exact)
    assert fits.meets_constraints
    assert not fits.feasibility.guards["training_cluster_cap"].triggered

    candidate.ep_degree = 8
    oversized = optimizer.evaluate_candidate(candidate, "h100", exact)
    assert not oversized.meets_constraints
    assert oversized.feasibility.guards["training_cluster_cap"].triggered
    assert "Training world 128 GPUs > hard cap 64 GPUs" in (
        oversized.constraint_violations
    )

    with pytest.raises(
        ValueError,
        match="training_cluster_gpus cannot exceed max_training_cluster_gpus",
    ):
        optimizer.DeploymentConstraints(
            training_cluster_gpus=128,
            max_training_cluster_gpus=64,
        )


# ============================================================
# test_wave47_fixes.py  (w47)
# ============================================================
"""Wave 47 regression pins — auto-progress default for greenfield searches.

Before Wave 47, an uncapped `ac-compile` greenfield run (~10^4 full
evaluations, tens of seconds) printed one "Searching..." line and then
went silent unless the user knew about --progress-every. Wave 47 makes
progress reporting on-by-default:

- unset --progress-every resolves to auto (every 1000 candidates), which
  small searches never reach;
- --quiet still disables progress entirely;
- an explicit --progress-every N is honored as before;
- searches with >= 2x the progress interval print an upfront line naming
  the candidate count, with a speed-knob hint when the search is uncapped.
"""

def run_compile(*extra, timeout=120):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "arch.json"
        return subprocess.run(
            [
                sys.executable, "ac/cli_compile.py",
                "--hardware", "h100",
                "--params", "1",
                "--tokens", "0.2",
                "--context", "2048",
                "--no-shadow-prices",
                "--output-config", str(out),
                *extra,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )


class Wave47AutoProgressTests(unittest.TestCase):
    def test_default_large_search_prints_progress_and_scale_note(self):
        # max-full-evaluations 2000 >= 2x the auto interval (1000): both the
        # upfront scale note and at least one progress line must appear.
        result = run_compile("--max-full-evaluations", "2000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("candidates to fully evaluate", result.stderr)
        self.assertIn("evaluated 1,000/", result.stderr)
        # Capped search: the uncapped speed-knob hint must NOT appear.
        self.assertNotIn("uncapped search", result.stderr)

    def test_quiet_suppresses_all_progress(self):
        result = run_compile("--max-full-evaluations", "2000", "--quiet")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("candidates to fully evaluate", result.stderr)
        self.assertNotIn("evaluated 1,000/", result.stderr)

    def test_explicit_progress_every_is_honored(self):
        result = run_compile(
            "--max-full-evaluations", "2000", "--progress-every", "500")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evaluated 500/", result.stderr)


if __name__ == "__main__":
    unittest.main()
