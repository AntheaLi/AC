"""MoE training/decode throughput and expert-parallel layout rules.

(Pins from Waves 19, 21, 24, and 25.)

  * MoE training throughput: EP-over-DP token accounting, a2a overlap,
    no phantom shared-expert EP allreduce; equal-active MoE within
    1.1-2.5x of dense per training-replica GPU; per-GPU column in
    pareto.csv.
  * EP=1 is a legal MoE execution plan (experts TP-sharded, vLLM-style).
  * The greenfield MoE / MoE-hybrid enumerators cap EP at the requested
    DP (the training-TPS math lays EP over DP), with a clear error for
    user-supplied EP options that have no survivors.
  * The decode `expert_load` system efficiency is streaming-class
    (0.42/0.38), not 0.20 — anything below the worst published stack's
    implied efficiency re-introduces the roofline double-count.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)

from ac.throughput_model import ArchConfig as TArch, throughput  # noqa: E402
from ac.optimizer import (  # noqa: E402
    DeploymentConstraints,
    _filter_ep_options_by_dp,
    generate_moe_candidates,
    generate_moe_hybrid_candidates,
)


def _per_gpu_train_tps(arch, ep, tp=8, pp=2):
    r = throughput(arch, "h100", tp_degree=tp, pp_degree=pp, dp_degree=8,
                   ep_degree=ep)
    return r.training_throughput_tokens_per_sec / (tp * pp)


def _dense_34b():
    return TArch(d_model=7168, n_layers=64, n_heads=56, d_head=128,
                 n_kv_heads=8, ffn_dim=18944, vocab_size=32000,
                 batch_size=8, seq_len=8192, precision="fp8")


def _moe_34b(shared=False):
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


def _small_moe_arch(batch=1):
    return TArch(
        d_model=2880, n_layers=36, n_heads=64, d_head=64, n_kv_heads=8,
        ffn_dim=2880, batch_size=batch, seq_len=2048,
        moe_config={"n_experts": 128, "top_k": 4, "expert_dim": 2880},
    )


class TestMoeTrainingThroughput:
    def test_equal_active_moe_within_published_band_of_dense(self):
        """The release-review band: real MoE training runs ~1.2-2.5x below
        equal-active dense per GPU, not ~20x."""
        dense = _per_gpu_train_tps(_dense_34b(), ep=1)
        for shared in (False, True):
            moe = _per_gpu_train_tps(_moe_34b(shared=shared), ep=8)
            ratio = dense / moe
            assert 1.1 <= ratio <= 2.5, (
                f"dense/MoE per-GPU training ratio {ratio:.2f} outside the "
                f"[1.1, 2.5] plausibility band (shared={shared}); "
                "EP token accounting or a2a overlap has regressed")

    def test_moe_tps_independent_of_ep_within_nvlink_domain(self):
        """EP lays over DP: growing EP within a node must not divide
        per-replica training throughput."""
        t4 = _per_gpu_train_tps(_moe_34b(), ep=4)
        t8 = _per_gpu_train_tps(_moe_34b(), ep=8)
        assert abs(t4 - t8) / t8 < 0.25, (
            f"per-GPU training TPS moved {abs(t4-t8)/t8*100:.0f}% between "
            "EP=4 and EP=8 — EP is leaking into the replica accounting")

    def test_cross_node_ep_costs_more_but_not_10x(self):
        t8 = _per_gpu_train_tps(_moe_34b(), ep=8)
        t32 = _per_gpu_train_tps(_moe_34b(), ep=32, pp=2)
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


class TestEp1IsLegalExecutionPlan(unittest.TestCase):
    def test_ep1_moe_throughput_runs_and_is_tp_sharded(self):
        arch = _small_moe_arch()
        r1 = throughput(arch, "h100", tp_degree=8, ep_degree=1)
        r2 = throughput(arch, "h100", tp_degree=8, ep_degree=2)
        self.assertGreater(r1.memory_footprint_per_gpu_gb, 0)
        # EP=1 keeps every expert (TP-sharded) resident: strictly more
        # per-GPU memory than EP=2, which halves resident experts.
        self.assertGreater(r1.memory_footprint_per_gpu_gb,
                           r2.memory_footprint_per_gpu_gb)


class TestEpDpCap(unittest.TestCase):
    """EP > DP is unreachable at greenfield enumeration."""

    def test_helper_drops_ep_above_dp(self):
        # default_ep_options for b200 goes to 72; DP=4 must cap to {2, 4}.
        self.assertEqual(
            _filter_ep_options_by_dp([1, 2, 4, 8, 16, 32, 64, 72], dp=4),
            [2, 4],
        )

    def test_helper_keeps_ep_le_dp(self):
        self.assertEqual(
            _filter_ep_options_by_dp([2, 4, 8], dp=8),
            [2, 4, 8],
        )

    def test_helper_drops_ep_lt_2(self):
        # EP=1 is unreachable — every rank would hold every expert.
        self.assertEqual(
            _filter_ep_options_by_dp([1, 2], dp=8),
            [2],
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


class TestEpDpCliEndToEnd(unittest.TestCase):
    """End-to-end: `ac-compile` doesn't print the confessional warning."""

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


class TestExpertLoadEfficiency(unittest.TestCase):
    """Decode expert_load efficiency is streaming-class, not 0.2."""

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


if __name__ == "__main__":
    unittest.main()
