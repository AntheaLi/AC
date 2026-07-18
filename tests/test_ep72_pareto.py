"""Gate-2 Task C: EP=72 Pareto smoke test (V1/V2 in miniature).

(Pins added in gate2-wave1; no existing assertions were modified.)

Three layers, all small-budget / fast-path:

  1. MECHANISM — one fixed K2-class MoE shape (288 experts, top-8) priced
     by the throughput model at EP=8 and EP=72 on h100 vs gb200_nvl72.
     The rack-scale domain must flip the EP economics: EP=72 wins decode
     TBT on the NVL72 while losing badly on the single-node H100.
  2. SEARCH — a tiny optimize() run proves EP=72 MoE candidates are
     generated, evaluated, and land in all_evaluated on gb200_nvl72.
  3. CLI — `ac-compile --hardware gb200_nvl72|h800` end-to-end smoke
     (both new enum values accepted, exit 0).
"""

import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)

from ac.throughput_model import ArchConfig as TArch, throughput  # noqa: E402


def _k2_class_moe(batch=256):
    """~45B-active / ~530B-total 288-expert top-8 MoE (K2-class shape,
    scaled so EP=8 still fits H100 HBM at fp8 — the contrast must come
    from the interconnect, not from a memory cliff)."""
    moe = {"n_experts": 288, "top_k": 8, "expert_dim": 1408,
           "shared_expert": {"ffn_dim": 2048, "precision": "fp8"}}
    return TArch(d_model=7168, n_layers=61, n_heads=64, d_head=128,
                 n_kv_heads=8, ffn_dim=18944, vocab_size=32000,
                 batch_size=batch, seq_len=8192, precision="fp8",
                 weight_precision="fp8", activation_precision="fp8",
                 moe_config=moe)


def _tbt(arch, hw, ep, tp=1, pp=1, dp=2304):
    r = throughput(arch, hw, tp_degree=tp, pp_degree=pp, dp_degree=dp,
                   ep_degree=ep)
    return r.decode_time_per_token_ms, r


class TestEp72Mechanism(unittest.TestCase):
    def test_ep72_wins_on_nvl72_loses_on_h100(self):
        arch = _k2_class_moe()
        tbt8_nvl, _ = _tbt(arch, "gb200_nvl72", 8)
        tbt72_nvl, _ = _tbt(arch, "gb200_nvl72", 72)
        tbt72_h100, _ = _tbt(arch, "h100", 72)
        # V2 core: on the 72-GPU domain, EP=72 beats EP=8 on decode TBT.
        self.assertLess(
            tbt72_nvl, tbt8_nvl,
            f"EP=72 TBT {tbt72_nvl:.2f}ms not better than EP=8 "
            f"{tbt8_nvl:.2f}ms on gb200_nvl72")
        # V1 core: the same EP=72 layout on single-node H100 is >=3x worse
        # than on the rack (slower fabric + all-to-all over 50 GB/s IB).
        # Archived mechanism probe (validation/c_nvl72/runs/probe_mechanism.txt):
        # 106.7 ms vs 28.7 ms at serving batch 256.
        self.assertGreater(
            tbt72_h100 / max(tbt72_nvl, 1e-9), 3.0,
            f"h100 EP=72 {tbt72_h100:.2f}ms vs nvl72 {tbt72_nvl:.2f}ms "
            "— domain contrast too small")


class TestEp72SearchSmoke(unittest.TestCase):
    def _constraints(self, **kw):
        from ac.optimizer import DeploymentConstraints
        base = dict(
            target_params_b=13.0, training_tokens=int(500e9),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16,
            allow_moe=True, max_total_params_b=600,
            moe_n_experts_options=[288], moe_top_k_options=[8],
            moe_granularity_targets=[1.0],
            ep_options=[8, 16, 32, 72],
            training_cluster_gpus=2304,   # unlock EP>DP filter (dp probe)
            max_candidates=200, max_full_evaluations=60,
        )
        base.update(kw)
        return DeploymentConstraints(**base)

    def test_ep72_candidates_evaluated_on_nvl72(self):
        from ac.optimizer import optimize
        result = optimize("gb200_nvl72", self._constraints())
        moe_eps = sorted({
            int(getattr(ev.arch, "ep_degree", 1) or 1)
            for ev in result.all_evaluated
            if getattr(ev.arch, "moe", None)
        })
        self.assertIn(72, moe_eps,
                      f"no EP=72 MoE candidate evaluated on gb200_nvl72; "
                      f"got EPs {moe_eps}")

    def test_ep72_absent_for_indivisible_expert_count(self):
        """384 % 72 != 0 — the enumerator must skip EP=72 for the true K2
        expert count rather than fabricate a fractional layout."""
        from ac.optimizer import generate_moe_candidates
        cands = generate_moe_candidates(
            "gb200_nvl72", self._constraints(moe_n_experts_options=[384]))
        eps = sorted({
            int(getattr(c, "ep_degree", 1) or 1)
            for c in cands if getattr(c, "moe", None)
        })
        self.assertNotIn(72, eps)
        self.assertIn(32, eps)


class TestNewHardwareCliSmoke(unittest.TestCase):
    def _run_compile(self, hw, td):
        out = os.path.join(td, "arch.json")
        p = subprocess.run(
            [sys.executable, "-m", "ac.cli_compile",
             "--hardware", hw,
             "--params", "7", "--tokens", "2", "--context", "8192",
             "--serving-tbt", "50", "--serving-batch", "32",
             "--tp", "8", "--pp", "1", "--dp", "8",
             "--max-candidates", "60",
             "--output-config", out,
             "--no-shadow-prices", "--quiet"],
            capture_output=True, text=True, cwd=REPO, timeout=120,
        )
        self.assertEqual(p.returncode, 0,
                         msg=f"{hw} compile failed: {p.stderr[-1500:]}")
        self.assertTrue(os.path.exists(out))

    def test_gb200_nvl72_cli(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_compile("gb200_nvl72", td)

    def test_h800_cli(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_compile("h800", td)


if __name__ == "__main__":
    unittest.main()
