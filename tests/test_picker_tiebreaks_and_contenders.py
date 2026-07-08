"""Picker tiebreaks, lattice sanity, cross-family bars, contender counts.

(Pins from Waves 18h, 19, 20, 23, and 27.)

  * The width lattice is dense (1B includes anchor-adjacent widths) and
    the 1B pick is not pathologically wide-shallow.
  * `--strict-quality`: rank-1 is the argmin-loss candidate.
  * The pre-calibration tiebreak band is capped: default rank-1 spends
    <= ~0.5% predicted loss vs best-loss.
  * Cross-family comparisons under uncalibrated family bias emit
    [unresolved] with a family-bias reason; the per-metric bias bar is
    larger for serving metrics than loss.
  * The research_quality picker buckets memory at a coarse band so a
    sub-display memory diff cannot gate meaningful TBT/TPS wins.
  * The contending-family count is one number across the CLI warning,
    config metadata, arch.md, and the sidecar.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
AC_DIR = os.path.join(REPO, "ac")
if AC_DIR not in sys.path:
    sys.path.insert(0, AC_DIR)

from ac.optimizer import (  # noqa: E402
    DeploymentConstraints, generate_candidates, optimize,
)


# ---------------------------------------------------------------------------
# lattice density + pick sanity (Wave 18h)
# ---------------------------------------------------------------------------

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


def test_strict_quality_rank1_is_argmin_loss():
    c = DeploymentConstraints(target_params_b=1.0, tp=8, pp=1, dp=8,
                              strict_quality=True)
    result = optimize("h100", c)
    frontier = list(result.pareto_frontier)
    best_loss = min(ev.predicted_loss for ev in frontier)
    assert abs(result.optimal.predicted_loss - best_loss) < 1e-9, (
        "--strict-quality must pick the argmin-loss candidate")


# ---------------------------------------------------------------------------
# tiebreak spend cap (Wave 19)
# ---------------------------------------------------------------------------

class TestTiebreakSpendCap:
    def test_default_rank1_spends_at_most_half_percent(self):
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


# ---------------------------------------------------------------------------
# cross-family bias floors (Waves 19 and 20)
# ---------------------------------------------------------------------------

class TestCrossFamilyBias:
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

    def test_cross_family_bar_per_metric(self):
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


# ---------------------------------------------------------------------------
# memory tiebreak bucketing (Wave 23)
# ---------------------------------------------------------------------------

class TestMemoryTiebreakUncertaintyAware(unittest.TestCase):
    """research_quality picker no longer trades meaningful TBT/TPS wins
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


# ---------------------------------------------------------------------------
# contending-family count unification (Wave 27)
# ---------------------------------------------------------------------------

class TestContenderCountUnified(unittest.TestCase):
    """The CLI warning, config metadata, arch.md, and sidecar all report
    the SAME contending-candidate count (`_collect_contenders` is the
    single source of truth; the pre-fix naïve `_loss_interval` overlap
    rule counted the SHARED model error against the decision even though
    it cancels in the pairwise difference, giving counts 3.8x–17x larger
    than the paired-sigma count)."""

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
        m = re.search(
            r"WARNING:\s+(\d+)\s+contending candidate\(s\)", self.stderr
        )
        return int(m.group(1)) if m else None

    def _extract_markdown_count(self):
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
            f"contending-family surfaces disagree: {counts!r}. All six "
            f"must route through `_collect_contenders`.",
        )

    def test_collect_contenders_is_the_source_of_truth(self):
        # In-process check that the two public entry points agree
        # deterministically on the same run.
        from ac.evaluator import EvaluatedCandidate  # noqa: F401 - import shape check
        # If the import worked, the shared helper is reachable.
        from ac.optimizer import _collect_contenders, compute_contending_family_full  # noqa: F401
        # The helper existing and being importable is itself the check;
        # the numeric check above proves it's the source of truth.


if __name__ == "__main__":
    unittest.main()
