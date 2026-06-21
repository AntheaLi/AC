"""Direct unit tests for evaluator helpers.

The CLI smoke tests in test_cli_smoke.py exercise the full pipeline end-
to-end, but the B1/B2/dedup fixes live in narrow helpers that benefit
from microscope-level coverage. These tests instantiate `CandidateArch`
objects directly and call the helpers, so any regression that changes
the diff/dedup logic is caught immediately (no need to wait for a CLI
roundtrip).

Tests in this file MUST be cheap (<10ms each). Anything that needs the
full evaluator pipeline belongs in test_cli_smoke.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the package importable; the package itself uses a sys.path hack
# but the test runner doesn't pick it up implicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ac import evaluator  # noqa: E402
from ac.optimizer import CandidateArch  # noqa: E402


def _mk_arch(**overrides) -> CandidateArch:
    """Minimal viable CandidateArch with safe defaults.

    Only the fields the diff helpers read are interesting; everything
    else falls back to the dataclass default.
    """
    base = dict(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=32,
        ffn_dim=14336,
        vocab_size=32000,
    )
    base.update(overrides)
    return CandidateArch(**base)


class ArchChangesTests(unittest.TestCase):
    """Covers `_arch_changes` — the helper that produces the field-level
    diff table in the delta-eval report."""

    def test_identical_arches_produce_no_changes(self):
        a = _mk_arch()
        b = _mk_arch()
        self.assertEqual(evaluator._arch_changes(a, b), [])

    def test_scalar_field_change_surfaces(self):
        a = _mk_arch(n_kv_heads=8)
        b = _mk_arch(n_kv_heads=4)
        changes = evaluator._arch_changes(a, b)
        names = {ch["field"] for ch in changes}
        self.assertIn("n_kv_heads", names)

    def test_densify_first_k_surfaces_n_dense_ffn_layers(self):
        """B1: densify_first_k changes only n_dense_ffn_layers (+ totals
        if you recompute), no top-level scalar. The diff must surface it.
        """
        a = _mk_arch(n_dense_ffn_layers=0, moe={"n_experts": 64, "top_k": 4,
                                                "expert_dim": 8192})
        b = _mk_arch(n_dense_ffn_layers=4, moe={"n_experts": 64, "top_k": 4,
                                                "expert_dim": 8192})
        changes = evaluator._arch_changes(a, b)
        names = {ch["field"] for ch in changes}
        self.assertIn("n_dense_ffn_layers", names)

    def test_moe_topology_change_surfaces(self):
        """B1: change_moe_topology shifts n_experts/top_k inside the moe
        dict. Without the fix these were invisible.
        """
        a = _mk_arch(moe={"n_experts": 128, "top_k": 4, "expert_dim": 8192})
        b = _mk_arch(moe={"n_experts": 64, "top_k": 4, "expert_dim": 8192})
        changes = evaluator._arch_changes(a, b)
        names = {ch["field"] for ch in changes}
        self.assertIn("moe.n_experts", names)
        self.assertNotIn("moe.top_k", names)
        self.assertNotIn("moe.expert_dim", names)

    def test_state_layout_change_surfaces(self):
        """B1: add_state_layers sets n_state_layers/n_attention_layers
        and a state_config dict. The diff must show all of them.
        """
        a = _mk_arch()  # all attention, no state
        b = _mk_arch(
            n_state_layers=8,
            n_attention_layers=24,
            state_config={"type": "mamba2", "d_state": 64,
                          "n_heads": 64, "d_head": 64},
            hybrid_ratio="1:3",
            placement_strategy="interleaved",
        )
        changes = evaluator._arch_changes(a, b)
        names = {ch["field"] for ch in changes}
        self.assertIn("state_enabled", names)
        self.assertIn("state.n_layers", names)
        self.assertIn("attention.n_layers", names)
        self.assertIn("state.hybrid_ratio", names)
        self.assertIn("state.placement_strategy", names)

    def test_mla_swap_surfaces_attention_type_and_latent(self):
        """B1: swap_attention_to_mla sets attention_type + MLA latent
        dims. The canonical fields must appear in the diff.
        """
        a = _mk_arch(attention_type="full")
        b = _mk_arch(
            attention_type="mla",
            mla_kv_latent_dim=256,
            mla_q_latent_dim=1536,
            mla_rope_head_dim=64,
        )
        changes = evaluator._arch_changes(a, b)
        names = {ch["field"] for ch in changes}
        self.assertIn("attention.type", names)
        self.assertIn("attention.mla_kv_latent_dim", names)
        self.assertIn("attention.mla_q_latent_dim", names)
        self.assertIn("attention.mla_rope_head_dim", names)

    def test_swa_swap_surfaces_window(self):
        a = _mk_arch(attention_type="full", swa_window=0)
        b = _mk_arch(attention_type="swa", swa_window=2048)
        names = {ch["field"] for ch in evaluator._arch_changes(a, b)}
        self.assertIn("attention.type", names)
        self.assertIn("attention.sliding_window", names)

    def test_state_default_placement_does_not_emit_ghost(self):
        """The CandidateArch defaults `placement_strategy="none"` and
        `n_state_layers=0` should not surface as diff rows when both
        sides hold the defaults.
        """
        a = _mk_arch(placement_strategy="none", n_state_layers=0)
        b = _mk_arch(placement_strategy="none", n_state_layers=0)
        changes = evaluator._arch_changes(a, b)
        names = {ch["field"] for ch in changes}
        self.assertNotIn("state.placement_strategy", names)
        self.assertNotIn("state.n_layers", names)

    def test_ep_degree_change_surfaces(self):
        a = _mk_arch(ep_degree=1, moe={"n_experts": 64, "top_k": 4,
                                       "expert_dim": 8192})
        b = _mk_arch(ep_degree=8, moe={"n_experts": 64, "top_k": 4,
                                       "expert_dim": 8192})
        names = {ch["field"] for ch in evaluator._arch_changes(a, b)}
        self.assertIn("parallelism.expert_parallel", names)

    def test_cp_method_change_surfaces(self):
        a = _mk_arch(cp_method="ring")
        b = _mk_arch(cp_method="ulysses")
        names = {ch["field"] for ch in evaluator._arch_changes(a, b)}
        self.assertIn("parallelism.cp_method", names)


class DedupeFieldChangesTests(unittest.TestCase):
    """Covers `_dedupe_field_changes` — the alias-collapsing pass that
    keeps the canonical row when the canonical path and the sidecar
    path both emit a row for the same conceptual field.
    """

    def test_empty_passthrough(self):
        self.assertEqual(evaluator._dedupe_field_changes([]), [])

    def test_no_duplicates_unchanged(self):
        rows = [
            {"field": "d_model", "baseline": 4096, "candidate": 4608},
            {"field": "n_layers", "baseline": 32, "candidate": 28},
        ]
        self.assertEqual(evaluator._dedupe_field_changes(rows), rows)

    def test_exact_duplicate_keeps_first(self):
        rows = [
            {"field": "attention.type", "baseline": "full", "candidate": "mla"},
            {"field": "attention.type", "baseline": "full", "candidate": "mla"},
        ]
        out = evaluator._dedupe_field_changes(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["field"], "attention.type")

    def test_alias_collapse_keeps_canonical(self):
        """`attention.mla_latent_dim` is the legacy sidecar label;
        `attention.mla_kv_latent_dim` is the canonical name. The dedup
        pass should keep whichever row arrived first (the canonical, by
        ordering convention) and drop the alias.
        """
        rows = [
            {"field": "attention.mla_kv_latent_dim",
             "baseline": 0, "candidate": 256},
            {"field": "attention.mla_latent_dim",
             "baseline": None, "candidate": 256},
        ]
        out = evaluator._dedupe_field_changes(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["field"], "attention.mla_kv_latent_dim")

    def test_alias_collapse_preserves_order(self):
        rows = [
            {"field": "d_model", "baseline": 4096, "candidate": 4608},
            {"field": "attention.mla_kv_latent_dim",
             "baseline": 0, "candidate": 256},
            {"field": "attention.mla_latent_dim",  # alias of the above
             "baseline": None, "candidate": 256},
            {"field": "n_layers", "baseline": 32, "candidate": 28},
        ]
        out = evaluator._dedupe_field_changes(rows)
        self.assertEqual(
            [ch["field"] for ch in out],
            ["d_model", "attention.mla_kv_latent_dim", "n_layers"],
        )


class MetricDeltaTests(unittest.TestCase):
    """Covers `_metric` — the helper that builds MetricDelta rows."""

    def test_zero_baseline_zero_delta_is_neutral(self):
        m = evaluator._metric("x", 0.0, 0.0)
        self.assertEqual(m.pct_change, 0.0)
        self.assertEqual(m.direction, "neutral")

    def test_zero_baseline_nonzero_delta_is_infinite(self):
        """The audit-mentioned Fix #4: real moves off a zero baseline
        must produce a signed +/-inf sentinel rather than silently
        reporting 0% change.
        """
        m = evaluator._metric("x", 0.0, 1.0)
        self.assertTrue(m.pct_change == float("inf"))
        # Direction is set by sign + lower_is_better convention.
        self.assertEqual(m.direction, "worsens")

    def test_lower_is_better_decreasing_improves(self):
        m = evaluator._metric("latency_ms", 10.0, 9.0, lower_is_better=True)
        self.assertEqual(m.direction, "improves")
        self.assertAlmostEqual(m.pct_change, -10.0, places=4)

    def test_higher_is_better_decreasing_worsens(self):
        m = evaluator._metric("tps", 1000.0, 900.0, lower_is_better=False)
        self.assertEqual(m.direction, "worsens")
        self.assertAlmostEqual(m.pct_change, -10.0, places=4)

    def test_pct_change_serializes_inf_as_none(self):
        """Audit Fix #4 invariant: math.inf is not strict JSON and
        breaks downstream parsers. The MetricDelta.as_dict() form must
        convert to None so the JSON is still valid.
        """
        m = evaluator._metric("x", 0.0, 1.0)
        d = m.as_dict()
        self.assertIsNone(d["pct_change"])


if __name__ == "__main__":
    unittest.main()
