"""Wave 11 — cell-output schema cleanup regression tests.

Per plan/redesign/11-cell-output-schema-cleanup.md. Locks in:

  * Every grid row gains a `decision` block with the canonical
    "answer + uncertainty" fields.
  * Every grid row gains a `diagnostics` block with alternates, shadow
    prices, search stats, smoothing flags, and justification.
  * Legacy `optimal` / `pareto` / etc. fields remain populated for the
    transition period (back-compat with the web app, CLI, family-view
    renderer).
  * `decision.predicted_loss` matches the legacy `optimal.loss` for the
    same winner — no behavior change.
  * Rows with no feasible solution emit a minimal `decision` with
    `no_feasible_solution: True`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _synthetic_row(family="dense", with_optimal=True):
    """A minimal synthetic grid row for testing the schema split helper."""
    row = {
        "hw": "h100", "params_B": 7, "tokens_T": 2.0,
        "context_length": 8192, "arch_mode": family,
        "state_type": None,
        "tp": 1, "pp": 1, "dp": 1,
        "candidates": 100, "feasible": 80,
        "candidates_enumerated_raw": 12000,
        "pareto_size": 12,
        "time_s": 1.5,
        "shadow_prices": [{"desc": "...", "delta_pct": 0.5}],
        "justification": "synthetic",
        "pareto": [{"d_model": 4096, "loss": 2.05},
                   {"d_model": 4096, "loss": 2.06}],
    }
    if with_optimal:
        opt = {
            "d_model": 4096, "n_layers": 32, "n_heads": 32, "d_head": 128,
            "n_kv_heads": 8, "ffn_dim": 14336, "vocab_size": 128000,
            "weight_prec": "bf16", "ffn_prec": "bf16", "kv_bits": 16,
            "tp": 1, "pp": 1, "ep_degree": 1, "cp_degree": 1, "dp": 1,
            "loss": 2.020, "uncertainty_total_pct": 3.0,
            "tbt_ms": 12.0, "ttft_ms": 50.0, "mem_gb": 30.0,
            "train_tps": 8000, "confidence": "medium",
            "attention_type": "gqa",
            "quality_terms": {"architecture_residual": {"value": 0.01}},
        }
        if family in ("moe", "moe_hybrid"):
            opt.update({"n_experts": 8, "top_k": 2, "expert_dim": 4096})
        if family in ("hybrid", "moe_hybrid"):
            opt.update({"n_state_layers": 24, "n_attention_layers": 8,
                       "placement_strategy": "first_periodic_last"})
        row["optimal"] = opt
    return row


class DecisionBlockCompletenessTests(unittest.TestCase):
    """Every populated row must have every decision.* field present."""

    def test_decision_block_completeness_dense(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("dense")
        _build_decision_diagnostics(row)
        d = row["decision"]
        # Required scalar fields
        for k in ("predicted_loss", "predicted_loss_uncertainty_pct",
                  "predicted_tbt_ms", "predicted_ttft_ms",
                  "predicted_mem_gb", "predicted_training_tps",
                  "confidence", "family"):
            self.assertIn(k, d, f"decision missing scalar field: {k}")
        # Required sub-blocks
        for k in ("shape", "precision", "parallelism", "attention"):
            self.assertIn(k, d, f"decision missing sub-block: {k}")
        self.assertEqual(d["family"], "dense")
        # MoE/state sub-blocks should be None for a dense row.
        self.assertIsNone(d["moe"])
        self.assertIsNone(d["state"])

    def test_decision_block_completeness_moe_hybrid(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("moe_hybrid")
        _build_decision_diagnostics(row)
        d = row["decision"]
        self.assertEqual(d["family"], "moe_hybrid")
        self.assertIsNotNone(d["moe"])
        self.assertIsNotNone(d["state"])
        self.assertEqual(d["moe"]["n_experts"], 8)
        self.assertEqual(d["state"]["n_state_layers"], 24)


class DiagnosticsSeparableTests(unittest.TestCase):
    """diagnostics is fully separable from decision."""

    def test_diagnostics_block_present(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("dense")
        _build_decision_diagnostics(row)
        diag = row["diagnostics"]
        for k in ("alternatives", "shadow_prices", "arch_dim_shadow_prices",
                  "quality_residual_breakdown", "search_stats",
                  "smoothing", "justification"):
            self.assertIn(k, diag, f"diagnostics missing field: {k}")
        # Search stats populated from row-level counters.
        self.assertEqual(diag["search_stats"]["candidates_generated"], 100)
        self.assertEqual(diag["search_stats"]["pareto_size"], 12)


class BackCompatLegacyTests(unittest.TestCase):
    """Legacy fields must still be populated after the helper runs."""

    def test_legacy_optimal_field_preserved(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("dense")
        _build_decision_diagnostics(row)
        # Legacy `optimal` field still present and untouched.
        self.assertIn("optimal", row)
        self.assertEqual(row["optimal"]["d_model"], 4096)
        self.assertEqual(row["optimal"]["loss"], 2.020)

    def test_legacy_pareto_preserved(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("dense")
        _build_decision_diagnostics(row)
        # Legacy `pareto` array still present.
        self.assertIn("pareto", row)
        self.assertEqual(len(row["pareto"]), 2)


class DecisionConsistencyTests(unittest.TestCase):
    """decision.predicted_loss must equal legacy optimal.loss."""

    def test_decision_consistency_with_legacy(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("dense")
        _build_decision_diagnostics(row)
        self.assertEqual(
            row["decision"]["predicted_loss"],
            row["optimal"]["loss"],
            "decision.predicted_loss must equal optimal.loss — Wave 11 "
            "must be a pure refactor with no behavior change."
        )


class NoFeasibleSolutionTests(unittest.TestCase):
    """Rows with no feasible solution get a minimal decision block."""

    def test_no_feasible_solution_decision(self):
        from _generator_payload import _build_decision_diagnostics
        row = _synthetic_row("dense", with_optimal=False)
        # Simulate a no-feasible row: clear `optimal`.
        row["optimal"] = None
        _build_decision_diagnostics(row)
        d = row["decision"]
        self.assertTrue(d.get("no_feasible_solution"))
        self.assertIsNone(d.get("predicted_loss"))
        # diagnostics still present (search stats, justification).
        self.assertIn("diagnostics", row)
        self.assertIn("search_stats", row["diagnostics"])


if __name__ == "__main__":
    unittest.main()
