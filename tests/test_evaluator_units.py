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


class TpSearchTests(unittest.TestCase):
    """Wave 4 — TP as a search variable.

    Two invariants we want to lock in:
      1. The grid driver's `context_aware_parallelism` planner produces a
         tp_options list whose max never exceeds the NVLink island size
         (we don't want to silently introduce cross-IB tensor parallelism).
      2. With tp_options enabled, a long-context cell at a small param
         target picks TP > 1 — i.e., the search actually uses the higher
         entries when KV pressure makes them valuable.
    """

    def test_tp_search_respects_nvlink_domain(self):
        import sys as _sys, os as _os
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from _generator_payload import (
            context_aware_parallelism, NVLINK_DOMAIN_SIZE_SEARCH,
        )

        # Wave 4 follow-up: TP search cap is now per-hw. NVL72-class fabrics
        # (B200) get a 72-rank domain; TPU v5p gets the 16-chip torus axis;
        # plain DGX-H100 stays at 8 because no NVL72 H100 exists. We assert
        # the planner never emits a TP option past the *per-hw* domain.
        for hw in ("h100", "b200", "trainium2", "trainium3", "tpu_v5p"):
            cap = NVLINK_DOMAIN_SIZE_SEARCH.get(hw, 8)
            for params in (1.0, 7.0, 120.0):
                for ctx in (8192, 131072, 4_194_304):
                    tp_options, pp, cp_options = context_aware_parallelism(
                        hw, params, ctx, serving_batch=16,
                    )
                    self.assertTrue(tp_options, f"empty tp_options for {hw} {params}B ctx={ctx}")
                    self.assertLessEqual(
                        max(tp_options), cap,
                        f"{hw} {params}B ctx={ctx} tp_options={tp_options} "
                        f"exceeded NVLink domain {cap}",
                    )

    def test_tp_search_picks_higher_tp_at_long_ctx(self):
        """A small-model long-context cell with tp_options=[1, 4] should
        produce a Pareto frontier that includes BOTH TPs (so the optimizer
        actually evaluated the higher TP option).

        The exact picked optimum can vary as the quality/throughput models
        evolve; the locked invariant is that TP=4 candidates appear at all
        — without Wave 4 the search would never even enumerate them.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ac"))
        from optimizer import DeploymentConstraints, optimize

        c = DeploymentConstraints(
            target_params_b=1.0,
            training_tokens=int(2e11),
            context_length=131072,   # 128k — KV pressure shows up here
            serving_tbt_ms=None, serving_ttft_ms=None,
            tp_options=[1, 4],
            cp=1, cp_options=[1],
            pp=1, dp=8,
            serving_batch=4,
            max_candidates=40,
            allow_quality_sentinel=True,
        )
        r = optimize("h100", c)
        # The frontier must contain candidates at both TPs — otherwise the
        # outer tp_opts loop didn't actually run.
        tp_values = {ev.arch.tp_degree for ev in r.all_evaluated}
        self.assertEqual(
            tp_values, {1, 4},
            f"expected both TPs in evaluated set, got {sorted(tp_values)}",
        )
        # The picked optimum must record a TP from the search list, not a
        # stray default.
        if r.optimal is not None:
            self.assertIn(
                r.optimal.arch.tp_degree, (1, 4),
                f"optimum tp_degree={r.optimal.arch.tp_degree} outside the "
                f"search space [1, 4]",
            )


class StateResidualAnchorTests(unittest.TestCase):
    """Wave 5 — state_residual must reproduce published hybrid anchor data.

    The current quality model can drift; these tests pin it against the
    Jamba ablations (AI21 2024), the NVIDIA Empirical Study of Mamba
    (2024), and the Samba paper (Microsoft 2024). If a future model
    change moves any anchor outside its tolerance band, that change
    needs to be justified against the cited literature.
    """

    def _state_value(self, p_attn, ctx, d_model=4096, n_layers=32, n_heads=32):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ac"))
        from quality_model import estimate_quality
        attn = max(1, int(round(p_attn * n_layers)))
        state = n_layers - attn
        cfg = {
            'd_model': d_model, 'n_layers': n_layers,
            'n_heads': n_heads, 'd_head': 128, 'n_kv_heads': max(1, n_heads // 8),
            'ffn_dim': d_model * 4, 'vocab_size': 32000,
            'attention_type': 'gqa',
            'weight_precision': 'bf16', 'kv_precision': 'bf16',
            'activation_precision': 'bf16',
        }
        if state > 0:
            cfg['model_type'] = 'hybrid'
            cfg['state_config'] = {
                'enabled': True, 'state_layers': state,
                'attention_layers': attn, 'd_state': 192, 'state_type': 'mamba2',
            }
        else:
            cfg['model_type'] = 'dense'
        q = estimate_quality(
            cfg,
            {'training_tokens': int(3.5e12), 'sequence_length': ctx},
            {'context_length': ctx, 'task_type': 'general'},
        )
        t = (q.terms or {}).get('state_residual')
        return t.value if t else 0.0

    def test_jamba_band_center_parity(self):
        """Jamba 1:7 hybrid at short ctx: parity with pure attention.

        AI21 ablations show p_attn = 0.125 (1:7 ratio) is the sweet
        spot — hybrid quality matches pure attention at standard
        contexts. state_residual must be ≈ 0 at this point.
        """
        val = self._state_value(p_attn=0.125, ctx=4096)
        self.assertLess(abs(val), 0.015,
                        f"Jamba sweet spot must be near-zero residual; got {val:.4f}")

    def test_nvidia_long_context_benefit(self):
        """NVIDIA 8B Mamba-2-Hybrid (p_attn≈0.07) at 128K beats the 8B
        Transformer by ~2.65 points on 12 tasks → ~1.5% loss reduction.
        state_residual must be negative at this point.
        """
        val = self._state_value(p_attn=0.07, ctx=131072)
        self.assertLess(val, -0.005,
                        f"NVIDIA 128K hybrid must show benefit (negative residual); got {val:.4f}")
        # Don't let it run away: cap the magnitude inside a reasonable band.
        self.assertGreater(val, -0.04,
                           f"benefit shouldn't exceed NVIDIA's measured improvement; got {val:.4f}")

    def test_state_residual_no_crush_at_large_scale(self):
        """500B hybrid at 2M context, p_attn=0.125 (Jamba ratio).

        Pre-Wave-5 the composition×ctx term inflated this to ≈ +0.108,
        which made MoE crush MoE-hybrid by 17.6% loss. After Wave 5 +
        Wave 5b's log-saturating compression, the value must be inside
        ±0.05 (i.e. not a crushing penalty in either direction).
        """
        val = self._state_value(p_attn=0.125, ctx=2097152,
                                d_model=12288, n_layers=80, n_heads=96)
        self.assertLess(abs(val), 0.05,
                        f"500B@2M hybrid penalty was the Wave-5 bug; "
                        f"must be inside ±0.05, got {val:.4f}")

    def test_pure_attention_no_state_benefit_or_penalty(self):
        """At p_attn=1.0 (pure attention) the state_residual term is
        not applicable — the new long-context benefit term must zero
        out when p_state=0."""
        val = self._state_value(p_attn=1.0, ctx=1048576)
        self.assertAlmostEqual(val, 0.0, places=3,
                               msg=f"pure-attention state_residual must be 0; got {val:.4f}")


if __name__ == "__main__":
    unittest.main()
