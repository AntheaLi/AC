"""Wave 18c tests — ServingWorkloadSpec and canonical scenarios.

Covers the six bullets from spec §Tests that live at the workload layer:

  * Changing max_context_length alone does not create fresh prefill work.
  * Cached and cold 1M workloads have same decode KV cost but different
    prefill cost.
  * Disaggregated prefill/decode preserves separate metrics and topology.

Plus validation-invariant tests.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ac.serving_workload import (
    CACHE_HIT_RATE_CACHED,
    CACHE_HIT_RATE_INCREMENTAL,
    INCREMENTAL_SESSION_FRESH_PROMPT,
    INTERACTIVE_FRESH_PROMPT,
    MEMORY_TIER_HBM,
    MEMORY_TIER_HBM_SPILL,
    TOPOLOGY_COLOCATED,
    TOPOLOGY_DISAGGREGATED,
    ServingWorkloadSpec,
    WorkloadRegistry,
    WorkloadScenario,
    canonical_workloads,
)


class ValidationTests(unittest.TestCase):

    def _base(self, **overrides):
        kw = dict(
            name="test", max_context_length=32768,
            fresh_prompt_tokens=8192, cached_prefix_tokens=0,
            decode_kv_length=8192, output_tokens=512,
            cache_hit_rate=0.0, serving_batch=16, concurrency=256,
            available_gpus=64,
            allowed_memory_tiers=(MEMORY_TIER_HBM,),
            prefill_topology=TOPOLOGY_COLOCATED,
            decode_topology=TOPOLOGY_COLOCATED,
        )
        kw.update(overrides)
        return kw

    def test_fresh_plus_cached_cannot_exceed_max_context(self):
        with self.assertRaises(ValueError):
            ServingWorkloadSpec(**self._base(
                fresh_prompt_tokens=10000, cached_prefix_tokens=30000,
                max_context_length=32768,
            ))

    def test_decode_kv_bounded_by_max_context(self):
        with self.assertRaises(ValueError):
            ServingWorkloadSpec(**self._base(
                max_context_length=8192, decode_kv_length=32768,
            ))

    def test_cache_hit_rate_in_unit_interval(self):
        with self.assertRaises(ValueError):
            ServingWorkloadSpec(**self._base(cache_hit_rate=1.5))
        with self.assertRaises(ValueError):
            ServingWorkloadSpec(**self._base(cache_hit_rate=-0.1))

    def test_positive_batch_concurrency_gpus(self):
        for field in ("serving_batch", "concurrency", "available_gpus"):
            with self.assertRaises(ValueError):
                ServingWorkloadSpec(**self._base(**{field: 0}))

    def test_topology_values_validated(self):
        with self.assertRaises(ValueError):
            ServingWorkloadSpec(**self._base(prefill_topology="magic"))


class CanonicalScenarioTests(unittest.TestCase):
    """Spec §Canonical scenarios — the five reference workloads per ctx row."""

    def test_all_five_scenarios_generated(self):
        ws = canonical_workloads(131072)
        expected = {s.value for s in WorkloadScenario}
        self.assertEqual(set(ws.keys()), expected)

    def test_interactive_ordinary_has_no_cached_prefix(self):
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        self.assertEqual(w.cached_prefix_tokens, 0)
        self.assertEqual(w.cache_hit_rate, 0.0)
        self.assertEqual(w.fresh_prompt_tokens,
                         min(131072, INTERACTIVE_FRESH_PROMPT))
        # decode KV equals live context after prefill = fresh prompt.
        self.assertEqual(w.decode_kv_length, w.fresh_prompt_tokens)

    def test_interactive_cached_long_context_uses_full_kv(self):
        w = canonical_workloads(int(1e6))[
            WorkloadScenario.INTERACTIVE_CACHED_LONG_CONTEXT.value]
        self.assertEqual(w.fresh_prompt_tokens, INTERACTIVE_FRESH_PROMPT)
        self.assertEqual(w.cached_prefix_tokens, int(1e6) - INTERACTIVE_FRESH_PROMPT)
        self.assertEqual(w.decode_kv_length, int(1e6))
        self.assertEqual(w.cache_hit_rate, CACHE_HIT_RATE_CACHED)

    def test_incremental_session_uses_bigger_fresh(self):
        w = canonical_workloads(int(1e6))[
            WorkloadScenario.INCREMENTAL_LONG_SESSION.value]
        self.assertEqual(w.fresh_prompt_tokens, INCREMENTAL_SESSION_FRESH_PROMPT)
        self.assertEqual(w.cached_prefix_tokens,
                         int(1e6) - INCREMENTAL_SESSION_FRESH_PROMPT)
        self.assertEqual(w.decode_kv_length, int(1e6))
        self.assertEqual(w.cache_hit_rate, CACHE_HIT_RATE_INCREMENTAL)

    def test_cold_full_ingestion_pays_full_prefill(self):
        w = canonical_workloads(int(2e6))[
            WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value]
        self.assertEqual(w.fresh_prompt_tokens, int(2e6))
        self.assertEqual(w.cached_prefix_tokens, 0)
        self.assertEqual(w.decode_kv_length, int(2e6))
        self.assertTrue(w.is_cold_ingestion)


class ContextIndependenceTests(unittest.TestCase):
    """Spec §Tests bullet 1 — changing max_context_length alone does not
    create fresh prefill work for the interactive paths."""

    def test_interactive_ordinary_fresh_prompt_is_flat_above_8k(self):
        for C in [32_768, 131_072, 1_048_576, 4_194_304]:
            w = canonical_workloads(C)[
                WorkloadScenario.INTERACTIVE_ORDINARY.value]
            self.assertEqual(w.fresh_prompt_tokens, INTERACTIVE_FRESH_PROMPT,
                f"ordinary fresh prompt must stay at 8k at C={C}")

    def test_interactive_cached_fresh_prompt_is_flat_above_8k(self):
        for C in [32_768, 131_072, 1_048_576, 4_194_304]:
            w = canonical_workloads(C)[
                WorkloadScenario.INTERACTIVE_CACHED_LONG_CONTEXT.value]
            self.assertEqual(w.fresh_prompt_tokens, INTERACTIVE_FRESH_PROMPT,
                f"cached fresh prompt must stay at 8k at C={C}")

    def test_incremental_session_fresh_prompt_is_flat_above_32k(self):
        for C in [131_072, 1_048_576, 4_194_304]:
            w = canonical_workloads(C)[
                WorkloadScenario.INCREMENTAL_LONG_SESSION.value]
            self.assertEqual(w.fresh_prompt_tokens,
                             INCREMENTAL_SESSION_FRESH_PROMPT,
                f"incremental fresh prompt must stay at 32k at C={C}")


class CachedVsColdCostTests(unittest.TestCase):
    """Spec §Tests bullet 2 — cached and cold 1M workloads have same decode KV
    but different prefill cost."""

    def test_decode_kv_matches_between_cached_and_cold(self):
        C = 1_048_576
        ws = canonical_workloads(C)
        cached = ws[WorkloadScenario.INTERACTIVE_CACHED_LONG_CONTEXT.value]
        cold = ws[WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value]
        self.assertEqual(cached.decode_kv_length, cold.decode_kv_length)
        self.assertEqual(cached.decode_kv_length, C)

    def test_prefill_effective_tokens_differs(self):
        C = 1_048_576
        ws = canonical_workloads(C)
        cached = ws[WorkloadScenario.INTERACTIVE_CACHED_LONG_CONTEXT.value]
        cold = ws[WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value]
        self.assertLess(cached.effective_prefill_tokens,
                        cold.effective_prefill_tokens,
                        "cached workload must charge fewer effective prefill tokens")
        # Cold ingestion charges the full 1M.
        self.assertEqual(cold.effective_prefill_tokens, C)
        # Cached charges fresh (8k) + 10% of cached (~104k).
        expected_cached = 8192 + int(round(0.10 * (C - 8192)))
        self.assertEqual(cached.effective_prefill_tokens, expected_cached)


class DisaggregationTests(unittest.TestCase):
    """Spec §Tests bullet 6 — disaggregated prefill/decode preserves separate
    metrics and topology at the workload layer."""

    def test_disaggregated_flag_toggles(self):
        w = canonical_workloads(
            131072,
            prefill_topology=TOPOLOGY_DISAGGREGATED,
            decode_topology=TOPOLOGY_COLOCATED,
        )[WorkloadScenario.INTERACTIVE_ORDINARY.value]
        self.assertTrue(w.is_disaggregated)
        self.assertEqual(w.prefill_topology, TOPOLOGY_DISAGGREGATED)
        self.assertEqual(w.decode_topology, TOPOLOGY_COLOCATED)

    def test_colocated_default(self):
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        self.assertFalse(w.is_disaggregated)


class SpillTierTests(unittest.TestCase):

    def test_spill_not_permitted_by_default(self):
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        self.assertFalse(w.spill_permitted)

    def test_spill_opt_in(self):
        w = canonical_workloads(
            131072,
            allowed_memory_tiers=(MEMORY_TIER_HBM, MEMORY_TIER_HBM_SPILL),
        )[WorkloadScenario.INTERACTIVE_ORDINARY.value]
        self.assertTrue(w.spill_permitted)


class WorkloadRegistryTests(unittest.TestCase):

    def test_register_canonical_populates_all_five(self):
        r = WorkloadRegistry()
        r.register_canonical(131072)
        self.assertEqual(len(r.workloads), 5)
        self.assertEqual({w.name for w in r.workloads.values()},
                         {s.value for s in WorkloadScenario})

    def test_as_dict_roundtrip(self):
        r = WorkloadRegistry()
        r.register_canonical(131072)
        d = r.as_dict()
        self.assertEqual(len(d), 5)
        # Each entry is serializable.
        for name, w in d.items():
            self.assertEqual(w["name"], name)
            self.assertIn("fresh_prompt_tokens", w)


if __name__ == "__main__":
    unittest.main()
