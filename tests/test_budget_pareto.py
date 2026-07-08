"""Wave 18b tests — budget-matched comparisons and scenario Pareto frontiers.

Covers the six bullets from `plan/redesign/18b-budget-and-serving-pareto.md`
§Tests:

  1. A 120B-total/5B-active MoE appears near the 5B active column and the
     120B total column, never the 120B active column.
  2. Equal-budget tolerances are enforced using measured candidate metrics.
  3. A slow but physical candidate remains in diagnostics.
  4. Epsilon-equivalent candidates collapse into one Pareto neighborhood.
  5. TBT, TTFT, throughput, and GPU-seconds can independently change dominance.
  6. Every rendered row exposes active/total size and topology.

Plus a few sanity checks on the derived metrics, the JSON/Markdown renderers,
and the matrix-level driver.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ac.budget_pareto import (
    BudgetMatrix,
    CandidateMetrics,
    ComparisonBudget,
    Matrix,
    MatrixCell,
    MatrixKey,
    ParetoScenario,
    Topology,
    epsilon_dominated,
    extract_metrics,
    match_budget,
    pareto_frontier,
    render_json,
    render_markdown,
    render_matrix_json,
    render_matrix_markdown,
)


def make_metric(
    *,
    identity: str = "dense",
    active: float = 7e9,
    total: float = 7e9,
    loss: float = 2.0,
    tbt: float = 20.0,
    ttft: float = 200.0,
    mem: float = 40.0,
    gpu_s_req: float = 0.02,
    train_flops: float = 1e20,
    train_gpu_s: float = 1e5,
    replica_gpus: int = 8,
    agg_tps: float = 1000.0,
    rps: float = 2.0,
    tp: int = 8, pp: int = 1, ep: int = 1, cp: int = 1, dp: int = 8, batch: int = 16,
) -> CandidateMetrics:
    return CandidateMetrics(
        identity_label=identity,
        active_params=int(active),
        total_params=int(total),
        training_flops=train_flops,
        training_gpu_seconds=train_gpu_s,
        prefill_gpu_seconds=ttft / 1000.0 * replica_gpus,
        decode_gpu_seconds=tbt / 1000.0 * 512 * replica_gpus,
        serving_gpu_seconds_per_request=gpu_s_req,
        replica_gpus=replica_gpus,
        aggregate_output_tokens_per_second=agg_tps,
        requests_per_second=rps,
        predicted_loss=loss,
        ttft_ms=ttft,
        tbt_ms=tbt,
        memory_per_gpu_gb=mem,
        topology=Topology(tp=tp, pp=pp, ep=ep, cp=cp, dp=dp, serving_batch=batch),
    )


class ClassificationBucketsTests(unittest.TestCase):
    """Spec §Tests #1 — 120B-total / 5B-active MoE must NOT land in the
    120B active column."""

    def test_moe_lands_in_active_column_not_total_column(self):
        moe = make_metric(identity="moe", active=5e9, total=120e9)
        dense_5b_anchor = make_metric(identity="dense", active=5e9, total=5e9)
        dense_120b_anchor = make_metric(identity="dense", active=120e9, total=120e9)

        # Equal-active projection at the 5B anchor: MoE included.
        matched_active_5b = match_budget(
            moe, ComparisonBudget.EQUAL_ACTIVE_PARAMS,
            float(dense_5b_anchor.active_params),
        )
        self.assertTrue(matched_active_5b,
                        "5B-active MoE must match equal_active_params at 5B anchor")

        # Equal-active projection at the 120B anchor: MoE excluded.
        matched_active_120b = match_budget(
            moe, ComparisonBudget.EQUAL_ACTIVE_PARAMS,
            float(dense_120b_anchor.active_params),
        )
        self.assertFalse(matched_active_120b,
                         "5B-active MoE MUST NOT match equal_active_params at 120B anchor")

        # Equal-total projection at the 120B anchor: MoE included.
        matched_total_120b = match_budget(
            moe, ComparisonBudget.EQUAL_TOTAL_PARAMS,
            float(dense_120b_anchor.total_params),
        )
        self.assertTrue(matched_total_120b,
                        "120B-total MoE must match equal_total_params at 120B anchor")


class BudgetToleranceTests(unittest.TestCase):
    """Spec §Tests #2 — tolerances (±5% training, ±10% serving cost) are
    enforced on measured candidate metrics."""

    def test_training_flops_pm5pct(self):
        anchor = make_metric(train_flops=1e20)
        just_inside = make_metric(train_flops=1.04e20)
        edge_inside = make_metric(train_flops=1.049e20)
        just_outside = make_metric(train_flops=1.06e20)
        self.assertTrue(match_budget(just_inside,
            ComparisonBudget.EQUAL_TRAINING_FLOPS, anchor.training_flops))
        self.assertTrue(match_budget(edge_inside,
            ComparisonBudget.EQUAL_TRAINING_FLOPS, anchor.training_flops))
        self.assertFalse(match_budget(just_outside,
            ComparisonBudget.EQUAL_TRAINING_FLOPS, anchor.training_flops))

    def test_serving_cost_pm10pct(self):
        anchor = make_metric(gpu_s_req=0.02)
        inside = make_metric(gpu_s_req=0.021)  # +5%
        edge_inside = make_metric(gpu_s_req=0.022)  # +10%
        outside = make_metric(gpu_s_req=0.025)  # +25%
        b = ComparisonBudget.EQUAL_SERVING_GPU_SECONDS_PER_REQUEST
        self.assertTrue(match_budget(inside, b, anchor.serving_gpu_seconds_per_request))
        self.assertTrue(match_budget(edge_inside, b,
                        anchor.serving_gpu_seconds_per_request))
        self.assertFalse(match_budget(outside, b,
                        anchor.serving_gpu_seconds_per_request))

    def test_training_gpu_seconds_uses_training_tolerance(self):
        anchor = make_metric(train_gpu_s=1e6)
        edge_inside = make_metric(train_gpu_s=1.049e6)  # +4.9%
        just_outside = make_metric(train_gpu_s=1.06e6)  # +6%
        b = ComparisonBudget.EQUAL_TRAINING_GPU_SECONDS
        self.assertTrue(match_budget(edge_inside, b, anchor.training_gpu_seconds))
        self.assertFalse(match_budget(just_outside, b, anchor.training_gpu_seconds))


class DiagnosticFrontierPreservationTests(unittest.TestCase):
    """Spec §Tests #3 — a physically valid but operationally extreme
    candidate remains in the diagnostic frontier."""

    def test_slow_but_physical_candidate_in_unconstrained_diagnostic(self):
        # An outrageous candidate that would fail every operational threshold.
        outrageous = make_metric(
            identity="dense", loss=2.0, tbt=99999.0, ttft=999999.0,
            gpu_s_req=99.0, replica_gpus=4096,
        )
        good = make_metric(identity="dense", loss=2.0, tbt=10.0, ttft=100.0)
        frontier = pareto_frontier(
            [outrageous, good], ParetoScenario.UNCONSTRAINED_DIAGNOSTIC)
        # Diagnostic scenario has no axes so nothing is dominated.
        self.assertEqual(len(frontier), 2,
            "unconstrained_diagnostic must preserve every physical candidate")

    def test_slow_candidate_survives_interactive_when_it_wins_a_different_axis(self):
        # Slow-but-cheap: high TBT but low replica_gpus and low training FLOPs.
        cheap_slow = make_metric(identity="dense", loss=2.0, tbt=500.0,
                                 replica_gpus=1)
        fast_expensive = make_metric(identity="dense", loss=2.0, tbt=15.0,
                                     replica_gpus=64)
        frontier = pareto_frontier(
            [cheap_slow, fast_expensive], ParetoScenario.INTERACTIVE_SERVING)
        ids = {id(m) for m in frontier}
        self.assertIn(id(cheap_slow), ids,
            "cheap_slow must remain on interactive frontier (wins on replica_gpus)")
        self.assertIn(id(fast_expensive), ids,
            "fast_expensive must remain on interactive frontier (wins on tbt)")


class EpsilonEquivalenceTests(unittest.TestCase):
    """Spec §Tests #4 — epsilon-equivalent candidates collapse into one
    Pareto neighborhood."""

    def test_epsilon_ties_do_not_dominate(self):
        # Two candidates within 0.5% loss (below the 1% eps).
        a = make_metric(loss=1.900, tbt=20.0, ttft=200.0, gpu_s_req=0.02,
                        replica_gpus=8)
        b = make_metric(loss=1.909, tbt=20.0, ttft=200.0, gpu_s_req=0.02,
                        replica_gpus=8)
        self.assertFalse(epsilon_dominated(a, b, ParetoScenario.INTERACTIVE_SERVING),
            "b does not strictly beat a on any axis by > eps")
        self.assertFalse(epsilon_dominated(b, a, ParetoScenario.INTERACTIVE_SERVING),
            "a does not strictly beat b on any axis by > eps")

    def test_beyond_epsilon_dominance(self):
        # a is strictly worse than b on loss by 5% (well above 1% eps).
        worse = make_metric(loss=2.10, tbt=20.0, ttft=200.0, gpu_s_req=0.02,
                            replica_gpus=8)
        better = make_metric(loss=2.00, tbt=20.0, ttft=200.0, gpu_s_req=0.02,
                             replica_gpus=8)
        self.assertTrue(epsilon_dominated(worse, better,
                        ParetoScenario.INTERACTIVE_SERVING))


class IndependentAxisDominanceTests(unittest.TestCase):
    """Spec §Tests #5 — TBT, TTFT, throughput, GPU-seconds can independently
    change dominance."""

    def test_ttft_alone_removes_dominance(self):
        base = make_metric(loss=2.0, tbt=20.0, ttft=200.0, gpu_s_req=0.02,
                           replica_gpus=8)
        # Same as base except huge TTFT — should remove dominance in
        # interactive but survive throughput (where TTFT isn't an axis).
        big_ttft = make_metric(loss=2.0, tbt=20.0, ttft=200000.0,
                               gpu_s_req=0.02, replica_gpus=8)
        # In INTERACTIVE, base epsilon-dominates big_ttft (better TTFT).
        self.assertTrue(epsilon_dominated(big_ttft, base,
                        ParetoScenario.INTERACTIVE_SERVING))
        # In THROUGHPUT, TTFT is not an axis → base does NOT dominate on TTFT.
        # They should tie on every axis THROUGHPUT cares about.
        self.assertFalse(epsilon_dominated(big_ttft, base,
                         ParetoScenario.THROUGHPUT_SERVING))

    def test_throughput_axis_swaps_winner(self):
        # A has lower loss but low throughput; B has higher loss but massive
        # throughput. Under INTERACTIVE, A dominates. Under THROUGHPUT, neither.
        a = make_metric(loss=1.90, tbt=25.0, ttft=250.0, gpu_s_req=0.03,
                        replica_gpus=8, agg_tps=500.0, rps=1.0)
        b = make_metric(loss=1.95, tbt=25.0, ttft=250.0, gpu_s_req=0.03,
                        replica_gpus=8, agg_tps=10000.0, rps=25.0)
        # INTERACTIVE doesn't use agg_tps/rps, so A dominates B on loss.
        self.assertTrue(epsilon_dominated(b, a,
                        ParetoScenario.INTERACTIVE_SERVING))
        # THROUGHPUT uses agg_tps and rps; the two are incomparable.
        self.assertFalse(epsilon_dominated(b, a,
                         ParetoScenario.THROUGHPUT_SERVING))
        self.assertFalse(epsilon_dominated(a, b,
                         ParetoScenario.THROUGHPUT_SERVING))

    def test_training_frontier_is_orthogonal_to_serving(self):
        # High-loss but very cheap-to-train candidate vs low-loss expensive.
        cheap_train = make_metric(loss=2.00, train_flops=1e19,
                                  train_gpu_s=1e4)
        pricey_train = make_metric(loss=1.95, train_flops=1e21,
                                   train_gpu_s=1e6)
        # In TRAINING, both survive: cheap_train wins FLOPs & GPU-s, pricey
        # wins loss. Neither eps-dominates.
        self.assertFalse(epsilon_dominated(cheap_train, pricey_train,
                         ParetoScenario.TRAINING))
        self.assertFalse(epsilon_dominated(pricey_train, cheap_train,
                         ParetoScenario.TRAINING))


class RendererExposesSizeAndTopologyTests(unittest.TestCase):
    """Spec §Tests #6 — every rendered row exposes active/total size and topology."""

    def _build_cell(self) -> MatrixCell:
        driver = BudgetMatrix(hardware="h100", context_length=131072,
                              training_tokens=int(20e12))
        anchor = make_metric(identity="dense", active=70e9, total=70e9,
                             loss=1.9, tbt=15.0, ttft=200.0)
        driver.add(anchor)
        driver.add(make_metric(identity="moe", active=37e9, total=671e9,
                               loss=1.85, tbt=25.0))
        driver.add(make_metric(identity="hybrid", active=70e9, total=70e9,
                               loss=1.92, tbt=8.0))
        return driver.build_cell(anchor)

    def test_markdown_contains_active_total_topology_per_row(self):
        cell = self._build_cell()
        md = render_markdown(cell)
        # Header presence
        self.assertIn("active", md.lower())
        self.assertIn("total", md.lower())
        self.assertIn("topology", md.lower())
        # A specific topology label from Topology.label()
        self.assertIn("TP=", md)
        self.assertIn("batch=", md)
        # Every non-empty frontier row must show both active and total
        # values (in "B" format from _fmt_params). Assert on a known cell.
        self.assertIn("70.00B", md)  # dense reference

    def test_json_contains_active_total_topology_per_row(self):
        cell = self._build_cell()
        js = render_json(cell)
        views = js["budget_views"]
        for view_name, view in views.items():
            for m in view["matched"]:
                self.assertIn("active_params", m)
                self.assertIn("total_params", m)
                self.assertIn("topology", m)
                self.assertIn("tp", m["topology"])
                self.assertIn("serving_batch", m["topology"])
        for name, sf in js["scenario_frontiers"].items():
            for m in sf["frontier"]:
                self.assertIn("active_params", m)
                self.assertIn("total_params", m)
                self.assertIn("topology", m)


class MetricExtractionTests(unittest.TestCase):
    """Sanity checks on extract_metrics — derived metrics come from the
    surrounding EvaluatedCandidate + DeploymentConstraints pair."""

    def _mkev(self, *, tp=8, pp=1, ep=1, cp=1, ffn=14336, d=4096, L=32,
              tbt=20.0, ttft=200.0, mem=40.0, train_tps=5000.0, loss=1.9,
              moe=None):
        # Minimal synthetic EvaluatedCandidate-shaped object.
        class Arch:
            pass
        arch = Arch()
        arch.d_model = d
        arch.n_layers = L
        arch.n_heads = 32
        arch.n_kv_heads = 8
        arch.d_head = 128
        arch.ffn_dim = ffn
        arch.vocab_size = 32000
        arch.moe = moe
        arch.state_config = None
        arch.n_dense_ffn_layers = 0
        arch.tp_degree = tp
        arch.pp_degree = pp
        arch.ep_degree = ep
        arch.cp_degree = cp

        class Throughput:
            def __init__(self):
                self.prefill_time_ms = ttft
        tput = Throughput()

        class Ev:
            pass
        ev = Ev()
        ev.arch = arch
        ev.predicted_loss = loss
        ev.training_tps = train_tps
        ev.serving_tbt_ms = tbt
        ev.serving_ttft_ms = ttft
        ev.memory_per_gpu_gb = mem
        ev.throughput = tput
        return ev

    def _mkconstraints(self, *, dp=8, serving_batch=16, output_len=512,
                       training_tokens=int(20e12)):
        class C:
            pass
        c = C()
        c.dp = dp
        c.serving_batch = serving_batch
        c.output_len = output_len
        c.training_tokens = training_tokens
        return c

    def test_active_equals_total_for_dense(self):
        ev = self._mkev()
        c = self._mkconstraints()
        m = extract_metrics(ev, c)
        self.assertEqual(m.identity_label, "dense")
        self.assertGreater(m.active_params, 0)
        self.assertEqual(m.active_params, m.total_params)

    def test_replica_gpus_is_tp_pp_cp(self):
        ev = self._mkev(tp=8, pp=2, cp=2)
        c = self._mkconstraints()
        m = extract_metrics(ev, c)
        self.assertEqual(m.replica_gpus, 8 * 2 * 2)
        self.assertEqual(m.topology.tp, 8)
        self.assertEqual(m.topology.pp, 2)
        self.assertEqual(m.topology.cp, 2)

    def test_topology_carries_serving_batch(self):
        ev = self._mkev()
        c = self._mkconstraints(serving_batch=32)
        m = extract_metrics(ev, c)
        self.assertEqual(m.topology.serving_batch, 32)

    def test_serving_gpu_seconds_per_request_scales_inversely_with_batch(self):
        ev = self._mkev()
        c_small = self._mkconstraints(serving_batch=1)
        c_big = self._mkconstraints(serving_batch=32)
        m_small = extract_metrics(ev, c_small)
        m_big = extract_metrics(ev, c_big)
        # Per-request cost should be lower at larger batch (per spec).
        self.assertLess(m_big.serving_gpu_seconds_per_request,
                        m_small.serving_gpu_seconds_per_request)

    def test_training_flops_scales_with_active_and_tokens(self):
        ev = self._mkev()
        c = self._mkconstraints(training_tokens=int(2e12))
        m2t = extract_metrics(ev, c)
        c20 = self._mkconstraints(training_tokens=int(20e12))
        m20 = extract_metrics(ev, c20)
        # 10× tokens → 10× training FLOPs.
        self.assertAlmostEqual(m20.training_flops / m2t.training_flops, 10.0,
                               places=3)


class BudgetMatrixDriverTests(unittest.TestCase):
    """Sanity checks on the BudgetMatrix driver."""

    def test_build_cell_emits_all_five_budget_views(self):
        driver = BudgetMatrix(hardware="h100", context_length=131072,
                              training_tokens=int(20e12))
        anchor = make_metric(active=70e9, total=70e9)
        driver.add(anchor)
        driver.add(make_metric(identity="moe", active=37e9, total=671e9))
        cell = driver.build_cell(anchor)
        expected = {b.value for b in ComparisonBudget}
        self.assertEqual(set(cell.budget_views.keys()), expected)

    def test_build_cell_emits_all_four_scenario_frontiers(self):
        driver = BudgetMatrix(hardware="h100", context_length=131072,
                              training_tokens=int(20e12))
        anchor = make_metric(active=70e9, total=70e9)
        driver.add(anchor)
        cell = driver.build_cell(anchor)
        expected = {s.value for s in ParetoScenario}
        self.assertEqual(set(cell.scenario_frontiers.keys()), expected)

    def test_unconstrained_diagnostic_preserves_all(self):
        driver = BudgetMatrix()
        for i in range(5):
            driver.add(make_metric(loss=1.9 + 0.01 * i))
        cell = driver.build_cell(make_metric(loss=1.9))
        diag = cell.scenario_frontiers[
            ParetoScenario.UNCONSTRAINED_DIAGNOSTIC.value]
        self.assertEqual(len(diag.frontier), 5)


class MatrixAggregationTests(unittest.TestCase):
    """Matrix-level driver: aggregates cells and renders the (hw × ctx) view."""

    def test_matrix_json_lists_cells_and_axes(self):
        matrix = Matrix(training_tokens=int(20e12))
        for hw in ["h100"]:
            for ctx in [32768, 131072]:
                for ref in [7.0, 70.0]:
                    driver = BudgetMatrix(hardware=hw, context_length=ctx,
                                          training_tokens=int(20e12))
                    anchor = make_metric(active=int(ref * 1e9),
                                         total=int(ref * 1e9))
                    driver.add(anchor)
                    cell = driver.build_cell(anchor)
                    matrix.add_cell(MatrixKey(hw, ctx, ref), cell)
        js = render_matrix_json(matrix)
        self.assertEqual(js["training_tokens"], int(20e12))
        self.assertIn("h100", js["hardware"])
        self.assertEqual(js["contexts"], [32768, 131072])
        self.assertEqual(js["reference_active_bs"], [7.0, 70.0])
        self.assertEqual(len(js["cells"]), 4)

    def test_matrix_markdown_header_and_status_present(self):
        matrix = Matrix(training_tokens=int(20e12))
        for ctx in [32768, 131072]:
            driver = BudgetMatrix(hardware="h100", context_length=ctx,
                                  training_tokens=int(20e12))
            driver.add(make_metric(active=70e9, total=70e9, loss=1.9))
            cell = driver.build_cell(make_metric(active=70e9, total=70e9))
            matrix.add_cell(MatrixKey("h100", ctx, 70.0), cell)
        md = render_matrix_markdown(matrix)
        self.assertIn("Wave 18b matrix", md)
        self.assertIn("h100", md)
        # Fallback status label carries the loss-argmin tag so it's not
        # confused with 18d's confidence-aware decision.
        self.assertIn("[loss-argmin]", md)

    def test_status_fn_override(self):
        matrix = Matrix(training_tokens=int(20e12))
        driver = BudgetMatrix(hardware="h100", context_length=32768)
        driver.add(make_metric(active=70e9, total=70e9))
        cell = driver.build_cell(make_metric(active=70e9, total=70e9))
        matrix.add_cell(MatrixKey("h100", 32768, 70.0), cell)
        md = render_matrix_markdown(matrix,
                                    status_fn=lambda c: "CUSTOM_STATUS")
        self.assertIn("CUSTOM_STATUS", md)


if __name__ == "__main__":
    unittest.main()
