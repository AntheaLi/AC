"""Wave 18c tests — physical status separation and operational flags.

Covers the remaining spec §Tests bullets:

  * A 60-second TTFT candidate remains physically feasible and is flagged.
  * Unsupported positional/context encoding is physically unsupported.
  * Cross-node TP/EP/CP is flagged rather than silently treated as local.
  * HBM spill routes to physical when spill tier is disallowed, operational
    when permitted.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ac.budget_pareto import CandidateMetrics, Topology, attach_operational_flags
from ac.serving_workload import (
    MEMORY_TIER_HBM,
    MEMORY_TIER_HBM_SPILL,
    ServingWorkloadSpec,
    WorkloadScenario,
    canonical_workloads,
)
from ac.operational_flags import (
    DEFAULT_THRESHOLDS,
    ExtendedFeasibility,
    OperationalAssessment,
    OperationalFlag,
    PhysicalStatus,
    build_extended_feasibility,
    evaluate_operational_flags,
    hbm_spill_physical,
)


def make_metric(
    *, loss=1.9, tbt=20.0, ttft=200.0, mem=40.0, gpu_s_req=0.02,
    replica_gpus=8, tp=8, pp=1, ep=1, cp=1, batch=16,
) -> CandidateMetrics:
    return CandidateMetrics(
        identity_label="dense",
        active_params=7_000_000_000, total_params=7_000_000_000,
        predicted_loss=loss, tbt_ms=tbt, ttft_ms=ttft,
        memory_per_gpu_gb=mem,
        serving_gpu_seconds_per_request=gpu_s_req,
        replica_gpus=replica_gpus,
        topology=Topology(tp=tp, pp=pp, ep=ep, cp=cp, serving_batch=batch),
    )


@dataclass
class _MockHardware:
    nvlink_domain_size: int = 8
    gpus_per_node: int = 8
    vendor: str = "nvidia"


class InteractiveFlagsTests(unittest.TestCase):
    """Spec §Tests — 60s TTFT stays physically feasible and gets flagged."""

    def test_60s_ttft_is_feasible_and_flagged(self):
        m = make_metric(ttft=60_000.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertEqual(ext.physical_status, PhysicalStatus.FEASIBLE)
        self.assertTrue(ext.is_feasible)
        self.assertIn(OperationalFlag.EXTREME_TTFT.value, ext.operational.flags)

    def test_extreme_tbt_flag(self):
        m = make_metric(tbt=500.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertIn(OperationalFlag.EXTREME_TBT.value, ext.operational.flags)
        # Still physically feasible.
        self.assertTrue(ext.is_feasible)

    def test_low_batch_efficiency_flag(self):
        m = make_metric(batch=1)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertIn(OperationalFlag.LOW_BATCH_EFFICIENCY.value,
                      ext.operational.flags)

    def test_no_flags_on_healthy_candidate(self):
        m = make_metric(ttft=200.0, tbt=20.0, gpu_s_req=0.02, replica_gpus=8, batch=16)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertTrue(ext.is_feasible)
        # Only cross_node_collective_risk might not fire — everything else clean.
        self.assertNotIn(OperationalFlag.EXTREME_TBT.value, ext.operational.flags)
        self.assertNotIn(OperationalFlag.EXTREME_TTFT.value, ext.operational.flags)
        self.assertNotIn(OperationalFlag.LOW_BATCH_EFFICIENCY.value,
                         ext.operational.flags)


class ColdIngestionThresholdSwitchTests(unittest.TestCase):
    """Cold-ingestion scenarios raise TTFT + replica thresholds so a 60s TTFT
    doesn't flag at long context ingestion, but a 20-minute one does."""

    def test_cold_ingestion_ttft_threshold_is_600s(self):
        # A 60s TTFT is EXTREME under interactive but fine under cold.
        m = make_metric(ttft=60_000.0)
        w_cold = canonical_workloads(int(1e6))[
            WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value]
        w_int = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext_cold = build_extended_feasibility(m, w_cold, hardware=_MockHardware())
        ext_int = build_extended_feasibility(m, w_int, hardware=_MockHardware())
        self.assertNotIn(OperationalFlag.EXTREME_TTFT.value,
                         ext_cold.operational.flags)
        self.assertIn(OperationalFlag.EXTREME_TTFT.value, ext_int.operational.flags)

    def test_cold_ingestion_replica_threshold_is_4096(self):
        m = make_metric(replica_gpus=1024)
        w_cold = canonical_workloads(int(1e6))[
            WorkloadScenario.COLD_FULL_CONTEXT_INGESTION.value]
        w_int = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext_cold = build_extended_feasibility(m, w_cold, hardware=_MockHardware())
        ext_int = build_extended_feasibility(m, w_int, hardware=_MockHardware())
        self.assertNotIn(OperationalFlag.OVERSIZED_REPLICA.value,
                         ext_cold.operational.flags)
        self.assertIn(OperationalFlag.OVERSIZED_REPLICA.value,
                      ext_int.operational.flags)


class CrossNodeCollectiveRiskTests(unittest.TestCase):
    """Spec §Tests — cross-node TP/EP/CP is flagged rather than silently
    treated as local."""

    def test_tp_beyond_nvlink_domain_flagged(self):
        m = make_metric(tp=32)  # H100 NVLink domain = 8
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertIn(OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value,
                      ext.operational.flags)
        # Still physically feasible.
        self.assertTrue(ext.is_feasible)

    def test_ep_beyond_nvlink_domain_flagged(self):
        m = make_metric(tp=8, ep=16)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertIn(OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value,
                      ext.operational.flags)

    def test_cp_beyond_nvlink_domain_flagged(self):
        m = make_metric(tp=8, cp=16)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertIn(OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value,
                      ext.operational.flags)

    def test_within_nvlink_domain_not_flagged(self):
        m = make_metric(tp=8, ep=4, cp=1)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware())
        self.assertNotIn(OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value,
                         ext.operational.flags)

    def test_b200_larger_domain(self):
        m = make_metric(tp=32)  # Fits in B200 NVL72 domain
        b200 = _MockHardware(nvlink_domain_size=72, gpus_per_node=72)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=b200)
        self.assertNotIn(OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value,
                         ext.operational.flags,
                         "TP=32 must not be cross-node on B200 NVL72")


class UnsupportedIsPhysicalTests(unittest.TestCase):
    """Spec §Tests — unsupported positional/context encoding is physically
    unsupported. The 18c contract routes this through PhysicalStatus."""

    def test_unsupported_status_passes_through(self):
        m = make_metric()
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(
            m, w, hardware=_MockHardware(),
            physical_status=PhysicalStatus.UNSUPPORTED,
            physical_message="attention_pattern csa not yet implemented",
        )
        self.assertEqual(ext.physical_status, PhysicalStatus.UNSUPPORTED)
        self.assertFalse(ext.is_feasible)
        # Operational assessment is still populated for auditing.
        self.assertIsInstance(ext.operational, OperationalAssessment)

    def test_model_validation_failure_status(self):
        m = make_metric()
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(
            m, w, hardware=_MockHardware(),
            physical_status=PhysicalStatus.MODEL_VALIDATION_FAILURE,
            physical_message="1M anchor missing",
        )
        self.assertEqual(ext.physical_status,
                         PhysicalStatus.MODEL_VALIDATION_FAILURE)
        self.assertFalse(ext.is_feasible)


class HBMSpillRoutingTests(unittest.TestCase):
    """HBM spill: physical when tier disallowed, operational when permitted."""

    def test_spill_physical_when_disallowed(self):
        m = make_metric(mem=100.0)  # exceeds 80 GB
        w = canonical_workloads(
            131072,
            allowed_memory_tiers=(MEMORY_TIER_HBM,),
        )[WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware(), hbm_gb=80.0)
        self.assertEqual(ext.physical_status, PhysicalStatus.INFEASIBLE)
        self.assertIn(OperationalFlag.HBM_SPILL.value, ext.operational.flags)

    def test_spill_operational_when_permitted(self):
        m = make_metric(mem=100.0)
        w = canonical_workloads(
            131072,
            allowed_memory_tiers=(MEMORY_TIER_HBM, MEMORY_TIER_HBM_SPILL),
        )[WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware(), hbm_gb=80.0)
        self.assertEqual(ext.physical_status, PhysicalStatus.FEASIBLE)
        self.assertIn(OperationalFlag.HBM_SPILL.value, ext.operational.flags)

    def test_no_spill_no_flag(self):
        m = make_metric(mem=40.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        ext = build_extended_feasibility(m, w, hardware=_MockHardware(), hbm_gb=80.0)
        self.assertNotIn(OperationalFlag.HBM_SPILL.value, ext.operational.flags)
        self.assertEqual(ext.physical_status, PhysicalStatus.FEASIBLE)

    def test_hbm_spill_physical_helper_agrees(self):
        spills, physical = hbm_spill_physical(
            100.0, 80.0, (MEMORY_TIER_HBM,))
        self.assertTrue(spills)
        self.assertTrue(physical)
        spills, physical = hbm_spill_physical(
            100.0, 80.0, (MEMORY_TIER_HBM, MEMORY_TIER_HBM_SPILL))
        self.assertTrue(spills)
        self.assertFalse(physical)
        spills, physical = hbm_spill_physical(
            40.0, 80.0, (MEMORY_TIER_HBM,))
        self.assertFalse(spills)
        self.assertFalse(physical)


class ThresholdsAndBudgetsTests(unittest.TestCase):

    def test_custom_threshold_override(self):
        m = make_metric(tbt=50.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        # Default threshold is 200ms; lower it to 30ms so 50ms flags.
        ext = build_extended_feasibility(
            m, w, hardware=_MockHardware(),
            thresholds={OperationalFlag.EXTREME_TBT.value: 30.0},
        )
        self.assertIn(OperationalFlag.EXTREME_TBT.value, ext.operational.flags)

    def test_gpu_seconds_budget_override(self):
        m = make_metric(gpu_s_req=1.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        # Default budget is 10 gpu-s/req; lower to 0.5 so 1.0 flags.
        ext = build_extended_feasibility(
            m, w, hardware=_MockHardware(),
            gpu_seconds_budget=0.5,
        )
        self.assertIn(OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value,
                      ext.operational.flags)

    def test_default_thresholds_populated(self):
        # Every default flag has a measured value in the assessment.
        m = make_metric()
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        assessment = evaluate_operational_flags(m, w, hardware=_MockHardware())
        for name in (OperationalFlag.EXTREME_TBT.value,
                     OperationalFlag.EXTREME_TTFT.value,
                     OperationalFlag.LOW_BATCH_EFFICIENCY.value,
                     OperationalFlag.OVERSIZED_REPLICA.value,
                     OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value,
                     OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value):
            self.assertIn(name, assessment.measured,
                          f"{name} must have a measured value")


class BudgetParetoIntegrationTests(unittest.TestCase):
    """`attach_operational_flags` populates CandidateMetrics.extended_feasibility."""

    def test_attach_updates_metrics(self):
        m1 = make_metric(tbt=20.0)
        m2 = make_metric(tbt=500.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        attach_operational_flags([m1, m2], w, hardware=_MockHardware())
        self.assertIsNotNone(m1.extended_feasibility)
        self.assertIsNotNone(m2.extended_feasibility)
        self.assertNotIn(OperationalFlag.EXTREME_TBT.value,
                         m1.extended_feasibility.operational.flags)
        self.assertIn(OperationalFlag.EXTREME_TBT.value,
                      m2.extended_feasibility.operational.flags)

    def test_as_row_includes_extended_feasibility(self):
        m = make_metric(tbt=500.0)
        w = canonical_workloads(131072)[
            WorkloadScenario.INTERACTIVE_ORDINARY.value]
        attach_operational_flags([m], w, hardware=_MockHardware())
        row = m.as_row()
        self.assertIn("extended_feasibility", row)
        self.assertEqual(row["extended_feasibility"]["physical_status"],
                         PhysicalStatus.FEASIBLE.value)
        self.assertIn(OperationalFlag.EXTREME_TBT.value,
                      row["extended_feasibility"]["operational"]["flags"])


if __name__ == "__main__":
    unittest.main()
