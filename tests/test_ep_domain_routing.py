"""Gate-2 Task C: all-to-all intra/inter-domain routing numerics.

(Pins added in gate2-wave1; no existing assertions were modified.)

`_moe_alltoall_cost` must:

  * route ALL volume over the intra-domain fabric when ep <= domain size
    (0.67 ring efficiency on `intra_node_bw_gb_s`);
  * split traffic when ep > domain: the intra share rides NVLink, the
    inter share rides `inter_node_bw_gb_s` with the node-limited-routing
    dedup, and the two concurrent legs resolve as max(t_intra, t_inter);
  * make EP=72 cheap on a 72-GPU domain (GB200 NVL72) and expensive on an
    8-GPU domain (H100/H800) — the core rack-scale economics of Task C;
  * move monotonically as the domain size changes (sensitivity contract
    behind validation experiment V3).

The stress vector's all_to_all axis reads its link bandwidth from
`stress._link_bw_bytes_s`; we pin that it routes by the same domain rule.
"""

import os
import sys
import unittest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)

from ac.throughput_model import (  # noqa: E402
    HardwareConfig,
    _moe_alltoall_cost,
    load_hardware,
)
from ac.stress import _link_bw_bytes_s  # noqa: E402

V = 1.0e9  # 1 GB of per-layer all-to-all volume (dispatch + combine)


def _toy_hw(intra_gbps, inter_gbps, domain, gpus_per_node=None):
    return HardwareConfig(
        vendor="nvidia",
        accelerator_family="hopper",
        accelerator_name="toy",
        compute_units=132,
        compute_unit_type="SM",
        hbm_capacity_gb=80,
        hbm_bandwidth_tb_s=3.35,
        peak_flops_tf={"bf16": 495},
        bytes_per_element={"bf16": 2},
        supported_precisions=["bf16"],
        fused_attention_efficiency={"bf16": 0.8},
        fused_attention_kernel="toy",
        interconnect={
            "type": "nvlink",
            "intra_node_bw_gb_s": intra_gbps,
            "inter_node_bw_gb_s": inter_gbps,
            "topology": "toy",
        },
        gpus_per_node=gpus_per_node or domain,
        nvlink_domain_size=domain,
    )


def _expected_intra(volume, ep, intra_gbps):
    return (ep - 1) / ep * volume / (0.67 * intra_gbps * 1e9)


def _expected_split(volume, ep, domain, intra_gbps, inter_gbps, dedup=0.5):
    off_rank = (ep - 1) / ep
    f_intra = (domain - 1) / (ep - 1)
    f_inter = 1.0 - f_intra
    t_intra = off_rank * volume * f_intra / (0.67 * intra_gbps * 1e9)
    t_inter = off_rank * volume * f_inter * dedup / (0.67 * inter_gbps * 1e9)
    return max(t_intra, t_inter)


class TestAllToAllDomainRouting(unittest.TestCase):
    def test_ep1_kernel_floor(self):
        hw = _toy_hw(900, 50, 8)
        self.assertAlmostEqual(_moe_alltoall_cost(V, 1, hw), 1e-6)

    def test_intra_domain_branch_exact(self):
        """ep <= domain: pure intra-domain pricing, hand-computed."""
        hw = _toy_hw(900, 50, 8)
        for ep in (2, 4, 8):
            got = _moe_alltoall_cost(V, ep, hw)
            want = _expected_intra(V, ep, 900)
            self.assertAlmostEqual(got / want, 1.0, places=9)

    def test_beyond_domain_split_exact(self):
        """ep > domain: hierarchical split, hand-computed max() of legs."""
        hw = _toy_hw(900, 50, 8)
        for ep in (16, 32, 72):
            got = _moe_alltoall_cost(V, ep, hw)
            want = _expected_split(V, ep, 8, 900, 50)
            self.assertAlmostEqual(got / want, 1.0, places=9,
                                   msg=f"ep={ep} split mismatch")

    def test_inter_leg_dominates_beyond_domain(self):
        """With an 18:1 intra:inter ratio, the inter leg is the binding
        one for ep >> domain — that's the whole point of the model."""
        hw = _toy_hw(900, 50, 8)
        ep = 72
        off_rank = (ep - 1) / ep
        f_inter = 1.0 - (8 - 1) / (ep - 1)
        t_inter = off_rank * V * f_inter * 0.5 / (0.67 * 50e9)
        got = _moe_alltoall_cost(V, ep, hw)
        self.assertAlmostEqual(got / t_inter, 1.0, places=9)

    def test_ep72_nvl72_vs_h100(self):
        """The Task-C headline contrast, at the model level: EP=72 on a
        72-GPU NVLink domain vs an 8-GPU node domain."""
        nvl72 = load_hardware("gb200_nvl72")
        h100 = load_hardware("h100")
        cost_nvl = _moe_alltoall_cost(V, 72, nvl72)
        cost_h100 = _moe_alltoall_cost(V, 72, h100)
        # NVL72: everything rides the 1800 GB/s domain fabric.
        self.assertAlmostEqual(
            cost_nvl / _expected_intra(V, 72, 1800), 1.0, places=9)
        # H100: inter-node IB leg dominates -> >10x more expensive.
        self.assertGreater(cost_h100 / cost_nvl, 10.0)

    def test_cost_monotonic_in_domain_size(self):
        """Fixed ep=32: cost must be non-increasing as the domain grows
        8 -> 16 -> 32 -> 72 (the V3 sensitivity contract)."""
        costs = []
        for domain in (8, 16, 32, 72):
            hw = _toy_hw(900, 50, domain)
            costs.append(_moe_alltoall_cost(V, 32, hw))
        for smaller, larger in zip(costs, costs[1:]):
            self.assertGreaterEqual(smaller, larger)
        # And the endpoints differ materially (the domain actually matters).
        self.assertGreater(costs[0] / costs[-1], 2.0)

    def test_h800_slower_than_h100_within_domain(self):
        """H800's 400 GB/s NVLink makes even in-domain all-to-all 2.25x
        slower than H100's 900 GB/s — the export-SKU penalty."""
        h800 = load_hardware("h800")
        h100 = load_hardware("h100")
        c800 = _moe_alltoall_cost(V, 8, h800)
        c100 = _moe_alltoall_cost(V, 8, h100)
        self.assertAlmostEqual(c800 / c100, 900.0 / 400.0, places=6)


class TestStressAxisDomainRouting(unittest.TestCase):
    """The stress vector's all_to_all axis must reflect the same domain
    rule: link BW = intra when ranks fit the domain, inter beyond it."""

    def test_nvl72_ep72_uses_intra(self):
        hw = load_hardware("gb200_nvl72")
        bw = _link_bw_bytes_s(hw, 72)
        self.assertEqual(bw, hw.interconnect["intra_node_bw_gb_s"] * 1e9)

    def test_nvl72_beyond_rack_uses_inter(self):
        hw = load_hardware("gb200_nvl72")
        bw = _link_bw_bytes_s(hw, 144)
        self.assertEqual(bw, hw.interconnect["inter_node_bw_gb_s"] * 1e9)

    def test_h100_ep72_uses_inter(self):
        hw = load_hardware("h100")
        bw = _link_bw_bytes_s(hw, 72)
        self.assertEqual(bw, hw.interconnect["inter_node_bw_gb_s"] * 1e9)

    def test_h800_ep8_uses_reduced_intra(self):
        hw = load_hardware("h800")
        bw = _link_bw_bytes_s(hw, 8)
        self.assertEqual(bw, 400e9)

    def test_old_targets_unchanged(self):
        for name, ranks, want_gbps in (("h100", 8, 900), ("h100", 16, 50),
                                       ("b200", 8, 1800), ("b200", 16, 100)):
            hw = load_hardware(name)
            self.assertEqual(_link_bw_bytes_s(hw, ranks), want_gbps * 1e9,
                             f"{name} ranks={ranks} link bw drifted")


if __name__ == "__main__":
    unittest.main()
