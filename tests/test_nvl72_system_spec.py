"""Gate-2 Task C: rack-scale system-spec loading (GB200 NVL72 / H800).

(Pins added in gate2-wave1; no existing assertions were modified.)

  * The optional `system` block in a hardware spec JSON is parsed into the
    legacy fabric fields (nvlink_domain_size, interconnect bandwidths,
    gpus_per_rack) and is authoritative when both are present.
  * Specs WITHOUT a `system` block keep single-node semantics exactly —
    the loader's new fields default to None and every legacy field passes
    through unchanged (zero-regression contract for old targets).
  * gb200_nvl72 exposes the 72-GPU NVLink5 domain at 1800 GB/s intra /
    50 GB/s inter with B200 chip params; h800 is h100_sxm with NVLink
    400 GB/s.
"""

import json
import os
import sys
import tempfile
import unittest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)

from ac.throughput_model import (  # noqa: E402
    HardwareConfig,
    _nvlink_domain_size,
    load_hardware,
)


class TestNvl72SystemSpec(unittest.TestCase):
    def test_gb200_nvl72_system_block_parsed(self):
        hw = load_hardware("gb200_nvl72")
        self.assertEqual(_nvlink_domain_size(hw), 72)
        self.assertEqual(hw.nvlink_domain_size, 72)
        self.assertEqual(hw.gpus_per_rack, 72)
        self.assertEqual(hw.gpus_per_node, 72)
        self.assertEqual(hw.chip, "b200")
        # System-block bandwidths land on the legacy interconnect fields.
        self.assertEqual(hw.interconnect["intra_node_bw_gb_s"], 1800)
        self.assertEqual(hw.interconnect["inter_node_bw_gb_s"], 50)
        # Chip-level params inherited from b200.json.
        self.assertEqual(hw.hbm_capacity_gb, 192)
        self.assertEqual(hw.hbm_bandwidth_tb_s, 8.0)
        self.assertEqual(hw.peak_flops_tf["bf16"], 1125)
        self.assertEqual(hw.accelerator_family, "blackwell")

    def test_h800_is_h100_with_reduced_nvlink(self):
        h800 = load_hardware("h800")
        h100 = load_hardware("h100")
        self.assertEqual(h800.interconnect["intra_node_bw_gb_s"], 400)
        self.assertEqual(h100.interconnect["intra_node_bw_gb_s"], 900)
        # Everything else about the chip is identical silicon.
        for field in ("compute_units", "hbm_capacity_gb", "hbm_bandwidth_tb_s",
                      "peak_flops_tf", "datasheet_peak_flops_tf",
                      "supported_precisions", "gpus_per_node",
                      "nvlink_domain_size", "accelerator_family"):
            self.assertEqual(getattr(h800, field), getattr(h100, field),
                             f"h800 differs from h100 in {field}")
        self.assertEqual(h800.interconnect["inter_node_bw_gb_s"],
                         h100.interconnect["inter_node_bw_gb_s"])
        self.assertEqual(_nvlink_domain_size(h800), 8)

    def test_system_block_is_authoritative_over_legacy_fields(self):
        """A spec carrying BOTH legacy fields and a conflicting system
        block must resolve to the system block's values."""
        spec = {
            "vendor": "nvidia",
            "accelerator_family": "blackwell",
            "accelerator_name": "toy",
            "compute_units": 1,
            "compute_unit_type": "SM",
            "hbm_capacity_gb": 16,
            "hbm_bandwidth_tb_s": 1.0,
            "peak_flops_tf": {"bf16": 1},
            "bytes_per_element": {"bf16": 2},
            "supported_precisions": ["bf16"],
            "fused_attention_efficiency": {"bf16": 0.8},
            "fused_attention_kernel": "toy",
            "interconnect": {
                "type": "nvlink",
                "intra_node_bw_gb_s": 111,
                "inter_node_bw_gb_s": 11,
                "topology": "toy",
            },
            "nvlink_domain_size": 8,
            "gpus_per_node": 8,
            "system": {
                "nvlink_domain_size": 32,
                "intra_domain_bandwidth_gbps": 999,
                "inter_domain_bandwidth_gbps": 33,
                "gpus_per_rack": 32,
            },
        }
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
            path = f.name
        try:
            hw = HardwareConfig.from_json(path)
        finally:
            os.unlink(path)
        self.assertEqual(hw.nvlink_domain_size, 32)
        self.assertEqual(hw.gpus_per_rack, 32)
        self.assertEqual(hw.interconnect["intra_node_bw_gb_s"], 999)
        self.assertEqual(hw.interconnect["inter_node_bw_gb_s"], 33)

    def test_old_specs_fallback_to_single_node_semantics(self):
        """Specs without a `system` block: new optional fields default to
        None and every legacy field is untouched (zero-regression)."""
        expectations = {
            # name: (domain, intra, inter, gpus_per_node)
            "h100": (8, 900, 50, 8),
            "b200": (8, 1800, 100, 8),
            "tpu_v5p": (16, 800, 800, 4),
            "tpu_v5e": (8, 400, 400, 4),
            "trainium2": (16, 1280, 400, 16),
            "trainium3": (32, 2400, 800, 32),
        }
        for name, (domain, intra, inter, gpn) in expectations.items():
            hw = load_hardware(name)
            self.assertEqual(
                _nvlink_domain_size(hw), domain, f"{name} domain drifted")
            self.assertEqual(hw.interconnect["intra_node_bw_gb_s"], intra,
                             f"{name} intra bw drifted")
            self.assertEqual(hw.interconnect["inter_node_bw_gb_s"], inter,
                             f"{name} inter bw drifted")
            per_node = hw.gpus_per_node if hw.vendor == "nvidia" \
                else hw.chips_per_host
            self.assertEqual(per_node, gpn, f"{name} node size drifted")
            # New optional fields must default to absent — this is what
            # keeps old-target behavior byte-identical.
            self.assertIsNone(hw.gpus_per_rack, f"{name} gained a rack")
            self.assertIsNone(hw.chip, f"{name} gained a chip ref")

    def test_new_targets_in_default_ep_search_space(self):
        from ac.lattice_engine import default_ep_options
        nvl_opts = default_ep_options("gb200_nvl72", for_moe=True)
        for want in (8, 16, 32, 72):
            self.assertIn(want, nvl_opts)
        h800_opts = default_ep_options("h800", for_moe=True)
        self.assertEqual(h800_opts, [2, 4, 8])
        # Old targets' default search spaces are unchanged.
        self.assertEqual(default_ep_options("h100", for_moe=True), [2, 4, 8])
        self.assertEqual(
            default_ep_options("b200", for_moe=True),
            [2, 4, 8, 16, 32, 64, 72])


if __name__ == "__main__":
    unittest.main()
