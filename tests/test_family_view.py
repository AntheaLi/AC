"""Wave 2b Step 2b.3: snapshot tests for the per-arch-family comparison renderer.

Locks the rendering format so cosmetic changes don't silently break the
v1-web app's expectations or shift the CLI output that researchers might
script against.

Four cells exercise the interesting cases:
- 1B @ 8k: small model, dense dominates (no MoE feasible at this size)
- 13B @ 8k: the chat-example cell (loss-vs-serving knee illustrated)
- 120B @ 1M: triggers HBM-spill annotation
- 1000B @ 4M: 4-family large-scale comparison
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AC = os.path.normpath(os.path.join(_HERE, "..", "ac"))
if _AC not in sys.path:
    sys.path.insert(0, _AC)

from report import (  # noqa: E402  (sys.path mutation must happen first)
    _fmt_ctx,
    _fmt_tbt_delta,
    _pretty_arch,
    render_family_comparison,
)


# ---------------------------------------------------------------------------
# Unit tests on the small helpers
# ---------------------------------------------------------------------------

def test_pretty_arch_known_modes():
    assert _pretty_arch("dense") == "dense"
    assert _pretty_arch("hybrid") == "hybrid"
    assert _pretty_arch("moe") == "MoE"
    assert _pretty_arch("moe_hybrid") == "MoE-hybrid"
    assert _pretty_arch("moe_hybrid", "mamba2") == "MoE-hybrid"


def test_pretty_arch_unknown_falls_through():
    # Unknown modes are passed through verbatim so the renderer never crashes
    # on a future arch_mode string the optimizer might add.
    assert _pretty_arch("retnet") == "retnet"


def test_fmt_ctx_buckets():
    assert _fmt_ctx(8192) == "8k"
    assert _fmt_ctx(32768) == "32k"
    assert _fmt_ctx(131072) == "128k"
    assert _fmt_ctx(1048576) == "1M"
    assert _fmt_ctx(4194304) == "4M"
    assert _fmt_ctx(512) == "512"  # below the k threshold


def test_fmt_tbt_delta_small_moves_render_as_percent():
    assert _fmt_tbt_delta(0.4) == "≈same decode"
    assert _fmt_tbt_delta(15.0) == "15% slower decode"
    assert _fmt_tbt_delta(-30.0) == "30% faster decode"


def test_fmt_tbt_delta_large_moves_render_as_multiplier():
    # Slower side switches to × at >= 100%
    assert _fmt_tbt_delta(150.0) == "2.5× slower decode"
    assert _fmt_tbt_delta(1000.0) == "11.0× slower decode"
    # Faster side switches to × at <= -50%
    assert _fmt_tbt_delta(-50.0) == "2.0× faster decode"
    assert _fmt_tbt_delta(-92.3) == "13.0× faster decode"


# ---------------------------------------------------------------------------
# Renderer snapshot tests — golden strings inline so reviewers see the format
# ---------------------------------------------------------------------------

# Chat-example cell: 13B @ 8k h100. The four families with the loss-vs-serving
# knee that motivated the redesign — MoE-hybrid wins on loss by a hair but
# pure dense is 13× faster to serve at +6% loss.
_FAMILIES_13B_8K = [
    {"arch_mode": "moe_hybrid", "state_type": "mamba2",
     "loss": 1.8192, "loss_delta_pct": 0.0, "tbt_ms": 142.0, "tbt_delta_pct": 0.0,
     "ttft_ms": 309.0, "mem_gb": 39.2, "spill_tier": "fits"},
    {"arch_mode": "moe", "state_type": None,
     "loss": 1.8465, "loss_delta_pct": 1.5, "tbt_ms": 194.0, "tbt_delta_pct": 36.6,
     "ttft_ms": 132.0, "mem_gb": 34.0, "spill_tier": "fits"},
    {"arch_mode": "dense", "state_type": None,
     "loss": 1.9304, "loss_delta_pct": 6.1, "tbt_ms": 11.0, "tbt_delta_pct": -92.3,
     "ttft_ms": 164.0, "mem_gb": 42.0, "spill_tier": "fits"},
    {"arch_mode": "hybrid", "state_type": "mamba2",
     "loss": 1.9639, "loss_delta_pct": 7.9, "tbt_ms": 12.0, "tbt_delta_pct": -91.5,
     "ttft_ms": 578.0, "mem_gb": 55.0, "spill_tier": "fits"},
]


def test_render_chat_example_cell():
    out = render_family_comparison(_FAMILIES_13B_8K, 13.0, 8192)
    # Wave 20 (feedback #1): the cross-family dense row carries the †
    # serving-bias marker (its |TBT delta| clears the moe-vs-dense floor
    # but remains a pre-calibration estimate); the hybrid row has no
    # anchor-measured floor (no hybrid serving anchors yet) so it is
    # unmarked; the footnote explains the marker.
    expected = (
        "\n13B @ 8k      loss        TBT         TTFT       mem\n"
        "  MoE-hybrid   1.8192     142 ms      309 ms     39 GB\n"
        "  MoE          1.8465     194 ms      132 ms     34 GB   (+1.5% loss, 37% slower decode)\n"
        "  dense        1.9304    11.0 ms      164 ms     42 GB   (+6.1% loss, 13.0× faster decode†)\n"
        "  hybrid       1.9639    12.0 ms      578 ms     55 GB   (+7.9% loss, 11.8× faster decode)\n"
        "  † cross-family decode/TTFT deltas are pre-calibration estimates "
        "under anchor-measured serving bias (see family_bias_v1.json for "
        "current per-family numbers); treat magnitudes, not just sub-floor "
        "deltas, with "
        "caution until a serving pack is fitted.\n"
    )
    assert out == expected


# 1B @ 8k: small model — at this size the optimizer-rollup typically has
# dense as the sole feasible family (MoE active params dominate, hybrid
# state residual hurts more than KV cache).
_FAMILIES_1B_8K = [
    {"arch_mode": "dense", "state_type": None,
     "loss": 2.1343, "loss_delta_pct": 0.0, "tbt_ms": 1.9, "tbt_delta_pct": 0.0,
     "ttft_ms": 17.8, "mem_gb": 7.1, "spill_tier": "fits"},
]


def test_render_single_family_cell():
    out = render_family_comparison(_FAMILIES_1B_8K, 1.0, 8192)
    expected = (
        "\n1B @ 8k      loss        TBT         TTFT       mem\n"
        "  dense        2.1343    1.90 ms     17.8 ms      7 GB\n"
    )
    assert out == expected


# 120B @ 1M: large model + long ctx → memory pressure → some families spill.
_FAMILIES_120B_1M = [
    {"arch_mode": "moe", "state_type": None,
     "loss": 1.7760, "loss_delta_pct": 0.0, "tbt_ms": 2429.0, "tbt_delta_pct": 0.0,
     "ttft_ms": 36611.0, "mem_gb": 95.0, "spill_tier": "nvlink"},
    {"arch_mode": "moe_hybrid", "state_type": "mamba2",
     "loss": 1.7850, "loss_delta_pct": 0.5, "tbt_ms": 1656.0, "tbt_delta_pct": -31.8,
     "ttft_ms": 69475.0, "mem_gb": 62.6, "spill_tier": "fits"},
    {"arch_mode": "dense", "state_type": None,
     "loss": 1.8926, "loss_delta_pct": 6.6, "tbt_ms": 440.0, "tbt_delta_pct": -81.9,
     "ttft_ms": 27622.0, "mem_gb": 34.0, "spill_tier": "fits"},
]


def test_render_spill_annotation():
    out = render_family_comparison(_FAMILIES_120B_1M, 120.0, 1048576)
    # The "nvlink spill" tag must appear on the spilled row.
    assert "[nvlink spill]" in out
    # Non-spilled rows must NOT carry the tag.
    assert out.count("[nvlink spill]") == 1
    # Family ordering is loss-sorted.
    lines = [line for line in out.split("\n") if line.startswith("  ")]
    assert lines[0].startswith("  MoE ")
    assert lines[1].startswith("  MoE-hybrid")
    assert lines[2].startswith("  dense")


# 1000B @ 4M: all four families present at extreme scale, exercises wide
# loss column and large mem values.
_FAMILIES_1000B_4M = [
    {"arch_mode": "moe_hybrid", "state_type": "mamba2",
     "loss": 1.7862, "loss_delta_pct": 0.0, "tbt_ms": 453.1, "tbt_delta_pct": 0.0,
     "ttft_ms": 51408.7, "mem_gb": 52.7, "spill_tier": "fits"},
    {"arch_mode": "moe", "state_type": None,
     "loss": 1.7988, "loss_delta_pct": 0.7, "tbt_ms": 694.7, "tbt_delta_pct": 53.3,
     "ttft_ms": 30294.0, "mem_gb": 44.4, "spill_tier": "fits"},
    {"arch_mode": "hybrid", "state_type": "mamba2",
     "loss": 1.8741, "loss_delta_pct": 4.9, "tbt_ms": 432.0, "tbt_delta_pct": -4.7,
     "ttft_ms": 77819.4, "mem_gb": 67.2, "spill_tier": "fits"},
    {"arch_mode": "dense", "state_type": None,
     "loss": 1.8747, "loss_delta_pct": 5.0, "tbt_ms": 1979.9, "tbt_delta_pct": 337.0,
     "ttft_ms": 102344.5, "mem_gb": 129.9, "spill_tier": "fits"},
]


def test_render_four_family_extreme_scale():
    out = render_family_comparison(_FAMILIES_1000B_4M, 1000.0, 4194304)
    # Wave 20 (feedback #1): dense-vs-MoE-hybrid is cross-family, so the
    # dense row carries the † serving-bias marker + footnote.
    expected = (
        "\n1000B @ 4M      loss        TBT         TTFT       mem\n"
        "  MoE-hybrid   1.7862     453 ms    51409 ms     53 GB\n"
        "  MoE          1.7988     695 ms    30294 ms     44 GB   (+0.7% loss, 53% slower decode)\n"
        "  hybrid       1.8741     432 ms    77819 ms     67 GB   (+4.9% loss, 5% faster decode)\n"
        "  dense        1.8747    1980 ms   102344 ms    130 GB   (+5.0% loss, 4.4× slower decode†)\n"
        "  † cross-family decode/TTFT deltas are pre-calibration estimates "
        "under anchor-measured serving bias (see family_bias_v1.json for "
        "current per-family numbers); treat magnitudes, not just sub-floor "
        "deltas, with "
        "caution until a serving pack is fitted.\n"
    )
    assert out == expected


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_render_empty_returns_empty_string():
    assert render_family_comparison([], 13.0, 8192) == ""


def test_render_missing_fields_does_not_crash():
    # The renderer must tolerate a minimal family dict (e.g. produced by a
    # back-compat path on legacy data).
    minimal = [{"arch_mode": "dense", "loss": 2.0, "tbt_ms": 10.0}]
    out = render_family_comparison(minimal, 1.0, 8192)
    assert "dense" in out
    assert "2.0000" in out
