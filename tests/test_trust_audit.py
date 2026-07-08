"""Wave 18e — trust audit core module tests (sensitivity, invariants, frontier).

Covers the non-public-anchor portion of the audit surface. Public-anchor tests
live in test_public_model_anchors.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac"))

from ac.trust_audit import (  # noqa: E402
    CellSpec,
    InvariantResult,
    check_active_total_consistency,
    check_context_capacity_independent_of_fresh_prompt,
    check_family_coverage_before_pruning,
    check_tp_cost_nonfree_above_island,
    run_frontier_feasibility_suite,
    run_sensitivity_suite,
    render_public_anchor_markdown,
)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_active_total_consistency_pass():
    r = check_active_total_consistency(37.0, 671.0, 37.0, 671.0)
    assert r.passed
    assert r.name == "active_total_consistency"


def test_active_total_consistency_fail_when_reported_diverges():
    r = check_active_total_consistency(37.0, 671.0, 37.0, 500.0)
    assert not r.passed


def test_tp_cost_nonfree_above_island_catches_free_lunch():
    """Regression against the Wave 17 PP-at-decode / TP-cost bug that made
    TP=32 look faster than TP=8."""
    # TBT dropping from 25ms (island) to 5ms (above-island) is the bug shape.
    r = check_tp_cost_nonfree_above_island(25.0, 5.0)
    assert not r.passed, "cross-island TP AR must not reduce TBT below intra-island"
    # Same or higher above-island is expected physical behaviour.
    r2 = check_tp_cost_nonfree_above_island(25.0, 30.0)
    assert r2.passed


def test_family_coverage_before_pruning_fires_when_family_lost():
    """Wave 8b cheap-rank bug regression: MoE candidates were 100% pruned."""
    families_enabled = ["dense", "moe", "hybrid"]
    r = check_family_coverage_before_pruning(
        per_family_counts_before={"dense": 100, "moe": 200, "hybrid": 50},
        per_family_counts_after={"dense": 40, "moe": 0, "hybrid": 20},
        families_enabled=families_enabled,
    )
    assert not r.passed
    assert "moe" in r.detail


def test_family_coverage_all_present():
    families_enabled = ["dense", "moe"]
    r = check_family_coverage_before_pruning(
        per_family_counts_before={"dense": 100, "moe": 200},
        per_family_counts_after={"dense": 40, "moe": 40},
        families_enabled=families_enabled,
    )
    assert r.passed


def test_context_capacity_independent_of_fresh_prompt():
    r = check_context_capacity_independent_of_fresh_prompt(True, True)
    assert r.passed
    r = check_context_capacity_independent_of_fresh_prompt(True, False)
    assert not r.passed


# ---------------------------------------------------------------------------
# Sensitivity suite
# ---------------------------------------------------------------------------


def test_sensitivity_suite_runs_and_reports_stability():
    """Smoke test: sensitivity suite completes without error on a small cell
    and populates the expected fields."""
    cell = CellSpec(
        hardware="h100",
        target_params_b=1.0,
        context_length=8192,
        training_tokens=int(2e12),
        allow_moe=False,
        allow_state=False,
        serving_batch=4,
        tp=1,
    )
    report = run_sensitivity_suite(cell, base_cap=10)
    assert report["schema_version"] == "wave18e.sensitivity.v1"
    assert len(report["runs"]) == 3  # 1x, 2x, 4x
    assert 0.0 <= report["winner_stability_fraction"] <= 1.0
    assert 0.0 <= report["contender_retention_fraction"] <= 1.0
    assert isinstance(report["unique_winner_allowed"], bool)


# ---------------------------------------------------------------------------
# Frontier feasibility suite
# ---------------------------------------------------------------------------


def test_frontier_feasibility_suite_smoke():
    """Frontier feasibility suite runs and returns a well-formed report."""
    r = run_frontier_feasibility_suite("h100")
    assert r["schema_version"] == "wave18e.frontier_feasibility.v1"
    assert r["n_total"] == 4
    assert 0 <= r["n_pass"] <= r["n_total"]
    for a in r["anchors"]:
        assert "context_length" in a
        assert "physically_feasible" in a


# ---------------------------------------------------------------------------
# Report renderers
# ---------------------------------------------------------------------------


def test_render_public_anchor_markdown_shape():
    """Markdown renderer produces the expected header and blocking marker."""
    fake_report = {
        "schema_version": "wave18e.audit.v1",
        "tolerances_kind": "pre_calibration",
        "tolerances_in_use": {
            "loss": 0.10, "tbt_ms": 0.25, "ttft_ms": 0.30, "mem_gb": 0.15,
        },
        "counts": {"pass": 3, "fail": 1, "skipped": 2, "error": 0,
                   "unrepresentable": 0, "blocking": 1},
        "block_publication": True,
        "anchors": [
            {
                "id": "x", "display_name": "X", "status": "pass",
                "override_recorded": False, "override_justification": "",
                "error": "", "failed_metrics": [],
                "metrics": [{
                    "metric": "loss", "predicted": 1.8, "published": 1.79,
                    "rel_err": 0.006, "tolerance": 0.10, "passed": True,
                }],
                "breakdown": [],
            },
        ],
    }
    md = render_public_anchor_markdown(fake_report)
    assert "Public-model predictive accuracy audit" in md
    assert "block_publication:** YES" in md
    assert "pre_calibration" in md
