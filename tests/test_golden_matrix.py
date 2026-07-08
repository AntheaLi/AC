"""Wave 15 — Pillar B: golden decision-matrix regression test.

Re-runs the 32-cell golden matrix and asserts against the committed fixture
at `tests/fixtures/golden_h100_decision_matrix.json`.

The contract:
  * Identical family picks per cell.
  * Loss within ±2%.
  * TBT within ±5%.
  * Shape (d_model, n_layers) identical.

Any drift fails the test. Intentional drift requires re-running
`scripts/regen_golden_matrix.py --accept-drift "reason"` and re-committing
the fixture. The reason is recorded in `.regen-history.txt` so the
provenance of every cell-moving update is auditable.

This catches: any change that silently moves cells. The Wave 8b prune bug
would have been caught here — the golden matrix would have shown every
cell as dense before stratification; the stratified fix visibly flips
cells, which is exactly the kind of change a reviewer should see before
merging.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "golden_h100_decision_matrix.json"

# Tolerances. Loss is the tightest because the chinchilla spine should be
# stable across runs; TBT is looser because the picker can break ties on
# different throughput axes when the loss bucket is tied.
LOSS_TOLERANCE_PCT = 2.0
TBT_TOLERANCE_PCT = 5.0
MEM_TOLERANCE_PCT = 10.0


def _family_of(arch) -> str:
    has_moe = bool(getattr(arch, "moe", None))
    has_state = bool(getattr(arch, "state_config", None)) and \
                getattr(arch, "n_state_layers", 0) > 0
    if has_moe and has_state: return "moe_hybrid"
    if has_moe:               return "moe"
    if has_state:             return "hybrid"
    return "dense"


def _recompute_cell(rec):
    """Reproduce one cell from the fixture's stored grid spec."""
    from ac.optimizer import DeploymentConstraints, optimize
    size_b = float(rec["size_b"])
    ctx = int(rec["ctx"])
    serving = rec["serving"]
    batch = 8 if serving == "short_serving" else 4
    c = DeploymentConstraints(
        target_params_b=size_b,
        training_tokens=int(2e12),
        context_length=ctx,
        serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=batch,
        tp=1, pp=1, dp=1,
        allow_moe=True,
        allow_state=True if size_b <= 100.0 else False,
        max_candidates=50,
        max_full_evaluations=20,
        allow_quality_sentinel=True,
        param_tolerance=0.15,
    )
    return optimize("h100", c)


class GoldenMatrixRegressionTests(unittest.TestCase):
    """Locked: any drift in the 32-cell decision matrix fails this test.

    The test is SLOW (~30s) because it re-runs the full grid. It exists
    as the matrix-level regression net — for fast iteration use
    `tests/test_matrix_invariants.py` (Pillar A), which tests properties
    not snapshots."""

    @classmethod
    def setUpClass(cls):
        if not FIXTURE_PATH.exists():
            raise unittest.SkipTest(
                f"Golden fixture missing at {FIXTURE_PATH}. Run "
                "`python scripts/regen_golden_matrix.py` to create it."
            )
        with open(FIXTURE_PATH) as f:
            cls.fixture = json.load(f)

    def test_fixture_schema(self):
        """Sanity: fixture must have the expected shape."""
        self.assertEqual(self.fixture.get("schema_version"), "wave15.golden.v1")
        self.assertEqual(self.fixture.get("hardware"), "h100")
        cells = self.fixture.get("cells", [])
        self.assertEqual(
            len(cells), 32,
            f"Golden matrix should have 32 cells, has {len(cells)}. "
            f"Re-run `scripts/regen_golden_matrix.py` if the spec changed."
        )

    def test_decision_matrix_unchanged(self):
        """Re-run each cell and assert family, shape, and metrics match
        the committed fixture within tolerance.

        SLOW (~30s): re-runs the full 32-cell grid. Gated behind
        `AC_RUN_GOLDEN_MATRIX=1` so the default fast suite skips it.
        Run nightly and before any release per
        `plan/redesign/scrutinization-protocol.md`."""
        import os as _os
        if _os.environ.get("AC_RUN_GOLDEN_MATRIX") != "1":
            self.skipTest(
                "Set AC_RUN_GOLDEN_MATRIX=1 to run the golden-matrix "
                "regression. Default-off per the nightly/release-gate "
                "protocol."
            )
        drift_records = []
        for rec in self.fixture["cells"]:
            cell_label = (
                f"{int(rec['size_b'])}B ctx={rec['ctx']} {rec['serving']}"
            )
            try:
                r = _recompute_cell(rec)
            except Exception as exc:
                drift_records.append(
                    f"{cell_label}: re-eval raised {type(exc).__name__}: {exc}"
                )
                continue

            committed_family = rec.get("family")
            if r.optimal is None:
                if committed_family is not None:
                    drift_records.append(
                        f"{cell_label}: optimizer returned None (committed "
                        f"had family={committed_family})"
                    )
                continue

            new_family = _family_of(r.optimal.arch)
            if new_family != committed_family:
                drift_records.append(
                    f"{cell_label}: family {committed_family} → {new_family}"
                )

            committed_picked = rec.get("picked") or {}
            new_d = int(r.optimal.arch.d_model)
            new_L = int(r.optimal.arch.n_layers)
            if (committed_picked.get("d_model") != new_d
                    or committed_picked.get("n_layers") != new_L):
                drift_records.append(
                    f"{cell_label}: shape "
                    f"{committed_picked.get('d_model')}x"
                    f"{committed_picked.get('n_layers')} → {new_d}x{new_L}"
                )

            old_loss = float(committed_picked.get("loss", 0.0))
            new_loss = float(r.optimal.predicted_loss)
            if old_loss > 0:
                drift = abs(new_loss - old_loss) / old_loss * 100
                if drift > LOSS_TOLERANCE_PCT:
                    drift_records.append(
                        f"{cell_label}: loss {old_loss:.4f} → "
                        f"{new_loss:.4f} ({drift:.2f}% > "
                        f"{LOSS_TOLERANCE_PCT}% tolerance)"
                    )

            old_tbt = float(committed_picked.get("tbt_ms", 0.0))
            new_tbt = float(r.optimal.serving_tbt_ms)
            if old_tbt > 0:
                drift = abs(new_tbt - old_tbt) / old_tbt * 100
                if drift > TBT_TOLERANCE_PCT:
                    drift_records.append(
                        f"{cell_label}: TBT {old_tbt:.2f}ms → "
                        f"{new_tbt:.2f}ms ({drift:.2f}% > "
                        f"{TBT_TOLERANCE_PCT}% tolerance)"
                    )

        self.assertEqual(
            drift_records, [],
            "Golden matrix drift detected:\n  " + "\n  ".join(drift_records)
            + "\n\nIf this drift is intentional (e.g., constant edit, "
            "calibration update), re-run `python scripts/regen_golden_matrix.py"
            " --accept-drift \"reason\"` and commit the updated fixture."
        )


if __name__ == "__main__":
    unittest.main()
