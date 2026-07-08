"""Wave 18h: ledger-vs-metadata consistency gate for shipped reference configs.

A reference config whose declared metadata.params disagree with what its own
architecture block computes poisons every delta-eval / modifier / trust-audit
run against it. The gpt_oss_120b fixture shipped for weeks declaring ~120B
while its architecture block computed 655B; mai_thinking_1 declared 962B while
computing 2553B. This test blocks that class of regression: every JSON under
configs/ that declares metadata.params must agree with the computed ledger to
within 2% (total and active).

The same check runs at load time in ac.baseline.load_baseline_model (loud
stderr WARNING); this test is the release gate.
"""

import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ac.baseline import load_baseline_model  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
CONFIG_PATHS = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))

TOLERANCE = 0.02


@pytest.mark.parametrize("path", CONFIG_PATHS,
                         ids=[os.path.basename(p) for p in CONFIG_PATHS])
def test_reference_config_ledger_matches_metadata(path):
    with open(path) as f:
        config = json.load(f)
    declared = (config.get("metadata", {}) or {}).get("params", {}) or {}
    if not declared:
        pytest.skip(f"{os.path.basename(path)} declares no metadata.params")

    baseline = load_baseline_model(path)
    cand = baseline.candidate
    computed = {
        "total_b": cand.total_params / 1e9,
        "active_b": (cand.active_params or cand.total_params) / 1e9,
    }
    for key, computed_b in computed.items():
        decl = declared.get(key)
        if decl is None:
            continue
        decl_f = float(decl)
        assert decl_f > 0, f"{path}: metadata.params.{key} must be positive"
        rel_err = abs(computed_b - decl_f) / decl_f
        assert rel_err <= TOLERANCE, (
            f"{os.path.basename(path)}: metadata.params.{key}={decl_f:g}B but "
            f"the architecture block computes {computed_b:.2f}B "
            f"({rel_err * 100:.1f}% off, tolerance {TOLERANCE * 100:.0f}%). "
            "Fix the config metadata or the architecture block — a fabricated "
            "reference baseline corrupts every comparison made against it."
        )


@pytest.mark.parametrize("path", CONFIG_PATHS,
                         ids=[os.path.basename(p) for p in CONFIG_PATHS])
def test_reference_config_loads_without_ledger_warning(path):
    baseline = load_baseline_model(path)
    ledger_warnings = [w for w in baseline.warnings if "LEDGER MISMATCH" in w]
    assert not ledger_warnings, (
        f"{os.path.basename(path)} raises ledger warnings at load time: "
        f"{ledger_warnings}"
    )
