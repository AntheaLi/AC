#!/usr/bin/env python3
"""Gate-2 Wave-1 Task E demo — USD TCO layer on the two reference configs.

  ac_version:             0.4.0
  quality_model_version:  effective_capacity_v2
  git_commit:             c170cda
  experiment_date:        2026-07-17
  agent wave:             gate2-wave1

Runs ac-compile (modifier mode, same invocation family as the golden
snapshot) for configs/mistral_7b.json and configs/gpt_oss_120b.json, then
post-processes each emitted config.json with ac.pricing.attach_cost_block
and writes the augmented configs plus a cost summary next to the runs.

This script is the wiring the orchestrator reproduces inside the CLI with
--cost-usd (see integration_note.md); it changes NO existing files.

Usage:
    python validation/e_pricing/run_demo.py            # full: compile + price
    python validation/e_pricing/run_demo.py --price-only   # skip compile
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

sys.path.insert(0, REPO)
from ac.pricing import attach_cost_block  # noqa: E402

DEMOS = {
    "mistral_7b": {
        "baseline_config": "configs/mistral_7b.json",
        # Workload = the compile's own constraint block; no overrides.
        "workload": {},
    },
    "gpt_oss_120b": {
        "baseline_config": "configs/gpt_oss_120b.json",
        "workload": {},
    },
}

COMPILE_ARGS = [
    "--hardware", "h100",
    "--tp-options", "4,8",
    "--quality-risk-budget-pct", "1.0",
    "--allow-quality-spending",
    "--quiet",
]


def run_compile(name: str, spec: dict) -> str:
    out_dir = os.path.join(RUNS, name)
    config_path = os.path.join(out_dir, "config.json")
    cmd = [
        "ac-compile",
        "--baseline-config", spec["baseline_config"],
        *COMPILE_ARGS,
        "--out", out_dir,
    ]
    print(f"[demo] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO, check=True)
    return config_path


def price_run(name: str, spec: dict) -> dict:
    config_path = os.path.join(RUNS, name, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    priced = attach_cost_block(
        config, "h100", spec["workload"], price_tier="on_demand")
    out_path = os.path.join(RUNS, name, "config_with_cost.json")
    with open(out_path, "w") as f:
        json.dump(priced, f, indent=2)
    block = priced["metadata"]["predicted"]["cost_estimate_usd"]
    print(f"[demo] {name}: wrote {out_path}")
    return block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-only", action="store_true",
                    help="skip ac-compile; price existing runs/*/config.json")
    args = ap.parse_args()

    blocks = {}
    for name, spec in DEMOS.items():
        if not args.price_only:
            run_compile(name, spec)
        blocks[name] = price_run(name, spec)

    lines = [
        "# Task E demo — cost_estimate_usd on reference configs",
        "",
        "ac_version 0.4.0 | quality_model_version effective_capacity_v2 | "
        "git_commit c170cda | experiment_date 2026-07-17 | agent wave "
        "gate2-wave1",
        "",
        "Hardware target: h100 (AWS p5.48xlarge on-demand list price, "
        "$6.88/GPU-hr; see ac/pricing_specs/h100.json). All figures are "
        "LIST-price estimates, not quotes.",
        "",
        "| config | training_total (USD) | serving_per_1m_tokens (USD) | "
        "annual_serving_at_load (USD) |",
        "|---|---|---|---|",
    ]
    for name, b in blocks.items():
        lines.append(
            f"| {name} | {b['training_total']:,.2f} | "
            f"{b['serving_per_1m_tokens']:,.4f} | "
            f"{b['annual_serving_at_load']:,.2f} |"
        )
    summary_path = os.path.join(HERE, "cost_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[demo] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
