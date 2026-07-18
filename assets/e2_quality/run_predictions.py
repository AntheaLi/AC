#!/usr/bin/env python3
# validation/e2_quality/run_predictions.py
# ----------------------------------------------------------------------------
# Gate 2 wave-1 header:
#   ac_version:            0.4.0
#   quality_model_version: effective_capacity_v2
#   git_commit:            c170cda
#   experiment_date:       2026-07-17
#   agent_wave:            gate2-wave1
#
# E2 quality-anchor prediction runner (wave 1 of 2: AC predictions ONLY; no
# observed/real loss values are gathered here — that is wave 2).
#
# For each anchor config in model_configs/ and each prior variant:
#   * stock    — AC's shipped in-code DEFAULT_QUALITY_CONSTANTS
#                (AC_QUALITY_DEFAULTS explicitly unset)
#   * pre2024  — AC_QUALITY_DEFAULTS=validation/e2_quality/priors_pre2024.yaml
#                (temporal-holdout overlay from build_priors_pre2024.py)
# the script
#   1. runs the REAL CLI pipeline (smoke-load + prediction):
#        ac-delta-eval --baseline-config <cfg> --hardware h100 \
#          --training-tokens <published T> \
#          --pretraining-context-length <published C> --context-length <C> \
#          --apply scale_n_layers:delta=0 --no-pareto --json \
#          --out runs/<anchor>__<variant>
#      scale_n_layers with delta=0 is a no-op transformation, so the
#      evaluation's baseline arm IS the anchor architecture;
#   2. re-evaluates the same (config, constraints) in-process through
#      ac.optimizer.evaluate_candidate to read quality.uncertainty_total
#      (not serialized by the CLI's evaluation.json), and CROSS-CHECKS the
#      in-process predicted_loss against the CLI's value;
#   3. emits the loss CI with AC's canonical convention
#      (optimizer._loss_interval): ci = loss * (1 +/- uncertainty_total).
#
# Evaluation context is set equal to each model's published pretraining
# context, so the context_utility term is identically zero and the anchor
# measures the training-distribution loss at the published token count.
#
# Output: validation/e2_quality/ac_predictions.json
# ----------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ac.baseline import load_baseline_model  # noqa: E402
from ac.optimizer import (  # noqa: E402
    DeploymentConstraints,
    evaluate_candidate,
)

CONFIG_DIR = HERE / "model_configs"
RUNS_DIR = HERE / "runs"
OUT_PATH = HERE / "ac_predictions.json"
PRIORS_PATH = HERE / "priors_pre2024.yaml"

ANCHORS = [
    {
        "anchor_id": "e2_olmo2_7b",
        "family": "OLMo-2",
        "config": "olmo2_7b.json",
        "provenance": "olmo2_7b.provenance.json",
    },
    {
        "anchor_id": "e2_pythia_1p4b",
        "family": "Pythia",
        "config": "pythia_1p4b.json",
        "provenance": "pythia_1p4b.provenance.json",
    },
    {
        "anchor_id": "e2_pythia_12b",
        "family": "Pythia",
        "config": "pythia_12b.json",
        "provenance": "pythia_12b.provenance.json",
    },
    {
        "anchor_id": "e2_smollm3_3b",
        "family": "SmolLM3",
        "config": "smollm3_3b.json",
        "provenance": "smollm3_3b.provenance.json",
    },
]

VARIANTS = ["stock", "pre2024"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variant_env(variant: str) -> dict:
    env = dict(os.environ)
    if variant == "pre2024":
        env["AC_QUALITY_DEFAULTS"] = str(PRIORS_PATH)
    else:
        env.pop("AC_QUALITY_DEFAULTS", None)
    return env


def _apply_env(variant: str) -> None:
    if variant == "pre2024":
        os.environ["AC_QUALITY_DEFAULTS"] = str(PRIORS_PATH)
    else:
        os.environ.pop("AC_QUALITY_DEFAULTS", None)


def run_cli(anchor: dict, variant: str, tokens: int, ctx: int) -> dict:
    out_dir = RUNS_DIR / f"{anchor['anchor_id']}__{variant}"
    cmd = [
        "ac-delta-eval",
        "--baseline-config", str(CONFIG_DIR / anchor["config"]),
        "--hardware", "h100",
        "--training-tokens", str(tokens),
        "--pretraining-context-length", str(ctx),
        "--context-length", str(ctx),
        "--apply", "scale_n_layers:delta=0",
        "--no-pareto",
        "--json",
        "--out", str(out_dir),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=_variant_env(variant),
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI failed for {anchor['anchor_id']} [{variant}]:\n"
            f"cmd: {' '.join(cmd)}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    with (out_dir / "evaluation.json").open() as f:
        evaluation = json.load(f)
    return {
        "cmd": cmd,
        "evaluation": evaluation,
        "stderr_tail": proc.stderr.strip().splitlines()[-5:],
    }


def run_inprocess(anchor: dict, variant: str, tokens: int, ctx: int) -> dict:
    _apply_env(variant)
    bm = load_baseline_model(str(CONFIG_DIR / anchor["config"]))
    candidate = bm.candidate
    # Mirror cli_delta_eval's parallelism resolution: explicit flag >
    # config parallelism block > 8/1/8. No CLI flags are passed here, so
    # the config block governs (tp/pp are 0 on the candidate when the
    # config omits them).
    cfg_par = (bm.config.get("parallelism") or {})
    resolved_tp = int(getattr(candidate, "tp_degree", 0) or 8)
    resolved_pp = int(getattr(candidate, "pp_degree", 0) or 1)
    resolved_dp = int(cfg_par.get("data_parallel") or 8)
    candidate.tp_degree = resolved_tp
    candidate.pp_degree = resolved_pp
    constraints = DeploymentConstraints(
        target_params_b=candidate.total_params / 1e9,
        training_tokens=tokens,
        unique_training_tokens=None,
        pretraining_context_length=ctx,
        quality_model_version="effective_capacity_v2",
        context_length=ctx,
        prompt_len=ctx,
        serving_tbt_ms=50.0,
        serving_ttft_ms=2000.0,
        serving_batch=1,
        tp=resolved_tp,
        pp=resolved_pp,
        dp=resolved_dp,
    )
    ev = evaluate_candidate(candidate, "h100", constraints)
    q = ev.quality
    return {
        "predicted_loss": float(ev.predicted_loss),
        "uncertainty_total": float(getattr(q, "uncertainty_total", 0.0)),
        "chinchilla_baseline": float(getattr(q, "chinchilla_baseline", 0.0)),
        "confidence": str(getattr(q, "confidence", "")),
        "dominant_penalty": str(getattr(q, "dominant_penalty", "")),
        "quality_sentinel": bool(getattr(q, "quality_sentinel", False)),
        "uncertainty_breakdown": {
            k: float(v)
            for k, v in (getattr(q, "uncertainty_breakdown", {}) or {}).items()
        },
        "term_values": {
            k: float(getattr(t, "value", 0.0))
            for k, t in (getattr(q, "terms", {}) or {}).items()
        },
        "ledger_total_params_b": float(candidate.total_params_b),
        "ledger_active_params_b": float(candidate.active_params_b),
        "baseline_warnings": list(bm.warnings),
        "meets_constraints": bool(getattr(ev, "meets_constraints", True)),
    }


def main() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for anchor in ANCHORS:
        with (CONFIG_DIR / anchor["provenance"]).open() as f:
            prov = json.load(f)
        tokens = int(prov["prediction_inputs"]["training_tokens"])
        ctx = int(prov["prediction_inputs"]["pretraining_context_length"])
        eval_ctx = int(prov["prediction_inputs"]["evaluation_context_length"])
        assert eval_ctx == ctx, anchor["anchor_id"]

        variants_out = {}
        for variant in VARIANTS:
            cli = run_cli(anchor, variant, tokens, ctx)
            ip = run_inprocess(anchor, variant, tokens, ctx)
            cli_loss = float(
                cli["evaluation"]["metrics"]["predicted_loss"]["baseline"])
            cli_loss_cand = float(
                cli["evaluation"]["metrics"]["predicted_loss"]["candidate"])
            rel_diff = abs(cli_loss - ip["predicted_loss"]) / max(
                1e-12, abs(ip["predicted_loss"]))
            crosscheck = {
                "cli_baseline_loss": cli_loss,
                "cli_candidate_loss": cli_loss_cand,
                "inprocess_loss": ip["predicted_loss"],
                "rel_diff": rel_diff,
                "pass": bool(
                    rel_diff < 1e-9 and abs(cli_loss_cand - cli_loss) < 1e-12),
            }
            if not crosscheck["pass"]:
                raise RuntimeError(
                    f"cross-check failed for {anchor['anchor_id']} [{variant}]: "
                    f"{crosscheck}")
            u = ip["uncertainty_total"]
            loss = ip["predicted_loss"]
            variants_out[variant] = {
                "predicted_loss": round(loss, 6),
                "loss_ci_low": round(loss * (1.0 - u), 6),
                "loss_ci_high": round(loss * (1.0 + u), 6),
                "uncertainty_total_pct": round(u * 100.0, 4),
                "chinchilla_baseline": round(ip["chinchilla_baseline"], 6),
                "confidence": ip["confidence"],
                "dominant_penalty": ip["dominant_penalty"],
                "quality_sentinel": ip["quality_sentinel"],
                "uncertainty_breakdown_pct": {
                    k: round(v * 100.0, 4)
                    for k, v in ip["uncertainty_breakdown"].items()
                },
                "term_values": ip["term_values"],
                "ledger_total_params_b": ip["ledger_total_params_b"],
                "ledger_active_params_b": ip["ledger_active_params_b"],
                "baseline_warnings": ip["baseline_warnings"],
                "cli_cmd": cli["cmd"],
                "cli_stderr_tail": cli["stderr_tail"],
                "crosscheck": crosscheck,
            }
        records.append({
            "anchor_id": anchor["anchor_id"],
            "family": anchor["family"],
            "config": f"validation/e2_quality/model_configs/{anchor['config']}",
            "provenance": (
                f"validation/e2_quality/model_configs/{anchor['provenance']}"),
            "training_tokens": tokens,
            "pretraining_context_length": ctx,
            "evaluation_context_length": eval_ctx,
            "variants": variants_out,
            "delta_pre2024_minus_stock": {
                "predicted_loss": round(
                    variants_out["pre2024"]["predicted_loss"]
                    - variants_out["stock"]["predicted_loss"], 6),
                "uncertainty_total_pct": round(
                    variants_out["pre2024"]["uncertainty_total_pct"]
                    - variants_out["stock"]["uncertainty_total_pct"], 4),
            },
        })

    payload = {
        "_header": {
            "ac_version": "0.4.0",
            "quality_model_version": "effective_capacity_v2",
            "git_commit": "c170cda",
            "experiment_date": "2026-07-17",
            "agent_wave": "gate2-wave1",
            "study": "E2 quality anchors — AC predictions only (wave 1 of 2); "
                     "no observed loss values recorded here by design",
        },
        "priors": {
            "stock": "ac.quality_model.DEFAULT_QUALITY_CONSTANTS "
                     "(AC_QUALITY_DEFAULTS unset)",
            "pre2024": "validation/e2_quality/priors_pre2024.yaml",
            "pre2024_sha256": _sha256(PRIORS_PATH),
        },
        "ci_convention": "loss_ci = predicted_loss * (1 +/- uncertainty_total) "
                         "(ac.optimizer._loss_interval)",
        "evaluation_protocol": {
            "hardware": "h100",
            "cli": "ac-delta-eval --apply scale_n_layers:delta=0 --no-pareto "
                   "--json (no-op delta; baseline arm is the anchor)",
            "workload_context": "evaluation context == published pretraining "
                                "context -> context_utility term == 0",
            "unique_training_tokens": None,
        },
        "anchors": records,
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_PATH}")
    for r in records:
        for v in VARIANTS:
            vv = r["variants"][v]
            print(
                f"{r['anchor_id']:>18} [{v:>7}] loss={vv['predicted_loss']:.4f} "
                f"CI=[{vv['loss_ci_low']:.4f}, {vv['loss_ci_high']:.4f}] "
                f"unc={vv['uncertainty_total_pct']:.2f}% "
                f"conf={vv['confidence']} xcheck={vv['crosscheck']['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
