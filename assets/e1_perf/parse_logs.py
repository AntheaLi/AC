#!/usr/bin/env python3
# =============================================================================
# E1 performance anchors — vLLM raw-log parser (wave 1 scaffold)
#
# ac_version: 0.4.0
# quality_model_version: effective_capacity_v2
# git_commit: c170cda
# experiment_date: 2026-07-17
# agent_wave: gate2-wave1
#
# Turns raw_logs/ artifacts produced by run_benchmarks.sh into anchor records
# matching the plan's anchors.json schema (第二道-A-锚点验证研究.md section 1.3).
# Predicted fields are pre-filled from ac_predictions.json (locked at wave 1);
# observed fields stay null until the rented-node run exists. The parser is
# idempotent: re-running it after measurements arrive fills observed values
# and abs_rel_err_pct in place, without touching predictions.
#
# Usage:
#   python3 parse_logs.py                      # scaffold mode (observed=null)
#   python3 parse_logs.py --raw-dir raw_logs   # after the rented-node run
# =============================================================================
"""vLLM bench/serve raw-log parser for the E1 anchor study."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

HEADER = {
    "ac_version": "0.4.0",
    "quality_model_version": "effective_capacity_v2",
    "git_commit": "c170cda",
    "experiment_date": "2026-07-17",
    "agent_wave": "gate2-wave1",
}

HERE = os.path.dirname(os.path.abspath(__file__))

# metric name -> (bench percentile key, detailed-array key, unit)
METRIC_SPEC = {
    "decode_tbt_ms_p50": {"bench_key": "p50_tpot_ms", "fallback_key": "median_tpot_ms",
                          "detail_key": "tpots", "unit": "ms"},
    "ttft_ms_p95": {"bench_key": "p95_ttft_ms", "fallback_key": "median_ttft_ms",
                    "detail_key": "ttfts", "unit": "ms"},
    "peak_memory_gb": {"unit": "GiB"},  # from nvidia-smi sampling, not bench JSON
}

_WEIGHTS_RE = re.compile(r"Loading model weights took\s+([0-9.]+)\s*GiB", re.I)
_KVTOK_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", re.I)


def _flatten(x):
    out = []
    for item in x:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _percentile(values, q):
    vals = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not vals:
        return None
    k = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def load_bench_observed(raw_dir, anchor_key, metric):
    """Observed latency value from <anchor>_bench.json, or None."""
    path = os.path.join(raw_dir, f"{anchor_key}_bench.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        bench = json.load(fh)
    spec = METRIC_SPEC[metric]
    if spec["bench_key"] in bench and bench[spec["bench_key"]] is not None:
        return float(bench[spec["bench_key"]]), path
    if spec["fallback_key"] in bench and bench[spec["fallback_key"]] is not None:
        return float(bench[spec["fallback_key"]]), path
    detail = bench.get(spec["detail_key"])
    if detail:
        q = 50.0 if "p50" in metric else 95.0
        val = _percentile(_flatten(detail), q)
        if val is not None:
            return float(val), path
    return None, path


def load_memory_observed(raw_dir, anchor_key):
    """Peak per-GPU memory.used (GiB) from <anchor>_nvidia_smi.csv, or None."""
    path = os.path.join(raw_dir, f"{anchor_key}_nvidia_smi.csv")
    if not os.path.exists(path):
        return None, None
    peak_per_gpu = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                idx = int(row[0].strip())
                mib = float(row[1].strip().rstrip(" MiB"))
            except ValueError:
                continue
            peak_per_gpu[idx] = max(peak_per_gpu.get(idx, 0.0), mib)
    if not peak_per_gpu:
        return None, path
    return max(peak_per_gpu.values()) / 1024.0, path


def load_server_notes(raw_dir, anchor_key):
    """Cross-check facts from <anchor>_server.log (weights GiB, KV tokens)."""
    path = os.path.join(raw_dir, f"{anchor_key}_server.log")
    notes = {}
    if not os.path.exists(path):
        return notes
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return notes
    w = [float(m) for m in _WEIGHTS_RE.findall(text)]
    if w:
        notes["vllm_weights_gib_per_gpu"] = max(w)
    kv = [int(m.replace(",", "")) for m in _KVTOK_RE.findall(text)]
    if kv:
        notes["vllm_kv_cache_tokens_per_gpu"] = max(kv)
    return notes


def build_records(predictions_path, raw_dir):
    with open(predictions_path) as fh:
        preds = json.load(fh)
    records = []
    for a in preds["anchors"]:
        key, metric = a["anchor_key"], a["metric"]
        if metric == "peak_memory_gb":
            observed, obs_path = load_memory_observed(raw_dir, key)
        else:
            observed, obs_path = load_bench_observed(raw_dir, key, metric)
        notes = load_server_notes(raw_dir, key)

        abs_rel = None
        if observed is not None and a["predicted"]:
            abs_rel = abs(observed - a["predicted"]) / a["predicted"] * 100.0

        rec = {
            # --- plan section 1.3 schema keys ---
            "anchor_id": f"e1_{key}_{metric}",
            "experiment": "E1",
            "model": a["model"],
            "hardware": "h100",
            "metric": metric,
            "predicted": a["predicted"],
            "observed": observed,
            "abs_rel_err_pct": abs_rel,
            "provenance": (
                {
                    "tier": "T1",
                    "source": f"vllm {notes.get('vllm_version', '')} bench/serve on rented "
                              f"H100 SXM x8; raw artifact: {os.path.relpath(obs_path, HERE)}"
                              if obs_path else "raw_logs/",
                    "date": datetime.fromtimestamp(
                        os.path.getmtime(obs_path), tz=timezone.utc
                    ).date().isoformat() if obs_path else None,
                }
                if observed is not None else None
            ),
            "ac_version": HEADER["ac_version"],
            "quality_model_version": HEADER["quality_model_version"],
            # --- scaffold extension keys (explicitly additive) ---
            "anchor_key": key,
            "tp": a["tp"],
            "ep": a["ep"],
            "status": "observation_pending" if observed is None else "observed",
            "predicted_provenance": {
                "tier": "model-prediction",
                "ac_command": a["ac_command"],
                "ac_raw_output": a["ac_raw_output"],
                "config_path": a["config_path"],
                "date": HEADER["experiment_date"],
            },
            "observed_notes": notes or None,
        }
        records.append(rec)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=os.path.join(HERE, "raw_logs"))
    ap.add_argument("--predictions", default=os.path.join(HERE, "ac_predictions.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "anchors_e1_parsed.json"))
    args = ap.parse_args(argv)

    records = build_records(args.predictions, args.raw_dir)
    payload = {
        "_header": {**HEADER,
                    "file_role": "E1 parsed anchor records (anchors.json schema, "
                                 "plan section 1.3); observed=null until the "
                                 "rented-node run exists"},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": os.path.abspath(args.raw_dir),
        "n_records": len(records),
        "n_observed": sum(1 for r in records if r["observed"] is not None),
        "records": records,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}: {payload['n_records']} records, "
          f"{payload['n_observed']} with observations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
