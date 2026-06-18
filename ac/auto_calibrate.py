"""
Auto-calibration utilities for AC lab deployments.

The fitter consumes lab measurement rows with predicted and observed metrics,
then writes a calibration pack:

  calibration_pack.json       full machine-readable fit summary
  quality_overrides.json      YAML-compatible overlay for AC_QUALITY_DEFAULTS
  hardware_specs/*.json       hardware specs with fitted efficiency constants
  report.md                   human-readable fit notes and usage

Input rows may be JSON, JSONL, or CSV. Field names are intentionally tolerant;
examples:

  {
    "id": "h100_mistral_7b_decode",
    "hardware": "h100",
    "predicted_loss": 2.03,
    "observed_loss": 2.08,
    "predicted_uncertainty_total_pct": 3.1,
    "predicted_training_tps": 11800,
    "observed_training_tps": 10400,
    "predicted_serving_tbt_ms": 6.2,
    "observed_serving_tbt_ms": 7.1
  }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_HERE = Path(__file__).resolve().parent
_DEFAULT_SPEC_DIR = _HERE / "hardware_specs"

_HARDWARE_SPEC_FILES = {
    "h100": "h100_sxm.json",
    "b200": "b200.json",
    "tpu_v5e": "tpu_v5e.json",
    "tpu_v5p": "tpu_v5p.json",
    "trainium2": "trainium2.json",
    "trn2": "trainium2.json",
    "trainium3": "trainium3.json",
    "trn3": "trainium3.json",
}

_CANONICAL_HW = {
    "h100_sxm": "h100",
    "h100": "h100",
    "b200": "b200",
    "tpu_v5e": "tpu_v5e",
    "tpu_v5p": "tpu_v5p",
    "trainium2": "trainium2",
    "trn2": "trainium2",
    "trainium3": "trainium3",
    "trn3": "trainium3",
}

_ALIASES = {
    "id": ("id", "name", "run_id", "case_id"),
    "hardware": ("hardware", "hw", "accelerator", "metadata.input_hardware"),
    "predicted_loss": (
        "predicted_loss",
        "prediction.loss",
        "predicted.loss",
        "quality.predicted_loss",
        "metadata.predicted.predicted_loss",
    ),
    "observed_loss": (
        "observed_loss",
        "actual_loss",
        "measured_loss",
        "observed.loss",
        "actual.loss",
        "measured.loss",
    ),
    "predicted_uncertainty_total_pct": (
        "predicted_uncertainty_total_pct",
        "uncertainty_total_pct",
        "predicted.uncertainty_total_pct",
        "metadata.predicted.uncertainty_total_pct",
    ),
    "predicted_training_tps": (
        "predicted_training_tps",
        "training_tps_pred",
        "predicted.training_tps",
        "metadata.predicted.training_throughput_tokens_per_sec",
    ),
    "observed_training_tps": (
        "observed_training_tps",
        "actual_training_tps",
        "measured_training_tps",
        "observed.training_tps",
    ),
    "predicted_serving_tbt_ms": (
        "predicted_serving_tbt_ms",
        "predicted_tbt_ms",
        "serving_tbt_ms_pred",
        "predicted.serving_tbt_ms",
        "metadata.predicted.serving_tbt_ms",
    ),
    "observed_serving_tbt_ms": (
        "observed_serving_tbt_ms",
        "actual_serving_tbt_ms",
        "measured_serving_tbt_ms",
        "observed.tbt_ms",
    ),
    "predicted_prefill_time_ms": (
        "predicted_prefill_time_ms",
        "predicted_ttft_ms",
        "predicted.serving_ttft_ms",
        "metadata.predicted.serving_ttft_ms",
    ),
    "observed_prefill_time_ms": (
        "observed_prefill_time_ms",
        "actual_prefill_time_ms",
        "observed_ttft_ms",
        "measured_ttft_ms",
    ),
    "predicted_memory_per_gpu_gb": (
        "predicted_memory_per_gpu_gb",
        "predicted.memory_per_gpu_gb",
        "metadata.predicted.memory_per_gpu_gb",
    ),
    "observed_memory_per_gpu_gb": (
        "observed_memory_per_gpu_gb",
        "actual_memory_per_gpu_gb",
        "measured_memory_per_gpu_gb",
    ),
}


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    try:
        cleaned = str(value).strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        out = float(cleaned)
        return out if math.isfinite(out) else None
    except ValueError:
        return None


def _nested_get(row: Dict[str, Any], path: str) -> Any:
    if path in row:
        return row[path]
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _value(row: Dict[str, Any], key: str) -> Any:
    for alias in _ALIASES[key]:
        got = _nested_get(row, alias)
        if got is not None:
            return got
    return None


def _numeric(row: Dict[str, Any], key: str) -> Optional[float]:
    return _to_float(_value(row, key))


def _read_measurements(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("measurements", "records", "runs", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(
        "Measurement JSON must be a list or an object with measurements/records/runs/data."
    )


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def _coverage(errors_pct: Sequence[float], uncertainties_pct: Sequence[float]) -> float:
    if not errors_pct:
        return 0.0
    ok = 0
    for err, unc in zip(errors_pct, uncertainties_pct):
        if abs(err) <= max(0.0, unc):
            ok += 1
    return ok / len(errors_pct)


def _canonical_hw(raw: Any, default: Optional[str]) -> Optional[str]:
    key = str(raw or default or "").strip().lower()
    return _CANONICAL_HW.get(key, key or None)


def _fit_quality(
    rows: Sequence[Dict[str, Any]],
    *,
    target_coverage: float,
    default_uncertainty_pct: float,
    min_uncertainty_pct: float,
) -> Dict[str, Any]:
    errors_pct: List[float] = []
    uncertainties_pct: List[float] = []
    examples: List[Dict[str, Any]] = []
    for row in rows:
        pred = _numeric(row, "predicted_loss")
        obs = _numeric(row, "observed_loss")
        if pred is None or obs is None or pred <= 0:
            continue
        unc = _numeric(row, "predicted_uncertainty_total_pct")
        if unc is None:
            unc = default_uncertainty_pct
        unc = max(float(unc), min_uncertainty_pct)
        err = (obs - pred) / pred * 100.0
        errors_pct.append(err)
        uncertainties_pct.append(unc)
        if len(examples) < 8:
            examples.append({
                "id": _value(row, "id") or f"row_{len(errors_pct)}",
                "predicted_loss": pred,
                "observed_loss": obs,
                "relative_error_pct": round(err, 4),
                "predicted_uncertainty_pct": round(unc, 4),
            })

    if not errors_pct:
        return {
            "n": 0,
            "calibration_multiplier": 1.0,
            "coverage_before": 0.0,
            "coverage_after": 0.0,
            "median_bias_pct": 0.0,
            "abs_error_p50_pct": 0.0,
            "abs_error_target_quantile_pct": 0.0,
            "examples": examples,
        }

    ratios = [
        abs(err) / max(unc, 1e-9)
        for err, unc in zip(errors_pct, uncertainties_pct)
    ]
    multiplier = max(1.0, _quantile(ratios, target_coverage))
    scaled_unc = [u * multiplier for u in uncertainties_pct]
    abs_errors = [abs(e) for e in errors_pct]
    return {
        "n": len(errors_pct),
        "calibration_multiplier": round(multiplier, 6),
        "coverage_before": round(_coverage(errors_pct, uncertainties_pct), 4),
        "coverage_after": round(_coverage(errors_pct, scaled_unc), 4),
        "median_bias_pct": round(_median(errors_pct), 6),
        "mean_bias_pct": round(_mean(errors_pct), 6),
        "abs_error_p50_pct": round(_quantile(abs_errors, 0.50), 6),
        "abs_error_p90_pct": round(_quantile(abs_errors, 0.90), 6),
        "abs_error_target_quantile_pct": round(_quantile(abs_errors, target_coverage), 6),
        "median_predicted_uncertainty_pct": round(_median(uncertainties_pct), 6),
        "examples": examples,
    }


def _ratio_pairs(
    rows: Sequence[Dict[str, Any]],
    predicted_key: str,
    observed_key: str,
    *,
    hardware: str,
) -> List[float]:
    ratios = []
    for row in rows:
        pred = _numeric(row, predicted_key)
        obs = _numeric(row, observed_key)
        if pred is None or obs is None or pred <= 0 or obs <= 0:
            continue
        row_hw = _canonical_hw(_value(row, "hardware"), None)
        if row_hw != hardware:
            continue
        ratios.append(obs / pred)
    return ratios


def _fit_hardware(
    rows: Sequence[Dict[str, Any]],
    *,
    default_hardware: Optional[str],
) -> Dict[str, Any]:
    hardware_ids = sorted({
        hw for hw in (
            _canonical_hw(_value(row, "hardware"), default_hardware)
            for row in rows
        )
        if hw
    })
    fits: Dict[str, Any] = {}
    for hw in hardware_ids:
        training = _ratio_pairs(
            rows, "predicted_training_tps", "observed_training_tps",
            hardware=hw)
        decode = _ratio_pairs(
            rows, "predicted_serving_tbt_ms", "observed_serving_tbt_ms",
            hardware=hw)
        prefill = _ratio_pairs(
            rows, "predicted_prefill_time_ms", "observed_prefill_time_ms",
            hardware=hw)
        memory = _ratio_pairs(
            rows, "predicted_memory_per_gpu_gb", "observed_memory_per_gpu_gb",
            hardware=hw)

        fit = {
            "n_training_tps": len(training),
            "n_serving_tbt": len(decode),
            "n_prefill": len(prefill),
            "n_memory": len(memory),
            "training_tps_observed_over_predicted": (
                round(_median(training), 6) if training else None
            ),
            "serving_tbt_observed_over_predicted": (
                round(_median(decode), 6) if decode else None
            ),
            "prefill_time_observed_over_predicted": (
                round(_median(prefill), 6) if prefill else None
            ),
            "memory_observed_over_predicted": (
                round(_median(memory), 6) if memory else None
            ),
        }
        fit["training_efficiency_multiplier"] = fit["training_tps_observed_over_predicted"]
        fit["decode_efficiency_multiplier"] = (
            round(1.0 / fit["serving_tbt_observed_over_predicted"], 6)
            if fit["serving_tbt_observed_over_predicted"] else None
        )
        fit["prefill_efficiency_multiplier"] = (
            round(1.0 / fit["prefill_time_observed_over_predicted"], 6)
            if fit["prefill_time_observed_over_predicted"] else None
        )
        fits[hw] = fit
    return fits


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _copy_and_calibrate_specs(
    *,
    base_spec_dir: Path,
    out_spec_dir: Path,
    hardware_fits: Dict[str, Any],
    min_efficiency: float,
    max_efficiency: float,
) -> Dict[str, Any]:
    out_spec_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    warnings = []
    for src in sorted(base_spec_dir.glob("*.json")):
        dst = out_spec_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(src.name)

    updates: Dict[str, Any] = {}
    for hw, fit in hardware_fits.items():
        filename = _HARDWARE_SPEC_FILES.get(hw)
        if not filename:
            warnings.append(f"No known hardware spec filename for {hw}; skipped.")
            continue
        path = out_spec_dir / filename
        if not path.exists():
            warnings.append(f"Hardware spec {filename} missing in {base_spec_dir}; skipped.")
            continue
        with path.open() as f:
            spec = json.load(f)
        cal = dict(spec.get("calibration", {}))
        before = dict(cal)
        for key, multiplier_key in (
            ("training_system_efficiency", "training_efficiency_multiplier"),
            ("decode_system_efficiency", "decode_efficiency_multiplier"),
            ("prefill_system_efficiency", "prefill_efficiency_multiplier"),
        ):
            multiplier = fit.get(multiplier_key)
            if multiplier is None or key not in cal:
                continue
            cal[key] = round(
                _clamp(float(cal[key]) * float(multiplier),
                       min_efficiency, max_efficiency),
                6,
            )
        spec["calibration"] = cal
        notes = spec.setdefault("notes", {})
        notes["auto_calibration"] = (
            "Generated by ac-auto-calibrate from lab measurements. "
            "Review sample count and coverage before treating as production."
        )
        with path.open("w") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")
        updates[hw] = {
            "file": str(path),
            "before": before,
            "after": cal,
            "fit": fit,
        }
    return {"copied": copied, "updates": updates, "warnings": warnings}


def _quality_overlay(fit: Dict[str, Any], *, target_coverage: float) -> Dict[str, Any]:
    source = (
        "ac-auto-calibrate "
        + datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    return {
        "uncertainty": {
            "calibration_multiplier": fit["calibration_multiplier"],
            "calibration_source": source,
            "calibration_target_coverage": target_coverage,
            "calibration_n": fit["n"],
            "calibration_coverage_before": fit["coverage_before"],
            "calibration_coverage_after": fit["coverage_after"],
            "calibration_median_bias_pct": fit["median_bias_pct"],
            "calibration_abs_error_p90_pct": fit.get("abs_error_p90_pct", 0.0),
        }
    }


def _render_report(pack: Dict[str, Any]) -> str:
    q = pack["quality"]
    lines = [
        "# AC Auto-Calibration Report",
        "",
        f"Generated: {pack['generated_at']}",
        "",
        "## Quality Uncertainty",
        "",
        f"- Rows used: {q['n']}",
        f"- Target coverage: {pack['target_coverage']:.2f}",
        f"- Coverage before: {q['coverage_before']:.2%}",
        f"- Coverage after: {q['coverage_after']:.2%}",
        f"- Calibration multiplier: {q['calibration_multiplier']:.3f}",
        f"- Median relative loss bias: {q['median_bias_pct']:+.3f}%",
        f"- P90 absolute relative loss error: {q.get('abs_error_p90_pct', 0.0):.3f}%",
        "",
        "Use `quality_overrides.json` via:",
        "",
        "```bash",
        "AC_QUALITY_DEFAULTS=/path/to/calibration_pack/quality_overrides.json ac-compile ...",
        "```",
        "",
        "## Hardware Efficiency",
        "",
    ]
    hardware = pack["hardware"]
    if not hardware:
        lines.append("No hardware throughput/latency rows were available.")
    else:
        lines.extend([
            "| Hardware | Train Eff x | Decode Eff x | Prefill Eff x | Train n | TBT n | Prefill n |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for hw, fit in sorted(hardware.items()):
            lines.append(
                f"| {hw} | "
                f"{_fmt_optional(fit.get('training_efficiency_multiplier'))} | "
                f"{_fmt_optional(fit.get('decode_efficiency_multiplier'))} | "
                f"{_fmt_optional(fit.get('prefill_efficiency_multiplier'))} | "
                f"{fit.get('n_training_tps', 0)} | "
                f"{fit.get('n_serving_tbt', 0)} | "
                f"{fit.get('n_prefill', 0)} |"
            )
        lines.extend([
            "",
            "Use calibrated hardware specs via:",
            "",
            "```bash",
            "AC_HARDWARE_SPEC_DIR=/path/to/calibration_pack/hardware_specs ac-compile ...",
            "```",
        ])
    warnings = pack.get("spec_output", {}).get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
    lines.extend([
        "",
        "## Notes",
        "",
        "- Quality calibration scales uncertainty intervals; it does not bias-correct the predicted loss point estimate.",
        "- Hardware calibration adjusts system-efficiency constants from median observed/predicted ratios.",
        "- Keep separate packs per cluster topology, kernel stack, scheduler policy, and model family when those differ materially.",
    ])
    return "\n".join(lines) + "\n"


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def fit_command(args: argparse.Namespace) -> int:
    rows = _read_measurements(Path(args.measurements))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_coverage = float(args.target_coverage)
    if not 0.0 < target_coverage < 1.0:
        print("ERROR: --target-coverage must be between 0 and 1.", file=sys.stderr)
        return 2

    quality = _fit_quality(
        rows,
        target_coverage=target_coverage,
        default_uncertainty_pct=float(args.default_quality_uncertainty_pct),
        min_uncertainty_pct=float(args.min_quality_uncertainty_pct),
    )
    hardware = _fit_hardware(rows, default_hardware=args.hardware)
    base_spec_dir = Path(args.base_hardware_spec_dir or os.environ.get(
        "AC_HARDWARE_SPEC_DIR", _DEFAULT_SPEC_DIR))
    spec_output = _copy_and_calibrate_specs(
        base_spec_dir=base_spec_dir,
        out_spec_dir=out_dir / "hardware_specs",
        hardware_fits=hardware,
        min_efficiency=float(args.min_efficiency),
        max_efficiency=float(args.max_efficiency),
    )
    overlay = _quality_overlay(quality, target_coverage=target_coverage)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pack = {
        "schema_version": "ac_lab_calibration_0.1",
        "generated_at": generated_at,
        "measurements_path": str(Path(args.measurements)),
        "rows_loaded": len(rows),
        "target_coverage": target_coverage,
        "quality": quality,
        "quality_overlay": overlay,
        "hardware": hardware,
        "spec_output": spec_output,
        "usage": {
            "quality_env": f"AC_QUALITY_DEFAULTS={out_dir / 'quality_overrides.json'}",
            "hardware_env": f"AC_HARDWARE_SPEC_DIR={out_dir / 'hardware_specs'}",
        },
    }

    (out_dir / "quality_overrides.json").write_text(
        json.dumps(overlay, indent=2) + "\n")
    (out_dir / "calibration_pack.json").write_text(
        json.dumps(pack, indent=2) + "\n")
    (out_dir / "report.md").write_text(_render_report(pack))

    print(f"Wrote calibration pack to: {out_dir}")
    print(f"Quality rows: {quality['n']}  multiplier={quality['calibration_multiplier']:.3f}")
    if hardware:
        print("Hardware fits: " + ", ".join(sorted(hardware.keys())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ac-auto-calibrate",
        description="Fit lab-local AC calibration packs from observed measurements.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit", help="Fit quality uncertainty and hardware efficiency factors")
    fit.add_argument("--measurements", required=True,
                     help="JSON, JSONL, or CSV measurement rows")
    fit.add_argument("--out", required=True,
                     help="Output directory for calibration_pack.json and overlays")
    fit.add_argument("--hardware", default=None,
                     help="Default hardware id for rows that omit hardware")
    fit.add_argument("--target-coverage", type=float, default=0.90,
                     help="Target quality uncertainty coverage quantile")
    fit.add_argument("--default-quality-uncertainty-pct", type=float, default=3.0,
                     help="Fallback predicted uncertainty when rows omit it")
    fit.add_argument("--min-quality-uncertainty-pct", type=float, default=0.5,
                     help="Floor used when computing uncertainty coverage")
    fit.add_argument("--base-hardware-spec-dir", default=None,
                     help="Base hardware_specs directory to copy and tune")
    fit.add_argument("--min-efficiency", type=float, default=0.03,
                     help="Lower clamp for fitted system-efficiency constants")
    fit.add_argument("--max-efficiency", type=float, default=0.95,
                     help="Upper clamp for fitted system-efficiency constants")
    fit.set_defaults(func=fit_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
