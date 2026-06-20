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
import hashlib
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
    "architecture_family": (
        "architecture_family",
        "arch_family",
        "model_family",
        "family",
        "architecture.family",
        "metadata.architecture_family",
    ),
    "model_type": (
        "model_type",
        "architecture.model_type",
        "metadata.model_type",
    ),
    "active_params_b": (
        "active_params_b",
        "active_params_B",
        "active_params",
        "metadata.predicted.active_params_b",
        "metadata.predicted.active_params_B",
    ),
    "total_params_b": (
        "total_params_b",
        "total_params_B",
        "total_params",
        "metadata.predicted.total_params_b",
        "metadata.predicted.total_params_B",
    ),
    "training_tokens": (
        "training_tokens",
        "tokens",
        "pretrain_tokens",
        "metadata.training_tokens",
        "metadata.input_constraints.training_tokens",
    ),
    "context_length": (
        "context_length",
        "sequence_length",
        "seq_len",
        "metadata.input_constraints.context_length",
    ),
}

_EVAL_MAP_ALIASES = (
    "evals",
    "eval_scores",
    "observed_evals",
    "observed_eval_scores",
    "actual_evals",
    "measured_evals",
    "benchmarks",
    "observed.benchmarks",
)

_PREDICTED_EVAL_MAP_ALIASES = (
    "predicted_evals",
    "predicted_eval_scores",
    "predicted.benchmarks",
    "metadata.predicted.evals",
    "metadata.predicted.eval_scores",
)

_EVAL_FEATURE_NAMES = [
    "intercept",
    "predicted_loss",
    "log_active_params_b",
    "log_total_params_b",
    "log_training_tokens_t",
    "log_context_length",
    "is_moe",
    "is_state_or_hybrid",
]


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


def _normalise_eval_name(name: Any) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _flatten(row: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def _map_at(row: Dict[str, Any], aliases: Sequence[str]) -> Dict[str, Any]:
    for alias in aliases:
        got = _nested_get(row, alias)
        if isinstance(got, dict):
            return got
    return {}


def _collect_observed_evals(row: Dict[str, Any]) -> Dict[str, float]:
    """Extract observed eval scores from tolerant map or flat field names."""
    evals: Dict[str, float] = {}
    for key, value in _map_at(row, _EVAL_MAP_ALIASES).items():
        score = _to_float(value)
        if score is not None:
            evals[_normalise_eval_name(key)] = score

    flat = _flatten(row)
    for key, value in flat.items():
        score = _to_float(value)
        if score is None:
            continue
        norm = _normalise_eval_name(key.replace(".", "_"))
        prefixes = ("observed_", "actual_", "measured_", "eval_")
        for prefix in prefixes:
            if norm.startswith(prefix) and not norm.startswith("observed_loss"):
                eval_name = norm[len(prefix):]
                if eval_name not in {
                    "loss", "training_tps", "serving_tbt_ms",
                    "prefill_time_ms", "memory_per_gpu_gb",
                } and not eval_name.startswith("scores_"):
                    evals.setdefault(_normalise_eval_name(eval_name), score)
    return evals


def _collect_predicted_evals(row: Dict[str, Any]) -> Dict[str, float]:
    evals: Dict[str, float] = {}
    for key, value in _map_at(row, _PREDICTED_EVAL_MAP_ALIASES).items():
        score = _to_float(value)
        if score is not None:
            evals[_normalise_eval_name(key)] = score

    flat = _flatten(row)
    for key, value in flat.items():
        score = _to_float(value)
        if score is None:
            continue
        norm = _normalise_eval_name(key.replace(".", "_"))
        if norm.startswith("predicted_") and not norm.startswith("predicted_loss"):
            eval_name = norm[len("predicted_"):]
            if eval_name not in {
                "training_tps", "serving_tbt_ms", "prefill_time_ms",
                "memory_per_gpu_gb", "uncertainty_total_pct",
            } and not eval_name.startswith("eval_scores_"):
                evals.setdefault(_normalise_eval_name(eval_name), score)
    return evals


def _family(row: Dict[str, Any]) -> str:
    raw = _value(row, "architecture_family")
    if raw is None:
        raw = _value(row, "id")
    text = str(raw or "unknown").strip().lower()
    return text or "unknown"


def _scale_to_billions(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value > 1e6:
        return value / 1e9
    return value


def _tokens_to_trillions(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value > 1e6:
        return value / 1e12
    return value


def _eval_features(row: Dict[str, Any]) -> List[float]:
    active_b = _scale_to_billions(_numeric(row, "active_params_b"))
    total_b = _scale_to_billions(_numeric(row, "total_params_b"))
    if active_b is None:
        active_b = total_b
    if total_b is None:
        total_b = active_b
    tokens_t = _tokens_to_trillions(_numeric(row, "training_tokens"))
    context = _numeric(row, "context_length")
    model_type = str(_value(row, "model_type") or _family(row)).lower()
    predicted_loss = _numeric(row, "predicted_loss") or 0.0

    return [
        1.0,
        float(predicted_loss),
        math.log(max(active_b or 1.0, 1e-9)),
        math.log(max(total_b or active_b or 1.0, 1e-9)),
        math.log(max(tokens_t or 1.0, 1e-9)),
        math.log(max(context or 1.0, 1.0)),
        1.0 if "moe" in model_type or "mixture" in model_type else 0.0,
        1.0 if any(k in model_type for k in ("state", "hybrid", "mamba", "linear")) else 0.0,
    ]


def _solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(b)
    aug = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        aug[col] = [v / denom for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-12:
                continue
            aug[row] = [
                aug[row][i] - factor * aug[col][i]
                for i in range(n + 1)
            ]
    return [aug[i][-1] for i in range(n)]


def _fit_ridge(xs: Sequence[Sequence[float]], ys: Sequence[float], alpha: float) -> List[float]:
    d = len(xs[0])
    xtx = [[0.0 for _ in range(d)] for _ in range(d)]
    xty = [0.0 for _ in range(d)]
    for x, y in zip(xs, ys):
        for i in range(d):
            xty[i] += x[i] * y
            for j in range(d):
                xtx[i][j] += x[i] * x[j]
    for i in range(1, d):
        xtx[i][i] += alpha
    solved = _solve_linear_system(xtx, xty)
    if solved is None:
        # Fall back to a mean-only model if the design matrix is singular.
        coefs = [0.0 for _ in range(d)]
        coefs[0] = _mean(list(ys))
        return coefs
    return solved


def _predict(coefs: Sequence[float], x: Sequence[float]) -> float:
    return float(sum(c * v for c, v in zip(coefs, x)))


def _rmse(errors: Sequence[float]) -> float:
    if not errors:
        return 0.0
    return math.sqrt(sum(e * e for e in errors) / len(errors))


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


def _ratio_scatter_p90_pct(ratios: Sequence[float]) -> Optional[float]:
    if len(ratios) < 2:
        return None
    med = _median(ratios)
    if med <= 0:
        return None
    residuals = [abs((r / med) - 1.0) * 100.0 for r in ratios]
    return round(_quantile(residuals, 0.90), 6)


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
            "training_scatter_p90_pct": _ratio_scatter_p90_pct(training),
            "serving_tbt_scatter_p90_pct": _ratio_scatter_p90_pct(decode),
            "prefill_scatter_p90_pct": _ratio_scatter_p90_pct(prefill),
            "memory_scatter_p90_pct": _ratio_scatter_p90_pct(memory),
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


def _fit_one_eval(
    examples: Sequence[Tuple[List[float], float, str, str]],
    *,
    ridge_alpha: float,
    min_eval_rows: int,
    min_eval_families: int,
) -> Dict[str, Any]:
    xs = [x for x, _, _, _ in examples]
    ys = [y for _, y, _, _ in examples]
    families = sorted({fam for _, _, fam, _ in examples})
    coefs = _fit_ridge(xs, ys, ridge_alpha)
    preds = [_predict(coefs, x) for x in xs]
    train_errors = [p - y for p, y in zip(preds, ys)]

    heldout_errors: List[float] = []
    heldout_rows = 0
    heldout_family_reports = []
    if len(families) >= 2:
        for fam in families:
            train = [ex for ex in examples if ex[2] != fam]
            test = [ex for ex in examples if ex[2] == fam]
            if len(train) < 2 or not test:
                continue
            fam_coefs = _fit_ridge([x for x, _, _, _ in train],
                                   [y for _, y, _, _ in train],
                                   ridge_alpha)
            fam_errors = [_predict(fam_coefs, x) - y for x, y, _, _ in test]
            heldout_errors.extend(fam_errors)
            heldout_rows += len(test)
            heldout_family_reports.append({
                "family": fam,
                "n": len(test),
                "rmse": round(_rmse(fam_errors), 6),
                "mae": round(_mean([abs(e) for e in fam_errors]), 6),
            })

    train_rmse = _rmse(train_errors)
    heldout_rmse = _rmse(heldout_errors) if heldout_errors else None
    status = "validated"
    warnings = []
    if len(examples) < min_eval_rows:
        status = "experimental"
        warnings.append(
            f"Only {len(examples)} rows; recommended minimum is {min_eval_rows}."
        )
    if len(families) < min_eval_families:
        status = "experimental"
        warnings.append(
            f"Only {len(families)} architecture families; recommended minimum is {min_eval_families}."
        )
    if heldout_rmse is None:
        status = "experimental"
        warnings.append("Held-out architecture-family CV was not available.")

    score_min = min(ys)
    score_max = max(ys)
    uncertainty = heldout_rmse if heldout_rmse is not None else train_rmse
    if len(examples) < 3:
        uncertainty = max(uncertainty, abs(score_max - score_min) or 1.0)

    return {
        "status": status,
        "n": len(examples),
        "families": families,
        "feature_names": list(_EVAL_FEATURE_NAMES),
        "coefficients": [round(c, 10) for c in coefs],
        "train_rmse": round(train_rmse, 6),
        "train_mae": round(_mean([abs(e) for e in train_errors]), 6),
        "heldout_family_rmse": (
            round(heldout_rmse, 6) if heldout_rmse is not None else None
        ),
        "heldout_family_rows": heldout_rows,
        "heldout_family_cv": heldout_family_reports,
        "uncertainty": round(float(uncertainty), 6),
        "score_min": round(score_min, 6),
        "score_max": round(score_max, 6),
        "examples": [
            {"id": row_id, "family": fam, "observed": round(y, 6)}
            for _, y, fam, row_id in examples[:8]
        ],
        "warnings": warnings,
    }


def _fit_eval_models(
    rows: Sequence[Dict[str, Any]],
    *,
    ridge_alpha: float,
    min_eval_rows: int,
    min_eval_families: int,
) -> Dict[str, Any]:
    observed_by_eval: Dict[str, List[Tuple[List[float], float, str, str]]] = {}
    residual_by_eval: Dict[str, List[Tuple[List[float], float, str, str]]] = {}

    for i, row in enumerate(rows):
        row_id = str(_value(row, "id") or f"row_{i + 1}")
        family = _family(row)
        features = _eval_features(row)
        observed = _collect_observed_evals(row)
        predicted = _collect_predicted_evals(row)
        for name, score in observed.items():
            if name in {
                "loss", "training_tps", "serving_tbt_ms",
                "prefill_time_ms", "memory_per_gpu_gb",
            }:
                continue
            observed_by_eval.setdefault(name, []).append(
                (features, score, family, row_id)
            )
            if name in predicted:
                residual_by_eval.setdefault(name, []).append(
                    (features, score - predicted[name], family, row_id)
                )

    evals: Dict[str, Any] = {}
    residuals: Dict[str, Any] = {}
    warnings = []
    for name, examples in sorted(observed_by_eval.items()):
        if len(examples) < 2:
            warnings.append(f"Eval {name} has fewer than 2 rows; skipped.")
            continue
        evals[name] = _fit_one_eval(
            examples,
            ridge_alpha=ridge_alpha,
            min_eval_rows=min_eval_rows,
            min_eval_families=min_eval_families,
        )
    for name, examples in sorted(residual_by_eval.items()):
        if len(examples) < 2:
            continue
        residuals[name] = _fit_one_eval(
            examples,
            ridge_alpha=ridge_alpha,
            min_eval_rows=min_eval_rows,
            min_eval_families=min_eval_families,
        )

    if not evals:
        status = "not_configured"
    elif all(v.get("status") == "validated" for v in evals.values()):
        status = "validated"
    else:
        status = "experimental"

    return {
        "schema_version": "ac_eval_models_0.1",
        "status": status,
        "feature_names": list(_EVAL_FEATURE_NAMES),
        "ridge_alpha": ridge_alpha,
        "min_eval_rows": min_eval_rows,
        "min_eval_families": min_eval_families,
        "evals": evals,
        "eval_residuals": residuals,
        "warnings": warnings,
    }


def _range(values: Sequence[float]) -> Optional[Dict[str, float]]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return {"min": round(min(vals), 6), "max": round(max(vals), 6)}


def _calibration_domains(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    active = []
    total = []
    tokens = []
    context = []
    eval_names = set()
    for row in rows:
        active_b = _scale_to_billions(_numeric(row, "active_params_b"))
        total_b = _scale_to_billions(_numeric(row, "total_params_b"))
        tokens_t = _tokens_to_trillions(_numeric(row, "training_tokens"))
        ctx = _numeric(row, "context_length")
        if active_b is not None:
            active.append(active_b)
        if total_b is not None:
            total.append(total_b)
        if tokens_t is not None:
            tokens.append(tokens_t)
        if ctx is not None:
            context.append(ctx)
        eval_names.update(_collect_observed_evals(row).keys())
    return {
        "active_params_b": _range(active),
        "total_params_b": _range(total),
        "training_tokens_t": _range(tokens),
        "context_length": _range(context),
        "hardware": sorted({
            hw for hw in (
                _canonical_hw(_value(row, "hardware"), None)
                for row in rows
            )
            if hw
        }),
        "architecture_families": sorted({_family(row) for row in rows}),
        "evals": sorted(eval_names),
    }


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fit_status_and_warnings(
    *,
    quality: Dict[str, Any],
    hardware: Dict[str, Any],
    eval_models: Dict[str, Any],
    min_quality_rows: int,
    min_hardware_rows: int,
    max_hardware_scatter_p90_pct: float,
) -> Tuple[str, List[str]]:
    warnings = []
    if quality.get("n", 0) < min_quality_rows:
        warnings.append(
            f"Quality uncertainty fit used {quality.get('n', 0)} rows; recommended minimum is {min_quality_rows}."
        )
    for hw, fit in sorted(hardware.items()):
        for metric_key, label in (
            ("n_training_tps", "training TPS"),
            ("n_serving_tbt", "decode latency"),
            ("n_prefill", "prefill latency"),
        ):
            n = int(fit.get(metric_key, 0) or 0)
            if 0 < n < min_hardware_rows:
                warnings.append(
                    f"{hw} {label} calibration used {n} rows; recommended minimum is {min_hardware_rows}."
                )
        for scatter_key, label in (
            ("training_scatter_p90_pct", "training TPS"),
            ("serving_tbt_scatter_p90_pct", "decode latency"),
            ("prefill_scatter_p90_pct", "prefill latency"),
        ):
            scatter = fit.get(scatter_key)
            if scatter is not None and float(scatter) > max_hardware_scatter_p90_pct:
                warnings.append(
                    f"{hw} {label} post-fit P90 scatter is {float(scatter):.2f}%; "
                    f"target is <= {max_hardware_scatter_p90_pct:.2f}%."
                )
    if eval_models.get("status") == "experimental":
        warnings.append("Eval models are experimental; inspect held-out-family CV before using them for sign-off.")
    if eval_models.get("status") == "not_configured":
        warnings.append("No eval score rows were supplied; pack calibrates loss uncertainty and hardware only.")

    if warnings:
        return "experimental", warnings
    return "production_ready", []


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


def _quality_overlay(
    fit: Dict[str, Any],
    *,
    target_coverage: float,
    status: str,
    warnings: Sequence[str],
    domains: Dict[str, Any],
    eval_models: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
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
        },
        "lab_calibration": {
            "schema_version": "ac_lab_calibration_0.2",
            "status": status,
            "warnings": list(warnings),
            "domains": domains,
            "provenance": provenance,
        },
        "eval_models": eval_models,
    }


def _render_report(pack: Dict[str, Any]) -> str:
    q = pack["quality"]
    lines = [
        "# AC Auto-Calibration Report",
        "",
        f"Generated: {pack['generated_at']}",
        f"Fit status: **{pack['fit_status']}**",
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
            "| Hardware | Train Eff x | Decode Eff x | Prefill Eff x | Train scatter P90 | TBT scatter P90 | Prefill scatter P90 | Train n | TBT n | Prefill n |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for hw, fit in sorted(hardware.items()):
            lines.append(
                f"| {hw} | "
                f"{_fmt_optional(fit.get('training_efficiency_multiplier'))} | "
                f"{_fmt_optional(fit.get('decode_efficiency_multiplier'))} | "
                f"{_fmt_optional(fit.get('prefill_efficiency_multiplier'))} | "
                f"{_fmt_optional(fit.get('training_scatter_p90_pct'))} | "
                f"{_fmt_optional(fit.get('serving_tbt_scatter_p90_pct'))} | "
                f"{_fmt_optional(fit.get('prefill_scatter_p90_pct'))} | "
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
    eval_models = pack.get("eval_models", {})
    lines.extend(["", "## Eval Models", ""])
    if eval_models.get("status") == "not_configured":
        lines.append("No eval score rows were supplied.")
    else:
        lines.extend([
            f"Status: **{eval_models.get('status', 'unknown')}**",
            "",
            "| Eval | Rows | Families | Train RMSE | Held-out family RMSE | Status |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for name, fit in sorted(eval_models.get("evals", {}).items()):
            lines.append(
                f"| {name} | {fit.get('n', 0)} | "
                f"{len(fit.get('families', []))} | "
                f"{_fmt_optional(fit.get('train_rmse'))} | "
                f"{_fmt_optional(fit.get('heldout_family_rmse'))} | "
                f"{fit.get('status', 'unknown')} |"
            )
        residuals = eval_models.get("eval_residuals", {})
        if residuals:
            lines.extend([
                "",
                "Residual fits were also learned for evals that supplied both predicted and observed scores: "
                + ", ".join(sorted(residuals.keys())) + ".",
            ])
    domains = pack.get("calibration_domains", {})
    lines.extend(["", "## Calibration Domain", ""])
    for key in ("active_params_b", "total_params_b", "training_tokens_t", "context_length"):
        rng = domains.get(key)
        if rng:
            lines.append(f"- {key}: {rng['min']} to {rng['max']}")
    if domains.get("architecture_families"):
        lines.append("- architecture_families: " + ", ".join(domains["architecture_families"]))
    if domains.get("evals"):
        lines.append("- evals: " + ", ".join(domains["evals"]))
    warnings = pack.get("spec_output", {}).get("warnings", [])
    warnings = list(warnings) + list(pack.get("warnings", []))
    warnings.extend(eval_models.get("warnings", []))
    for fit in eval_models.get("evals", {}).values():
        warnings.extend(fit.get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for w in dict.fromkeys(warnings):
            lines.append(f"- {w}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Quality calibration scales uncertainty intervals; it does not bias-correct the predicted loss point estimate.",
        "- Hardware calibration adjusts system-efficiency constants from median observed/predicted ratios.",
        "- Eval models are ridge fits with held-out-family checks when multiple architecture families are present.",
        "- Keep separate packs per cluster topology, kernel stack, scheduler policy, and model family when those differ materially.",
    ])
    return "\n".join(lines) + "\n"


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def fit_command(args: argparse.Namespace) -> int:
    measurements_path = Path(args.measurements)
    rows = _read_measurements(measurements_path)
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
    eval_models = _fit_eval_models(
        rows,
        ridge_alpha=float(args.eval_ridge_alpha),
        min_eval_rows=int(args.min_eval_rows),
        min_eval_families=int(args.min_eval_families),
    )
    domains = _calibration_domains(rows)
    provenance = {
        "measurements_path": str(measurements_path),
        "measurements_sha256": _fingerprint(measurements_path),
        "rows_loaded": len(rows),
        "command": {
            "target_coverage": target_coverage,
            "eval_ridge_alpha": float(args.eval_ridge_alpha),
            "min_quality_rows": int(args.min_quality_rows),
            "min_eval_rows": int(args.min_eval_rows),
            "min_eval_families": int(args.min_eval_families),
            "min_hardware_rows": int(args.min_hardware_rows),
            "max_hardware_scatter_p90_pct": float(args.max_hardware_scatter_p90_pct),
        },
    }
    fit_status, fit_warnings = _fit_status_and_warnings(
        quality=quality,
        hardware=hardware,
        eval_models=eval_models,
        min_quality_rows=int(args.min_quality_rows),
        min_hardware_rows=int(args.min_hardware_rows),
        max_hardware_scatter_p90_pct=float(args.max_hardware_scatter_p90_pct),
    )
    base_spec_dir = Path(args.base_hardware_spec_dir or os.environ.get(
        "AC_HARDWARE_SPEC_DIR", _DEFAULT_SPEC_DIR))
    spec_output = _copy_and_calibrate_specs(
        base_spec_dir=base_spec_dir,
        out_spec_dir=out_dir / "hardware_specs",
        hardware_fits=hardware,
        min_efficiency=float(args.min_efficiency),
        max_efficiency=float(args.max_efficiency),
    )
    overlay = _quality_overlay(
        quality,
        target_coverage=target_coverage,
        status=fit_status,
        warnings=fit_warnings,
        domains=domains,
        eval_models=eval_models,
        provenance=provenance,
    )
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pack = {
        "schema_version": "ac_lab_calibration_0.2",
        "generated_at": generated_at,
        "fit_status": fit_status,
        "warnings": fit_warnings,
        "measurements_path": str(measurements_path),
        "measurements_sha256": provenance["measurements_sha256"],
        "rows_loaded": len(rows),
        "target_coverage": target_coverage,
        "quality": quality,
        "quality_overlay": overlay,
        "hardware": hardware,
        "eval_models": eval_models,
        "calibration_domains": domains,
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
    if eval_models.get("evals"):
        print(
            "Eval models: "
            + ", ".join(
                f"{name}:{fit.get('status', 'unknown')}"
                for name, fit in sorted(eval_models["evals"].items())
            )
        )
    if fit_warnings:
        print(f"Fit status: {fit_status} ({len(fit_warnings)} warning(s))")
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
    fit.add_argument("--min-quality-rows", type=int, default=12,
                     help="Recommended minimum loss rows before pack is production-ready")
    fit.add_argument("--eval-ridge-alpha", type=float, default=1.0,
                     help="Ridge regularization strength for eval score models")
    fit.add_argument("--min-eval-rows", type=int, default=12,
                     help="Recommended minimum rows per eval before the eval model is validated")
    fit.add_argument("--min-eval-families", type=int, default=3,
                     help="Recommended minimum architecture families per eval")
    fit.add_argument("--min-hardware-rows", type=int, default=3,
                     help="Recommended minimum rows per hardware metric")
    fit.add_argument("--max-hardware-scatter-p90-pct", type=float, default=15.0,
                     help="Maximum post-fit P90 hardware scatter before pack is marked experimental")
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
