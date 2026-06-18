# AC Auto-Calibration Report

Generated: 2026-06-18T15:27:34+00:00

## Quality Uncertainty

- Rows used: 2
- Target coverage: 0.90
- Coverage before: 100.00%
- Coverage after: 100.00%
- Calibration multiplier: 1.000
- Median relative loss bias: +2.880%
- P90 absolute relative loss error: 3.213%

Use `quality_overrides.json` via:

```bash
AC_QUALITY_DEFAULTS=/path/to/calibration_pack/quality_overrides.json ac-compile ...
```

## Hardware Efficiency

| Hardware | Train Eff x | Decode Eff x | Prefill Eff x | Train n | TBT n | Prefill n |
|---|---:|---:|---:|---:|---:|---:|
| h100 | 0.869 | 0.868 | 0.876 | 2 | 2 | 2 |

Use calibrated hardware specs via:

```bash
AC_HARDWARE_SPEC_DIR=/path/to/calibration_pack/hardware_specs ac-compile ...
```

## Notes

- Quality calibration scales uncertainty intervals; it does not bias-correct the predicted loss point estimate.
- Hardware calibration adjusts system-efficiency constants from median observed/predicted ratios.
- Keep separate packs per cluster topology, kernel stack, scheduler policy, and model family when those differ materially.
