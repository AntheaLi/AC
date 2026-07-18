"""Wave 15 — Pillar B: regenerate the golden decision-matrix fixture.

Produces `tests/fixtures/golden_h100_decision_matrix.json`. The committed
fixture is the regression contract; running this script overwrites it,
which should ONLY be done when an intentional change moves cells (e.g.
constant edit, calibration update). The test
`tests/test_golden_matrix.py::test_decision_matrix_unchanged` will fail
loudly on any unexplained drift.

The matrix is intentionally small (32 cells: 4 sizes × 4 contexts × 2
serving modes) so the regen completes in ~30s and human review of the
diff stays tractable.

Usage:
    python scripts/regen_golden_matrix.py [--accept-drift "reason"]

When `--accept-drift` is provided, the script writes the new fixture and
also records the reason in a sidecar `.regen-history.txt` so the
provenance of every cell-moving update is auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Golden matrix scope: small, sharp, H100-only until Wave 10 (hw-blind quality)
# lands. The 32 cells span the spec's enumerated dimensions.
GOLDEN_HW = "h100"
GOLDEN_SIZES_B = [1.0, 7.0, 70.0, 500.0]
GOLDEN_CTXS = [32768, 131072, 1048576, 2097152]
GOLDEN_SERVING_MODES = [
    {"name": "short_serving",  "tbt": None, "ttft": None, "batch": 8},
    {"name": "long_serving",   "tbt": None, "ttft": None, "batch": 4},
]

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "golden_h100_decision_matrix.json"
HISTORY_PATH = ROOT / "tests" / "fixtures" / "golden_h100_decision_matrix.regen-history.txt"

# Eval budget per cell — tight enough to finish in <1s, loose enough to
# admit the picked optimum reliably. The full 32-cell regen completes in
# <30s on a 2025-era laptop with these caps; larger cells (500B + state)
# are explicitly downscoped via the allow_state=False rule below.
MAX_CANDIDATES = 50
MAX_FULL_EVALUATIONS = 20


def _family_of(arch) -> str:
    """Wave 18a: route through the canonical ArchitectureSignature so the
    golden-matrix regen and the optimizer/report/generator can no longer
    disagree about family labels."""
    try:
        from ac.architecture import architecture_signature
        return architecture_signature(arch).legacy_family
    except (ValueError, ImportError):
        has_moe = bool(getattr(arch, "moe", None))
        has_state = bool(getattr(arch, "state_config", None)) and \
                    getattr(arch, "n_state_layers", 0) > 0
        if has_moe and has_state: return "moe_hybrid"
        if has_moe:               return "moe"
        if has_state:             return "hybrid"
        return "dense"


def _cell_record(target_b, ctx, smode):
    """Run the optimizer for one cell and serialize the contract fields.

    The fixture stores only the fields the regression test cares about:
      - picked family
      - rounded loss, TBT, memory
      - shape (d_model, n_layers, tp_degree)
    Auxiliary fields (training_tps, prefill, full Pareto) are not stored
    because they're more noise-prone and would inflate the diff."""
    from ac.optimizer import DeploymentConstraints, optimize

    c = DeploymentConstraints(
        target_params_b=float(target_b),
        training_tokens=int(2e12),
        context_length=int(ctx),
        serving_tbt_ms=smode["tbt"],
        serving_ttft_ms=smode["ttft"],
        serving_batch=int(smode["batch"]),
        tp=1, pp=1, dp=1,
        allow_moe=True,
        allow_state=True if target_b <= 100.0 else False,  # state only viable at small/mid
        max_candidates=MAX_CANDIDATES,
        max_full_evaluations=MAX_FULL_EVALUATIONS,
        allow_quality_sentinel=True,
        param_tolerance=0.15,
    )

    t0 = time.time()
    r = optimize(GOLDEN_HW, c)
    elapsed = time.time() - t0

    rec = {
        "size_b": float(target_b),
        "ctx": int(ctx),
        "serving": smode["name"],
        "elapsed_s": round(elapsed, 2),
    }
    if r.optimal is None:
        rec["family"] = None
        rec["picked"] = None
        rec["note"] = "no_feasible_solution"
        return rec

    arch = r.optimal.arch
    rec["family"] = _family_of(arch)
    rec["picked"] = {
        "d_model": int(arch.d_model),
        "n_layers": int(arch.n_layers),
        "tp_degree": int(getattr(arch, "tp_degree", 1)),
        "loss": round(float(r.optimal.predicted_loss), 4),
        "tbt_ms": round(float(r.optimal.serving_tbt_ms), 2),
        "mem_gb": round(float(r.optimal.memory_per_gpu_gb), 2),
    }
    return rec


PARTIAL_PATH = ROOT / "tests" / "fixtures" / ".golden_h100_decision_matrix.partial.json"


def _cell_key(size_b, ctx, serving_name):
    return f"{size_b}|{ctx}|{serving_name}"


def regen(resume: bool = True) -> dict:
    """Walk the 32-cell grid and produce a fixture-shaped dict.

    Resumable: writes a partial fixture after every cell so that the
    sandboxed 45s bash budget can complete the regen across multiple
    invocations. Each invocation skips cells already present in the
    partial file. When all 32 cells are done, the partial is promoted
    to the committed fixture path."""
    cells_by_key = {}
    if resume and PARTIAL_PATH.exists():
        try:
            with open(PARTIAL_PATH) as f:
                stash = json.load(f)
            for rec in stash.get("cells", []):
                k = _cell_key(rec["size_b"], rec["ctx"], rec["serving"])
                cells_by_key[k] = rec
            print(f"Resuming: {len(cells_by_key)} cells already in partial.",
                  file=sys.stderr)
        except Exception:
            pass

    total = len(GOLDEN_SIZES_B) * len(GOLDEN_CTXS) * len(GOLDEN_SERVING_MODES)
    done = len(cells_by_key)
    PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for size_b in GOLDEN_SIZES_B:
        for ctx in GOLDEN_CTXS:
            for smode in GOLDEN_SERVING_MODES:
                k = _cell_key(size_b, ctx, smode["name"])
                if k in cells_by_key:
                    continue
                rec = _cell_record(size_b, ctx, smode)
                cells_by_key[k] = rec
                done += 1
                fam = rec.get("family") or "—"
                print(f"  [{done}/{total}] {int(size_b)}B ctx={ctx} "
                      f"{smode['name']}: family={fam} "
                      f"({rec.get('elapsed_s', 0):.1f}s)",
                      file=sys.stderr)
                # Stash after every cell so a SIGTERM mid-run still
                # preserves progress.
                with open(PARTIAL_PATH, "w") as f:
                    json.dump({"cells": list(cells_by_key.values())}, f)

    # Re-emit in deterministic order so the committed fixture diff is stable.
    cells = []
    for size_b in GOLDEN_SIZES_B:
        for ctx in GOLDEN_CTXS:
            for smode in GOLDEN_SERVING_MODES:
                k = _cell_key(size_b, ctx, smode["name"])
                if k in cells_by_key:
                    cells.append(cells_by_key[k])
    return {
        "schema_version": "wave15.golden.v1",
        "hardware": GOLDEN_HW,
        "training_tokens": int(2e12),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_candidates": MAX_CANDIDATES,
        "max_full_evaluations": MAX_FULL_EVALUATIONS,
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accept-drift",
        type=str,
        default=None,
        help="One-line justification for accepting that this regen will "
             "move cells. Recorded in the sidecar history file.",
    )
    args = parser.parse_args()

    print(f"Regenerating golden matrix → {FIXTURE_PATH.relative_to(ROOT)}",
          file=sys.stderr)
    new = regen()

    # If the fixture exists, surface a diff summary so the human reviewer
    # can spot-check before committing.
    old = None
    if FIXTURE_PATH.exists():
        try:
            with open(FIXTURE_PATH) as f:
                old = json.load(f)
        except Exception:
            old = None

    moved = 0
    if old is not None and isinstance(old.get("cells"), list):
        old_by_key = {(c["size_b"], c["ctx"], c["serving"]): c
                      for c in old["cells"]}
        for c in new["cells"]:
            k = (c["size_b"], c["ctx"], c["serving"])
            o = old_by_key.get(k)
            if o is None:
                continue
            if (o.get("family") != c.get("family")
                    or (o.get("picked") or {}).get("d_model")
                        != (c.get("picked") or {}).get("d_model")
                    or (o.get("picked") or {}).get("n_layers")
                        != (c.get("picked") or {}).get("n_layers")):
                moved += 1
                print(
                    f"  [drift] {int(c['size_b'])}B ctx={c['ctx']} "
                    f"{c['serving']}: family {o.get('family')}→{c.get('family')}",
                    file=sys.stderr,
                )

    if moved > 0 and not args.accept_drift:
        print(
            f"\nABORT: {moved} cells moved relative to committed fixture. "
            "Provide --accept-drift \"reason\" to overwrite. The test "
            "tests/test_golden_matrix.py will continue to fail until the "
            "fixture is re-committed.",
            file=sys.stderr,
        )
        return 1

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(new, f, indent=2)
    print(f"Wrote {FIXTURE_PATH}", file=sys.stderr)

    if args.accept_drift:
        with open(HISTORY_PATH, "a") as f:
            f.write(
                f"{new['generated_at_utc']}\t{moved} cells moved\t{args.accept_drift}\n"
            )
        print(f"Recorded drift acceptance → {HISTORY_PATH}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
