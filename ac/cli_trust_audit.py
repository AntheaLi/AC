"""Wave 18e — trust audit CLI (`ac-trust-audit`).

Usage:
  ac-trust-audit --public-anchors [--tightened] [--json OUT.json] [--md OUT.md]
  ac-trust-audit --frontier-feasibility [--hardware h100]
  ac-trust-audit --all

Exit code is nonzero when `block_publication == True`.

This module is the package-registered entry point (see `pyproject.toml
[project.scripts]`). `scripts/run_trust_audit.py` is preserved as a thin
compat shim that re-exports `main` from here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from ac.trust_audit import (
    run_frontier_feasibility_suite,
    run_public_anchor_suite,
    render_public_anchor_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ac-trust-audit",
        description=(
            "Wave 18e trust audit: public-model predictive accuracy + "
            "1M/2M physical feasibility anchors. Exit code is nonzero "
            "when the audit reports block_publication."
        ),
    )
    p.add_argument("--public-anchors", action="store_true",
                   help="Run the public-model predictive-accuracy audit.")
    p.add_argument("--frontier-feasibility", action="store_true",
                   help="Run the 1M/2M physical anchor suite.")
    p.add_argument("--all", action="store_true",
                   help="Run every audit stage.")
    p.add_argument("--tightened", action="store_true",
                   help="Use post-calibration tightened tolerances "
                        "(Wave 19 release gate).")
    p.add_argument("--hardware", default="h100",
                   help="Hardware target for the audit (default: h100).")
    p.add_argument("--out", metavar="DIR", default=None,
                   help="Output directory; writes audit.json + report.md "
                        "with standard names (matches every other "
                        "subcommand). --json/--md override individually.")
    p.add_argument("--json", metavar="PATH", default=None,
                   help="Write machine-readable audit report to this path.")
    p.add_argument("--md", metavar="PATH", default=None,
                   help="Write the public-anchor markdown report to this path.")
    p.add_argument("--registry", metavar="PATH", default=None,
                   help="Override public-anchor registry JSON path (default: "
                        "tests/fixtures/public_model_anchors_v1.json).")
    p.add_argument("--override", metavar="ANCHOR:JUSTIFICATION",
                   action="append", default=[],
                   help="Record an override justification for a failing anchor "
                        "(non-blocking). Repeatable.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Wave 28: fail fast on a broken AC_QUALITY_DEFAULTS /
    # AC_HARDWARE_SPEC_DIR — otherwise every anchor evaluation errors
    # individually and the audit reads like a model-accuracy failure.
    try:
        from ac.quality_model import validate_calibration_environment
    except ImportError:
        from quality_model import validate_calibration_environment
    try:
        validate_calibration_environment(getattr(args, "hardware", None))
    except Exception as e:
        print(f"ERROR: invalid calibration environment: {e}", file=sys.stderr)
        return 2

    # Wave 20 (feedback #6): `--out DIR` with no stage flag implies --all.
    # The README always documented `ac-trust-audit --out DIR` as "regenerate
    # everything"; make the CLI match the documented intent instead of
    # erroring.
    if args.out and not (args.public_anchors or args.frontier_feasibility
                         or args.all):
        args.all = True

    if not (args.public_anchors or args.frontier_feasibility or args.all):
        parser.error(
            "pick at least one of --public-anchors / "
            "--frontier-feasibility / --all (or pass --out DIR, "
            "which implies --all)")

    reports = {}
    block = False

    if args.public_anchors or args.all:
        justifications = {}
        for spec in args.override:
            if ":" not in spec:
                parser.error(
                    f"malformed override {spec!r}; expected "
                    f"ANCHOR:JUSTIFICATION")
            k, v = spec.split(":", 1)
            justifications[k.strip()] = v.strip()
        r = run_public_anchor_suite(
            registry_path=Path(args.registry) if args.registry else None,
            justifications=justifications,
            tightened=args.tightened,
            hardware_override=args.hardware,
        )
        reports["public_anchors"] = r
        print(render_public_anchor_markdown(r))
        if r["block_publication"]:
            block = True

    if args.frontier_feasibility or args.all:
        r = run_frontier_feasibility_suite(hardware=args.hardware)
        reports["frontier_feasibility"] = r
        print(f"\nFrontier feasibility: {r['n_pass']}/{r['n_total']} anchors pass.")
        for a in r["anchors"]:
            mark = "✓" if a["physically_feasible"] else "✗"
            print(f"  {mark} ctx={a['context_length']} {a['scenario']}"
                  + (f" ({a['reason']})" if a["reason"] else ""))
        if r["block_publication"]:
            block = True

    # Wave 19 (P2): --out DIR convenience, standard names.
    if getattr(args, "out", None):
        os.makedirs(args.out, exist_ok=True)
        if not args.json:
            args.json = os.path.join(args.out, "audit.json")
        if not args.md:
            args.md = os.path.join(args.out, "report.md")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"\nWrote {args.json}", file=sys.stderr)
    if args.md and "public_anchors" in reports:
        with open(args.md, "w") as f:
            f.write(render_public_anchor_markdown(reports["public_anchors"]))
        print(f"Wrote {args.md}", file=sys.stderr)

    # Wave 19 (P1-5): regenerate the per-family loss-bias table alongside
    # the audit. This is the fixture decision.assess_decision uses to raise
    # the winner bar for cross-family comparisons pre-calibration; labs that
    # extend the anchor registry should re-run with --out to refresh it.
    if getattr(args, "out", None) and "public_anchors" in reports:
        try:
            fb = _family_bias_from_audit(reports["public_anchors"],
                                         registry_path=args.registry)
            fb_path = os.path.join(args.out, "family_bias.json")
            with open(fb_path, "w") as f:
                json.dump(fb, f, indent=2)
            print(f"Wrote {fb_path} (copy over "
                  "ac/calibration/family_bias_v1.json to refresh the "
                  "decision layer's cross-family bar)", file=sys.stderr)
        except Exception as exc:  # never block the audit on the extra table
            print(f"family-bias table skipped: {exc}", file=sys.stderr)

    return 1 if block else 0




# Metrics the family-bias table aggregates. Wave 20 (feedback #1): the
# wave-19 table recorded LOSS bias only, while serving_cost / latency
# profiles and every "N× faster decode" justification leaned on TBT/TTFT
# predictions whose anchor errors spanned -94%…+93% on MoE at the time
# (Wave 25 narrowed this to roughly −55%…+40% by fixing the decode
# expert_load efficiency double-count; the current span lives in
# family_bias_v1.json) — model error with no caveat. Aggregate every
# anchor-audited metric so the decision layer can floor them all.
FAMILY_BIAS_METRICS = ("loss", "tbt_ms", "ttft_ms", "mem_gb")


def _family_bias_from_audit(public_anchors: dict,
                            registry_path=None) -> dict:
    """Per-family mean signed error from an anchor run, per metric.

    Wave 19 (P1-5) introduced the loss-only table; Wave 20 generalizes it
    to every audited metric (schema v2). The v1 loss keys are preserved at
    the family level so older readers keep working.
    """
    import statistics
    from pathlib import Path as _Path
    reg_path = registry_path or str(
        _Path(__file__).resolve().parent.parent
        / "tests" / "fixtures" / "public_model_anchors_v1.json")
    with open(reg_path) as f:
        reg = json.load(f)
    fam_of = {e["id"]: (e.get("arch") or {}).get("family", "dense")
              for e in reg.get("entries", [])}
    # by_family[fam][metric] -> [(anchor_id, signed_err_pct), ...]
    by_family: dict = {}
    for a in public_anchors.get("anchors", []):
        if a.get("status") in ("skipped", "unrepresentable", "error"):
            continue
        fam = fam_of.get(a.get("id"), "dense")
        for m in a.get("metrics", []):
            name = m.get("metric")
            if name not in FAMILY_BIAS_METRICS or not m.get("published"):
                continue
            if m.get("rel_err") is None:
                continue
            by_family.setdefault(fam, {}).setdefault(name, []).append(
                (a.get("id"), m["rel_err"] * 100.0))
    out = {"schema_version": "wave20.family_bias.v2",
           "source": "ac-trust-audit --public-anchors",
           "families": {}}

    def _stats(rows):
        errs = [e for _, e in rows]
        return {
            "mean_signed_err_pct": round(statistics.mean(errs), 2),
            "stdev_err_pct": round(
                statistics.stdev(errs) if len(errs) > 1
                else abs(errs[0]) * 0.5, 2),
            "n": len(errs),
            "anchors": {i: round(e, 2) for i, e in rows},
        }

    for fam, per_metric in by_family.items():
        entry: dict = {"metrics": {}}
        for metric, rows in per_metric.items():
            entry["metrics"][metric] = _stats(rows)
        # v1 compatibility keys (loss only), consumed by older readers.
        loss = entry["metrics"].get("loss")
        if loss:
            entry["mean_signed_loss_err_pct"] = loss["mean_signed_err_pct"]
            entry["stdev_loss_err_pct"] = loss["stdev_err_pct"]
            entry["n_anchors"] = loss["n"]
            entry["anchors"] = loss["anchors"]
        out["families"][fam] = entry
    return out


if __name__ == "__main__":
    sys.exit(main())
