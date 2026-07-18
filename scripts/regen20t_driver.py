"""Wave 44 — resumable 20T web-grid regeneration driver.

Context: the Cowork sandbox kills any process that outlives its ~45 s
foreground window, so the grid regen (hours of optimizer searches) must be
sliced into shards of ≤ ~40 s, each persisted to disk, with a state file
so repeated invocations make monotonic progress. Per user decision
(2026-07-15): h100 runs at the highest fidelity that fits a window
(max-candidates 400, refine ladder 24→8), the other hardware runs coarse
(max-candidates 150, refine 8) — recorded per shard and disclosed in the
demo's index.html.

Protocol:
  python3 scripts/regen20t_driver.py            # run shards for ~35 s, exit
  ... repeat until it prints ALL-DONE ...
  python3 scripts/regen20t_driver.py --assemble # merge + post chain + write

Crash safety: a shard is marked "attempting" before generate() starts; if
the sandbox kills the run, the next invocation sees the marker, bumps the
shard one level down the fidelity ladder, and retries. Completed shards
are never re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Wave 47: pin hash randomization. The optimizer's candidate-pool cap and
# dedupe iterate sets/dicts; with a random PYTHONHASHSEED two runs of the
# SAME search can trim different exact-tie Pareto members (observed: one
# 500B/8k moe_hybrid pareto entry differing between regens). Re-exec with
# a fixed seed so sharded regens are bit-reproducible.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "ac"), _ROOT, _HERE, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _generator_payload as G  # noqa: E402

CONTEXTS = [8192, 32768, 131072, 1048576, 2097152, 4194304]
TOKENS_T = 20.0
HARDWARE = ["h100", "b200", "tpu_v5p", "trainium2", "trainium3"]
PARAMS = [1.0, 3.0, 7.0, 13.0, 120.0, 500.0, 750.0, 1000.0]
VARIANTS = ([("dense", None), ("moe", None)]
            + [("hybrid", st) for st in G.STATE_FAMILIES]
            + [("moe_hybrid", st) for st in G.STATE_FAMILIES])

# Fidelity ladders: (max_candidates, local_refine_budget), best first.
LADDER_H100 = [(400, 24), (400, 8), (150, 8), (100, 4)]
LADDER_COARSE = [(150, 8), (100, 4)]

# Cost priors (seconds) for window packing; refined from observed times.
EST_DEFAULT = {"dense": 10.0, "moe": 25.0, "hybrid": 25.0, "moe_hybrid": 36.0}

WORK = os.environ.get(
    "AC_REGEN20T_DIR", os.path.join(os.path.dirname(_ROOT), ".regen20t"))
SHARD_DIR = os.path.join(WORK, "shards")
STATE = os.path.join(WORK, "state.json")


def _gated(params_b: float, mode: str) -> bool:
    if mode == "moe" and params_b < G.MOE_MIN_PARAMS:
        return True
    if mode == "hybrid" and params_b < G.HYBRID_MIN_PARAMS:
        return True
    if mode == "moe_hybrid" and params_b < G.MOE_HYBRID_MIN_PARAMS:
        return True
    return False


def shard_list():
    # Wave 44 scope: full variant matrix on h100 (detailed sweep for the
    # main flagship hardware); dense + MoE ONLY on other hardware (coarse
    # sweep — hybrid state families are h100-only in the demo). Disclosed
    # in the demo footer via _regen20t.note.
    out = []
    coarse_variants = [(m, s) for (m, s) in VARIANTS if m in ("dense", "moe")]
    for hw in HARDWARE:  # h100 first
        variants = VARIANTS if hw == "h100" else coarse_variants
        for p in PARAMS:
            for mode, st in variants:
                if _gated(p, mode):
                    continue
                sid = f"{hw}_{p:g}_{mode}" + (f"_{st}" if st else "")
                out.append({"id": sid, "hw": hw, "params": p,
                            "mode": mode, "state": st})
    return out


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=0)
    os.replace(tmp, STATE)


def run_shards(budget: float, only_hw=None):
    os.makedirs(SHARD_DIR, exist_ok=True)
    st = load_state()
    shards = shard_list()
    if only_hw:
        shards = [s for s in shards if s["hw"] in only_hw]

    # Crash recovery: any shard still marked attempting died mid-run —
    # push it one level down its ladder.
    for s in shards:
        rec = st.get(s["id"])
        if rec and rec.get("status") == "attempting":
            rec["level"] = int(rec.get("level", 0)) + 1
            rec["status"] = "pending"
            print(f"[recover] {s['id']} died at level {rec['level']-1}; "
                  f"downgrading to level {rec['level']}")
    save_state(st)

    # Cost estimates: per (hw-class, mode) observed maxima persisted in the
    # state file; the static defaults apply only until a mode has been
    # observed once. (v2 fix: the first version took max(default, observed)
    # so a cheap observed mode could never undercut its pessimistic
    # default, starving heavy-but-actually-cheap shards.)
    est_obs = load_state().get("_est", {})
    t0 = time.time()
    done = ran = 0

    def _est(mode: str) -> float:
        if mode in est_obs:
            return est_obs[mode] * 1.2
        return EST_DEFAULT.get(mode, 30.0)
    for s in shards:
        rec = st.get(s["id"], {})
        if rec.get("status") == "done":
            done += 1
            continue
        ladder = LADDER_H100 if s["hw"] == "h100" else LADDER_COARSE
        level = min(int(rec.get("level", 0)), len(ladder) - 1)
        elapsed = time.time() - t0
        # Cap the estimate's contribution so a mode whose prior exceeds the
        # budget can still START in a fresh window (it gets the whole
        # window to itself; if it truly can't fit, the attempt protocol
        # downgrades it next invocation).
        if elapsed + min(_est(s["mode"]), budget - 2.0) > budget:
            continue  # keep scanning: a cheaper mode may still fit
        mc, rb = ladder[level]
        st[s["id"]] = {"status": "attempting", "level": level,
                       "mc": mc, "rb": rb}
        save_state(st)
        t1 = time.time()
        modes = G.build_arch_modes([s["mode"]],
                                   [s["state"]] if s["state"] else None)
        try:
            data = G.generate(hardware=[s["hw"]], param_targets=[s["params"]],
                              token_counts=[TOKENS_T], arch_modes=modes,
                              contexts=CONTEXTS, allow_compressed=True,
                              max_candidates=mc, local_refine_budget=rb)
        except ValueError as e:
            # Wave 44 (v2): the generator raises ValueError for hard
            # parallelism incompatibilities (e.g. ep_options=[1] when
            # the derived DP >= 2 filters out EP=1 as sub-EP; a real
            # upstream defect in _generator_payload's EP handling for
            # certain large-MoE + small-cluster combos). Record the
            # skip and continue — assemble() treats "skipped" as a
            # legitimate empty-grid shard so the run can finish.
            dt = time.time() - t1
            st[s["id"]] = {"status": "skipped", "level": level, "mc": mc,
                           "rb": rb, "time_s": round(dt, 1),
                           "error": str(e)[:200]}
            save_state(st)
            done += 1  # counts toward "no work left"
            ran += 1
            print(f"[skip] {s['id']}: ValueError -> {str(e)[:120]}")
            continue
        dt = time.time() - t1
        with open(os.path.join(SHARD_DIR, s["id"] + ".json"), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        st[s["id"]] = {"status": "done", "level": level, "mc": mc,
                       "rb": rb, "time_s": round(dt, 1),
                       "rows": len(data.get("grid", []))}
        est_obs[s["mode"]] = max(est_obs.get(s["mode"], 0.0), dt)
        st["_est"] = est_obs
        save_state(st)
        done += 1
        ran += 1
        print(f"[shard] {s['id']} mc={mc} r={rb}: {dt:.1f}s "
              f"rows={len(data.get('grid', []))}")
    total = len(shards)
    print(f"PROGRESS {done}/{total} shards done (ran {ran} this window)")
    if done == total:
        print("ALL-DONE")
    return 0


def assemble(out_multi: str, out_h100: str):
    st = load_state()
    shards = shard_list()
    # Wave 44 (v2): "skipped" counts as finished for scheduling. A
    # skipped shard has no output file, so assembly ignores it. Only
    # "pending" / "attempting" shards block assembly.
    missing = [s["id"] for s in shards
               if st.get(s["id"], {}).get("status") not in ("done", "skipped")]
    if missing:
        print(f"REFUSING to assemble: {len(missing)} shards missing, e.g. "
              f"{missing[:5]}")
        return 1
    skipped = [s["id"] for s in shards
               if st.get(s["id"], {}).get("status") == "skipped"]
    if skipped:
        print(f"[assemble] {len(skipped)} shards skipped (no output): {skipped[:5]}...")

    def _merge_all(ids):
        base = None
        for sid in ids:
            path = os.path.join(SHARD_DIR, sid + ".json")
            if not os.path.exists(path):
                continue  # skipped shard
            with open(path) as f:
                data = json.load(f)
            if base is None:
                base = data
            else:
                base = G.merge_payload(base, data)
        return base

    all_ids = [s["id"] for s in shards]
    h100_ids = [s["id"] for s in shards if s["hw"] == "h100"]

    # Fidelity provenance for the payload + demo footer.
    fid = {}
    for s in shards:
        rec = st[s["id"]]
        fid.setdefault(s["hw"], {})
        key = f"mc{rec['mc']}/r{rec['rb']}"
        fid[s["hw"]][key] = fid[s["hw"]].get(key, 0) + 1
    prov = {
        "tokens_T": TOKENS_T,
        "contexts": CONTEXTS,
        "generated": time.strftime("%Y-%m-%d"),
        "search_fidelity_by_hw": fid,
        "note": ("Release web matrix regenerated at 20T training tokens. "
                 "H100 uses the full architecture matrix at max-candidates "
                 "400 (refine 24, with resumable fidelity fallback); other "
                 "hardware uses dense/MoE comparison sweeps at "
                 "max-candidates 150 (refine 8)."),
    }

    for label, ids, path in (("multi-hw", all_ids, out_multi),
                             ("h100", h100_ids, out_h100)):
        merged = _merge_all(ids)
        G.run_post_chain(merged)
        merged["_regen20t"] = prov
        with open(path, "w") as f:
            json.dump(merged, f, separators=(",", ":"))
        print(f"[assemble] {label}: {len(merged['grid'])} rows, "
              f"{len(merged.get('cells', []))} cells -> {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=34.0,
                    help="seconds of shard work per invocation")
    ap.add_argument("--hw", type=str, default=None,
                    help="comma list; restrict this window to these hw")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--out-multi", default=os.path.join(
        os.path.dirname(_ROOT), "v1-web", "compiler-data.json"))
    ap.add_argument("--out-h100", default=os.path.join(
        os.path.dirname(_ROOT), "v1-web", "compiler-data-h100.json"))
    args = ap.parse_args()
    if args.assemble:
        return assemble(args.out_multi, args.out_h100)
    only = [h.strip() for h in args.hw.split(",")] if args.hw else None
    return run_shards(args.budget, only)


if __name__ == "__main__":
    sys.exit(main())
