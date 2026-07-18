"""Wave 47 helper: regenerate the non-h100 coarse shards (dense + moe,
max-candidates 150 / refine 8) that the shipped 20T multi-hw payload used,
into the same shards dir as regen20t_driver, resumably (skip existing).

Usage: python3 scripts/_regen20t_coarse_shards.py [--budget 34]
Env:   AC_REGEN20T_DIR (same as the driver)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

# Wave 47: pin hash randomization (see regen20t_driver.py) so sharded
# regens are bit-reproducible.
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
HW = ["b200", "tpu_v5p", "trainium2", "trainium3"]
PARAMS = [1.0, 3.0, 7.0, 13.0, 120.0, 500.0, 750.0, 1000.0]
WORK = os.environ.get(
    "AC_REGEN20T_DIR", os.path.join(os.path.dirname(_ROOT), ".regen20t"))
SHARD_DIR = os.path.join(WORK, "shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=34.0)
    args = ap.parse_args()
    os.makedirs(SHARD_DIR, exist_ok=True)
    t0 = time.time()
    todo = done = 0
    for hw in HW:
        for p in PARAMS:
            for mode in ("dense", "moe"):
                if mode == "moe" and p < G.MOE_MIN_PARAMS:
                    continue
                sid = f"{hw}_{p:g}_{mode}"
                path = os.path.join(SHARD_DIR, sid + ".json")
                todo += 1
                if os.path.exists(path):
                    done += 1
                    continue
                if time.time() - t0 > args.budget:
                    continue
                t1 = time.time()
                data = G.generate(
                    hardware=[hw], param_targets=[p], token_counts=[20.0],
                    arch_modes=G.build_arch_modes([mode], None),
                    contexts=CONTEXTS, allow_compressed=True,
                    max_candidates=150, local_refine_budget=8)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                os.replace(tmp, path)
                done += 1
                print(f"[shard] {sid}: {time.time()-t1:.1f}s "
                      f"rows={len(data.get('grid', []))}")
    print(f"PROGRESS {done}/{todo}")
    if done == todo:
        print("ALL-DONE")


if __name__ == "__main__":
    main()
