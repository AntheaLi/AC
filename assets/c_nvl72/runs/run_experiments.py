#!/usr/bin/env python3
"""Gate-2 Task C — official experiment runner (V1–V3 analysis artifacts).

Header (required on every validation artifact):
  ac_version: 0.4.0
  quality_model_version: effective_capacity_v2
  git_commit: c170cda
  experiment_date: 2026-07-17
  agent wave: "gate2-wave1"

Reads prereg.md + prereg_amendment1.md + prereg_amendment2.md (protocol).
Writes raw JSON artifacts to validation/c_nvl72/runs/. Does not modify
any package code or existing files.

Parts:
  M  mechanism table — fixed 1.08T-total K2-class MoE arch, EP sweep on
     h100 vs gb200_nvl72 (V1 b/c, V2 b/c core numbers + stress axes)
  S  search-level evidence — in-process optimize() on both targets with
     the frozen protocol (allow_mla, batch 64); all_evaluated dumps,
     frontier EP composition, same-shape EP-controlled TBT contrasts
     (V1 a, V2 a)
  V3 domain sensitivity — fixed-arch EP sweep with nvlink_domain_size
     overridden to 8/16/32/72 (monotonicity)
  V2b true-K2 384-expert search on gb200_nvl72 (EP=72 must be absent)
"""

import copy
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
RUNS = os.path.dirname(os.path.abspath(__file__))

HEADER = {
    "ac_version": "0.4.0",
    "quality_model_version": "effective_capacity_v2",
    "git_commit": "c170cda",
    "experiment_date": "2026-07-17",
    "agent_wave": "gate2-wave1",
}

from ac.throughput_model import ArchConfig as TArch, throughput  # noqa: E402
from ac.stress import Workload, compute_throughput_stress  # noqa: E402
from ac.optimizer import DeploymentConstraints, optimize  # noqa: E402


def _k2_class_arch(batch=64, kv_precision="int8"):
    """1.08T-total / 45.7B-active 288-expert top-8 MoE (K2-class ratio),
    GQA-8 attention, fp8 weights."""
    moe = {"n_experts": 288, "top_k": 8, "expert_dim": 2816,
           "shared_expert": {"ffn_dim": 2048, "precision": "fp8"}}
    return TArch(d_model=7168, n_layers=61, n_heads=64, d_head=128,
                 n_kv_heads=8, ffn_dim=18944, vocab_size=32000,
                 batch_size=batch, seq_len=8192, precision="fp8",
                 weight_precision="fp8", activation_precision="fp8",
                 kv_precision=kv_precision,
                 moe_config=moe)


def _row(hw, ep, arch):
    r = throughput(arch, hw, tp_degree=1, pp_degree=1, dp_degree=2304,
                   ep_degree=ep)
    lb = r.decode_layer_breakdown
    a2a = getattr(lb, "alltoall_s", 0.0) if lb else 0.0
    tot = lb.total_s if lb else 0.0
    return {
        "hardware": hw, "ep": ep, "batch": arch.batch_size,
        "decode_tbt_ms": r.decode_time_per_token_ms,
        "train_tps": r.training_throughput_tokens_per_sec,
        "mem_per_gpu_gb": r.memory_footprint_per_gpu_gb,
        "spill_tier": r.spill_tier,
        "decode_a2a_share": a2a / max(tot, 1e-12),
    }


def part_mechanism():
    out = dict(HEADER)
    out["part"] = "M_mechanism_table"
    out["arch"] = {
        "d_model": 7168, "n_layers": 61, "n_experts": 288, "top_k": 8,
        "expert_dim": 2816, "shared_dim": 2048, "attention": "gqa-8",
        "weight_precision": "fp8", "kv_precision": "int8",
        "total_params_b": 1080, "active_params_b": 45.7,
    }
    rows = []
    stress_rows = []
    for batch in (32, 64):
        arch = _k2_class_arch(batch=batch)
        for hw in ("h100", "gb200_nvl72"):
            for ep in (8, 16, 32, 72):
                rows.append(_row(hw, ep, arch))
    # Stress vectors (decode + training phase) at the contrast endpoints.
    arch = _k2_class_arch(batch=64)
    for phase in ("decode", "training"):
        for hw in ("h100", "gb200_nvl72"):
            for ep in (8, 72):
                wl = Workload(batch_size=64, prefill_seq_len=8192,
                              decode_kv_len=8192, training_seq_len=8192,
                              phase=phase)
                sv = compute_throughput_stress(
                    arch, hw, wl, tp_degree=1, pp_degree=1, dp_degree=2304,
                    ep_degree=ep)
                stress_rows.append({
                    "phase": phase, "hardware": hw, "ep": ep,
                    "all_to_all": sv.all_to_all,
                    "all_reduce": sv.all_reduce,
                    "hbm_capacity": sv.hbm_capacity,
                    "hbm_bw_decode": sv.hbm_bw_decode,
                })
    out["ep_sweep"] = rows
    out["stress_axes"] = stress_rows
    path = os.path.join(RUNS, "mechanism_table.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path, rows, stress_rows


def _constraints(**kw):
    base = dict(
        target_params_b=32.0, training_tokens=int(15e12),
        context_length=8192, tp=1, pp=1, dp=1,
        serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=64,
        allow_moe=True, allow_mla=True, max_total_params_b=1200,
        moe_n_experts_options=[288], moe_top_k_options=[8],
        moe_granularity_targets=[1.0, 0.5],
        ep_options=[8, 16, 32, 72],
        training_cluster_gpus=2304,
        max_candidates=200)
    base.update(kw)
    return DeploymentConstraints(**base)


def _dump_search(hw, fname):
    r = optimize(hw, _constraints())
    rows = []
    for ev in r.all_evaluated:
        a = ev.arch
        if not getattr(a, "moe", None):
            continue
        rows.append({
            "ep": int(getattr(a, "ep_degree", 1) or 1),
            "d_model": a.d_model, "n_layers": a.n_layers,
            "n_experts": a.moe.get("n_experts"),
            "top_k": a.moe.get("top_k"),
            "expert_dim": a.moe.get("expert_dim"),
            "attention_type": getattr(a, "attention_type", "full"),
            "weight_precision": getattr(a, "weight_precision", "bf16"),
            "total_params_b": a.total_params / 1e9,
            "predicted_loss": ev.predicted_loss,
            "serving_tbt_ms": ev.serving_tbt_ms,
            "training_tps": ev.training_tps,
            "memory_per_gpu_gb": ev.memory_per_gpu_gb,
            "meets_constraints": ev.meets_constraints,
            "on_pareto_frontier": any(x is ev for x in r.pareto_frontier),
        })
    frontier_eps = sorted({row["ep"] for row in rows
                           if row["on_pareto_frontier"]})
    # Same-shape controlled contrast: group by shape key, compare TBT
    # across EP within identical (d_model, layers, expert_dim, precision).
    groups = {}
    for row in rows:
        key = (row["d_model"], row["n_layers"], row["expert_dim"],
               row["weight_precision"], row["attention_type"])
        groups.setdefault(key, []).append(row)
    contrasts = []
    for key, grp in sorted(groups.items()):
        eps = {g["ep"]: g for g in grp}
        if len(eps) >= 3:
            contrasts.append({
                "shape": {"d_model": key[0], "n_layers": key[1],
                          "expert_dim": key[2], "weight_precision": key[3],
                          "attention_type": key[4]},
                "by_ep": {str(ep): {"tbt_ms": g["serving_tbt_ms"],
                                    "loss": g["predicted_loss"],
                                    "mem_gb": g["memory_per_gpu_gb"]}
                          for ep, g in sorted(eps.items())},
            })
    best = {}
    for row in rows:
        if not row["meets_constraints"]:
            continue
        ep = row["ep"]
        if ep not in best or row["serving_tbt_ms"] < best[ep]["serving_tbt_ms"]:
            best[ep] = row
    out = dict(HEADER)
    out.update({
        "part": "S_search", "hardware": hw,
        "candidates_evaluated": r.candidates_evaluated,
        "moe_rows": len(rows),
        "evaluated_eps": sorted({row["ep"] for row in rows}),
        "frontier_eps": frontier_eps,
        "best_feasible_tbt_by_ep": {str(k): v for k, v in sorted(best.items())},
        "same_shape_ep_contrasts": contrasts,
        "rows": rows,
    })
    path = os.path.join(RUNS, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path, out


def part_v3():
    """Fixed-arch EP sweep with nvlink_domain_size overridden 8/16/32/72
    via AC_HARDWARE_SPEC_DIR spec copies (no package edits)."""
    import importlib
    import ac.throughput_model as tm

    src = os.path.join(REPO, "ac", "hardware_specs")
    tmp = tempfile.mkdtemp(prefix="ac_v3_specs_")
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(tmp, f))
    base_spec = json.load(open(os.path.join(src, "gb200_nvl72.json")))

    table = []
    for dom in (8, 16, 32, 72):
        d = copy.deepcopy(base_spec)
        d["system"]["nvlink_domain_size"] = dom
        d["nvlink_domain_size"] = dom
        d["gpus_per_node"] = dom
        d["accelerator_name"] = f"gb200-domain-{dom}"
        with open(os.path.join(tmp, "gb200_nvl72.json"), "w") as f:
            json.dump(d, f)
        os.environ["AC_HARDWARE_SPEC_DIR"] = tmp
        importlib.reload(tm)
        arch = _k2_class_arch(batch=64)
        for ep in (8, 16, 32, 72):
            r = tm.throughput(arch, "gb200_nvl72", tp_degree=1, pp_degree=1,
                              dp_degree=2304, ep_degree=ep)
            table.append({
                "nvlink_domain_size": dom, "ep": ep,
                "decode_tbt_ms": r.decode_time_per_token_ms,
                "train_tps": r.training_throughput_tokens_per_sec,
            })
    os.environ.pop("AC_HARDWARE_SPEC_DIR", None)
    importlib.reload(tm)
    shutil.rmtree(tmp, ignore_errors=True)

    by_dom = {}
    for row in table:
        by_dom.setdefault(row["nvlink_domain_size"], []).append(row)
    summary = {}
    for dom, rows in sorted(by_dom.items()):
        argmax_tps = max(rows, key=lambda r: r["train_tps"])
        argmin_tbt = min(rows, key=lambda r: r["decode_tbt_ms"])
        summary[str(dom)] = {
            "argmax_train_tps_ep": argmax_tps["ep"],
            "argmin_tbt_ep": argmin_tbt["ep"],
            "train_tps_by_ep": {str(r["ep"]): r["train_tps"] for r in rows},
            "tbt_by_ep": {str(r["ep"]): r["decode_tbt_ms"] for r in rows},
        }
    out = dict(HEADER)
    out.update({"part": "V3_domain_sensitivity", "rows": table,
                "summary_by_domain": summary})
    path = os.path.join(RUNS, "v3_sensitivity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path, out


def part_v2b():
    """True K2 shape: 384 experts. EP=72 must be absent (384 % 72 != 0)."""
    r = optimize("gb200_nvl72", _constraints(moe_n_experts_options=[384]))
    eps = sorted({int(getattr(ev.arch, "ep_degree", 1) or 1)
                  for ev in r.all_evaluated if getattr(ev.arch, "moe", None)})
    frontier_eps = sorted({int(getattr(ev.arch, "ep_degree", 1) or 1)
                           for ev in r.pareto_frontier
                           if getattr(ev.arch, "moe", None)})
    best = {}
    for ev in r.all_evaluated:
        a = ev.arch
        if not getattr(a, "moe", None) or not ev.meets_constraints:
            continue
        ep = int(getattr(a, "ep_degree", 1) or 1)
        if ep not in best or ev.serving_tbt_ms < best[ep][0]:
            best[ep] = (ev.serving_tbt_ms, ev.predicted_loss,
                        a.total_params / 1e9)
    out = dict(HEADER)
    out.update({
        "part": "V2b_k2_384_experts", "hardware": "gb200_nvl72",
        "evaluated_eps": eps, "frontier_eps": frontier_eps,
        "ep72_present": 72 in eps,
        "best_feasible_tbt_by_ep": {
            str(k): {"tbt_ms": v[0], "loss": v[1], "total_b": v[2]}
            for k, v in sorted(best.items())},
    })
    path = os.path.join(RUNS, "v2b_384_experts.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path, out


if __name__ == "__main__":
    p1, rows, stress_rows = part_mechanism()
    print("wrote", p1, f"({len(rows)} ep rows, {len(stress_rows)} stress rows)")
    p2, _ = _dump_search("h100", "search_h100.json")
    print("wrote", p2)
    p3, _ = _dump_search("gb200_nvl72", "search_nvl72.json")
    print("wrote", p3)
    p4, _ = part_v3()
    print("wrote", p4)
    p5, _ = part_v2b()
    print("wrote", p5)
