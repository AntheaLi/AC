# E3 — Decision Retrodiction (anchor-report section)

ac_version 0.4.0 · quality_model_version effective_capacity_v2 · git
c170cda (prereg committed at ba3bbd5, scaffolding b32064c) · executed
2026-07-17 · wave gate2-wave1 → wave-2 execution.

Preregistered protocol: `validation/prereg/e3_cases.md` (frozen wave 1;
wave-2 fills appended). Raw outputs: `validation/e3_retrodiction/runs/`.
Configs + provenance: `validation/e3_retrodiction/model_configs/`.

## Verdict table

| Case | Retrodicted decision | Verdict | Key measured evidence |
|---|---|---|---|
| 1a | V3: MHA→MLA | **HIT** | Relieved all 3 binding axes (HBM-BW-decode 0.99→0.89, HBM-capacity 1.93→0.57, KV-footprint 1.52→0.21); quality **gain** −0.0042; EXPANDS_FRONTIER, dominates 12 |
| 1b | V3: GQA-8→MLA | **HIT** | Relieved hbm_bw_decode; quality gain −0.0095; EXPANDS_FRONTIER; relief breadth < 1a as pre-registered |
| 2 | Llama-3: MHA→GQA-8 | **PARTIAL-HIT** (frontier position) | GQA-8 relieves both KV axes at quality +0.0001; MQA arm pays +0.0211, dominated by 51; MHA-noop arm gains nothing. GQA-8 lands INTERIOR (dom 2), not literally on frontier |
| 3 | Mistral: SWA-4096 | **HIT** | SWA arm ON_FRONTIER and dominates the GQA-only comparator; KV-footprint 1.60→0.05; side-effect: introduces tc_util_prefill axis entry |
| 4 | first-K-dense k=3 (V3) / k=1 (K2) | **HIT** | k=3 quality gain −0.0088 EXPANDS_FRONTIER; k=1 −0.0029 EQUIVALENT; ordering k=3>k=1 matches V3's own choice |
| 5 | K2: wide-EP on NVL72-class fabric | **HIT** (arm b partial) | h100 EP=64: all_to_all 0.006→0.228 (38×, band relaxed), TBT 12.1 ms; gb200_nvl72 EP=64: all_to_all 0.015, TBT 5.05 ms (2.39×), hbm_capacity relieved 1.65→0.57, EXPANDS_FRONTIER; C's uncapped search: EP=72 on frontier (11 rows), best EP=72 TBT 9.0 < best EP=8 33.9 ms |
| N1 | MHA-7B @32k stresses KV axis | hit (control) | hbm_bw_decode = 1.016 binding on single H100 |
| N2 | d_model=4104 lattice penalty | partial (control) | tc_util_prefill 0.852 vs 0.752 aligned (+13 %); misaligned also gains hbm_bw_decode binding; aligned twin already "loaded" — letter unattainable |
| N3 | MoE spill feasibility boundary | hit per D1 audit record (TP1 arm partial) | literal 7B: feasible, no axes (audit arithmetic confirmed, plan-literal falsified per D1); 430B TP8: hbm_capacity 1.41 violated-but-feasible; 430B TP1: dominated by 64, decode 7.5× — rejected via penalty+dominance, sentinel missed by GB/GiB slip (9.87× HBM < 10×) |

**Score vs the pre-registered bar (≥ 4/5 case hits): 4 HIT + 1
PARTIAL-HIT of 5 cases — bar met.** Controls: 1 hit, 2 partials (both
attributed to control-design letters, not model defects).

## Per-case reasoning

**Case 1 (V3 MLA).** AC reproduces DeepSeek's central claim from the V2
paper ("MLA achieves superior performance compared with MHA, and meanwhile
significantly reduces the KV cache", arXiv:2405.04434v5) in both
directions at once: from MHA the swap relieves *all three* binding stress
axes — "Applying MLA relieves HBM-BW-decode 0.99→0.89; HBM-capacity
1.93→0.57; KV-footprint 1.52→0.21. Quality gain: -0.0034 attention,
-0.0008 shape-law" — and from GQA-8 it still pays ("Quality gain: -0.0099
attention"), with narrower relief, exactly the pre-registered asymmetry.
The quality model scoring MLA as a *gain* over MHA (not merely a cheap
approximation) is the strongest single retrodictive signal in E3.

**Case 2 (Llama-3 GQA-8).** The sweet-spot structure is fully reproduced:
GQA-8 "relieves HBM-capacity 1.31→0.41; KV-footprint 1.00→0.12. Quality
cost: +0.0001", while the MQA extreme pays +0.0210 attention quality and
is dominated by 51 candidates, and the MHA-noop extreme buys nothing.
Partial rather than full hit because AC's frontier membership test places
GQA-8 INTERIOR (dominated by 2 candidates) — the pre-registered letter was
frontier membership. Execution surfaced a labeling error in the prereg
itself (D7): group_size counts query heads per KV group, so arms 2/3 were
named inversely; the corrected mapping preserves the substantive claim.

**Case 3 (Mistral SWA).** Under the long_context preset the SWA arm is the
unique frontier candidate and the full-attention GQA comparator is
dominated — AC's own dominator count identifies the SWA arm as the
preferred point, matching Mistral's "SWA to handle longer sequences at
smaller cost". Measured KV-footprint relief (1.60→0.05) exceeds the
paper's "8× cache reduction at 32K". The introduced tc_util_prefill entry
(windowed-prefill tiling) is an honest model side-effect worth a
calibration look, not a falsification.

**Case 4 (first-K-dense).** Both historical choices beat the all-MoE
counterfactual on the quality residual at negligible cost — "Quality gain:
-0.0090 MoE" for k=3 (EXPANDS_FRONTIER, dominates 4), "−0.0030 MoE" for
k=1 — and the k=3 > k=1 ordering on the V3 backbone matches V3's actual
choice of 3 dense layers there. The quality model demonstrably "sees"
dense early layers, which was the falsification condition.

**Case 5 (K2 wide-EP).** The h100 arm reproduces the cross-node all-to-all
tax in the exact channel C characterized: all_to_all score 0.006→0.228
(38×) with link_bw_ep collapsing to 50 GB/s, and the tax surfaces in TBT
(12.1 ms h100 vs 5.05 ms gb200_nvl72 at EP=64, 2.39×; C measured 1.99× at
its frozen point, V1(b)). The axis band stays *relaxed* at 0.228 — the
self-damping caveat C already flagged (V1(d): h100 0.240 vs NVL72 0.015) —
which is why arm (b) is a partial. On gb200_nvl72 the same EP=64 arm
"relieves HBM-capacity 1.65→0.57" and EXPANDS_FRONTIER; and C's uncapped
search artifact (V2(a)/V2(b)) supplies the frontier-level claim: EP=72
enters the NVL72 Pareto frontier (11 rows) with the best TBT of any EP
(9.0 vs 33.9 ms for EP=8). My own greenfield arm, capped at 30k candidates
by the 300 s sandbox, selected a dense arch and is recorded as
execution-capped, not falsifying (D8).

**Controls.** N1 confirms the KV/bandwidth axis binds exactly where
arithmetic says it must (hbm_bw_decode 1.016 at 32k MHA). N2 confirms the
lattice penalty's sign and magnitude channel (tc_util_prefill 0.852 vs
0.752) but the aligned twin already sits in the "loaded" band, so the
"penalty absent in aligned" letter was unachievable — a control-design
error. N3 confirms both feasibility-boundary mechanisms: the audit record
(D1) predicted the literal 7B arm must be feasible (it is, no binding
axes), the 430B TP8 arm sits in the violated-but-feasible soft band
(1.41×), and the TP1 arm is crushed by the continuous penalty
(hbm_capacity 11.27, decode 7.5× slower, dominated by 64) — the hard
sentinel itself was missed by a GB-vs-GiB unit slip in my own arithmetic
(9.87× < 10× HBM), not by any model misbehavior.

## Error taxonomy (non-hits)

| Entry | Case | Category | Detail |
|---|---|---|---|
| E-1 | 2 | constraint-reconstruction error | group_size semantics inverted in prereg arm labels (D7); frontier-membership letter missed (INTERIOR, dom 2) |
| E-2 | 5(b) | prior gap (threshold calibration) | all_to_all score rises 38× but band stays relaxed (self-damping; corroborates C V1(d)) — "binding/heavily penalized" letter too strong for the axis-band mechanism |
| E-3 | N2 | constraint-reconstruction error | aligned twin already in "loaded" tc_util band at 32k; "absent in aligned" letter unattainable |
| E-4 | N3-TP1 | constraint-reconstruction error | GB vs GiB unit slip: 848.17 GB = 9.87× HBM < 10× sentinel (D9) |
| E-5 | 5(d) own run | execution artifact (not scored against model) | 300 s sandbox forced --max-candidates 30000; dense selection under cap; EP-frontier evidence cited from C's uncapped artifact (D8) |

No model defects were attributed: every non-hit traces to prereg/control
wording, unit arithmetic, threshold calibration expectations, or the
execution sandbox.
