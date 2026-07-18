# README hardware-table rows — Task C (gb200_nvl72 / h800)

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: c170cda
- experiment_date: 2026-07-17
- agent wave: "gate2-wave1"

> Orchestrator: insert the two rows below into the README
> "Supported components → Hardware targets" table (README.md ~line 765),
> keeping the existing column order. Do NOT apply by editing this file;
> this is a snippet only. The `--hardware` enum line (~line 140) should
> read: `--hardware {h100, b200, gb200_nvl72, h800, tpu_v5p, tpu_v5e, trainium2, trn2, trainium3, trn3}`.

## Rows to insert (after the **NVIDIA B200** row)

| **NVIDIA GB200 NVL72** (rack-scale, 72× B200) | 2 250 / 4 500 / 9 000 (MXFP4) | 192 GB ×72 | NVLink 5 domain = 72 GPUs (1.8 TB/s per GPU, 130 TB/s rack); IB scale-out 400 Gb/s per GPU | wmma + MX |
| **NVIDIA H800 SXM** (H100 export SKU) | 990 / 1980 / — | 80 GB | NVLink 4 reduced (400 GB/s) | wmma 16×16 |

## Footnote to add directly under the table

`gb200_nvl72` and `h800` are *system-level* targets: the per-chip numbers
are identical to B200 / H100 respectively (same silicon); what changes is
the fabric. `gb200_nvl72` models the full rack as one NVLink domain
(`nvlink_domain_size=72` in `ac/hardware_specs/gb200_nvl72.json`), which
is what makes rack-scale expert parallelism (EP up to 72) priceable —
see `validation/c_nvl72/report.md` (EP=72 on the Pareto frontier for a
1T-class 288-expert MoE on NVL72 vs mandatory spill + a −21% training
all-to-all tax on single-node H100). `h800` models the export-restricted
H100 SKU whose NVLink is capped at 400 GB/s. Rack power, cooling, and
failure rates are intentionally not modeled (roadmap).

## Provenance for the two rows (accessed 2026-07-17)

- NVIDIA GB200 NVL72 product page (72-GPU NVLink domain, 1.8 TB/s per
  GPU, 130 TB/s aggregate): https://www.nvidia.com/en-us/data-center/gb200-nvl72/ — T2
- GB200 NVL72 scale-out 400 Gb/s per GPU (ConnectX-7 NDR): NVIDIA
  partner spec sheets (https://www.spheron.network/gpu-rental/gb200/) — T3
- NVIDIA H800 GPU datasheet (mirrored; H100 silicon rows + NVLink
  400 GB/s): https://www.scribd.com/document/777167019/NVIDIA-H800-GPU-Datasheet — T2/T3
