#!/usr/bin/env bash
# Audit grid: 3 capabilities × 3 reference configs + NSA/YOCO + arg validation.
# Run from the repo root. Writes everything to ./out/audit/.
set -e
OUT=out/audit
mkdir -p "$OUT"

echo "=== 1. Greenfield × Mistral-7B-ish (dense) ==="
python ac/cli_compile.py \
    --hardware h100 --params 7 --tokens 2 --context 8192 \
    --serving-tbt 50 --serving-batch 32 --tp 8 --pp 1 --dp 8 \
    --output-config "$OUT/mistral_arch.json" \
    --output-justification "$OUT/mistral_arch.md" \
    --output-pareto "$OUT/mistral_pareto.csv" \
    --no-shadow-prices

echo
echo "=== 2. Greenfield × GPT-OSS-120B-ish (MoE 128 × top-4) ==="
python ac/cli_compile.py \
    --hardware h100 --params 5.1 --tokens 4 --context 32768 \
    --serving-tbt 80 --serving-batch 16 --tp 8 --pp 1 --dp 8 \
    --allow-moe --moe-n-experts 128 --moe-top-k 4 \
    --output-config "$OUT/gpt_oss_arch.json" \
    --output-justification "$OUT/gpt_oss_arch.md" \
    --output-pareto "$OUT/gpt_oss_pareto.csv" \
    --no-shadow-prices

echo
echo "=== 3. Modifier × GPT-OSS-120B baseline ==="
python ac/cli_compile.py \
    --hardware h100 --tp 8 --pp 1 --dp 8 \
    --baseline-config configs/gpt_oss_120b.json \
    --out "$OUT/gpt_oss_modifier" \
    --allow-quality-spending --quality-risk-budget-pct 2

echo
echo "=== 4. Delta-eval × Mistral-7B + GQA(group_size=8) ==="
python ac/cli_delta_eval.py \
    --baseline-config configs/mistral_7b.json \
    --hardware h100 --tp 8 --workload long_context \
    --apply swap_attention_to_gqa --apply-args group_size=8 \
    --out "$OUT/mistral_delta_gqa"

echo
echo "=== 5. Delta-eval × GPT-OSS-120B + MLA(latent_dim=256) ==="
python ac/cli_delta_eval.py \
    --baseline-config configs/gpt_oss_120b.json \
    --hardware h100 --tp 8 --workload chat \
    --apply swap_attention_to_mla --apply-args latent_dim=256 \
    --out "$OUT/gpt_oss_delta_mla"

echo
echo "=== 6. NSA + YOCO emission ==="
python ac/cli_compile.py \
    --hardware h100 --params 7 --tokens 2 --context 32768 \
    --serving-tbt 50 --serving-batch 16 --tp 8 --pp 1 --dp 8 \
    --nsa --nsa-select-top-k 16 --nsa-window-size 512 \
    --yoco --yoco-n-self-attn-layers 1 \
    --output-config "$OUT/nsa_yoco.json" \
    --output-justification "$OUT/nsa_yoco.md" \
    --output-pareto "$OUT/nsa_yoco.csv" \
    --no-shadow-prices

echo
echo "=== 7. Argument validation (expects pre-eval error) ==="
python ac/cli_delta_eval.py \
    --baseline-config configs/mistral_7b.json --hardware h100 --tp 8 \
    --apply swap_attention_to_gqa --apply-args n_kv_heads=4 \
    || true

echo
echo "=== 8. Stress probe ==="
python ac/cli_stress.py stress --known Mistral-7B --hw h100 \
    --batch 32 --decode-kv 32768 --tp 8

echo
echo "All audit tests complete. Outputs in $OUT/."
