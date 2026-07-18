#!/usr/bin/env bash
# Golden quickstart snapshot — release-gate §1.3 regression harness.
#
# Runs the three README quickstart commands (greenfield / modifier /
# delta-eval) and stores their output files, timestamp-scrubbed, under an
# output directory. Re-run into a second directory and `diff -r` the two:
# every byte must match (prediction-layer determinism gate).
#
# Usage:
#   scripts/golden_snapshot.sh [OUT_DIR]     # default: out/golden
#   scripts/golden_snapshot.sh out/golden_new
#   diff -r out/golden out/golden_new        # must be empty
#
# Scrubbed fields (the only allowed diffs):
#   - "generated_at": "..."   JSON timestamps
#   - "search_time_sec": N    wall-clock seconds embedded in config JSON
#   - "Search time: N.Ns"     wall-clock line in the justification report
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-out/golden}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for cli in ac-compile ac-delta-eval; do
    if ! command -v "$cli" >/dev/null 2>&1; then
        echo "error: $cli not on PATH — run \`pip install -e .\` first" >&2
        exit 1
    fi
done

echo "[snapshot] 1/3 greenfield: 7B dense on h100"
ac-compile \
  --hardware h100 --params 7 --tokens 2 --context 8192 \
  --serving-tbt 50 --serving-batch 32 --tp 8 --pp 1 --dp 8 \
  --output-config "$WORK/greenfield/arch.json" \
  --output-justification "$WORK/greenfield/arch.md" \
  --output-pareto "$WORK/greenfield/pareto.csv" \
  --output-shadow-prices "$WORK/greenfield/shadow_prices.json" \
  --quiet

echo "[snapshot] 2/3 modifier: mistral-7b local Pareto"
ac-compile \
  --baseline-config configs/mistral_7b.json \
  --hardware h100 --tp-options 4,8 \
  --quality-risk-budget-pct 1.0 --allow-quality-spending \
  --out "$WORK/modifier" \
  --quiet

echo "[snapshot] 3/3 delta-eval: GQA-8 on mistral-7b @ 32k"
ac-delta-eval \
  --baseline-config configs/mistral_7b.json \
  --hardware h100 --tp 8 --workload long_context \
  --apply swap_attention_to_gqa --apply-args group_size=8 \
  --out "$WORK/delta_gqa"

mkdir -p "$OUT_DIR/greenfield" "$OUT_DIR/modifier" "$OUT_DIR/delta_gqa"
cp "$WORK"/greenfield/* "$OUT_DIR/greenfield/"
cp "$WORK"/modifier/*   "$OUT_DIR/modifier/"
cp "$WORK"/delta_gqa/*  "$OUT_DIR/delta_gqa/"

# Scrub the only fields allowed to differ between runs.
find "$OUT_DIR" -type f \( -name '*.json' -o -name '*.md' -o -name '*.csv' \) -print0 |
while IFS= read -r -d '' f; do
    perl -pi -e 's/"generated_at": "[^"]*"/"generated_at": "<SCRUBBED>"/g;
                 s/"search_time_sec": [0-9.]+/"search_time_sec": "<SCRUBBED>"/g;
                 s/Search time: [0-9.]+s/Search time: <SCRUBBED>/g' "$f"
done

echo "[snapshot] wrote $(find "$OUT_DIR" -type f | wc -l | tr -d ' ') files -> $OUT_DIR"
echo "[snapshot] verify determinism: scripts/golden_snapshot.sh out/golden_new && diff -r '$OUT_DIR' out/golden_new"
