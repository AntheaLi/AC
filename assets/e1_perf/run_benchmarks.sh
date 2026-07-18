#!/usr/bin/env bash
# =============================================================================
# E1 performance anchors — vLLM measurement runner (rented H100 SXM 80GB x8 node)
#
# ac_version: 0.4.0
# quality_model_version: effective_capacity_v2
# git_commit: c170cda
# experiment_date: 2026-07-17
# agent_wave: gate2-wave1
#
# Pre-registered protocol: validation/prereg/e1_predictions.md (wave 1, locked
# before any measurement). Do not change flags without recording an erratum in
# that file. Outputs land in raw_logs/ next to this script:
#   <anchor>_server.log      vLLM server log (startup memory accounting kept)
#   <anchor>_bench.json      vllm bench serve result (p50/p95/p99 ttft,tpot,itl)
#   <anchor>_nvidia_smi.csv  per-GPU memory.used sampled at 1 Hz
#   <anchor>_deviations.txt  only written when the EP fallback fires
#   env_snapshot.txt         GPU/driver/vllm fingerprint for the T1 provenance
#
# Gated models (meta-llama/*) require:  export HF_TOKEN=hf_...   (license accepted)
# =============================================================================
set -euo pipefail

VLLM_VERSION="0.25.1"     # PyPI 2026-07-14; pinned in pre-registration
RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw_logs"
PORT="${PORT:-8000}"
NUM_PROMPTS=32
INPUT_LEN=2048
OUTPUT_LEN=512
MAX_MODEL_LEN=2560

mkdir -p "$RAW_DIR"

# --- 0. install pin (idempotent) --------------------------------------------
python3 -m pip show "vllm" 2>/dev/null | grep -q "Version: ${VLLM_VERSION}" \
  || python3 -m pip install "vllm==${VLLM_VERSION}"

# --- 1. environment fingerprint (T1 provenance) ------------------------------
{
  date -u "+utc %Y-%m-%dT%H:%M:%SZ"
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  python3 -m pip show vllm torch | grep -E "Name|Version"
} > "$RAW_DIR/env_snapshot.txt" 2>&1 || true

# --- 2. anchor loop ----------------------------------------------------------
# fields: anchor_key|hf_repo|tp|kv_bytes_per_gpu|extra_serve_flags
# kv_bytes_per_gpu: 2*K,V * n_layers * (n_kv_heads/TP) * d_head * 2B * 2560 tok
# (official architecture dims, NOT AC outputs; floor 64 MiB; see prereg section 2.2)
ANCHORS=(
  "mistral_7b_tp1|mistralai/Mistral-7B-v0.1|1|335544320|"
  "llama31_8b_tp1|meta-llama/Llama-3.1-8B|1|335544320|"
  "qwen3_8b_tp1|Qwen/Qwen3-8B|1|377487360|"
  "qwen3_32b_tp2|Qwen/Qwen3-32B|2|335544320|"
  "llama33_70b_tp4|meta-llama/Llama-3.3-70B-Instruct|4|209715200|"
  "llama33_70b_tp8|meta-llama/Llama-3.3-70B-Instruct|8|104857600|"
  "gpt_oss_120b_tp8ep8|openai/gpt-oss-120b|8|67108864|--enable-expert-parallel"
)

serve_and_bench () {
  local anchor="$1" repo="$2" tp="$3" kvb="$4" extra="$5"
  echo "================ ${anchor} (${repo}, tp=${tp}) ================"

  # start 1 Hz memory sampler BEFORE the server so weight-load peak is captured
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -lms 1000 \
    > "$RAW_DIR/${anchor}_nvidia_smi.csv" 2>/dev/null &
  local smi_pid=$!

  # shellcheck disable=SC2086
  vllm serve "$repo" \
    --tensor-parallel-size "$tp" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --kv-cache-memory-bytes "$kvb" \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --port "$PORT" $extra \
    > "$RAW_DIR/${anchor}_server.log" 2>&1 &
  local srv_pid=$!

  # wait for readiness (large downloads can take a long time)
  local waited=0
  until curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo "server died during startup; see ${anchor}_server.log"
      # fallback: gpt-oss without --enable-expert-parallel (pre-registered deviation path)
      if [[ -n "$extra" ]]; then
        echo "retrying without extra flags: $extra" | tee "$RAW_DIR/${anchor}_deviations.txt"
        kill "$smi_pid" 2>/dev/null || true
        serve_and_bench "$anchor" "$repo" "$tp" "$kvb" ""
        return
      fi
      kill "$smi_pid" 2>/dev/null || true
      return 1
    fi
    sleep 15; waited=$((waited+15))
    if (( waited > 5400 )); then
      echo "timeout waiting for ${anchor} server"; kill "$srv_pid" "$smi_pid" 2>/dev/null || true
      return 1
    fi
  done

  # the benchmark itself (concurrency 1 == AC --serving-batch 1)
  vllm bench serve \
    --backend openai --model "$repo" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --ignore-eos \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
    --save-result --save-detailed \
    --result-dir "$RAW_DIR" --result-filename "${anchor}_bench.json" \
    --metadata "anchor=${anchor}" "tp=${tp}" "input_len=${INPUT_LEN}" \
               "output_len=${OUTPUT_LEN}" "concurrency=1" "vllm=${VLLM_VERSION}" \
    > "$RAW_DIR/${anchor}_bench_stdout.log" 2>&1 || true

  # let the last decode finish flushing, then tear down
  sleep 5
  kill "$srv_pid" 2>/dev/null || true
  wait "$srv_pid" 2>/dev/null || true
  kill "$smi_pid" 2>/dev/null || true
}

for row in "${ANCHORS[@]}"; do
  IFS='|' read -r anchor repo tp kvb extra <<< "$row"
  serve_and_bench "$anchor" "$repo" "$tp" "$kvb" "$extra"
done

echo "all anchors done -> $RAW_DIR"
