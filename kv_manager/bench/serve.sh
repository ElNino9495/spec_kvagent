#!/usr/bin/env bash
# Launch one system under test with an identical GPU KV budget.
#
#   CUDA_VISIBLE_DEVICES=1 bash kv_manager/bench/serve.sh <system> [extra vllm serve args]
#
# systems:
#   vllm          plain vLLM 0.27.1 (prefix caching; evicted blocks are recomputed)
#   vllm_offload  vLLM + its built-in OffloadingConnector (CPU tier, reloads everything in bf16)
#   kvm_bf16      AgentKVConnector, reload everything in bf16 (sanity: should match vllm_offload)
#   kvm_int4      AgentKVConnector, uniform 4-bit reload (block 0 + recent window bf16)
#   kvm_int2      AgentKVConnector, uniform 2-bit reload (block 0 + recent window bf16)
#
# env: MODEL, PORT (8000), KV_BYTES (GPU KV budget, 16 GiB), CPU_BYTES (128 GiB),
#      MAX_LEN (65536), STATS (AgentKVConnector per-request stats JSONL),
#      DRY_RUN=1 prints the vllm command instead of running it
set -euo pipefail

SYSTEM=${1:?usage: serve.sh <vllm|vllm_offload|kvm_bf16|kvm_int4|kvm_int2> [vllm args]}
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=${MODEL:-unsloth/Llama-3.1-8B-Instruct}
PORT=${PORT:-8000}
KV_BYTES=${KV_BYTES:-$((16 * 1024 ** 3))}
CPU_BYTES=${CPU_BYTES:-$((128 * 1024 ** 3))}
MAX_LEN=${MAX_LEN:-65536}
STATS=${STATS:-$ROOT/results/replay/${SYSTEM}.kvm_stats.jsonl}

args=(serve "$MODEL" --port "$PORT" --attention-backend FLASH_ATTN --enable-prefix-caching
      --kv-cache-memory-bytes "$KV_BYTES" --max-model-len "$MAX_LEN")

case "$SYSTEM" in
  vllm) ;;
  vllm_offload)
    extra="{\"cpu_bytes_to_use\": $CPU_BYTES}"
    args+=(--kv-transfer-config "{\"kv_connector\": \"OffloadingConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": $extra}") ;;
  kvm_bf16|kvm_int4|kvm_int2)
    mode=${SYSTEM#kvm_}
    [ "$mode" = bf16 ] || mode="uniform_$mode"
    mkdir -p "$(dirname "$STATS")"
    extra="{\"mode\": \"$mode\", \"cpu_bytes\": $CPU_BYTES, \"stats_path\": \"$STATS\"}"
    args+=(--kv-transfer-config "{\"kv_connector\": \"AgentKVConnector\", \"kv_connector_module_path\": \"kvm.connector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": $extra}") ;;
  *) echo "unknown system: $SYSTEM" >&2; exit 2 ;;
esac

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${DRY_RUN:-}" ]; then
  printf '%s\n' "$ROOT/envt/.venv/bin/vllm" "${args[@]}" "$@"
  exit 0
fi
exec "$ROOT/envt/.venv/bin/vllm" "${args[@]}" "$@"
