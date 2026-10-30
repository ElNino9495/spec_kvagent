#!/usr/bin/env bash
# Run the replay benchmark against several systems, one after another, on one GPU.
#
#   CUDA_VISIBLE_DEVICES=1 bash kv_manager/bench/run_systems.sh "vllm vllm_offload kvm_bf16 kvm_int4 kvm_int2" \
#       --agents 16
#
# Extra args go to bench.replay (timing). Results: kv_manager/results/replay/<system>.jsonl, server logs
# alongside. Env vars of serve.sh (MODEL, PORT, KV_BYTES, CPU_BYTES, MAX_LEN) apply to every system.
set -euo pipefail

SYSTEMS=${1:?usage: run_systems.sh "<systems>" [replay args]}
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=${PORT:-8000}
OUT="$ROOT/results/replay"
mkdir -p "$OUT"
cd "$ROOT"

for s in $SYSTEMS; do
  echo "=== $s"
  bash bench/serve.sh "$s" > "$OUT/$s.server.log" 2>&1 &
  pid=$!
  until curl -sf "localhost:$PORT/health" > /dev/null; do
    kill -0 "$pid" 2> /dev/null || { echo "server for $s exited; see $OUT/$s.server.log" >&2; exit 1; }
    sleep 5
  done
  tag=()
  [[ "$s" == kvm_* ]] && tag=(--tag)
  envt/.venv/bin/python -m bench.replay --url "http://localhost:$PORT" --out "$OUT/$s.jsonl" "${tag[@]}" "$@" \
    | tee "$OUT/$s.summary.txt"
  kill "$pid"
  wait "$pid" 2> /dev/null || true
done
