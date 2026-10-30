# 6. Architecture

The system is an out-of-tree **vLLM 0.27.1 KV connector**. vLLM loads it by module path, and it
needs no changes to vLLM and no custom kernels. Code lives in
[`kv_manager/kvm/`](../kv_manager/kvm/).

```
vllm serve unsloth/Llama-3.1-8B-Instruct --attention-backend FLASH_ATTN --kv-cache-memory-bytes <budget> \
  --kv-transfer-config '{"kv_connector": "AgentKVConnector", "kv_connector_module_path": "kvm.connector",
                         "kv_role": "kv_both", "kv_connector_extra_config": {"mode": "uniform_int2", "cpu_bytes": ...}}'
```

## Background: vLLM's KV connector API

vLLM stores KV in fixed-size **blocks** of 16 tokens and keeps finished requests' blocks as a
**prefix cache**. Each full block is identified by a chained hash of its tokens. A KV connector is
created twice:
- once on the **scheduler** side, which decides what to load and save;
- once on the **worker** side, which owns the GPU tensors and does the copies.

The scheduler side passes per-step **metadata** to the worker side.

| Hook | Side | What we do |
|---|---|---|
| `get_num_new_matched_tokens(request, num_computed)` | Scheduler | Count the blocks beyond vLLM's own GPU prefix hit that are stored on CPU, and report them as an **async** load. vLLM parks the request in `WAITING_FOR_REMOTE_KVS` and keeps serving others. |
| `update_state_after_alloc(request, blocks, n)` | Scheduler | vLLM has assigned GPU blocks. Queue a `LoadJob` (GPU block ids, CPU slots, per-block precision). |
| `request_finished(request, block_ids)` | Scheduler | Queue a `SaveJob` for new full prompt blocks. Return `True` so vLLM keeps the blocks until our copy is done. |
| `build_connector_meta` / `update_connector_output` | Scheduler | Send jobs to the worker; learn which saves and loads finished |
| `register_kv_caches(kv_caches)` | Worker | Receive the per-layer KV tensors, check the FlashAttention layout, and allocate the pinned CPU pools |
| `start_load_kv(...)` | Worker | Called every engine step: start this step's saves and loads on a side CUDA stream |
| `get_finished(...)` | Worker | Poll CUDA events and report finished sends and receives back to the scheduler |

## Components

| File | Role |
|---|---|
| [`quant.py`](../kv_manager/kvm/quant.py) | Block-wise KIVI-style quantization in PyTorch: `pack_blocks` and `unpack_blocks` (whole blocks to and from one byte row), `block_bytes` (byte accounting). The offline gate experiment uses the same functions. |
| [`sched.py`](../kv_manager/kvm/sched.py) | Scheduler-side bookkeeping in pure Python: `BlockIndex` (block hash → CPU slot, LRU eviction, pins), `plan_bits` (precision per block), `SchedulerState` (load and save jobs, stats) |
| [`transfer.py`](../kv_manager/kvm/transfer.py) | Worker-side `KVTransfer`: pinned host pools, gather and scatter of blocks across layers, device↔host copies, quantize and dequantize, all on a side CUDA stream with completion events |
| [`connector.py`](../kv_manager/kvm/connector.py) | `AgentKVConnector`, thin glue between the vLLM hooks and the two classes above |
| `draft.py` (M2) | Draft model owner and scorer |
| [`../bench/`](../kv_manager/bench/) | Replay client, server launcher and multi-system runner |

## Save path (tool call issued)

1. vLLM calls `request_finished`. We take the request's full blocks: prompt blocks only by
   default, and never past `num_computed_tokens`.
2. We keep the blocks whose hash is not already stored, then allocate CPU slots (evicting
   least-recently-used ready blocks if needed) and queue a `SaveJob`. We return `True`.
3. In the next step's `start_load_kv`, the worker:
   - makes the side stream wait for the compute stream;
   - gathers the blocks from every layer in chunks of 64 blocks;
   - copies them to the bf16 pool, packs the low-bit copy on the GPU and copies that to the
     low-bit pool;
   - records an event.
4. `get_finished` reports the request as sent once the event completes. The scheduler marks the
   slots ready and vLLM frees or caches the GPU blocks.

## Load path (tool returned)

1. The new request's prompt shares its hash chain with the earlier turns. vLLM first takes
   whatever blocks are still in its GPU prefix cache, then asks us how many of the following
   blocks are on CPU.
2. After allocation, `plan_bits` assigns a precision to each block: block 0 and the last
   `ceil(64 / 16) = 4` blocks get bf16, and the rest get the mode's low precision. We pin the
   slots against eviction and queue a `LoadJob`.
3. The worker copies each precision group in chunks: host to device, unpack and dequantize if
   low-bit, then scatter into vLLM's paged blocks. It records an event.
4. Once the event completes, `get_finished` reports the request as received. vLLM moves it back to
   the waiting queue and prefills the new tokens on top of the reloaded history.

## Storage format

A CPU **slot** holds one block (16 tokens) for **all layers**:

- **bf16 pool:** `(slots, layers, kv_heads, 16, 2 × head_dim)`. This is vLLM's FlashAttention
  logical layout, with keys in the first half of the last dimension and values in the second.
- **Low-bit pool** (`uniform_int4` / `uniform_int2`): one byte row per slot, made of these
  sections in order:

  | Section | Shape per slot | Type |
  |---|---|---|
  | key codes | layers × heads × 16 × head_dim·bits/8 | uint8, packed 8/bits per byte |
  | key scale, key zero-point | layers × heads × 1 × head_dim each | fp16 (per channel, over the block's tokens) |
  | value codes | layers × heads × 16 × head_dim·bits/8 | uint8 |
  | value scale, value zero-point | layers × heads × 16 × 1 each | fp16 (per token, over channels) |

For Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128) a slot is 2 MiB in bf16, 656 KiB at
4-bit and 400 KiB at 2-bit.

Pools are allocated once, at their exact size, and pinned with `cudaHostRegister`.
`pin_memory=True` would round each allocation up to the next power of two.

## Correctness rules

| Rule | Why |
|---|---|
| CPU blocks are keyed by **vLLM's block hash** | A content-addressed store: a later turn matches earlier turns automatically, and identical prefixes are deduplicated |
| A stored slot is **never overwritten** | The CPU store always holds the bf16 original |
| A block ever reloaded at low precision is **never saved again** | vLLM caches the reloaded (lossy) block under the original hash. Saving it would replace the original with a degraded copy. |
| A request aborted **while its load is in flight** is not also saved | Otherwise vLLM would receive both a "sent" and a "received" completion for one request and crash on an assertion |
| Requests with `prompt_logprobs` bypass the connector | vLLM recomputes their whole prompt (`skip_reading_prefix_cache`) |
| Only blocks fully covered by computed tokens are saved | The last sampled token has no KV yet |
| Index uploads use pinned memory and `non_blocking` | A pageable upload would synchronize the side stream on every chunk |
| Side-stream work waits for the compute stream first | Saves read blocks only after the forward pass that wrote them |

The scheduler-side rules are covered by CPU unit tests in
[`kv_manager/tests/`](../kv_manager/tests/).

## Memory accounting

vLLM creates the connector **after** measuring GPU memory. Anything the connector allocates on the
GPU is therefore not counted: the draft model in M2, its KV, and transfer buffers (up to 64 blocks
at a time). All systems run with an explicit, identical `--kv-cache-memory-bytes`, so the target's
KV budget is the same everywhere and there is headroom for the connector.

## M2: the draft inside the connector

The worker-side connector loads the 1B draft (Hugging Face `transformers`) on the same GPU. It
keeps one draft KV cache per agent session, identified by `kv_transfer_params.session_id`.

```
tool returns ─┬─ load 2-bit copies of all evicted blocks, dequantize ──────┐
              └─ draft: read tool output, 4×10-token guesses, score blocks ┴─ load bf16 for top-k% ─ report done ─ target prefill
```

- **At save time:** the draft also reads the agent's newest message, off the critical path.
- **Why not vLLM's built-in speculative-decoding drafter:**
  - In `gpu_model_runner.execute_model`, it runs after the target's forward pass in every step,
    but the scores are needed before the target reads the tool output.
  - Its layers are merged into the target's KV cache group (`UniformTypeKVCacheSpecs`), so its KV
    would be freed and offloaded together with the target's.

## Current limitations

- Single GPU (tensor parallelism untested), FlashAttention backend only, and full-attention
  models only. vLLM's hybrid KV cache manager is turned off automatically for this connector.
- LRU eviction is simple and has no awareness of sessions or tool durations. That kind of
  retention policy is Continuum's territory and could be combined later.
- The CPU pools are allocated and pinned when vLLM starts, which takes a while for large `cpu_bytes`.
- Not yet run on a GPU (see [Phases](04-phases.md#m1-vllm-system-without-the-draft)).
