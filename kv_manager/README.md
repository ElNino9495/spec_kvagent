# kv_manager: draft-guided precision for KV reload after tool-call pauses

Working directory for the new idea, built on vLLM 0.27.1 (the baseline). No custom kernels:
everything is plain PyTorch plus vLLM's existing plugin points. For the background, goals and
phases, read the project docs in [`../docs/`](../docs/README.md). This README covers the commands.

## Idea

1. **Pause.** When an agent's turn ends in a tool call, the target model's KV blocks are copied
   to CPU memory, both in bf16 and as a 2-bit copy, and the GPU blocks are freed for other agents.
2. **Tool returns.** The draft model (Llama-3.2-1B, whose own KV stays on the GPU) prefills the
   tool output and makes N short rollouts. Each history block gets the Phase 1 consensus score
   `S_i = mean_j max_t A_j[t, i]`.
3. **Reload.** The top-k% blocks, the first block and the recent window come back in bf16. The
   rest come back as 2-bit and are dequantized into ordinary bf16 vLLM blocks. Every token is
   still present and positions are unchanged, so vLLM's attention runs untouched.

Target/draft: `Llama-3.1-8B-Instruct` / `Llama-3.2-1B-Instruct`. They share a tokenizer, so
their blocks line up exactly. The draft's KV is 32 KB/token, a quarter of the target's
128 KB/token. (A Qwen3-8B/0.6B pair would only save 1.3x: 144 vs 112 KB/token.)

Why it might pay off: reloading 32k tokens of Llama-8B KV is 4.2 GB, about 0.2 s over PCIe,
and it sits on the critical path to the first token after the tool returns. With many agents
resuming at once, PCIe is shared between them.

Main risk: uniform 4-bit reload needs no draft and moves about the same number of bytes. Phase 1
showed that 2-bit on everything already costs only +0.054 NLL. The gate experiment below decides
whether the draft earns its place.

Decided so far:
- **Baselines:** plain vLLM 0.27.1 (prefix caching, recompute after eviction), and vLLM with its
  built-in CPU offload (`OffloadingConnector`, which reloads everything in bf16).
- **Reload:** load-then-upgrade. When the tool returns, start loading the 2-bit copy of every
  block right away. When the draft's scores arrive, overwrite the top blocks with bf16. This
  hides the draft's latency behind the 2-bit transfer.
- **Closest prior work:** SpecPrefill (ICML 2025, arXiv:2502.02789). It also uses a
  Llama-3.2-1B draft's lookahead attention to score tokens, but it *drops* unselected prompt
  tokens at prefill. We keep every token and only change reload precision across tool pauses.
  Native vLLM support was proposed and closed as not planned (vllm-project/vllm#39060), because
  vLLM has no sparse KV cache beyond sliding window. Keeping all tokens avoids that problem.

## Gate experiment (run first)

[`experiments/gate/`](experiments/gate/) uses the 31 tool-call pauses from 10 real SWE-bench agent runs
(histories of 8k–34k tokens). For each one, it reloads the history at mixed precision, prefills the tool
output on top, and measures the damage to the real next agent message (ΔNLL and top-1 agreement) against
the bytes reloaded:

- Baselines: uniform int4 and uniform int2 (no draft).
- Proposed: draft consensus (N=4, L=10), scored after the tool returns.
- Cheaper variant: draft window, which uses attention from the tool output only, with no rollouts.
- SpecPrefill's scoring, exactly as in its reference code: 8 greedy lookahead queries, softmax over
  prompt keys, avg-pool 13, max over layers and heads, mean over steps.
- Controls: recency, random, and the oracle (the target's attention from the real next message).
- Settings: top blocks at bf16 or int4 with the rest at int2, keep 5/10/20/40%, block sizes 16 and 64.
- Quantization uses [`kvm/quant.py`](kvm/quant.py), the same per-block format the system will use.

The go/no-go rule was fixed before running (see `plot_gate.py`). A draft method passes if it does
either of the following:
- It moves no more bytes than uniform int4 and at least halves its ΔNLL (paired 95% CI below 0).
- It moves at most 75% of uniform int4's bytes without being worse (CI upper bound < +0.01 nats).

```bash
cd kv_manager && source envt/.venv/bin/activate
python -m experiments.gate.data                                    # download recordings (CPU)
CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_gate --max-samples 1 --hist-lens 4000  # smoke test
CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_gate         # 1 x A100 80GB, ~40 GB
python -m experiments.gate.plot_gate                               # results/gate/summary.md + pareto plots
```

## System design

Everything is a vLLM 0.27.1 plugin, with no fork and no custom kernels.

- **`kvm/connector.py`** (M1, built): an out-of-tree `KVConnectorBase_V1`, loaded with
  `kv_connector_module_path`. CPU blocks are keyed by vLLM's own prefix-cache block hashes, so
  an agent's next turn matches its earlier turns automatically. Bookkeeping is in
  `kvm/sched.py` and copies are in `kvm/transfer.py`.
  - **Save:** when a request finishes (the agent issued a tool call), `request_finished`
    returns True, so vLLM keeps the blocks until our copy is done. Every new full prompt block
    goes to pinned CPU memory in bf16, plus a packed low-bit copy (`kvm.quant`, quantized on the
    GPU) unless mode is `bf16`. Stored blocks are never overwritten, so a block reloaded at low
    precision keeps its original bf16 copy.
  - **Load:** for blocks no longer in vLLM's GPU prefix cache, `get_num_new_matched_tokens`
    returns an async load. vLLM parks the request and keeps serving others while the worker
    copies the blocks in on a side CUDA stream and dequantizes the low-bit ones into ordinary
    bf16 vLLM blocks. Block 0 and the recent window always come back in bf16.
  - Requests with `prompt_logprobs` skip the connector, because vLLM recomputes their whole
    prompt.
- **`kvm/draft.py`** (M2): the draft (Llama-3.2-1B, HF transformers) runs inside vLLM's worker
  process on the same GPU, owned by the connector. It keeps its own KV for each agent session
  (`kv_transfer_params.session_id`), outside vLLM's block pool. Load becomes load-then-upgrade:
  1. copy the 2-bit blocks to the GPU and dequantize them;
  2. run the draft on the tool output and score the history blocks;
  3. copy the top blocks in bf16 over the 2-bit ones;
  4. report the load as finished.

  We don't use vLLM's built-in drafter for two reasons:
  - It runs *after* the target forward pass (`gpu_model_runner.execute_model` →
    `propose_draft_token_ids`), but the scores are needed before the target sees the tool output.
  - Its layers share block ids with the target (`UniformTypeKVCacheSpecs`), so its KV would be
    offloaded together with the target's.
- **Modes** (`kv_connector_extra_config.mode`):
  - M1: `bf16` (sanity check, should match vLLM's own offload), `uniform_int4`, `uniform_int2`.
  - M2: `draft` (consensus), `specprefill`, `recency`, `random`.
- **Memory:** the connector is created after vLLM profiles GPU memory, so the draft's weights,
  its KV and transfer buffers are not counted. Every system therefore runs with the same fixed
  `--kv-cache-memory-bytes` budget.
- **`bench/replay.py`** (M1, built): replays the recorded SWE-bench runs as N concurrent agents
  against `vllm serve`.
  - Each turn sends the whole conversation so far as token ids. It is teacher-forced with the
    real agent messages, so every system sees identical contexts.
  - Turns are separated by simulated tool delays (0.5 s plus an exponential with mean 2 s).
  - It records time to first token after each tool return, end-to-end time and throughput.
  - Per-request load and save stats (blocks, bytes) come from the connector's `stats_path`.
  - Quality is not measured in vLLM: requesting `prompt_logprobs` makes vLLM recompute the
    whole prompt, so it would never read the reloaded KV. Quality comes from the gate experiment,
    which uses the same `kvm.quant` format. Measuring it inside vLLM would need forced decoding
    through a logits-processor plugin (M3).

Build plan:
1. **M1 (done, CPU-tested only):** the connector with `bf16`, `uniform_int4` and `uniform_int2`,
   the replay benchmark, and launch scripts for both baselines. This is the fallback system if
   the gate says NO-GO.
2. **M2:** the draft inside the connector, with load-then-upgrade.
3. **M3:** concurrency sweeps, ablations, in-vLLM quality via forced decoding, and plots.

### Running M1 (needs one GPU)

```bash
cd kv_manager
DRY_RUN=1 bash bench/serve.sh kvm_int2            # print the exact vllm serve command
CUDA_VISIBLE_DEVICES=1 bash bench/run_systems.sh "vllm vllm_offload kvm_bf16 kvm_int4 kvm_int2" --agents 16
```

`serve.sh` starts one system:

| System | What it is |
|---|---|
| `vllm` | plain vLLM |
| `vllm_offload` | vLLM + `OffloadingConnector` |
| `kvm_bf16`, `kvm_int4`, `kvm_int2` | AgentKVConnector in each mode |

All systems run with `--attention-backend FLASH_ATTN` and the same `KV_BYTES` GPU KV budget
(default 16 GiB). Make `KV_BYTES` small relative to agents × context so that blocks get evicted
during tool pauses. `run_systems.sh` starts each server, waits for `/health`, runs the replay,
and writes `results/replay/<system>.{jsonl,summary.txt,server.log}`.

## Environment

Requirements: Linux x86_64, NVIDIA driver >= 580 (CUDA 13.0), and `python3` with `pip` on PATH
(used only to bootstrap `uv`).

```bash
bash kv_manager/envt/setup_env.sh
source kv_manager/envt/.venv/bin/activate
```

- `envt/requirements.txt`: top-level pins (vLLM 0.27.1 plus the versions its CUDA CI tested with).
- `envt/requirements.lock.txt`: every transitive dependency pinned. Commit it so everyone gets an
  identical environment.
- After editing `requirements.txt`, run `bash kv_manager/envt/setup_env.sh --relock`.
- uv, the managed Python 3.12, the venv and the download cache all live under `envt/` (gitignored).

Tests (CPU only): `cd kv_manager && python -m pytest tests -q`

## Layout

```
kv_manager/
├── envt/                  requirements.txt, requirements.lock.txt, setup_env.sh (+ gitignored venv)
├── kvm/
│   ├── quant.py           block-wise int2/int4 KV quantization, shared by experiments and system
│   ├── sched.py           scheduler-side bookkeeping (CPU slots, LRU, load/save jobs)
│   ├── transfer.py        worker-side copies and (de)quantization on a side CUDA stream
│   └── connector.py       AgentKVConnector (vLLM KV connector plugin)
├── bench/                 replay.py, serve.sh, run_systems.sh
├── experiments/gate/      data.py, run_gate.py, plot_gate.py
├── tests/                 CPU unit tests
└── results/               gate/ and replay/ outputs
```
