# 5. Benchmarks

This page describes the data, models, metrics and systems, and the protocol of both benchmarks.
The commands are in [`kv_manager/README.md`](../kv_manager/README.md).

## Workload: real agent recordings

All current experiments use real runs of **mini-SWE-agent** on **SWE-bench Verified**: real GitHub
issues from open-source Python projects. In each run the agent reads a bug report, runs shell
commands (`grep`, `sed`, `python -c`, tests), reads the outputs and works toward a fix. The
recordings come from
[`john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash`](https://github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash).
We use the longest run from each of 10 projects:

| Recording | Agent turns | Full conversation (Llama tokens) |
|---|---|---|
| sphinx-doc__sphinx-8548 | 106 | 56,036 |
| sympy__sympy-12419 | 97 | 50,076 |
| django__django-11265 | 95 | 46,804 |
| pydata__xarray-4687 | 88 | 33,904 |
| astropy__astropy-13398 | 59 | 31,381 |
| pytest-dev__pytest-5787 | 41 | 28,591 |
| mwaskom__seaborn-3069 | 72 | 27,640 |
| psf__requests-2931 | 62 | 22,176 |
| scikit-learn__scikit-learn-11310 | 66 | 21,598 |
| pallets__flask-5014 | 43 | 18,464 |

Each recording is formatted in the Llama-3 chat format: system, task, then alternating agent
messages (with the command they ran) and tool outputs (`ipython` role). Phase 0 also used ALFWorld
traces and synthetic filler. The download and formatting code is in
[`kv_manager/experiments/gate/data.py`](../kv_manager/experiments/gate/data.py).

## Models

| Role | Model | KV per token |
|---|---|---|
| Target (the agent's model) | `unsloth/Llama-3.1-8B-Instruct` (same weights as Meta's, no login needed) | 128 KB |
| Draft (scorer) | `unsloth/Llama-3.2-1B-Instruct` | 32 KB |

Phases 0 and 1 used the 1B model only.

## Metrics

| Metric | Definition | Better |
|---|---|---|
| **ΔNLL** | Increase in mean negative log-likelihood (nats/token) of the **real** next agent message, compared with a full-precision history. As a rough guide, +0.03 is barely noticeable, +0.3 a clear loss, +1.0 serious damage. | Lower |
| **Top-1 agreement** | Fraction of positions where the model's most likely next token is unchanged | Higher |
| **Recall vs. oracle** | Share of the oracle's top blocks that a scorer also picks. The oracle uses the real next message's attention. | Higher |
| **Bytes fraction** | Reload bytes divided by a full bf16 reload, including scales and the always-bf16 blocks | Lower |
| **TTFT after tool return** | Time from sending a resumed turn to receiving its first token. This is the **headline latency**. | Lower |
| **End-to-end time, throughput** | Per-request latency, and requests and prompt tokens per second over a run | Lower / higher |
| **Load and save bytes** | Per-request bytes moved by the connector, from its stats log | Lower |
| **Draft scoring latency** | Time for the draft's work after the tool output arrives | Lower |

## Benchmark 1: gate experiment (offline)

**Purpose:** decide whether the draft earns its place (RQ1 and RQ2 in [Goals](03-goals.md)) before
building it into vLLM.

**Samples:** each recording is cut at the first tool-call pause whose history reaches 8k, 16k, 24k
and 32k tokens. This gives **31 pauses** with histories of 8k–34k tokens. Each sample has three
parts:
- **history**: everything up to the agent message that issued the tool call. This is what sits
  in CPU memory during the pause.
- **fresh**: the tool's output and the assistant header, read in at full precision.
- **target**: the real next agent message, up to 256 tokens.

**Per sample:**
1. Compute the target model's history KV.
2. Make 4-bit and 2-bit copies with the exact system format (`kvm/quant.py`, per block).
3. Score the history blocks after the tool returns, using each method:

   | Method | How a block is scored |
   |---|---|
   | `consensus` (proposed) | Draft reads the tool output, then 4 sampled guesses × 10 tokens. Score = mean over guesses of the max attention over steps. |
   | `draft_window` | Draft attention from the last 32 tool-output tokens only (no guesses) |
   | `specprefill` | SpecPrefill's reference scoring: 8 greedy lookahead queries, softmax over prompt keys, average-pool of 13, max over layers and heads, mean over steps |
   | `recency` | Newest blocks first |
   | `random` | Random |
   | `oracle` | Target's attention from the real next message (upper bound; not usable in practice) |

4. For each scheme, rebuild the history at mixed precision, read the tool output on top in bf16,
   and teacher-force the target message. Block 0, the recent 64 tokens and the partial last block
   are always bf16.

   | Scheme | Precision |
   |---|---|
   | `uniform_int4`, `uniform_int2` | All other blocks at that precision, no draft |
   | `<method>`, top = bf16 or int4, keep 5/10/20/40% | Top-scored blocks at the top precision, the rest at int2 |

5. Repeat for block sizes 16 and 64 tokens.

**Decision rule:** fixed in code before any run; see [Goals](03-goals.md#success-criteria).

**Outputs:**
- `results/gate/metrics.csv`, one row per sample × block size × scheme;
- `samples.csv`, with draft latencies;
- `pareto_B16.png` and `pareto_B64.png`, quality against bytes;
- `summary.md`, with paired comparisons, consensus vs. SpecPrefill, and the GO/NO-GO verdict.

**Cost:** one A100 80 GB, about 40 GB of memory, roughly 30–60 minutes (our estimate).

## Benchmark 2: replay (online serving)

**Purpose:** measure real serving performance in vLLM (RQ3).

**Systems under test.** All run vLLM 0.27.1 with `--attention-backend FLASH_ATTN`, prefix caching
on, and the **same GPU KV budget** (`--kv-cache-memory-bytes`, 16 GiB by default):

| System | Description |
|---|---|
| `vllm` | **Baseline 1.** Plain vLLM. KV evicted from the GPU is recomputed. |
| `vllm_offload` | **Baseline 2.** vLLM + its built-in `OffloadingConnector`. Evicted KV is reloaded from CPU in bf16. |
| `kvm_bf16` | Our connector, reloading everything in bf16. A sanity check that should behave like `vllm_offload`. |
| `kvm_int4`, `kvm_int2` | Our connector, uniform 4-bit or 2-bit reload (block 0 and recent window in bf16) |
| `kvm_draft` (M2) | Our connector, 2-bit reload with draft-selected bf16 upgrades |

**Protocol** ([`kv_manager/bench/replay.py`](../kv_manager/bench/replay.py)):
- **N concurrent agents.** Each replays one recording, cycling through the 10 and starting at a
  random time within a 10 s window.
- **Requests.** Each turn sends the whole conversation so far as token ids to `/v1/completions`.
  The reply is streamed, 64 tokens with end-of-sequence ignored, and the time to first token is
  recorded.
- **Teacher forcing.** The next turn uses the real agent message and tool output from the
  recording, not the generated text. Every system therefore sees identical contexts.
- **Simulated tool delay** between turns: 0.5 s plus an exponential with mean 2 s.
- **Session marker.** Each agent's system prompt starts with a unique marker, so copies of the
  same recording do not share cached blocks.
- **Eviction pressure.** Set the GPU KV budget small relative to agents × context, otherwise
  nothing is evicted and every system looks the same.

**Outputs:**
- `results/replay/<system>.jsonl`: one line per request, with agent, turn, prompt length, tool
  delay, TTFT, end-to-end time and generated tokens;
- `<system>.summary.txt`: p50, p90 and p99 TTFT after tool return, plus throughput;
- `<system>.kvm_stats.jsonl`, for our connector: per-request loaded, bf16 and low-bit blocks,
  bytes and saved blocks;
- `<system>.server.log`.

**Why there is no quality number from vLLM yet.** Asking vLLM for `prompt_logprobs` makes it
recompute the whole prompt (`skip_reading_prefix_cache`), so it would never read the reloaded KV.
For now, quality comes from the gate experiment, which uses the same quantization code. M3 adds
forced decoding through a logits-processor plugin, so quality can be measured on exactly the KV
vLLM reloaded.

## Environment and hardware

- **Software:** Python 3.12, vLLM 0.27.1, torch 2.13.0 (CUDA 13.0) and transformers 5.14.1. Every
  dependency is pinned in
  [`kv_manager/envt/requirements.lock.txt`](../kv_manager/envt/requirements.lock.txt). Build with
  `bash kv_manager/envt/setup_env.sh`.
- **GPUs:** NVIDIA A100 80 GB (driver ≥ 580). Phase 1 used a Colab T4.
- **Tests:** `cd kv_manager && python -m pytest tests -q` runs on CPU in about 10 s.

## Reproducibility checklist

- [ ] Environment built from the lock file, with `vllm.__version__ == "0.27.1"`
- [ ] Recordings downloaded with `python -m experiments.gate.data`
- [ ] Gate run, with `results/gate/summary.md` committed
- [ ] Replay runs at a documented `KV_BYTES`, agent count and seed, with the summaries committed
- [ ] The same GPU type and the same `KV_BYTES` for every system being compared
