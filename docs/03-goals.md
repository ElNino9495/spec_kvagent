# 3. Goals

## Goal

Cut the cost of resuming an agent after a tool call: the bytes moved and the time to the first new
token. Keep the history intact and close to full-precision quality, with nothing but a vLLM
plugin.

## Research questions

| ID | Question | Answered by |
|---|---|---|
| RQ1 | At the same reload bytes, does **draft-guided** mixed precision lose less quality than **uniform** 4-bit or 2-bit? | Gate experiment ([Benchmarks](05-benchmarks.md#benchmark-1-gate-experiment-offline)) |
| RQ2 | Is lookahead scoring (consensus) worth its latency, compared with cheaper scorers: attention from the tool output only, SpecPrefill's scoring, recency, random? | Gate experiment, which reports quality and draft latency |
| RQ3 | In a real serving engine, how much do lower-precision reloads cut **time to first token after a tool returns**, and how much do they raise **throughput with many concurrent agents**, compared with plain vLLM and vLLM with CPU offload? | Replay benchmark (M1 for uniform modes, M2/M3 for the draft) |
| RQ4 | What does the precision loss cost end to end, in next-message likelihood and later in agent task success? | Gate (offline) now; forced-decoding measurement inside vLLM in M3 |

## Hypotheses

- **H1 (from Phase 1):** keeping every token at low precision is far less damaging than evicting
  the same tokens.
- **H2:** looking ahead with a draft model finds the blocks the next message needs better than
  looking back or looking only at the tool output.
- **H3:** at long contexts and high concurrency, PCIe reload time is a large part of the
  time to first token after a tool returns. Cutting reload bytes 3× or more shortens it noticeably.
- **H4:** with the draft running alongside the 2-bit load (load-then-upgrade), scoring adds little
  or nothing to that time.

## Success criteria

**Gate (pre-registered, fixed in code before any run).** A draft-guided configuration passes if,
at some block size, either:

- it moves **no more bytes** than uniform 4-bit and **at least halves** its ΔNLL (paired 95%
  bootstrap CI of the difference entirely below 0), or
- it moves **at most 75%** of uniform 4-bit's bytes and is **not worse** (CI upper bound below
  +0.01 nats/token).

If nothing passes, the draft is dropped. The fallback system is uniform low-bit reload (M1),
which still answers RQ3.

**System (M3).** The exact targets will be fixed before the M3 runs, once the gate has chosen the
operating point. We will report, at equal GPU KV budget:

- the median and p90 time to first token after a tool return;
- requests per second;
- PCIe bytes per resume;
- ΔNLL of the real next message.

Each is reported against both baselines across concurrency levels.

## Non-goals

- **No custom GPU kernels and no vLLM fork.** Everything uses plain PyTorch and vLLM's public plugin points.
- **No GPU-side KV compression.** Reloaded blocks are ordinary bf16. The GPU KV budget is the same for all systems.
- **Not lossless.** Precision loss is the trade-off we measure. A lossless mode would get nothing from the draft.
- **No training.** The draft and target are off-the-shelf models.
- **No changes to the agent.** Agents, prompts and tools are replayed exactly as recorded.
- **Not yet:** tensor parallelism, attention backends other than FlashAttention, hybrid or sliding-window models.

## Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Uniform 4-bit matches draft-guided quality | The draft is unnecessary | The gate decides before M2 is built. M1 is a complete system without a draft. |
| Draft scoring is slow (HF eager attention, 10 decode steps for 4 batched guesses) | Adds latency on resume | Load-then-upgrade overlaps scoring with the 2-bit load. The gate measures scoring latency, and cheaper scorers are compared. |
| PCIe is not the bottleneck at low concurrency | Small gains | Report concurrency sweeps and context-length breakdowns, and set the GPU KV budget so eviction actually happens |
| vLLM does not count the draft's GPU memory | Out-of-memory errors or unfair comparisons | Fix `--kv-cache-memory-bytes` identically for all systems and leave headroom |
| Quality cannot be measured inside vLLM with `prompt_logprobs` (it recomputes the prompt) | No in-system quality numbers | The gate measures offline with the same quantization code. M3 adds forced decoding through a logits-processor plugin. |
| Phase 1 results came from a 1B model used as both draft and target | Recall may differ for the 1B→8B pair | The gate uses the real 8B target and 1B draft |

Next: [Phases](04-phases.md).
