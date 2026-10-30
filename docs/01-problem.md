# 1. The problem

## The KV cache in one paragraph

A language model writes text one token at a time. To choose the next token, every layer
*attends* to all earlier tokens, using a **key** and a **value** vector for each one. Recomputing
these for the whole text at every step would be far too slow, so the model computes them once and
keeps them. That store is the **KV cache**. Its size is fixed per token and grows in a straight line
with the length of the text:

```
KV bytes per token = layers × 2 (key and value) × KV heads × head size × 2 bytes (bf16)
```

| Model | KV per token | KV for a 32k-token context |
|---|---|---|
| Llama-3.2-1B-Instruct | 32 KB | 1 GiB |
| Llama-3.1-8B-Instruct | 128 KB | 4 GiB |
| Llama-3.3-70B-Instruct | 320 KB | 10 GiB |

An 80 GB GPU running an 8B model (about 16 GB of weights) therefore has room for the KV of only
a handful of long conversations at once.

## Why agents make it worse

An LLM **agent** works on a task over many turns. Each turn, it writes a message with an action,
such as a shell command, and an external **tool** runs it. The tool's output is appended to the
conversation, and the agent writes its next message. This has three consequences:

1. **The context gets long.** In our 10 real coding-agent runs on SWE-bench (mini-SWE-agent),
   conversations reach 18k–56k tokens after 41–106 turns ([Phase 1](04-phases.md#phase-1)).
2. **The model pauses on every turn while the tool runs.** It sits idle for seconds or longer, but its
   KV cache is still needed for the next turn.
3. **Many agents run at once.** A serving system handles many sessions, and they compete for GPU
   memory and for the CPU↔GPU link (**PCIe**).

Phase 1 measured the cost on a small model. At 56k tokens the KV cache was 1.7 GB for the 1B
model. Decoding slowed from about 42 to about 3.3 tokens/s, and reading each new tool output cost
about 7× more than at the start of the run.

## What a serving engine can do during a tool pause

When an agent's request finishes (it has issued a tool call), the engine has three options for
that agent's KV cache:

| Option | Cost |
|---|---|
| **Keep it on the GPU** | Idle KV occupies memory that other agents need. This limits how many agents fit, and when memory runs out something gets evicted anyway. |
| **Evict and recompute later** | The next turn must re-read the whole history (a full *prefill*). For an 8B model at 32k tokens that is on the order of seconds, by our estimate. |
| **Offload to CPU memory and reload later** | All 4 GiB must cross PCIe before the agent can continue. At a practical 20–25 GB/s that is roughly 0.15–0.2 s per resume, on the critical path. With many agents resuming at once, they share the link. |

vLLM, the most widely used open-source serving engine, does the second by default. With its
`OffloadingConnector` plugin it does the third. Our baselines are exactly these two configurations
([Benchmarks](05-benchmarks.md)).

## The problem we solve

> When an agent resumes after a tool call, how can it get its history back on the GPU with far
> fewer bytes than a full-precision reload, without losing the information it needs and without
> changing the serving engine's internals?

## Why existing approaches fall short

| Approach | What it does | Why it is not enough here |
|---|---|---|
| **Eviction** (H2O, SnapKV, SpecKV) | Permanently drops tokens judged unimportant | Agents often need old context again, for example a file printed 30 turns ago. In Phase 1, evicting did **7–18× more damage** than quantizing the same tokens. |
| **Uniform quantization** (KIVI) | Stores every token at 2 or 4 bits | Simple and strong, but blind to content: important and unimportant tokens get the same precision. |
| **Retention policies** (Continuum) | Keeps an agent's KV on the GPU for a predicted tool duration (a time-to-live) | Reduces reloads but keeps full-size KV resident. Whatever expires is reloaded or recomputed in full. |
| **CPU offload** (vLLM `OffloadingConnector`, LMCache) | Moves KV to CPU memory and back | Always moves every byte at full precision. |
| **Draft-guided sparse prefill** (SpecPrefill) | A small draft model picks important prompt tokens, and the target model reads only those | Drops tokens (lossy in the eviction sense). vLLM declined native support because it has no sparse KV cache. |

The full comparison is in [Related work](07-related-work.md). The next page explains our approach:
[The idea](02-idea.md).
