# 7. Related work

How this project relates to prior work. **Our position:** we use a draft model's lookahead to
choose **precision**, not to choose which tokens to keep. We apply it at the moment an agent's KV
is **reloaded after a tool call**, and we build it as a plugin for an unmodified production engine.

## At a glance

| Work | Decides | Using | Tokens dropped? | Where it acts | Relation to us |
|---|---|---|---|---|---|
| H2O | Which tokens to keep | Accumulated past attention | Yes | GPU KV during decoding | Eviction baseline in Phase 1 |
| SnapKV | Which tokens to keep, per head | Attention from a window at the end of the prompt | Yes | GPU KV after prefill | "Look back" scorer in Phase 1 |
| KIVI | Precision (uniform) | Fixed recipe: keys per channel, values per token, recent tokens in full precision | No | GPU KV | **Our quantization recipe.** Uniform int4 and int2 are our strongest baselines |
| StreamingLLM | Which tokens to keep | First tokens (attention sinks) and a recent window | Yes | GPU KV | Why block 0 and the recent window are always bf16 |
| SpecKV | Which tokens to keep | A draft model's lookahead (single output) | Yes | GPU KV | Closest scoring idea. We use several sampled guesses and quantize instead of dropping |
| SpecPrefill | Which prompt tokens the target reads | A draft's lookahead attention | Yes | Prefill | Closest system idea. Included as a scorer in our gate experiment |
| Quest | Which KV pages to read per query | Per-page key min/max vs. the current query | Not stored, but skipped in attention | GPU attention (custom kernel) | Query-aware page selection, which needs kernels |
| TriForce | Which KV chunks a mid-level draft reads | Chunk-mean keys of the target | No (lossless) | Speculative decoding, optional CPU offload | Inspired the "index + offload" direction |
| Continuum | How long to keep an agent's KV on the GPU | Predicted tool duration and reload cost (time-to-live) | No | Scheduling across tool calls | Same setting (tool pauses). Retention is complementary to our reload precision |
| vLLM `OffloadingConnector`, LMCache | Where KV lives (GPU, CPU, disk) | Prefix-cache hashes, LRU or ARC | No | Storage tiers | **Baseline 2.** We change only what is reloaded and at which precision |
| PASTE, Speculative Actions, SPEX | Which tools or branches to run early | Predictable agent control flow | n/a | Agent execution | Speculate over actions, not KV memory. Complementary |

## Details

**Eviction: H2O, SnapKV, SpecKV.** These keep a token budget by permanently deleting
low-importance tokens. They differ in how importance is estimated:
- H2O uses accumulated past attention ("heavy hitters").
- SnapKV uses attention from an observation window at the end of the prompt.
- SpecKV uses a small draft model's lookahead output, which predicts future importance better.

Phase 1 reproduced this ordering on agent traces: lookahead beat look-back, which beat random. It
also showed that in agent contexts, evicting does 7–18× more damage than quantizing the same
tokens. That is why we keep every token.

**Quantization: KIVI.** Asymmetric 2-bit quantization of the KV cache, with keys quantized per
channel (they have outlier channels) and values per token. It needs no tuning and is near-lossless
in many tasks. We use its recipe per 16-token block, and **uniform KIVI-style 4-bit and 2-bit are
the bars the draft must beat**. In Phase 1, uniform 2-bit already cost only +0.054 nats/token.

**Draft-guided selection: SpecPrefill.** A small draft model (e.g. Llama-3.2-1B) reads the prompt
and runs a few lookahead decoding steps. Its attention scores each token, and the large model reads
only the top tokens, with their original positions. This greatly cuts time to first token, at
some quality loss on hard tasks. It is the closest work to Phase 1's consensus scoring. There are
three differences:
- we never drop tokens; we choose a precision per block;
- we act on KV reloaded after a tool pause, not on a fresh prompt;
- we use the mean over several sampled guesses of the max over steps, while SpecPrefill uses one
  greedy lookahead with the max over layers and heads.

A native vLLM integration was proposed and closed as not planned, because vLLM has no sparse KV
cache ([vllm-project/vllm#39060](https://github.com/vllm-project/vllm/issues/39060)). Our design
avoids that problem by keeping every token. SpecPrefill's exact scoring is one of the methods in
our gate experiment.

**Retrieval and offloading: TriForce, Quest.**
- **TriForce** speeds up long-context generation losslessly with two levels of speculative
  decoding. The middle level reads a small "retrieval cache" of KV chunks, chosen by their mean
  keys. It can keep the full cache in CPU memory.
- **Quest** selects KV pages per query using key bounds, inside a custom attention kernel.

Both act within one continuous decode. We act across tool-call pauses, and we need no custom
kernels.

**Agent-aware serving: Continuum.** Continuum keeps an agent's KV on the GPU during a tool call,
for a time-to-live set from the predicted tool duration and the cost of reloading. This reduces
how often KV is evicted. We reduce what an eviction costs, by reloading fewer bytes. The two
combine naturally: Continuum decides **whether** to keep KV on the GPU, and we decide **how much**
to bring back.

**Storage tiers: vLLM `OffloadingConnector`, LMCache.** These move KV between GPU, CPU and disk
using prefix-cache hashes, and always reload at full precision. `OffloadingConnector` is our second
baseline, and our connector follows the same API and hashing.

**Speculative agents: PASTE, Speculative Actions, SPEX.** These speculate on the agent's external
actions (running tools early, expanding reasoning branches in parallel) to hide latency.
They are complementary: we speculate on what the model will *read*, to manage memory.

## References

| Work | Reference |
|---|---|
| H2O | Zhang et al., NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| SnapKV | Li et al., 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| KIVI | Liu et al., ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| StreamingLLM | Xiao et al., ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| SpecKV | Galim et al., *Draft-based Approximate Inference for LLMs*, ICLR 2026. [arXiv:2506.08373](https://arxiv.org/abs/2506.08373) |
| SpecPrefill | Liu et al., ICML 2025. [arXiv:2502.02789](https://arxiv.org/abs/2502.02789), [code](https://github.com/Jingyu6/speculative_prefill) |
| Quest | Tang et al., ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) |
| TriForce | Sun et al., COLM 2024. [arXiv:2404.11912](https://arxiv.org/abs/2404.11912) |
| Continuum | Li et al., 2025. [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) |
| vLLM | Kwon et al., SOSP 2023 (PagedAttention). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| PASTE, Speculative Actions, SPEX | See [`proposal/references.bib`](../proposal/references.bib) (entries still being completed) |
