# Glossary

| Term | Meaning |
|---|---|
| **Agent** | An LLM that works on a task over many turns, calling external tools (shell commands, APIs) and reading their outputs |
| **Attention sink** | The first few tokens of a sequence, which receive a lot of attention regardless of content. Dropping or degrading them hurts quality, so block 0 is always kept in bf16. |
| **bf16** | 16-bit "brain float" format, the full precision used for KV here. 2 bytes per value. |
| **Block** | vLLM's unit of KV storage: 16 consecutive tokens, all layers' keys and values. Our CPU store, precision choices and scores all work per block. |
| **Block hash** | vLLM's chained hash of a block's tokens (and all tokens before it). Equal hashes mean equal KV, which is how prefix caching and our CPU store find reusable blocks. |
| **Consensus score** | Our token importance: `S_i = (1/N) Σ_j max_t A_j[t, i]`. It takes the attention from N sampled draft guesses of L tokens each, max over steps, then the mean over guesses. Summed per block. |
| **Decode** | Generating output tokens one at a time. Each step reads the whole KV cache. |
| **Draft model** | A small, cheap model from the same family as the target (here Llama-3.2-1B for Llama-3.1-8B). It is used to predict what the target will need. |
| **Eviction** | Permanently deleting KV entries (tokens). The opposite of our approach. |
| **Forced decoding** | Making the model "generate" a fixed, known text so its log-probabilities can be read. Planned for in-vLLM quality measurement (M3). |
| **Gate experiment** | The offline go/no-go test (Phase 2) deciding whether draft-guided precision beats uniform 4-bit |
| **HF / transformers** | Hugging Face's model library. Used for the offline experiments and for the draft model. |
| **KIVI** | A KV quantization recipe: keys quantized per channel, values per token, asymmetric min-max with a scale and zero-point |
| **KV cache** | The stored key and value vectors of every earlier token, for every layer. Its size grows linearly with context length. |
| **KV connector** | vLLM's plugin interface for loading and saving KV outside the GPU (`KVConnectorBase_V1`). Our system is one. |
| **Load-then-upgrade** | Our reload order: bring back every block at 2-bit right away, then overwrite the draft's top blocks with bf16 before the model continues |
| **Lookahead / rollout / guess** | A few tokens the draft generates to anticipate what the agent will write next. Their attention reveals which history they rely on. |
| **mini-SWE-agent** | A minimal coding agent whose recorded runs on SWE-bench Verified are our workload |
| **Mixed precision** | Different blocks stored or reloaded at different bit widths (bf16, 4-bit, 2-bit) |
| **NLL / ΔNLL** | Negative log-likelihood (nats per token) of the real next agent message. ΔNLL is its increase relative to a full-precision history; 0 means no damage. |
| **Offload / reload** | Moving KV from GPU to CPU memory during a pause, and back when the agent resumes |
| **Oracle** | A scorer that uses the real future (the actual next message). An upper bound, not usable in practice. |
| **PCIe** | The CPU↔GPU link. Practical throughput is roughly 20–25 GB/s per direction on our A100s (estimate). Reload time is bytes divided by this. |
| **Pinned memory** | Page-locked CPU memory that the GPU can copy from asynchronously |
| **Prefill** | Reading prompt tokens into the model to compute their KV. Recomputing a long history is a large prefill. |
| **Prefix caching** | vLLM reusing KV blocks from earlier requests that share the same leading tokens |
| **Quantization (2-bit, 4-bit)** | Storing each value as a 2- or 4-bit code plus a per-group scale and zero-point (fp16). 2-bit with scales is about 0.20× the bytes of bf16 here; 4-bit is about 0.32×. |
| **Recent window** | The newest history tokens (64 here), always reloaded in bf16 |
| **RoPE** | Rotary position embedding. Keys carry their position, which is why dropping tokens complicates things and keeping all tokens does not. |
| **SpecPrefill** | Prior work that uses a draft model's lookahead attention to pick which prompt tokens the target reads (and drops the rest) |
| **SWE-bench Verified** | A benchmark of real GitHub issues from Python projects, used for coding agents |
| **Target model** | The large model the agent actually uses (Llama-3.1-8B-Instruct here) |
| **Teacher forcing** | Feeding the recorded real text instead of the model's own output, so all systems see identical contexts |
| **Tool-call pause** | The time between an agent issuing a tool call and receiving its output. The model is idle, but its KV is still needed. |
| **TTFT** | Time to first token. "TTFT after tool return" is our headline latency. |
| **vLLM** | The open-source LLM serving engine we build on (version 0.27.1) |
