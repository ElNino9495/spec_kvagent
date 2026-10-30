Idea 3: Keep the draft model's KV on the GPU as an index
Between turns, offload the target model's KV entirely but keep the draft model's KV resident. It's about 10–50× smaller, depending on the model pair.

When the tool returns, the draft runs attention over its full cache, which costs little, and finds the top-k blocks. Only those target KV blocks get prefetched.
Overlap the prefetch with prefilling the tool output. The new tokens are processed while the needed old KV blocks arrive over PCIe.
How it differs from TriForce: TriForce does this inside a single request. You apply it across tool-call pauses and across memory tiers, and you add a scheduling policy for which requests' draft caches stay resident.


TriForce speeds up generation over very long contexts using two levels of speculative decoding. It's still lossless: the final output is exactly what the full model would produce. The paper is Sun et al., COLM 2024, "TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding."

One correction to how I described Idea 3 earlier: in TriForce, the small draft model is not the index. The important KV blocks are chosen using the target model's own keys. That actually helps your novelty claim, as explained below.

1. The problem TriForce solves
With long contexts, the bottleneck in decoding moves from loading weights to loading the KV cache.

Take Llama-2-7B as an example. Its KV cache needs about 0.5 MB per token, from 32 layers × (keys + values) × 4096 dimensions × 2 bytes. At 128K tokens that's about 64 GB of KV, compared with about 14 GB of weights. Every decoding step has to read all of it, so most of each step is spent moving KV through memory.

2. Three observations it builds on
Attention is sparse. A small part of the KV cache (a few thousand tokens out of 128K) captures most of the attention mass.
Important tokens stay important for a while. The KV that matters for token t is mostly the same for tokens t+1 … t+k. So you can select the important KV once and reuse it for several steps.
A tiny model with a tiny cache is fast but poor. Its guesses are only good for drafting a few tokens ahead.
3. The three-level hierarchy
Level 1  Tiny draft model (Llama-68M)      + StreamingLLM cache (attention sinks + recent window)
            │  drafts γ1 tokens
            ▼
Level 2  Target model weights              + RETRIEVAL cache (sparse, ~4K tokens, on GPU)
            │  verifies Level 1, then acts as the draft for Level 3 (γ2 tokens)
            ▼
Level 3  Target model weights              + FULL KV cache
            │  final verification → lossless
Level 1 to Level 2: this pair fixes the weight-loading cost. The 68M model is very cheap to run.
Level 2 to Level 3: this pair fixes the KV-loading cost. Level 2 reads only about 4K tokens of KV instead of 128K, and Level 3 reads the full KV only once per batch of γ2 tokens.
4. How the retrieval cache works (the part most relevant to you)
Split the full KV cache into chunks of about 8–16 tokens.
For each chunk, store one representative key, the mean of its keys.
Take the target model's current query vector, compute a dot product with every representative key, and keep the top-scoring chunks up to the budget.
Rebuild this selection once per verification cycle, not every token. This relies on observation 2.
The index is built from the target model's own keys, averaged per chunk. The 68M draft never sees the full context. It only has a short StreamingLLM window.

5. Offloading mode
TriForce also has a setting where the full KV cache lives in CPU memory:

The retrieval cache (about 4K tokens) and the Level 1 model stay on the GPU.
Level 3 verification has to stream the full KV over PCIe. That's expensive, but it happens once per γ2 accepted tokens, so the cost is spread out.
The paper reports roughly 2.3× speedup on an A100 and much larger gains (around 7–8×) in the offloading setup on consumer GPUs. Check the exact numbers in the paper before citing them.

6. Where TriForce stops, and where your idea starts
TriForce assumes	Agentic reality
One long, static prompt
Context grows turn by turn with tool outputs
Continuous decoding
Generation pauses for seconds or minutes during tool calls
Single request
Many concurrent agents competing for GPU memory and PCIe bandwidth
Index uses the target model's chunk-mean keys, on GPU
Your index is the draft model's full-context KV, kept resident while the target KV is offloaded
Lossless only: full KV checked every cycle
You can offer a lossless mode and a cheaper lossy mode
So your pitch is:

TriForce selects important KV within a single continuous decode. We select it across tool-call pauses, using a small draft model's full-context KV as an index that stays on the GPU. That lets us offload the target model's KV entirely during a pause, and prefetch only what's needed while the tool output is being prefilled.

7. Your comparison setup
System	What happens during a tool pause	What happens when the tool returns
vLLM + offloading (CPU offload or LMCache connector)
KV is evicted or offloaded when memory pressure requires it, with no knowledge of tools
Reload everything, or recompute it
Continuum + offloading
KV is pinned for a time-to-live (TTL) based on predicted tool duration, and offloaded after expiry
Reload everything if offloaded
Your system
Target KV fully offloaded. Draft KV (small) stays on GPU. The draft predicts important blocks
Prefetch only the top-k target blocks while prefilling the tool output. Fetch the rest lazily, or never (lossy mode)
One design decision to make early is lossless versus lossy:

Lossless: the target model eventually verifies with the full KV, as in TriForce. Your benefit then comes from better overlap and from fewer full KV loads at the moment the tool returns.
Lossy: the target model attends only to the prefetched blocks. This saves much more, but you have to report task success rates (SWE-bench resolve rate, τ-bench pass rate) against the KV budget.
I'd implement both and present them as two operating points.

Add two ablations, because reviewers will ask for them:

A TriForce-style index (target chunk-mean keys kept on GPU) instead of the draft KV. This isolates whether the draft model's KV is actually a better or cheaper index.
Random or recency-based top-k prefetch. This shows the draft model's predictions are doing real work.
Metrics to report:

Time to first token after the tool returns, which is your headline number.
End-to-end agent job completion time.
Throughput with N concurrent agents.
PCIe bytes moved.
GPU memory used per agent.
Task accuracy, in lossy mode.
A useful next step is a rough cost model: draft KV size compared with target KV size, PCIe time for top-k blocks compared with all blocks, and the prefill time of a typical tool output. That tells you early whether the prefetch can actually hide behind the prefill. Want me to work through those numbers for a specific model pair, such as Qwen3-8B with Qwen3-0.6B?



