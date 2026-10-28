# Speculative KV Cache Quantization for Agentic Serving: Experiments (Part 1)

## 1. The project in one paragraph

A language model keeps a **KV cache**, a stored key and value vector for every token it has already read, so it
does not recompute them for every new token. The cache grows by a fixed amount with every token. In **agentic
workloads**, a model works on a task for dozens or hundreds of turns: it reads tool outputs (file contents, grep
results, test logs), decides on an action, and reads more output. The context can reach tens of thousands of tokens,
and the KV cache becomes a major memory and speed bottleneck. Existing fixes either **evict** tokens (H2O, SnapKV,
SpecKV), which loses them permanently, or **quantize everything uniformly** (KIVI), which does not look at content.
This project proposes **speculative multi-rollout KV cache quantization**:

1. A cheap draft model makes **N** short guesses of what the agent will write next.
2. Every earlier token gets a **consensus attention score** across those guesses:
   `S_i = (1/N) Σ_j max_t A^(j)_{t,i}`, where `A^(j)_{t,i}` is how much step `t` of guess `j` attends to token `i`.
3. High-scoring tokens stay at full precision (FP16/BF16). Everything else is **quantized to low bit-width**
   (2-bit or 4-bit) instead of being thrown away.

---

## 2. Part 0: the baseline (what was done before)

### Goal

Measure how a plain, unoptimised KV cache behaves under an agent-like workload, to confirm the problem exists and to
set up the measurement tools.

### What was done

The script `src/baseline_experiment.py` loads **Llama-3.2-1B-Instruct** with HuggingFace `transformers` and **no KV
cache optimisations**, then simulates an agent session:

- Each turn, it appends one **tool output** to the context. The tool outputs are taken from real agent runs.
- It keeps the KV cache from earlier turns (`past_key_values`), so the cache keeps growing.
- It lets the model generate **64 tokens** of its own reply. These tokens stay in the context.
- Every turn, it logs the context length, prefill time, decode time and speed, and allocated and reserved GPU memory.

### Data used in Part 0

| Run | Data | Turns | Final context |
|---|---|---|---|
| SWE-bench (Django-10554) | real tool outputs from one coding-agent run | 126 | 35,293 tokens |
| SWE-bench (SymPy-14248) | real tool outputs from one coding-agent run | 108 | 34,993 tokens |
| ALFWorld | 36 household-task games joined together | 514 | 48,030 tokens |
| Synthetic | filler text (smoke test only) | 500 | 144,000 tokens |

### What Part 0 showed

- **GPU memory in use grows in a straight line with context length.** For the SWE-bench runs it went from about
  2.3 GB (model weights) to about 3.4 GB at 35k tokens.
- **PyTorch reserves far more memory than it uses.** On ALFWorld, 19.1 GB was reserved for 3.8 GB actually in use.
  On the synthetic run, 93 GB was reserved for 6.8 GB in use.
- **Decoding slows down as the context grows.** On the synthetic run, speed fell from 124 to 74 tokens/s by 144k
  tokens. Part 0 ran on a large data-centre GPU, so at the ~35k-token lengths of the real runs the slowdown was small
  (178 → 170 tokens/s).

### What Part 0 left open

- The context held **only the tool outputs**, with the model's own made-up 64-token replies in between. It did not
  hold the task description or the agent's real messages and commands, so it was not the real conversation.
- The KV cache size was not measured directly, only total GPU memory.
- Only a few recordings were used.
- The core idea of the proposal, consensus scoring, had not been tested yet.

---

## 3. Part 1: the current approach

Part 1 has two experiments.

### Experiment 1: a realistic baseline on more data

**Question:** how do memory and speed behave when the model holds the *whole, real* agent conversation, across many
different tasks?

- **1a: full-recording replay.** **10 real agent recordings** are replayed turn by turn. Each is the longest
  recording from a different open-source project, 18k–56k tokens. The model's memory holds the full real
  conversation: system prompt, task, the agent's real messages and commands, and the real tool outputs. The KV cache
  size is measured **directly from the cache tensors**.
- **1b: Part 0 method on the same data.** One of these recordings (`django__django-11265`) is run again through the
  **unchanged Part 0 script**, so the old method and the new method can be compared on identical data.

### Experiment 2: does consensus scoring work?

**Question:** if draft guesses are used to decide which tokens to keep at full precision, how much does the model's
prediction of the agent's *real* next message suffer? Is it better to quantize the rest or evict it?

Conversations are cut just before a real agent message. Several token-scoring methods choose which tokens to keep,
the rest are either **quantized to 2-bit** (the proposed approach) or **evicted** (the H2O/SnapKV/SpecKV approach),
and the damage to the real next message is measured.

---

## 4. How the experiments were run

### Setup

| | |
|---|---|
| Hardware | Google Colab, NVIDIA T4 GPU (15 GB) |
| Model | `unsloth/Llama-3.2-1B-Instruct` (same weights as Meta's Llama-3.2-1B-Instruct; no login needed) |
| Precision | fp16 |
| KV cache shape | 16 layers × 8 KV heads × 64 dims, keys + values |
| Software | HuggingFace `transformers`, PyTorch, matplotlib |

### Data: real agent recordings

The recordings are real runs of **mini-SWE-agent** on **SWE-bench Verified**, which contains real GitHub issues from
open-source Python projects. In each recording, the agent reads a bug report, runs shell commands (e.g. `grep`,
`sed`, `python -c`, running tests), reads the outputs, and works towards a fix. A data-preparation step downloads each
recording and converts it into two formats:

- **Replay format:** the full conversation (system, task, agent messages, tool outputs), used by Experiment 1a and
  Experiment 2.
- **Part 0 format:** the tool outputs only, one per line, exactly what the Part 0 script expects. Used by
  Experiment 1b.

The 10 recordings in Experiment 1:

| Recording | Turns | Final context (tokens) |
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

### Experiment 1a: full-recording replay

Each recording is played back to the model turn by turn, exactly as it happened:

```
turn 0:  system prompt + task (the bug report)
turn 1:  agent's real message + the command it ran  →  the tool's real output
turn 2:  agent's real message + the command it ran  →  the tool's real output
...
```

On every turn:

1. **Prefill:** the new agent message and tool output are read in, in the Llama-3 chat format. The KV cache from all
   earlier turns is kept.
2. **Decode:** the model writes its own next message, up to 64 tokens, stopping at Llama-3's end-of-turn token
   `<|eot_id|>`. This is timed to measure decode speed.
3. **Undo:** the generated tokens are removed from the cache, because the *real* agent message goes in next turn. The
   cache therefore always holds exactly the real conversation.
4. **Log:** context length, prefill and decode time, KV cache size measured from the cache tensors, and allocated and
   reserved GPU memory.

### Experiment 1b: the Part 0 method on the same recording

`django__django-11265` is run through the unchanged Part 0 script with its input in the Part 0 format.

| | 1a: full recording | 1b: Part 0 method |
|---|---|---|
| What goes into memory | task + real agent messages + real tool outputs | **only the tool outputs** |
| Between tool outputs | the agent's real message | the model's **own 64 generated tokens** |
| Model's own writing | timed, then removed from memory | **stays in memory** and builds up |
| Stops writing at | Llama-3's end-of-turn token | never stops early (always 64 tokens) |
| Text format | Llama-3 chat format | raw text joined together |
| KV cache size | measured directly | only total GPU memory |

### Experiment 2: consensus scoring proof-of-concept

**Contexts.** Three recordings (django-11265, sympy-12419, sphinx-8548) are each cut at about 2k, 3k, 4k and 5k
tokens, giving **12 contexts**. Each cut is at a turn boundary, just **before a real agent message**. That real
message (up to 256 tokens) is the "target" used to measure damage.

**For each context:**

1. **Prefill** the context once.
2. **Draft guesses.** The model makes **8 sampled guesses** of **10 tokens** each (temperature 1.0, top-p 0.95), plus
   one greedy guess. The attention each guess pays to every earlier token is recorded. In this proof-of-concept the
   draft model is the same 1B model.
3. **Score every context token** with each method:

   | Method | How a token is scored |
   |---|---|
   | **Consensus N=1/2/4/8 (proposed)** | `S_i = mean over guesses of (max over guess steps of attention to token i)`, using the first N sampled guesses |
   | **Single greedy guess** (SpecKV-style) | the same formula with one greedy guess |
   | **Past attention** (SnapKV-style) | attention from the last 32 context tokens (looks back, not ahead) |
   | **Random** | random scores (sanity check) |
   | **Oracle** | attention from the *real* next message. It knows the future, so it is an upper bound and not usable in practice |

4. **Keep the top tokens.** For each method, the top **5 / 10 / 20 / 40%** of the middle tokens stay at full precision.
   The first 4 tokens (attention sinks) and the last 64 tokens (recent window) are always kept.
5. **Treat the rest one of two ways:**
   - **Quantize to 2-bit** (the proposed approach), KIVI-style asymmetric min-max quantization. Keys use per-channel
     scales over groups of 32 tokens; values use per-token scales. The quantization is *simulated*
     (quantize → dequantize), so it measures the effect on quality, not real memory savings.
   - **Evict**: the tokens are masked out entirely, as H2O/SnapKV/SpecKV do.
6. **Measure damage.** The real next agent message is fed in, and the model's predictions are compared with the
   untouched cache:
   - **Δ NLL:** the increase in negative log-likelihood of the real message. 0 means no damage; lower is better.
     As a rough guide, +0.03 is barely noticeable, +0.3 is a clear loss and +1.0 is serious damage.
   - **Top-1 agreement:** how often the model's most likely next token is the same as with the untouched cache.
   - **Recall:** what share of the oracle's top tokens a method also picked.
7. A **baseline** that quantizes *everything* to 2-bit (nothing kept at full precision) is also measured.

---

## 5. Experiment 1 results and graphs

### Summary table (all 10 recordings)

| Recording | Final context | KV cache | KB/token | Decode tok/s start → end | Slowdown | Reserved peak |
|---|---|---|---|---|---|---|
| sphinx-8548 | 56,036 | 1.71 GB | 32.0 | 41.5 → 3.3 | 92% | 14.1 GB |
| sympy-12419 | 50,076 | 1.53 GB | 32.0 | 42.9 → 3.7 | 91% | 14.3 GB |
| django-11265 | 46,804 | 1.43 GB | 32.0 | 41.1 → 3.9 | 90% | 14.1 GB |
| xarray-4687 | 33,904 | 1.04 GB | 32.0 | 39.6 → 5.3 | 87% | 8.2 GB |
| astropy-13398 | 31,381 | 0.96 GB | 32.0 | 40.6 → 5.7 | 86% | 4.8 GB |
| pytest-5787 | 28,591 | 0.87 GB | 32.0 | 33.7 → 6.2 | 82% | 5.3 GB |
| seaborn-3069 | 27,640 | 0.84 GB | 32.0 | 40.4 → 6.4 | 84% | 10.3 GB |
| requests-2931 | 22,176 | 0.68 GB | 32.0 | 42.5 → 7.8 | 82% | 10.1 GB |
| scikit-learn-11310 | 21,598 | 0.66 GB | 32.0 | 41.9 → 8.0 | 81% | 6.2 GB |
| flask-5014 | 18,464 | 0.56 GB | 32.0 | 42.6 → 9.1 | 79% | 5.8 GB |

### Graph 1: KV cache size and memory breakdown (`exp1_memory.png`)

**Left panel: KV cache size, all 10 recordings.** The x-axis is context length (thousand tokens); the y-axis is KV
cache size (GB) measured from the cache tensors. Each dot is one turn, and the dashed line is the theoretical value.

- Every turn of every recording falls **exactly on the theoretical line of 32 KB per token**
  (16 layers × 8 KV heads × 64 dims × 2 for K and V × 2 bytes).
- **Meaning:** cache size depends only on the number of tokens, not on what the tokens are. The cost is fully
  predictable, so savings are easy to calculate: 2-bit storage would need about 4 KB per token instead of 32 KB.

**Right panel: memory breakdown for the longest recording (sphinx-8548).** Blue is the KV cache, orange is allocated
memory (weights + cache, i.e. memory actually in use), and green is memory reserved by PyTorch.

- Allocated memory grows smoothly from about 2.3 GB (model weights) to about 4 GB.
- Reserved memory climbs in a **saw-tooth** to about 14 GB (nearly the whole T4), drops back to about 5 GB, and climbs
  again, four times.
- **Meaning:** the data in use is only about 4 GB, but PyTorch holds about **3.5× more**. Each turn the cache is copied
  into a slightly larger block, and temporary attention buffers for long contexts leave gaps the allocator cannot
  reuse. The gaps pile up until the GPU is full, then everything is freed at once and the cycle repeats.
  Fragmentation is a real serving problem on top of the cache size itself.

### Graph 2: decode speed and prefill cost (`exp1_latency.png`)


**Left panel: decode speed.** The x-axis is context length; the y-axis is tokens generated per second.

- All 10 recordings form one clean curve: about **42 tok/s** at 2k tokens, about 15 at 10k, about 8.5 at 20k, about
  4.6 at 40k, and about **3.3 at 56k**, roughly **13× slower**.
- **Meaning:** to write each new token, the model reads the entire KV cache, so decoding is limited by memory
  bandwidth and slows as the cache grows. A smaller cache means less to read, so **compression should make the
  model faster, not just smaller**.

**Right panel: prefill cost per 1,000 new tokens.** Only turns that read at least 200 new tokens are shown, because
smaller chunks give noisy numbers.

- Prefill cost rises roughly linearly, from about **200 ms** at the start to about **1,400–1,500 ms** at 40k–55k
  tokens (single points up to about 2,100 ms), about **7× more expensive**.
- **Meaning:** every new token must attend to everything already in memory. The spread comes from differing chunk
  sizes (a short `grep` result vs a whole file).

### Graph 3: GPU memory per recording (`exp1_per_recording.png`)

There is one panel per recording, all on the same scale, with the same colours as Graph 1 (right).

- **Blue (KV cache) and orange (allocated) are straight lines with the same slope in all 10 panels.** Recordings
  differ only in how far they extend.
- **Green (reserved) is never straight.** It stays flat and then climbs in steps. The three longest recordings
  (sphinx, sympy, django) reach the 13–14 GB limit, and sphinx and sympy then show the saw-tooth. Shorter ones
  (pytest, astropy, scikit-learn, flask) stay around 3–6 GB.
- Peak reserved memory does **not** follow context length neatly. requests-2931 reserved about 10 GB at only 22k
  tokens, while astropy-13398 stayed under 5 GB at 31k.
- **Meaning:** the real data grows the same predictable way everywhere. The wasted space depends on the **size and
  order of the tool outputs**, because large chunks need large temporary buffers, and it gets worse as the
  conversation gets longer.

### Graph 4: full recording vs the Part 0 method (`exp1_replay_vs_part0.png`)

Both lines come from the same recording (django-11265). Blue is the full recording (1a); orange is the Part 0
method (1b).

**Left panel: context growth** (x = turn, y = context length).

- The full recording starts about 2k tokens higher (the task, which the Part 0 method never includes) and stays
  above. It ends at **46,804 tokens vs 41,923**.
- Both lines have the same staircase shape and the same big jumps (around turns 57–68). Those turns are where the
  tool printed large files, which both methods include.
- **Meaning:** the Part 0 method holds about **12% less** than the real conversation. It drops the task (about 2,000
  tokens) and the agent's messages (about 8,900 tokens), and adds the model's own replies instead (about 6,000
  tokens, 14% of its memory).

**Right panel: decode speed** (x = context length, y = tokens/s).

- After the first ~2k tokens, **the two curves lie on top of each other** (15.5 vs 15.0 tok/s at 10k, 8.5 vs 8.8 at
  20k, 4.6 vs 4.6 at 40k).
- The Part 0 curve has one brief dip near 1.5k tokens (43 → 18 tok/s), a one-off warm-up artefact at the very start.
- **Meaning:** decode speed depends only on how long the context is, not on what is in it. The Part 0 method's
  **speed results are valid**, but its **memory numbers undercount** a real agent conversation.

| | 1a: full recording | 1b: Part 0 method |
|---|---|---|
| Turns | 95 | 94 |
| Final context | 46,804 | 41,923 |
| Tokens generated per turn | 44.5 (stops naturally) | 64 (always) |
| Decode tok/s at the end | 3.9 (at 46.8k) | 4.4 (at 41.9k) |

The end speeds differ only because 1a ends at a longer context; at the same length they match.

---

## 6. Experiment 2 results and graphs

**Colours used in the Experiment 2 graphs:**

| Colour | Method |
|---|---|
| Blue | Consensus, N=4 (proposed) |
| Orange | Single greedy guess (SpecKV-style) |
| Green | Past attention (SnapKV-style) |
| Yellow | Random |
| Pink, dotted | Oracle (uses the real next message) |
| Grey dashed | Everything 2-bit (nothing kept at full precision) |

### Summary numbers

**Damage, Δ NLL (lower is better), mean over 12 contexts.** Quantizing *everything* to 2-bit gives **+0.054**.

| Method | Quantize, keep 5% | 10% | 20% | 40% | Evict, keep 5% | 10% | 20% | 40% |
|---|---|---|---|---|---|---|---|---|
| Consensus N=1 | +0.037 | +0.014 | +0.023 | −0.000 | +0.519 | +0.323 | +0.142 | +0.057 |
| Consensus N=2 | +0.068 | +0.042 | +0.029 | −0.004 | +0.564 | +0.333 | +0.116 | +0.061 |
| **Consensus N=4** | +0.058 | +0.043 | **+0.007** | **−0.008** | **+0.496** | **+0.317** | +0.123 | +0.060 |
| Consensus N=8 | +0.068 | +0.022 | +0.017 | −0.006 | +0.475 | +0.292 | +0.100 | +0.061 |
| Single greedy guess | +0.058 | +0.032 | +0.026 | +0.010 | +0.609 | +0.337 | +0.173 | +0.054 |
| Past attention (SnapKV) | +0.047 | +0.028 | +0.033 | +0.035 | +0.928 | +0.628 | +0.333 | +0.148 |
| Random | +0.083 | +0.033 | +0.056 | +0.038 | +1.110 | +0.908 | +0.510 | +0.232 |
| Oracle | +0.051 | +0.030 | +0.002 | +0.006 | −0.005 | +0.001 | +0.002 | −0.003 |

**Top-1 agreement with the uncompressed model (higher is better).**

| Method | Quantize, keep 5% | 10% | 20% | 40% | Evict, keep 5% | 10% | 20% | 40% |
|---|---|---|---|---|---|---|---|---|
| Consensus N=4 | 93.5% | 92.9% | 95.2% | 97.3% | 80.7% | 85.6% | 92.4% | 96.8% |
| Single greedy guess | 92.3% | 94.1% | 95.4% | 96.1% | 78.9% | 85.9% | 92.4% | 95.7% |
| Past attention (SnapKV) | 91.8% | 91.5% | 92.7% | 94.8% | 72.6% | 79.4% | 85.9% | 92.9% |
| Random | 91.6% | 90.2% | 91.7% | 91.7% | 65.5% | 68.9% | 76.3% | 86.5% |
| Oracle | 93.0% | 95.2% | 96.7% | 98.2% | 92.9% | 93.8% | 96.2% | 98.2% |

### Graph 5: which tokens to keep? (`exp2_quality_vs_keep.png`), the main result


The x-axis is the % of middle tokens kept at full precision; the y-axis is Δ NLL. Error bars show mean ± standard
error over 12 contexts. **The two panels use very different y-scales** (about 0.1 vs about 1.2).

**Left panel: rest quantized to 2-bit (proposed approach).**

- All damage is small, between about −0.01 and +0.08 NLL, even with 95% of tokens at 2-bit.
- At keep 5–10%, the lines are tangled and the error bars overlap. No method clearly beats quantizing everything
  (+0.054).
- At keep **20–40%**, consensus N=4 drops to **+0.007 and −0.008**, matching the oracle (+0.002, +0.006). The single
  greedy guess is worse (+0.026, +0.010), SnapKV-style stays flat at about +0.03, and random is about +0.04–0.06.
- **Meaning:** quantization is gentle. With enough tokens kept, choosing them by consensus brings the damage
  down to about zero, as good as knowing the future.

**Right panel: rest evicted (H2O / SnapKV / SpecKV approach).**

- Damage is **10–50× larger**, up to +1.1 NLL.
- The ranking is clear and consistent at every keep level: **consensus < single greedy guess < SnapKV-style <
  random**. At keep 5%: 0.50 vs 0.61 vs 0.93 vs 1.11.
- The oracle stays at about 0, so a perfect choice of tokens exists. The methods differ in how well they find it.
- **Meaning:** when tokens are deleted, choosing well matters a lot. Guessing ahead does about **half the damage** of
  looking back, and several guesses beat one guess.

**Comparing the panels:** for consensus N=4 at keep 10%, quantizing does +0.043 damage vs +0.317 for evicting (about
**7× less**). At keep 20% it is +0.007 vs +0.123 (about **18× less**). This supports the central claim of the
proposal: **quantize instead of evicting**.

### Graph 6: do more draft guesses help? (`exp2_n_sweep.png`)

The x-axis is the number of sampled draft guesses N (1, 2, 4, 8); the y-axis is Δ NLL. **Here the line colours mean
keep levels, not methods:** blue = keep 5%, orange = keep 10%, green = keep 20%.

- **Rest evicted:** damage falls slightly and fairly steadily as N grows: 0.519 → 0.475 (keep 5%), 0.323 → 0.292
  (keep 10%), 0.142 → 0.100 (keep 20%). N=2 is a small bump, and N=8 is the best on every line.
- **Rest quantized:** no clear trend. The lines zig-zag in a tiny range (0.007–0.07), within the noise seen in
  Graph 5.
- **Meaning:** more guesses help when mistakes are costly (eviction). With quantization the damage is already near
  zero, so N barely matters. N = 4–8 is a sensible choice, which matches the proposal.

### Graph 7: how many of the truly needed tokens does each method find? (`exp2_recall.png`)

"Truly needed" means the oracle's top tokens, the ones the real next message attended to most. Each bar is the share
of those a method also picked.

| Method | top 5% | top 10% | top 20% | top 40% |
|---|---|---|---|---|
| Consensus N=4 | 43.4% | 52.3% | 64.4% | 75.3% |
| Consensus N=8 | 43.7% | 52.9% | 65.7% | 76.8% |
| Single greedy guess | 42.2% | 50.4% | 61.6% | 73.3% |
| Past attention (SnapKV) | 21.5% | 32.2% | 47.7% | 63.4% |
| Random | 4.8% | 9.5% | 19.9% | 40.4% |

- Random equals the keep fraction, as it should, which serves as a sanity check.
- Draft guesses find about **5× more** needed tokens than chance at top 10%, and about **2× more** than looking back at
  top 5%.
- Consensus is ahead of a single greedy guess in every group, but only by 1–3 points. Two sampled guesses share only
  57% of their top-10% tokens (Jaccard 0.57; consensus vs greedy 0.66), so averaging adds some coverage, but the gain
  is modest.
- **Meaning:** this is the clearest evidence that **looking ahead with draft guesses finds the important tokens**.

### Graph 8: where attention lands (`exp2_score_profile.png`)

This graph shows one context from sphinx-8548, about 5,000 tokens long. The x-axis is token position; the y-axis is
the importance score. Background colours: yellow = task, green = agent messages, pink = tool outputs. The top panel is
the consensus score (4 guesses); the bottom panel is the oracle score (the real next message). Sinks and the recent
window are left out.

- In **both** panels the spikes sit at the **start of each agent message** (green bands), and the largest spikes are
  near the end (the most recent agent messages).
- The long task description (first ~1,100 tokens) and most tool output score close to zero.
- The oracle's attention is more spread out. It also lands on the most recent tool outputs (positions ~4,600–5,000)
  and on scattered tokens inside tool output. Consensus is sharper and catches the main peaks but misses some of the
  spread, which matches the ~50% recall in Graph 7.
- **Meaning:** what the agent itself did recently matters most for its next step. Draft guesses point at the same
  places as the real future, and long, old tool output is the best candidate for compression.

### Token reports (`top_tokens_*.md`)

For three contexts, these reports show the draft guesses, the real next message, the 15 most and 15 least important
tokens, and which part of the conversation the top tokens come from.

| Recording | Agent messages: share of context | Agent messages: share of top 10% | Tool output: share of context | Tool output: share of top 10% |
|---|---|---|---|---|
| django-11265 | 8% | **40%** | 41% | 29% |
| sphinx-8548 | 10% | **41%** | 62% | 27% |
| sympy-12419 | 19% | **42%** | 48% | 26% |

- **Most important tokens:** the wording of earlier agent messages, e.g. "I will print lines 135 to 155 of …". The
  draft guesses copy that pattern ("I will view lines 810 to 820 of …"). The real agent sometimes does something
  different (e.g. searches with `grep`), but the guesses still point at the right earlier context: the same file and
  the same kind of action.
- **Least important tokens:** low-value boilerplate such as traceback file paths, long `tests/roots/...` file
  listings, and repeated `DeprecationWarning` lines. These are natural candidates for 2-bit storage.

---

## 7. Key findings

1. **The KV cache costs exactly 32 KB per token** for this model, on every recording. It is fully predictable and grows
   linearly; at 56k tokens it is 1.71 GB.
2. **Long agent sessions are slow.** Decoding gets up to **~13× slower** (79–92% slowdown) and reading new text about
   **7× more expensive** as the context grows from 2k to 50k+ tokens.
3. **PyTorch reserves far more than it uses:** up to about 14 GB reserved for about 4 GB of real data, because of
   fragmentation and temporary buffers.
4. **The Part 0 method undercounts real memory by about 12%,** but its speed measurements are valid, since speed
   depends only on context length.
5. **Draft guesses find the important tokens:** about 52% of the truly needed tokens vs 32% for looking back and 10%
   for random (top 10%).
6. **Several guesses beat one guess, modestly.** Under eviction, damage falls steadily from N=1 to N=8.
7. **Quantizing is 7–18× less damaging than evicting.** With 20–40% of tokens kept at full precision, consensus scoring
   brings the damage to about zero, matching the oracle.

---

## 9. Files in this folder

| Path | What it is |
|---|---|
| `Part1_KV_Cache_Experiment.ipynb` | The Colab notebook that writes all scripts, runs both experiments and shows every plot |
| `Part1_KV_Cache_Experiment_with_summaries.ipynb` | The same notebook with a written summary under each plot section |
| `kv_cache_results/RESULTS_EXPLAINED.md` | A longer, graph-by-graph explanation of the results |
| `kv_cache_results/exp1/replay_<id>.csv` | Experiment 1a, per-turn log for each recording |
| `kv_cache_results/exp1/part0_method_django__django-11265.csv` | Experiment 1b, per-turn log (Part 0 method) |
| `kv_cache_results/exp1/exp1_summary.md` / `.csv` | One row per recording (table in Section 5) |
| `kv_cache_results/exp1/exp1_replay_vs_part0.md` | 1a vs 1b comparison table |
| `kv_cache_results/exp1/*.png` | Graphs 1–4 |
| `kv_cache_results/exp2/metrics.csv` | Experiment 2, every method × policy × keep level × context |
| `kv_cache_results/exp2/agreement.csv` | Overlap between the guesses' top-token sets |
| `kv_cache_results/exp2/score_profiles.json` | Per-token scores behind Graph 8 |
| `kv_cache_results/exp2/config.json` | The exact settings used for Experiment 2 |
| `kv_cache_results/exp2/exp2_summary.md` | All Experiment 2 tables (Section 6) |
| `kv_cache_results/exp2/top_tokens_<id>.md` | Token reports for three contexts |
| `kv_cache_results/exp2/*.png` | Graphs 5–8 |
| `kv_cache_results.zip` | All of the above results in one archive |
