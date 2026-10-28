# Speculative KV Cache Quantization for Agentic Serving
## Project summary

The KV cache grows linearly with context length, which is a serious memory
bottleneck for long-horizon, multi-turn agentic LLM workloads. Hard eviction
(H2O, SnapKV) permanently drops tokens that may matter later; uniform
quantization (KIVI) is safer but not content-aware.

We propose a **speculative multi-rollout KV cache quantization** scheme:

1. When GPU memory pressure crosses a threshold, fork the current context to
   a lightweight draft model.
2. Run `N` parallel stochastic draft rollouts (`N ∈ [4, 8]`, lookahead
   `L = 10`) to sample diverse plausible continuations.
3. Score each context token by **consensus attention** across rollouts:
   `S_i = (1/N) * Σ_j max_t A^(j)_{t,i}`.
4. Keep high-consensus tokens (plus attention sinks / recent window) in
   FP16/BF16; quantize the rest to 4-bit or 2-bit.


## Repo layout

```
.
├── proposal/
│   ├── proposal.tex        
│   └── references.bib      
├── src/
│   ├── baseline_experiment.py   # Phase 0: profile vanilla HF KV cache under simulated agentic load
│   └── plot_baseline.py         # plots memory/latency vs. context length from the baseline CSV
├── data/
│   └── sample_tool_outputs.jsonl  # example trace schema — replace with real SWE-bench/BFCL traces
├── results/                # baseline_experiment.py output lands here (gitignored)
└── requirements.txt
```

## Phase 0: baseline

Profiles the default HuggingFace `transformers` KV cache (no optimizations)
under a simulated agentic loop: each turn appends a tool-output chunk to the
growing context (reusing `past_key_values` across turns), then generates the
next step. Logs per-turn context length, prefill/decode latency, and GPU
memory to CSV.

```bash
pip install -r requirements.txt

# smoke test with synthetic filler text — NOT for reported results
python src/baseline_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-turns 5 --gen-len 16


!python src/baseline_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tool-outputs data/real_single_task.jsonl \
    --num-turns 126 --gen-len 64 --out results/baseline_real.csv

!python src/plot_baseline.py --csv results/baseline_real.csv --out results/baseline_real_plot.png


!python src/baseline_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tool-outputs data/real_alfworld.jsonl \
    --num-turns 514 --gen-len 64 --out results/baseline_alfworld.csv
!python src/plot_baseline.py --csv results/baseline_alfworld.csv --out results/baseline_alfworld_plot.png


python src/plot_baseline.py --csv results/baseline.csv --out results/baseline_plot.png

!python src/baseline_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tool-outputs data/real_single_task_sympy.jsonl \
    --num-turns 108 --gen-len 64 --out results/baseline_sympy.csv
!python src/plot_baseline.py --csv results/baseline_sympy.csv --out results/baseline_sympy_plot.png

```

## Phase 1: real-trace baseline + consensus scoring

All Phase 1 code, the Colab notebook, results and graphs are in the `Phase1/` folder (see its README).

- **Exp 1:** Replayed 10 new SWE-bench trajectories (18k–56k tokens) with the full conversation in
  context (task + agent messages + tool outputs). The KV cache is 32 KB/token (1.7 GB at 56k tokens),
  decode drops from ~42 to ~3–9 tok/s, and reserved memory fills the GPU (~14 GB) while only ~4 GB is
  used. The Phase 0 method on the same trajectory gives ~12% less context at the same speed.
- **Exp 2:** Proof-of-concept of consensus scoring. Draft guesses find ~52% of the truly needed
  tokens (SnapKV-style 32%, random 10%). Quantizing the rest to 2-bit does 7–18× less damage than
  evicting it. With 4–8 guesses and the top 10–20% kept in FP16, quality loss is near zero at a
  ~3–5× smaller cache.
