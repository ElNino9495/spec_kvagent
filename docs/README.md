# spec_kvagent documentation

**spec_kvagent** is a research project on serving LLM *agents* efficiently. It targets the
**KV cache**, the per-token memory a language model keeps while it works. When an agent pauses for a
tool call, our system parks that cache in CPU memory. When the tool returns, a small *draft* model
predicts which parts of the history the next step will need. Those parts are reloaded at full
precision and everything else at 2-bit. No token is ever thrown away, and the system is a plain
plugin for [vLLM](https://github.com/vllm-project/vllm) 0.27.1: no fork, no custom GPU kernels.

If you are new here, read the pages in this order:

| # | Page | What you will learn |
|---|---|---|
| 1 | [The problem](01-problem.md) | What the KV cache is, why agents make it a bottleneck, and why existing fixes fall short |
| 2 | [The idea](02-idea.md) | How the system works turn by turn, and why it is designed this way |
| 3 | [Goals](03-goals.md) | Research questions, hypotheses, success criteria, non-goals and risks |
| 4 | [Phases](04-phases.md) | What has been done, what each phase found, and what comes next |
| 5 | [Benchmarks](05-benchmarks.md) | Datasets, models, metrics, baselines and how every experiment is run |
| 6 | [Architecture](06-architecture.md) | How the system plugs into vLLM, data formats and correctness rules |
| 7 | [Related work](07-related-work.md) | How this differs from H2O, SnapKV, KIVI, SpecKV, SpecPrefill, TriForce, Continuum and others |
| – | [Glossary](glossary.md) | Every term and abbreviation used in these docs |

## Status at a glance

| Phase | Question | Status |
|---|---|---|
| 0. Baseline profiling | Is the KV cache really a problem for agents? | Done |
| 1. Real traces and scoring proof of concept | Can a draft model find the tokens that matter? Is quantizing better than evicting? | Done |
| 2. Gate experiment | Does draft-guided precision beat plain uniform 4-bit? | Built, not run yet |
| M1. vLLM system, no draft | Offload at tool pauses and reload at uniform 16, 4 or 2 bits | Built, CPU-tested only |
| M2. vLLM system with draft | Draft scoring inside vLLM and load-then-upgrade | Planned |
| M3. Evaluation | Concurrency sweeps, ablations, quality measured inside vLLM | Planned |

Details are in [Phases](04-phases.md).

## Repository map

| Path | Contents |
|---|---|
| [`proposal/`](../proposal/) | The original course proposal (LaTeX), which describes the first version of the idea |
| [`src/`](../src/), [`data/`](../data/), [`results/`](../results/) | Phase 0: baseline profiling scripts, traces and results |
| [`Phase_1/`](../Phase_1/) | Phase 1: Colab notebook, results archive and a detailed write-up |
| [`kv_manager/`](../kv_manager/) | The current system and experiments (vLLM connector, gate experiment, replay benchmark, environment). Its [README](../kv_manager/README.md) has the commands |
| [`docs/`](./) | These pages |
| [`TEMP.md`](../TEMP.md) | Working notes on TriForce and offloading, from which the current direction grew |

## Where to start, by role

- **Want the idea in five minutes?** Read [The problem](01-problem.md) and the first two sections of [The idea](02-idea.md).
- **Running experiments?** Read [Benchmarks](05-benchmarks.md), then follow [`kv_manager/README.md`](../kv_manager/README.md).
- **Changing the code?** Read [Architecture](06-architecture.md). Run the CPU tests with
  `cd kv_manager && python -m pytest tests -q`.
- **Writing the paper?** Read [Goals](03-goals.md), [Phases](04-phases.md) and [Related work](07-related-work.md).
