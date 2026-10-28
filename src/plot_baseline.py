"""
Plot results produced by baseline_experiment.py.

Usage:
    python plot_baseline.py --csv results/baseline.csv --out results/baseline_plot.png
"""

import argparse
import csv

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="baseline_plot.png")
    args = parser.parse_args()

    turns, context_lens, prefill_ms, decode_ms, mem_alloc, mem_reserved = [], [], [], [], [], []

    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turns.append(int(row["turn"]))
            context_lens.append(int(row["context_len"]))
            prefill_ms.append(float(row["prefill_ms"]))
            decode_ms.append(float(row["decode_ms"]))
            mem_alloc.append(float(row["mem_allocated_gb"]))
            mem_reserved.append(float(row["mem_reserved_gb"]))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(context_lens, mem_alloc, label="Allocated")
    axes[0].plot(context_lens, mem_reserved, label="Reserved", linestyle="--")
    axes[0].set_ylabel("GPU memory (GB)")
    axes[0].legend()
    axes[0].set_title("KV cache memory growth vs. context length")

    axes[1].plot(context_lens, prefill_ms)
    axes[1].set_ylabel("Prefill latency (ms)")
    axes[1].set_title("Prefill latency vs. context length")

    axes[2].plot(context_lens, decode_ms, color="orange")
    axes[2].set_ylabel("Decode latency (ms)")
    axes[2].set_xlabel("Context length (tokens)")
    axes[2].set_title("Decode latency vs. context length")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
