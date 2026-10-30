"""Plots and go/no-go summary for the gate experiment.

    cd kv_manager && python -m experiments.gate.plot_gate

Writes pareto_B<block>.png (mean dNLL vs bytes reloaded) and summary.md to the results dir.

Go/no-go (fixed before running): a draft method (consensus, draft_window or specprefill) passes if, at
some block size, one of its configurations
  (a) moves no more bytes than uniform_int4 and at least halves its damage (mean dNLL
      <= 0.5 x uniform_int4's, paired 95% CI of the difference entirely below 0), or
  (b) moves <= 75% of uniform_int4's bytes and is non-inferior (CI upper bound < +0.01 nats).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRAFT_METHODS = ("consensus", "draft_window", "specprefill")
MARGIN = 0.01
BYTES_RATIO = 0.75


def paired_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results" / "gate"))
    args = p.parse_args()
    out = Path(args.results)
    df = pd.read_csv(out / "metrics.csv")
    df["hi"] = df["hi"].fillna("")
    key = ["instance_id", "turn"]

    lines = ["# Gate experiment summary", "",
             f"{df[key].drop_duplicates().shape[0]} samples. dNLL is in nats per target token "
             "(0 = identical to bf16 reload). bytes = reload bytes / bf16 reload bytes.", ""]
    verdicts = []
    for B, g in df.groupby("block_size"):
        u4 = g[g.method == "uniform_int4"].set_index(key)
        u4_bytes = u4.bytes_frac.mean()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (method, hi), m in g[g.hi != ""].groupby(["method", "hi"]):
            agg = m.groupby("keep_frac")[["bytes_frac", "delta_nll"]].mean()
            ax.plot(agg.bytes_frac, agg.delta_nll, marker="o", ls="-" if hi == "bf16" else "--",
                    label=f"{method} (hi={hi})", lw=2.2 if method in DRAFT_METHODS else 1.2)
        for name, mk in (("uniform_int4", "*"), ("uniform_int2", "X")):
            u = g[g.method == name]
            ax.scatter(u.bytes_frac.mean(), u.delta_nll.mean(), marker=mk, s=160, c="k", zorder=5, label=name)
        ax.axhline(0, c="grey", lw=0.8)
        ax.set_xlabel("bytes reloaded / bf16 bytes")
        ax.set_ylabel("mean ΔNLL of real next message")
        ax.set_title(f"KV reload quality vs bytes (block = {B} tokens)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"pareto_B{B}.png", dpi=150)
        plt.close(fig)

        lines += [f"## Block size {B}", "",
                  f"uniform_int4: bytes {u4_bytes:.3f}, dNLL {u4.delta_nll.mean():+.4f}, "
                  f"top-1 {u4.top1_agree.mean():.3f}  |  uniform_int2: bytes "
                  f"{g[g.method == 'uniform_int2'].bytes_frac.mean():.3f}, dNLL "
                  f"{g[g.method == 'uniform_int2'].delta_nll.mean():+.4f}", "",
                  "| method | hi | keep | bytes | dNLL | top-1 | recall | dNLL - uniform_int4 [95% CI] | passes |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for (method, hi, keep), m in g[g.hi != ""].groupby(["method", "hi", "keep_frac"]):
            m = m.set_index(key)
            diff = (m.delta_nll - u4.delta_nll.loc[m.index]).to_numpy()
            lo, hi_ci = paired_ci(diff)
            b = m.bytes_frac.mean()
            halves = m.delta_nll.mean() <= 0.5 * u4.delta_nll.mean()
            ok = ((b <= u4_bytes and hi_ci < 0 and halves)
                  or (b <= BYTES_RATIO * u4_bytes and hi_ci < MARGIN))
            if ok and method in DRAFT_METHODS:
                verdicts.append(f"{method} hi={hi} keep={keep:g} B={B}")
            lines.append(f"| {method} | {hi} | {keep:g} | {b:.3f} | {m.delta_nll.mean():+.4f} | "
                         f"{m.top1_agree.mean():.3f} | {m.recall_vs_oracle.mean():.2f} | "
                         f"{diff.mean():+.4f} [{lo:+.4f}, {hi_ci:+.4f}] | {'yes' if ok else ''} |")
        lines.append("")

        sp = g[g.method == "specprefill"].set_index(key + ["hi", "keep_frac"])
        cons = g[g.method == "consensus"].set_index(key + ["hi", "keep_frac"])
        if len(sp) and len(cons):
            lines += [f"Consensus vs SpecPrefill scoring (B={B}, negative = consensus better):", "",
                      "| hi | keep | dNLL consensus - specprefill [95% CI] |", "|---|---|---|"]
            d = (cons.delta_nll - sp.delta_nll.loc[cons.index]).reset_index()
            for (hi, keep), m in d.groupby(["hi", "keep_frac"]):
                lo, up = paired_ci(m.delta_nll.to_numpy())
                lines.append(f"| {hi} | {keep:g} | {m.delta_nll.mean():+.4f} [{lo:+.4f}, {up:+.4f}] |")
            lines.append("")

    lines += ["## Verdict", "",
              ("**GO**: draft-guided precision beats uniform int4 for: " + "; ".join(verdicts)) if verdicts
              else "**NO-GO**: no draft-guided configuration beats uniform int4 under the pre-set rule.", ""]
    s = pd.read_csv(out / "samples.csv")
    lines += ["## Draft scoring latency after tool return (HF eager, pessimistic)", "",
              f"draft_window: median {s.draft_window_ms.median():.0f} ms, "
              f"specprefill: median {s.specprefill_ms.median():.0f} ms, "
              f"consensus: median {s.consensus_ms.median():.0f} ms (history median {s.hist_len.median():.0f} tokens)", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
