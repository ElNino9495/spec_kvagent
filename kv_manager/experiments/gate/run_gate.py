"""Gate experiment: does draft-guided mixed precision beat uniform 4-bit for KV reload?

Each sample is one tool-call pause from a real SWE-bench agent run (see data.py):
  1. Prefill the target's history KV (what sits offloaded on CPU during the pause).
  2. Build 2-bit and 4-bit versions of it with kvm.quant (the exact per-block system format).
  3. Score history blocks after the tool returns:
       consensus     draft prefills the tool output on its own resident history KV, then
                     N sampled rollouts of L tokens; S_i = mean_j max_t A_j[t, i]  (proposed)
       draft_window  draft attention from the last W fresh tokens (no rollouts, cheaper)
       specprefill   SpecPrefill's scoring (arXiv:2502.02789, reference code): 8 greedy lookahead
                     queries, softmax over prompt keys only, avg-pool 13, max over layers and
                     heads, mean over steps
       recency       newest blocks first
       random
       oracle        target attention from the real next message (upper bound, offline only)
  4. For every scheme, reload the history at mixed precision, prefill the tool output on top
     in bf16, teacher-force the real next agent message and compare with the bf16 reference.

Schemes (block 0 and the recent window are always bf16; the trailing partial block is
recomputed in bf16, as vLLM only offloads full blocks):
  uniform_int4 / uniform_int2        every middle block at that precision, no draft
  <method>, hi in {bf16, int4}, k    top-k fraction of middle blocks at `hi`, rest at int2

Outputs (in --out-dir): metrics.csv (sample x block size x scheme), samples.csv.

    cd kv_manager
    python -m experiments.gate.data                                   # download recordings
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_gate --max-samples 1 --hist-lens 4000  # smoke
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_gate        # full run
    python -m experiments.gate.plot_gate
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from experiments.gate import data
from kvm.quant import block_bytes, roundtrip_blocks

ROOT = Path(__file__).resolve().parents[2]
BITS = {"bf16": 16, "int4": 4, "int2": 2}


def load_model(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    return tok, model


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class AttnRecorder:
    """Forward hooks on every attention layer, reducing attention onto the first `ctx` keys.

    Switches the model to eager attention while active (sdpa does not return weights).
    take() returns, for everything run since the last take():
      mean  (batch, q_len, ctx)  head-averaged attention, averaged over layers (Phase 1)
      peak  (batch, ctx)         SpecPrefill's reduction for the last query: softmax over the
                                 `ctx` keys only, avg-pool, max over heads and layers
    """

    def __init__(self, model, ctx: int, pool: int = 13):
        self.model, self.ctx, self.pool = model, ctx, pool
        self.mean, self.peak, self.n, self.handles = None, None, 0, []

    def _hook(self, module, args, out):
        w = out[1][..., : self.ctx]
        m = w.mean(dim=1, dtype=torch.float32)
        self.mean = m if self.mean is None else self.mean + m
        last = w[:, :, -1].float()
        last = last / last.sum(dim=-1, keepdim=True)
        b, h, c = last.shape
        last = torch.nn.functional.avg_pool1d(last.reshape(b * h, 1, c), self.pool, stride=1,
                                              padding=self.pool // 2).reshape(b, h, c).amax(dim=1)
        self.peak = last if self.peak is None else torch.maximum(self.peak, last)
        self.n += 1

    def __enter__(self):
        self.model.set_attn_implementation("eager")
        self.handles = [layer.self_attn.register_forward_hook(self._hook)
                        for layer in self.model.model.layers]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.model.set_attn_implementation("sdpa")

    def take(self) -> tuple[torch.Tensor, torch.Tensor]:
        out = (self.mean / self.n, self.peak)
        self.mean, self.peak, self.n = None, None, 0
        return out


@torch.no_grad()
def prefill(model, ids: torch.Tensor, cache=None, chunk: int = 2048):
    out = None
    for s in range(0, ids.shape[1], chunk):
        out = model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True,
                    logits_to_keep=1)
        cache = out.past_key_values
    return cache, out.logits[:, -1].float()


@torch.no_grad()
def run_turn(model, layer_kv, n_layers, fresh, target, recorder=None):
    """Reload history KV (layer_kv(i) -> (k, v)), prefill the tool output in bf16, then
    teacher-force the real next message. Returns (log p of each target token, argmax)."""
    cache = DynamicCache()
    for i in range(n_layers):
        cache.update(*layer_kv(i), i)
    cache, last = prefill(model, fresh, cache)
    if recorder is not None:
        with recorder:
            out = model(input_ids=target, past_key_values=cache, use_cache=True)
    else:
        out = model(input_ids=target, past_key_values=cache, use_cache=True)
    logits = torch.cat([last, out.logits[0, :-1].float()])
    logp = torch.log_softmax(logits, dim=-1).gather(-1, target[0, :, None]).squeeze(-1)
    return logp, logits.argmax(dim=-1)


def _sample(logits, temperature, top_p, gen):
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    sorted_p[sorted_p.cumsum(dim=-1) - sorted_p > top_p] = 0
    choice = torch.multinomial(sorted_p / sorted_p.sum(-1, keepdim=True), 1, generator=gen)
    return idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def draft_scores(draft, hist, fresh, args, gen):
    """Token scores over the history from the draft, computed after the tool returns.
    The draft's history KV is resident in the real system, so only the work after the
    tool output arrives is timed."""
    P, F = hist.shape[1], fresh.shape[1]
    cache, _ = prefill(draft, hist)
    _sync(hist.device)
    t0 = time.perf_counter()

    w = min(args.window, F)
    if F > w:
        cache, _ = prefill(draft, fresh[:, :-w], cache)
    rec = AttnRecorder(draft, P + F)
    with rec:
        out = draft(input_ids=fresh[:, -w:], past_key_values=cache, use_cache=True, logits_to_keep=1)
    mean, peak = rec.take()
    window = mean[0, :, :P].sum(dim=0)
    _sync(hist.device)
    t_base = time.perf_counter() - t0
    base_kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]

    def fork(n):
        c = DynamicCache()
        for i, (k, v) in enumerate(base_kv):
            c.update(k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1), i)
        return c

    # SpecPrefill: last prompt token + (lookahead - 1) greedy tokens, one query each
    t1 = time.perf_counter()
    sp_steps, c1, logits = [peak[0]], fork(1), out.logits[:, -1].float()
    with rec:
        for _ in range(args.sp_lookahead - 1):
            o = draft(input_ids=logits.argmax(dim=-1)[:, None], past_key_values=c1, use_cache=True)
            c1, logits = o.past_key_values, o.logits[:, -1].float()
            sp_steps.append(rec.take()[1][0])
    specprefill = torch.stack(sp_steps).mean(dim=0)[:P]
    del c1
    _sync(hist.device)
    t_sp = time.perf_counter() - t1

    # Consensus (proposed): N sampled rollouts of L tokens
    t2 = time.perf_counter()
    branched, logits = fork(args.n), out.logits[:, -1].float().repeat(args.n, 1)
    steps = []
    with rec:
        for _ in range(args.horizon):
            tok = _sample(logits, args.temperature, args.top_p, gen)
            o = draft(input_ids=tok[:, None], past_key_values=branched, use_cache=True)
            branched, logits = o.past_key_values, o.logits[:, -1].float()
            steps.append(rec.take()[0][:, -1, :P])
    consensus = torch.stack(steps, dim=1).max(dim=1).values.mean(dim=0)
    _sync(hist.device)
    t_cons = time.perf_counter() - t2
    return ({"draft_window": window, "specprefill": specprefill, "consensus": consensus},
            {"draft_window_ms": 1e3 * t_base, "specprefill_ms": 1e3 * (t_base + t_sp),
             "consensus_ms": 1e3 * (t_base + t_cons)})


def top_blocks(block_scores, middle, frac):
    n = int(round(frac * middle.numel()))
    if n == 0:
        return middle[:0]
    return middle[block_scores[middle].topk(n).indices]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--draft", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--instances", nargs="*", default=data.INSTANCES)
    p.add_argument("--hist-lens", nargs="*", type=int, default=[8000, 16000, 24000, 32000])
    p.add_argument("--max-target-tokens", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=0, help="stop after this many samples (0 = all)")
    p.add_argument("--block-sizes", nargs="*", type=int, default=[16, 64])
    p.add_argument("--keep-fracs", nargs="*", type=float, default=[0.05, 0.10, 0.20, 0.40])
    p.add_argument("--hi", nargs="*", default=["bf16", "int4"], choices=["bf16", "int4"])
    p.add_argument("--recent", type=int, default=64, help="recent-window tokens always kept bf16")
    p.add_argument("--n", type=int, default=4, help="draft rollouts N")
    p.add_argument("--horizon", type=int, default=10, help="rollout length L")
    p.add_argument("--window", type=int, default=32, help="fresh tokens used by draft_window")
    p.add_argument("--sp-lookahead", type=int, default=8, help="SpecPrefill lookahead queries")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "gate"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, samples_path = out_dir / "metrics.csv", out_dir / "samples.csv"
    done = set()
    if args.resume and samples_path.exists():
        with open(samples_path) as f:
            done = {(r["instance_id"], int(r["turn"])) for r in csv.DictReader(f)}
    else:
        metrics_path.unlink(missing_ok=True)
        samples_path.unlink(missing_ok=True)

    print(f"loading {args.target} and {args.draft}")
    dev = args.device
    tok, target = load_model(args.target, dev)
    dtok, draft = load_model(args.draft, dev)
    assert tok.get_vocab() == dtok.get_vocab(), "target and draft must share a tokenizer"
    cfg = target.config
    n_layers, kv_heads = cfg.num_hidden_layers, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    n_done = 0
    for instance_id in args.instances:
        for s in data.make_samples(tok, data.load(instance_id), args.hist_lens, args.max_target_tokens):
            if args.max_samples and n_done >= args.max_samples:
                return
            if (instance_id, s["turn"]) in done:
                continue
            n_done += 1
            gen = torch.Generator(device=dev).manual_seed(args.seed + s["turn"])
            hist = torch.tensor([s["hist_ids"]], device=dev)
            fresh = torch.tensor([s["fresh_ids"]], device=dev)
            tgt = torch.tensor([s["target_ids"]], device=dev)
            P = hist.shape[1]
            print(f"\n{instance_id} turn {s['turn']}: history {P}, tool output {fresh.shape[1]}, "
                  f"target {tgt.shape[1]} tokens", flush=True)

            cache, _ = prefill(target, hist)
            K16 = [layer.keys for layer in cache.layers]
            V16 = [layer.values for layer in cache.layers]
            del cache

            rec = AttnRecorder(target, P)
            ref_logp, ref_pred = run_turn(target, lambda i: (K16[i], V16[i]), n_layers, fresh, tgt, rec)
            ref_nll = -ref_logp.mean().item()
            tok_scores = {"oracle": rec.take()[0][0].max(dim=0).values}
            ds, timing = draft_scores(draft, hist, fresh, args, gen)
            tok_scores.update(ds)
            print(f"  ref NLL {ref_nll:.3f} | draft scoring: window {timing['draft_window_ms']:.0f} ms, "
                  f"specprefill {timing['specprefill_ms']:.0f} ms, consensus {timing['consensus_ms']:.0f} ms",
                  flush=True)

            rows = []
            for B in args.block_sizes:
                nb = P // B
                n_recent = min(nb - 1, math.ceil(args.recent / B))
                middle = torch.arange(1, nb - n_recent, device=dev)
                if middle.numel() == 0:
                    continue
                K4 = [roundtrip_blocks(k, 4, B, "key") for k in K16]
                V4 = [roundtrip_blocks(v, 4, B, "value") for v in V16]
                K2 = [roundtrip_blocks(k, 2, B, "key") for k in K16]
                V2 = [roundtrip_blocks(v, 2, B, "value") for v in V16]

                blk = {m: sc[: nb * B].reshape(nb, B).sum(dim=-1) for m, sc in tok_scores.items()}
                blk["recency"] = torch.arange(nb, device=dev, dtype=torch.float32)
                blk["random"] = torch.rand(nb, device=dev, generator=gen)
                bf16_bytes = P * kv_heads * head_dim * 4 * n_layers

                def evaluate(block_bits):
                    tok_bits = torch.full((P,), 16, device=dev, dtype=torch.int16)
                    tok_bits[: nb * B] = block_bits.repeat_interleave(B)
                    m4 = (tok_bits == 4)[None, None, :, None]
                    m2 = (tok_bits == 2)[None, None, :, None]

                    def layer_kv(i):
                        k = torch.where(m2, K2[i], torch.where(m4, K4[i], K16[i]))
                        v = torch.where(m2, V2[i], torch.where(m4, V4[i], V16[i]))
                        return k, v

                    logp, pred = run_turn(target, layer_kv, n_layers, fresh, tgt)
                    nbytes = sum(block_bytes(int(b), B, kv_heads, head_dim) * int(c)
                                 for b, c in zip(*block_bits.unique(return_counts=True)))
                    nbytes = nbytes * n_layers + (P - nb * B) * kv_heads * head_dim * 4 * n_layers
                    return {"bytes_frac": nbytes / bf16_bytes,
                            "delta_nll": -logp.mean().item() - ref_nll,
                            "top1_agree": (pred == ref_pred).float().mean().item()}

                base = {"instance_id": instance_id, "turn": s["turn"], "hist_len": P,
                        "block_size": B, "ref_nll": ref_nll}
                for name, bits in (("uniform_int4", 4), ("uniform_int2", 2)):
                    bb = torch.full((nb,), 16, device=dev, dtype=torch.int16)
                    bb[middle] = bits
                    rows.append({**base, "method": name, "hi": "", "keep_frac": 0.0,
                                 "recall_vs_oracle": "", **evaluate(bb)})

                for frac in args.keep_fracs:
                    oracle_set = set(top_blocks(blk["oracle"], middle, frac).tolist())
                    for method, sc in blk.items():
                        chosen = top_blocks(sc, middle, frac)
                        recall = (len(set(chosen.tolist()) & oracle_set) / len(oracle_set)
                                  if oracle_set else float("nan"))
                        for hi in args.hi:
                            bb = torch.full((nb,), 16, device=dev, dtype=torch.int16)
                            bb[middle] = 2
                            bb[chosen] = BITS[hi]
                            rows.append({**base, "method": method, "hi": hi, "keep_frac": frac,
                                         "recall_vs_oracle": round(recall, 4), **evaluate(bb)})
                del K4, V4, K2, V2

                u4 = next(r for r in rows if r["block_size"] == B and r["method"] == "uniform_int4")
                c = [r for r in rows if r["block_size"] == B and r["method"] == "consensus"
                     and r["hi"] == "bf16" and r["keep_frac"] == 0.10]
                if c:
                    print(f"  B={B}: uniform_int4 dNLL {u4['delta_nll']:+.4f} @ {u4['bytes_frac']:.2f} | "
                          f"consensus bf16 10% dNLL {c[0]['delta_nll']:+.4f} @ {c[0]['bytes_frac']:.2f}",
                          flush=True)

            _append(metrics_path, rows)
            _append(samples_path, [{"instance_id": instance_id, "turn": s["turn"], "hist_len": P,
                                    "fresh_len": fresh.shape[1], "target_len": tgt.shape[1],
                                    "ref_nll": ref_nll, **timing}])
            del K16, V16
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
