"""Replay recorded SWE-bench agent runs as N concurrent agents against a vLLM server.

Each agent replays one recording turn by turn. A turn sends the whole conversation so far, as
token ids, to /v1/completions, streams the reply and records the time to first token. The next
turn is teacher-forced with the real agent message and tool output from the recording, so every
system under test sees identical contexts; generated tokens are only used for timing. Between
turns the agent sleeps for a simulated tool delay, during which its KV can be evicted or offloaded.

    cd kv_manager
    python -m bench.replay --url http://localhost:8000 --agents 16 --out results/replay/kvm_int2.jsonl

Quality is not measured here: vLLM recomputes the whole prompt whenever prompt_logprobs is
requested (skip_reading_prefix_cache), so it would never see the reloaded KV. Reload quality
comes from experiments/gate, which uses the same kvm.quant format.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from experiments.gate import data

ROOT = Path(__file__).resolve().parents[1]


def build_agent(tok, instance_id: str, agent: int) -> tuple[list[int], list[list[int]]]:
    """(prefix ids, [ids appended before each turn's request, ...])."""
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    prefix, turns = data.conversation(data.load(instance_id), tok.bos_token or "", nonce=f"[agent {agent}] ")
    steps = [[]] + [enc(agent_msg + data.EOT) + enc(data.msg("tool", tool) + data.ASSISTANT_HEADER)
                    for agent_msg, tool, _ in turns]
    return [t for p in prefix for t in enc(p)], steps


async def timed_request(session, url, body) -> dict:
    t0 = time.perf_counter()
    ttft, n_chunks, usage = None, 0, None
    async with session.post(f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
    out = {"ttft": ttft, "e2e": time.perf_counter() - t0, "chunks": n_chunks}
    if usage:
        out["gen_tokens"] = usage.get("completion_tokens")
        out["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return out


async def run_agent(i, args, tok, model, session, out, rng):
    instance = args.instances[i % len(args.instances)]
    ids, steps = build_agent(tok, instance, i)
    await asyncio.sleep(rng.uniform(0, args.stagger))
    for turn, append in enumerate(steps[: args.max_turns]):
        ids = ids + append
        if len(ids) + args.gen_tokens > args.max_context:
            break
        if turn > 0:
            delay = args.tool_delay_min + rng.expovariate(1 / args.tool_delay_mean)
            await asyncio.sleep(delay)
        else:
            delay = 0.0
        body = {"model": model, "temperature": 0.0}
        if args.tag:
            body["kv_transfer_params"] = {"tag": f"a{i}-t{turn}"}
        rec = {"agent": i, "instance": instance, "turn": turn, "prompt_tokens": len(ids),
               "tool_delay": delay, "t_send": time.time()}
        body.update(prompt=ids, max_tokens=args.gen_tokens, ignore_eos=True, stream=True,
                    stream_options={"include_usage": True})
        rec.update(await timed_request(session, args.url, body))
        out.write(json.dumps(rec) + "\n")
        out.flush()


def summarize(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    resumed = [r for r in rows if r["turn"] > 0]
    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")
    lines = [f"{len(rows)} requests from {len({r['agent'] for r in rows})} agents"]
    if rows and "ttft" in rows[0]:
        t = [r["ttft"] for r in resumed if r.get("ttft") is not None]
        span = max(r["t_send"] + r["e2e"] for r in rows) - min(r["t_send"] for r in rows)
        lines += [f"TTFT after tool return: p50 {pct(t, .5):.3f}s  p90 {pct(t, .9):.3f}s  "
                  f"p99 {pct(t, .99):.3f}s  mean {statistics.fmean(t):.3f}s" if t else "no resumed turns",
                  f"wall {span:.1f}s, {len(rows) / span:.2f} req/s, "
                  f"{sum(r['prompt_tokens'] for r in rows) / span:.0f} prompt tok/s"]
    return "\n".join(lines)


async def main_async(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model
        if model is None:
            async with session.get(f"{args.url}/v1/models") as r:
                model = (await r.json())["data"][0]["id"]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as out:
            rng = random.Random(args.seed)
            seeds = [rng.randrange(2**31) for _ in range(args.agents)]
            await asyncio.gather(*(run_agent(i, args, tok, model, session, out, random.Random(seeds[i]))
                                   for i in range(args.agents)))
    print(summarize(out_path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default=None, help="served model name (default: first from /v1/models)")
    p.add_argument("--tokenizer", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--instances", nargs="*", default=data.INSTANCES)
    p.add_argument("--max-turns", type=int, default=1000)
    p.add_argument("--max-context", type=int, default=65536, help="stop an agent past this many tokens")
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--tool-delay-mean", type=float, default=2.0, help="mean of the exponential part (s)")
    p.add_argument("--tool-delay-min", type=float, default=0.5, help="constant part of the delay (s)")
    p.add_argument("--stagger", type=float, default=10.0, help="agents start uniformly within this window (s)")
    p.add_argument("--tag", action="store_true", help="send kv_transfer_params tags (AgentKVConnector only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results" / "replay" / "run.jsonl"))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
