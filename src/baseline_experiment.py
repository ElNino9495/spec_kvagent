"""
Baseline KV cache profiling under simulated agentic load.

Runs a manual turn loop: each turn appends a "tool output" chunk to a growing
context (reusing the model's KV cache across turns via past_key_values), then
generates the model's next step. Logs per-turn context length, prefill/decode
latency, and GPU memory usage.

Usage:
    python baseline_experiment.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --tool-outputs sample_tool_outputs.jsonl \
        --num-turns 50 --gen-len 64 --out results/baseline.csv

If --tool-outputs is omitted, synthetic filler text is used instead. This is
only for smoke-testing that the harness runs end to end — do NOT use
synthetic-text results in the report. Source real trace chunks (e.g. from
SWE-bench / mini-swe-agent / BFCL trajectories) before collecting real numbers.

Optional: pass --mem-cap-gb to stop the loop once allocated memory crosses a
threshold, simulating a memory-constrained serving environment on hardware
where a single session wouldn't otherwise hit real pressure.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tool_outputs(path: str, num_turns: int) -> list[str]:
    """Load tool-output text chunks from a JSONL trace file.

    Expected format: one JSON object per line with a "tool_output" (or
    "text" / "content") field holding the chunk's raw text. Adapt the key
    name here if your trace source uses a different schema.
    """
    chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("tool_output") or obj.get("text") or obj.get("content")
            if text:
                chunks.append(text)
    if len(chunks) < num_turns:
        raise ValueError(
            f"Trace file only has {len(chunks)} chunks but --num-turns={num_turns}. "
            "Provide more trace data or reduce --num-turns."
        )
    return chunks[:num_turns]


def synthetic_tool_outputs(num_turns: int, tokenizer, approx_tokens: int = 300) -> list[str]:
    """Fallback filler text generator for smoke-testing only.

    NOT representative of real agent workloads -- prefer --tool-outputs with
    real trace data for any numbers that go into the report.
    """
    base = (
        "Reading file src/module.py lines 1-120. Found function definitions: "
        "parse_config, load_data, run_pipeline. No syntax errors detected. "
        "Test suite output: 12 passed, 1 failed (test_edge_case). "
    )
    n_repeats = max(1, approx_tokens // max(1, len(tokenizer.encode(base))))
    chunk = base * n_repeats
    return [chunk for _ in range(num_turns)]


def get_mem_stats():
    return {
        "mem_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "mem_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="HF model id or local path, e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--tool-outputs", type=str, default=None,
                         help="Path to a JSONL file of real tool-output chunks. "
                              "If omitted, synthetic filler text is used (smoke-test only).")
    parser.add_argument("--num-turns", type=int, default=50)
    parser.add_argument("--gen-len", type=int, default=64,
                         help="Max number of tokens to generate per turn (the decode step).")
    parser.add_argument("--mem-cap-gb", type=float, default=None,
                         help="If set, stop the loop once memory_allocated exceeds this many GB.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                         choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out", type=str, default="results/baseline.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This script expects a CUDA GPU to collect meaningful memory stats.")

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    print(f"Loading model {args.model} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    if args.tool_outputs:
        tool_outputs = load_tool_outputs(args.tool_outputs, args.num_turns)
    else:
        print("WARNING: no --tool-outputs provided; using synthetic filler text. "
              "Fine for smoke-testing the harness, NOT for results you report.")
        tool_outputs = synthetic_tool_outputs(args.num_turns, tokenizer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "turn", "context_len", "chunk_len", "gen_len",
        "prefill_ms", "decode_ms", "decode_tok_per_s",
        "mem_allocated_gb", "mem_reserved_gb",
    ]

    past_key_values = None
    context_len = 0

    torch.cuda.reset_peak_memory_stats()

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for turn in range(args.num_turns):
            chunk_text = tool_outputs[turn]
            chunk_ids = tokenizer(chunk_text, return_tensors="pt").input_ids.to(device)
            chunk_len = chunk_ids.shape[1]

            # --- PREFILL: process the new tool-output chunk against the carried-over cache ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(
                    input_ids=chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000.0

            past_key_values = out.past_key_values
            context_len += chunk_len
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            del out 

            # --- DECODE: generate the model's next step, one token at a time ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            generated_tokens = 0
            with torch.no_grad():
                for _ in range(args.gen_len):
                    step_out = model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = step_out.past_key_values
                    next_token = step_out.logits[:, -1:, :].argmax(dim=-1)
                    generated_tokens += 1
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t0) * 1000.0
            context_len += generated_tokens

            decode_tok_per_s = generated_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
            mem = get_mem_stats()

            row = {
                "turn": turn,
                "context_len": context_len,
                "chunk_len": chunk_len,
                "gen_len": generated_tokens,
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "decode_tok_per_s": round(decode_tok_per_s, 2),
                "mem_allocated_gb": round(mem["mem_allocated_gb"], 3),
                "mem_reserved_gb": round(mem["mem_reserved_gb"], 3),
            }
            writer.writerow(row)
            f_out.flush()
            print(
                f"turn {turn:3d} | ctx={context_len:6d} | "
                f"prefill={prefill_ms:7.1f}ms | decode={decode_ms:7.1f}ms "
                f"({decode_tok_per_s:5.1f} tok/s) | "
                f"mem_alloc={mem['mem_allocated_gb']:.2f}GB mem_reserved={mem['mem_reserved_gb']:.2f}GB"
            )

            if args.mem_cap_gb is not None and mem["mem_allocated_gb"] >= args.mem_cap_gb:
                print(f"Memory cap of {args.mem_cap_gb} GB reached at turn {turn}. Stopping.")
                break

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nDone. Peak memory allocated: {peak_gb:.2f} GB. Results written to {out_path}")


if __name__ == "__main__":
    main()
