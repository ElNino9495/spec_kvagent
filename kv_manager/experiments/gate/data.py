"""Real agent recordings for the gate experiment, cut into (history, fresh, target) samples.

Source: mini-SWE-agent runs on SWE-bench Verified (same recordings as Phase 1):
github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash

A sample mirrors one tool-call pause:
  history  everything up to and including the agent message that issued the tool call.
           This is the KV that sits offloaded on CPU during the pause.
  fresh    the tool output that just came back + the assistant header. Prefilled fresh
           in full precision on top of the reloaded history.
  target   the real next agent message, teacher-forced to measure damage.

    python -m experiments.gate.data            # download + convert all recordings
"""

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "swe"
TRAJ_REPO = "john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash"
RAW_URL = f"https://raw.githubusercontent.com/{TRAJ_REPO}/main"

# Longest recording from each of 10 projects (Phase 1, Experiment 1): 18k-56k tokens.
INSTANCES = [
    "sphinx-doc__sphinx-8548",
    "django__django-11265",
    "sympy__sympy-12419",
    "astropy__astropy-13398",
    "pytest-dev__pytest-5787",
    "pydata__xarray-4687",
    "mwaskom__seaborn-3069",
    "psf__requests-2931",
    "scikit-learn__scikit-learn-11310",
    "pallets__flask-5014",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())


def _agent_text(msg: dict) -> str:
    """The agent's sentence plus the shell command(s) it ran."""
    parts = [msg.get("content") or ""]
    for call in msg.get("tool_calls") or []:
        args = call.get("function", {}).get("arguments", "")
        try:
            cmd = json.loads(args).get("command", args)
        except (json.JSONDecodeError, AttributeError):
            cmd = args
        parts.append(f"```bash\n{cmd}\n```")
    return "\n\n".join(p for p in parts if p)


def prepare(instance_id: str, data_dir: Path = DATA_DIR) -> Path:
    raw = data_dir / "raw" / f"{instance_id}.traj.json"
    _download(f"{RAW_URL}/trajs/{instance_id}.traj.json", raw)
    traj = json.loads(raw.read_text(encoding="utf-8"))
    messages = []
    for m in traj["messages"]:
        role = m.get("role")
        if role in ("system", "user", "tool") and isinstance(m.get("content"), str):
            messages.append({"role": role, "content": m["content"]})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": _agent_text(m)})
    out = data_dir / "replay" / f"{instance_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"instance_id": instance_id, "messages": messages}), encoding="utf-8")
    return out


def load(instance_id: str, data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / "replay" / f"{instance_id}.json"
    if not path.exists():
        prepare(instance_id, data_dir)
    return json.loads(path.read_text(encoding="utf-8"))


# Llama-3 chat format; tool results use the "ipython" role.
def msg(role: str, content: str) -> str:
    role = "ipython" if role == "tool" else role
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"


ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>"


def conversation(traj: dict, bos: str, nonce: str = "") -> tuple[list[str], list[tuple[str, str, str]]]:
    """Returns (prefix pieces, [(agent message, tool output, next agent message), ...]).

    `nonce` is prepended to the system prompt so replayed copies of one recording do not
    share prefix-cache blocks.
    """
    msgs = traj["messages"]
    head = [m for m in msgs if m["role"] in ("system", "user")]
    prefix = [bos] + [msg(m["role"], (nonce if i == 0 else "") + m["content"]) for i, m in enumerate(head)]
    prefix.append(ASSISTANT_HEADER)
    rest = [m for m in msgs if m["role"] in ("assistant", "tool")]
    pairs, pending = [], None
    for m in rest:
        if m["role"] == "assistant":
            pending = m["content"]
        elif m["role"] == "tool" and pending is not None:
            pairs.append((pending, m["content"]))
            pending = None
    agents = [a for a, _ in pairs] + ([pending] if pending is not None else [])
    turns = [(a, t, agents[i + 1]) for i, (a, t) in enumerate(pairs) if i + 1 < len(agents)]
    return prefix, turns


def make_samples(tokenizer, traj: dict, hist_lens: list[int], max_target_tokens: int) -> list[dict]:
    """One sample per requested history length: the first tool-call pause whose history
    reaches that many tokens."""
    enc = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    prefix, turns = conversation(traj, tokenizer.bos_token or "")
    hist = [t for p in prefix for t in enc(p)]
    samples, wants = [], sorted(hist_lens)
    for k, (agent, tool, nxt) in enumerate(turns):
        hist = hist + enc(agent + EOT)
        if wants and len(hist) >= wants[0]:
            while wants and len(hist) >= wants[0]:
                wants.pop(0)
            samples.append({
                "instance_id": traj["instance_id"],
                "turn": k + 1,
                "hist_ids": list(hist),
                "fresh_ids": enc(msg("tool", tool) + ASSISTANT_HEADER),
                "target_ids": enc(nxt + EOT)[:max_target_tokens],
            })
        hist = hist + enc(msg("tool", tool) + ASSISTANT_HEADER)
        if not wants:
            break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=INSTANCES)
    args = parser.parse_args()
    for i in args.ids:
        print(f"{i}: {prepare(i)}")


if __name__ == "__main__":
    main()
