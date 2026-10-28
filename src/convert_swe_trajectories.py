"""
Convert real mini-SWE-agent trajectory files (from the SWE-bench "experiments"
leaderboard repo) into the JSONL schema baseline_experiment.py expects.

Each trajectory is a JSON file with a "messages" list; role=="tool" messages
contain the actual shell/tool output the agent saw at each step (e.g. grep
results, file contents, test run output, patch application results). This
script pulls those out, in order, across one or more trajectory files, and
writes them as one real_tool_outputs.jsonl with the same {"tool_output": "..."}
schema as data/sample_tool_outputs.jsonl -- so it's a drop-in replacement, but
with authentic agent tool-call text instead of synthetic filler.

Source of the trajectories (real submissions to the SWE-bench Verified
leaderboard, mini-SWE-agent format, Gemini 3.5 Flash):
    https://github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash

To get more / different trajectories, browse other submissions in
    https://github.com/swe-bench/experiments
(look at evaluation/<split>/<date>_<model>/metadata.yaml -> assets.trajs for a
submission whose trajectories are hosted directly on GitHub rather than S3)
and point --traj-dir at a cloned copy of its trajs/ folder.

Usage:
    python src/convert_swe_trajectories.py \
        --traj-dir /tmp/traj-sample/trajs \
        --num-trajectories 15 \
        --out data/real_tool_outputs.jsonl
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, required=True,
                         help="Directory of *.traj.json trajectory files.")
    parser.add_argument("--num-trajectories", type=int, default=1,
                         help="How many trajectory files to concatenate. "
                              "Default is 1 -- a single, coherent, real agent "
                              "session solving one real issue. Only raise "
                              "this if you deliberately want to simulate one "
                              "long-lived session handling several unrelated "
                              "tasks back to back (and say so in your report).")
    parser.add_argument("--instance-id", type=str, default=None,
                         help="Pick one specific trajectory by its instance id "
                              "(the filename without .traj.json), e.g. "
                              "django__django-10554. Overrides --num-trajectories.")
    parser.add_argument("--longest", action="store_true",
                         help="Instead of taking the first N files alphabetically, "
                              "pick the N longest trajectories (most tool-call "
                              "turns), which gives you more usable data per real task.")
    parser.add_argument("--out", type=str, default="data/real_tool_outputs.jsonl")
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.traj_dir, "*.traj.json")))
    if not all_files:
        raise FileNotFoundError(f"No *.traj.json files found in {args.traj_dir}")

    if args.instance_id:
        files = [f for f in all_files
                  if os.path.basename(f) == f"{args.instance_id}.traj.json"]
        if not files:
            raise FileNotFoundError(f"No trajectory found for instance_id={args.instance_id}")
    elif args.longest:
        def tool_msg_count(path):
            with open(path) as fh:
                d = json.load(fh)
            return sum(1 for m in d.get("messages", []) if m.get("role") == "tool")
        files = sorted(all_files, key=tool_msg_count, reverse=True)[: args.num_trajectories]
    else:
        files = all_files[: args.num_trajectories]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_written = 0
    with open(args.out, "w") as f_out:
        for path in files:
            with open(path) as f_in:
                traj = json.load(f_in)
            for msg in traj.get("messages", []):
                if msg.get("role") != "tool":
                    continue
                content = msg.get("content")
                if not content or not isinstance(content, str):
                    continue
                f_out.write(json.dumps({"tool_output": content}) + "\n")
                n_written += 1

    print(f"Wrote {n_written} real tool-output chunks from "
          f"{len(files)} trajectories to {args.out}")


if __name__ == "__main__":
    main()
