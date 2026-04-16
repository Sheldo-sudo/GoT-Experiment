import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def _resolve_input_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, "trajectories.jsonl")
        if os.path.exists(candidate):
            return candidate
    return path_or_dir


def _load_many_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        path = _resolve_input_path(p)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def export_preference_pairs(
    input_paths: List[str],
    output_path: str,
    min_reward_gap: float = 0.1,
) -> int:
    trajectories = _load_many_jsonl(input_paths)
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for traj in trajectories:
        key = str(traj.get("sample_id"))
        by_sample[key].append(traj)

    records = []
    for _, group in by_sample.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: float(x.get("reward", 0.0)))
        rejected = group[0]
        chosen = group[-1]
        chosen_reward = float(chosen.get("reward", 0.0))
        rejected_reward = float(rejected.get("reward", 0.0))
        if chosen_reward - rejected_reward < min_reward_gap:
            continue

        prompt = chosen.get("final_prompt") or chosen.get("query", "")
        chosen_text = chosen.get("final_response") or chosen.get("final_answer", "")
        rejected_text = rejected.get("final_response") or rejected.get("final_answer", "")
        if not prompt or not chosen_text or not rejected_text:
            continue

        records.append(
            {
                "instruction": str(prompt),
                "input": "",
                "chosen": str(chosen_text),
                "rejected": str(rejected_text),
                "metadata": {
                    "sample_id": chosen.get("sample_id"),
                    "task": chosen.get("routed_task"),
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                },
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export preference pairs from trajectories.")
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="comma-separated run_dir or trajectories.jsonl paths",
    )
    parser.add_argument("--output", type=str, required=True, help="output jsonl path")
    parser.add_argument("--min_reward_gap", type=float, default=0.1)
    args = parser.parse_args()

    inputs = [x.strip() for x in args.inputs.split(",") if x.strip()]
    count = export_preference_pairs(
        input_paths=inputs,
        output_path=args.output,
        min_reward_gap=args.min_reward_gap,
    )
    print(f"Exported {count} preference pairs to {args.output}")
