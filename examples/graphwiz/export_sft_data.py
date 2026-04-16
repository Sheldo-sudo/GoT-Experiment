import argparse
import json
import os
from typing import Any, Dict, List


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_input_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, "trajectories.jsonl")
        if os.path.exists(candidate):
            return candidate
    return path_or_dir


def export_sft(
    input_path: str,
    output_path: str,
    min_reward: float = 0.4,
    require_correct: bool = True,
) -> int:
    input_path = _resolve_input_path(input_path)
    trajectories = _load_jsonl(input_path)

    records = []
    for traj in trajectories:
        reward = float(traj.get("reward", 0.0))
        is_correct = bool(traj.get("is_correct", False))
        if reward < min_reward:
            continue
        if require_correct and not is_correct:
            continue

        prompt = traj.get("final_prompt") or traj.get("query", "")
        response = traj.get("final_response") or traj.get("final_answer", "")
        if not prompt or not response:
            continue

        records.append(
            {
                "instruction": str(prompt),
                "input": "",
                "output": str(response),
                "metadata": {
                    "sample_id": traj.get("sample_id"),
                    "task": traj.get("routed_task"),
                    "reward": reward,
                    "is_correct": is_correct,
                    "gold_answer": traj.get("gold_answer", ""),
                },
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SFT data from trajectories.")
    parser.add_argument("--input", type=str, required=True, help="run_dir or trajectories.jsonl")
    parser.add_argument("--output", type=str, required=True, help="output jsonl path")
    parser.add_argument("--min_reward", type=float, default=0.4)
    parser.add_argument("--require_correct", type=int, default=1)
    args = parser.parse_args()

    count = export_sft(
        input_path=args.input,
        output_path=args.output,
        min_reward=args.min_reward,
        require_correct=bool(args.require_correct),
    )
    print(f"Exported {count} SFT samples to {args.output}")
