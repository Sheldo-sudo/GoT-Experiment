import argparse
import json
import os
from typing import Any, Dict, List


def _resolve_input_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, "trajectories.jsonl")
        if os.path.exists(candidate):
            return candidate
    return path_or_dir


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def export_rl(
    input_path: str,
    output_path: str,
    include_step_rewards: bool = True,
) -> int:
    path = _resolve_input_path(input_path)
    trajectories = _load_jsonl(path)

    records = []
    for traj in trajectories:
        prompt = traj.get("final_prompt") or traj.get("query", "")
        response = traj.get("final_response") or traj.get("final_answer", "")
        if not prompt or not response:
            continue

        rec = {
            "prompt": str(prompt),
            "response": str(response),
            "reward": float(traj.get("reward", 0.0)),
            "metadata": {
                "sample_id": traj.get("sample_id"),
                "task": traj.get("routed_task"),
                "is_correct": bool(traj.get("is_correct", False)),
                "reward_components": traj.get("reward_components", {}),
                "token_usage": traj.get("token_usage", {}),
            },
        }
        if include_step_rewards:
            rec["step_rewards"] = traj.get("step_rewards", [])
        records.append(rec)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RL/GRPO data from trajectories.")
    parser.add_argument("--input", type=str, required=True, help="run_dir or trajectories.jsonl")
    parser.add_argument("--output", type=str, required=True, help="output jsonl path")
    parser.add_argument("--include_step_rewards", type=int, default=1)
    args = parser.parse_args()

    count = export_rl(
        input_path=args.input,
        output_path=args.output,
        include_step_rewards=bool(args.include_step_rewards),
    )
    print(f"Exported {count} RL samples to {args.output}")
