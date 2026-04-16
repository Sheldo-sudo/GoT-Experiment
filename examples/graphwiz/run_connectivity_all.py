import argparse
import csv
import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure local repository source has higher priority than site-packages.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from graph_of_thoughts import controller, language_models

try:
    from . import utils
    from .graphwiz_got import (
        StrongStructuredGraphWizParser,
        StrongStructuredGraphWizPrompter,
        build_strong_structured_got,
    )
except ImportError:
    import utils
    from graphwiz_got import (
        StrongStructuredGraphWizParser,
        StrongStructuredGraphWizPrompter,
        build_strong_structured_got,
    )


def parse_data_ids(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def snapshot_lm_usage(lm) -> Dict[str, float]:
    """
    记录当前 LM 的累计 token / cost 计数。
    尽量兼容不同字段名。
    """
    prompt_tokens = getattr(lm, "prompt_tokens", 0) or 0
    completion_tokens = getattr(lm, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(lm, "response_tokens", 0) or 0

    cost = getattr(lm, "cost", 0.0) or 0.0

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost": float(cost),
    }


def extract_final_answer_from_executor(executor) -> str:
    """
    优先从 Controller.get_final_thoughts() 里拿最终答案。
    """
    try:
        final_groups = executor.get_final_thoughts()
        candidates: List[str] = []
        for group in final_groups:
            if not group:
                continue
            for thought in group:
                state = getattr(thought, "state", None)
                if isinstance(state, dict):
                    current = state.get("current", "")
                    if current:
                        candidates.append(current)
        if candidates:
            return candidates[-1]
    except Exception:
        pass
    return ""


def evaluate_sample(sample: Dict[str, Any], final_answer: str) -> bool:
    """
    复用主线当前的 GroundTruth 逻辑。
    """
    state = {
        "task": sample["task"],
        "original": sample["query"],
        "gold": sample["answer"],
        "current": final_answer,
    }
    return bool(utils.graphwiz_ground_truth(state))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_connectivity_all(
    budget: float,
    lm_name: str,
    source: str = "test",
    subset: str = "connectivity",
    max_samples: Optional[int] = None,
    data_ids: Optional[List[int]] = None,
    local_json_path: Optional[str] = None,
    data_root: str = "./data",
    prefer_local: bool = True,
) -> str:
    """
    跑完整个 connectivity 数据集，并统计：
    - solved / failed
    - prompt / completion / total tokens
    - cost
    """
    data = utils.load_graphwiz_samples(
        source=source,
        subset=subset,
        max_samples=max_samples,
        local_json_path=local_json_path,
        data_root=data_root,
        prefer_local=prefer_local,
    )

    if data_ids:
        selected_data = [data[i] for i in data_ids]
    else:
        selected_data = data

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(results_dir)

    run_dir = os.path.join(results_dir, f"connectivity_eval_{lm_name}_{timestamp}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "graphs"))

    logging.basicConfig(
        filename=os.path.join(run_dir, "run.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    lm_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )

    rows: List[Dict[str, Any]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    solved_count = 0
    failed_count = 0

    for idx, sample in enumerate(selected_data):
        if budget <= 0.0:
            logging.error("Budget depleted before sample %s", sample["id"])
            break

        logging.info("Running connectivity sample idx=%s sample_id=%s", idx, sample["id"])

        lm_builder = getattr(language_models, "create_language_model", None)
        if callable(lm_builder):
            lm = lm_builder(
                lm_config_path,
                model_name=lm_name,
                cache=True,
            )
        else:
            lm = language_models.ChatGPT(
                lm_config_path,
                model_name=lm_name,
                cache=True,
            )

        before = snapshot_lm_usage(lm)

        operations_graph = build_strong_structured_got("connectivity")

        executor = controller.Controller(
            lm,
            operations_graph,
            StrongStructuredGraphWizPrompter(),
            StrongStructuredGraphWizParser(),
            {
                "sample_id": sample["id"],
                "task": "connectivity",
                "original": sample["query"],
                "gold": sample["answer"],
                "current": "",
                "phase": 0,
                "part": "root",
                "branch_goal": "",
                "method": "strong_structured::connectivity",
                "meta": sample["meta"],
            },
        )

        try:
            executor.run()
        except Exception as e:
            logging.exception("Exception while running sample_id=%s: %s", sample["id"], e)

        output_json_path = os.path.join(run_dir, "graphs", f"{sample['id']}.json")
        executor.output_graph(output_json_path)

        after = snapshot_lm_usage(lm)

        prompt_delta = max(0, after["prompt_tokens"] - before["prompt_tokens"])
        completion_delta = max(0, after["completion_tokens"] - before["completion_tokens"])
        cost_delta = max(0.0, after["cost"] - before["cost"])
        total_delta = prompt_delta + completion_delta

        final_answer = extract_final_answer_from_executor(executor)
        solved = evaluate_sample(sample, final_answer)

        if solved:
            solved_count += 1
        else:
            failed_count += 1

        total_prompt_tokens += prompt_delta
        total_completion_tokens += completion_delta
        total_cost += cost_delta
        budget -= cost_delta

        rows.append(
            {
                "index": idx,
                "sample_id": sample["id"],
                "task": sample["task"],
                "solved": int(solved),
                "failed": int(not solved),
                "prompt_tokens": prompt_delta,
                "completion_tokens": completion_delta,
                "total_tokens": total_delta,
                "cost": round(cost_delta, 6),
                "final_answer": final_answer,
                "gold_answer": sample["answer"],
                "output_json": output_json_path,
            }
        )

    # 写逐样本 CSV
    csv_path = os.path.join(run_dir, "sample_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "sample_id",
                "task",
                "solved",
                "failed",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
                "final_answer",
                "gold_answer",
                "output_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # 写总 summary
    summary = {
        "source": source,
        "subset": subset,
        "lm_name": lm_name,
        "num_samples_run": len(rows),
        "solved_count": solved_count,
        "failed_count": failed_count,
        "accuracy": (solved_count / len(rows)) if rows else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_cost": round(total_cost, 6),
        "csv_path": csv_path,
        "results_dir": run_dir,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return run_dir


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()

    parser_.add_argument("--source", type=str, default="test", choices=["train", "test", "rft"])
    parser_.add_argument("--subset", type=str, default="connectivity")
    parser_.add_argument("--local_json_path", type=str, default="")
    parser_.add_argument("--data_root", type=str, default="./data")
    parser_.add_argument("--prefer_local", type=int, default=1)
    parser_.add_argument("--data_ids", type=str, default="")
    parser_.add_argument("--max_samples", type=int, default=0)
    parser_.add_argument("--budget", type=float, default=100.0)
    parser_.add_argument("--lm_name", type=str, default="chatgpt")

    args = parser_.parse_args()
    data_ids = parse_data_ids(args.data_ids)
    max_samples = args.max_samples if args.max_samples > 0 else None

    run_connectivity_all(
        budget=args.budget,
        lm_name=args.lm_name,
        source=args.source,
        subset=args.subset,
        max_samples=max_samples,
        data_ids=data_ids,
        local_json_path=args.local_json_path if args.local_json_path else None,
        data_root=args.data_root,
        prefer_local=bool(args.prefer_local),
    )