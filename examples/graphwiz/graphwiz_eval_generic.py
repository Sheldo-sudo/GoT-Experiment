import argparse
import csv
import datetime
import importlib
import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from graph_of_thoughts import controller, language_models

try:
    from . import utils
    from .graphwiz_got import (
        StrongStructuredGraphWizParser,
        StrongStructuredGraphWizPrompter,
        build_strong_structured_got,
        route_task_name,
    )
except ImportError:
    import utils
    from graphwiz_got import (
        StrongStructuredGraphWizParser,
        StrongStructuredGraphWizPrompter,
        build_strong_structured_got,
        route_task_name,
    )


TASK_MODULE_CANDIDATES = {
    "connectivity": ["connectivity_task"],
    "bipartite": ["bipartite_task"],
    "cycle": ["cycle_task"],
    "topology": ["topology_task"],
    "shortest_path": ["shortest_path"],
    "flow": ["flow_task"],
    "triangle": ["triangle_task"],
    "hamilton": ["hamilton_task"],
    "substructure": ["substructure_task"],
}


def parse_data_ids(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def snapshot_lm_usage(lm) -> Dict[str, float]:
    """
    尽量兼容不同语言模型实现上的字段名。
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
    优先从 Controller.get_final_thoughts() 中抽取最终答案。
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


def build_eval_state(sample: Dict[str, Any], final_answer: str, routed_task: str) -> Dict[str, Any]:
    return {
        "task": routed_task,
        "original": sample["query"],
        "gold": sample["answer"],
        "current": final_answer,
    }


def import_task_module(task_name: str):
    """
    懒加载各 task 文件，避免顶层强依赖。
    """
    task_name = route_task_name(task_name)
    candidates = TASK_MODULE_CANDIDATES.get(task_name, [])

    for module_name in candidates:
        try:
            return importlib.import_module(f".{module_name}", package=__package__)
        except Exception:
            pass
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass

    return None


def _infer_task_from_text(text: str) -> str:
    """
    当 sample['task'] 不可靠时，从文本中兜底猜任务类型。
    尽量按更具体的任务优先匹配，避免误判到 connectivity / generic。
    """
    t = (text or "").strip().lower()
    if not t:
        return "generic"

    # 更具体的关键词优先
    if "topology sorting" in t or "topological sorting" in t or "topological ordering" in t:
        return "topology"
    if "shortest path" in t:
        return "shortest_path"
    if "maximum flow" in t or ("source" in t and "sink" in t and "capacity" in t and "flow" in t):
        return "flow"
    if "hamiltonian" in t:
        return "hamilton"
    if "bipartite" in t:
        return "bipartite"
    if "subgraph" in t or "substructure" in t or "pattern graph" in t:
        return "substructure"
    if "triangle" in t:
        return "triangle"
    if "cycle" in t:
        return "cycle"

    # connectivity 放后面，避免把 shortest path / topology 误吸进去
    if "connectivity" in t or "connected" in t or "path between" in t:
        return "connectivity"

    return "generic"


def infer_routed_task(
    sample: Dict[str, Any],
    subset: Optional[str] = None,
    local_json_path: Optional[str] = None,
) -> str:
    """
    多级任务识别：
    1. subset（最可靠，比如 --subset topology）
    2. sample['task']
    3. meta.dataset_name
    4. local_json_path
    5. query 文本关键词兜底

    这样即使数据里 task 被写成 generic，也能正确分发到专用任务模块。
    """
    # 1) subset 最可靠
    if subset:
        routed = route_task_name(subset)
        if routed != "generic":
            return routed

    # 2) 样本自带 task
    raw_task = str(sample.get("task", "")).strip()
    routed = route_task_name(raw_task)
    if routed != "generic":
        return routed

    # 3) meta.dataset_name
    meta = sample.get("meta", {}) or {}
    dataset_name = str(meta.get("dataset_name", "")).strip().lower()
    guessed = _infer_task_from_text(dataset_name)
    if guessed != "generic":
        return guessed

    # 4) local_json_path
    local_path_text = str(local_json_path or "").strip().lower()
    guessed = _infer_task_from_text(local_path_text)
    if guessed != "generic":
        return guessed

    # 5) query 文本兜底
    query = str(sample.get("query", "")).strip().lower()
    guessed = _infer_task_from_text(query)
    if guessed != "generic":
        return guessed

    return "generic"


def get_task_runtime(task_name: str):
    """
    返回 (operations_graph, prompter_obj, parser_obj)
    优先使用任务专属实现；失败时退回 generic strong_structured。
    """
    module = import_task_module(task_name)
    if module is not None:
        build_graph_fn = getattr(module, "build_graph", None)
        get_prompter_fn = getattr(module, "get_prompter", None)
        get_parser_fn = getattr(module, "get_parser", None)

        if callable(build_graph_fn) and callable(get_prompter_fn) and callable(get_parser_fn):
            return build_graph_fn(), get_prompter_fn(), get_parser_fn()

    return (
        build_strong_structured_got(task_name),
        StrongStructuredGraphWizPrompter(),
        StrongStructuredGraphWizParser(),
    )


def get_task_ground_truth_fn(task_name: str) -> Callable[[Dict[str, Any]], bool]:
    """
    优先使用各任务自己的 ground_truth。
    如果导入失败，再退回 utils.graphwiz_ground_truth。
    """
    module = import_task_module(task_name)
    if module is not None:
        gt_fn = getattr(module, "ground_truth", None)
        if callable(gt_fn):
            return gt_fn
    return lambda state: bool(utils.graphwiz_ground_truth(state))


def evaluate_sample(
    sample: Dict[str, Any],
    final_answer: str,
    routed_task: Optional[str] = None,
) -> bool:
    """
    用任务专属 GT 判定正确性。
    """
    routed_task = route_task_name(
        routed_task or sample.get("routed_task") or sample.get("task", "")
    )
    state = build_eval_state(sample, final_answer, routed_task)
    gt_fn = get_task_ground_truth_fn(routed_task)

    try:
        return bool(gt_fn(state))
    except Exception:
        return bool(utils.graphwiz_ground_truth(state))


def init_task_stats() -> Dict[str, Any]:
    return {
        "num_samples_run": 0,
        "correct_count": 0,
        "wrong_count": 0,
        "accuracy": 0.0,
        "accuracy_percent": "0.00%",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }


def build_controller_state(sample: Dict[str, Any], routed_task: str) -> Dict[str, Any]:
    return {
        "sample_id": sample["id"],
        "task": routed_task,
        "original": sample["query"],
        "gold": sample["answer"],
        "current": "",
        "phase": 0,
        "part": "root",
        "branch_goal": "",
        "method": f"strong_structured::{routed_task}",
        "meta": sample.get("meta", {}),
    }


def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _convert_to_per_token(cost_value: float, unit: str) -> float:
    unit = (unit or "").strip().lower()
    if unit in {"per_token", "token", "1"}:
        return float(cost_value)
    if unit in {"per_1k", "per_1000", "1k"}:
        return float(cost_value) / 1000.0
    if unit in {"per_1m", "per_1000000", "1m"}:
        return float(cost_value) / 1_000_000.0
    # 默认按字段名语义：token_cost -> per_token
    return float(cost_value)


def load_model_pricing(config_path: str, model_name: str) -> Dict[str, float]:
    """
    从 language_models/config.json 中读取单价，返回每 token 单价。
    支持：
    - prompt_token_cost / response_token_cost
    - input_token_cost / output_token_cost
    - prompt_cost_per_1k / completion_cost_per_1k
    - pricing_unit: per_token / per_1k / per_1m
    """
    if not os.path.exists(config_path):
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    if not isinstance(cfg, dict):
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    block = cfg.get(model_name)
    if not isinstance(block, dict):
        models_block = cfg.get("models")
        if isinstance(models_block, dict):
            block = models_block.get(model_name)

    if not isinstance(block, dict):
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    unit = str(block.get("pricing_unit", "per_token"))

    prompt_token_cost = _first_not_none(
        block.get("prompt_token_cost"),
        block.get("input_token_cost"),
    )
    completion_token_cost = _first_not_none(
        block.get("response_token_cost"),
        block.get("completion_token_cost"),
        block.get("output_token_cost"),
    )

    if prompt_token_cost is not None or completion_token_cost is not None:
        return {
            "prompt_per_token": _convert_to_per_token(float(prompt_token_cost or 0.0), unit),
            "completion_per_token": _convert_to_per_token(float(completion_token_cost or 0.0), unit),
        }

    prompt_cost_per_1k = _first_not_none(
        block.get("prompt_cost_per_1k"),
        block.get("input_cost_per_1k"),
    )
    completion_cost_per_1k = _first_not_none(
        block.get("response_cost_per_1k"),
        block.get("completion_cost_per_1k"),
        block.get("output_cost_per_1k"),
    )

    if prompt_cost_per_1k is not None or completion_cost_per_1k is not None:
        return {
            "prompt_per_token": float(prompt_cost_per_1k or 0.0) / 1000.0,
            "completion_per_token": float(completion_cost_per_1k or 0.0) / 1000.0,
        }

    return {"prompt_per_token": 0.0, "completion_per_token": 0.0}


def estimate_cost_from_tokens(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Dict[str, float],
) -> float:
    return (
        prompt_tokens * float(pricing.get("prompt_per_token", 0.0)) +
        completion_tokens * float(pricing.get("completion_per_token", 0.0))
    )


def run_graphwiz_eval(
    budget: float,
    lm_name: str,
    source: str = "test",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
    data_ids: Optional[List[int]] = None,
    local_json_path: Optional[str] = None,
    data_root: str = "./data",
    prefer_local: bool = True,
    use_cache: bool = False,
) -> str:
    """
    通用版 GraphWiz 统计脚本：
    - 支持任意 subset
    - 支持混合 task 自动路由
    - 统计 token / cost / accuracy
    - 与 graphwiz_got.py 的兼容接口保持一致
    """
    initial_budget = float(budget)

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

    source_tag = source if not subset else f"{source}_{subset}"
    cache_tag = "cache_on" if use_cache else "cache_off"
    run_dir = os.path.join(results_dir, f"{source_tag}_eval_{lm_name}_{cache_tag}_{timestamp}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "graphs"))

    logging.basicConfig(
        filename=os.path.join(run_dir, "run.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": source,
                "subset": subset,
                "max_samples": max_samples,
                "local_json_path": local_json_path,
                "data_root": data_root,
                "prefer_local": prefer_local,
                "num_selected_data": len(selected_data),
                "lm_name": lm_name,
                "initial_budget": initial_budget,
                "use_cache": use_cache,
                "mode": "graphwiz_eval_generic",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lm_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )
    pricing = load_model_pricing(lm_config_path, lm_name)

    rows: List[Dict[str, Any]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    correct_count = 0
    wrong_count = 0

    task_stats: Dict[str, Dict[str, Any]] = defaultdict(init_task_stats)

    for idx, sample in enumerate(selected_data):
        if budget <= 0.0:
            logging.error("Budget depleted before sample %s", sample["id"])
            break

        raw_task = str(sample.get("task", "")).strip()
        routed_task = infer_routed_task(
            sample,
            subset=subset,
            local_json_path=local_json_path,
        )
        sample["routed_task"] = routed_task

        logging.info(
            "Running idx=%s sample_id=%s raw_task=%s routed_task=%s budget_left=%s",
            idx,
            sample["id"],
            raw_task,
            routed_task,
            budget,
        )

        ensure_dir(os.path.join(run_dir, "graphs", routed_task))

        lm = language_models.ChatGPT(
            lm_config_path,
            model_name=lm_name,
            cache=use_cache,
        )

        before = snapshot_lm_usage(lm)

        operations_graph, task_prompter, task_parser = get_task_runtime(routed_task)

        logging.info(
            "Dispatch runtime sample_id=%s routed_task=%s graph=%s prompter=%s parser=%s",
            sample["id"],
            routed_task,
            type(operations_graph).__name__,
            type(task_prompter).__name__,
            type(task_parser).__name__,
        )

        executor = controller.Controller(
            lm,
            operations_graph,
            task_prompter,
            task_parser,
            build_controller_state(sample, routed_task),
        )

        error_message = ""
        try:
            executor.run()
        except Exception as e:
            error_message = str(e)
            logging.exception(
                "Exception while running sample_id=%s raw_task=%s routed_task=%s: %s",
                sample["id"],
                raw_task,
                routed_task,
                e,
            )

        output_json_path = os.path.join(
            run_dir,
            "graphs",
            routed_task,
            f"{sample['id']}.json",
        )

        try:
            executor.output_graph(output_json_path)
        except Exception as e:
            logging.exception("Failed to output graph for sample_id=%s: %s", sample["id"], e)

        after = snapshot_lm_usage(lm)

        prompt_delta = max(0, after["prompt_tokens"] - before["prompt_tokens"])
        completion_delta = max(0, after["completion_tokens"] - before["completion_tokens"])
        raw_cost_delta = max(0.0, after["cost"] - before["cost"])
        total_delta = prompt_delta + completion_delta

        if raw_cost_delta > 0.0:
            cost_delta = raw_cost_delta
            cost_source = "lm.cost"
        else:
            cost_delta = estimate_cost_from_tokens(
                prompt_tokens=prompt_delta,
                completion_tokens=completion_delta,
                pricing=pricing,
            )
            cost_source = "token_fallback"

        final_answer = extract_final_answer_from_executor(executor)
        is_correct = evaluate_sample(sample, final_answer, routed_task=routed_task)

        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        total_prompt_tokens += prompt_delta
        total_completion_tokens += completion_delta
        total_cost += cost_delta
        budget -= cost_delta

        task_stats[routed_task]["num_samples_run"] += 1
        task_stats[routed_task]["correct_count"] += int(is_correct)
        task_stats[routed_task]["wrong_count"] += int(not is_correct)
        task_stats[routed_task]["prompt_tokens"] += prompt_delta
        task_stats[routed_task]["completion_tokens"] += completion_delta
        task_stats[routed_task]["total_tokens"] += total_delta
        task_stats[routed_task]["total_cost"] += cost_delta

        rows.append(
            {
                "index": idx,
                "sample_id": sample["id"],
                "raw_task": raw_task,
                "routed_task": routed_task,
                "subset": subset if subset is not None else "",
                "is_correct": int(is_correct),
                "is_wrong": int(not is_correct),
                "prompt_tokens": prompt_delta,
                "completion_tokens": completion_delta,
                "total_tokens": total_delta,
                "cost": round(cost_delta, 6),
                "cost_source": cost_source,
                "final_answer": final_answer,
                "gold_answer": sample["answer"],
                "output_json": output_json_path,
                "error_message": error_message,
            }
        )

        logging.info(
            "Finished sample_id=%s correct=%s prompt_tokens=%s completion_tokens=%s cost=%s cost_source=%s budget_left=%s",
            sample["id"],
            is_correct,
            prompt_delta,
            completion_delta,
            cost_delta,
            cost_source,
            budget,
        )

    for task_name, stats in task_stats.items():
        n = stats["num_samples_run"]
        acc = (stats["correct_count"] / n) if n else 0.0
        stats["accuracy"] = acc
        stats["accuracy_percent"] = f"{acc * 100:.2f}%"
        stats["total_cost"] = round(stats["total_cost"], 6)

    csv_path = os.path.join(run_dir, "sample_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "sample_id",
                "raw_task",
                "routed_task",
                "subset",
                "is_correct",
                "is_wrong",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
                "cost_source",
                "final_answer",
                "gold_answer",
                "output_json",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    task_summary_path = os.path.join(run_dir, "task_summary.json")
    with open(task_summary_path, "w", encoding="utf-8") as f:
        json.dump(task_stats, f, ensure_ascii=False, indent=2)

    task_csv_path = os.path.join(run_dir, "task_summary.csv")
    with open(task_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "num_samples_run",
                "correct_count",
                "wrong_count",
                "accuracy",
                "accuracy_percent",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "total_cost",
            ],
        )
        writer.writeheader()
        for task_name, stats in sorted(task_stats.items()):
            writer.writerow(
                {
                    "task": task_name,
                    "num_samples_run": stats["num_samples_run"],
                    "correct_count": stats["correct_count"],
                    "wrong_count": stats["wrong_count"],
                    "accuracy": stats["accuracy"],
                    "accuracy_percent": stats["accuracy_percent"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["total_tokens"],
                    "total_cost": stats["total_cost"],
                }
            )

    num_samples_run = len(rows)
    overall_accuracy = (correct_count / num_samples_run) if num_samples_run else 0.0

    summary = {
        "source": source,
        "subset": subset,
        "lm_name": lm_name,
        "use_cache": use_cache,
        "num_samples_selected": len(selected_data),
        "num_samples_run": num_samples_run,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "accuracy": overall_accuracy,
        "accuracy_percent": f"{overall_accuracy * 100:.2f}%",
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "initial_budget": round(initial_budget, 6),
        "spent_budget": round(total_cost, 6),
        "remaining_budget": round(budget, 6),
        "total_cost": round(total_cost, 6),
        "csv_path": csv_path,
        "task_summary_path": task_summary_path,
        "task_csv_path": task_csv_path,
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
    parser_.add_argument("--subset", type=str, default="")
    parser_.add_argument("--local_json_path", type=str, default="")
    parser_.add_argument("--data_root", type=str, default="./data")
    parser_.add_argument("--prefer_local", type=int, default=1)
    parser_.add_argument("--data_ids", type=str, default="")
    parser_.add_argument("--max_samples", type=int, default=0)
    parser_.add_argument("--budget", type=float, default=100.0)
    parser_.add_argument("--lm_name", type=str, default="chatgpt")
    parser_.add_argument("--use_cache", type=int, default=0)

    args = parser_.parse_args()

    data_ids = parse_data_ids(args.data_ids)
    max_samples = args.max_samples if args.max_samples > 0 else None
    subset = args.subset if args.subset else None

    run_graphwiz_eval(
        budget=args.budget,
        lm_name=args.lm_name,
        source=args.source,
        subset=subset,
        max_samples=max_samples,
        data_ids=data_ids,
        local_json_path=args.local_json_path if args.local_json_path else None,
        data_root=args.data_root,
        prefer_local=bool(args.prefer_local),
        use_cache=bool(args.use_cache),
    )