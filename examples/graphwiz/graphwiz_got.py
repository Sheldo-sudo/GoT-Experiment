import argparse
import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from graph_of_thoughts import controller, language_models, operations, parser, prompter

# 兼容从 examples 目录或仓库根目录执行
try:
    from . import utils
except ImportError:
    import utils


DECOMPOSE_PROMPT = """You are solving a graph reasoning problem.

Task name:
{task}

Original problem:
{query}

Break the original problem into exactly 2 self-contained subproblems that would help solve the original question.
The 2 subproblems must be useful, non-overlapping, and together should support the final answer.

Return ONLY valid JSON in the following format:
{{
  "Subproblem 1": "...",
  "Subproblem 2": "..."
}}
"""


SOLVE_SUBPROBLEM_PROMPT = """You are solving one subproblem of a graph reasoning task.

Task name:
{task}

Original problem:
{original_query}

Subproblem to solve:
{subproblem}

Please solve ONLY this subproblem.
Be concise but explicit.
End your response with a final sub-result in the format:
### <your final sub-result>
"""


AGGREGATE_PROMPT = """You are given an original graph reasoning problem and two solved subproblems.

Task name:
{task}

Original problem:
{original_query}

Subproblem analyses:
{subproblem_bundle}

Use the subproblem analyses to answer the ORIGINAL problem.
Be explicit but concise.
Your final line MUST be:
### <final answer>
"""


DIRECT_SOLVE_PROMPT = """You are solving a graph reasoning problem.

Task name:
{task}

Problem:
{query}

Solve the problem directly.
Your final line MUST be:
### <final answer>
"""


SUPPORTED_TASKS = {
    "connectivity",
    "cycle",
    "bipartite",
    "topology",
    "shortest_path",
    "flow",
    "triangle",
    "hamilton",
    "substructure",
}


def _canonical_task_name(task: str) -> str:
    if hasattr(utils, "canonical_task_name"):
        return utils.canonical_task_name(task)
    return str(task).strip().lower()


def route_task_name(task: str) -> str:
    """
    为 graphwiz_eval_generic.py 提供兼容接口。
    如果是支持的任务则返回规范化任务名，否则返回 generic。
    """
    task = _canonical_task_name(task)
    if task in SUPPORTED_TASKS:
        return task
    return "generic"


def _is_got_like_method(method: str) -> bool:
    """
    兼容两种调用方式：
    1. 旧版：got
    2. 评测脚本：strong_structured::<task>
    """
    if not method:
        return False
    return method.startswith("got") or method.startswith("strong_structured::")


class GraphWizPrompter(prompter.Prompter):
    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        assert len(state_dicts) >= 1, "aggregation_prompt requires at least one state"

        original_query = state_dicts[0]["original"]
        task = state_dicts[0].get("task", "unknown")

        # sort by "Subproblem 1", "Subproblem 2", ...
        def sort_key(s: Dict[str, Any]) -> str:
            return str(s.get("part", ""))

        chunks: List[str] = []
        for idx, st in enumerate(sorted(state_dicts, key=sort_key), start=1):
            part = st.get("part", f"Subproblem {idx}")
            subproblem = st.get("subproblem", "")
            current = st.get("current", "")
            chunks.append(
                f"{part}:\n"
                f"Question: {subproblem}\n"
                f"Answer:\n{current}\n"
            )

        subproblem_bundle = "\n".join(chunks)

        return AGGREGATE_PROMPT.format(
            task=task,
            original_query=original_query,
            subproblem_bundle=subproblem_bundle,
        )

    def generate_prompt(
        self,
        num_branches: int,
        original: str,
        current: str,
        task: str,
        method: str,
        phase: int,
        subproblem: str = "",
        **kwargs,
    ) -> str:
        if not _is_got_like_method(method):
            return DIRECT_SOLVE_PROMPT.format(task=task, query=original)

        if phase == 0:
            return DECOMPOSE_PROMPT.format(task=task, query=original)

        if phase == 1:
            return SOLVE_SUBPROBLEM_PROMPT.format(
                task=task,
                original_query=original,
                subproblem=subproblem,
            )

        # fallback direct solve / refine
        return DIRECT_SOLVE_PROMPT.format(task=task, query=original)

    def improve_prompt(self, **kwargs) -> str:
        return ""

    def validation_prompt(self, **kwargs) -> str:
        return ""

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        return ""


class GraphWizParser(parser.Parser):
    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> List[Dict]:
        new_states = []
        for text in texts:
            cleaned = utils.clean_response(text)
            if not cleaned:
                continue
            new_states.append(
                {
                    "current": cleaned,
                    "phase": 3,
                    "part": "final",
                }
            )
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states: List[Dict] = []

        phase = state.get("phase", 0)
        method = state.get("method", "")

        for text in texts:
            cleaned = utils.clean_response(text)

            # GoT first step: decompose into JSON
            if _is_got_like_method(method) and phase == 0:
                obj = utils.extract_first_json_object(cleaned)
                if obj is None or len(obj) < 2:
                    obj = utils.default_decomposition(
                        state.get("task", ""),
                        state.get("original", ""),
                    )

                # keep only first 2 items
                items = list(obj.items())[:2]
                for key, value in items:
                    if not isinstance(value, str):
                        value = str(value)
                    new_states.append(
                        {
                            "current": "",
                            "phase": 1,
                            "part": str(key).strip(),
                            "subproblem": value.strip(),
                        }
                    )
                continue

            # GoT second step: solve subproblem
            if _is_got_like_method(method) and phase == 1:
                if cleaned:
                    new_states.append(
                        {
                            "current": cleaned,
                            "phase": 2,
                        }
                    )
                continue

            # fallback
            if cleaned:
                new_states.append(
                    {
                        "current": cleaned,
                        "phase": phase + 1,
                    }
                )

        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        if not texts:
            return {}
        return {"current": utils.clean_response(texts[0])}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        return True

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        # not used in this integration because we use python scoring_function
        return [0.0 for _ in states]


# ===== 兼容 graphwiz_eval_generic.py 的导出接口 =====

class StrongStructuredGraphWizPrompter(GraphWizPrompter):
    """
    兼容前者接口名。
    实际逻辑仍复用当前通用 GoT Prompter。
    """
    pass


class StrongStructuredGraphWizParser(GraphWizParser):
    """
    兼容前者接口名。
    实际逻辑仍复用当前通用 GoT Parser。
    """
    pass


def build_strong_structured_got(task_name: str) -> operations.GraphOfOperations:
    """
    兼容前者接口名。
    当前后者是 task-agnostic 的通用 GoT 图，因此这里忽略 task_name，
    直接返回 got() 构建出的图。
    """
    _ = task_name
    return got()


def got() -> operations.GraphOfOperations:
    """
    A task-agnostic GoT graph:
    original problem -> decompose into 2 subproblems -> solve each ->
    aggregate -> final answer
    """
    graph = operations.GraphOfOperations()

    # Step 1: generate decomposition
    plans = operations.Generate(1, 1)
    graph.append_operation(plans)

    keep_nodes = []
    for i in range(1, 3):
        part_name = f"Subproblem {i}"

        selector = operations.Selector(
            lambda thoughts, part_name=part_name: [
                thought for thought in thoughts if thought.state.get("part") == part_name
            ]
        )
        selector.add_predecessor(plans)
        graph.add_operation(selector)

        solve_subproblem = operations.Generate(1, 3)
        solve_subproblem.add_predecessor(selector)
        graph.add_operation(solve_subproblem)

        score_subproblem = operations.Score(
            1, False, utils.graphwiz_format_score
        )
        score_subproblem.add_predecessor(solve_subproblem)
        graph.add_operation(score_subproblem)

        keep_best_subproblem = operations.KeepBestN(1, False)
        keep_best_subproblem.add_predecessor(score_subproblem)
        graph.add_operation(keep_best_subproblem)

        keep_nodes.append(keep_best_subproblem)

    # Step 2: aggregate subproblem answers
    aggregate = operations.Aggregate(3)
    for node in keep_nodes:
        node.add_successor(aggregate)
    graph.add_operation(aggregate)

    score_final = operations.Score(1, False, utils.graphwiz_format_score)
    score_final.add_predecessor(aggregate)
    graph.add_operation(score_final)

    keep_best_final = operations.KeepBestN(1, False)
    keep_best_final.add_predecessor(score_final)
    graph.add_operation(keep_best_final)

    ground_truth = operations.GroundTruth(utils.graphwiz_ground_truth)
    ground_truth.add_predecessor(keep_best_final)
    graph.add_operation(ground_truth)

    return graph


def parse_data_ids(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run(
    data_ids: Optional[List[int]],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
    source: str = "test",
    subset: Optional[str] = "connectivity",
    max_samples: Optional[int] = 20,
    local_json_path: Optional[str] = None,
) -> float:
    """
    Run GraphWiz data inside the GoT framework.
    """
    orig_budget = budget

    data = utils.load_graphwiz_samples(
        source=source,
        subset=subset,
        max_samples=max_samples,
        local_json_path=local_json_path,
    )

    if data_ids:
        selected_data = [data[i] for i in data_ids]
    else:
        selected_data = data

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    method_tag = "-".join([m.__name__ for m in methods])
    source_tag = source if not subset else f"{source}_{subset}"
    folder_name = f"{source_tag}_{lm_name}_{method_tag}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder, exist_ok=True)

    with open(os.path.join(results_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": source,
                "subset": subset,
                "max_samples": max_samples,
                "local_json_path": local_json_path,
                "num_selected_data": len(selected_data),
                "methods": [m.__name__ for m in methods],
                "lm_name": lm_name,
                "budget": budget,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__), exist_ok=True)

    lm_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )

    for sample in selected_data:
        logging.info("Running sample_id=%s task=%s", sample["id"], sample["task"])

        if budget <= 0.0:
            logging.error("Budget depleted. Stopping before sample %s", sample["id"])
            break

        for method in methods:
            if budget <= 0.0:
                logging.error("Budget depleted. Stop method loop.")
                break

            logging.info("Running method=%s budget_left=%s", method.__name__, budget)

            lm = language_models.ChatGPT(
                lm_config_path,
                model_name=lm_name,
                cache=True,
            )

            operations_graph = method()

            executor = controller.Controller(
                lm,
                operations_graph,
                GraphWizPrompter(),
                GraphWizParser(),
                {
                    "sample_id": sample["id"],
                    "task": sample["task"],
                    "original": sample["query"],
                    "gold": sample["answer"],
                    "current": "",
                    "phase": 0,
                    "part": "root",
                    "subproblem": "",
                    "method": method.__name__,
                    "meta": sample["meta"],
                },
            )

            try:
                executor.run()
            except Exception as e:
                logging.exception("Exception while running sample %s: %s", sample["id"], e)

            out_path = os.path.join(
                results_folder,
                method.__name__,
                f"{sample['id']}.json",
            )
            executor.output_graph(out_path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()

    parser_.add_argument(
        "--source",
        type=str,
        default="test",
        choices=["train", "test", "rft"],
        help="GraphInstruct source",
    )
    parser_.add_argument(
        "--subset",
        type=str,
        default="connectivity",
        help="subset/config name for GraphInstruct-Test; train/rft 时可忽略",
    )
    parser_.add_argument(
        "--local_json_path",
        type=str,
        default="",
        help="optional local json path; if set, it overrides remote HF loading",
    )
    parser_.add_argument(
        "--data_ids",
        type=str,
        default="",
        help="comma separated indices in the loaded sample list, e.g. 0,1,2",
    )
    parser_.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="maximum number of samples to load",
    )
    parser_.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="LM budget in dollars",
    )
    parser_.add_argument(
        "--lm_name",
        type=str,
        default="chatgpt",
        help="model name defined in GoT language model config",
    )
    parser_.add_argument(
        "--method",
        type=str,
        default="got",
        choices=["got"],
        help="currently this integration focuses on GoT",
    )

    args = parser_.parse_args()

    method_map = {
        "got": got,
    }

    data_ids = parse_data_ids(args.data_ids)
    methods = [method_map[args.method]]

    spent = run(
        data_ids=data_ids,
        methods=methods,
        budget=args.budget,
        lm_name=args.lm_name,
        source=args.source,
        subset=args.subset,
        max_samples=args.max_samples,
        local_json_path=args.local_json_path if args.local_json_path else None,
    )

    print(f"Finished. Spent budget: {spent}")