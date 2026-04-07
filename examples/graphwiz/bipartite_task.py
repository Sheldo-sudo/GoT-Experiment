from typing import Any, Dict, List

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "bipartite"
METHOD_NAME = "structured::bipartite"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "TwoColorAttempt",
        "goal": "Construct a two-coloring / two-partition explanation if the graph is bipartite.",
        "num_generate": 4,
        "keep_n": 1,
    },
    {
        "part": "OddCycleSearch",
        "goal": "Search for an odd-cycle witness that would prove the graph is not bipartite.",
        "num_generate": 4,
        "keep_n": 1,
    },
]


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for bipartite checking.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "TwoColorAttempt":
        return f"""You are solving the TwoColorAttempt branch.

Original problem:
{original_query}

Goal:
Construct a two-coloring or two-partition explanation if possible.

End with:
### <branch conclusion>
"""
    return f"""You are solving the OddCycleSearch branch.

Original problem:
{original_query}

Goal:
Search for an odd-cycle witness that would prove the graph is not bipartite.

End with:
### <branch conclusion>
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for bipartite checking.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

If there is a valid two-coloring explanation and no concrete odd-cycle witness, answer Yes.
If there is a concrete odd-cycle witness, answer No.

Your final line must be exactly one of:
### Yes
### No
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the final answer for bipartite checking.

Original problem:
{original_query}

Previous answer:
{current}

Your final line must be exactly one of:
### Yes
### No
"""


def search_score(state: Dict[str, Any]) -> float:
    text = utils.clean_response(state.get("current", ""))
    part = state.get("part", "")
    low = text.lower()

    if not text:
        return 100.0

    score = 0.0
    if "###" not in text:
        score += 8.0
    if len(text) < 8:
        score += 10.0

    if part == "TwoColorAttempt":
        if not any(k in low for k in ["color", "partition", "set a", "set b", "two-color", "two color"]):
            score += 14.0
    elif part == "OddCycleSearch":
        if not any(k in low for k in ["odd cycle", "triangle", "cycle", "not bipartite"]):
            score += 14.0
    elif part == "final":
        if not utils.validate_yesno_response(state):
            score += 25.0

    score += min(len(text) / 3000.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    return utils.validate_yesno_response(state)


def ground_truth(state: Dict[str, Any]) -> bool:
    truth = utils.graph_bipartite_truth(state.get("original", ""))
    pred = utils.extract_yes_no(state.get("current", ""))
    if truth is not None and pred is not None:
        return (pred == "yes") == truth
    return utils.graphwiz_ground_truth(state)


def build_graph():
    return build_task_graph(
        branches=BRANCHES,
        search_score_fn=search_score,
        final_validator=final_validator,
        ground_truth_fn=ground_truth,
        aggregate_responses=3,
    )


def get_prompter():
    return BaseTaskPrompter(
        task_name=TASK_NAME,
        phase0_prompt_fn=phase0_prompt,
        branch_prompt_fn=branch_prompt,
        aggregate_prompt_fn=aggregate_prompt,
        improve_prompt_fn=improve_prompt,
    )


def get_parser():
    return BaseTaskParser(BRANCHES)