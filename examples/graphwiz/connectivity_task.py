from typing import Any, Dict, List

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "connectivity"
METHOD_NAME = "structured::connectivity"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "PathWitness",
        "goal": "Find one explicit path witness from the source node to the target node if connectivity holds.",
        "num_generate": 4,
        "keep_n": 1,
    },
    {
        "part": "SeparationCheck",
        "goal": "Look only for concrete disconnection evidence such as different connected components, unreachable expansion, or isolation.",
        "num_generate": 4,
        "keep_n": 1,
    },
]


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for connectivity.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "PathWitness":
        return f"""You are solving the PathWitness branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Find ONE explicit path witness from the source node to the target node if connectivity holds.

Rules:
1. Focus only on finding an explicit path witness.
2. Prefer a short, concrete node sequence such as 9 -> 0 -> 8 or [9, 0, 8].
3. Do not discuss disconnection cases in this branch.
4. If you find a path, the final line must be:
### Path: [a->b->c]
5. If you cannot find an explicit path, the final line must be:
### No explicit path found
"""
    return f"""You are solving the SeparationCheck branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Look ONLY for concrete disconnection evidence.

Rules:
1. Do NOT give a path witness in this branch.
2. Do NOT conclude "No" unless you have concrete evidence such as different connected components, an isolated side, or unreachable expansion.
3. If you find concrete disconnection evidence, the final line must be:
### Disconnection evidence: <brief reason>
4. If you do NOT find concrete disconnection evidence, the final line must be:
### No disconnection evidence
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given multiple branch analyses from a strictly structured Graph-of-Thought workflow.

Task:
connectivity

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. If the PathWitness branch contains any explicit path witness such as [a->b->c] or a clear node sequence from source to target, you MUST answer Yes.
2. Only answer No if there is NO explicit path witness and the SeparationCheck branch provides concrete disconnection evidence.
3. "No disconnection evidence" is NOT evidence for No.
4. When branches conflict, trust an explicit path witness over speculative disconnection analysis.

Your final line must be exactly one of:
### Yes
### No
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""You previously produced an invalid or weak final answer for connectivity.

Original problem:
{original_query}

Previous answer:
{current}

Repair the answer.

Your final line must be exactly one of:
### Yes
### No
"""


def search_score(state: Dict[str, Any]) -> float:
    task = utils.canonical_task_name(state.get("task", ""))
    part = state.get("part", "")
    text = utils.clean_response(state.get("current", ""))
    low = text.lower()

    if not text:
        return 100.0

    score = 0.0
    if "###" not in text:
        score += 8.0
    if len(text) < 8:
        score += 10.0
    if len(text) > 3000:
        score += 6.0

    if task == "connectivity":
        if part == "PathWitness":
            seq = utils.extract_sequence_from_text(text)
            yn = utils.extract_yes_no(text)
            if seq is None:
                score += 18.0
            if yn not in {None, "yes"}:
                score += 12.0
            if "no explicit path found" in low or "no path" in low:
                score += 16.0

        elif part == "SeparationCheck":
            if utils.extract_sequence_from_text(text) is not None:
                score += 12.0

            has_disconnection_phrase = any(
                k in low for k in [
                    "disconnection evidence",
                    "different component",
                    "different connected component",
                    "isolated",
                    "unreachable",
                    "cannot reach",
                    "no disconnection evidence",
                    "same component",
                    "same connected component",
                ]
            )
            if not has_disconnection_phrase:
                score += 12.0

            if "there is a path" in low or "path from" in low:
                score += 8.0

        elif part == "final":
            if not utils.validate_yesno_response(state):
                score += 25.0
            if utils.extract_yes_no(text) is None:
                score += 10.0

    score += min(len(text) / 3000.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    return utils.validate_yesno_response(state)


def ground_truth(state: Dict[str, Any]) -> bool:
    truth = utils.graph_connectivity_truth(state.get("original", ""))
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