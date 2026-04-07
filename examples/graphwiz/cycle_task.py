from typing import Any, Dict, List

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "cycle"
METHOD_NAME = "structured::cycle"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "CycleWitness",
        "goal": "Find one explicit cycle witness if a cycle exists.",
        "num_generate": 4,
        "keep_n": 1,
    },
    {
        "part": "AcyclicCheck",
        "goal": "Check whether the graph is acyclic by elimination, traversal reasoning, or absence of back-edge / revisitation evidence.",
        "num_generate": 4,
        "keep_n": 1,
    },
]


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for cycle detection.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "CycleWitness":
        return f"""You are solving the CycleWitness branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Find ONE explicit cycle witness if a cycle exists.

Rules:
1. Focus only on finding a concrete cycle witness.
2. Prefer a short explicit node sequence such as [1, 4, 7, 1].
3. Do not discuss "acyclic" in this branch.
4. If you find a cycle, the final line must be:
### Cycle: [a->b->c->a]
5. If you do not find an explicit cycle, the final line must be:
### No explicit cycle found
"""
    return f"""You are solving the AcyclicCheck branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Look ONLY for concrete acyclic evidence.

Rules:
1. Do NOT provide a cycle witness in this branch.
2. Do NOT conclude "No" unless you have concrete acyclic evidence such as successful elimination, no back-edge, or a valid DAG-style explanation.
3. If you find concrete acyclic evidence, the final line must be:
### Acyclic evidence: <brief reason>
4. If you do NOT find concrete acyclic evidence, the final line must be:
### No acyclic evidence
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given multiple branch analyses for cycle detection.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. If the CycleWitness branch contains any explicit cycle witness, you MUST answer Yes.
2. Only answer No if there is NO explicit cycle witness and the AcyclicCheck branch provides concrete acyclic evidence.
3. "No acyclic evidence" is NOT evidence for No.
4. When branches conflict, trust an explicit cycle witness over speculative acyclic analysis.

Your final line must be exactly one of:
### Yes
### No
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the final answer for cycle detection.

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
    if len(text) > 3000:
        score += 6.0

    if part == "CycleWitness":
        seq = utils.extract_sequence_from_text(text)
        yn = utils.extract_yes_no(text)
        if seq is None and "cycle:" not in low:
            score += 18.0
        if yn not in {None, "yes"}:
            score += 12.0
        if "no explicit cycle found" in low or "no cycle" in low:
            score += 16.0

    elif part == "AcyclicCheck":
        if utils.extract_sequence_from_text(text) is not None:
            score += 12.0

        has_acyclic_phrase = any(
            k in low for k in [
                "acyclic evidence",
                "no back-edge",
                "no back edge",
                "topological",
                "elimination",
                "dag",
                "no acyclic evidence",
            ]
        )
        if not has_acyclic_phrase:
            score += 12.0

        if "cycle:" in low or "there is a cycle" in low:
            score += 8.0

    elif part == "final":
        if not utils.validate_yesno_response(state):
            score += 25.0

    score += min(len(text) / 3000.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    return utils.validate_yesno_response(state)


def _has_cycle_undirected(graph: Dict[str, Any]) -> bool:
    adj = utils.build_adj(graph, force_undirected=True)
    seen = set()

    def dfs(u: int, parent: int) -> bool:
        seen.add(u)
        for v, _ in adj.get(u, []):
            if v == parent:
                continue
            if v in seen:
                return True
            if dfs(v, u):
                return True
        return False

    for node in graph["nodes"]:
        if node not in seen and dfs(node, -1):
            return True
    return False


def _has_cycle_directed(graph: Dict[str, Any]) -> bool:
    adj = utils.build_adj(graph, force_undirected=False)
    color = {u: 0 for u in graph["nodes"]}

    def dfs(u: int) -> bool:
        color[u] = 1
        for v, _ in adj.get(u, []):
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    for node in graph["nodes"]:
        if color[node] == 0 and dfs(node):
            return True
    return False


def ground_truth(state: Dict[str, Any]) -> bool:
    graph = utils.parse_graph_from_query(state.get("original", ""))
    pred = utils.extract_yes_no(state.get("current", ""))
    if pred is None:
        return utils.graphwiz_ground_truth(state)

    truth = _has_cycle_directed(graph) if graph["directed"] else _has_cycle_undirected(graph)
    return (pred == "yes") == truth


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