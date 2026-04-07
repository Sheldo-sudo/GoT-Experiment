import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "triangle"
METHOD_NAME = "structured::triangle"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "HighWeightCandidateScan",
        "goal": "Identify high-weight nodes and prioritize plausible triangle candidates among them.",
        "num_generate": 4,
        "keep_n": 1,
    },
    {
        "part": "TriangleVerification",
        "goal": "Check whether candidate triples are actual 3-cliques and compute their node-weight sums exactly.",
        "num_generate": 4,
        "keep_n": 2,
    },
    {
        "part": "AlternativeComparison",
        "goal": "Compare the best verified triangle against remaining plausible alternatives and keep the largest sum.",
        "num_generate": 3,
        "keep_n": 1,
    },
]


_WEIGHT_RE = re.compile(r"\[(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\]")
_EDGE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _clean_text(text: str) -> str:
    fn = getattr(utils, "clean_response", None)
    if callable(fn):
        return fn(text)
    return (text or "").strip()


def _extract_last_number(text: str) -> Optional[float]:
    fn = getattr(utils, "extract_last_number", None)
    if callable(fn):
        return fn(text)
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    return float(matches[-1])


def _parse_triangle_instance(query: str) -> Tuple[Dict[int, float], Set[Tuple[int, int]]]:
    weights: Dict[int, float] = {}
    for node_str, weight_str in _WEIGHT_RE.findall(query or ""):
        weights[int(node_str)] = float(weight_str)

    edges: Set[Tuple[int, int]] = set()
    for a_str, b_str in _EDGE_RE.findall(query or ""):
        a, b = int(a_str), int(b_str)
        if a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        edges.add(edge)
    return weights, edges


def _max_triangle_sum(weights: Dict[int, float], edges: Set[Tuple[int, int]]) -> Optional[float]:
    nodes = sorted(weights)
    best: Optional[float] = None
    for a, b, c in combinations(nodes, 3):
        if (min(a, b), max(a, b)) not in edges:
            continue
        if (min(a, c), max(a, c)) not in edges:
            continue
        if (min(b, c), max(b, c)) not in edges:
            continue
        cur = float(weights[a] + weights[b] + weights[c])
        if best is None or cur > best:
            best = cur
    return best


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for the maximum triangle-sum task.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "HighWeightCandidateScan":
        return f"""You are solving the HighWeightCandidateScan branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Identify high-weight nodes and prioritize plausible triangle candidates among them.

Rules:
1. A candidate is useful only if the three nodes might form a full triangle.
2. End with exactly one final line:
### CandidateSum: <number>
"""
    if part == "TriangleVerification":
        return f"""You are solving the TriangleVerification branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Check whether candidate triples are actual 3-cliques and compute their node-weight sums exactly.

Rules:
1. Three interconnected nodes means all three pairwise edges must exist.
2. End with exactly one final line:
### VerifiedSum: <number>
"""
    return f"""You are solving the AlternativeComparison branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Compare the best verified triangle against remaining plausible alternatives and keep the largest sum.

Rules:
1. Reject any triple missing even one edge.
2. End with exactly one final line:
### BestSum: <number>
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for the maximum triangle-sum task.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. Three interconnected nodes must form a full triangle (all three pairwise edges present).
2. The final answer is the maximum NODE-WEIGHT sum among valid triangles.
3. Output only the final numeric answer.

Your final line must be:
### <number>
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the maximum triangle-sum answer.

Original problem:
{original_query}

Previous answer:
{current}

Rules:
1. Count node weights, not edge weights.
2. Use only triples whose three pairwise edges all exist.
3. Output only the final numeric answer.

Your final line must be:
### <number>
"""


def search_score(state: Dict[str, Any]) -> float:
    text = _clean_text(state.get("current", ""))
    part = state.get("part", "")
    low = text.lower()

    if not text:
        return 100.0

    score = 0.0
    if "###" not in text:
        score += 8.0
    if _extract_last_number(text) is None:
        score += 18.0
    if len(text) > 2500:
        score += 6.0

    if part == "HighWeightCandidateScan":
        if not any(k in low for k in ["weight", "candidate", "high", "triangle"]):
            score += 10.0
    elif part == "TriangleVerification":
        if not any(k in low for k in ["verify", "pairwise", "clique", "triangle", "sum"]):
            score += 12.0
    elif part == "AlternativeComparison":
        if not any(k in low for k in ["alternative", "compare", "best", "maximum", "larger"]):
            score += 12.0
    elif part == "final":
        if not final_validator(state):
            score += 25.0

    score += min(len(text) / 3000.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    return _extract_last_number(state.get("current", "")) is not None


def ground_truth(state: Dict[str, Any]) -> bool:
    pred = _extract_last_number(state.get("current", ""))
    if pred is None:
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    weights, edges = _parse_triangle_instance(state.get("original", ""))
    truth = _max_triangle_sum(weights, edges)
    if truth is None:
        gold = _extract_last_number(state.get("gold", ""))
        if gold is None:
            fn = getattr(utils, "graphwiz_ground_truth", None)
            return fn(state) if callable(fn) else False
        return abs(pred - gold) < 1e-6
    return abs(pred - truth) < 1e-6


def build_graph():
    return build_task_graph(
        branches=BRANCHES,
        search_score_fn=search_score,
        final_validator=final_validator,
        ground_truth_fn=ground_truth,
        aggregate_responses=4,
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
