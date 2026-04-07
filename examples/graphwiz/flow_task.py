import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "flow"
METHOD_NAME = "structured::flow"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "SourceSinkBounds",
        "goal": "Compute tight upper bounds from source outgoing capacity, sink incoming capacity, and obvious bottlenecks / cuts.",
        "num_generate": 3,
        "keep_n": 1,
    },
    {
        "part": "AugmentingPathPlan",
        "goal": "Construct a feasible lower bound by combining capacity-compatible augmenting paths without violating shared-edge capacities.",
        "num_generate": 5,
        "keep_n": 2,
    },
    {
        "part": "OptimalityCheck",
        "goal": "Show that the best feasible lower bound already matches the strongest valid upper bound, so the value is optimal.",
        "num_generate": 3,
        "keep_n": 1,
    },
]

# 支持两种边格式：
# 1) (0 -> 1, 3)
# 2) (0, 1, 3)
_FLOW_EDGE_ARROW_RE = re.compile(r"\((\d+)\s*->\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_FLOW_EDGE_TUPLE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_FLOW_PAIR_PATTERNS = [
    re.compile(r"(?:maximum\s+flow|flow)\s+from\s+node\s+(\d+)\s+to\s+node\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"source\s*(?:node)?\s*(\d+)\s*(?:and|,)?\s*sink\s*(?:node)?\s*(\d+)", flags=re.IGNORECASE),
    re.compile(r"from\s+(\d+)\s+to\s+(\d+)", flags=re.IGNORECASE),
]
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
    try:
        return float(matches[-1])
    except Exception:
        return None


def _extract_final_line(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _looks_truncated(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return False
    if "###" not in t and len(t) > 1200:
        return True
    tail = t.rstrip()[-6:]
    bad_suffixes = (":", ",", "(", "[", "{", "->", "-", "=")
    return any(t.rstrip().endswith(s) for s in bad_suffixes) or tail.endswith("...")


def _parse_flow_instance(query: str) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
    text = query or ""

    edges: List[Tuple[int, int, float]] = []
    nodes = set()

    arrow_edges = _FLOW_EDGE_ARROW_RE.findall(text)
    tuple_edges = _FLOW_EDGE_TUPLE_RE.findall(text)

    if arrow_edges:
        raw_edges = arrow_edges
    else:
        raw_edges = tuple_edges

    for u_str, v_str, w_str in raw_edges:
        try:
            u, v = int(u_str), int(v_str)
            w = float(w_str)
        except Exception:
            continue
        edges.append((u, v, w))
        nodes.add(u)
        nodes.add(v)

    pair = None
    for pat in _FLOW_PAIR_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                pair = (int(m.group(1)), int(m.group(2)))
                nodes.add(pair[0])
                nodes.add(pair[1])
                break
            except Exception:
                pass

    n = max(nodes) + 1 if nodes else 0
    return {"n": n, "edges": edges, "directed": True}, pair


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for maximum flow.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "SourceSinkBounds":
        return f"""You are solving the SourceSinkBounds branch of a structured Graph-of-Thought workflow for maximum flow.

Original problem:
{original_query}

Goal:
Compute the strongest easy upper bound.

Rules:
1. Use only source-outgoing capacity, sink-incoming capacity, or a valid bottleneck / cut.
2. Never add incompatible path capacities together.
3. Keep the reasoning short.
4. End with exactly one final line:
### UpperBound: <number>
"""
    if part == "AugmentingPathPlan":
        return f"""You are solving the AugmentingPathPlan branch of a structured Graph-of-Thought workflow for maximum flow.

Original problem:
{original_query}

Goal:
Construct a feasible lower bound by combining capacity-compatible augmenting paths.

Rules:
1. Only count simultaneously feasible flow.
2. If two paths share an edge, respect the shared capacity.
3. Prefer 2-4 concise path/cut statements, not long prose.
4. End with exactly one final line:
### FeasibleFlow: <number>
"""
    return f"""You are solving the OptimalityCheck branch of a structured Graph-of-Thought workflow for maximum flow.

Original problem:
{original_query}

Goal:
Show that the best feasible value is already optimal by matching it to an upper bound.

Rules:
1. Compare one feasible flow value against one valid upper bound.
2. If they match, certify optimality.
3. Keep the explanation brief.
4. End with exactly one final line:
### CertifiedMaxFlow: <number>
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for a maximum-flow problem.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. Output only the final maximum-flow value.
2. Trust a value only if it is supported as a feasible flow and not contradicted by a tighter upper bound.
3. Reject over-counting caused by incompatible shared edges.
4. Keep the final answer extremely short.

Your final line must be:
### <number>
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the maximum-flow answer.

Original problem:
{original_query}

Previous answer:
{current}

Rules:
1. The final answer must be only one numeric maximum-flow value.
2. Fix any over-counting of incompatible augmenting paths.
3. Prefer a value that is feasible and also upper-bounded by the same cut.
4. Keep it very short.

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
        score += 10.0
    if _extract_last_number(text) is None:
        score += 22.0
    if _looks_truncated(text):
        score += 16.0
    if len(text) > 1800:
        score += 10.0
    elif len(text) > 900:
        score += 4.0

    if part == "SourceSinkBounds":
        if not any(k in low for k in ["upperbound", "upper bound", "incoming", "outgoing", "cut", "bottleneck"]):
            score += 12.0
    elif part == "AugmentingPathPlan":
        if not any(k in low for k in ["path", "feasible", "augment", "residual", "flow"]):
            score += 10.0
        if "feasibleflow" not in low and "feasible flow" not in low:
            score += 8.0
    elif part == "OptimalityCheck":
        if not any(k in low for k in ["optimal", "certified", "cut", "upper bound", "match"]):
            score += 12.0
    elif part == "final":
        if not final_validator(state):
            score += 30.0

    score += min(len(text) / 2500.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    text = state.get("current", "")
    value = _extract_last_number(text)
    if value is None:
        return False
    if _looks_truncated(text):
        return False
    final_line = _extract_final_line(text)
    if not final_line:
        return False
    if "###" not in final_line and len(_clean_text(text)) > 120:
        return False
    return True


def _edmonds_karp(graph: Dict[str, Any], source: int, sink: int) -> Optional[float]:
    n = graph["n"]
    if n <= 0 or source < 0 or sink < 0 or source >= n or sink >= n:
        return None

    cap = [[0.0 for _ in range(n)] for _ in range(n)]
    for u, v, w in graph["edges"]:
        if w < -1e-12:
            return None
        if w > 0:
            cap[u][v] += float(w)

    flow = 0.0
    while True:
        parent = [-1] * n
        parent[source] = source
        q = deque([source])

        while q and parent[sink] == -1:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and cap[u][v] > 1e-12:
                    parent[v] = u
                    q.append(v)

        if parent[sink] == -1:
            break

        aug = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            aug = min(aug, cap[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= aug
            cap[v][u] += aug
            v = u

        flow += aug

    return flow


def ground_truth(state: Dict[str, Any]) -> bool:
    pred = _extract_last_number(state.get("current", ""))
    if pred is None:
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    graph, pair = _parse_flow_instance(state.get("original", ""))
    if pair is None or not graph["edges"]:
        gold_num = _extract_last_number(state.get("gold", ""))
        if gold_num is None:
            fn = getattr(utils, "graphwiz_ground_truth", None)
            return fn(state) if callable(fn) else False
        return abs(pred - gold_num) < 1e-6

    truth = _edmonds_karp(graph, pair[0], pair[1])
    if truth is None:
        gold_num = _extract_last_number(state.get("gold", ""))
        if gold_num is not None:
            return abs(pred - gold_num) < 1e-6
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

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