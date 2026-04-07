import heapq
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "shortest_path"
METHOD_NAME = "structured::shortest_path"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "CandidatePathComputation",
        "goal": "Construct one strong candidate shortest path and compute its total weight carefully.",
        "num_generate": 5,
        "keep_n": 2,
    },
    {
        "part": "DistanceRelaxation",
        "goal": "Reason using tentative distances / relaxations from the source in a Dijkstra-like or Bellman-Ford-like way.",
        "num_generate": 3,
        "keep_n": 1,
    },
    {
        "part": "AlternativeAudit",
        "goal": "Compare the candidate against plausible alternatives and reject paths that are longer, cyclic, or unsupported.",
        "num_generate": 3,
        "keep_n": 1,
    },
]

# 支持：
# 1) (0, 1, 3)
# 2) (0 -> 1, 3)
_EDGE_TUPLE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_EDGE_ARROW_RE = re.compile(r"\((\d+)\s*->\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_PAIR_PATTERNS = [
    re.compile(r"(?:shortest\s+path\s+from\s+node|from\s+node)\s+(\d+)\s+to\s+node\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"(?:shortest\s+path\s+from|from)\s+(\d+)\s+to\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"source\s*(?:node)?\s*(\d+)\s*(?:and|,)?\s*target\s*(?:node)?\s*(\d+)", flags=re.IGNORECASE),
]
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NO_PATH_RE = re.compile(r"\bno\s+path\b", flags=re.IGNORECASE)


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


def _extract_no_path(text: str) -> bool:
    return bool(_NO_PATH_RE.search(text or ""))


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
    if t.rstrip().endswith((":", ",", "(", "[", "{", "->", "-", "=")):
        return True
    return t.rstrip().endswith("...")


def _infer_directed(query: str, has_arrow_edges: bool) -> bool:
    text = (query or "").lower()
    if has_arrow_edges:
        return True
    if "directed graph" in text or "directed weighted graph" in text or "digraph" in text:
        return True
    if "undirected graph" in text:
        return False
    return False


def _parse_shortest_instance(query: str) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
    text = query or ""
    arrow_edges = _EDGE_ARROW_RE.findall(text)
    tuple_edges = _EDGE_TUPLE_RE.findall(text)

    directed = _infer_directed(text, bool(arrow_edges))
    raw_edges = arrow_edges if arrow_edges else tuple_edges

    edges: List[Tuple[int, int, float]] = []
    nodes = set()

    for a_str, b_str, w_str in raw_edges:
        try:
            a, b = int(a_str), int(b_str)
            w = float(w_str)
        except Exception:
            continue
        edges.append((a, b, w))
        nodes.add(a)
        nodes.add(b)

    pair = None
    for pat in _PAIR_PATTERNS:
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
    return {"n": n, "edges": edges, "directed": directed}, pair


def _build_adj(graph: Dict[str, Any]) -> Dict[int, List[Tuple[int, float]]]:
    adj = {i: [] for i in range(graph["n"])}
    directed = bool(graph.get("directed", False))
    for u, v, w in graph["edges"]:
        adj[u].append((v, float(w)))
        if not directed:
            adj[v].append((u, float(w)))
    return adj


def _has_negative_edge(graph: Dict[str, Any]) -> bool:
    return any(float(w) < 0 for _, _, w in graph["edges"])


def _dijkstra_distance(graph: Dict[str, Any], source: int, target: int) -> Optional[float]:
    n = graph["n"]
    if n <= 0 or source < 0 or target < 0 or source >= n or target >= n:
        return None
    if _has_negative_edge(graph):
        return None

    adj = _build_adj(graph)
    dist = [float("inf")] * n
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        cur_d, u = heapq.heappop(pq)
        if cur_d > dist[u] + 1e-12:
            continue
        if u == target:
            return cur_d
        for v, w in adj.get(u, []):
            nd = cur_d + w
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return None if dist[target] == float("inf") else dist[target]


def _bellman_ford_distance(graph: Dict[str, Any], source: int, target: int) -> Optional[float]:
    n = graph["n"]
    if n <= 0 or source < 0 or target < 0 or source >= n or target >= n:
        return None

    dist = [float("inf")] * n
    dist[source] = 0.0

    edges = []
    for u, v, w in graph["edges"]:
        edges.append((u, v, float(w)))
        if not graph.get("directed", False):
            edges.append((v, u, float(w)))

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] == float("inf"):
                continue
            nd = dist[u] + w
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                updated = True
        if not updated:
            break

    # 如果存在可达负环，当前任务通常不应出现；保守返回 None 交给 fallback
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w + 1e-12 < dist[v]:
            return None

    return None if dist[target] == float("inf") else dist[target]


def _shortest_distance(graph: Dict[str, Any], source: int, target: int) -> Optional[float]:
    if _has_negative_edge(graph):
        return _bellman_ford_distance(graph, source, target)
    return _dijkstra_distance(graph, source, target)


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for shortest path.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "CandidatePathComputation":
        return f"""You are solving the CandidatePathComputation branch.

Original problem:
{original_query}

Goal:
Construct one strong candidate shortest path and compute its total weight carefully.

Rules:
1. Internally you may mention a path, but keep it brief.
2. The final answer must contain only the final distance or 'No path'.
3. Do not dump long path enumerations.
4. End with exactly one final line in one of these forms:
### <number>
### No path
"""
    if part == "DistanceRelaxation":
        return f"""You are solving the DistanceRelaxation branch.

Original problem:
{original_query}

Goal:
Reason using tentative distances / relaxations from the source.

Rules:
1. Prefer compact distance updates over enumerating many full paths.
2. Keep the explanation short.
3. End with exactly one final line in one of these forms:
### RelaxedDistance: <number>
### No path
"""
    return f"""You are solving the AlternativeAudit branch.

Original problem:
{original_query}

Goal:
Compare the candidate against plausible alternatives and reject paths that are longer, cyclic, or unsupported.

Rules:
1. Cyclic or repeated-node detours should not improve a normal shortest-path answer unless negative cycles are explicitly required.
2. Keep the explanation short.
3. End with exactly one final line in one of these forms:
### AuditedDistance: <number>
### No path
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for shortest path.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. The final answer is the shortest-path WEIGHT, not the path string.
2. Prefer values supported by both a concrete candidate and relaxation-style reasoning.
3. Reject cyclic, unsupported, or obviously longer alternatives.
4. Keep the final answer extremely short.

Your final line must be one of:
### <number>
### No path
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the shortest-path answer.

Original problem:
{original_query}

Previous answer:
{current}

Rules:
1. Output only the final shortest-path distance, not a full path.
2. Fix mistakes caused by missing a lighter alternative or confusing directed vs undirected edges.
3. If unreachable, answer No path.
4. Keep it very short.

Your final line must be one of:
### <number>
### No path
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
    if _extract_last_number(text) is None and not _extract_no_path(text):
        score += 20.0
    if _looks_truncated(text):
        score += 16.0
    if len(text) > 1800:
        score += 10.0
    elif len(text) > 900:
        score += 4.0

    if part == "CandidatePathComputation":
        if not any(k in low for k in ["path", "weight", "total", "candidate"]):
            score += 10.0
    elif part == "DistanceRelaxation":
        if not any(k in low for k in ["distance", "relax", "tentative", "update", "frontier"]):
            score += 12.0
    elif part == "AlternativeAudit":
        if not any(k in low for k in ["alternative", "shorter", "audit", "compare", "cycle"]):
            score += 12.0
    elif part == "final":
        if not final_validator(state):
            score += 30.0

    score += min(len(text) / 2500.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    text = state.get("current", "")
    ok = _extract_last_number(text) is not None or _extract_no_path(text)
    if not ok:
        return False
    if _looks_truncated(text):
        return False
    final_line = _extract_final_line(text).lower()
    if not final_line:
        return False
    if "###" not in final_line and len(_clean_text(text)) > 140:
        return False
    return True


def ground_truth(state: Dict[str, Any]) -> bool:
    text = state.get("current", "")
    graph, pair = _parse_shortest_instance(state.get("original", ""))

    if pair is None or not graph["edges"]:
        gold_num = _extract_last_number(state.get("gold", ""))
        pred = _extract_last_number(text)
        if gold_num is not None and pred is not None:
            return abs(pred - gold_num) < 1e-6
        if _extract_no_path(state.get("gold", "")) and _extract_no_path(text):
            return True
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    truth = _shortest_distance(graph, pair[0], pair[1])
    pred = _extract_last_number(text)

    if truth is None:
        if _extract_no_path(text):
            return True
        gold_num = _extract_last_number(state.get("gold", ""))
        if gold_num is not None and pred is not None:
            return abs(pred - gold_num) < 1e-6
        return False

    return pred is not None and abs(pred - truth) < 1e-6


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