import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "substructure"
METHOD_NAME = "structured::substructure"

BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "PatternRoleExtraction",
        "goal": "Summarize the pattern graph by role signatures such as in-degree, out-degree, anchors, and rare edge motifs.",
        "num_generate": 3,
        "keep_n": 1,
    },
    {
        "part": "CandidateMapping",
        "goal": "Propose a concrete injective mapping from pattern nodes to host nodes if one exists.",
        "num_generate": 5,
        "keep_n": 2,
    },
    {
        "part": "EdgeCoverageAudit",
        "goal": "Check whether every required pattern edge is preserved under the mapping, or identify a missing required edge.",
        "num_generate": 3,
        "keep_n": 1,
    },
]

_YES_RE = re.compile(r"\byes\b", flags=re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", flags=re.IGNORECASE)

# 支持：
# 1) (a -> b)
# 2) (a, b)   仅作为兜底
_EDGE_ARROW_RE = re.compile(r"\(([^()]+?)\s*->\s*([^()]+?)\)")
_EDGE_TUPLE_RE = re.compile(r"\(([^(),]+?)\s*,\s*([^(),]+?)\)")

# 识别一些常见 host/pattern 分界标记
_PATTERN_SPLIT_MARKERS = [
    "The nodes of subgraph G'",
    "The nodes of subgraph G’",
    "subgraph G'",
    "subgraph G’",
    "pattern graph",
    "pattern subgraph",
    "query graph",
]

# 尝试提取节点列表
_NODE_LIST_PATTERNS = [
    re.compile(r"(?:nodes|vertices)\s+of\s+subgraph\s+g[’']?\s*(?:are|:)\s*([A-Za-z0-9_,\s]+)", flags=re.IGNORECASE),
    re.compile(r"(?:nodes|vertices)\s*(?:are|:)\s*([A-Za-z0-9_,\s]+)", flags=re.IGNORECASE),
]


def _clean_text(text: str) -> str:
    fn = getattr(utils, "clean_response", None)
    if callable(fn):
        return fn(text)
    return (text or "").strip()


def _extract_yes_no(text: str) -> Optional[bool]:
    fn = getattr(utils, "extract_yes_no", None)
    if callable(fn):
        out = fn(text)
        if out is not None:
            s = str(out).strip().lower()
            if s == "yes":
                return True
            if s == "no":
                return False
    if _YES_RE.search(text or ""):
        return True
    if _NO_RE.search(text or ""):
        return False
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
    if t.rstrip().endswith((":", ",", "(", "[", "{", "->", "-", "=")):
        return True
    return t.rstrip().endswith("...")


def _token_is_numeric(tok: str) -> bool:
    return tok.strip().isdigit()


def _normalize_node_token(tok: str) -> str:
    return tok.strip().strip("{}[]() ").strip("'")


def _split_host_and_pattern_text(text: str) -> Tuple[str, str]:
    for marker in _PATTERN_SPLIT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            return text[:idx], text[idx:]
    return text, ""


def _extract_node_list(text: str) -> Set[str]:
    out: Set[str] = set()
    for pat in _NODE_LIST_PATTERNS:
        for m in pat.finditer(text or ""):
            chunk = m.group(1)
            for tok in re.findall(r"[A-Za-z0-9_]+", chunk):
                out.add(_normalize_node_token(tok))
    return out


def _extract_edges(text: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []

    arrow_edges = _EDGE_ARROW_RE.findall(text or "")
    if arrow_edges:
        for a_raw, b_raw in arrow_edges:
            a = _normalize_node_token(a_raw)
            b = _normalize_node_token(b_raw)
            if a and b:
                edges.append((a, b))
        return edges

    for a_raw, b_raw in _EDGE_TUPLE_RE.findall(text or ""):
        a = _normalize_node_token(a_raw)
        b = _normalize_node_token(b_raw)
        if a and b:
            edges.append((a, b))

    return edges


def _parse_substructure_instance(query: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    text = query or ""
    host_text, pattern_text = _split_host_and_pattern_text(text)

    host_edges_all = _extract_edges(host_text)
    pattern_edges_all = _extract_edges(pattern_text)

    host_nodes: Set[str] = set()
    pattern_nodes: Set[str] = set()

    # 优先使用 marker 分段后的结果
    host_edges: Set[Tuple[int, int]] = set()
    for a, b in host_edges_all:
        if _token_is_numeric(a) and _token_is_numeric(b):
            host_edges.add((int(a), int(b)))
            host_nodes.add(a)
            host_nodes.add(b)

    pattern_edges: Set[Tuple[str, str]] = set()
    for a, b in pattern_edges_all:
        pattern_edges.add((a, b))
        pattern_nodes.add(a)
        pattern_nodes.add(b)

    # 如果 marker 解析不到 pattern，就用整段文本做一次“数值 host / 非数值 pattern”分离
    if not pattern_edges and not pattern_nodes:
        all_edges = _extract_edges(text)
        for a, b in all_edges:
            if _token_is_numeric(a) and _token_is_numeric(b):
                host_edges.add((int(a), int(b)))
                host_nodes.add(a)
                host_nodes.add(b)
            else:
                pattern_edges.add((a, b))
                pattern_nodes.add(a)
                pattern_nodes.add(b)

    # 补节点列表
    for tok in _extract_node_list(host_text):
        if _token_is_numeric(tok):
            host_nodes.add(tok)

    for tok in _extract_node_list(pattern_text):
        if not _token_is_numeric(tok):
            pattern_nodes.add(tok)

    # 如果 pattern 仍为空，则说明解析失败，不允许误判 True
    if not host_nodes or not pattern_nodes:
        return None, None

    host = {
        "nodes": sorted(int(x) for x in host_nodes),
        "edges": host_edges,
        "out": {int(u): set() for u in host_nodes},
        "in": {int(u): set() for u in host_nodes},
    }
    for u, v in host_edges:
        host["out"][u].add(v)
        host["in"][v].add(u)

    pattern = {
        "nodes": sorted(pattern_nodes),
        "edges": pattern_edges,
        "out": {u: set() for u in pattern_nodes},
        "in": {u: set() for u in pattern_nodes},
    }
    for u, v in pattern_edges:
        pattern["out"][u].add(v)
        pattern["in"][v].add(u)

    return host, pattern


def _has_subgraph_isomorphism(host: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
    p_nodes: List[str] = list(pattern["nodes"])
    h_nodes: List[int] = list(host["nodes"])

    if not p_nodes:
        return False
    if len(p_nodes) > len(h_nodes):
        return False

    order = sorted(
        p_nodes,
        key=lambda u: (
            -(len(pattern["out"][u]) + len(pattern["in"][u])),
            -len(pattern["out"][u]),
            -len(pattern["in"][u]),
            u,
        ),
    )

    candidates: Dict[str, List[int]] = {}
    for pu in p_nodes:
        pout = len(pattern["out"][pu])
        pin = len(pattern["in"][pu])
        cand = []
        for hu in h_nodes:
            if len(host["out"][hu]) >= pout and len(host["in"][hu]) >= pin:
                cand.append(hu)
        if not cand:
            return False
        candidates[pu] = cand

    mapping: Dict[str, int] = {}
    used: Set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True

        pu = order[idx]
        for hu in candidates[pu]:
            if hu in used:
                continue

            ok = True
            for pv, hv in mapping.items():
                if (pu, pv) in pattern["edges"] and (hu, hv) not in host["edges"]:
                    ok = False
                    break
                if (pv, pu) in pattern["edges"] and (hv, hu) not in host["edges"]:
                    ok = False
                    break

            if not ok:
                continue

            mapping[pu] = hu
            used.add(hu)
            if backtrack(idx + 1):
                return True
            used.remove(hu)
            del mapping[pu]

        return False

    return backtrack(0)


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for substructure matching.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "PatternRoleExtraction":
        return f"""You are solving the PatternRoleExtraction branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Summarize the pattern graph by anchor roles, in-degree / out-degree signatures, and rare motifs.

Rules:
1. Focus on the hardest pattern nodes to match.
2. Keep it concise.
3. End with exactly one final line:
### AnchorSummary: <very short summary>
"""
    if part == "CandidateMapping":
        return f"""You are solving the CandidateMapping branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Propose a concrete injective mapping from pattern nodes to host nodes if one exists.

Rules:
1. Conclude Yes only if the mapping can preserve all required pattern edges.
2. Otherwise conclude No.
3. Do not print a very long mapping table.
4. End with exactly one final line:
### Yes
or
### No
"""
    return f"""You are solving the EdgeCoverageAudit branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Check whether every required pattern edge is preserved under the candidate mapping, or identify a missing required edge.

Rules:
1. Missing even one required directed edge means failure.
2. Keep the explanation short.
3. End with exactly one final line:
### Yes
or
### No
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for substructure matching.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. This is a Yes/No task.
2. Prefer Yes only when a concrete injective mapping survives all required pattern edges.
3. Prefer No when every plausible anchor fails because at least one required pattern edge is missing.
4. Keep the final answer extremely short.

Your final line must be:
### Yes
or
### No
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the substructure-matching answer.

Original problem:
{original_query}

Previous answer:
{current}

Rules:
1. Output only Yes or No.
2. Use Yes only if the pattern graph can be embedded into the host graph with an injective mapping.
3. Extra host edges are allowed; missing required pattern edges are not.
4. Keep it very short.

Your final line must be:
### Yes
or
### No
"""


def _validate_substructure_response(state: Dict[str, Any]) -> bool:
    text = state.get("current", "")
    if _extract_yes_no(text) is None:
        return False
    if _looks_truncated(text):
        return False
    final_line = _extract_final_line(text)
    if not final_line:
        return False
    if "###" not in final_line and len(_clean_text(text)) > 80:
        return False
    return True


def search_score(state: Dict[str, Any]) -> float:
    text = _clean_text(state.get("current", ""))
    part = state.get("part", "")
    low = text.lower()

    if not text:
        return 100.0

    score = 0.0

    if "###" not in text:
        score += 10.0
    if part != "PatternRoleExtraction" and _extract_yes_no(text) is None:
        score += 20.0
    if part == "PatternRoleExtraction" and "anchorsummary" not in low and "anchor summary" not in low:
        score += 10.0
    if _looks_truncated(text):
        score += 16.0
    if len(text) > 1800:
        score += 10.0
    elif len(text) > 900:
        score += 4.0

    if part == "PatternRoleExtraction":
        if not any(k in low for k in ["pattern", "anchor", "in-degree", "out-degree", "role", "motif"]):
            score += 12.0
    elif part == "CandidateMapping":
        if not any(k in low for k in ["mapping", "injective", "match", "candidate"]):
            score += 10.0
    elif part == "EdgeCoverageAudit":
        if not any(k in low for k in ["edge", "required", "missing", "audit", "preserve"]):
            score += 12.0
    elif part == "final":
        if not _validate_substructure_response(state):
            score += 30.0

    score += min(len(text) / 2500.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    return _validate_substructure_response(state)


def ground_truth(state: Dict[str, Any]) -> bool:
    pred = _extract_yes_no(state.get("current", ""))
    if pred is None:
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    host, pattern = _parse_substructure_instance(state.get("original", ""))
    if host is None or pattern is None:
        gold = _extract_yes_no(state.get("gold", ""))
        if gold is not None:
            return pred == gold
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    truth = _has_subgraph_isomorphism(host, pattern)
    return pred == truth


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