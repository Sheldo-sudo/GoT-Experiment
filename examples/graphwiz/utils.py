# examples/graphwiz/utils.py

import json
import os
import re
from typing import Any, Dict, List, Optional
from collections import deque

from datasets import get_dataset_config_names, load_dataset


def get_graphinstruct_test_subsets() -> List[str]:
    """
    Return all available subsets/configs for GraphInstruct-Test.
    """
    return get_dataset_config_names("GraphWiz/GraphInstruct-Test")


def _load_json_samples_from_file(
    json_path: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ds = load_dataset("json", data_files=json_path, split="train")

    samples: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        query = ex.get("query", ex.get("input_prompt", "")).strip()
        answer = ex.get("answer", "").strip()
        task = ex.get("task", "unknown")
        sid = ex.get("index", i)

        if sid is None:
            sid = i
        try:
            sid = int(sid)
        except Exception:
            sid = i

        meta = {}
        for k, v in ex.items():
            if k not in {"query", "input_prompt", "answer", "task"}:
                meta[k] = v

        samples.append(
            {
                "id": sid,
                "task": str(task),
                "query": query,
                "answer": answer,
                "meta": {
                    "dataset_name": f"local_json::{os.path.abspath(json_path)}",
                    **meta,
                },
            }
        )

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def _build_local_candidate_paths(
    source: str,
    subset: Optional[str],
    data_root: str,
) -> List[str]:
    """
    Build a robust list of possible local JSON paths.
    We deliberately try multiple common layouts so the loader is easier to use.

    Examples for source='test', subset='connectivity', data_root='./data':
    - ./data/connectivity.json
    - ./data/connectivity.jsonl
    - ./data/test_connectivity.json
    - ./data/test_connectivity.jsonl
    - ./data/test/connectivity.json
    - ./data/test/connectivity.jsonl
    - ./data/GraphInstruct-Test/connectivity.json
    - ./data/GraphInstruct-Test/connectivity.jsonl
    """
    data_root = os.path.abspath(data_root)
    source = (source or "").strip().lower()

    candidates: List[str] = []

    def add(path: str) -> None:
        if path not in candidates:
            candidates.append(path)

    if subset:
        subset = subset.strip()

        names = [
            f"{subset}.json",
            f"{subset}.jsonl",
            f"{source}_{subset}.json",
            f"{source}_{subset}.jsonl",
            f"{subset}_{source}.json",
            f"{subset}_{source}.jsonl",
        ]

        subdirs = [
            "",
            source,
            "GraphInstruct-Test" if source == "test" else "",
            "GraphWiz",
            os.path.join("GraphWiz", "GraphInstruct-Test") if source == "test" else "",
        ]

        for subdir in subdirs:
            if not subdir:
                for name in names:
                    add(os.path.join(data_root, name))
            else:
                for name in names:
                    add(os.path.join(data_root, subdir, name))

        # 常见特化路径：./data/test/connectivity.json
        add(os.path.join(data_root, source, f"{subset}.json"))
        add(os.path.join(data_root, source, f"{subset}.jsonl"))

    else:
        names = [
            f"{source}.json",
            f"{source}.jsonl",
        ]

        subdirs = [
            "",
            source,
            "GraphInstruct",
            "GraphInstruct-RFT-72K" if source == "rft" else "",
            "GraphWiz",
            os.path.join("GraphWiz", "GraphInstruct") if source == "train" else "",
            os.path.join("GraphWiz", "GraphInstruct-RFT-72K") if source == "rft" else "",
        ]

        for subdir in subdirs:
            if not subdir:
                for name in names:
                    add(os.path.join(data_root, name))
            else:
                for name in names:
                    add(os.path.join(data_root, subdir, name))

    return candidates


def _normalize_hf_dataset_to_samples(
    ds,
    dataset_name: str,
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        query = ex.get("query", ex.get("input_prompt", "")).strip()
        answer = ex.get("answer", "").strip()
        task = ex.get("task", subset if subset else "unknown")
        sid = ex.get("index", i)

        if sid is None:
            sid = i
        try:
            sid = int(sid)
        except Exception:
            sid = i

        meta = {}
        for k, v in ex.items():
            if k not in {"query", "input_prompt", "answer", "task"}:
                meta[k] = v

        samples.append(
            {
                "id": sid,
                "task": str(task),
                "query": query,
                "answer": answer,
                "meta": {
                    "dataset_name": dataset_name,
                    **meta,
                },
            }
        )

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def load_graphwiz_samples(
    source: str = "test",
    subset: Optional[str] = "connectivity",
    max_samples: Optional[int] = None,
    local_json_path: Optional[str] = None,
    data_root: str = "./data",
    prefer_local: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load GraphWiz / GraphInstruct style samples and normalize them to:
    {
        "id": int,
        "task": str,
        "query": str,
        "answer": str,
        "meta": dict
    }

    Backward compatible with the old signature:
        load_graphwiz_samples(source, subset, max_samples, local_json_path)

    New behavior:
        - local_json_path has highest priority
        - prefer_local=True: search local files under data_root first and REQUIRE local loading
        - prefer_local=False: load from remote HF datasets

    source:
        - "train" -> GraphWiz/GraphInstruct
        - "test"  -> GraphWiz/GraphInstruct-Test (requires subset)
        - "rft"   -> GraphWiz/GraphInstruct-RFT-72K
    """
    source = (source or "test").strip().lower()

    # 1) Explicit local file path: highest priority.
    if local_json_path:
        local_json_path = os.path.abspath(local_json_path)
        if not os.path.exists(local_json_path):
            raise FileNotFoundError(f"local_json_path not found: {local_json_path}")
        return _load_json_samples_from_file(local_json_path, max_samples=max_samples)

    # 2) Prefer local loading from data_root.
    if prefer_local:
        candidate_paths = _build_local_candidate_paths(
            source=source,
            subset=subset,
            data_root=data_root,
        )

        for path in candidate_paths:
            if os.path.exists(path):
                return _load_json_samples_from_file(path, max_samples=max_samples)

        raise FileNotFoundError(
            "prefer_local=True, but no local GraphWiz file was found.\n"
            f"source={source}, subset={subset}, data_root={os.path.abspath(data_root)}\n"
            f"Tried paths:\n" + "\n".join(candidate_paths)
        )

    # 3) Remote HF loading branch.
    if source == "train":
        ds = load_dataset("GraphWiz/GraphInstruct", split="train")
        dataset_name = "GraphWiz/GraphInstruct"

    elif source == "test":
        if not subset:
            raise ValueError(
                "source='test' 时必须提供 subset，例如 connectivity / bipartite / ..."
            )
        ds = load_dataset("GraphWiz/GraphInstruct-Test", subset, split="test")
        dataset_name = f"GraphWiz/GraphInstruct-Test::{subset}"

    elif source == "rft":
        ds = load_dataset("GraphWiz/GraphInstruct-RFT-72K", split="train")
        dataset_name = "GraphWiz/GraphInstruct-RFT-72K"

    else:
        raise ValueError(f"不支持的 source: {source}")

    return _normalize_hf_dataset_to_samples(
        ds=ds,
        dataset_name=dataset_name,
        subset=subset,
        max_samples=max_samples,
    )


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def clean_response(text: str) -> str:
    return strip_code_fences(text).strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first {...} block and parse it as JSON.
    """
    text = strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def extract_final_answer(text: str) -> str:
    """
    GraphWiz answers often end with: ### Yes / ### No / ### <number> / ### <path>.
    We normalize by taking the last ### segment if present.
    """
    text = clean_response(text)
    if "###" in text:
        return text.split("###")[-1].strip()
    return text.strip()


def normalize_text(text: str) -> str:
    text = extract_final_answer(text)
    text = text.strip().lower()
    text = text.replace("，", ",")
    text = text.replace("：", ":")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .\n\t")


def normalize_compact(text: str) -> str:
    text = normalize_text(text)
    text = text.replace(" -> ", "->")
    text = re.sub(r"\s+", "", text)
    return text


def extract_yes_no(text: str) -> Optional[str]:
    s = normalize_text(text)
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return None


def extract_last_number(text: str) -> Optional[float]:
    s = normalize_text(text)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def looks_like_sequence(text: str) -> bool:
    s = normalize_text(text)
    if "[" in s and "]" in s:
        return True
    if "->" in s:
        return True
    if "order" in s or "path" in s or "sequence" in s:
        return True
    return False


def detect_task_family(task: str = "", query: str = "") -> str:
    """
    Loose family detection so the integration can work across multiple graph tasks.
    """
    s = f"{task} {query}".lower()

    if "topolog" in s:
        return "order"

    if "shortest path" in s or "shortest_path" in s:
        if "weight of the shortest path" in s or "distance" in s or "length" in s:
            return "numeric"
        return "path"

    if "max flow" in s or "maximum flow" in s or "flow" in s:
        return "numeric"

    if "connect" in s:
        return "yesno"

    if "bipartite" in s:
        return "yesno"

    if "cycle" in s and "hamilton" not in s:
        return "yesno"

    if "triangle" in s and "how many" not in s and "count" not in s:
        return "yesno"

    if "subgraph" in s and ("yes or no" in s or "whether" in s or "is there" in s):
        return "yesno"

    if "hamilton" in s:
        if "yes or no" in s or "whether" in s or "is there" in s:
            return "yesno"
        return "path"

    return "generic"


def default_decomposition(task: str, query: str) -> Dict[str, str]:
    """
    Fallback decomposition in case the LM does not return valid JSON.
    """
    family = detect_task_family(task, query)

    if family == "yesno":
        return {
            "Subproblem 1": "Extract the graph structure, the queried nodes or target property, and identify the key local evidence needed for the decision.",
            "Subproblem 2": "Use the extracted evidence to determine whether the answer should be Yes or No, and explain the minimal reasoning path.",
        }

    if family == "numeric":
        return {
            "Subproblem 1": "Extract all nodes, edges, weights or capacities, and identify the source, sink, or target pair relevant to the calculation.",
            "Subproblem 2": "Use the extracted graph information to compute the required numeric result and verify the final value.",
        }

    if family in {"path", "order"}:
        return {
            "Subproblem 1": "Extract the graph constraints, all relevant nodes, and the structural conditions that any valid path or ordering must satisfy.",
            "Subproblem 2": "Construct a valid candidate path or ordering and verify that it satisfies all constraints from the original graph problem.",
        }

    return {
        "Subproblem 1": "Extract the graph structure, entities, and constraints from the problem statement.",
        "Subproblem 2": "Reason over the extracted structure to derive the final answer and verify it against the task requirement.",
    }


def graphwiz_format_score(state: Dict[str, Any]) -> float:
    """
    A lightweight non-gold scoring function for GoT search.
    Lower is better.

    This intentionally does NOT use state['gold'], otherwise search would leak labels.
    """
    text = clean_response(state.get("current", ""))
    if not text:
        return 100.0

    task = state.get("task", "")
    query = state.get("original", "")
    family = detect_task_family(task, query)

    score = 0.0

    if "###" not in text:
        score += 8.0

    final = extract_final_answer(text)
    if not final:
        score += 25.0

    if family == "yesno":
        if extract_yes_no(text) is None:
            score += 25.0

    elif family == "numeric":
        if extract_last_number(text) is None:
            score += 25.0

    elif family in {"path", "order"}:
        if not looks_like_sequence(text):
            score += 12.0

    # too short or too long are both suspicious
    n = len(text)
    if n < 8:
        score += 10.0
    if n > 4000:
        score += 6.0

    # slight preference for concise answers
    score += min(n / 2000.0, 5.0)

    return float(score)


def graphwiz_ground_truth(state: Dict[str, Any]) -> bool:
    """
    Final evaluator used by operations.GroundTruth.
    """
    task = state.get("task", "")
    query = state.get("original", "")
    gold = state.get("gold", "")
    pred = state.get("current", "")

    family = detect_task_family(task, query)

    if family == "yesno":
        gy = extract_yes_no(gold)
        py = extract_yes_no(pred)
        return gy is not None and py is not None and gy == py

    if family == "numeric":
        gn = extract_last_number(gold)
        pn = extract_last_number(pred)
        if gn is None or pn is None:
            return False
        return abs(gn - pn) < 1e-6

    if family in {"path", "order"}:
        g = normalize_compact(gold)
        p = normalize_compact(pred)
        if not g or not p:
            return False
        return g == p or g in p or p in g

    g = normalize_compact(gold)
    p = normalize_compact(pred)
    if not g or not p:
        return False
    return g == p or g in p or p in g


def canonical_task_name(task: str) -> str:
    t = str(task or "").strip().lower()
    alias = {
        "shortest": "shortest_path",
        "shortestpath": "shortest_path",
        "shortest_path": "shortest_path",
        "topological": "topology",
        "topological_sorting": "topology",
    }
    return alias.get(t, t)


def validate_yesno_response(state: Dict[str, Any]) -> bool:
    return extract_yes_no(state.get("current", "")) is not None


def extract_sequence_from_text(text: str) -> Optional[List[int]]:
    s = clean_response(text or "")
    if not s:
        return None

    # 1) Prefer last bracketed list
    bracket_matches = re.findall(r"\[([^\[\]]+)\]", s, flags=re.S)
    for content in reversed(bracket_matches):
        nums = re.findall(r"-?\d+", content)
        if nums:
            return [int(x) for x in nums]

    # 2) Arrow-style path
    if "->" in s:
        nums = re.findall(r"-?\d+", s)
        if nums:
            return [int(x) for x in nums]

    return None


def parse_graph_from_query(query: str) -> Dict[str, Any]:
    text = str(query or "")
    low = text.lower()

    directed_edges = re.findall(r"\((\d+)\s*->\s*(\d+)\)", text)
    undirected_edges = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)

    if directed_edges:
        directed = True
        raw_edges = directed_edges
    else:
        raw_edges = undirected_edges
        if "undirected graph" in low:
            directed = False
        elif "directed graph" in low or "digraph" in low:
            directed = True
        else:
            directed = False

    edges = []
    nodes = set()
    for a_str, b_str in raw_edges:
        a, b = int(a_str), int(b_str)
        edges.append((a, b))
        nodes.add(a)
        nodes.add(b)

    range_match = re.search(r"numbered\s+from\s+(\d+)\s+to\s+(\d+)", low)
    if range_match:
        lo = int(range_match.group(1))
        hi = int(range_match.group(2))
        if lo <= hi:
            nodes.update(range(lo, hi + 1))
        else:
            nodes.update(range(hi, lo + 1))

    return {
        "nodes": sorted(nodes),
        "edges": edges,
        "directed": directed,
    }


def build_adj(graph: Dict[str, Any], force_undirected: bool = False) -> Dict[int, List]:
    nodes = graph.get("nodes", [])
    adj = {u: [] for u in nodes}
    directed = bool(graph.get("directed", False)) and not force_undirected

    for edge in graph.get("edges", []):
        if len(edge) >= 2:
            u, v = int(edge[0]), int(edge[1])
            w = 1.0
            if len(edge) >= 3:
                try:
                    w = float(edge[2])
                except Exception:
                    w = 1.0
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, [])
            if not directed:
                adj[v].append((u, w))
    return adj


def _extract_node_pair_for_connectivity(query: str) -> Optional[List[int]]:
    low = str(query or "").lower()
    patterns = [
        r"path\s+between\s+node\s+(\d+)\s+and\s+node\s+(\d+)",
        r"between\s+node\s+(\d+)\s+and\s+node\s+(\d+)",
        r"from\s+node\s+(\d+)\s+to\s+node\s+(\d+)",
        r"from\s+(\d+)\s+to\s+(\d+)",
    ]
    for p in patterns:
        m = re.search(p, low)
        if m:
            return [int(m.group(1)), int(m.group(2))]
    return None


def graph_connectivity_truth(query: str) -> Optional[bool]:
    graph = parse_graph_from_query(query)
    pair = _extract_node_pair_for_connectivity(query)
    if not pair:
        return None
    source, target = pair
    if source == target:
        return True

    adj = build_adj(graph, force_undirected=True)
    if source not in adj or target not in adj:
        return False

    seen = {source}
    q = deque([source])
    while q:
        u = q.popleft()
        for v, _ in adj.get(u, []):
            if v == target:
                return True
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False


def graph_bipartite_truth(query: str) -> Optional[bool]:
    graph = parse_graph_from_query(query)
    if not graph.get("nodes"):
        return None

    adj = build_adj(graph, force_undirected=True)
    color: Dict[int, int] = {}
    for start in graph["nodes"]:
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v, _ in adj.get(u, []):
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True