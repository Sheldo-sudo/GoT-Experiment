import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "topology"
METHOD_NAME = "structured::topology"

# 设计思路：
# 1) 小图/中图/大图自适应 prompt，避免大图长篇展开。
# 2) 更关键：自定义 TopologyParser，在 parser 阶段对候选序列做程序化 Kahn 修复。
#    这样模型只需要提供“偏好排序”，最终输出由代码保证满足所有边约束。
# 3) 接口不变：build_graph / get_prompter / get_parser 仍然保留。
BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "ConstraintExtraction",
        "goal": "Extract a small set of anchor precedence constraints implied by directed edges.",
        "num_generate": 1,
        "keep_n": 1,
    },
    {
        "part": "CandidateOrdering",
        "goal": "Construct one complete candidate topological ordering containing every node exactly once.",
        "num_generate": 2,
        "keep_n": 2,
    },
]


def _clean_response(text: str) -> str:
    clean_fn = getattr(utils, "clean_response", None)
    if callable(clean_fn):
        try:
            return clean_fn(text)
        except Exception:
            pass
    return str(text or "").strip()


def _extract_sequence_from_text(text: str) -> Optional[List[int]]:
    """
    从回答中提取候选序列。
    优先取最后一个 ### 后面的内容；若失败，再从全文兜底提取。
    """
    if not text:
        return None

    s = str(text)
    candidate_regions: List[str] = []

    idx = s.rfind("###")
    if idx != -1:
        candidate_regions.append(s[idx + 3 :].strip())

    candidate_regions.append(s)

    for region in candidate_regions:
        # 优先匹配最后一个 [ ... ] 块
        bracket_matches = re.findall(r"\[([^\[\]]+)\]", region, flags=re.S)
        for content in reversed(bracket_matches):
            nums = re.findall(r"-?\d+", content)
            if nums:
                return [int(x) for x in nums]

        # 再尝试最后几行中“像序列”的一行
        lines = [ln.strip() for ln in region.splitlines() if ln.strip()]
        for line in reversed(lines):
            nums = re.findall(r"-?\d+", line)
            if len(nums) >= 2 and (
                "," in line
                or " " in line
                or line.startswith("[")
                or line.startswith("(")
                or line.startswith("###")
            ):
                return [int(x) for x in nums]

    return None


def _format_sequence(seq: List[int]) -> str:
    return "### [" + ", ".join(str(x) for x in seq) + "]"


def _parse_topology_query(original: str) -> Tuple[Optional[int], List[Tuple[int, int]]]:
    """
    直接从题面解析：
    - 节点总数 n
    - 有向边列表 edges
    """
    text = str(original or "")

    # 解析类似 "The nodes are numbered from 0 to 42"
    m = re.search(r"nodes\s+are\s+numbered\s+from\s+0\s+to\s+(\d+)", text, flags=re.I)
    n = int(m.group(1)) + 1 if m else None

    # 解析所有 (u->v)
    edges = [(int(a), int(b)) for a, b in re.findall(r"\((\d+)\s*->\s*(\d+)\)", text)]

    # 若没有明确 n，则从边里反推
    if n is None and edges:
        n = max(max(u, v) for u, v in edges) + 1

    return n, edges


def _graph_stats(original: str) -> Tuple[int, int]:
    n, edges = _parse_topology_query(original)
    return int(n or 0), len(edges)


def _size_bucket(original: str) -> str:
    """
    small / medium / large
    """
    n, m = _graph_stats(original)
    if n <= 12 and m <= 40:
        return "small"
    if n <= 24 and m <= 120:
        return "medium"
    return "large"


def _normalize_preference_orders(n: int, preferred_orders: List[List[int]]) -> List[Dict[int, int]]:
    """
    把候选序列变成位置映射，只保留 [0, n-1] 且不重复的节点。
    """
    rank_maps: List[Dict[int, int]] = []

    for order in preferred_orders:
        seen = set()
        filtered: List[int] = []
        for x in order:
            if not isinstance(x, int):
                continue
            if x < 0 or x >= n:
                continue
            if x in seen:
                continue
            seen.add(x)
            filtered.append(x)

        if filtered:
            rank_maps.append({node: idx for idx, node in enumerate(filtered)})

    return rank_maps


def _solve_topology_with_preferences(
    original: str,
    preferred_orders: Optional[List[List[int]]] = None,
) -> Optional[List[int]]:
    """
    程序化拓扑排序：
    - 一定满足所有边约束
    - 若有 preferred_orders，则在 Kahn 选零入度节点时尽量贴近这些候选顺序
    """
    n, edges = _parse_topology_query(original)
    if n is None or n <= 0:
        return None

    outgoing: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n

    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            outgoing[u].append(v)
            indeg[v] += 1

    rank_maps = _normalize_preference_orders(n, preferred_orders or [])

    def priority(node: int):
        # 没有任何偏好时，退化成按节点编号选
        if not rank_maps:
            return (1, n, n, node)

        total = 0
        matched = 0
        best = n + 1

        for mp in rank_maps:
            pos = mp.get(node)
            if pos is None:
                total += n
            else:
                total += pos
                matched += 1
                if pos < best:
                    best = pos

        # 优先使用在偏好序列中出现过的点，再按平均靠前程度选
        return (0 if matched > 0 else 1, total, best, node)

    available = {i for i in range(n) if indeg[i] == 0}
    result: List[int] = []

    while available:
        u = min(available, key=priority)
        available.remove(u)
        result.append(u)

        for v in outgoing[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                available.add(v)

    if len(result) != n:
        # 有环或解析异常
        return None

    return result


def _validate_topology_response(state: Dict[str, Any]) -> bool:
    """
    语义校验：
    1. 能解析出一个序列
    2. 序列恰好包含所有节点且不重复
    3. 每条边 u->v 都满足 pos[u] < pos[v]
    """
    text = state.get("current", "")
    seq = _extract_sequence_from_text(text)
    if not seq:
        return False

    original = state.get("original", "")
    n, edges = _parse_topology_query(original)
    if n is None or n <= 0:
        return False

    if len(seq) != n:
        return False

    if set(seq) != set(range(n)):
        return False

    pos = {node: i for i, node in enumerate(seq)}

    for u, v in edges:
        if u not in pos or v not in pos:
            return False
        if pos[u] >= pos[v]:
            return False

    return True


def _count_edge_violations(original: str, seq: Optional[List[int]]) -> int:
    """
    统计候选序列违反了多少条边约束。
    """
    if not seq:
        return 10**9

    n, edges = _parse_topology_query(original)
    if n is None or len(seq) != n or set(seq) != set(range(n)):
        return 10**8

    pos = {node: i for i, node in enumerate(seq)}
    bad = 0
    for u, v in edges:
        if pos[u] >= pos[v]:
            bad += 1
    return bad


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for topological ordering.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    bucket = _size_bucket(original_query)
    n, m = _graph_stats(original_query)

    if part == "ConstraintExtraction":
        if bucket == "small":
            return f"""You are solving the ConstraintExtraction branch.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Goal:
Extract only a few useful precedence constraints implied by directed edges.

Rules:
1. Do NOT restate all edges.
2. Extract only 4 to 8 anchor constraints of the form "u before v".
3. Keep the answer concise.
4. End with exactly one final line beginning with ###

End with:
### <constraint summary>
"""
        if bucket == "medium":
            return f"""You are solving the ConstraintExtraction branch.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Goal:
Extract only a small anchor set of precedence constraints useful for checking a topological order.

Rules:
1. Do NOT list all edges.
2. Output only 6 to 10 anchor constraints of the form "u before v".
3. No long explanation.
4. End with exactly one final line beginning with ###

End with:
### <constraint summary>
"""
        return f"""You are solving the ConstraintExtraction branch.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Goal:
Extract a very small anchor set of precedence constraints for later checking.

Rules:
1. This is a large graph. Do NOT restate the full edge list.
2. Output only 6 to 10 short anchor constraints of the form "u before v".
3. No step-by-step explanation.
4. Keep the whole answer short.
5. End with exactly one final line beginning with ###

End with:
### <constraint summary>
"""

    if bucket == "small":
        return f"""You are solving the CandidateOrdering branch.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Goal:
Construct one complete candidate topological ordering containing every node exactly once.

Rules:
1. Use a valid topological-ordering process such as in-degree removal.
2. Every node must appear exactly once.
3. For every directed edge u->v, u must appear before v.
4. At most 2 short sentences of reasoning.
5. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""

    if bucket == "medium":
        return f"""You are solving the CandidateOrdering branch.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Goal:
Construct one complete candidate topological ordering containing every node exactly once.

Rules:
1. Think silently using a topological-ordering process such as in-degree removal.
2. Do NOT restate the edge list.
3. Do NOT write a long indegree table.
4. At most 1 short sentence before the final answer.
5. Every node must appear exactly once.
6. For every directed edge u->v, u must appear before v.
7. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""

    return f"""You are solving the CandidateOrdering branch.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Goal:
Construct one complete candidate topological ordering containing every node exactly once.

Rules:
1. This is a large graph. Think silently.
2. Do NOT restate the edges.
3. Do NOT list indegrees for all nodes.
4. Do NOT write step-by-step reasoning.
5. Prefer outputting ONLY the final answer line, or at most one very short sentence plus the final line.
6. Every node must appear exactly once.
7. For every directed edge u->v, u must appear before v.
8. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    bucket = _size_bucket(original_query)
    n, m = _graph_stats(original_query)

    if bucket == "large":
        return f"""You are given branch analyses for topological ordering.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Branch analyses:
{branch_bundle}

Task:
Return one valid topological ordering containing every node exactly once.

Rules:
1. This is a large graph. Do NOT restate the edges.
2. Do NOT write long explanations.
3. Use the branch hints only to confirm or repair a candidate ordering.
4. Output one ordering only.
5. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""
    return f"""You are given branch analyses for topological ordering.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Branch analyses:
{branch_bundle}

Task:
Use the extracted constraints to confirm or repair the candidate ordering.
Return one valid topological ordering containing every node exactly once.

Rules:
1. The ordering must contain every node exactly once.
2. For every directed edge u->v, u must appear before v.
3. Output one ordering only.
4. Keep the explanation minimal.
5. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""


def improve_prompt(original_query: str, current: str) -> str:
    bucket = _size_bucket(original_query)
    n, m = _graph_stats(original_query)

    if bucket == "large":
        return f"""Repair the topological ordering answer.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Previous answer:
{current}

Rules:
1. This is a large graph. Do NOT write a long explanation.
2. Output one repaired ordering only.
3. Every node must appear exactly once.
4. For every directed edge u->v, u must appear before v.
5. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""
    return f"""Repair the topological ordering answer.

Original problem:
{original_query}

Graph size:
nodes={n}, edges={m}

Previous answer:
{current}

Rules:
1. Every node appears exactly once.
2. For every directed edge u->v, u must appear before v.
3. Output one ordering only.
4. Keep the explanation minimal.
5. The LAST line must be exactly in this format and nothing after it:
### [0, 3, 1, 2, ...]
"""


def _contains_lengthy_reasoning_markers(text: str) -> bool:
    low = text.lower()
    markers = [
        "step-by-step",
        "step 1",
        "step 2",
        "kahn",
        "indegree",
        "in-degree",
        "adjacency",
        "scan edges",
        "let's go",
        "let us go",
        "i'll",
        "i will",
    ]
    return any(m in low for m in markers)


def search_score(state: Dict[str, Any]) -> float:
    text = _clean_response(state.get("current", ""))
    part = state.get("part", "")
    original = state.get("original", "")
    bucket = _size_bucket(original)

    if not text:
        return 100.0

    score = 0.0

    if "###" not in text:
        score += 8.0

    if len(text) < 8:
        score += 10.0

    score += min(len(text) / 6000.0, 3.0)

    if part == "ConstraintExtraction":
        low = text.lower()
        if not any(k in low for k in ["before", "preced", "constraint", "must appear", "u before v"]):
            score += 12.0

        if bucket == "large":
            if len(text) > 450:
                score += 12.0
            if _contains_lengthy_reasoning_markers(text):
                score += 8.0
        elif bucket == "medium":
            if len(text) > 700:
                score += 6.0
        else:
            if len(text) > 1200:
                score += 4.0

    elif part == "CandidateOrdering":
        seq = _extract_sequence_from_text(text)
        if seq is None:
            score += 20.0
        else:
            n, _ = _parse_topology_query(original)
            if n is not None:
                if len(seq) != n:
                    score += 12.0
                if len(set(seq)) != len(seq):
                    score += 12.0

            bad = _count_edge_violations(original, seq)
            if bad < 10**7:
                score += min(float(bad), 25.0)

            if _validate_topology_response(
                {
                    "original": original,
                    "current": text,
                }
            ):
                score -= 8.0

        if bucket == "large":
            if len(text) > 550:
                score += 16.0
            if len(text) > 900:
                score += 12.0
            if _contains_lengthy_reasoning_markers(text):
                score += 10.0
        elif bucket == "medium":
            if len(text) > 1000:
                score += 8.0
            if _contains_lengthy_reasoning_markers(text):
                score += 4.0
        else:
            if len(text) > 1800:
                score += 4.0

    elif part == "final":
        if not _validate_topology_response(state):
            score += 30.0
        else:
            score -= 10.0

        if bucket == "large":
            if len(text) > 300:
                score += 8.0
        elif bucket == "medium":
            if len(text) > 700:
                score += 4.0

    return float(max(score, 0.0))


def final_validator(state: Dict[str, Any]) -> bool:
    return _validate_topology_response(state)


def ground_truth(state: Dict[str, Any]) -> bool:
    # topology 是多解任务，不能按字符串匹配
    return _validate_topology_response(state)


class TopologyParser(BaseTaskParser):
    """
    关键修复：
    - 对 CandidateOrdering / aggregate / improve 阶段的序列做程序化 Kahn 修复
    - 把模型输出当作“偏好顺序”，最终答案由代码保证合法
    """

    def _repair_from_preferences(
        self,
        original: str,
        preferred_orders: List[List[int]],
    ) -> Optional[str]:
        solved = _solve_topology_with_preferences(original, preferred_orders)
        if solved is None:
            return None
        return _format_sequence(solved)

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        phase = state.get("phase", 0)
        part = state.get("part", "")
        original = state.get("original", "")

        # phase 0 保持父类逻辑：展开分支
        if phase == 0:
            return super().parse_generate_answer(state, texts)

        new_states: List[Dict] = []

        for text in texts:
            cleaned = _clean_response(text)
            if not cleaned:
                continue

            repaired = cleaned
            if part == "CandidateOrdering":
                seq = _extract_sequence_from_text(cleaned)
                preferred_orders = [seq] if seq else []
                fixed = self._repair_from_preferences(original, preferred_orders)
                if fixed:
                    repaired = fixed

            new_states.append(
                {
                    "current": repaired,
                    "phase": phase + 1,
                }
            )

        return new_states

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> List[Dict]:
        if not states:
            return []

        original = states[0].get("original", "")
        preferred_orders: List[List[int]] = []

        # 优先收集 CandidateOrdering 分支保留下来的候选
        for st in states:
            if st.get("part") == "CandidateOrdering":
                seq = _extract_sequence_from_text(st.get("current", ""))
                if seq:
                    preferred_orders.append(seq)

        # 也收集 aggregate 模型输出中的候选
        for text in texts:
            seq = _extract_sequence_from_text(_clean_response(text))
            if seq:
                preferred_orders.append(seq)

        fixed = self._repair_from_preferences(original, preferred_orders)

        new_states: List[Dict] = []
        if fixed:
            new_states.append(
                {
                    "current": fixed,
                    "phase": 3,
                    "part": "final",
                    "branch_goal": "",
                }
            )
            return new_states

        # 极端情况下程序修复失败，再退回父类
        return super().parse_aggregation_answer(states, texts)

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        original = state.get("original", "")
        preferred_orders: List[List[int]] = []

        seq0 = _extract_sequence_from_text(state.get("current", ""))
        if seq0:
            preferred_orders.append(seq0)

        for text in texts:
            seq = _extract_sequence_from_text(_clean_response(text))
            if seq:
                preferred_orders.append(seq)

        fixed = self._repair_from_preferences(original, preferred_orders)
        if fixed:
            return {
                "current": fixed,
                "phase": state.get("phase", 3),
                "part": state.get("part", "final"),
            }

        if not texts:
            return {}
        return {
            "current": _clean_response(texts[0]),
            "phase": state.get("phase", 3),
            "part": state.get("part", "final"),
        }


def build_graph():
    return build_task_graph(
        branches=BRANCHES,
        search_score_fn=search_score,
        final_validator=final_validator,
        ground_truth_fn=ground_truth,
        aggregate_responses=1,
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
    return TopologyParser(BRANCHES)