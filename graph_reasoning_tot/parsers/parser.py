"""
parsers/parser.py
==================
Answer extraction + scoring for all 9 GraphInstruct tasks.

CONFIRMED demo formats (from real GraphWiz demo files):
--------------------------------------------------------
Task          Answer tag format                   GT example
-----------   ---------------------------------   ----------
connectivity  ### Yes.  / ### No.                 "Yes" / "No"
cycle         ### Yes.  / ### No.                 "Yes" / "No"
bipartite     ### Yes.  / ### No.                 "Yes" / "No"
hamilton      ### Yes, [0,1,4,5,3,2].             "Yes" / "No"
subgraph      ### Yes.  / ### No.                 "Yes" / "No"
shortest_path ### 3.    / ### 5.                  "3"   / "5"
maximum_flow  ### 10.   / ### 8.                  "10"  / "8"
topology      ### [0, 1, 2, 3, 4].               "0 1 2 3 4"
triplet       ### 21.   / ### 16.                 "21"  / "16"

CRITICAL DISCOVERY — triplet task:
------------------------------------
  triplet.txt is NOT triangle counting.
  It asks: "Find the maximum sum of the weights of three interconnected nodes."
  Input includes node weights: [0, 2] [1, 9] [2, 6] ...
  GT is an integer (the max weight sum), e.g. "21", "16"
  Scoring: exact integer match against GT string (no NetworkX needed)

CRITICAL DISCOVERY — hamilton answer format:
--------------------------------------------
  ### Yes, [0,1,4,5,3,2].   ← Yes followed by comma + path
  ### No.
  We extract only Yes/No, ignoring the path.

CRITICAL DISCOVERY — topology answer format:
--------------------------------------------
  ### [0, 1, 2, 3, 4].      ← list with brackets and spaces
  Sequence extraction must handle brackets.
  Validation via NetworkX (any valid topological order accepted).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

try:
    from graph_of_thoughts.parser import Parser       # type: ignore
except ImportError:
    class Parser:                                      # type: ignore
        pass

logger = logging.getLogger(__name__)


# ── Core extraction helpers ───────────────────────────────────────────────

def _hashtag_content(text: str) -> Optional[str]:
    """
    Extract content after ### marker.
    Handles:
        ### Yes.
        ### No.
        ### Yes, [0,1,4,5,3,2].
        ### 21.
        ### [0, 1, 2, 3, 4].
        ### 10.
    Returns the raw string after ###, stripped of trailing period/whitespace.
    """
    m = re.search(r"###\s*(.+?)\.?\s*$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _yes_no(text: str) -> Optional[str]:
    """
    Extract Yes/No from text. Checks ### tag first, then full text.
    Handles  '### Yes, [0,1,4,5,3,2].'  correctly (takes 'Yes' before comma).
    """
    tag = _hashtag_content(text)
    if tag:
        # Yes may be followed by comma + path in hamilton: "Yes, [0,1,4,5,3,2]"
        t = tag.lower()
        if t.startswith("yes"):
            return "Yes"
        if t.startswith("no"):
            return "No"

    # Fallback: scan lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in [lines[0] if lines else "", lines[-1] if lines else "", text]:
        t = line.lower()
        if re.search(r"\byes\b", t):
            return "Yes"
        if re.search(r"\bno\b", t):
            return "No"
    return None


def _number(text: str) -> Optional[float]:
    """
    Extract numeric answer. Checks ### tag first.
    Ignores <<...>> calculation annotations in shortest_path demos.
    """
    tag = _hashtag_content(text)
    if tag:
        m = re.search(r"-?\d+(?:\.\d+)?", tag)
        if m:
            return float(m.group())
    # Fallback: last number in text (more reliable than first)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None


def _integer(text: str) -> Optional[int]:
    v = _number(text)
    return int(v) if v is not None else None


def _sequence(text: str) -> Optional[List[int]]:
    """
    Extract node sequence for topology.
    Handles: ### [0, 1, 2, 3, 4].
    Also handles bare: 0 1 2 3 4
    Returns list of ints, or None if fewer than 2 numbers found.
    """
    tag = _hashtag_content(text)
    if tag:
        nums = re.findall(r"\d+", tag)
        if len(nums) >= 2:
            return [int(n) for n in nums]

    # Fallback: look for the longest run of integers on any line
    best: List[int] = []
    for line in text.splitlines():
        nums = re.findall(r"\d+", line)
        if len(nums) > len(best):
            best = [int(n) for n in nums]
    return best if len(best) >= 2 else None


# ── Public extraction entry point ─────────────────────────────────────────

def extract_answer(raw: str, task: str) -> Any:
    """
    Parse raw LLM output into a typed answer.

    Returns
    -------
    "Yes" / "No"    connectivity, cycle, bipartite, hamilton, subgraph
    float           shortest_path, maximum_flow
    int             triplet   (max weight sum of 3 interconnected nodes)
    List[int]       topology
    None            if extraction fails
    """
    if task in ("connectivity", "cycle", "bipartite", "hamilton", "subgraph"):
        return _yes_no(raw)

    if task in ("shortest_path", "maximum_flow"):
        return _number(raw)

    if task == "triplet":           # ← weighted node clique sum, NOT triangle count
        return _integer(raw)

    if task == "topology":
        return _sequence(raw)

    return None


# ── Scoring ───────────────────────────────────────────────────────────────

def _norm_yesno(s: str) -> str:
    # Strip GraphWiz "### " prefix from ground_truth if present (e.g. "### Yes" -> "Yes")
    t = str(s).strip()
    if t.startswith("###"):
        t = t.lstrip("#").strip()
    return "Yes" if t.lower() in ("yes", "true", "1") else "No"


def _topology_valid(seq: List[int], raw_prompt: str) -> bool:
    """Validate any topological order via NetworkX using edges from prompt."""
    if not _HAS_NX:
        return False
    G = nx.DiGraph()
    # Edge formats in topology prompts: (i->j)
    for m in re.finditer(r"\((\d+)->(\d+)\)", raw_prompt):
        G.add_edge(int(m.group(1)), int(m.group(2)))
    nodes = set(G.nodes())
    if set(seq) != nodes or len(seq) != len(nodes):
        return False
    pos = {node: idx for idx, node in enumerate(seq)}
    return all(pos[u] < pos[v] for u, v in G.edges())


def score_answer(parsed: Any, task: str, ground_truth: str) -> float:
    """
    Compare parsed answer against ground_truth string from dataset JSON.
    Returns 1.0 (correct) or 0.0 (wrong / parse failure).

    ground_truth is the raw 'answer' field from *_test.json, e.g.:
        "Yes", "No", "3", "21", "0 1 2 3 4"
    """
    if parsed is None:
        return 0.0
    gt = ground_truth.strip()

    # ── Binary tasks ──────────────────────────────────────────────────────
    if task in ("connectivity", "cycle", "bipartite", "hamilton", "subgraph"):
        return 1.0 if _norm_yesno(str(parsed)) == _norm_yesno(gt) else 0.0

    # ── Numeric tasks ─────────────────────────────────────────────────────
    if task in ("shortest_path", "maximum_flow"):
        try:
            return 1.0 if abs(float(parsed) - float(gt)) < 1e-6 else 0.0
        except (ValueError, TypeError):
            return 0.0

    # ── Triplet: max weight sum of 3 interconnected nodes ─────────────────
    if task == "triplet":
        try:
            return 1.0 if int(parsed) == int(gt) else 0.0
        except (ValueError, TypeError):
            return 0.0

    # ── Topology: exact GT match (conservative; use score_answer_with_prompt
    #              for NetworkX-validated flexible matching) ─────────────────
    if task == "topology":
        if not isinstance(parsed, list):
            return 0.0
        gt_seq = [int(x) for x in gt.split() if x.isdigit()]
        return 1.0 if parsed == gt_seq else 0.0

    return 0.0


def score_answer_with_prompt(
    parsed: Any,
    task: str,
    ground_truth: str,
    raw_prompt: str,
) -> float:
    """
    Enhanced scorer for topology: accepts ANY valid topological ordering.
    For all other tasks delegates to score_answer().
    """
    if task != "topology":
        return score_answer(parsed, task, ground_truth)
    if not isinstance(parsed, list):
        return 0.0
    gt_seq = [int(x) for x in ground_truth.split() if x.isdigit()]
    if parsed == gt_seq:
        return 1.0
    # Accept any other valid topological order
    return 1.0 if _topology_valid(parsed, raw_prompt) else 0.0


# ── GoT Parser class (forward-compat for CoT/ToT/GoT runners) ────────────

class GraphInstructParser(Parser):

    def parse_generate_answer(self, state, texts):
        task = state.get("task", "")
        results = []
        for text in texts:
            s = dict(state)
            s["raw_answer"]    = text
            s["parsed_answer"] = extract_answer(text, task)
            s["score"]         = 0.0
            results.append(s)
        return results

    def parse_score_answer(self, states, texts):
        scores = []
        for s in states:
            sc = score_answer_with_prompt(
                s.get("parsed_answer"),
                s.get("task", ""),
                s.get("ground_truth", ""),
                s.get("raw_prompt", ""),
            )
            scores.append(sc)
        return scores

    def parse_aggregation_answer(self, orig, curr, texts):
        return {}

    def parse_improve_answer(self, state, texts):
        return {}

    def parse_validation_answer(self, state, texts):
        return [True] * len(texts)