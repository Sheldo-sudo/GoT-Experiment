# prompters/prompter.py
"""
GoT Prompter for GraphInstruct tasks.

GoT Operation Flow per task:
  1. Generate  — 生成初始推理（多个独立分支）
  2. Score     — 对每个分支打分（0-10）
  3. Aggregate — 合并最优分支
  4. Refine    — 基于聚合结果精化最终答案
"""

from __future__ import annotations
import os, sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demos import load_demo

try:
    from graph_of_thoughts.prompter import Prompter
except ImportError:
    class Prompter:
        pass


class GoTPrompter(Prompter):

    def __init__(self):
        self._demo_cache: Dict[str, str] = {}

    def _demo(self, task: str) -> str:
        if task not in self._demo_cache:
            self._demo_cache[task] = load_demo(task)
        return self._demo_cache[task]

    # ── 1. Generate: 生成初始推理 ──────────────────────────────────────
    def build_generate_prompt(self, task: str, raw_question: str) -> str:
        demo = self._demo(task)
        return (
            f"{demo}\n\n"
            f"{raw_question.strip()}\n"
            f"A: Let's reason step by step about this graph problem."
        )

    # ── 2. Score: 对推理打分 ────────────────────────────────────────────
    def build_score_prompt(self, task: str, raw_question: str, thought: str) -> str:
        return (
            f"Rate the correctness of the following graph reasoning on a scale "
            f"from 0 to 10, where 10 is perfectly correct.\n"
            f"Only output a single integer between 0 and 10.\n\n"
            f"Problem: {raw_question.strip()}\n\n"
            f"Reasoning:\n{thought.strip()}\n\n"
            f"Score (0-10):"
        )

    # ── 3. Aggregate: 合并多个推理分支 ─────────────────────────────────
    def build_aggregate_prompt(
        self, task: str, raw_question: str, thoughts: List[str]
    ) -> str:
        formatted = "\n\n".join(
            f"Reasoning {i+1}:\n{t.strip()}"
            for i, t in enumerate(thoughts)
        )
        return (
            f"You are given multiple reasoning attempts for a graph problem. "
            f"Combine the best insights from each to produce a single, "
            f"improved reasoning that leads to the correct answer.\n\n"
            f"Problem: {raw_question.strip()}\n\n"
            f"{formatted}\n\n"
            f"Combined best reasoning:"
        )

    # ── 4. Refine: 精化最终答案 ─────────────────────────────────────────
    def build_refine_prompt(
        self, task: str, raw_question: str, aggregated: str
    ) -> str:
        demo = self._demo(task)
        return (
            f"{demo}\n\n"
            f"{raw_question.strip()}\n"
            f"A: {aggregated.strip()}\n"
            f"Therefore, the final answer is ###"
        )

    # ── Fallback ────────────────────────────────────────────────────────
    def build_prompt_from_raw(self, task: str, raw_question: str) -> str:
        demo = self._demo(task)
        return f"{demo}\n\n{raw_question.strip()}\nA: Let's think step by step."

    # ── GoT Controller interface ────────────────────────────────────────
    def generate_prompt(self, num_branches, original, source, target, **kw) -> str:
        import json
        state = json.loads(original) if isinstance(original, str) else original
        return self.build_generate_prompt(state["task"], state.get("raw_prompt", ""))

    def score_prompt(self, state_dicts: List, **kw) -> str:
        if not state_dicts:
            return ""
        s = state_dicts[0]
        return self.build_score_prompt(s["task"], s["raw_prompt"], s.get("thought", ""))

    def aggregation_prompt(self, state_dicts: List, **kw) -> str:
        if not state_dicts:
            return ""
        task     = state_dicts[0]["task"]
        question = state_dicts[0]["raw_prompt"]
        thoughts = [s.get("thought", "") for s in state_dicts]
        return self.build_aggregate_prompt(task, question, thoughts)

    def improve_prompt(self, state_dicts: List, **kw) -> str:
        if not state_dicts:
            return ""
        s = state_dicts[0]
        return self.build_refine_prompt(s["task"], s["raw_prompt"], s.get("thought", ""))

    def validation_prompt(self, state_dicts: List, **kw) -> str:
        return ""