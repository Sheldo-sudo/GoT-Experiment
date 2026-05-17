"""
prompters/prompter.py
======================
ToT Prompter for GraphInstruct tasks — GraphWiz demo format.

ToT prompt structure:
---------------------
  1. build_thought_prompt  — 生成下一步推理分支
  2. build_eval_prompt     — 评估当前推理路径 (sure/maybe/impossible)
  3. build_final_prompt    — 基于最优路径得出最终答案
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demos import load_demo                             # noqa: E402

try:
    from graph_of_thoughts.prompter import Prompter    # type: ignore
except ImportError:
    class Prompter:                                     # type: ignore
        pass


class ToTPrompter(Prompter):
    """
    Tree-of-Thoughts prompter for GraphInstruct tasks.

    Three prompt types:
        build_thought_prompt  — generate next reasoning step (breadth branches)
        build_eval_prompt     — evaluate current thought path (sure/maybe/impossible)
        build_final_prompt    — extract final answer from best thought path
    """

    def __init__(self):
        self._demo_cache: Dict[str, str] = {}

    # ── Demo caching ──────────────────────────────────────────────────────
    def _demo(self, task: str) -> str:
        if task not in self._demo_cache:
            self._demo_cache[task] = load_demo(task)
        return self._demo_cache[task]

    # ── Thought generation prompt ─────────────────────────────────────────
    def build_thought_prompt(
        self,
        task: str,
        raw_question: str,
        partial_thought: str = "",
    ) -> str:
        """
        Generate the next reasoning step.

        If partial_thought is empty  → prompt for Step 1.
        If partial_thought is given  → prompt for the next step.
        """
        demo = self._demo(task)
        if partial_thought:
            return (
                f"{demo}\n\n"
                f"{raw_question.strip()}\n"
                f"A: Let's think step by step.\n"
                f"{partial_thought.strip()}\n"
                f"Next step:"
            )
        else:
            return (
                f"{demo}\n\n"
                f"{raw_question.strip()}\n"
                f"A: Let's think step by step.\nStep 1:"
            )

    # ── Evaluation prompt ─────────────────────────────────────────────────
    def build_eval_prompt(
        self,
        task: str,
        raw_question: str,
        thought: str,
    ) -> str:
        """
        Evaluate whether the current reasoning path is correct.

        Expected model output: 'sure', 'maybe', or 'impossible'.
        Scoring: sure=1.0, maybe=0.5, impossible=0.0
        """
        return (
            f"Evaluate whether the following reasoning step for a graph problem "
            f"is correct and helpful towards the final answer.\n"
            f"Answer with exactly one word: 'sure', 'maybe', or 'impossible'.\n\n"
            f"Problem: {raw_question.strip()}\n\n"
            f"Reasoning so far:\n{thought.strip()}\n\n"
            f"Is this reasoning correct and on track to solve the problem? "
            f"Answer (sure/maybe/impossible):"
        )

    # ── Final answer prompt ───────────────────────────────────────────────
    def build_final_prompt(
        self,
        task: str,
        raw_question: str,
        best_thought: str,
    ) -> str:
        """
        Extract the final answer from the best reasoning path.
        Uses the same demo format as I/O and CoT for consistent parsing.
        """
        demo = self._demo(task)
        return (
            f"{demo}\n\n"
            f"{raw_question.strip()}\n"
            f"A: {best_thought.strip()}\n"
            f"Therefore, the final answer is ###"
        )

    # ── Fallback: direct CoT prompt (used if tree search fails) ──────────
    def build_prompt_from_raw(self, task: str, raw_question: str) -> str:
        """
        Fallback to CoT-style prompt when no valid candidates survive pruning.
        Ensures the runner always gets a usable output.
        """
        demo = self._demo(task)
        return f"{demo}\n\n{raw_question.strip()}\nA: Let's think step by step."

    # ── GoT Controller interface stubs ────────────────────────────────────
    def generate_prompt(
        self,
        num_branches: int,
        original: str,
        source: Optional[str],
        target: Optional[str],
        **kwargs,
    ) -> str:
        import json
        state = json.loads(original) if isinstance(original, str) else original
        return self.build_thought_prompt(
            task=state["task"],
            raw_question=state.get("raw_prompt", ""),
            partial_thought=state.get("partial_thought", ""),
        )

    def aggregation_prompt(self, state_dicts: List, **kw) -> str:
        return ""

    def improve_prompt(self, state_dicts: List, **kw) -> str:
        return ""

    def validation_prompt(self, state_dicts: List, **kw) -> str:
        return ""

    def score_prompt(self, state_dicts: List, **kw) -> str:
        return ""