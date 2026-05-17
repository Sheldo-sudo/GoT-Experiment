"""
prompters/prompter.py
======================
I/O Prompter for GraphInstruct tasks — GraphWiz demo format.

Confirmed demo format (from connectivity.txt):
----------------------------------------------
  Determine if there is a path between two nodes in the graph.
  Note that (i,j) means that node i and node j are connected ...
  ...
  Below are several examples:
  Q: The nodes are numbered from 0 to 5, and the edges are: ...
  A: Node 1 is in the connected block ... ### No.
  Q: ...
  A: ... ### Yes.

I/O prompt construction (simplest possible):
--------------------------------------------
  {full demo file content}           ← loaded verbatim, no modification
  
  Q: {input_prompt from dataset}     ← appended directly, no reformatting
  A:                                 ← model continues from here

Key insight: input_prompt in GraphInstruct-Test is already in the exact
same surface form as the demo Q: lines, so we never need to reconstruct
edges — just paste the question as-is.
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


class IOPrompter(Prompter):
    """
    Few-shot I/O prompter.

    The only public method used by run_io.py is:
        build_prompt_from_raw(task, raw_question) -> str

    All GoT Controller interface methods are stubbed for forward-compatibility
    with future CoT / ToT / GoT runners.
    """

    def __init__(self):
        self._demo_cache: Dict[str, str] = {}

    # ── Demo caching ──────────────────────────────────────────────────────
    def _demo(self, task: str) -> str:
        if task not in self._demo_cache:
            self._demo_cache[task] = load_demo(task)
        return self._demo_cache[task]

    # ── Main entry point for run_io.py ────────────────────────────────────
    def build_prompt_from_raw(self, task: str, raw_question: str) -> str:
        """
        Build the complete I/O prompt.

        Structure:
            {demo file — verbatim}
            
            Q: {raw_question}
            A:

        The raw_question is the input_prompt field from the dataset JSON,
        used verbatim without any reformatting.
        """
        demo = self._demo(task)
        # input_prompt already contains full "Q: The nodes are..." prefix.
        # Do NOT add another Q: here — that causes double Q:Q: and confuses the model.
        return f"{demo}\n\n{raw_question.strip()}\nA:"

    # ── GoT Controller interface stubs ────────────────────────────────────
    # Kept for forward-compatibility when this prompter is reused in
    # CoT / ToT / GoT runners.

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
        return self.build_prompt_from_raw(
            task=state["task"],
            raw_question=state.get("raw_prompt", ""),
        )

    def aggregation_prompt(self, state_dicts: List, **kw) -> str:
        return ""

    def improve_prompt(self, state_dicts: List, **kw) -> str:
        return ""

    def validation_prompt(self, state_dicts: List, **kw) -> str:
        return ""

    def score_prompt(self, state_dicts: List, **kw) -> str:
        return ""