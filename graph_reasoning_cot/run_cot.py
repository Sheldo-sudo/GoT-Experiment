"""
run_io.py
=========
I/O baseline for GraphInstruct tasks using LLaMA-2-7B-Chat.

Server paths (pre-configured, matching actual server layout)
-------------------------------------------------------------
  Model : /home/wangzq28/GoT-Experiment/models/LLaMA2-7B-chat
  Test  : /home/wangzq28/GoT-Experiment/data/GraphInstruct-Test/
            → connectivity_test.json, cycle_test.json, ... (JSON arrays)
  Train : /home/wangzq28/GoT-Experiment/data/GraphInstruct/GraphInstruct.json

Dataset format (per entry in each *_test.json)
-----------------------------------------------
  {
    "index":        0,
    "input_prompt": "The nodes are numbered from 0 to 4, ...",
    "answer":       "Yes"          # ground truth string
    # optional: "task", "node_range", "edge_range"
  }

Demo format (GraphWiz CoT style, confirmed from connectivity.txt)
-----------------------------------------------------------------
  - Multi-line preamble describing the task
  - Q: ... A: ... ### Yes/No  (CoT reasoning with final answer tagged)

I/O prompt construction
-----------------------
  {full demo file content}
  
  Q: {input_prompt from dataset}
  A:

Answer extraction
-----------------
  Looks for  ### Yes  /  ### No  first (GraphWiz CoT format),
  then falls back to bare Yes / No in first line.

Usage
-----
  # Dry-run (no GPU, 5 samples per task)
  python run_io.py --dry_run --max_samples 5

  # V100 feasibility check (50 samples)
  python run_io.py --max_samples 50

  # Full dataset, all tasks
  python run_io.py

  # Single task
  python run_io.py --task connectivity

  # Custom paths
  python run_io.py --data_dir /path/to/test --output_dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Package root ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from language_models.llama_local import LlamaLocal
from prompters.prompter import IOPrompter
from parsers.parser import extract_answer, score_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_io")

# ── Server paths ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "/home/wangzq28/GoT-Experiment/models/LLaMA2-7B-chat"
DEFAULT_TEST_DIR   = "/home/wangzq28/GoT-Experiment/data/GraphInstruct-Test"
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results", "cot")

# ── Task definitions ──────────────────────────────────────────────────────
ALL_TASKS = [
    "connectivity",
    "cycle",
    "bipartite",
    "shortest_path",
    "triangle",
    "subgraph",
    "hamilton",
    "maximum_flow",
    "topology",
]

# Maps task name → actual filename stem in GraphInstruct-Test/
# Confirmed layout: {stem}_test.json
TASK_FILE_STEMS: Dict[str, str] = {
    "connectivity":  "connectivity",
    "cycle":         "cycle",
    "bipartite":     "bipartite",
    "shortest_path": "shortest",
    "triangle":      "triangle",
    "subgraph":      "substructure",
    "hamilton":      "hamilton",
    "maximum_flow":  "flow",
    "topology":      "topology",
}


# ── Data loading ──────────────────────────────────────────────────────────

def load_task_data(
    data_dir:    str,
    task:        str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load samples for `task` from GraphInstruct-Test directory.

    File layout (confirmed from server):
        {data_dir}/{stem}_test.json   — JSONL: one JSON object per line
                                        (NOT a JSON array — json.load() fails)

    Each line is a JSON object with:
        input_prompt : str   — the full question as-is
        answer       : str   — ground truth (e.g. "Yes", "3", "0 2 1 3")
        index        : int   — optional sample index
    """
    stem     = TASK_FILE_STEMS[task]
    filename = f"{stem}_test.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        logger.warning("File not found: %s — skipping task %s", filepath, task)
        return []

    # ── Parse JSONL: one JSON object per line ────────────────────────────
    # The files use .json extension but are JSONL format (confirmed by
    # "Extra data: line 2 column 1" error from json.load()).
    items = []
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d in %s: %s",
                               lineno, filename, e)
                continue

    samples = []
    for item in items:
        sample = {
            "task":          task,
            "raw_prompt":    item.get("input_prompt", ""),
            "ground_truth":  str(item.get("answer", "")).strip(),
            "index":         item.get("index", len(samples)),
        }
        samples.append(sample)

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info("Loaded %d samples for task=%-16s from %s",
                len(samples), task, filename)
    return samples


# ── Inference loop ────────────────────────────────────────────────────────

def run_cot_task(
    task:     str,
    samples:  List[Dict],
    lm:       LlamaLocal,
    prompter: IOPrompter,
    dry_run:  bool = False,
) -> List[Dict]:
    """
    Run I/O inference for one task. Returns list of result dicts.

    Prompt structure:
        {demo file content}         ← few-shot CoT examples (GraphWiz format)

        Q: {input_prompt}           ← raw question from dataset, no modification
        A:                          ← model completes from here
    """
    results = []
    logger.info("─── Task: %-14s │ %d samples ───", task, len(samples))

    for i, sample in enumerate(samples):
        prompt = prompter.build_prompt_from_raw(
            task=task,
            raw_question=sample["raw_prompt"],
        )

        t0 = time.time()
        if dry_run:
            raw_output = "Yes"          # dummy — pipeline validation only
        else:
            outputs    = lm.query(prompt, num_responses=1)
            raw_output = outputs[0] if outputs else ""
        latency = time.time() - t0

        parsed = extract_answer(raw_output, task)
        score  = score_answer(parsed, task, sample["ground_truth"])

        results.append({
            "index":         sample["index"],
            "task":          task,
            "prompt":        prompt,
            "raw_answer":    raw_output,
            "parsed_answer": str(parsed),
            "ground_truth":  sample["ground_truth"],
            "score":         score,
            "latency_s":     round(latency, 4),
        })

        # Progress logging
        if dry_run or (i + 1) % 10 == 0 or i == 0:
            running_acc = sum(r["score"] for r in results) / len(results)
            logger.info(
                "  [%3d/%3d] acc=%.3f | lat=%.2fs | raw=%r",
                i + 1, len(samples), running_acc, latency,
                raw_output[:60].replace("\n", " "),
            )

    return results


# ── Results serialisation ─────────────────────────────────────────────────

def save_results(
    results:     List[Dict],
    task:        str,
    output_dir:  str,
    token_usage: Dict,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-sample JSONL (one result per line for easy streaming/analysis)
    out_path = os.path.join(output_dir, f"cot_{task}_{ts}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n        = len(results)
    correct  = sum(r["score"] for r in results)
    accuracy = correct / n if n else 0.0
    avg_lat  = sum(r["latency_s"] for r in results) / n if n else 0.0

    summary = {
        "method":        "cot",
        "task":          task,
        "n_samples":     n,
        "n_correct":     int(correct),
        "accuracy":      round(accuracy, 4),
        "avg_latency_s": round(avg_lat, 4),
        "token_usage":   token_usage,
        "timestamp":     ts,
        "result_file":   out_path,
    }
    summary_path = os.path.join(output_dir, f"cot_{task}_{ts}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "  ✓ %-14s │ acc=%.4f (%d/%d) │ avg_lat=%.2fs │ tok_in=%d tok_out=%d",
        task, accuracy, int(correct), n, avg_lat,
        token_usage.get("prompt_tokens", 0),
        token_usage.get("completion_tokens", 0),
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="I/O baseline experiment for GraphInstruct (9 tasks)"
    )
    p.add_argument(
        "--config", default=None,
        help="Path to LM config JSON. Uses server defaults if omitted.",
    )
    p.add_argument(
        "--data_dir", default=DEFAULT_TEST_DIR,
        help=f"GraphInstruct-Test directory (default: {DEFAULT_TEST_DIR})",
    )
    p.add_argument(
        "--task", default="all",
        choices=ALL_TASKS + ["all"],
        help="Task to run (default: all)",
    )
    p.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Results directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per task. None = full dataset.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Skip LLM calls — validate pipeline only (no GPU required)",
    )
    return p.parse_args()


def main():
    args  = parse_args()
    tasks = ALL_TASKS if args.task == "all" else [args.task]

    # ── Init LM ──────────────────────────────────────────────────────────
    lm = LlamaLocal(config_path=args.config or "")
    if not args.config:
        lm.model_path     = DEFAULT_MODEL_PATH
        lm.backend        = "vllm"
        lm.max_new_tokens = 256
        lm.temperature    = 0.0
        lm.top_p          = 1.0

    prompter = IOPrompter()

    logger.info("Mode   : %s", "DRY-RUN (no GPU)" if args.dry_run else "LIVE")
    logger.info("Model  : %s", lm.model_path)
    logger.info("Data   : %s", args.data_dir)
    logger.info("Tasks  : %s", tasks)
    logger.info("Limit  : %s samples/task",
                args.max_samples if args.max_samples else "all")

    all_summaries: List[Dict] = []

    for task in tasks:
        samples = load_task_data(args.data_dir, task, args.max_samples)
        if not samples:
            logger.warning("No samples for %s — skipping.", task)
            continue

        lm.reset_token_counters()
        results = run_cot_task(task, samples, lm, prompter, args.dry_run)
        summary = save_results(results, task, args.output_dir, lm.token_usage())
        all_summaries.append(summary)

    # ── Cross-task summary ────────────────────────────────────────────────
    if all_summaries:
        logger.info("")
        logger.info("=" * 72)
        logger.info("%-18s %7s %6s %8s %10s %10s",
                    "Task", "Acc", "N", "Correct", "Tok-In", "Tok-Out")
        logger.info("-" * 72)
        for s in all_summaries:
            logger.info(
                "%-18s %7.4f %6d %8d %10d %10d",
                s["task"], s["accuracy"], s["n_samples"], s["n_correct"],
                s["token_usage"].get("prompt_tokens", 0),
                s["token_usage"].get("completion_tokens", 0),
            )
        logger.info("=" * 72)
        avg_acc = sum(s["accuracy"] for s in all_summaries) / len(all_summaries)
        logger.info("Overall average accuracy: %.4f  (%d tasks)",
                    avg_acc, len(all_summaries))

        # Save master summary
        master_path = os.path.join(
            args.output_dir,
            f"cot_all_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(master_path, "w") as f:
            json.dump({
                "method":       "cot",
                "tasks":        all_summaries,
                "average_acc":  round(avg_acc, 4),
            }, f, indent=2)
        logger.info("Master summary → %s", master_path)


if __name__ == "__main__":
    main()