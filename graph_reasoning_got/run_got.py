"""
run_got.py
==========
Graph-of-Thoughts baseline for GraphInstruct tasks.

GoT Operation Graph per sample:
  Generate(k=3) → Score → KeepBest(n=2) → Aggregate → Refine → Extract Answer

LLM calls per sample:
  - Generate:  1 call (k responses in one batch)
  - Score:     k calls (one per thought)
  - Aggregate: 1 call
  - Refine:    1 call
  Total: k + 3 calls  (default k=3 → 6 calls/sample)

Usage
-----
  python run_got.py --dry_run --max_samples 3
  python run_got.py --config config.json --task connectivity --max_samples 10
  python run_got.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from language_models.llama_local import LlamaLocal
from prompters.prompter import GoTPrompter
from parsers.parser import extract_answer, score_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_got")

DEFAULT_MODEL_PATH = "/home/wangzq28/GoT-Experiment/models/LLaMA2-7B-chat"
DEFAULT_TEST_DIR   = "/home/wangzq28/GoT-Experiment/data/GraphInstruct-Test"
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results", "got")
DEFAULT_K          = 3   # number of Generate branches

ALL_TASKS = [
    "connectivity", "cycle", "bipartite", "shortest_path",
    "triangle", "subgraph", "hamilton", "maximum_flow", "topology",
]

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
    data_dir: str, task: str, max_samples: Optional[int] = None
) -> List[Dict]:
    stem     = TASK_FILE_STEMS[task]
    filepath = os.path.join(data_dir, f"{stem}_test.json")
    if not os.path.exists(filepath):
        logger.warning("File not found: %s", filepath)
        return []

    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append({
                "task":         task,
                "raw_prompt":   item.get("input_prompt", ""),
                "ground_truth": str(item.get("answer", "")).strip(),
                "index":        item.get("index", len(samples)),
            })

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


# ── GoT core operation graph ──────────────────────────────────────────────

def _parse_score(score_output: str) -> float:
    """Extract numeric score 0-10 from model output."""
    import re
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', score_output)
    for n in nums:
        v = float(n)
        if 0 <= v <= 10:
            return v
    return 5.0  # default if parsing fails


def got_solve(
    task:         str,
    raw_question: str,
    lm:           LlamaLocal,
    prompter:     GoTPrompter,
    k:            int  = DEFAULT_K,
    dry_run:      bool = False,
) -> Tuple[str, int]:
    """
    GoT operation graph:
      Generate(k) → Score each → KeepBest(2) → Aggregate → Refine → Answer

    Returns: (raw_final_output, total_lm_calls)
    """
    total_calls = 0

    # ── Op 1: Generate k independent thoughts ────────────────────────────
    gen_prompt = prompter.build_generate_prompt(task, raw_question)
    if dry_run:
        thoughts = [f"Dry-run thought {i}" for i in range(k)]
    else:
        thoughts    = lm.query(gen_prompt, num_responses=k)
        total_calls += 1

    # ── Op 2: Score each thought ──────────────────────────────────────────
    scored: List[Tuple[str, float]] = []
    for thought in thoughts:
        score_prompt = prompter.build_score_prompt(task, raw_question, thought)
        if dry_run:
            sc = 7.0
        else:
            score_out = lm.query(score_prompt, num_responses=1)
            sc        = _parse_score(score_out[0] if score_out else "5")
            total_calls += 1
        scored.append((thought, sc))

    # ── Op 3: KeepBest — keep top-2 by score ─────────────────────────────
    scored.sort(key=lambda x: x[1], reverse=True)
    best_thoughts = [t for t, _ in scored[:2]]

    # ── Op 4: Aggregate best thoughts ────────────────────────────────────
    agg_prompt = prompter.build_aggregate_prompt(task, raw_question, best_thoughts)
    if dry_run:
        aggregated = "Dry-run aggregated thought"
    else:
        agg_out    = lm.query(agg_prompt, num_responses=1)
        aggregated = agg_out[0] if agg_out else best_thoughts[0]
        total_calls += 1

    # ── Op 5: Refine — extract final answer ──────────────────────────────
    refine_prompt = prompter.build_refine_prompt(task, raw_question, aggregated)
    if dry_run:
        raw_output = "Yes"
    else:
        refine_out = lm.query(refine_prompt, num_responses=1)
        raw_output = refine_out[0] if refine_out else ""
        total_calls += 1

    return raw_output, total_calls


# ── Inference loop ────────────────────────────────────────────────────────

def run_got_task(
    task:     str,
    samples:  List[Dict],
    lm:       LlamaLocal,
    prompter: GoTPrompter,
    k:        int  = DEFAULT_K,
    dry_run:  bool = False,
) -> List[Dict]:
    results        = []
    total_lm_calls = 0
    correct_so_far = 0

    pbar = tqdm(
        samples,
        desc=f"{task:<16}",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
    )

    for sample in pbar:
        t0 = time.time()

        raw_output, calls = got_solve(
            task         = task,
            raw_question = sample["raw_prompt"],
            lm           = lm,
            prompter     = prompter,
            k            = k,
            dry_run      = dry_run,
        )
        latency = time.time() - t0
        total_lm_calls += calls

        parsed = extract_answer(raw_output, task)
        score  = score_answer(parsed, task, sample["ground_truth"])
        correct_so_far += score

        results.append({
            "index":         sample["index"],
            "task":          task,
            "raw_answer":    raw_output,
            "parsed_answer": str(parsed),
            "ground_truth":  sample["ground_truth"],
            "score":         score,
            "latency_s":     round(latency, 4),
            "lm_calls":      calls,
            "got_k":         k,
        })

        running_acc = correct_so_far / len(results)
        pbar.set_postfix({
            "acc":   f"{running_acc:.3f}",
            "lat":   f"{latency:.1f}s",
            "calls": calls,
        })

    pbar.close()

    n         = len(results)
    acc       = correct_so_far / n if n else 0.0
    avg_lat   = sum(r["latency_s"] for r in results) / n if n else 0.0
    avg_calls = total_lm_calls / n if n else 0.0
    logger.info(
        "✓ %-16s │ acc=%.4f (%d/%d) │ avg_lat=%.2fs │ "
        "avg_calls=%.1f │ total_calls=%d",
        task, acc, int(correct_so_far), n,
        avg_lat, avg_calls, total_lm_calls,
    )
    return results


# ── Results serialisation ─────────────────────────────────────────────────

def save_results(
    results: List[Dict], task: str, output_dir: str,
    token_usage: Dict, k: int,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = os.path.join(output_dir, f"got_{task}_{ts}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n         = len(results)
    correct   = sum(r["score"] for r in results)
    accuracy  = correct / n if n else 0.0
    avg_lat   = sum(r["latency_s"] for r in results) / n if n else 0.0
    avg_calls = sum(r["lm_calls"] for r in results) / n if n else 0.0

    summary = {
        "method":        "got",
        "task":          task,
        "got_k":         k,
        "n_samples":     n,
        "n_correct":     int(correct),
        "accuracy":      round(accuracy, 4),
        "avg_latency_s": round(avg_lat, 4),
        "avg_lm_calls":  round(avg_calls, 2),
        "token_usage":   token_usage,
        "timestamp":     ts,
        "result_file":   out_path,
    }
    with open(os.path.join(output_dir, f"got_{task}_{ts}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Graph-of-Thoughts baseline for GraphInstruct (9 tasks)"
    )
    p.add_argument("--config",      default=None)
    p.add_argument("--data_dir",    default=DEFAULT_TEST_DIR)
    p.add_argument("--task",        default="all", choices=ALL_TASKS + ["all"])
    p.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--k",           type=int, default=DEFAULT_K,
                   help="Number of Generate branches (default: 3)")
    p.add_argument("--dry_run",     action="store_true")
    return p.parse_args()


def main():
    args  = parse_args()
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    k     = args.k

    lm = LlamaLocal(config_path=args.config or "")
    if not args.config:
        lm.model_path     = DEFAULT_MODEL_PATH
        lm.backend        = "vllm"
        lm.max_new_tokens = 512
        lm.temperature    = 0.1
        lm.top_p          = 1.0

    prompter = GoTPrompter()

    logger.info("Mode      : %s", "DRY-RUN (no GPU)" if args.dry_run else "LIVE")
    logger.info("Model     : %s", lm.model_path)
    logger.info("Data      : %s", args.data_dir)
    logger.info("Tasks     : %s", tasks)
    logger.info("GoT config: k=%d  (calls/sample ≈ %d)", k, k + 3)
    logger.info("Limit     : %s samples/task",
                args.max_samples if args.max_samples else "all")

    all_summaries = []
    task_pbar = tqdm(tasks, desc="Overall", unit="task",
                     position=0, leave=True, dynamic_ncols=True)

    for task in task_pbar:
        task_pbar.set_description(f"Overall [{task}]")
        samples = load_task_data(args.data_dir, task, args.max_samples)
        if not samples:
            logger.warning("No samples for %s — skipping.", task)
            continue

        lm.reset_token_counters()
        results = run_got_task(task, samples, lm, prompter, k, args.dry_run)
        summary = save_results(results, task, args.output_dir,
                               lm.token_usage(), k)
        all_summaries.append(summary)

    task_pbar.close()

    if all_summaries:
        print("\n" + "=" * 80)
        print(f"{'Task':<18} {'Acc':>7} {'N':>6} {'Correct':>8} "
              f"{'AvgCall':>8} {'Tok-In':>10} {'Tok-Out':>10}")
        print("-" * 80)
        for s in all_summaries:
            print(
                f"{s['task']:<18} {s['accuracy']:>7.4f} {s['n_samples']:>6} "
                f"{s['n_correct']:>8} {s['avg_lm_calls']:>8.1f} "
                f"{s['token_usage'].get('prompt_tokens',0):>10} "
                f"{s['token_usage'].get('completion_tokens',0):>10}"
            )
        print("=" * 80)
        avg_acc = sum(s["accuracy"] for s in all_summaries) / len(all_summaries)
        print(f"Overall average accuracy: {avg_acc:.4f}  ({len(all_summaries)} tasks)\n")

        master_path = os.path.join(
            args.output_dir,
            f"got_all_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(master_path, "w") as f:
            json.dump({
                "method":      "got",
                "got_k":       k,
                "tasks":       all_summaries,
                "average_acc": round(avg_acc, 4),
            }, f, indent=2)
        logger.info("Master summary → %s", master_path)


if __name__ == "__main__":
    main()