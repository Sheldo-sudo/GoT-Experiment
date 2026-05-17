"""
run_tot.py
=========
Tree-of-Thoughts baseline for GraphInstruct tasks using LLaMA-2-7B-Chat.

ToT Search Strategy: BFS (Breadth-First Search)
------------------------------------------------
  Each sample:
    1. Generate `tot_breadth` candidate thought steps
    2. Evaluate each with sure/maybe/impossible scoring
    3. Keep top `tot_breadth` candidates by score
    4. Repeat for `tot_depth` layers
    5. Use best surviving thought to generate final answer

Usage
-----
  python run_tot.py --dry_run --max_samples 3
  python run_tot.py --config config.json --task connectivity --max_samples 10
  python run_tot.py --config config.json
  python run_tot.py --config config.json --breadth 2 --depth 2
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

# ── Package root ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from language_models.llama_local import LlamaLocal
from prompters.prompter import ToTPrompter
from parsers.parser import extract_answer, score_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_tot")

# ── Server paths ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "/home/wangzq28/GoT-Experiment/models/LLaMA2-7B-chat"
DEFAULT_TEST_DIR   = "/home/wangzq28/GoT-Experiment/data/GraphInstruct-Test"
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results", "tot")

# ── ToT hyperparameters ───────────────────────────────────────────────────
DEFAULT_BREADTH = 3
DEFAULT_DEPTH   = 3

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
    stem     = TASK_FILE_STEMS[task]
    filename = f"{stem}_test.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        logger.warning("File not found: %s — skipping task %s", filepath, task)
        return []

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

    samples = []
    for item in items:
        samples.append({
            "task":         task,
            "raw_prompt":   item.get("input_prompt", ""),
            "ground_truth": str(item.get("answer", "")).strip(),
            "index":        item.get("index", len(samples)),
        })

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


# ── ToT core BFS search ───────────────────────────────────────────────────

def _eval_score(eval_output: str) -> float:
    text = eval_output.lower().strip()
    if "sure" in text:
        return 1.0
    elif "impossible" in text:
        return 0.0
    else:
        return 0.5


def tot_solve(
    task:         str,
    raw_question: str,
    lm:           LlamaLocal,
    prompter:     ToTPrompter,
    breadth:      int  = DEFAULT_BREADTH,
    depth:        int  = DEFAULT_DEPTH,
    dry_run:      bool = False,
) -> Tuple[str, int]:
    """
    BFS Tree-of-Thoughts search.
    Returns: (raw_final_output, total_lm_calls)
    """
    total_calls = 0
    candidates: List[Tuple[str, float]] = [("", 1.0)]

    for d in range(depth):
        next_candidates: List[Tuple[str, float]] = []

        for thought, _ in candidates:
            # Step 1: generate breadth new thought steps
            thought_prompt = prompter.build_thought_prompt(
                task, raw_question, thought
            )
            if dry_run:
                new_steps = [f"Dry-run step {d+1} branch {b}" for b in range(breadth)]
            else:
                new_steps    = lm.query(thought_prompt, num_responses=breadth)
                total_calls += 1

            for new_step in new_steps:
                new_thought = (thought + "\n" + new_step).strip()

                # Step 2: evaluate this thought path
                eval_prompt = prompter.build_eval_prompt(
                    task, raw_question, new_thought
                )
                if dry_run:
                    eval_out = "maybe"
                else:
                    eval_results = lm.query(eval_prompt, num_responses=1)
                    eval_out     = eval_results[0] if eval_results else "maybe"
                    total_calls += 1

                sc = _eval_score(eval_out)
                if sc > 0.0:
                    next_candidates.append((new_thought, sc))

        if not next_candidates:
            break

        # Step 3: keep top-breadth candidates
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = next_candidates[:breadth]

    # Step 4: generate final answer from best thought
    best_thought = candidates[0][0] if candidates else ""
    if best_thought:
        final_prompt = prompter.build_final_prompt(task, raw_question, best_thought)
    else:
        final_prompt = prompter.build_prompt_from_raw(task, raw_question)

    if dry_run:
        raw_output = "Yes"
    else:
        outputs    = lm.query(final_prompt, num_responses=1)
        raw_output = outputs[0] if outputs else ""
        total_calls += 1

    return raw_output, total_calls


# ── Inference loop ────────────────────────────────────────────────────────

def run_tot_task(
    task:     str,
    samples:  List[Dict],
    lm:       LlamaLocal,
    prompter: ToTPrompter,
    breadth:  int  = DEFAULT_BREADTH,
    depth:    int  = DEFAULT_DEPTH,
    dry_run:  bool = False,
) -> List[Dict]:
    results        = []
    total_lm_calls = 0
    correct_so_far = 0

    # tqdm progress bar — shows per-sample stats in postfix
    pbar = tqdm(
        samples,
        desc=f"{task:<16}",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
    )

    for sample in pbar:
        t0 = time.time()

        raw_output, calls = tot_solve(
            task         = task,
            raw_question = sample["raw_prompt"],
            lm           = lm,
            prompter     = prompter,
            breadth      = breadth,
            depth        = depth,
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
            "tot_breadth":   breadth,
            "tot_depth":     depth,
        })

        # Update tqdm postfix with live stats
        running_acc = correct_so_far / len(results)
        pbar.set_postfix({
            "acc":   f"{running_acc:.3f}",
            "lat":   f"{latency:.1f}s",
            "calls": calls,
        })

    pbar.close()

    # Per-task summary log after bar completes
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
    results:     List[Dict],
    task:        str,
    output_dir:  str,
    token_usage: Dict,
    breadth:     int,
    depth:       int,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = os.path.join(output_dir, f"tot_{task}_{ts}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n         = len(results)
    correct   = sum(r["score"] for r in results)
    accuracy  = correct / n if n else 0.0
    avg_lat   = sum(r["latency_s"] for r in results) / n if n else 0.0
    avg_calls = sum(r["lm_calls"] for r in results) / n if n else 0.0

    summary = {
        "method":        "tot",
        "task":          task,
        "tot_breadth":   breadth,
        "tot_depth":     depth,
        "n_samples":     n,
        "n_correct":     int(correct),
        "accuracy":      round(accuracy, 4),
        "avg_latency_s": round(avg_lat, 4),
        "avg_lm_calls":  round(avg_calls, 2),
        "token_usage":   token_usage,
        "timestamp":     ts,
        "result_file":   out_path,
    }
    summary_path = os.path.join(output_dir, f"tot_{task}_{ts}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tree-of-Thoughts baseline for GraphInstruct (9 tasks)"
    )
    p.add_argument("--config",      default=None)
    p.add_argument("--data_dir",    default=DEFAULT_TEST_DIR)
    p.add_argument("--task",        default="all", choices=ALL_TASKS + ["all"])
    p.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--breadth",     type=int, default=DEFAULT_BREADTH,
                   help="ToT breadth: candidates per layer (default: 3)")
    p.add_argument("--depth",       type=int, default=DEFAULT_DEPTH,
                   help="ToT depth: reasoning layers (default: 3)")
    p.add_argument("--dry_run",     action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    tasks   = ALL_TASKS if args.task == "all" else [args.task]
    breadth = args.breadth
    depth   = args.depth

    # ── Init LM ──────────────────────────────────────────────────────────
    lm = LlamaLocal(config_path=args.config or "")
    if not args.config:
        lm.model_path     = DEFAULT_MODEL_PATH
        lm.backend        = "vllm"
        lm.max_new_tokens = 256
        lm.temperature    = 0.7
        lm.top_p          = 0.95

    prompter = ToTPrompter()

    logger.info("Mode      : %s", "DRY-RUN (no GPU)" if args.dry_run else "LIVE")
    logger.info("Model     : %s", lm.model_path)
    logger.info("Data      : %s", args.data_dir)
    logger.info("Tasks     : %s", tasks)
    logger.info("ToT config: breadth=%d  depth=%d", breadth, depth)
    logger.info("Limit     : %s samples/task",
                args.max_samples if args.max_samples else "all")

    # Outer tqdm: across all tasks
    all_summaries: List[Dict] = []
    task_pbar = tqdm(tasks, desc="Overall", unit="task",
                     position=0, leave=True, dynamic_ncols=True)

    for task in task_pbar:
        task_pbar.set_description(f"Overall [{task}]")

        samples = load_task_data(args.data_dir, task, args.max_samples)
        if not samples:
            logger.warning("No samples for %s — skipping.", task)
            continue

        lm.reset_token_counters()
        results = run_tot_task(
            task, samples, lm, prompter, breadth, depth, args.dry_run
        )
        summary = save_results(
            results, task, args.output_dir, lm.token_usage(), breadth, depth
        )
        all_summaries.append(summary)

    task_pbar.close()

    # ── Final cross-task summary table ────────────────────────────────────
    if all_summaries:
        print("\n" + "=" * 80)
        print(f"{'Task':<18} {'Acc':>7} {'N':>6} {'Correct':>8} "
              f"{'AvgCall':>8} {'Tok-In':>10} {'Tok-Out':>10}")
        print("-" * 80)
        for s in all_summaries:
            print(
                f"{s['task']:<18} {s['accuracy']:>7.4f} {s['n_samples']:>6} "
                f"{s['n_correct']:>8} {s['avg_lm_calls']:>8.1f} "
                f"{s['token_usage'].get('prompt_tokens', 0):>10} "
                f"{s['token_usage'].get('completion_tokens', 0):>10}"
            )
        print("=" * 80)
        avg_acc = sum(s["accuracy"] for s in all_summaries) / len(all_summaries)
        print(f"Overall average accuracy: {avg_acc:.4f}  ({len(all_summaries)} tasks)\n")

        master_path = os.path.join(
            args.output_dir,
            f"tot_all_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(master_path, "w") as f:
            json.dump({
                "method":      "tot",
                "tot_breadth": breadth,
                "tot_depth":   depth,
                "tasks":       all_summaries,
                "average_acc": round(avg_acc, 4),
            }, f, indent=2)
        logger.info("Master summary → %s", master_path)


if __name__ == "__main__":
    main()