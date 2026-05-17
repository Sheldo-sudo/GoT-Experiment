"""
demos/__init__.py
Maps task name → demo filename (matches actual filenames on server).
"""
import os

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# Filename mapping: task_name -> actual .txt filename (no extension)
TASK_DEMO_FILES = {
    "connectivity":  os.path.join(DEMO_DIR, "connectivity.txt"),
    "cycle":         os.path.join(DEMO_DIR, "cycle.txt"),
    "bipartite":     os.path.join(DEMO_DIR, "bipartite.txt"),
    "shortest_path": os.path.join(DEMO_DIR, "shortest.txt"),   # file: shortest.txt
    "triangle":      os.path.join(DEMO_DIR, "triplet.txt"),    # file: triplet.txt
    "subgraph":      os.path.join(DEMO_DIR, "substructure.txt"),# file: substructure.txt
    "hamilton":      os.path.join(DEMO_DIR, "hamilton.txt"),
    "maximum_flow":  os.path.join(DEMO_DIR, "flow.txt"),       # file: flow.txt
    "topology":      os.path.join(DEMO_DIR, "topology.txt"),
}


def load_demo(task: str) -> str:
    """Return the raw few-shot demo string for a given task."""
    path = TASK_DEMO_FILES.get(task)
    if path is None:
        raise ValueError(
            f"Unknown task: {task!r}. Valid tasks: {sorted(TASK_DEMO_FILES)}"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Demo file not found: {path}\n"
            f"Please ensure demos/ contains: "
            + ", ".join(os.path.basename(p) for p in TASK_DEMO_FILES.values())
        )
    with open(path, "r") as f:
        return f.read().strip()