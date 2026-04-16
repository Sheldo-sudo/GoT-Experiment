from typing import Any, Dict, List


def _quality_from_search_score(score: float) -> float:
    # Lower search_score is better. Convert to [0, 1] quality.
    if score is None:
        return 0.0
    try:
        s = float(score)
    except Exception:
        return 0.0
    return 1.0 / (1.0 + max(s, 0.0))


def _extract_branch_steps(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = trajectory.get("steps", []) or []
    return [
        s
        for s in steps
        if s.get("part") not in {"final", "root", None, ""}
    ]


def _extract_final_step(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    steps = trajectory.get("steps", []) or []
    for step in reversed(steps):
        if step.get("part") == "final":
            return step
    return steps[-1] if steps else {}


def _branch_agreement(trajectory: Dict[str, Any]) -> float:
    """
    Generic branch agreement approximation:
    - yes/no tasks: consistency ratio
    - numeric tasks: inverse normalized std
    - others: exact normalized final token match ratio
    """
    branch_steps = _extract_branch_steps(trajectory)
    if len(branch_steps) <= 1:
        return 1.0

    routed_task = str(trajectory.get("routed_task", "")).lower()
    yesno_tasks = {"connectivity", "bipartite", "cycle", "hamilton", "substructure"}
    numeric_tasks = {"shortest_path", "flow", "triangle"}

    if routed_task in yesno_tasks:
        labels = []
        for s in branch_steps:
            text = str((s.get("state") or {}).get("current", "")).lower()
            if "yes" in text:
                labels.append("yes")
            elif "no" in text:
                labels.append("no")
        if len(labels) <= 1:
            return 0.5
        majority = max(labels.count("yes"), labels.count("no"))
        return float(majority / len(labels))

    if routed_task in numeric_tasks:
        values = []
        for s in branch_steps:
            text = str((s.get("state") or {}).get("current", ""))
            nums = []
            cur = ""
            for ch in text:
                if ch.isdigit() or ch in ".-":
                    cur += ch
                else:
                    if cur:
                        nums.append(cur)
                        cur = ""
            if cur:
                nums.append(cur)
            if nums:
                try:
                    values.append(float(nums[-1]))
                except Exception:
                    pass
        if len(values) <= 1:
            return 0.5
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        denom = abs(mean_val) + 1.0
        return max(0.0, 1.0 - (std / denom))

    compact = []
    for s in branch_steps:
        text = str((s.get("state") or {}).get("current", "")).strip().lower()
        compact.append(" ".join(text.split()))
    if len(compact) <= 1:
        return 0.5
    first = compact[0]
    same = sum(1 for x in compact if x == first)
    return float(same / len(compact))


def compute_trajectory_reward(
    trajectory: Dict[str, Any],
    alpha_tokens: float = 1e-6,
) -> Dict[str, Any]:
    """
    R = 1.0*Correct(final)
      + 0.2*ValidFinal(final)
      + 0.1*MeanBranchQuality
      + 0.1*BranchAgreement
      - 0.1*RepairUsed
      - alpha*TotalTokens
    """
    final_step = _extract_final_step(trajectory)
    branch_steps = _extract_branch_steps(trajectory)

    correct = 1.0 if trajectory.get("is_correct", False) else 0.0
    valid_final = 1.0 if final_step.get("final_validator") else 0.0
    branch_qualities = [
        _quality_from_search_score(step.get("search_score"))
        for step in branch_steps
    ]
    mean_branch_quality = (
        sum(branch_qualities) / len(branch_qualities) if branch_qualities else 0.0
    )
    branch_agreement = _branch_agreement(trajectory)
    repair_used = 1.0 if any(
        str(step.get("operation", "")).lower() == "validate_and_improve"
        for step in trajectory.get("steps", [])
    ) else 0.0
    total_tokens = float(trajectory.get("token_usage", {}).get("total_tokens", 0.0))

    reward = (
        1.0 * correct
        + 0.2 * valid_final
        + 0.1 * mean_branch_quality
        + 0.1 * branch_agreement
        - 0.1 * repair_used
        - float(alpha_tokens) * total_tokens
    )

    return {
        "reward": float(reward),
        "components": {
            "correct_final": correct,
            "valid_final": valid_final,
            "mean_branch_quality": float(mean_branch_quality),
            "branch_agreement": float(branch_agreement),
            "repair_used": repair_used,
            "total_tokens": total_tokens,
            "alpha_tokens": float(alpha_tokens),
        },
    }


def compute_step_level_shaping(
    trajectory: Dict[str, Any],
    gamma: float = 0.98,
) -> List[Dict[str, Any]]:
    """
    Minimal step-level shaping:
    step_reward = quality(step) + discounted_final_bonus
    """
    steps = trajectory.get("steps", []) or []
    final_bonus = 1.0 if trajectory.get("is_correct", False) else 0.0
    n = len(steps)
    shaped = []
    for i, step in enumerate(steps):
        q = _quality_from_search_score(step.get("search_score"))
        discounted = (gamma ** max(0, n - i - 1)) * final_bonus
        shaped.append(
            {
                "step_index": i,
                "operation": step.get("operation"),
                "part": step.get("part"),
                "phase": step.get("phase"),
                "step_reward": float(q + discounted),
                "quality_term": float(q),
                "discounted_final_bonus": float(discounted),
            }
        )
    return shaped
