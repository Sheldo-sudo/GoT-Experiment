from typing import Any, Callable, Dict, List

from graph_of_thoughts import operations, parser, prompter

try:
    from . import utils
except ImportError:
    import utils


class BaseTaskPrompter(prompter.Prompter):
    def __init__(
        self,
        task_name: str,
        phase0_prompt_fn: Callable[[str], str],
        branch_prompt_fn: Callable[[str, str, str, str], str],
        aggregate_prompt_fn: Callable[[str, str], str],
        improve_prompt_fn: Callable[[str, str], str],
    ) -> None:
        self.task_name = task_name
        self.phase0_prompt_fn = phase0_prompt_fn
        self.branch_prompt_fn = branch_prompt_fn
        self.aggregate_prompt_fn = aggregate_prompt_fn
        self.improve_prompt_fn = improve_prompt_fn

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        assert len(state_dicts) >= 1
        original_query = state_dicts[0]["original"]

        def sort_key(s: Dict[str, Any]) -> str:
            return str(s.get("part", ""))

        chunks: List[str] = []
        for st in sorted(state_dicts, key=sort_key):
            chunks.append(
                f"{st.get('part', 'Unknown')}:\n"
                f"Answer:\n{st.get('current', '')}\n"
            )

        branch_bundle = "\n".join(chunks)
        return self.aggregate_prompt_fn(original_query, branch_bundle)

    def generate_prompt(
        self,
        num_branches: int,
        original: str,
        current: str,
        task: str,
        method: str,
        phase: int,
        part: str = "",
        branch_goal: str = "",
        **kwargs,
    ) -> str:
        if phase == 0:
            return self.phase0_prompt_fn(task)
        return self.branch_prompt_fn(task, part, branch_goal, original)

    def improve_prompt(self, **kwargs) -> str:
        original = kwargs.get("original", "")
        current = kwargs.get("current", "")
        return self.improve_prompt_fn(original, current)

    def validation_prompt(self, **kwargs) -> str:
        return ""

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        return ""


class BaseTaskParser(parser.Parser):
    def __init__(self, branches: List[Dict[str, Any]]) -> None:
        self.branches = branches

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> List[Dict]:
        new_states = []
        for text in texts:
            cleaned = utils.clean_response(text)
            if cleaned:
                new_states.append(
                    {
                        "current": cleaned,
                        "phase": 3,
                        "part": "final",
                        "branch_goal": "",
                    }
                )
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        phase = state.get("phase", 0)

        if phase == 0:
            return [
                {
                    "current": "",
                    "phase": 1,
                    "part": branch["part"],
                    "branch_goal": branch["goal"],
                }
                for branch in self.branches
            ]

        new_states: List[Dict] = []
        for text in texts:
            cleaned = utils.clean_response(text)
            if cleaned:
                new_states.append(
                    {
                        "current": cleaned,
                        "phase": phase + 1,
                    }
                )
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        if not texts:
            return {}
        return {
            "current": utils.clean_response(texts[0]),
            "phase": state.get("phase", 3),
            "part": state.get("part", "final"),
        }

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        return True

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        return [0.0 for _ in states]


def add_branch(
    graph: operations.GraphOfOperations,
    plans: operations.Operation,
    part_name: str,
    num_generate: int,
    keep_n: int,
    score_fn,
) -> operations.Operation:
    selector = operations.Selector(
        lambda thoughts, part_name=part_name: [
            t for t in thoughts if t.state.get("part") == part_name
        ]
    )
    selector.add_predecessor(plans)
    graph.add_operation(selector)

    solve = operations.Generate(1, num_generate)
    solve.add_predecessor(selector)
    graph.add_operation(solve)

    score = operations.Score(1, False, score_fn)
    score.add_predecessor(solve)
    graph.add_operation(score)

    keep = operations.KeepBestN(keep_n, False)
    keep.add_predecessor(score)
    graph.add_operation(keep)

    return keep


def build_task_graph(
    branches: List[Dict[str, Any]],
    search_score_fn,
    final_validator,
    ground_truth_fn,
    aggregate_responses: int = 3,
) -> operations.GraphOfOperations:
    graph = operations.GraphOfOperations()

    plans = operations.Generate(1, 1)
    graph.append_operation(plans)

    branch_nodes = []
    for branch in branches:
        node = add_branch(
            graph=graph,
            plans=plans,
            part_name=branch["part"],
            num_generate=branch["num_generate"],
            keep_n=branch["keep_n"],
            score_fn=search_score_fn,
        )
        branch_nodes.append(node)

    aggregate = operations.Aggregate(aggregate_responses)
    for node in branch_nodes:
        aggregate.add_predecessor(node)
    graph.add_operation(aggregate)

    validate = operations.ValidateAndImprove(
        num_samples=1,
        improve=True,
        num_tries=1,
        validate_function=final_validator,
    )
    validate.add_predecessor(aggregate)
    graph.add_operation(validate)

    keep_valid = operations.KeepValid()
    keep_valid.add_predecessor(validate)
    graph.add_operation(keep_valid)

    score_final = operations.Score(1, False, search_score_fn)
    score_final.add_predecessor(keep_valid)
    graph.add_operation(score_final)

    keep_best = operations.KeepBestN(1, False)
    keep_best.add_predecessor(score_final)
    graph.add_operation(keep_best)

    gt = operations.GroundTruth(ground_truth_fn)
    gt.add_predecessor(keep_best)
    graph.add_operation(gt)

    return graph


def default_yesno_validator(state: Dict[str, Any]) -> bool:
    return utils.validate_yesno_response(state)


def default_numeric_validator(state: Dict[str, Any]) -> bool:
    return utils.extract_last_number(state.get("current", "")) is not None


def fallback_ground_truth(state: Dict[str, Any]) -> bool:
    return utils.graphwiz_ground_truth(state)