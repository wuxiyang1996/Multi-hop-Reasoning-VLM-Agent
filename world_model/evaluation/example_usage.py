"""
Example usage of the Synthetic Experience Evaluation module.

Demonstrates the full pipeline with three synthetic trajectories of
varying quality (high / medium / low) and a mock LLM judge.
Run standalone:  python -m world_model.evaluation.example_usage
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from world_model.evaluation.config import ExperienceEvaluationConfig, LLMJudgeConfig
from world_model.evaluation.schemas import (
    ExperienceRecord,
    ExperienceStep,
)
from world_model.evaluation.run_evaluation import evaluate_experiences


# ─── Mock LLM that returns dimension-aware scores ────────────────────

# Pre-programmed responses keyed by (record_id_pattern, dimension_keyword)
_MOCK_SCORES = {
    ("high", "fidelity"):              9,
    ("high", "consistency"):           9,
    ("high", "instruction"):           8,
    ("high", "diversity"):             7,
    ("high", "informativeness"):       8,
    ("high", "holistic"):              9,

    ("medium", "fidelity"):            6,
    ("medium", "consistency"):         7,
    ("medium", "instruction"):         5,
    ("medium", "diversity"):           6,
    ("medium", "informativeness"):     5,
    ("medium", "holistic"):            6,

    ("low", "fidelity"):               2,
    ("low", "consistency"):            3,
    ("low", "instruction"):            2,
    ("low", "diversity"):              4,
    ("low", "informativeness"):        2,
    ("low", "holistic"):               2,
}


def _mock_ask_model(prompt: str, **kwargs) -> str:
    """Deterministic mock LLM that inspects the prompt to pick scores."""
    prompt_lower = prompt.lower()

    # Detect record quality tier
    tier = "medium"
    if "high_quality" in prompt_lower:
        tier = "high"
    elif "low_quality" in prompt_lower:
        tier = "low"

    # Detect dimension
    dim = "holistic"
    for kw in ["fidelity", "consistency", "instruction", "diversity", "informativeness"]:
        if kw in prompt_lower:
            dim = kw
            break

    score = _MOCK_SCORES.get((tier, dim), 5)

    verdict = "ACCEPT" if score >= 7 else ("REFINE" if score >= 4 else "DISCARD")

    return json.dumps({
        "score": score,
        "evidence": [
            f"Mock evidence for {dim} (tier={tier})",
            f"Score determined by pre-programmed lookup",
        ],
        "issues": [] if score >= 7 else [f"Mock issue: {dim} score is {score}/10"],
        "verdict": verdict,
        "suggestions": [] if score >= 7 else [f"Consider improving {dim}"],
        "recommendation": verdict,
        "reasoning": f"Mock holistic reasoning for {tier} quality trajectory",
    })


# ─── Build toy trajectories ─────────────────────────────────────────


def _make_high_quality_record() -> ExperienceRecord:
    """Coherent Overcooked trajectory: pick onion -> bring to pot -> serve."""
    return ExperienceRecord(
        record_id="high_quality_0",
        steps=[
            ExperienceStep(
                state="Agent at (0,1). Counter has 3 onions. Pot is empty.",
                next_state="Agent at (1,1). Agent holds 1 onion. Counter has 2 onions.",
                instruction="Pick up an onion from the counter.",
                action="pick_onion",
                reward=0.0,
            ),
            ExperienceStep(
                state="Agent at (1,1). Agent holds 1 onion. Counter has 2 onions.",
                next_state="Agent at (2,1). Agent holds 1 onion. Counter has 2 onions.",
                instruction="Move toward the pot.",
                action="move_right",
                reward=0.0,
            ),
            ExperienceStep(
                state="Agent at (2,1). Agent holds 1 onion. Counter has 2 onions.",
                next_state="Agent at (2,1). Agent holds nothing. Pot has 1 onion.",
                instruction="Place the onion into the pot.",
                action="place_onion",
                reward=0.5,
            ),
            ExperienceStep(
                state="Agent at (2,1). Pot has 1 onion. Cooking timer: 0/20.",
                next_state="Agent at (2,1). Pot has 1 onion cooking. Timer: 20/20. Soup ready.",
                instruction="Wait for the soup to finish cooking.",
                action="wait",
                reward=0.0,
            ),
            ExperienceStep(
                state="Agent at (2,1). Soup ready in pot.",
                next_state="Agent at (3,1). Agent holds soup plate. Served to window.",
                instruction="Serve the soup to the serving window.",
                action="serve_soup",
                reward=5.0,
            ),
        ],
        synthesis_instructions="Pick onion, cook soup, serve it.",
        plan_steps=[
            "Pick up an onion from the counter.",
            "Move toward the pot.",
            "Place the onion into the pot.",
            "Wait for the soup to finish cooking.",
            "Serve the soup to the serving window.",
        ],
        env_hint="overcooked",
        skill_id="cook_and_serve",
    )


def _make_medium_quality_record() -> ExperienceRecord:
    """Partially consistent: one step skips a transition."""
    return ExperienceRecord(
        record_id="medium_quality_1",
        steps=[
            ExperienceStep(
                state="Agent at start. 3 plates on counter.",
                next_state="Agent holds plate. 2 plates on counter.",
                instruction="Pick up a plate.",
                action="pick_plate",
                reward=0.0,
            ),
            ExperienceStep(
                state="Agent holds plate. 2 plates on counter.",
                next_state="Agent at pot. Agent holds plate with soup.",
                instruction="Fill the plate with soup from the pot.",
                action="fill_plate",
                reward=0.5,
            ),
            ExperienceStep(
                state="Agent at pot. Agent holds plate with soup.",
                next_state="Agent at window. Soup delivered. Score +5.",
                instruction="Deliver the soup plate to the window.",
                action="deliver",
                reward=5.0,
            ),
        ],
        synthesis_instructions="Pick plate, get soup, deliver.",
        plan_steps=[
            "Pick up a plate.",
            "Fill the plate with soup from the pot.",
            "Deliver the soup plate to the window.",
        ],
        env_hint="overcooked",
    )


def _make_low_quality_record() -> ExperienceRecord:
    """Contradictory and implausible trajectory."""
    return ExperienceRecord(
        record_id="low_quality_2",
        steps=[
            ExperienceStep(
                state="Agent at (0,0). Empty kitchen.",
                next_state="Agent at (5,5). Agent holds cooked steak.",
                instruction="Pick up an onion.",
                action="teleport_with_steak",
                reward=0.0,
            ),
            ExperienceStep(
                state="Agent at (5,5). Agent holds cooked steak.",
                next_state="Agent at (0,0). Kitchen is now full of plates.",
                instruction="Place the onion in the pot.",
                action="reset_world",
                reward=-1.0,
            ),
            ExperienceStep(
                state="Agent at (0,0). Kitchen is now full of plates.",
                next_state="Agent at (0,0). Kitchen is now full of plates.",
                instruction="Serve the soup.",
                action="noop",
                reward=0.0,
            ),
        ],
        synthesis_instructions="Cook an onion soup.",
        plan_steps=["Pick up an onion.", "Place the onion in the pot.", "Serve the soup."],
        env_hint="overcooked",
    )


# ─── Main ────────────────────────────────────────────────────────────


def main():
    records = [
        _make_high_quality_record(),
        _make_medium_quality_record(),
        _make_low_quality_record(),
    ]

    config = ExperienceEvaluationConfig(
        llm=LLMJudgeConfig(
            model="mock",
            ask_model_fn=_mock_ask_model,
        ),
        run_holistic_pass=True,
    )

    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    summary = evaluate_experiences(records, config=config)

    print("\n" + "=" * 60)
    print(summary.format_for_llm())
    print("=" * 60)

    for rid, report in summary.reports.items():
        print(f"\n--- {rid} ---")
        print(report.format_for_llm())

    print(f"\nFull JSON report:\n{json.dumps(summary.to_dict(), indent=2)}")


if __name__ == "__main__":
    main()
