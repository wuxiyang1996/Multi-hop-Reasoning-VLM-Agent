#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from motif_transfer.multihorizon_replay import (
    analyze_multihorizon_rows,
    file_hash,
    stable_hash,
)
from motif_transfer.multihorizon_runner import (
    ForkState,
    ForkStep,
    PolicyDecision,
    PolicyHistoryStep,
    run_matched_multihorizon_snapshot,
)
from motif_transfer.source_multihorizon import (
    render_continuation_prompt,
    validate_plan,
)


_ACTION_RE = re.compile(r"ACTION\s*:\s*(\d+)(?:\s|$)", re.IGNORECASE)


def strict_action_number(reply: str, native_actions: Sequence[str]) -> str | None:
    matches = [int(value) for value in _ACTION_RE.findall(reply)]
    if not matches or len(set(matches)) != 1:
        return None
    index = matches[0] - 1
    if not 0 <= index < len(native_actions):
        return None
    return str(native_actions[index])


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        return None


def extract_official_value(info: Mapping[str, Any]) -> tuple[float | None, str | None]:
    direct_paths = (
        ("score_value",),
        ("perf_score",),
        ("eval_score",),
        ("score",),
        ("structured_state", "simulation", "episode_reward"),
        ("structured_state", "ram_watch", "score"),
    )
    for path in direct_paths:
        value: Any = info
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        number = _numeric(value)
        if number is not None:
            return number, ".".join(path)
    return None, None


def extract_official_success(info: Mapping[str, Any]) -> bool | None:
    for key in ("won", "success", "task_success", "official_success"):
        if key in info and info[key] is not None:
            return bool(info[key])
    return None


class SourceForkEnvironment:
    def __init__(self, *, game: str, maximum_steps: int) -> None:
        from env_wrappers.subprocess_env import SubprocessEnv
        from trainer.coevolution._state_to_markup import state_to_markup

        self._game = game
        self._maximum_steps = maximum_steps
        self._state_to_markup = state_to_markup
        self._env = SubprocessEnv(
            game=game,
            max_steps=maximum_steps,
            env_kind="gymv" if game.startswith("gymv_") else "orak",
        )
        self._step = 0

    def _state(
        self,
        observation: str,
        info: Mapping[str, Any],
        *,
        terminal: bool,
        truncated: bool,
    ) -> ForkState:
        actions = tuple(str(item) for item in info.get("action_names", ()))
        markup = self._state_to_markup(
            obs_nl=str(observation),
            info=dict(info),
            game=self._game,
            step=self._step,
        )
        official_value, _ = extract_official_value(info)
        return ForkState(
            state={
                "state_markup": markup,
                "structured_state_sha256": stable_hash(info.get("structured_state")),
            },
            admissible_actions=actions,
            terminal=terminal,
            truncated=truncated,
            observable=str(observation),
            official_value=official_value,
        )

    def reset(self, *, seed: int) -> ForkState:
        self._step = 0
        observation, info = self._env.reset(seed=seed)
        return self._state(
            str(observation), info, terminal=False, truncated=False,
        )

    def step(self, action: str) -> ForkStep:
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1
        state = self._state(
            str(observation), info,
            terminal=bool(terminated), truncated=bool(truncated),
        )
        official_value, value_source = extract_official_value(info)
        return ForkStep(
            state=state,
            reward=float(reward),
            official_success=extract_official_success(info),
            official_value=official_value,
            metadata={
                "official_value_source": value_source,
                "structured_state_sha256": stable_hash(info.get("structured_state")),
            },
        )

    def close(self) -> None:
        self._env.close()


class FrozenTreatmentPolicy:
    def __init__(self, client, *, model: str, snapshot: Mapping[str, Any]) -> None:
        self._client = client
        self._model = model
        self._snapshot = snapshot

    def choose_action(
        self,
        state: ForkState,
        *,
        treatment: str,
        decision_index: int,
        history: Sequence[PolicyHistoryStep],
    ) -> PolicyDecision:
        treatment_row = self._snapshot["treatments"][treatment]
        if decision_index == 0:
            return PolicyDecision(
                action=str(treatment_row["parsed_action"]),
                prompt_sha256=str(treatment_row["prompt_sha256"]),
                raw_response_sha256=str(treatment_row["raw_response_sha256"]),
                requested_adapter=treatment_row.get("requested_adapter"),
                used_adapter=treatment_row.get("used_adapter"),
                source="FROZEN_MATCHED_FIRST_ACTION",
                metadata={
                    "context_skill_id": treatment_row.get("context_skill_id"),
                },
            )
        branch_history = [(row.action, row.reward) for row in history]
        prompt = render_continuation_prompt(
            str(treatment_row["prompt"]),
            state_markup=str(state.state["state_markup"]),
            native_actions=state.admissible_actions,
            branch_history=branch_history,
        )
        requested_adapter = None if treatment == "B" else "action_taking"
        model_id = self._model if requested_adapter is None else requested_adapter
        request_seed = int(stable_hash({
            "model": model_id,
            "prompt": prompt,
        })[:8], 16)
        response = self._client.completions.create(
            model=model_id,
            prompt=prompt,
            temperature=0.0,
            max_tokens=128,
            stop=["\n\nGame state:", "\n\nAvailable actions", "<think", "<thinking"],
            extra_body={"seed": request_seed},
        )
        raw_response = response.choices[0].text if response.choices else ""
        action = strict_action_number(raw_response, state.admissible_actions)
        usage = response.usage
        return PolicyDecision(
            action=action or "__INVALID_POLICY_OUTPUT__",
            prompt_sha256=stable_hash(prompt),
            raw_response_sha256=stable_hash(raw_response),
            requested_adapter=requested_adapter,
            used_adapter=requested_adapter,
            request_seed=request_seed,
            source="LIVE_CLOSED_LOOP_CONTINUATION",
            metadata={
                "strict_parse_valid": action is not None,
                "prompt_tokens": int(usage.prompt_tokens if usage else 0),
                "completion_tokens": int(usage.completion_tokens if usage else 0),
                "response_model": str(response.model),
            },
        )


def _runtime_receipts(source_runtime: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "multihorizon_replay": Path(sys.modules[
            "motif_transfer.multihorizon_replay"
        ].__file__).resolve(),
        "multihorizon_runner": Path(sys.modules[
            "motif_transfer.multihorizon_runner"
        ].__file__).resolve(),
        "source_multihorizon": Path(sys.modules[
            "motif_transfer.source_multihorizon"
        ].__file__).resolve(),
        "source_subprocess_env": source_runtime / "env_wrappers/subprocess_env.py",
        "source_state_to_markup": (
            source_runtime / "trainer/coevolution/_state_to_markup.py"
        ),
    }
    return {name: file_hash(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute frozen h1/h2/h4/h8 source intervention forks."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-runtime", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    source_runtime = args.source_runtime.resolve()
    if not (source_runtime / "env_wrappers/subprocess_env.py").is_file():
        raise SystemExit(f"invalid source runtime: {source_runtime}")
    sys.path.insert(0, str(source_runtime))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    rows_path = args.output / "multihorizon_rows.jsonl"
    manifest_path = args.output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "claim_boundary": plan["claim_boundary"],
        "plan": str(args.plan.resolve()),
        "plan_sha256": file_hash(args.plan),
        "plan_content_sha256": plan["plan_sha256"],
        "source_runtime": str(source_runtime),
        "model": args.model,
        "endpoint_identity_sha256": stable_hash(args.endpoint),
        "runtime_code_sha256": _runtime_receipts(source_runtime),
        "completed_snapshots": 0,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from openai import OpenAI

    client = OpenAI(
        base_url=args.endpoint,
        api_key="EMPTY",
        timeout=180.0,
        max_retries=1,
    )
    all_rows: list[dict[str, Any]] = []
    try:
        with rows_path.open("a", encoding="utf-8") as stream:
            for snapshot in plan["snapshots"]:
                policy = FrozenTreatmentPolicy(
                    client, model=args.model, snapshot=snapshot,
                )
                rows = run_matched_multihorizon_snapshot(
                    lambda: SourceForkEnvironment(
                        game=str(plan["game"]),
                        maximum_steps=int(plan["maximum_steps"]),
                    ),
                    policy,
                    episode_seed=int(snapshot["episode_seed"]),
                    episode_id=str(snapshot["episode_id"]),
                    fork_step=int(snapshot["fork_step"]),
                    prefix_actions=[str(item) for item in snapshot["prefix_actions"]],
                    expected_fork_observable_hash=str(
                        snapshot["expected_fork_observable_hash"]
                    ),
                    split=str(snapshot["split"]),
                    maximum_horizon=int(plan["maximum_horizon"]),
                )
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                all_rows.extend(rows)
                manifest["completed_snapshots"] += 1
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    finally:
        client.close()

    report = analyze_multihorizon_rows(all_rows)
    report.update({
        "schema_version": 1,
        "claim_boundary": plan["claim_boundary"],
        "plan_content_sha256": plan["plan_sha256"],
        "rows_sha256": file_hash(rows_path),
    })
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update({
        "status": "COMPLETE",
        "rows_file": rows_path.name,
        "rows_sha256": file_hash(rows_path),
        "report_file": report_path.name,
        "report_sha256": file_hash(report_path),
        "gates": report["gates"],
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed_snapshots": manifest["completed_snapshots"],
        "gates": report["gates"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
