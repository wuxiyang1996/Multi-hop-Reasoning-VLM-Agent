#!/usr/bin/env python3
"""Run compact, instrumented source episodes and mechanical replay forks."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper  # noqa: E402
from env_wrappers.gym_like import make_gaming_env  # noqa: E402
from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from harness.provider_clients import (  # noqa: E402
    StrictOpenAIResponsesClient,
    load_literal_secret,
)
from harness.replay_fork import ReplayForkVerifier  # noqa: E402
from harness.source_evidence_store import write_source_evidence_batch  # noqa: E402
from trainer.coevolution.episode_runner import run_episode_async  # noqa: E402
from trainer.coevolution.vllm_client import AsyncVLLMClient  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _runtime_code_receipt() -> dict[str, str]:
    relative_paths = (
        "scripts/run_instrumented_source_smoke.py",
        "trainer/coevolution/episode_runner.py",
        "trainer/coevolution/vllm_client.py",
        "harness/reasoning_event_log.py",
        "harness/source_evidence_store.py",
        "harness/provider_clients.py",
    )
    result = {}
    for relative in relative_paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"runtime receipt file missing: {relative}")
        result[relative] = _sha256(path)
    return result


@dataclass
class _ActorResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    adapter: None
    provider_usage: Mapping[str, Any]


class OpenRouterRolloutActor:
    """The production runner's minimal async ``generate`` interface."""

    def __init__(self, endpoint: str, model: str, api_key: str) -> None:
        self.model = model
        self._client = StrictOpenAIClient(endpoint, timeout_s=180, api_key=api_key)

    async def generate(
        self, prompt: str, *, adapter=None, temperature=None,
        max_tokens=None, stop=None,
    ) -> _ActorResult:
        del adapter, temperature, stop
        request_seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)

        def _call_with_endpoint_retry():
            endpoint_retries = 0
            for attempt in range(4):
                try:
                    reply, usage = self._client.complete(
                        model=self.model, prompt=prompt,
                        max_tokens=int(max_tokens or 128), seed=request_seed,
                    )
                    usage = dict(usage)
                    usage["endpoint_retry_count"] = endpoint_retries
                    usage["request_seed"] = request_seed
                    return reply, usage
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status != 429 and not (isinstance(status, int) and status >= 500):
                        raise
                    if attempt == 3:
                        raise
                    endpoint_retries += 1
                    time.sleep(2 ** attempt)
            raise RuntimeError("unreachable endpoint retry state")

        reply, usage = await asyncio.to_thread(_call_with_endpoint_retry)
        return _ActorResult(
            text=reply,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=float(usage.get("latency_s", 0.0) or 0.0) * 1000,
            adapter=None,
            provider_usage=dict(usage),
        )

    def close(self) -> None:
        self._client.close()


class OpenAIResponsesRolloutActor:
    """OpenAI Responses API adapter preserving the same local Harness parser."""

    def __init__(
        self, endpoint: str, model: str, api_key: str, reasoning_effort: str,
        reasoning_token_reserve: int,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_token_reserve = reasoning_token_reserve
        self._client = StrictOpenAIResponsesClient(
            endpoint, timeout_s=180, api_key=api_key,
        )

    async def generate(
        self, prompt: str, *, adapter=None, temperature=None,
        max_tokens=None, stop=None,
    ) -> _ActorResult:
        del adapter, temperature, stop

        def _call_with_endpoint_retry():
            endpoint_retries = 0
            for attempt in range(4):
                try:
                    visible_token_budget = int(max_tokens or 128)
                    reply, usage = self._client.complete(
                        model=self.model,
                        prompt=prompt,
                        # OpenAI counts private reasoning and visible JSON in
                        # one max_output_tokens budget.  Preserve the runner's
                        # visible-output allowance by adding an explicit,
                        # separately receipted reasoning reserve.
                        max_tokens=(
                            visible_token_budget + self.reasoning_token_reserve
                        ),
                        reasoning_effort=self.reasoning_effort,
                    )
                    usage = dict(usage)
                    usage["harness_visible_token_budget"] = visible_token_budget
                    usage["api_reasoning_token_reserve"] = self.reasoning_token_reserve
                    usage["endpoint_retry_count"] = endpoint_retries
                    # Responses does not expose a sampling seed.  The prompt hash
                    # is the stable request identity; never pretend determinism.
                    usage["request_prompt_sha256"] = hashlib.sha256(
                        prompt.encode()
                    ).hexdigest()
                    return reply, usage
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status != 429 and not (isinstance(status, int) and status >= 500):
                        raise
                    if attempt == 3:
                        raise
                    endpoint_retries += 1
                    time.sleep(2 ** attempt)
            raise RuntimeError("unreachable endpoint retry state")

        reply, usage = await asyncio.to_thread(_call_with_endpoint_retry)
        return _ActorResult(
            text=reply,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=float(usage.get("latency_s", 0.0) or 0.0) * 1000,
            adapter=None,
            provider_usage=dict(usage),
        )

    def close(self) -> None:
        self._client.close()


class LocalCheckpointRolloutActor:
    """Adapter-aware local actor which never silently accepts base fallback."""

    def __init__(self, endpoints: list[str], model: str) -> None:
        self.model = model
        self._client = AsyncVLLMClient(
            base_urls=endpoints,
            model=model,
            default_temperature=0.0,
            default_max_tokens=128,
            timeout=180.0,
            max_retries=1,
        )
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, prompt: str, *, adapter=None, temperature=None,
        max_tokens=None, stop=None,
    ):
        # Bind stochastic sampling to the exact request instead of the shared
        # server request order.  A shadow observer can then make extra calls
        # without perturbing the source Decision policy's random stream.
        seed_payload = json.dumps(
            {"adapter": adapter, "prompt": prompt}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        )
        request_seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:8], 16)
        result = await self._client.generate(
            prompt,
            adapter=adapter,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_body={"seed": request_seed},
        )
        self.calls.append({
            "requested_adapter": adapter,
            "used_adapter": result.adapter,
            "nonempty_response": bool(result.text.strip()),
            "request_seed": request_seed,
        })
        # AsyncVLLMClient deliberately supports base fallback for production
        # availability.  Evidence collection has the opposite requirement:
        # a missing checkpoint must invalidate the run, not change treatment.
        if adapter in {"action_taking", "skill_selection"}:
            if result.adapter != adapter:
                raise RuntimeError(
                    f"checkpoint_adapter_fallback:{adapter}->{result.adapter}"
                )
            if not result.text.strip():
                raise RuntimeError(f"checkpoint_adapter_empty_response:{adapter}")
        return result

    async def close(self) -> None:
        for client in self._client._clients:  # owned clients; no public close API
            await client.close()


def _checkpoint_receipt(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    required = {
        "skill_selection": checkpoint / "adapters/decision/skill_selection",
        "action_taking": checkpoint / "adapters/decision/action_taking",
        "segment": checkpoint / "adapters/skillbank/segment",
        "contract": checkpoint / "adapters/skillbank/contract",
        "curator": checkpoint / "adapters/skillbank/curator",
    }
    files: dict[str, str] = {}
    for name, root in required.items():
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = root / filename
            if not path.is_file():
                raise SystemExit(f"missing checkpoint file: {path}")
            files[str(path.relative_to(checkpoint))] = _sha256(path)
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "path": str(checkpoint),
        "metadata": metadata,
        "files_sha256": files,
    }


class _SourceReplayAdapter:
    def __init__(self, game: str, max_steps: int) -> None:
        if game.startswith("gymv_"):
            from env_wrappers.subprocess_env import SubprocessEnv
            self._env = SubprocessEnv(
                game=game, max_steps=max_steps, env_kind="gymv",
            )
        else:
            base = GamingAgentNLWrapper(
                make_gaming_env(game=game, max_steps=max_steps)
            )
            if game == "tetris":
                from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
                self._env = TetrisMacroActionWrapper(base)
            else:
                self._env = base
        self._observation = ""
        self._actions: list[str] = []
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False

    def reset(self, *, seed: int):
        self._observation, info = self._env.reset(seed=seed)
        self._actions = [str(item) for item in info.get("action_names", ())]
        return self._observation

    def state_receipt(self):
        return self._observation

    def admissible_actions(self):
        return tuple(self._actions)

    def step(self, action: str):
        self._observation, reward, terminated, truncated, info = self._env.step(action)
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self._actions = [str(item) for item in info.get("action_names", ())]
        return self._observation

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


def _events_by_kind(result, kind: str):
    return [
        row for row in result.reasoning_event_log["events"]
        if row["kind"] == kind
    ]


def _run_replay_forks(results, *, game: str, seed_base: int, max_steps: int):
    receipts = []
    verifier = ReplayForkVerifier()
    for episode_index, result in enumerate(results):
        steps = _events_by_kind(result, "ENVIRONMENT_STEP")
        observations = _events_by_kind(result, "OBSERVATION")
        admissibility = _events_by_kind(result, "NATIVE_ADMISSIBILITY")
        if not steps or len(observations) < 2 or len(admissibility) < 2:
            continue
        prefix = [str(steps[0]["payload"]["executed_action"])]
        fork_observation = next(
            row for row in observations if int(row["payload"]["step"]) == 1
        )
        at_fork = next(
            row for row in admissibility if int(row["payload"]["step"]) == 1
        )
        original_next = str(steps[1]["payload"]["executed_action"]) if len(steps) > 1 else None
        alternatives = [
            str(item) for item in at_fork["payload"]["native_actions"]
            if str(item) != original_next
        ]
        for alternative_index, alternative in enumerate(alternatives):
            adapter = _SourceReplayAdapter(game, max_steps)
            try:
                receipt = verifier.run(
                    adapter,
                    intervention_id=(
                        f"{result.episode_id}.fork_step_1.alt_{alternative_index}"
                    ),
                    seed=seed_base + episode_index,
                    prefix_actions=prefix,
                    expected_fork_state_sha256=str(
                        fork_observation["payload"]["observable_state_sha256"]
                    ),
                    alternative_action=alternative,
                )
                receipts.append(asdict(receipt) | {"receipt_sha256": receipt.content_hash()})
            finally:
                adapter.close()
    return receipts


async def _main(args) -> int:
    checkpoint_receipt = None
    bank_receipt = None
    skill_bank = None
    if args.checkpoint is not None:
        checkpoint_receipt = _checkpoint_receipt(args.checkpoint)
        endpoints = [item.strip() for item in args.endpoint.split(",") if item.strip()]
        if not endpoints or any("openrouter.ai" in item.lower() for item in endpoints):
            raise SystemExit("--checkpoint requires local adapter-aware vLLM endpoint(s)")
        actor = LocalCheckpointRolloutActor(endpoints, args.model)
        if args.skill_bank is not None:
            from scripts.qwen3_decision_agent import load_skill_bank
            bank_path = args.skill_bank.resolve()
            if not bank_path.is_file():
                raise SystemExit(f"missing skill bank: {bank_path}")
            source_bank, query_engine = load_skill_bank(
                str(bank_path), use_query_engine=True,
            )
            if query_engine is None:
                raise RuntimeError("skill_query_engine_initialization_failed")
            skill_bank = query_engine
            bank_receipt = {
                "path": str(bank_path),
                "sha256": _sha256(bank_path),
                "n_skills": len(source_bank),
                "query_engine": type(query_engine).__name__,
                "candidate_retrieval": "semantic_select",
            }
    else:
        if args.skill_bank is not None:
            raise SystemExit("--skill-bank requires --checkpoint")
        # An explicit file is an explicit credential choice and must override
        # inherited scheduler/login-shell variables (which are often stale).
        key = ""
        if args.api_key_file is not None:
            try:
                key = load_literal_secret(args.api_key_file, args.api_key_env)
            except ValueError as exc:
                raise SystemExit(f"API key unavailable: {exc}") from exc
        if not key:
            key = os.environ.get(args.api_key_env, "").strip()
        if not key and args.provider == "openrouter":
            try:
                from API_func import open_router_api_key
                key = str(open_router_api_key or "").strip()
            except Exception:
                key = ""
        if not key:
            raise SystemExit(f"{args.provider} API key unavailable")
        if args.provider == "openai":
            actor = OpenAIResponsesRolloutActor(
                args.endpoint, args.model, key, args.reasoning_effort,
                args.openai_reasoning_token_reserve,
            )
        else:
            actor = OpenRouterRolloutActor(args.endpoint, args.model, key)
    try:
        async def _episode(index):
            return await run_episode_async(
                game=args.game,
                max_steps=args.max_steps,
                vllm_client=actor,
                skill_bank=skill_bank,
                temperature=0.0,
                stuck_window=args.max_steps + 1,
                min_steps_before_stuck=args.max_steps + 1,
                episode_seed=args.seed_base + index,
                record_reasoning_events=True,
                reasoning_backbone_harness=args.reasoning_backbone_harness,
                matched_policy_skill_id=args.matched_policy_skill_id,
            )

        if args.reasoning_backbone_harness or args.sequential_episodes:
            # Preserve a stable request order and avoid provider throttling
            # changing v2 receipt coverage. Endpoint-only retries happen inside
            # OpenRouterRolloutActor; model/schema failures are never redrawn.
            results = []
            for index in range(args.episodes):
                results.append(await _episode(index))
        else:
            results = await asyncio.gather(*[
                _episode(index) for index in range(args.episodes)
            ])
    finally:
        maybe_close = actor.close()
        if asyncio.iscoroutine(maybe_close):
            await maybe_close
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = write_source_evidence_batch(
        args.output, results,
        manifest_metadata={
            "game": args.game,
            "model": args.model,
            "provider": args.provider if checkpoint_receipt is None else "local_vllm",
            "reasoning_effort": (
                args.reasoning_effort if args.provider == "openai" else None
            ),
            "openai_reasoning_token_reserve": (
                args.openai_reasoning_token_reserve
                if args.provider == "openai" else None
            ),
            "adapter": (
                "checkpoint_multi_lora" if checkpoint_receipt is not None else None
            ),
            "lora_checkpoint_loaded": checkpoint_receipt is not None,
            "checkpoint_receipt": checkpoint_receipt,
            "skill_bank_receipt": bank_receipt,
            "episode_seed_base": args.seed_base,
            "max_steps": args.max_steps,
            "sequential_episodes": args.sequential_episodes,
            "git_commit": _git_commit(),
            "runtime_code_sha256": _runtime_code_receipt(),
            "working_tree_changes_may_exist": True,
            "reasoning_backbone_harness": args.reasoning_backbone_harness,
            "human_policy_hints": False,
            "policy_hint_profile": "NO_HUMAN_GAME_HINTS_V1",
            "split_contract": (
                "SORT_EPISODE_ID_THEN_ROUND_ROBIN_"
                "DISCOVERY_QUALIFICATION_HELD_OUT_V1"
            ),
            "matched_policy_primary_horizon": 8,
            "matched_policy_collection_horizon": 1,
            "multihorizon_execution_stage": "AFTER_DISCOVERY_CANDIDATE_FREEZE",
        },
        protocol_profile=(
            "source_agent_v2" if args.reasoning_backbone_harness else "source_agent"
        ),
    )
    receipts = _run_replay_forks(
        results, game=args.game, seed_base=args.seed_base, max_steps=args.max_steps,
    )
    replay_path = args.output / "replay_receipts.jsonl"
    replay_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts),
        encoding="utf-8",
    )
    manifest = dict(manifest)
    manifest["replay_forks"] = {
        "n_receipts": len(receipts),
        "status_counts": {
            status: sum(row["status"] == status for row in receipts)
            for status in sorted({row["status"] for row in receipts})
        },
        "file": "replay_receipts.jsonl",
        "sha256": _sha256(replay_path),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    if args.matched_policy_skill_id:
        matched = [
            row for result in results
            for row in getattr(result, "matched_policy_records", [])
        ]
        if not matched:
            raise RuntimeError(
                f"no matched policy snapshots for {args.matched_policy_skill_id}"
            )
        treatment_counts = Counter(str(row["treatment"]) for row in matched)
        expected_treatments = {"B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM"}
        if set(treatment_counts) != expected_treatments:
            raise RuntimeError(f"incomplete matched treatments: {dict(treatment_counts)}")
        snapshot_count = len(matched) // 4
        if any(value != snapshot_count for value in treatment_counts.values()):
            raise RuntimeError(f"unbalanced matched treatments: {dict(treatment_counts)}")
        # Persist expensive model outputs before emulator replay. The same file
        # is enriched and overwritten after every replay passes.
        matched_path = args.output / "matched_policy_records.jsonl"
        matched_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in matched),
            encoding="utf-8",
        )

        def _replay_matched(row):
            last_error = None
            for _attempt in range(3):
                adapter = None
                try:
                    adapter = _SourceReplayAdapter(args.game, args.max_steps)
                    receipt = ReplayForkVerifier().run(
                        adapter,
                        intervention_id=(
                            f"{row['episode_id']}.matched_step_{row['step']}.{row['treatment']}"
                        ),
                        seed=int(row["episode_seed"]),
                        prefix_actions=[str(item) for item in row["prefix_actions"]],
                        expected_fork_state_sha256=str(row["before_observable_sha256"]),
                        alternative_action=str(row["parsed_action"]),
                    )
                    return asdict(receipt) | {
                        "receipt_sha256": receipt.content_hash(),
                        "alternative_reward": adapter.last_reward,
                        "alternative_terminated": adapter.last_terminated,
                        "alternative_truncated": adapter.last_truncated,
                        "startup_attempt": _attempt + 1,
                    }
                except Exception as exc:
                    last_error = exc
                finally:
                    if adapter is not None:
                        adapter.close()
            raise RuntimeError("matched_replay_startup_retries_exhausted") from last_error

        with ThreadPoolExecutor(max_workers=args.matched_replay_workers) as pool:
            matched_replays = list(pool.map(_replay_matched, matched))
        replay_by_id = {
            str(row["intervention_id"]): row for row in matched_replays
        }
        for row in matched:
            intervention_id = (
                f"{row['episode_id']}.matched_step_{row['step']}.{row['treatment']}"
            )
            replay = replay_by_id[intervention_id]
            row["replay_receipt_sha256"] = replay["receipt_sha256"]
            row["replay_status"] = replay["status"]
            row["after_observable_sha256"] = replay["alternative_next_state_sha256"]

        matched_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in matched),
            encoding="utf-8",
        )
        matched_replay_path = args.output / "matched_policy_replays.jsonl"
        matched_replay_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in matched_replays),
            encoding="utf-8",
        )
        manifest["matched_policy_treatments"] = {
            "source_skill_id": args.matched_policy_skill_id,
            "snapshot_count": snapshot_count,
            "treatment_counts": dict(sorted(treatment_counts.items())),
            "replay_status_counts": dict(sorted(Counter(
                str(row["status"]) for row in matched_replays
            ).items())),
            "records_file": matched_path.name,
            "records_sha256": _sha256(matched_path),
            "replays_file": matched_replay_path.name,
            "replays_sha256": _sha256(matched_replay_path),
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    summary = {
        "episodes": len(results),
        "steps": [item.steps for item in results],
        "rewards": [item.total_reward for item in results],
        "protocol_failure_episodes": len(manifest["protocol_failures"]),
        "replay_forks": manifest["replay_forks"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not manifest["protocol_failures"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game", default="twenty_forty_eight")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--matched-policy-skill-id", default=None,
        help="Collect B/G-S/G+S/G+Random at snapshots selecting this exact skill ID.",
    )
    parser.add_argument(
        "--matched-replay-workers", type=int, default=4,
        help="Parallel environment workers for one-step matched replay receipts.",
    )
    parser.add_argument(
        "--sequential-episodes", action="store_true",
        help="Run resets/episodes serially so per-seed initial states cannot race.",
    )
    parser.add_argument("--endpoint", default="https://openrouter.ai/api")
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--provider", choices=("openrouter", "openai"), default="openrouter")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument(
        "--reasoning-effort", choices=("minimal", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument(
        "--openai-reasoning-token-reserve", type=int, default=512,
        help=(
            "Extra max_output_tokens reserved for private OpenAI reasoning; "
            "the original runner max_tokens remains the visible JSON budget."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Full co-evolution checkpoint root containing adapters/ and metadata.json.",
    )
    parser.add_argument(
        "--skill-bank", type=Path, default=None,
        help="Optional exact skill_bank.jsonl to use with the checkpoint.",
    )
    parser.add_argument(
        "--reasoning-backbone-harness", action="store_true",
        help=(
            "Collect Agent proposal/prediction and post-transition verification "
            "receipts; invalid cycles remain logged but cannot enter source programs."
        ),
    )
    args = parser.parse_args()
    if args.openai_reasoning_token_reserve < 0:
        parser.error("--openai-reasoning-token-reserve must be non-negative")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
