"""Leakage-safe DiscoveryWorld adapter with deterministic replay receipts.

DiscoveryWorld exposes a policy-facing observation API and a separate task
scorecard containing oracle knowledge.  This module keeps those channels
separate by construction: policies receive :class:`DiscoveryWorldObservation`,
while the scorecard can only be requested after an episode terminates.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import threading
from typing import Any, Callable, Mapping, Sequence


FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "criticalhypotheses",
        "criticalquestions",
        "oraclescorecard",
        "score",
        "scorecard",
        "scorecardnormalized",
        "scorenormalized",
        "taskscores",
    }
)


# DiscoveryWorld currently has two stochastic channels that are not controlled
# by its scenario seed: Python's process-global ``random`` module (used by NPC
# wandering) and an unseeded ``random.Random`` owned by each world object.  A
# paired intervention is invalid unless both channels are coupled across forks.
DETERMINISM_PROTOCOL = "discoveryworld-isolated-common-random-numbers-v1"
_OFFICIAL_RNG_LOCK = threading.RLock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def _jsonable(value: Any) -> Any:
    """Canonicalize official hidden-state snapshots for audit hashing only."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=_canonical_json)
    raise TypeError(f"unsupported DiscoveryWorld audit value: {type(value).__name__}")


def _canonical_audit_snapshot(value: Any) -> Any:
    """Remove only non-state wall-clock, rendering, and score-note metadata.

    DiscoveryWorld chooses some terrain sprite variants with Python's process
    global RNG.  The variants are rendering cache, not MDP state: semantic
    object type, attributes, location, contents, parts, tasks, and feed state
    remain in the snapshot.  The evaluator's ``associatedNotes`` field is a
    redundant human-readable rendering of score state; some tasks interpolate
    a Python set into it, making word order process-dependent.  Its structured
    score and UUID fields remain committed by the audit hash.
    """

    if isinstance(value, Mapping):
        output = {
            str(key): _canonical_audit_snapshot(item)
            for key, item in value.items()
            if str(key) not in {
                "runtime_seconds", "spriteModifiers", "spriteNames", "associatedNotes",
            }
        }
        associated_uuids = output.get("associatedUUIDs")
        if isinstance(associated_uuids, list):
            associated_uuids.sort(key=_canonical_json)
        # Rendering can reorder colocated top-level objects.  A world tile is a
        # semantic multiset; object identity and containment remain explicit.
        grid = output.get("grid")
        if isinstance(grid, list):
            for column in grid:
                if not isinstance(column, list):
                    continue
                for tile in column:
                    if isinstance(tile, list) and all(isinstance(row, Mapping) for row in tile):
                        tile.sort(key=lambda row: (int(row.get("uuid", -1)), str(row.get("type", ""))))
        return output
    if isinstance(value, (list, tuple)):
        return [_canonical_audit_snapshot(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_audit_snapshot(item) for item in value]
        return sorted(items, key=_canonical_json)
    return _jsonable(value)


def _assert_no_oracle_fields(value: Any, path: str = "policy") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("_", "").replace("-", "")
            if normalized in FORBIDDEN_POLICY_KEYS:
                raise ValueError(f"oracle field escaped into policy channel: {path}.{key}")
            _assert_no_oracle_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_oracle_fields(item, f"{path}[{index}]")


def _task_terminal(ui: Mapping[str, Any]) -> tuple[bool, bool]:
    progress = list(ui.get("taskProgress") or ())
    if not progress:
        raise ValueError("DiscoveryWorld observation has no taskProgress")
    terminal = all(bool(row.get("completed")) for row in progress)
    success = terminal and all(bool(row.get("completedSuccessfully")) for row in progress)
    return terminal, success


def _validate_action(
    action: Mapping[str, Any], known_actions: Mapping[str, Any], *, in_dialog: bool,
) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        raise TypeError("DiscoveryWorld action must be a JSON mapping")
    if in_dialog:
        if set(action) != {"chosen_dialog_option_int"}:
            raise ValueError("dialog action must contain only chosen_dialog_option_int")
        choice = action["chosen_dialog_option_int"]
        if not isinstance(choice, int) or isinstance(choice, bool) or choice < 0:
            raise ValueError("chosen_dialog_option_int must be a nonnegative integer")
        return {"chosen_dialog_option_int": choice}
    name = str(action.get("action") or "")
    if name not in known_actions:
        raise ValueError(f"unknown DiscoveryWorld action: {name!r}")
    required = tuple(str(value) for value in known_actions[name].get("args") or ())
    allowed = {"action", *required}
    missing = [key for key in required if key not in action]
    extra = sorted(set(map(str, action)) - allowed)
    if missing or extra:
        raise ValueError(f"invalid {name} arguments: missing={missing}, extra={extra}")
    return {key: action[key] for key in ("action", *required)}


@dataclass(frozen=True)
class DiscoveryWorldObservation:
    scenario: str
    difficulty: str
    seed: int
    episode_step: int
    ui: Mapping[str, Any]
    known_actions: Mapping[str, Any]
    teleport_locations: Mapping[str, Any]
    last_action_result: Mapping[str, Any] | None
    vision: Mapping[str, str] | None
    in_dialog: bool
    terminal: bool
    official_success: bool

    def policy_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        _assert_no_oracle_fields(payload)
        return payload

    @property
    def policy_state_sha256(self) -> str:
        return stable_hash(self.policy_payload())


@dataclass(frozen=True)
class DiscoveryWorldTransitionReceipt:
    schema_version: str
    scenario: str
    difficulty: str
    seed: int
    episode_step: int
    before_policy_state_sha256: str
    before_audit_world_sha256: str
    action: Mapping[str, Any]
    action_result: Mapping[str, Any]
    action_succeeded: bool
    tick_result: Mapping[str, Any]
    after_policy_state_sha256: str
    after_audit_world_sha256: str
    terminal: bool
    official_success: bool
    runtime_saw_oracle_scorecard: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scenario: str,
        difficulty: str,
        seed: int,
        episode_step: int,
        before: DiscoveryWorldObservation,
        before_audit_hash: str,
        action: Mapping[str, Any],
        action_result: Mapping[str, Any],
        tick_result: Mapping[str, Any],
        after: DiscoveryWorldObservation,
        after_audit_hash: str,
    ) -> "DiscoveryWorldTransitionReceipt":
        body = {
            "schema_version": "discoveryworld-transition-v1",
            "scenario": scenario,
            "difficulty": difficulty,
            "seed": int(seed),
            "episode_step": int(episode_step),
            "before_policy_state_sha256": before.policy_state_sha256,
            "before_audit_world_sha256": before_audit_hash,
            "action": dict(action),
            "action_result": dict(action_result),
            "action_succeeded": bool(action_result.get("success")),
            "tick_result": dict(tick_result),
            "after_policy_state_sha256": after.policy_state_sha256,
            "after_audit_world_sha256": after_audit_hash,
            "terminal": bool(after.terminal),
            "official_success": bool(after.official_success),
            "runtime_saw_oracle_scorecard": False,
        }
        return cls(receipt_sha256=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        expected = body.pop("receipt_sha256")
        return expected == stable_hash(body)


@dataclass(frozen=True)
class DiscoveryWorldEvaluationReceipt:
    schema_version: str
    scenario: str
    difficulty: str
    seed: int
    episode_steps: int
    terminal: bool
    official_success: bool
    scorecard: Any
    scorecard_sha256: str
    policy_runtime_saw_oracle_scorecard: bool


class DiscoveryWorldEnvironment:
    """A fresh-world environment whose policy channel excludes scorecards."""

    def __init__(
        self,
        *,
        scenario: str,
        difficulty: str,
        seed: int,
        max_steps: int,
        thread_id: int = 1,
        include_vision: bool = False,
        frame_dir: str | Path | None = None,
        api_factory: Callable[[int], Any] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.scenario = str(scenario)
        self.difficulty = str(difficulty)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.thread_id = int(thread_id)
        self.include_vision = bool(include_vision)
        self.frame_dir = Path(frame_dir).resolve() if frame_dir is not None else None
        self._api_factory = api_factory
        self.api: Any = None
        self.episode_step = 0
        self.current: DiscoveryWorldObservation | None = None
        self.current_audit_hash = ""
        self.terminated = False
        self._global_random_state: object | None = None

    def _reset_random_protocol(self) -> None:
        seed = _stable_seed(
            DETERMINISM_PROTOCOL, self.scenario, self.difficulty, self.seed,
        )
        self._global_random_state = random.Random(seed).getstate()

    @contextmanager
    def _isolated_official_randomness(self, *, seed_new_objects: bool = False):
        """Run official code with episode-local common random numbers.

        The official implementation uses Python's module-global RNG.  Saving and
        restoring it makes separate environment objects independent, while the
        lock prevents concurrent in-process episodes from interleaving draws.
        During scenario construction we additionally seed every Object RNG from
        stable episode identity plus the official UUID.  The patch is scoped to
        construction and the upstream checkout remains unchanged.
        """

        if self._global_random_state is None:
            raise RuntimeError("DiscoveryWorld random protocol was not initialized")
        with _OFFICIAL_RNG_LOCK:
            process_state = random.getstate()
            random.setstate(self._global_random_state)
            object_class = None
            original_init = None
            try:
                if seed_new_objects and self._api_factory is None:
                    from discoveryworld.objects.Object import Object

                    object_class = Object
                    original_init = Object.__init__
                    episode_identity = (
                        DETERMINISM_PROTOCOL,
                        self.scenario,
                        self.difficulty,
                        self.seed,
                    )

                    def deterministic_object_init(
                        obj: Any,
                        world: Any,
                        objectType: str,
                        objectName: str,
                        defaultSpriteName: str,
                        rngSeed: int | None = None,
                    ) -> None:
                        assert original_init is not None
                        original_init(
                            obj,
                            world,
                            objectType,
                            objectName,
                            defaultSpriteName,
                            rngSeed=rngSeed,
                        )
                        object_seed = (
                            int(rngSeed)
                            if rngSeed is not None
                            else _stable_seed(
                                *episode_identity,
                                "object-rng",
                                int(obj.uuid),
                                str(objectType),
                                str(objectName),
                            )
                        )
                        obj.seed(object_seed)

                    Object.__init__ = deterministic_object_init
                yield
            finally:
                self._global_random_state = random.getstate()
                if object_class is not None and original_init is not None:
                    object_class.__init__ = original_init
                random.setstate(process_state)

    def _make_api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory(self.thread_id)
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

        return DiscoveryWorldAPI(threadID=self.thread_id)

    def _audit_world_hash(self) -> str:
        world = getattr(self.api, "world", None)
        history = list(getattr(world, "worldHistory", ()) or ())
        getter = getattr(world, "getWorldHistoryAtStep", None)
        if not history or not callable(getter):
            raise RuntimeError("DiscoveryWorld did not expose an audit world snapshot")
        snapshot = getter(len(history) - 1)
        rng_states: dict[str, Any] = {
            "protocol": DETERMINISM_PROTOCOL,
            "isolated_python_random": self._global_random_state,
        }
        for label, owner, attribute in (
            ("api", self.api, "r"),
            ("world", world, "rng"),
            ("uuid_generator", getattr(world, "uuidGenerator", None), "random"),
        ):
            generator = getattr(owner, attribute, None)
            getstate = getattr(generator, "getstate", None)
            if callable(getstate):
                rng_states[label] = getstate()
        object_states = []
        get_objects = getattr(world, "getAllWorldObjects", None)
        if callable(get_objects):
            objects_by_uuid = {
                int(obj.uuid): obj
                for obj in get_objects()
                if hasattr(obj, "uuid")
            }
            for uuid, obj in sorted(objects_by_uuid.items()):
                getstate = getattr(getattr(obj, "rng", None), "getstate", None)
                if callable(getstate):
                    object_states.append({"uuid": uuid, "state": getstate()})
        rng_states["objects"] = object_states
        return stable_hash({
            "world_snapshot": _canonical_audit_snapshot(snapshot),
            "future_random_state": _jsonable(rng_states),
        })

    def _observe(self, last_action_result: Mapping[str, Any] | None) -> DiscoveryWorldObservation:
        raw = self.api.getAgentObservation(agentIdx=0)
        errors = list(raw.get("errors") or ())
        if errors:
            raise RuntimeError(f"DiscoveryWorld observation failed: {errors}")
        ui = raw.get("ui")
        if not isinstance(ui, Mapping):
            raise RuntimeError("DiscoveryWorld observation did not contain a UI mapping")
        natural_terminal, natural_success = _task_terminal(ui)
        terminal = natural_terminal or self.episode_step >= self.max_steps
        vision = raw.get("vision") if self.include_vision else None
        dialog_probe = getattr(self.api, "isAgentInDialog", None)
        if not callable(dialog_probe):
            raise RuntimeError("DiscoveryWorld did not expose dialog state")
        observation = DiscoveryWorldObservation(
            scenario=self.scenario,
            difficulty=self.difficulty,
            seed=self.seed,
            episode_step=self.episode_step,
            ui=dict(ui),
            known_actions=dict(self.api.listKnownActions(limited=False)),
            teleport_locations=dict(self.api.listTeleportLocationsDict()),
            last_action_result=(dict(last_action_result) if last_action_result is not None else None),
            vision=(dict(vision) if isinstance(vision, Mapping) else None),
            in_dialog=bool(dialog_probe(agentIdx=0)),
            terminal=terminal,
            official_success=natural_success,
        )
        observation.policy_payload()  # fail closed on oracle leakage
        return observation

    def reset(self) -> DiscoveryWorldObservation:
        self._reset_random_protocol()
        with self._isolated_official_randomness(seed_new_objects=True):
            self.api = self._make_api()
            if self.frame_dir is not None:
                self.frame_dir.mkdir(parents=True, exist_ok=True)
                self.api.FRAME_DIR = str(self.frame_dir) + "/"
            loaded = self.api.loadScenario(
                scenarioName=self.scenario,
                difficultyStr=self.difficulty,
                randomSeed=self.seed,
                numUserAgents=1,
            )
        if not loaded:
            raise RuntimeError(
                f"DiscoveryWorld failed to load {self.scenario}/{self.difficulty}/seed={self.seed}"
            )
        self.episode_step = 0
        self.terminated = False
        self.current = self._observe(None)
        self.current_audit_hash = self._audit_world_hash()
        self.terminated = self.current.terminal
        return self.current

    def step(
        self, action: Mapping[str, Any],
    ) -> tuple[DiscoveryWorldObservation, DiscoveryWorldTransitionReceipt]:
        if self.api is None or self.current is None:
            raise RuntimeError("reset must be called before step")
        if self.terminated:
            raise RuntimeError("cannot step a terminated DiscoveryWorld episode")
        clean_action = _validate_action(
            action, self.current.known_actions, in_dialog=self.current.in_dialog,
        )
        before = self.current
        before_audit = self.current_audit_hash
        with self._isolated_official_randomness():
            action_result = self.api.performAgentAction(agentIdx=0, actionJSON=clean_action)
            if not isinstance(action_result, Mapping):
                raise RuntimeError("DiscoveryWorld returned a non-mapping action result")
            tick_result = self.api.tick()
        if not isinstance(tick_result, Mapping) or not bool(tick_result.get("success")):
            raise RuntimeError(f"DiscoveryWorld tick failed: {tick_result}")
        self.episode_step += 1
        after = self._observe(action_result)
        after_audit = self._audit_world_hash()
        receipt = DiscoveryWorldTransitionReceipt.create(
            scenario=self.scenario,
            difficulty=self.difficulty,
            seed=self.seed,
            episode_step=self.episode_step,
            before=before,
            before_audit_hash=before_audit,
            action=clean_action,
            action_result=action_result,
            tick_result=tick_result,
            after=after,
            after_audit_hash=after_audit,
        )
        self.current = after
        self.current_audit_hash = after_audit
        self.terminated = after.terminal
        return after, receipt

    def replay_prefix(
        self,
        actions: Sequence[Mapping[str, Any]],
        *,
        expected_policy_state_sha256: str | None = None,
        expected_audit_world_sha256: str | None = None,
    ) -> tuple[DiscoveryWorldObservation, tuple[DiscoveryWorldTransitionReceipt, ...]]:
        observation = self.reset()
        receipts = []
        for action in actions:
            observation, receipt = self.step(action)
            receipts.append(receipt)
        if (
            expected_policy_state_sha256 is not None
            and observation.policy_state_sha256 != expected_policy_state_sha256
        ):
            raise RuntimeError("DiscoveryWorld replay policy-state hash mismatch")
        if (
            expected_audit_world_sha256 is not None
            and self.current_audit_hash != expected_audit_world_sha256
        ):
            raise RuntimeError("DiscoveryWorld replay hidden-state hash mismatch")
        return observation, tuple(receipts)

    def finalize_evaluation(self) -> DiscoveryWorldEvaluationReceipt:
        if self.api is None or self.current is None or not self.terminated:
            raise RuntimeError("oracle evaluation is permitted only after episode termination")
        scorecard = self.api.getTaskScorecard()
        return DiscoveryWorldEvaluationReceipt(
            schema_version="discoveryworld-evaluation-v1",
            scenario=self.scenario,
            difficulty=self.difficulty,
            seed=self.seed,
            episode_steps=self.episode_step,
            terminal=True,
            official_success=bool(self.current.official_success),
            scorecard=scorecard,
            scorecard_sha256=stable_hash(_jsonable(scorecard)),
            policy_runtime_saw_oracle_scorecard=False,
        )


__all__ = [
    "DETERMINISM_PROTOCOL",
    "DiscoveryWorldEnvironment",
    "DiscoveryWorldEvaluationReceipt",
    "DiscoveryWorldObservation",
    "DiscoveryWorldTransitionReceipt",
    "FORBIDDEN_POLICY_KEYS",
    "stable_hash",
]
