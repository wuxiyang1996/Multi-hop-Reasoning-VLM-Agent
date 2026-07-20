"""Deterministically replay source-program receipts against downloaded JSON."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from skill_bank.program_ir import CanonicalSkillProgram
from skill_bank.source_effects import extract_source_effects


def _hash_text(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class SourceReplayReceipt:
    program_id: str
    program_hash: str
    passed: bool
    n_evidence: int
    n_verified: int
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "program_hash": self.program_hash,
            "passed": self.passed,
            "n_evidence": self.n_evidence,
            "n_verified": self.n_verified,
            "failures": list(self.failures),
        }


class SourceReplayValidator:
    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root)
        self._episodes: Dict[Tuple[str, str], tuple[Path, bytes, Mapping[str, Any]]] = {}
        for path in sorted(self.source_root.glob("*/episode_*.json")):
            raw = path.read_bytes()
            payload = json.loads(raw)
            game = str(payload.get("game_name") or payload.get("metadata", {}).get("game") or path.parent.name)
            episode_id = str(payload.get("episode_id") or path.stem)
            key = (game, episode_id)
            if key in self._episodes:
                raise ValueError(f"duplicate source episode key: {key}")
            self._episodes[key] = (path, raw, payload)

    def validate(self, program: CanonicalSkillProgram) -> SourceReplayReceipt:
        program.validate()
        failures: List[str] = []
        verified = 0
        observed_effect_evidence: Dict[str, set[str]] = {}
        for evidence in program.evidence:
            episode_key = (evidence.key.game, evidence.key.episode_id)
            indexed = self._episodes.get(episode_key)
            if indexed is None:
                failures.append(f"missing_episode:{episode_key[0]}:{episode_key[1]}")
                continue
            path, raw, payload = indexed
            if hashlib.sha256(raw).hexdigest() != evidence.source_file_sha256:
                failures.append(f"file_hash_mismatch:{path}")
                continue
            experiences = payload.get("experiences") or []
            matches = [
                step for ordinal, step in enumerate(experiences)
                if int(step.get("idx", ordinal)) == evidence.key.step_index
            ]
            if len(matches) != 1:
                failures.append(
                    f"step_resolution_failed:{episode_key[1]}:{evidence.key.step_index}:{len(matches)}"
                )
                continue
            step = matches[0]
            observed = {
                "state": _hash_text(step.get("raw_state", step.get("state", ""))),
                "next": _hash_text(step.get("raw_next_state", step.get("next_state", ""))),
                "action": str(step.get("action") or "").strip(),
                "reward": float(step.get("reward") or 0.0),
                "done": bool(step.get("done")),
            }
            expected = {
                "state": evidence.state_sha256,
                "next": evidence.next_state_sha256,
                "action": evidence.action,
                "reward": evidence.reward,
                "done": evidence.done,
            }
            mismatches = [key for key in expected if observed[key] != expected[key]]
            if mismatches:
                failures.append(
                    f"transition_mismatch:{episode_key[1]}:{evidence.key.step_index}:{','.join(mismatches)}"
                )
                continue
            for effect in extract_source_effects(
                game=evidence.key.game,
                state=str(step.get("state") or ""),
                next_state=str(step.get("next_state") or ""),
                action=observed["action"],
                reward=observed["reward"],
                done=observed["done"],
            ):
                observed_effect_evidence.setdefault(effect, set()).add(evidence.key.stable_id())
            verified += 1
        for step in program.steps:
            for effect in step.effects:
                if effect.value_type != "typed_source_state_delta":
                    continue
                actual = observed_effect_evidence.get(effect.predicate, set())
                claimed = set(effect.evidence_step_ids)
                if actual != claimed:
                    failures.append(f"typed_effect_receipt_mismatch:{effect.predicate}")
        return SourceReplayReceipt(
            program_id=program.program_id,
            program_hash=program.content_hash(),
            passed=not failures and verified == len(program.evidence),
            n_evidence=len(program.evidence),
            n_verified=verified,
            failures=failures,
        )


__all__ = ["SourceReplayReceipt", "SourceReplayValidator"]
