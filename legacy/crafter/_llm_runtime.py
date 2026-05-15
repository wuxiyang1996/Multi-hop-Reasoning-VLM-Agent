"""LLM-backed implementations of the three Crafter teacher hooks.

Spec context: ``crafter/README.md`` §"Teacher-LLM integration" and
PLAN-SKILL-CRAFTER §2 ("Multi-run reasoning requirement"). Until this
module landed, the three setters

    Repairer.set_llm_repairer
    Hypothesizer.set_llm_proposer
    FailureDiagnoser.set_llm_diagnoser

were *dormant* (`None` defaults, deterministic rule path always wins).
This module is the README's "Step 0 prerequisite" — wire **one**
concrete hook through ``API_func.ask_model`` so the integration point
actually runs in production. Failure-mode 3 (constrained-JSON
decoding + retry + audit telemetry) is implemented inline as a single-
pass ``_call_json`` helper; future best-of-N / counterfactual
scaffolding (failure-mode 1) plugs in by replacing ``_call_json``
with a multi-pass driver, see ``crafter/README.md`` §integration
roadmap step 2.

Design constraints:

* **Crash-safe.** Every hook catches exceptions and returns ``None``
  so the rule path takes over (matches the contract documented at
  ``Repairer.repair`` / ``Hypothesizer.propose`` /
  ``FailureDiagnoser.diagnose``).
* **No new module deps.** Only ``API_func.ask_model`` +
  ``common.models.BACKBONE_TEACHER_MODEL`` — same import surface the
  rest of the project uses.
* **Telemetry as keyword args.** Each call optionally appends an
  audit row through a caller-supplied ``audit_sink`` (typically
  ``ArtifactStore.append_audit``) so the README's required counters
  (``crafter.llm.calls``, ``crafter.llm.parse_failures``,
  ``crafter.llm.fallthrough_to_rule``, ``crafter.llm.exceptions``)
  show up in the existing ``audit.jsonl`` stream without inventing a
  new transport.

The hooks intentionally produce *partial* proposals:

* ``LLMRepairer.__call__`` returns a ``PatchProposal`` with the
  patched protocol / contract filled in; the rationale + provenance
  fields are populated by the caller (`Repairer.repair`).
* ``LLMHypothesizer.__call__`` returns a ``HypothesisProposal``
  with ``novel_protocol`` + ``contract`` populated; the rationale
  / parent skill ids are set by `Hypothesizer.propose`.
* ``LLMDiagnoser.__call__`` returns a ``FailureDiagnosis`` with the
  recommended strategy + root cause filled in.

The JSON schemas the LLM is asked for are intentionally tiny — only
the fields that the dataclass requires. Any extra fields are
ignored. This keeps single-pass parse-failure rates manageable
without constrained decoding.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from common.enums import EVIDENCE_ROLES, RecoveryStrategy
from common.models import BACKBONE_TEACHER_MODEL
from data_structure.extensions.bank_mutation_proposal import (
    HypothesisProposal,
    PatchProposal,
)
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from .failure_memory import FailurePattern

logger = logging.getLogger("crafter.llm_runtime")

# ─────────────────────────────────────────────────────────────────────
# Public surface
# ─────────────────────────────────────────────────────────────────────

AuditSink = Callable[[Dict[str, Any]], None]

_DEFAULT_TEMP_REPAIR = 0.2     # patches should be near-deterministic
_DEFAULT_TEMP_DIAGNOSE = 0.0   # classification, want determinism
_DEFAULT_TEMP_HYPOTHESIZE = 0.4  # novel protocols, slight diversity
_DEFAULT_MAX_TOKENS = 1200


@dataclass
class LLMHookConfig:
    """Per-hook LLM call configuration.

    Defaults are tuned for ``BACKBONE_TEACHER_MODEL =
    "Qwen/Qwen3.5-35B-A3B"`` but every field is overridable so callers
    can flip to ``gpt-5.4`` for a smoke run without rebuilding the
    crafter.
    """

    model: str = field(default_factory=lambda: BACKBONE_TEACHER_MODEL)
    temperature: float = _DEFAULT_TEMP_REPAIR
    max_tokens: int = _DEFAULT_MAX_TOKENS
    enable_thinking: bool = False
    audit_sink: Optional[AuditSink] = None
    """Optional callback receiving one ``dict`` per LLM call. The dict
    has the schema documented in ``crafter/README.md`` §"telemetry
    counters" — kind / model / parse_ok / fallthrough / elapsed_ms /
    hook / exception."""


# ─────────────────────────────────────────────────────────────────────
# Diagnoser
# ─────────────────────────────────────────────────────────────────────


_DIAGNOSE_SYS_PROMPT = (
    "You are the failure-diagnosis component of a skill-bank crafter. "
    "Given one observed failure of a skill, classify it into a recovery "
    "strategy from a fixed enum and explain the root cause in <= 2 "
    "sentences. Respond with ONLY a JSON object on a single line."
)

_DIAGNOSE_SCHEMA_HINT = (
    "Respond with: {\"recommended_strategy\": <enum>, \"root_cause\": "
    "<string>, \"locus\": <string>, \"confidence\": <float in [0,1]>}.\n"
    "Allowed enum values for recommended_strategy: "
    + ", ".join(s.value for s in RecoveryStrategy)
)


class LLMDiagnoser:
    """Drop-in replacement for ``FailureDiagnoser._llm``.

    Usage::

        diagnoser = FailureDiagnoser()
        diagnoser.set_llm_diagnoser(LLMDiagnoser(LLMHookConfig(model="gpt-5.4")))
    """

    def __init__(self, config: Optional[LLMHookConfig] = None) -> None:
        self.config = config or LLMHookConfig(temperature=_DEFAULT_TEMP_DIAGNOSE)

    def __call__(self, trace: FailureTrace) -> Optional[FailureDiagnosis]:
        prompt = _render_diagnose_prompt(trace)
        payload = _call_json(
            prompt=prompt,
            cfg=self.config,
            hook_name="diagnoser",
            audit_sink=self.config.audit_sink,
        )
        if not isinstance(payload, dict):
            return None
        try:
            strategy = RecoveryStrategy(str(payload.get("recommended_strategy")))
        except ValueError:
            _emit_audit(self.config.audit_sink, {
                "kind": "crafter_llm",
                "hook": "diagnoser",
                "parse_failure": "invalid_recovery_strategy",
                "raw_value": payload.get("recommended_strategy"),
            })
            return None
        return FailureDiagnosis(
            failure_id=trace.failure_id,
            locus=str(payload.get("locus") or "unknown"),
            root_cause=str(payload.get("root_cause") or ""),
            recommended_strategy=strategy,
            confidence=_safe_float(payload.get("confidence"), default=0.5),
            notes="llm_diagnoser",
        )


# ─────────────────────────────────────────────────────────────────────
# Repairer
# ─────────────────────────────────────────────────────────────────────


_REPAIR_SYS_PROMPT = (
    "You are the skill-repair component of a multi-hop reasoning agent. "
    "Given one base skill and an observed failure pattern, produce an "
    "edited protocol + contract that fixes the failure. Keep the edit "
    "minimal: do not rewrite the whole protocol — insert / modify the "
    "smallest set of hops needed. Every hop dict MUST contain 'action' "
    "and 'payload' keys; 'action' must be one of "
    "GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE, VERIFY, RETRY. "
    "Respond with ONLY a JSON object on a single line."
)

_REPAIR_SCHEMA_HINT = (
    "Respond with: {\"patched_protocol\": [{\"action\": ..., \"payload\": "
    "{...}, \"notes\": ...}], \"patched_contract\": {\"preconditions\": "
    "[...], \"effects_add\": [...], \"effects_del\": [...], "
    "\"expected_evidence_roles\": [...], \"success_criteria\": [...], "
    "\"abort_criteria\": [...]}, \"recovery_strategy\": <enum>, "
    "\"rationale\": <string <= 200 chars>}.\n"
    "Allowed values for expected_evidence_roles: "
    + ", ".join(EVIDENCE_ROLES)
)


class LLMRepairer:
    """Drop-in for ``Repairer._llm``.

    Returns a fully-formed ``PatchProposal`` (the ``Repairer`` caller
    only inspects ``proposal is not None`` before persisting); when
    parsing fails this returns ``None`` and ``Repairer._rule_repair``
    takes over.
    """

    def __init__(self, config: Optional[LLMHookConfig] = None) -> None:
        self.config = config or LLMHookConfig(temperature=_DEFAULT_TEMP_REPAIR)

    def __call__(
        self,
        base: SkillRecord,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
    ) -> Optional[PatchProposal]:
        prompt = _render_repair_prompt(base, pattern, diagnosis)
        payload = _call_json(
            prompt=prompt,
            cfg=self.config,
            hook_name="repairer",
            audit_sink=self.config.audit_sink,
        )
        if not isinstance(payload, dict):
            return None

        protocol = _coerce_protocol(payload.get("patched_protocol"))
        contract = _coerce_contract(payload.get("patched_contract"))
        if not protocol:
            _emit_audit(self.config.audit_sink, {
                "kind": "crafter_llm",
                "hook": "repairer",
                "parse_failure": "empty_protocol",
            })
            return None

        try:
            strategy = RecoveryStrategy(str(payload.get("recovery_strategy")))
        except ValueError:
            strategy = diagnosis.recommended_strategy

        return PatchProposal(
            rationale=str(payload.get("rationale") or
                          f"llm_repair[{strategy.value}] for pattern={pattern.pattern_id}"),
            parent_skill_ids=[base.skill_id],
            seed_failure_ids=list(pattern.failure_ids),
            target_domains=list(base.feasible_domains),
            teacher_model=self.config.model,
            base_skill_id=base.skill_id,
            patched_protocol=protocol,
            patched_contract=contract,
            recovery_strategy=strategy.value,
            proposed_at=time.time(),
        )


# ─────────────────────────────────────────────────────────────────────
# Hypothesizer
# ─────────────────────────────────────────────────────────────────────


_HYPOTHESIZE_SYS_PROMPT = (
    "You are the novel-skill hypothesizer of a multi-hop reasoning "
    "agent. Given a recurring failure pattern and its diagnosis, "
    "design a small (2-6 hop) skill protocol that would have prevented "
    "the failure. Every hop dict MUST contain 'action' and 'payload' "
    "keys; 'action' must be one of "
    "GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE, VERIFY. "
    "Also produce a `strategic_description`: a 1-3 sentence "
    "natural-language paragraph that tells the actor WHEN to invoke "
    "this skill (the situation it solves) — this is the same shape "
    "the legacy Skill Bank Agent emits in `skill.strategic_description` "
    "and is what the actor's retrieval engine matches against. "
    "Respond with ONLY a JSON object on a single line."
)


# Fix-B: cap on the number of existing-concept lines we inject into
# the hypothesizer prompt.  Above this we cluster + take a
# representative per cluster — keeps prompt cost bounded even when
# the bank holds hundreds of skills.
_HYPOTHESIZE_EXISTING_CONCEPTS_CAP = 16


class LLMHypothesizer:
    """Drop-in for ``Hypothesizer._llm``."""

    def __init__(self, config: Optional[LLMHookConfig] = None) -> None:
        self.config = config or LLMHookConfig(temperature=_DEFAULT_TEMP_HYPOTHESIZE)

    def __call__(
        self,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
        *,
        existing_concepts: Optional[List[str]] = None,
    ) -> Optional[HypothesisProposal]:
        prompt = _render_hypothesize_prompt(
            pattern, diagnosis,
            existing_concepts=existing_concepts,
        )
        payload = _call_json(
            prompt=prompt,
            cfg=self.config,
            hook_name="hypothesizer",
            audit_sink=self.config.audit_sink,
        )
        if not isinstance(payload, dict):
            return None

        protocol = _coerce_protocol(payload.get("novel_protocol"))
        contract = _coerce_contract(payload.get("contract"))
        if not protocol:
            _emit_audit(self.config.audit_sink, {
                "kind": "crafter_llm",
                "hook": "hypothesizer",
                "parse_failure": "empty_protocol",
            })
            return None

        # Default target domains: union of pattern + diagnosis hint.
        target_domains = sorted(set(pattern.domains)) or []

        # Fix-D2: surface a natural-language `strategic_description`
        # so the writeback projector can populate the legacy bank's
        # `skill.strategic_description` field — the one the actor's
        # retrieval engine (`skill_agents/query.py`) embeds into its
        # similarity index. Falls back to `rationale` when the LLM
        # forgot the field, since `rationale` already has the right
        # shape (1-2 sentence summary).
        strategic_description = str(
            payload.get("strategic_description")
            or payload.get("description")
            or payload.get("rationale")
            or ""
        ).strip()
        if contract is not None and not contract.description:
            contract.description = strategic_description

        return HypothesisProposal(
            name=str(payload.get("name") or
                     f"hyp_for_{pattern.failure_class.lower() or 'unknown'}"),
            rationale=str(payload.get("rationale") or
                          f"llm_hypothesis for pattern={pattern.pattern_id}: "
                          f"{diagnosis.root_cause}"),
            parent_skill_ids=[pattern.skill_id] if pattern.skill_id else [],
            seed_failure_ids=list(pattern.failure_ids),
            target_domains=target_domains,
            teacher_model=self.config.model,
            novel_protocol=protocol,
            contract=contract,
            source_failure_pattern_ids=[pattern.pattern_id],
            proposed_at=time.time(),
            strategic_description=strategic_description,
        )


# ─────────────────────────────────────────────────────────────────────
# Convenience: install all three hooks on a SkillCrafterService
# ─────────────────────────────────────────────────────────────────────


def install_llm_hooks(
    crafter,                                                       # SkillCrafterService
    *,
    model: str = BACKBONE_TEACHER_MODEL,
    audit_sink: Optional[AuditSink] = None,
    enable_diagnoser: bool = False,
    enable_repairer: bool = True,
    enable_hypothesizer: bool = True,
) -> Dict[str, Any]:
    """Install the LLM-backed hooks on ``crafter`` and stamp the
    teacher model. Returns a small dict of which hooks were enabled
    (useful for logging).

    Defaults match the README's recommended Step-0 + Step-1 sequence:
    enable Repairer first (it has the smallest fail surface), then
    Hypothesizer; leave Diagnoser on the rule path because the rule
    table is already specific and the LLM offers little lift on a
    7-way classification.

    The crafter's ``teacher_model`` is updated so emitted proposals
    carry the correct provenance stamp regardless of which hooks are
    live.
    """
    common_cfg = LLMHookConfig(model=model, audit_sink=audit_sink)
    enabled: Dict[str, bool] = {}
    if enable_repairer:
        crafter._repairer.set_llm_repairer(LLMRepairer(common_cfg))    # type: ignore[attr-defined]
        enabled["repairer"] = True
    if enable_hypothesizer:
        crafter._hypothesizer.set_llm_proposer(LLMHypothesizer(common_cfg))  # type: ignore[attr-defined]
        enabled["hypothesizer"] = True
    if enable_diagnoser:
        crafter._diagnoser.set_llm_diagnoser(LLMDiagnoser(common_cfg))  # type: ignore[attr-defined]
        enabled["diagnoser"] = True
    crafter.set_teacher_model(model)
    return {"model": model, "hooks": enabled}


# ─────────────────────────────────────────────────────────────────────
# Internals — single-pass JSON LLM call + parsers
# ─────────────────────────────────────────────────────────────────────


def _call_json(
    *,
    prompt: str,
    cfg: LLMHookConfig,
    hook_name: str,
    audit_sink: Optional[AuditSink],
) -> Optional[Any]:
    """Single-pass LLM call returning a parsed JSON value (or ``None``).

    Catches ALL exceptions; never raises. Audit events are emitted
    when an audit_sink is provided so the dormant-vs-live status of
    the hook is observable.
    """
    # Lazy import: ``API_func`` pulls in optional SDKs on import; we
    # don't want to require ``anthropic`` etc. just because someone
    # imported ``crafter._llm_runtime``.
    try:
        from API_func import ask_model
    except Exception as exc:                                           # noqa: BLE001
        _emit_audit(audit_sink, {
            "kind": "crafter_llm",
            "hook": hook_name,
            "exception": f"import_API_func: {exc}",
        })
        return None

    t0 = time.time()
    raw: Optional[str] = None
    try:
        raw = ask_model(
            prompt,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            enable_thinking=cfg.enable_thinking,
        )
    except Exception as exc:                                           # noqa: BLE001
        _emit_audit(audit_sink, {
            "kind": "crafter_llm",
            "hook": hook_name,
            "model": cfg.model,
            "exception": str(exc),
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
        return None

    if not isinstance(raw, str) or not raw.strip():
        _emit_audit(audit_sink, {
            "kind": "crafter_llm",
            "hook": hook_name,
            "model": cfg.model,
            "parse_failure": "empty_response",
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
        return None

    if raw.startswith("Error"):
        _emit_audit(audit_sink, {
            "kind": "crafter_llm",
            "hook": hook_name,
            "model": cfg.model,
            "exception": raw[:200],
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
        return None

    parsed = _parse_json_loose(raw)
    _emit_audit(audit_sink, {
        "kind": "crafter_llm",
        "hook": hook_name,
        "model": cfg.model,
        "parse_ok": parsed is not None,
        "raw_len": len(raw),
        "elapsed_ms": int((time.time() - t0) * 1000),
    })
    return parsed


_FENCED_RX = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _parse_json_loose(text: str) -> Optional[Any]:
    """Parse JSON tolerant of code-fence wrapping + leading prose.

    Tries (in order):
      1. raw text
      2. ```json ... ``` fenced block
      3. first balanced { ... } substring
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCED_RX.search(text)
    if m:
        try:
            return json.loads(m.group("body"))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        return None
    return None


def _coerce_protocol(raw: Any) -> List[Dict[str, Any]]:
    """Coerce an arbitrary LLM blob into the ``protocol`` list shape.

    Defensive — drops items that don't look like ``{"action": str, ...}``.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if not isinstance(action, str) or not action.strip():
            continue
        out.append({
            "action": action.upper().strip(),
            "payload": dict(item.get("payload") or {}),
            "notes": str(item.get("notes") or ""),
        })
    return out


def _coerce_contract(raw: Any) -> Optional[SkillContract]:
    """Coerce an LLM blob into ``SkillContract``; returns ``None`` on
    bad input so the caller can fall back to the rule path."""
    if not isinstance(raw, dict):
        return None
    try:
        roles = [r for r in (raw.get("expected_evidence_roles") or []) if r in EVIDENCE_ROLES]
        return SkillContract(
            preconditions=_coerce_str_list(raw.get("preconditions")),
            effects_add=_coerce_str_list(raw.get("effects_add")),
            effects_del=_coerce_str_list(raw.get("effects_del")),
            belief_progress=_coerce_str_list(raw.get("belief_progress")),
            grounding_progress=_coerce_str_list(raw.get("grounding_progress")),
            expected_evidence_roles=roles,
            success_criteria=_coerce_str_list(raw.get("success_criteria")),
            abort_criteria=_coerce_str_list(raw.get("abort_criteria")),
            description=str(raw.get("description") or "").strip(),
        )
    except Exception:                                                  # noqa: BLE001
        return None


def _coerce_str_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]


def _safe_float(raw: Any, *, default: float) -> float:
    try:
        v = float(raw)
        if v != v:    # NaN
            return default
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def _emit_audit(sink: Optional[AuditSink], event: Dict[str, Any]) -> None:
    if sink is None:
        return
    try:
        sink(event)
    except Exception:                                                  # noqa: BLE001
        # Audit failures must never propagate.
        pass


# ─────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────


def _render_diagnose_prompt(trace: FailureTrace) -> str:
    return "\n".join([
        _DIAGNOSE_SYS_PROMPT,
        "",
        _DIAGNOSE_SCHEMA_HINT,
        "",
        "Failure trace:",
        json.dumps(trace.to_json(), indent=2, ensure_ascii=False, sort_keys=True),
    ])


def _render_repair_prompt(
    base: SkillRecord,
    pattern: FailurePattern,
    diagnosis: FailureDiagnosis,
) -> str:
    return "\n".join([
        _REPAIR_SYS_PROMPT,
        "",
        _REPAIR_SCHEMA_HINT,
        "",
        "Recommended recovery strategy: " + diagnosis.recommended_strategy.value,
        "Diagnosis root cause: " + (diagnosis.root_cause or ""),
        "",
        "Base skill:",
        json.dumps({
            "skill_id": base.skill_id,
            "name": base.name,
            "feasible_domains": list(base.feasible_domains),
            "protocol": base.protocol,
            "contract": base.contract.to_json(),
        }, indent=2, ensure_ascii=False, sort_keys=True),
        "",
        "Failure pattern:",
        json.dumps({
            "pattern_id": pattern.pattern_id,
            "failure_class": pattern.failure_class,
            "count": pattern.count,
            "domains": list(pattern.domains),
            "sample_abort_reasons": list(pattern.sample_abort_reasons),
            "failed_step_index": pattern.failed_step_index,
        }, indent=2, ensure_ascii=False, sort_keys=True),
    ])


def _render_hypothesize_prompt(
    pattern: FailurePattern,
    diagnosis: FailureDiagnosis,
    *,
    existing_concepts: Optional[List[str]] = None,
) -> str:
    """Render the Hypothesizer prompt body.

    ``existing_concepts`` (Fix-B) is an optional list of short
    descriptors — one per concept already represented in the
    crafter's bank — that the LLM is instructed NOT to paraphrase.
    Each string is the bank's per-skill ``"<name>: <short rationale>"``;
    callers in ``crafter.service`` cluster + cap the list to
    ``_HYPOTHESIZE_EXISTING_CONCEPTS_CAP`` lines so prompt cost
    stays bounded.

    The block is intentionally placed *before* the failure-pattern
    payload — putting it last would let the LLM treat it as
    "additional context" and ignore it; putting it up front makes
    it a constraint on the response.  See the v3 attribution summary
    §"Fix-B: prompt counter-collapse" for the empirical receipt.
    """
    lines: List[str] = [
        _HYPOTHESIZE_SYS_PROMPT,
        "",
        "Respond with: {\"name\": <string, short verb-phrase like "
        "\"Gate evidence before commit\">, \"strategic_description\": "
        "<string, 1-3 sentence paragraph telling the actor WHEN to "
        "invoke this skill — same shape as the legacy Skill Bank "
        "Agent's `skill.strategic_description`>, \"novel_protocol\": "
        "[{\"action\": ..., \"payload\": {...}, \"notes\": ...}], "
        "\"contract\": {\"preconditions\": [...], \"effects_add\": [...], "
        "\"effects_del\": [...], \"expected_evidence_roles\": [...], "
        "\"success_criteria\": [...], \"abort_criteria\": [...], "
        "\"description\": <string, optional natural-language paragraph; "
        "if omitted strategic_description is used as fallback>}, "
        "\"rationale\": <string <= 200 chars, internal Crafter audit text>}.",
        "Allowed values for expected_evidence_roles: " + ", ".join(EVIDENCE_ROLES),
        "",
    ]

    if existing_concepts:
        # Defensive: drop empties / non-strings, dedup case-insensitively
        # so two callers passing slightly-different surface forms of
        # the same concept don't double-bill the prompt.
        seen: set = set()
        clean: List[str] = []
        for c in existing_concepts:
            if not isinstance(c, str):
                continue
            t = c.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(t)
        clean = clean[:_HYPOTHESIZE_EXISTING_CONCEPTS_CAP]
        if clean:
            lines.extend([
                "EXISTING SKILL BANK already covers these concepts. "
                "Your `name` MUST NOT be a paraphrase of any line "
                "below, and your `novel_protocol` MUST encode a "
                "DIFFERENT failure-mitigation strategy (different "
                "hop sequence, different evidence roles, or a "
                "different precondition) from every line below.",
                *(f"- {c}" for c in clean),
                "",
            ])

    lines.extend([
        "Recommended recovery strategy: " + diagnosis.recommended_strategy.value,
        "Diagnosis root cause: " + (diagnosis.root_cause or ""),
        "",
        "Failure pattern:",
        json.dumps({
            "pattern_id": pattern.pattern_id,
            "failure_class": pattern.failure_class,
            "count": pattern.count,
            "domains": list(pattern.domains),
            "sample_abort_reasons": list(pattern.sample_abort_reasons),
        }, indent=2, ensure_ascii=False, sort_keys=True),
    ])
    return "\n".join(lines)


__all__ = [
    "AuditSink",
    "LLMDiagnoser",
    "LLMHookConfig",
    "LLMHypothesizer",
    "LLMRepairer",
    "install_llm_hooks",
]
