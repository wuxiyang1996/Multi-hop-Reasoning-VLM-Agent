"""Forward bind one :class:`SharedAbstractSkill` to a target task.

Pipeline (the user-described "harness/crafter to convert into the new
domain via LLM + validation"):

  1. Pick a SharedAbstractSkill ``A`` and a target task ``T``.
  2. Pull every existing binding of ``A`` (other tasks where ``A``
     is bound) from the SharedAbstractBank — these become FEW-SHOT
     EXAMPLES for the LLM converter.
  3. Pull a small slice of ``T``'s native bindings (mining-source
     skills already in PerTaskBank[T]) so the LLM sees what
     target-task vocabulary looks like.
  4. Call GPT-5.4 with (abstract template_steps + abstract
     protocol_steps + cross-task example bindings + target-task
     vocab examples) → propose a candidate ``BoundConcreteSkill``
     for ``T``.
  5. (Optional) hand the candidate to ``FewShotAdapter`` for
     execution validation.  When ``--harness validate`` is set we
     try to pull a small demo set for ``T`` and run the skill;
     otherwise we write the binding with ``binding_status="PENDING"``
     so an offline validation pass can sweep them later.
  6. Write the BoundConcreteSkill into PerTaskBank[T] and append a
     lineage entry of ``discovered_via="binding"`` to ``A``.

The script is resilient: any of (LLM call, harness, target task
demo loader) failing degrades to writing a ``PENDING`` /
``REJECTED`` binding with a diagnostic note, so the bank is always
left in a consistent state.

Single-skill usage::

    python scripts/bind_abstract_to_task.py \\
        --abstract-id "INSPECT/SETUP" \\
        --target-task candy_crush

Batch usage (all 42 strong cross-task candidates → candy_crush)::

    python scripts/bind_abstract_to_task.py \\
        --target-task candy_crush \\
        --batch-strong-candidates \\
        --max-binds 8

This is a *thin* driver: most of the legwork (prompt design,
template index, FewShotAdapter wiring) is reused from
``scripts/lift_skill_templates_gpt54.py`` and
``labeling_supplement/_phase4_transfer_cycle.py`` already.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
WORK = REPO.parent
for p in [str(WORK), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Pull in the OpenRouter client + JSON helpers from the lift script.
from scripts.lift_skill_templates_gpt54 import (                       # noqa: E402
    _extract_json, _get_openai_client, DEFAULT_MODEL,
)
from skill_bank.shared_abstract_bank import (                          # noqa: E402
    BoundConcreteSkill, LineageEntry, ProtocolStep, SharedAbstractSkill,
    SubEpisodeRef, TwoLayerSkillStore, hash_contract,
)

logger = logging.getLogger("bind_abstract_to_task")

DEFAULT_BANK_ROOT = REPO / "shared_skill_bank" / "_latest"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at re-grounding modality-agnostic skill "
    "skeletons into specific game / web / video tasks.  Your "
    "PRIMARY output is a STEP-BY-STEP PROTOCOL — the sequence of "
    "actions the agent will follow OVER TIME at runtime.  Each "
    "protocol step has a typed action verb (op), parameterised "
    "payload with semantic slot types, typed preconditions, and "
    "typed add/del effects.  The contract (preconditions / "
    "postconditions / example_predicates) is a static summary "
    "DERIVED from the protocol — not the source of truth.  "
    "Output strict JSON only."
)


def _render_examples(
    abstract: SharedAbstractSkill,
    *,
    bank: TwoLayerSkillStore,
    exclude_task: str,
    max_examples: int = 3,
) -> str:
    """Render up to ``max_examples`` existing bindings of ``abstract``
    in OTHER tasks for the LLM as few-shot examples.  We FAVOUR
    bindings that have a non-empty protocol — those teach the LLM
    what shape we want; static-contract-only ones are kept as a
    fallback only when no rich examples exist."""
    rich, sparse = [], []
    for L in abstract.lineage:
        if L.task == exclude_task:
            continue
        if L.discovered_via not in ("mining", "binding"):
            continue
        binding = bank.per_task(L.task).by_concrete_id(L.concrete_skill_id)
        if binding is None:
            continue
        if binding.protocol:
            rich.append((L, binding))
        elif binding.contract:
            sparse.append((L, binding))
    chosen = (rich + sparse)[:max_examples]
    if not chosen:
        return "(no existing bindings in other tasks; this is a cold-start binding.)"
    out: List[str] = []
    for L, b in chosen:
        c = b.contract or {}
        out.append(f"-- Existing binding in task '{L.task}' ({L.cohort}) --")
        out.append(f"  name          : {b.name}")
        if b.protocol:
            out.append(f"  protocol      :")
            for i, s in enumerate(b.protocol[:5]):
                out.append(
                    f"     [{i+1}] op={s.op:<8} "
                    f"payload={dict(s.payload)} "
                    f"slot_types={dict(s.slot_types)}"
                )
                if s.preconditions:
                    out.append(f"          pre  = {[p.get('type','?') for p in s.preconditions[:3]]}")
                if s.effects_add:
                    out.append(f"          eff+ = {[e.get('type','?') for e in s.effects_add[:3]]}")
                if s.notes:
                    out.append(f'          notes= "{s.notes[:80]}"')
        out.append(f"  preconditions : {c.get('preconditions') or []}")
        out.append(f"  postconditions: {c.get('postconditions') or []}")
        out.append("")
    return "\n".join(out).rstrip()


def _render_target_vocab(
    *, bank: TwoLayerSkillStore, target_task: str, max_examples: int = 4,
) -> str:
    """Render a few of the target task's NATIVE skills so the LLM
    sees what target-side vocabulary looks like."""
    natives = []
    for b in bank.per_task(target_task).records:
        if b.binding_source == "mining" and b.contract:
            natives.append(b)
        if len(natives) >= max_examples:
            break
    if not natives:
        return "(no native skills mined yet on this target task.)"
    lines: List[str] = []
    for b in natives:
        c = b.contract
        lines += [
            f"-- Native skill '{b.concrete_skill_id}' ({b.name}) --",
            f"  preconditions   : {c.get('preconditions') or []}",
            f"  example_preds   : {c.get('example_predicates') or []}",
            "",
        ]
    return "\n".join(lines).rstrip()


def _render_abstract_protocol(abstract: SharedAbstractSkill) -> str:
    """Render the abstract skeleton's protocol in a form the LLM can
    pattern-match on.  Slot types and effect types stay verbatim
    because they're already modality-agnostic; payload values are
    slot references (``${target}``)."""
    if not abstract.protocol_steps:
        return "  (no protocol skeleton recorded — derive from template_steps)"
    out: List[str] = []
    for i, s in enumerate(abstract.protocol_steps[:8]):
        out.append(
            f"  [{i+1}] op={s.op:<8} "
            f"payload={dict(s.payload)}  "
            f"slot_types={dict(s.slot_types)}"
        )
        if s.preconditions:
            out.append(f"        pre  = {[p.get('type','?') for p in s.preconditions[:3]]}")
        if s.effects_add:
            out.append(f"        eff+ = {[e.get('type','?') for e in s.effects_add[:3]]}")
        if s.effects_del:
            out.append(f"        eff- = {[e.get('type','?') for e in s.effects_del[:3]]}")
        if s.evidence_role:
            out.append(f"        evidence_role = {s.evidence_role}")
    return "\n".join(out)


def build_bind_prompt(
    abstract: SharedAbstractSkill,
    *, target_task: str,
    bank: TwoLayerSkillStore,
) -> str:
    sig = abstract.template_signature
    template_lines = [f"  [{i+1}] {s.op:<9} {s.predicate}"
                      for i, s in enumerate(abstract.template_steps)]
    proto_skel = _render_abstract_protocol(abstract)
    examples = _render_examples(abstract, bank=bank, exclude_task=target_task)
    target_vocab = _render_target_vocab(bank=bank, target_task=target_task)

    return "\n".join([
        f"ABSTRACT_SKILL_ID : {abstract.abstract_skill_id}",
        f"NAME              : {abstract.name}",
        f"TEMPLATE_SIGNATURE: {sig}",
        "",
        "TEMPLATE_STEPS (Layer-C, 2-5 modality-agnostic steps):",
        *(template_lines or ["  (none)"]),
        "",
        "ABSTRACT PROTOCOL SKELETON (slot types + effect types, "
        "modality-agnostic — KEEP THIS STRUCTURE):",
        proto_skel,
        "",
        "EXISTING BINDINGS IN OTHER TASKS (for analogy / shape):",
        examples,
        "",
        f"TARGET TASK: {target_task}",
        "TARGET-TASK NATIVE VOCABULARY (what predicate names look like here):",
        target_vocab,
        "",
        "TASK: Produce an EXECUTABLE binding for the TARGET task.  The",
        "primary output is the PROTOCOL — the step-by-step plan an agent",
        "will follow over time.  The contract is a derived static summary.",
        "",
        "Output STRICT JSON:",
        "{",
        '  "name": "<short human-readable name>",',
        '  "protocol": [',
        '    {',
        '      "op": "<EXEC | MOVE | INSPECT | COMPARE | KEEP | PLACE | APPROACH | SELECT | VERIFY | TRACK | DROP>",',
        '      "payload": {"<slot>": "${<slot>}", ...},   // slot REFERENCES, not values',
        '      "slot_types": {"<slot>": "<tracked_entity|container_entity|navigable_region|selectable_entity|effect_predicate|enum|any>"},',
        '      "preconditions": [{"type": "<entity_visible|attribute_value|player_state|...>", "args": {<target-task-vocab>}}],',
        '      "effects_add":   [{"type": "<entity_appeared|entity_disappeared|attribute_changed|cumulative_reward_increased>", "args": {<target-task-vocab>}}],',
        '      "effects_del":   [...],',
        '      "evidence_role": "<COMMIT|VERIFY|OBSERVATION|RECOVER>",',
        '      "notes": "<≤16 word description in TARGET task vocabulary>"',
        '    },',
        '    ...   // 2-5 steps, MIRRORING the abstract protocol structure',
        '  ],',
        '  "contract": {                                  // DERIVED summary',
        '    "preconditions":      ["..."],   // ≤6 facts in target vocab',
        '    "postconditions":     ["..."],   // ≤6 observable outcomes',
        '    "example_predicates": ["..."],   // 3-6 snake_case predicates',
        '    "eff_add":            ["..."],',
        '    "eff_del":            ["..."]',
        '  },',
        '  "rationale": "<≤80 word: why this re-grounding is faithful>"',
        "}",
        "",
        "RULES:",
        " - Each protocol step MUST keep the abstract step\\'s op semantics: ",
        "   if the abstract uses MOVE with slot_types {target: tracked_entity, ",
        "   direction: enum}, the binding MUST also be a MOVE with those ",
        "   slot types — only the args' VALUES change to target-task vocab.",
        " - Effect types come from a fixed enum.  Use entity_appeared / ",
        "   entity_disappeared / attribute_changed / cumulative_reward_increased",
        "   etc. — DO NOT invent new effect type names.",
        " - Predicates in `args` and notes use the TARGET task\\'s vocab.",
        "   E.g. for candy_crush say 'adjacent_swap' not 'enemy_dispatched'.",
        " - Never put button glyphs / DOM xpaths in `notes` or `args`.",
        " - The contract is a SUMMARY — it must be derivable from the ",
        "   protocol\\'s preconditions and effects.",
    ])


# ---------------------------------------------------------------------------
def llm_convert(
    abstract: SharedAbstractSkill,
    *, target_task: str,
    bank: TwoLayerSkillStore,
    model: str = DEFAULT_MODEL,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """One LLM call.  Returns ``(parsed_dict, raw_text)``."""
    client = _get_openai_client()
    prompt = build_bind_prompt(abstract, target_task=target_task, bank=bank)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        # Rich protocol output (op + payload + slot_types + typed pre/eff
        # + notes per step) is ~250 tokens/step; 4-5 steps + contract +
        # rationale needs ~1.6k.
        max_tokens=1800,
    )
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    return _extract_json(text), text


def _coerce_binding(
    parsed: Dict[str, Any], *,
    abstract: SharedAbstractSkill, target_task: str,
) -> Optional[BoundConcreteSkill]:
    if not isinstance(parsed, dict):
        return None

    # ── Protocol — PRIMARY ──────────────────────────────────────
    proto_raw = parsed.get("protocol") or []
    if not isinstance(proto_raw, list):
        return None
    protocol_steps: List[ProtocolStep] = []
    for step in proto_raw:
        if not isinstance(step, dict):
            continue
        # Coerce typed list fields back into list[dict]; the LLM
        # sometimes returns list[str] for preconditions / effects —
        # wrap those as {"type": ..., "args": {}}.
        def _norm_typed(v) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        out.append({
                            "type": str(x.get("type", "")),
                            "args": dict(x.get("args") or {}),
                        })
                    elif isinstance(x, str) and x.strip():
                        out.append({"type": x.strip(), "args": {}})
            return out

        protocol_steps.append(ProtocolStep(
            op=str(step.get("op", "")).upper(),
            payload=dict(step.get("payload") or {}),
            slot_types=dict(step.get("slot_types") or {}),
            preconditions=_norm_typed(step.get("preconditions")),
            effects_add=_norm_typed(step.get("effects_add") or step.get("eff_add")),
            effects_del=_norm_typed(step.get("effects_del") or step.get("eff_del")),
            evidence_role=str(step.get("evidence_role", "")).upper(),
            notes=str(step.get("notes", ""))[:200],
        ))
    if not protocol_steps:
        return None

    # ── Contract — DERIVED summary ──────────────────────────────
    contract_raw = parsed.get("contract") or {}
    if not isinstance(contract_raw, dict):
        contract_raw = {}
    contract: Dict[str, Any] = {}
    for k in ("preconditions", "postconditions", "example_predicates",
              "eff_add", "eff_del"):
        v = contract_raw.get(k) or []
        if not isinstance(v, list):
            v = [str(v)]
        contract[k] = [str(x).strip() for x in v if str(x).strip()][:8]

    # Belt-and-braces: if the LLM gave us a protocol but skipped the
    # contract, derive the static summary from protocol effect types.
    if not contract["eff_add"]:
        contract["eff_add"] = sorted({e["type"] for s in protocol_steps
                                       for e in s.effects_add if e.get("type")})[:6]
    if not contract["eff_del"]:
        contract["eff_del"] = sorted({e["type"] for s in protocol_steps
                                       for e in s.effects_del if e.get("type")})[:6]

    name = str(parsed.get("name") or abstract.name or abstract.abstract_skill_id)
    return BoundConcreteSkill(
        concrete_skill_id=abstract.abstract_skill_id,
        task=target_task,
        abstract_skill_id=abstract.abstract_skill_id,
        name=name,
        protocol=protocol_steps,
        sub_episodes=[],                 # forward bind has no rollouts yet
        contract=contract,
        binding_status="PENDING",        # validation hasn't fired yet
        binding_source="forward_convert",
        decorations={"rationale": str(parsed.get("rationale", ""))[:400]},
    )


# ---------------------------------------------------------------------------
def harness_validate(
    binding: BoundConcreteSkill, *,
    target_task: str, k: int = 2, max_demos: int = 2,
    pass_rate_min: float = 0.5,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Run the binding through ``FewShotAdapter`` if a target-task
    demo is available.  Returns ``(passed | None, diagnostics)``.

    ``passed = None`` means we couldn't even attempt validation
    (e.g. no demos, dispatcher missing); the caller should leave
    ``binding_status`` at ``PENDING``.
    """
    diag: Dict[str, Any] = {"validator": "FewShotAdapter"}
    try:
        from labeling_supplement._phase4_target_dispatch import (
            build_target,
        )
        from labeling_supplement._phase4_transfer_cycle import _run_transfer
        from common.enums import SkillStatus
        from data_structure.extensions.skill_record import SkillRecord
    except Exception as exc:
        diag["error"] = f"import_failed: {exc!r}"
        return None, diag

    diag["error"] = "skipped_for_now: harness validation pipeline not wired in this driver"
    return None, diag


# ---------------------------------------------------------------------------
def bind_one(
    *,
    abstract: SharedAbstractSkill,
    target_task: str,
    bank: TwoLayerSkillStore,
    model: str = DEFAULT_MODEL,
    do_harness_validate: bool = False,
) -> Dict[str, Any]:
    """Forward-bind ``abstract`` to ``target_task``.  Writes a
    BoundConcreteSkill into PerTaskBank[target_task] and appends a
    lineage entry on the abstract.  Returns a small report dict."""
    t0 = time.monotonic()
    parsed, raw = llm_convert(abstract, target_task=target_task,
                               bank=bank, model=model)
    if parsed is None:
        return {
            "abstract": abstract.abstract_skill_id,
            "target": target_task,
            "ok": False, "reason": "llm_no_json",
            "raw_excerpt": raw[:200],
        }
    binding = _coerce_binding(parsed, abstract=abstract, target_task=target_task)
    if binding is None:
        return {
            "abstract": abstract.abstract_skill_id,
            "target": target_task,
            "ok": False, "reason": "coerce_failed",
            "raw_excerpt": raw[:200],
        }

    validated, vdiag = (None, {}) if not do_harness_validate else \
        harness_validate(binding, target_task=target_task)
    if validated is True:
        binding.binding_status = "VALIDATED"
        binding.last_validation_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif validated is False:
        binding.binding_status = "REJECTED"
        binding.last_validation_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Persist via the high-level forward-bind API.
    chash = hash_contract(binding.contract)
    lineage = LineageEntry(
        task=target_task,
        concrete_skill_id=binding.concrete_skill_id,
        raw_skill_id=binding.concrete_skill_id,
        cohort="",  # caller can fill this in
        discovered_via="binding",
        is_native=False,
        contract_hash=chash,
        decorations={"binding_source": "forward_convert",
                     "model": model},
    )
    verdicts = bank.insert_validated_binding(
        concrete=binding,
        abstract_skill_id=abstract.abstract_skill_id,
        template_signature=abstract.template_signature,
        lineage=lineage,
    )
    return {
        "abstract": abstract.abstract_skill_id,
        "target":   target_task,
        "ok":       True,
        "binding_status": binding.binding_status,
        "validator_diag": vdiag,
        "verdicts": verdicts,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "rationale": binding.decorations.get("rationale", ""),
        "n_protocol_steps":  len(binding.protocol),
        "protocol_op_chain": [s.op for s in binding.protocol],
        "n_total_effects":   sum(len(s.effects_add) + len(s.effects_del)
                                  for s in binding.protocol),
        "n_preconditions":   len(binding.contract.get("preconditions", [])),
        "n_eff_add_summary": len(binding.contract.get("eff_add",     [])),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", default=str(DEFAULT_BANK_ROOT))
    ap.add_argument("--target-task", required=True)
    ap.add_argument("--abstract-id",
                    help="Single abstract skill_id to bind.  If omitted, must use --batch-strong-candidates.")
    ap.add_argument("--abstract-signature", default=None,
                    help="Disambiguate abstract IDs that have multiple signatures.")
    ap.add_argument("--batch-strong-candidates", action="store_true",
                    help="Bind every abstract with ≥2 cross-task lineages and a non-NO_TEMPLATE signature.")
    ap.add_argument("--max-binds", type=int, default=4,
                    help="Cap the number of bindings issued in batch mode.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--harness-validate", action="store_true",
                    help="Hand candidate to FewShotAdapter (currently a stub).")
    ap.add_argument("--out-report", default=None,
                    help="Optional JSON report path.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    bank = TwoLayerSkillStore(Path(args.bank_root))
    bank.abstract.load()
    logger.info("bank: %d abstracts, tasks=%s",
                bank.abstract.size, bank.list_tasks())

    # Pick targets.
    targets: List[SharedAbstractSkill] = []
    if args.abstract_id:
        cands = bank.abstract.by_abstract_id(args.abstract_id)
        if args.abstract_signature:
            cands = [c for c in cands if c.template_signature == args.abstract_signature]
        if not cands:
            logger.error("no abstract %r in bank", args.abstract_id)
            return 2
        targets = [cands[0]]
    elif args.batch_strong_candidates:
        unbound = bank.abstract.candidates_for_target_task(args.target_task)
        # Filter: has signature, ≥2 cross-task lineages, ≥1 binding lineage with a
        # contract somewhere (so we can render examples).
        strong = []
        for r in unbound:
            if r.template_signature == "NO_TEMPLATE":
                continue
            n_tasks = r.n_bound_tasks
            if n_tasks < 2:
                continue
            # need at least one MINING lineage so render_examples returns content
            if not any(L.discovered_via == "mining" for L in r.lineage):
                continue
            strong.append(r)
        strong.sort(key=lambda r: -r.n_bound_tasks)
        targets = strong[:args.max_binds]
    else:
        logger.error("either --abstract-id or --batch-strong-candidates required")
        return 2

    logger.info("binding %d abstract(s) -> %s", len(targets), args.target_task)

    reports: List[Dict[str, Any]] = []
    n_ok = 0
    for abs_rec in targets:
        logger.info("  bind  %-25s  sig=%s",
                    abs_rec.abstract_skill_id, abs_rec.template_signature)
        try:
            r = bind_one(
                abstract=abs_rec,
                target_task=args.target_task,
                bank=bank, model=args.model,
                do_harness_validate=args.harness_validate,
            )
        except Exception as exc:                                       # noqa: BLE001
            r = {"abstract": abs_rec.abstract_skill_id,
                 "target": args.target_task, "ok": False,
                 "reason": f"exception:{exc!r}"}
        reports.append(r)
        if r.get("ok"):
            n_ok += 1
        logger.info("    -> %s", json.dumps({k: v for k, v in r.items()
                                              if k != "rationale"}))

    if args.out_report:
        Path(args.out_report).write_text(
            json.dumps(reports, indent=2, ensure_ascii=False),
        )
        logger.info("report: %s", args.out_report)

    logger.info("done: %d/%d ok", n_ok, len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
