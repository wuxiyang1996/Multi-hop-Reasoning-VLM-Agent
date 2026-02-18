"""
Skill query engine — rich retrieval API over the Skill Bank.

Provides keyword matching, effect-based retrieval, and detailed skill
inspection.  Designed as the query backend for ``decision_agents``
(``query_skill`` tool) and the pipeline's ``SkillBankAgent.query_skill``.

Usage::

    from skill_agents.query import SkillQueryEngine
    from skill_agents.skill_bank.bank import SkillBankMVP

    bank = SkillBankMVP("bank.jsonl"); bank.load()
    engine = SkillQueryEngine(bank)

    results = engine.query("navigate to pot and place onion", top_k=3)
    details = engine.get_detail("nav_to_pot")
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.stage3_mvp.schemas import SkillEffectsContract, VerificationReport


def _tokenize(text: str) -> Set[str]:
    """Lowercase tokenisation for keyword overlap scoring."""
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _effect_set_for_contract(c: SkillEffectsContract) -> Set[str]:
    return (c.eff_add or set()) | (c.eff_del or set()) | (c.eff_event or set())


def _contract_summary(c: SkillEffectsContract) -> Dict[str, Any]:
    return {
        "skill_id": c.skill_id,
        "version": c.version,
        "eff_add": sorted(c.eff_add) if c.eff_add else [],
        "eff_del": sorted(c.eff_del) if c.eff_del else [],
        "eff_event": sorted(c.eff_event) if c.eff_event else [],
        "n_literals": c.total_literals,
        "n_instances": c.n_instances,
    }


class SkillQueryEngine:
    """Query interface over a SkillBankMVP.

    Supports:
    - **keyword query**: match query tokens against skill ID and effect literals.
    - **effect query**: find skills whose effects overlap with desired state changes.
    - **list / detail**: enumerate skills or inspect one in depth.
    """

    def __init__(self, bank: SkillBankMVP) -> None:
        self._bank = bank
        self._build_index()

    def _build_index(self) -> None:
        """Pre-tokenise skill IDs and effects for fast matching."""
        self._id_tokens: Dict[str, Set[str]] = {}
        self._effect_tokens: Dict[str, Set[str]] = {}
        self._effect_sets: Dict[str, Set[str]] = {}

        for sid in self._bank.skill_ids:
            self._id_tokens[sid] = _tokenize(sid)
            c = self._bank.get_contract(sid)
            if c is not None:
                eff = _effect_set_for_contract(c)
                self._effect_sets[sid] = eff
                self._effect_tokens[sid] = set()
                for lit in eff:
                    self._effect_tokens[sid] |= _tokenize(lit)
            else:
                self._effect_sets[sid] = set()
                self._effect_tokens[sid] = set()

    # ── Keyword query ────────────────────────────────────────────────

    def query(self, key: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Score every skill against a natural-language query key.

        Scoring: weighted combination of skill-ID token overlap and
        effect-literal token overlap.

        Returns up to *top_k* results, each a dict with ``skill_id``,
        ``score``, ``contract`` summary, and ``micro_plan`` (list of
        eff_add predicates as plan steps).
        """
        q_tokens = _tokenize(key)
        if not q_tokens:
            return self.list_all()[:top_k]

        scored: List[tuple] = []
        for sid in self._bank.skill_ids:
            id_score = _jaccard(q_tokens, self._id_tokens.get(sid, set()))
            eff_score = _jaccard(q_tokens, self._effect_tokens.get(sid, set()))
            total = 0.6 * id_score + 0.4 * eff_score
            scored.append((total, sid))

        scored.sort(key=lambda x: -x[0])
        results: List[Dict[str, Any]] = []
        for score, sid in scored[:top_k]:
            c = self._bank.get_contract(sid)
            entry: Dict[str, Any] = {
                "skill_id": sid,
                "score": round(score, 4),
            }
            if c is not None:
                entry["contract"] = _contract_summary(c)
                entry["micro_plan"] = [
                    {"action": None, "effect": lit}
                    for lit in sorted(c.eff_add or set())[:7]
                ]
            else:
                entry["contract"] = {}
                entry["micro_plan"] = []
            results.append(entry)
        return results

    # ── Effect-based query ───────────────────────────────────────────

    def query_by_effects(
        self,
        desired_add: Optional[set] = None,
        desired_del: Optional[set] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find skills whose contract effects best match desired state changes.

        Uses Jaccard similarity between the query effect set and each
        skill's effect set.
        """
        query_set = (desired_add or set()) | (desired_del or set())
        if not query_set:
            return self.list_all()[:top_k]

        scored: List[tuple] = []
        for sid, eff in self._effect_sets.items():
            sim = _jaccard(query_set, eff)
            if desired_add:
                c = self._bank.get_contract(sid)
                if c and c.eff_add:
                    add_overlap = len((desired_add or set()) & c.eff_add) / max(len(desired_add or set()), 1)
                    sim = 0.5 * sim + 0.5 * add_overlap
            scored.append((sim, sid))

        scored.sort(key=lambda x: -x[0])
        results: List[Dict[str, Any]] = []
        for score, sid in scored[:top_k]:
            c = self._bank.get_contract(sid)
            entry: Dict[str, Any] = {
                "skill_id": sid,
                "effect_match_score": round(score, 4),
            }
            if c is not None:
                entry["contract"] = _contract_summary(c)
            results.append(entry)
        return results

    # ── List / detail ────────────────────────────────────────────────

    def list_all(self) -> List[Dict[str, Any]]:
        """Return a compact summary for every skill in the bank."""
        entries: List[Dict[str, Any]] = []
        for sid in self._bank.skill_ids:
            c = self._bank.get_contract(sid)
            r = self._bank.get_report(sid)
            entry: Dict[str, Any] = {"skill_id": sid}
            if c is not None:
                entry["version"] = c.version
                entry["n_add"] = len(c.eff_add) if c.eff_add else 0
                entry["n_del"] = len(c.eff_del) if c.eff_del else 0
                entry["n_event"] = len(c.eff_event) if c.eff_event else 0
                entry["n_instances"] = c.n_instances
            if r is not None:
                entry["pass_rate"] = round(r.overall_pass_rate, 3)
            entries.append(entry)
        return entries

    def get_detail(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Full detail for one skill: contract + report + quality info."""
        c = self._bank.get_contract(skill_id)
        if c is None:
            return None

        detail: Dict[str, Any] = {"skill_id": skill_id}
        detail["contract"] = _contract_summary(c)

        r = self._bank.get_report(skill_id)
        if r is not None:
            detail["report"] = {
                "pass_rate": round(r.overall_pass_rate, 3),
                "n_instances": r.n_instances,
                "worst_segments": r.worst_segments[:5],
                "failure_signatures": dict(list(r.failure_signatures.items())[:5]),
            }
            detail["eff_add_rates"] = {
                k: round(v, 3) for k, v in r.eff_add_success_rate.items()
            }
            detail["eff_del_rates"] = {
                k: round(v, 3) for k, v in r.eff_del_success_rate.items()
            }
        return detail

    # ── Decision-agent convenience ───────────────────────────────────

    def query_for_decision_agent(
        self,
        key: str,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Convenience method returning a single best-match result dict
        compatible with the ``run_tool(TOOL_QUERY_SKILL, ...)`` interface
        in ``decision_agents.agent``.

        Returns ``{"skill_id": str|None, "micro_plan": list[dict]}``.
        """
        results = self.query(key, top_k=top_k)
        if not results:
            return {"skill_id": None, "micro_plan": []}

        best = results[0]
        return {
            "skill_id": best.get("skill_id"),
            "micro_plan": best.get("micro_plan", []) or [{"action": "proceed"}],
            "contract": best.get("contract", {}),
        }
