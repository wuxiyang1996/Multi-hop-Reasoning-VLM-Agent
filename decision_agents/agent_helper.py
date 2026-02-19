# This file defines helper functions for the VLM decision agent: state summarization,
# intention inference, episodic memory store, and skill-bank formatting.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from API_func import ask_model
except ImportError:
    ask_model = None


# ---------------------------------------------------------------------------
# State summary
# ---------------------------------------------------------------------------

def get_state_summary(
    observation: str,
    game: Optional[str] = None,
    model: Optional[str] = None,
    max_chars: int = 2000,
) -> str:
    """
    Produce a concise textual summary of the current observation for grounding.
    If observation is already short, return as-is; otherwise use LLM to summarize.
    """
    if not observation or not isinstance(observation, str):
        return ""
    obs = observation.strip()
    if len(obs) <= max_chars:
        return obs
    if ask_model is None:
        return obs[:max_chars] + "..."
    prompt = (
        "Summarize this game observation in 2-5 short sentences. "
        "Include: location/area, key entities, threats, objectives, and valid actions if mentioned. "
        "Keep it under 400 characters.\n\nObservation:\n" + obs[:4000]
    )
    return ask_model(prompt, model=model or "gpt-4o-mini", temperature=0.2, max_tokens=300)


# ---------------------------------------------------------------------------
# Intention inference
# ---------------------------------------------------------------------------

def infer_intention(
    summary_or_observation: str,
    game: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Infer current intention (short objective/subgoal) from state summary or observation.
    context can include: last_actions, progress_notes, task description.
    """
    if not summary_or_observation or not isinstance(summary_or_observation, str):
        return "Explore and survive."
    if ask_model is None:
        return "Complete objective."
    ctx = context or {}
    extra = ""
    if ctx.get("last_actions"):
        extra += "\nRecent actions: " + ", ".join(str(a) for a in ctx["last_actions"][-3:])
    if ctx.get("progress_notes"):
        extra += "\nProgress: " + " | ".join(ctx["progress_notes"])
    if ctx.get("task"):
        extra += "\nEpisode task: " + str(ctx["task"])
    prompt = (
        "Given this game state, output a single short phrase (under 15 words) "
        "describing the agent's current objective or subgoal. Be specific (e.g. 'Reach checkpoint', 'Avoid sniper and heal').\n\n"
        "State:\n" + summary_or_observation[:2500] + extra + "\n\nIntention phrase:"
    )
    out = ask_model(prompt, model=model or "gpt-4o-mini", temperature=0.2, max_tokens=80)
    return (out or "Complete objective.").strip()[:200]


# ---------------------------------------------------------------------------
# Episodic memory store (for query_memory)
# ---------------------------------------------------------------------------

class EpisodicMemoryStore:
    """Episodic memory with RAG-embedding retrieval (cosine similarity) and
    keyword-overlap fallback.

    When an ``embedder`` is provided (a ``TextEmbedderBase`` from
    ``rag.embedding``), every memory is embedded on ``add`` and queries are
    scored via cosine similarity.  The final score is a weighted mix of
    embedding similarity and keyword overlap so the system degrades
    gracefully if the embedding model is unavailable.

    When no embedder is provided, behaviour is identical to the original
    keyword-overlap-only store.
    """

    def __init__(
        self,
        max_entries: int = 500,
        embedder: Any = None,
        embedding_weight: float = 0.7,
    ) -> None:
        """
        Args:
            max_entries: Maximum number of memories to keep (FIFO eviction).
            embedder: Optional ``TextEmbedderBase`` (e.g. from
                ``rag.get_text_embedder()``).  Enables embedding retrieval.
            embedding_weight: Blend weight for embedding vs keyword score
                (0 = keyword only, 1 = embedding only).
        """
        self._entries: List[Dict[str, Any]] = []
        self._max_entries = max_entries
        self._embedder = embedder
        self._embedding_weight = embedding_weight
        self._memory_store: Any = None
        if embedder is not None:
            self._init_memory_store(embedder)

    def _init_memory_store(self, embedder: Any) -> None:
        try:
            from rag.retrieval import MemoryStore
            self._memory_store = MemoryStore(embedder=embedder, top_k=self._max_entries)
        except ImportError:
            self._memory_store = None

    def add(
        self,
        key: str,
        summary: str,
        action: Any = None,
        outcome: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add one memory entry and embed it if an embedder is available."""
        entry = {
            "key": key,
            "summary": summary,
            "action": action,
            "outcome": outcome,
            **(extra or {}),
        }
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        if self._memory_store is not None:
            text = (key + " " + summary).strip()
            if text:
                try:
                    self._memory_store.add_texts([text], payloads=[entry])
                except Exception:
                    pass

    def add_experience(self, state_summary: str, action: Any, next_state_summary: str, done: bool) -> None:
        """Convenience: add from a single experience."""
        key = state_summary[:200] if state_summary else ""
        self.add(key=key, summary=state_summary, action=action, outcome=next_state_summary, extra={"done": done})

    def query(self, query_key: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by embedding similarity + keyword overlap.

        If an embedder is available, scores are a weighted blend of cosine
        similarity and keyword overlap.  Otherwise falls back to keyword only.
        """
        if not query_key or not self._entries:
            return []

        keyword_scores = self._keyword_scores(query_key)

        if self._memory_store is not None and len(self._memory_store) > 0:
            try:
                ranked = self._memory_store.rank(query_key, k=len(self._entries))
                emb_scores = {idx: score for idx, score, _ in ranked}
            except Exception:
                emb_scores = {}
        else:
            emb_scores = {}

        w = self._embedding_weight if emb_scores else 0.0
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, entry in enumerate(self._entries):
            kw = keyword_scores[i]
            emb = emb_scores.get(i, 0.0)
            combined = w * emb + (1.0 - w) * kw
            scored.append((combined, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def _keyword_scores(self, query_key: str) -> List[float]:
        q_lower = query_key.lower()
        q_words = set(w for w in q_lower.split() if len(w) >= 2)
        scores: List[float] = []
        for e in self._entries:
            text = (e.get("key", "") + " " + e.get("summary", "")).lower()
            t_words = set(w for w in text.split() if len(w) >= 2)
            overlap = len(q_words & t_words) / max(len(q_words), 1)
            scores.append(overlap)
        return scores

    @property
    def has_embedder(self) -> bool:
        return self._memory_store is not None

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Skill bank formatting for agent prompt
# ---------------------------------------------------------------------------

def skill_bank_to_text(skill_bank: Any) -> str:
    """
    Format skill bank for inclusion in agent prompt (skill_ids + short effect summary).
    skill_bank can be SkillBankMVP, SkillBankAgent, or any object with
    .skill_ids and .get_contract(skill_id).
    """
    if skill_bank is None:
        return "(no skill bank)"

    # SkillBankAgent wraps a SkillBankMVP; unwrap if needed
    bank = getattr(skill_bank, "bank", skill_bank)

    try:
        ids = list(bank.skill_ids)[:50]
    except AttributeError:
        return "(no skill bank)"
    if not ids:
        return "(empty skill bank)"

    lines = [f"Available skills ({len(ids)}): " + ", ".join(ids)]
    for sid in ids[:15]:
        try:
            c = bank.get_contract(sid)
            if c is not None:
                add = getattr(c, "eff_add", set()) or set()
                dele = getattr(c, "eff_del", set()) or set()
                add_preview = ", ".join(sorted(add)[:3])
                parts = [f"add({len(add)})", f"del({len(dele)})"]
                if add_preview:
                    parts.append(f"e.g. {add_preview}")
                r = bank.get_report(sid) if hasattr(bank, "get_report") else None
                if r is not None:
                    parts.append(f"pass={r.overall_pass_rate:.0%}")
                lines.append(f"  - {sid}: {', '.join(parts)}")
        except Exception:
            lines.append(f"  - {sid}")
    return "\n".join(lines)


def query_skill_bank(skill_bank: Any, key: str, top_k: int = 1) -> Dict[str, Any]:
    """Query the skill bank and return a result compatible with the QUERY_SKILL tool.

    Supports SkillBankAgent (rich query), SkillQueryEngine, and plain SkillBankMVP
    (fallback to name matching).

    Returns ``{"skill_id": str|None, "micro_plan": list[dict], ...}``.
    """
    if skill_bank is None:
        return {"skill_id": None, "micro_plan": []}

    # SkillBankAgent has .query_skill()
    if hasattr(skill_bank, "query_skill"):
        results = skill_bank.query_skill(key, top_k=top_k)
        if results:
            best = results[0]
            return {
                "skill_id": best.get("skill_id"),
                "micro_plan": best.get("micro_plan", []) or [{"action": "proceed"}],
                "contract": best.get("contract", {}),
            }
        return {"skill_id": None, "micro_plan": []}

    # SkillQueryEngine
    if hasattr(skill_bank, "query_for_decision_agent"):
        return skill_bank.query_for_decision_agent(key, top_k=top_k)

    # Fallback: plain SkillBankMVP or similar — name match
    bank = getattr(skill_bank, "bank", skill_bank)
    try:
        ids = list(bank.skill_ids)
    except AttributeError:
        return {"skill_id": None, "micro_plan": []}

    key_lower = key.lower()
    skill_id = None
    for sid in ids:
        if sid.lower() in key_lower or key_lower in sid.lower():
            skill_id = sid
            break
    if skill_id is None and ids:
        skill_id = ids[0]

    if skill_id:
        c = bank.get_contract(skill_id)
        if c:
            add_set = getattr(c, "eff_add", set()) or set()
            steps = [{"action": None, "effect": lit} for lit in sorted(add_set)[:5]]
            return {"skill_id": skill_id, "micro_plan": steps or [{"action": "proceed"}]}

    return {"skill_id": None, "micro_plan": []}
