"""Shared abstract skill bank — the cross-game skill skeleton store.

The repository previously held three independent skill representations:

* ``labeling/skill_bank_*/<run>/<task>/skill_bank.jsonl`` — per-task
  mining output with concrete contracts (preconditions / effects /
  predicates).  Per-task IDs (e.g. ``COMMIT/CLEAR``).
* ``labeling/skill_templates/run_*/<cohort>/<task>/template_bank.jsonl``
  — Layer-C lifted, modality-agnostic 2-5 step templates.
* ``runs/<run>/transfer_log/usage.jsonl`` — production trainer usage,
  which decorated IDs with ``early:`` / ``mid:`` / ``late:`` phase
  prefixes, ``#v2`` crafter-evolution suffixes, or
  ``__translated_to__<task>`` translation suffixes.  None of these
  variants made it into the lift index, which is why
  ``TemplateIndex.coverage_on_prod_ids`` was 6.6 %.

This module unifies all three into a single two-layer store:

* :class:`SharedAbstractSkill` — the **cross-game skill skeleton**.
  Stores the modality-agnostic step skeleton (``template_steps``)
  AND the original multi-hop protocol skeleton (``protocol_steps``)
  that is *already* free of game-specific button / DOM tokens.  All
  game vocabulary lives in the lineage, never in the skeleton.
* :class:`BoundConcreteSkill` — one concrete (task, contract,
  protocol) binding for an abstract skill.  This is what the harness
  actually runs against.

Bidirectional flow (both required by the new pipeline):

  Forward (transfer):  SharedAbstractSkill --LLM convert--> candidate
  BoundConcreteSkill --harness validate--> committed binding.

  Backward (discovery): new skill mined / crafter-proposed in task X
  --LLM lift--> abstract template --upsert into SharedAbstractBank
  (new entry OR new lineage entry on existing abstract).

Storage layout (default ``shared_skill_bank/`` next to the lift run):

    shared_skill_bank/
      abstract.jsonl            ← SharedAbstractSkill records
      by_task/<task>/bindings.jsonl  ← BoundConcreteSkill records

Both files are JSONL.  Reads go through the reader classes which
de-dup on ``stable_key`` (see :meth:`SharedAbstractSkill.stable_key`).
Writes go through :meth:`SharedAbstractBank.upsert_abstract` /
:meth:`PerTaskBank.upsert_binding` — which append-only-write a new
record and let the next read pick the latest by ``updated_at``.

This module is dependency-free: no LLM clients, no harness imports,
just dataclasses + JSONL I/O.  The forward-convert / harness-
validate / LLM-lift wiring lives in ``scripts/bind_abstract_to_task.py``
and ``scripts/discover_skill_to_shared_bank.py``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("shared_abstract_bank")


# ---------------------------------------------------------------------------
# ID normalisation — the core fix for the 6.6 % production coverage hole.
# ---------------------------------------------------------------------------
# Trainer-side decorations layered over a stable mining ID:
#
#   early:COMMIT/CLEAR                  → COMMIT/CLEAR        (phase prefix)
#   mid:EXECUTE                          → EXECUTE             (phase prefix)
#   late:INSPECT/SETUP                   → INSPECT/SETUP       (phase prefix)
#   COMMIT/CLEAR#v2                      → COMMIT/CLEAR        (crafter version)
#   INSPECT/SETUP__translated_to__gymv_altered_beast
#                                        → INSPECT/SETUP       (translated)
#
# A skill_id is its OWN stem if none of these patterns match.  This
# function is INTENTIONALLY conservative: it never strips any unknown
# suffix / prefix, so legitimate IDs stay verbatim.

_PHASE_PREFIX = re.compile(r"^(?:early|mid|late):", re.IGNORECASE)
# ``#v<n>`` may be followed by a hash suffix (``:9a370dbc``) when the
# crafter promoted a versioned twin under a stable hash; we strip the
# whole ``#v<n>(:<hex>)?`` tail.  Examples seen in production:
#   COMMIT/CLEAR#v2                    → COMMIT/CLEAR
#   RECOVER/EVADE#v2:9a370dbc          → RECOVER/EVADE
_VERSION_TAG  = re.compile(r"#v\d+(?::[0-9a-f]+)?$", re.IGNORECASE)
_TRANSLATED_TAIL = re.compile(r"__translated_to__.+$", re.IGNORECASE)


def normalise_skill_id(skill_id: str) -> str:
    """Strip phase / version / translation decorations, returning the
    stable mining stem.  Idempotent.

    Examples
    --------
    >>> normalise_skill_id("late:INSPECT/SETUP__translated_to__gymv_altered_beast")
    'INSPECT/SETUP'
    >>> normalise_skill_id("COMMIT/CLEAR#v2")
    'COMMIT/CLEAR'
    >>> normalise_skill_id("skill-00f509f288")
    'skill-00f509f288'
    """
    if not isinstance(skill_id, str):
        return ""
    s = skill_id.strip()
    s = _PHASE_PREFIX.sub("", s, count=1)
    s = _TRANSLATED_TAIL.sub("", s, count=1)
    s = _VERSION_TAG.sub("", s, count=1)
    return s


def parse_skill_id_decorations(skill_id: str) -> Dict[str, Any]:
    """Reverse of :func:`normalise_skill_id` — return the decorations
    that were stripped.  Used by the consolidator to track lineage
    metadata (which task a translation targeted, which crafter version
    was used, ...).
    """
    if not isinstance(skill_id, str):
        return {}
    out: Dict[str, Any] = {}
    s = skill_id.strip()
    m = _PHASE_PREFIX.match(s)
    if m:
        out["phase"] = m.group(0).rstrip(":").lower()
        s = s[m.end():]
    m = _TRANSLATED_TAIL.search(s)
    if m:
        out["translated_to"] = m.group(0).removeprefix("__translated_to__")
        s = s[: m.start()]
    m = _VERSION_TAG.search(s)
    if m:
        tag = m.group(0).lstrip("#")
        if ":" in tag:
            ver, vhash = tag.split(":", 1)
            out["crafter_version"] = ver
            out["crafter_version_hash"] = vhash
        else:
            out["crafter_version"] = tag
        s = s[: m.start()]
    out["stem"] = s
    return out


# ---------------------------------------------------------------------------
# Step / Protocol primitives
# ---------------------------------------------------------------------------
@dataclass
class TemplateStep:
    """One step of the modality-agnostic Layer-C template."""
    op: str           # PERCEIVE / RECALL / COMPARE / FILTER / DECIDE / COMMIT / VERIFY / RECOVER
    predicate: str    # 6-12 word abstract description of what happens at this step

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemplateStep":
        return cls(
            op=str(d.get("op", "")).upper(),
            predicate=str(d.get("predicate", "")),
        )


@dataclass
class ProtocolStep:
    """One step of the original multi-hop reasoning protocol that
    produced the skill.  Game-vocabulary tokens (button glyphs, DOM
    selectors, gold answers, ...) are kept in ``payload`` — the
    skeleton itself stays modality-agnostic.

    Concretely, this is the form the mining pipeline emits in
    ``skill_bank.jsonl`` under ``protocol``: a list of dicts with
    ``op``, ``notes``, ``evidence_role``, and ``payload``.
    """
    op: str
    notes: str = ""
    evidence_role: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProtocolStep":
        return cls(
            op=str(d.get("op", "")),
            notes=str(d.get("notes", "")),
            evidence_role=str(d.get("evidence_role", "")),
            payload=dict(d.get("payload") or {}),
        )


@dataclass
class LineageEntry:
    """One concrete (task, skill_id) that binds to a SharedAbstractSkill."""
    task: str
    concrete_skill_id: str            # task-local ID (stem-form)
    raw_skill_id: str                 # the FULL ID as seen in production logs
    cohort: str                       # gymv_game / env_wr_game / web / vr_image / vr_video
    discovered_via: str               # 'mining' | 'crafter' | 'translation' | 'production_usage' | 'binding'
    is_native: bool                   # True iff the skill was first discovered IN this task
    n_uses: int = 0                   # times observed in production logs (if available)
    n_success: int = 0                # production successes
    n_translated_uses: int = 0        # production uses where it was a translation
    contract_hash: str = ""           # hash of the bound contract for change detection
    decorations: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineageEntry":
        return cls(
            task=str(d.get("task", "")),
            concrete_skill_id=str(d.get("concrete_skill_id", "")),
            raw_skill_id=str(d.get("raw_skill_id", "")),
            cohort=str(d.get("cohort", "")),
            discovered_via=str(d.get("discovered_via", "")),
            is_native=bool(d.get("is_native", False)),
            n_uses=int(d.get("n_uses", 0) or 0),
            n_success=int(d.get("n_success", 0) or 0),
            n_translated_uses=int(d.get("n_translated_uses", 0) or 0),
            contract_hash=str(d.get("contract_hash", "")),
            decorations=dict(d.get("decorations") or {}),
            notes=str(d.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# SharedAbstractSkill — the "skill skeleton" stored in the cross-game bank
# ---------------------------------------------------------------------------
@dataclass
class SharedAbstractSkill:
    """The cross-game skeleton of one skill.

    Stable key: ``stable_key()`` = ``(skill_id_stem, template_signature)``
    so two skills with the same name but a different procedural
    skeleton (rare but possible) are kept as separate abstracts.
    """
    abstract_skill_id: str             # stable cross-game ID (skill_id stem)
    name: str                           # human-readable name
    template_signature: str             # "PERCEIVE → COMPARE → DECIDE → COMMIT → VERIFY"
    template_steps: List[TemplateStep] = field(default_factory=list)
    protocol_steps: List[ProtocolStep] = field(default_factory=list)
    lineage: List[LineageEntry] = field(default_factory=list)
    cohorts_seen: List[str] = field(default_factory=list)
    discovered_via: str = "mining"     # initial discovery channel
    schema_version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # ── stable identity ───────────────────────────────────────────
    def stable_key(self) -> Tuple[str, str]:
        return (self.abstract_skill_id, self.template_signature)

    @property
    def n_lineage(self) -> int:
        return len(self.lineage)

    @property
    def n_native_tasks(self) -> int:
        return sum(1 for L in self.lineage if L.is_native)

    @property
    def n_bound_tasks(self) -> int:
        return len({L.task for L in self.lineage})

    @property
    def total_production_uses(self) -> int:
        return sum(L.n_uses for L in self.lineage)

    @property
    def total_production_successes(self) -> int:
        return sum(L.n_success for L in self.lineage)

    # ── serialisation ─────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstract_skill_id":    self.abstract_skill_id,
            "name":                 self.name,
            "template_signature":   self.template_signature,
            "template_steps":       [asdict(s) for s in self.template_steps],
            "protocol_steps":       [asdict(s) for s in self.protocol_steps],
            "lineage":              [asdict(L) for L in self.lineage],
            "cohorts_seen":         sorted(set(self.cohorts_seen)),
            "discovered_via":       self.discovered_via,
            "schema_version":       self.schema_version,
            "created_at":           self.created_at,
            "updated_at":           self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SharedAbstractSkill":
        return cls(
            abstract_skill_id=str(d.get("abstract_skill_id", "")),
            name=str(d.get("name", "")),
            template_signature=str(d.get("template_signature", "")),
            template_steps=[TemplateStep.from_dict(s) for s in (d.get("template_steps") or [])],
            protocol_steps=[ProtocolStep.from_dict(s) for s in (d.get("protocol_steps") or [])],
            lineage=[LineageEntry.from_dict(L) for L in (d.get("lineage") or [])],
            cohorts_seen=list(d.get("cohorts_seen") or []),
            discovered_via=str(d.get("discovered_via", "mining")),
            schema_version=int(d.get("schema_version", 1)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )

    # ── lineage upserts (used by the discovery path) ──────────────
    def upsert_lineage(self, entry: LineageEntry) -> bool:
        """Append ``entry`` if we don't already have a (task,
        concrete_skill_id) match; else MERGE counts onto the
        existing entry.  Returns True iff a new entry was added."""
        for L in self.lineage:
            if L.task == entry.task and L.concrete_skill_id == entry.concrete_skill_id:
                L.n_uses += entry.n_uses
                L.n_success += entry.n_success
                L.n_translated_uses += entry.n_translated_uses
                if entry.contract_hash and not L.contract_hash:
                    L.contract_hash = entry.contract_hash
                if entry.is_native and not L.is_native:
                    L.is_native = True
                if entry.decorations:
                    L.decorations.update(entry.decorations)
                return False
        self.lineage.append(entry)
        if entry.cohort and entry.cohort not in self.cohorts_seen:
            self.cohorts_seen.append(entry.cohort)
        return True


# ---------------------------------------------------------------------------
# BoundConcreteSkill — what lives in PerTaskBank, what the harness runs
# ---------------------------------------------------------------------------
@dataclass
class BoundConcreteSkill:
    """A SharedAbstractSkill bound to a specific task.

    ``contract`` and ``protocol`` carry the **task-specific**
    grounding (game vocabulary, DOM selectors, predicate names that
    actually exist on the target).  ``abstract_skill_id`` is the
    cross-game reference."""
    concrete_skill_id: str             # task-local ID (e.g. "COMMIT/CLEAR" — the stem)
    task: str
    abstract_skill_id: Optional[str]   # ref into SharedAbstractBank
    name: str = ""
    contract: Dict[str, Any] = field(default_factory=dict)
    protocol: List[Dict[str, Any]] = field(default_factory=list)
    binding_status: str = "PENDING"    # PENDING | VALIDATED | REJECTED | NATIVE_DRAFT
    binding_source: str = "mining"     # mining | crafter | forward_convert | translation | seed
    n_episodes_verified: int = 0
    pass_rate: float = 0.0
    last_validation_at: str = ""
    raw_skill_id: str = ""              # original ID as seen on disk / in production
    decorations: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundConcreteSkill":
        return cls(
            concrete_skill_id=str(d.get("concrete_skill_id", "")),
            task=str(d.get("task", "")),
            abstract_skill_id=d.get("abstract_skill_id"),
            name=str(d.get("name", "")),
            contract=dict(d.get("contract") or {}),
            protocol=list(d.get("protocol") or []),
            binding_status=str(d.get("binding_status", "PENDING")),
            binding_source=str(d.get("binding_source", "mining")),
            n_episodes_verified=int(d.get("n_episodes_verified", 0) or 0),
            pass_rate=float(d.get("pass_rate", 0.0) or 0.0),
            last_validation_at=str(d.get("last_validation_at", "")),
            raw_skill_id=str(d.get("raw_skill_id", "")),
            decorations=dict(d.get("decorations") or {}),
            schema_version=int(d.get("schema_version", 1)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )


# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_contract(contract: Dict[str, Any]) -> str:
    blob = json.dumps(contract, sort_keys=True, ensure_ascii=False)
    return "sha1:" + hashlib.sha1(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SharedAbstractBank — append-only JSONL with in-memory index
# ---------------------------------------------------------------------------
class SharedAbstractBank:
    """Append-only JSONL store for :class:`SharedAbstractSkill`.

    On read, records are de-duplicated by ``stable_key``; the latest
    ``updated_at`` wins.  On write, a fresh record is appended (no
    in-place update) so the file doubles as an audit trail.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._abstract_path = self.root / "abstract.jsonl"
        self._records: Dict[Tuple[str, str], SharedAbstractSkill] = {}
        self._loaded = False

    # ── I/O ───────────────────────────────────────────────────────
    def load(self) -> None:
        self._records.clear()
        self._loaded = True
        if not self._abstract_path.exists():
            return
        with self._abstract_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rec = SharedAbstractSkill.from_dict(d)
                key = rec.stable_key()
                old = self._records.get(key)
                if old is None or rec.updated_at >= old.updated_at:
                    self._records[key] = rec

    @property
    def abstract_path(self) -> Path:
        return self._abstract_path

    @property
    def records(self) -> List[SharedAbstractSkill]:
        if not self._loaded:
            self.load()
        return list(self._records.values())

    @property
    def size(self) -> int:
        return len(self.records)

    # ── upsert (idempotent — backward discovery + forward consolidation) ──
    def upsert_abstract(self, rec: SharedAbstractSkill) -> str:
        """Insert or merge ``rec``.  Returns ``"new" | "merged"``."""
        if not self._loaded:
            self.load()
        key = rec.stable_key()
        if not rec.created_at:
            rec.created_at = _now_iso()
        rec.updated_at = _now_iso()

        existing = self._records.get(key)
        verdict: str
        if existing is None:
            self._records[key] = rec
            verdict = "new"
        else:
            for L in rec.lineage:
                existing.upsert_lineage(L)
            for c in rec.cohorts_seen:
                if c not in existing.cohorts_seen:
                    existing.cohorts_seen.append(c)
            if not existing.template_steps and rec.template_steps:
                existing.template_steps = rec.template_steps
            if not existing.protocol_steps and rec.protocol_steps:
                existing.protocol_steps = rec.protocol_steps
            if not existing.name and rec.name:
                existing.name = rec.name
            existing.updated_at = _now_iso()
            verdict = "merged"

        self._append(self._records[key])
        return verdict

    def upsert_lineage(
        self, abstract_skill_id: str, template_signature: str,
        entry: LineageEntry,
    ) -> str:
        """Append a single lineage entry to an existing abstract.  If
        the abstract isn't found, returns ``"missing"`` and does
        nothing — the caller is expected to upsert the abstract
        first."""
        if not self._loaded:
            self.load()
        key = (abstract_skill_id, template_signature)
        existing = self._records.get(key)
        if existing is None:
            return "missing"
        existing.upsert_lineage(entry)
        existing.updated_at = _now_iso()
        if entry.cohort and entry.cohort not in existing.cohorts_seen:
            existing.cohorts_seen.append(entry.cohort)
        self._append(existing)
        return "merged"

    def _append(self, rec: SharedAbstractSkill) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._abstract_path.open("a") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    # ── lookup helpers ───────────────────────────────────────────
    def by_abstract_id(self, abstract_skill_id: str) -> List[SharedAbstractSkill]:
        if not self._loaded:
            self.load()
        return [r for r in self._records.values()
                if r.abstract_skill_id == abstract_skill_id]

    def by_signature(self, template_signature: str) -> List[SharedAbstractSkill]:
        if not self._loaded:
            self.load()
        return [r for r in self._records.values()
                if r.template_signature == template_signature]

    def candidates_for_target_task(
        self, target_task: str, *, exclude_already_bound: bool = True,
    ) -> List[SharedAbstractSkill]:
        """Return abstract skills that haven't yet been bound to
        ``target_task`` — the forward-bind candidate set."""
        if not self._loaded:
            self.load()
        out: List[SharedAbstractSkill] = []
        for r in self._records.values():
            already = any(L.task == target_task for L in r.lineage)
            if exclude_already_bound and already:
                continue
            out.append(r)
        return out


# ---------------------------------------------------------------------------
# PerTaskBank — append-only JSONL for one task's bindings
# ---------------------------------------------------------------------------
class PerTaskBank:
    """Append-only JSONL store for :class:`BoundConcreteSkill` of one task.

    File path: ``<root>/by_task/<task>/bindings.jsonl``.

    De-dup key on read: ``(concrete_skill_id, task)``; latest
    ``updated_at`` wins.
    """

    def __init__(self, root: Path, task: str) -> None:
        self.root = Path(root)
        self.task = task
        self._bindings_path = self.root / "by_task" / task / "bindings.jsonl"
        self._records: Dict[str, BoundConcreteSkill] = {}
        self._loaded = False

    def load(self) -> None:
        self._records.clear()
        self._loaded = True
        if not self._bindings_path.exists():
            return
        with self._bindings_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rec = BoundConcreteSkill.from_dict(d)
                old = self._records.get(rec.concrete_skill_id)
                if old is None or rec.updated_at >= old.updated_at:
                    self._records[rec.concrete_skill_id] = rec

    @property
    def bindings_path(self) -> Path:
        return self._bindings_path

    @property
    def records(self) -> List[BoundConcreteSkill]:
        if not self._loaded:
            self.load()
        return list(self._records.values())

    @property
    def size(self) -> int:
        return len(self.records)

    def upsert_binding(self, rec: BoundConcreteSkill) -> str:
        """Idempotent upsert.  Returns ``"new" | "updated"``."""
        if not self._loaded:
            self.load()
        if not rec.created_at:
            rec.created_at = _now_iso()
        rec.updated_at = _now_iso()
        rec.task = self.task

        old = self._records.get(rec.concrete_skill_id)
        verdict: str
        if old is None:
            self._records[rec.concrete_skill_id] = rec
            verdict = "new"
        else:
            old.contract = rec.contract or old.contract
            old.protocol = rec.protocol or old.protocol
            old.binding_status = rec.binding_status or old.binding_status
            old.binding_source = rec.binding_source or old.binding_source
            old.pass_rate = rec.pass_rate or old.pass_rate
            old.n_episodes_verified = max(
                rec.n_episodes_verified, old.n_episodes_verified,
            )
            old.last_validation_at = rec.last_validation_at or old.last_validation_at
            old.abstract_skill_id = rec.abstract_skill_id or old.abstract_skill_id
            old.raw_skill_id = rec.raw_skill_id or old.raw_skill_id
            if rec.decorations:
                old.decorations.update(rec.decorations)
            old.updated_at = _now_iso()
            self._records[rec.concrete_skill_id] = old
            verdict = "updated"

        self._bindings_path.parent.mkdir(parents=True, exist_ok=True)
        with self._bindings_path.open("a") as f:
            f.write(json.dumps(self._records[rec.concrete_skill_id].to_dict(),
                               ensure_ascii=False) + "\n")
        return verdict

    def by_concrete_id(self, concrete_skill_id: str) -> Optional[BoundConcreteSkill]:
        if not self._loaded:
            self.load()
        return self._records.get(concrete_skill_id)


# ---------------------------------------------------------------------------
# Composite store — convenience wrapper holding both layers
# ---------------------------------------------------------------------------
class TwoLayerSkillStore:
    """Convenience wrapper around the SharedAbstractBank + a lazily-
    instantiated PerTaskBank cache.  This is the surface intended for
    use by the trainer / orchestrator."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.abstract = SharedAbstractBank(self.root)
        self._task_banks: Dict[str, PerTaskBank] = {}

    def per_task(self, task: str) -> PerTaskBank:
        if task not in self._task_banks:
            self._task_banks[task] = PerTaskBank(self.root, task)
            self._task_banks[task].load()
        return self._task_banks[task]

    def list_tasks(self) -> List[str]:
        by_task_dir = self.root / "by_task"
        if not by_task_dir.exists():
            return []
        return sorted(p.name for p in by_task_dir.iterdir() if p.is_dir())

    # ── high-level API used by orchestrator hooks ─────────────────
    def insert_discovered_skill(
        self,
        *,
        concrete: BoundConcreteSkill,
        abstract: SharedAbstractSkill,
        lineage: LineageEntry,
    ) -> Dict[str, str]:
        """The DISCOVERY path: a new skill was mined / proposed by
        crafter inside one task.  Insert it into PerTaskBank AND
        upsert the lifted abstract into SharedAbstractBank with a
        lineage entry pointing back to the concrete record.

        Returns a tiny verdict dict: ``{abstract: 'new'|'merged',
        binding: 'new'|'updated'}``.
        """
        abstract.upsert_lineage(lineage)
        abstract_verdict = self.abstract.upsert_abstract(abstract)
        binding_verdict = self.per_task(concrete.task).upsert_binding(concrete)
        return {"abstract": abstract_verdict, "binding": binding_verdict}

    def insert_validated_binding(
        self,
        *,
        concrete: BoundConcreteSkill,
        abstract_skill_id: str,
        template_signature: str,
        lineage: LineageEntry,
    ) -> Dict[str, str]:
        """The FORWARD-BIND path: an existing abstract was converted
        + harness-validated for a new task.  The abstract record
        itself is unchanged; only its lineage gets a new entry, and
        a BoundConcreteSkill is written to the per-task bank."""
        binding_verdict = self.per_task(concrete.task).upsert_binding(concrete)
        abstract_verdict = self.abstract.upsert_lineage(
            abstract_skill_id, template_signature, lineage,
        )
        return {"abstract": abstract_verdict, "binding": binding_verdict}


# ---------------------------------------------------------------------------
__all__ = [
    "BoundConcreteSkill",
    "LineageEntry",
    "ProtocolStep",
    "PerTaskBank",
    "SharedAbstractBank",
    "SharedAbstractSkill",
    "TemplateStep",
    "TwoLayerSkillStore",
    "hash_contract",
    "normalise_skill_id",
    "parse_skill_id_decorations",
]
