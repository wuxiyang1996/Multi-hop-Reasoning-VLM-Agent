#!/usr/bin/env python
"""scripts.skillbridge_eval.run_promotion_attribution
─────────────────────────────────────────────────────
Take the per-mode ``proposals.jsonl`` files emitted by
:mod:`scripts.skillbridge_eval.run_smoke_attribution` (which writes
the *crafter-side* schema produced by
``data_structure.extensions.bank_mutation_proposal.proposal_to_json``)
and feed them through the live promotion gate
(``labeling_supplement/decide_promotion_gpt54.py``) so we can read
the **promote-rate** for each mode — i.e. the share of crafter
proposals that survive Stage 0 ⊕ 1 ⊕ 2 ⊕ 4 (and Stage 3 placeholder
in the default ``offline-synthetic`` gate mode).

Why a separate runner instead of plumbing this into the smoke runner:

1. The smoke runner is intentionally Stage-0-only and self-contained
   (just the static checks ``GateService._run_static`` does), so it
   can run with **zero** auxiliary inputs (no bank_run, no actions
   trail, no judge model). That keeps it under a minute on 50 VTB
   samples and is enough to surface diversity / boilerplate signal.
2. The promotion gate has hard schema requirements that the
   crafter-side ``proposal_to_json`` schema does not satisfy:
     * ``proposal_kind`` (lowercase) instead of ``type`` (PascalCase)
     * ``target_skill_id`` instead of ``base_skill_id`` (for patches)
     * ``patch_kind`` instead of ``recovery_strategy``
     * ``proposer`` enum (composer / generalizer / hypothesizer / reflector)
     * ``adapter_plan`` (per-domain reuse vs synthesize verdict)
     * ``target_domains`` MUST be the full ``ALL_FIVE_DOMAINS`` set
       to clear Stage-0 (PLAN-SKILL-CRAFTER §0.1).
   We translate row-by-row using the same logic as
   ``trainer/coevolution/_crafter_hook._to_offline_row`` (the
   canonical adapter the trainer ships with), then write the
   translated rows under ``<adapter_root>/<corpus>/<source>/proposals.jsonl``
   so the promotion gate's ``_discover_pairs`` (which only walks
   ``CORPORA = ("gym_v", "env_wrappers")``) can find them.
3. The gate also requires ``<bank_run>/<corpus>/<source>/skill_bank.jsonl``
   to exist for any pair it processes. The smoke runs are **seeded**
   from ``labeling/skill_bank_out/run_20260430_030637/env_wrappers/twenty_forty_eight/skill_bank.jsonl``,
   so we naturally fold the smoke output under that ``(corpus, source)``
   pair and reuse the same bank_run for the gate. The mismatch
   between "gate thinks this is a 2048 promotion" and "the proposal
   actually came from a VTB failure" is *intentional* for this
   smoke — we're measuring **gate pass-rate of LLM-minted vs
   rule-minted proposals on identical Stage 0/1/2/4 plumbing**, not
   the cross-domain transferability of the proposed skill itself.

Outputs:

  <output_dir>/<mode>/                              ← adapter root the gate eats
    env_wrappers/twenty_forty_eight/proposals.jsonl
  <output_dir>/<mode>/_promotion/                   ← gate's per-mode output
    _run_summary.json
    _run_meta.json
    env_wrappers/twenty_forty_eight/promotion_decisions.jsonl
    env_wrappers/twenty_forty_eight/skill_evaluations.jsonl
  <output_dir>/_promotion_summary.json              ← cross-mode roll-up
  <output_dir>/_promotion_summary.md                ← human-readable table

Usage::

    python -m scripts.skillbridge_eval.run_promotion_attribution \\
        --smoke-dir labeling_supplement/episode_reflections_out/_smoke_attr_v2 \\
        --bank-run labeling/skill_bank_out/run_20260430_030637 \\
        --output-dir labeling_supplement/episode_reflections_out/_promotion_attr_v2

By default the gate runs in ``offline-synthetic`` mode (no LLM judge,
deterministic Stage 0 + LIMITED_PASS placeholders for Stages 1-4 —
the documented Phase-1 floor). Pass ``--gate-mode offline-with-llm-judge``
to add the 35B-A3B judge call (one per proposal), which actually
contributes a *FAIL* signal (otherwise lane_b_llm patches always pass).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mirror trainer-hook constants so the adapter row matches byte-for-byte
# what the live trainer would have written, even if those constants
# evolve. Keeps the gate's interpretation identical for either
# producer.
from trainer.coevolution._crafter_hook import (  # noqa: E402
    ALL_FIVE_DOMAINS,
    _default_adapter_plan,
)

logger = logging.getLogger("skillbridge_eval.run_promotion_attribution")

# ---------------------------------------------------------------------------
# proposal_to_json (crafter side) → _to_offline_row (gate side) translator
# ---------------------------------------------------------------------------

# Map ``type`` (PascalCase from ``proposal_to_json``) → tuple of
# (proposal_kind for the gate, proposer enum). The proposer enum is
# the *deterministic* default — for proposals that came from the
# LLM hooks we override to ``llm_crafter`` further down.
_TYPE_TO_KIND: Dict[str, Tuple[str, str]] = {
    "PatchProposal":       ("patch",       "reflector"),
    "RetireProposal":      ("retire",      "reflector"),
    "ComposeProposal":     ("compose",     "composer"),
    "GeneralizeProposal":  ("transfer",    "generalizer"),
    "HypothesisProposal":  ("hypothesize", "hypothesizer"),
}


def _translate_row(
    row: Dict[str, Any],
    *,
    domain: str,
    is_llm: bool,
) -> Dict[str, Any]:
    """Translate one ``proposal_to_json`` dict into the offline-mirror
    JSONL row shape ``decide_promotion_gpt54.py::_OfflineProposal.from_json``
    expects.

    Mirrors :func:`trainer.coevolution._crafter_hook._to_offline_row`
    field-by-field — kept inline so we can run without instantiating
    a ``BankMutationProposal`` (the JSONL on disk is already a flat
    dict and the dataclass round-trip would force us to reconstruct
    ``SkillContract`` etc. for fields we never read at the gate).
    """
    type_name = row.get("type", "")
    kind, default_proposer = _TYPE_TO_KIND.get(type_name, (type_name.lower(), "reflector"))
    out: Dict[str, Any] = {
        "proposal_id": row.get("proposal_id", ""),
        "rationale":   row.get("rationale", ""),
        "proposer":    "llm_crafter" if is_llm else default_proposer,
        # PLAN-SKILL-CRAFTER §0.1 / §2.5 — the gate's Stage 0 static
        # check rejects any proposal whose target_domains ≠
        # ALL_FIVE_DOMAINS. The smoke originals only carry the
        # source domain, so we expand here to match the trainer's
        # offline-mirror contract.
        "target_domains": list(ALL_FIVE_DOMAINS),
        "adapter_plan":  _default_adapter_plan(domain),
        "proposal_kind": kind,
    }
    if kind == "patch":
        out["target_skill_id"] = row.get("base_skill_id", "")
        out["patch_kind"] = row.get("recovery_strategy") or "protocol_patch"
        out["evidence_role"] = _evidence_role_from_contract_blob(row.get("patched_contract"))
        out["seed_failure_ids"] = list(row.get("seed_failure_ids", []))
        if row.get("patched_protocol"):
            out["patched_protocol"] = list(row["patched_protocol"])
        if row.get("patched_contract") is not None:
            out["patched_contract"] = row["patched_contract"]
    elif kind == "retire":
        out["target_skill_id"] = row.get("target_skill_id", "")
        out["retire_reason"] = row.get("reason") or "evidence-starved"
        out["evidence_role"] = ""
    elif kind == "compose":
        out["components"] = list(row.get("component_skill_ids", []))
        out["compose_op"] = "sequence"
        out["evidence_role"] = _evidence_role_from_contract_blob(row.get("contract"))
        if row.get("composed_protocol"):
            out["composed_protocol"] = list(row["composed_protocol"])
        if row.get("contract") is not None:
            out["contract"] = row["contract"]
        if row.get("name"):
            out["name"] = row["name"]
    elif kind == "transfer":
        out["source_skill_id"] = row.get("base_skill_id", "")
        out["source_domain"] = row.get("source_domain") or "gymv"
        td = row.get("target_domain") or ""
        out["new_adapter_per_target"] = {td: True} if td else {}
        out["slot_remap_per_target"] = {td: dict(row.get("slot_remap", {}))} if td else {}
        out["evidence_role"] = _evidence_role_from_contract_blob(row.get("contract"))
        if row.get("abstracted_protocol"):
            out["abstracted_protocol"] = list(row["abstracted_protocol"])
        if row.get("contract") is not None:
            out["contract"] = row["contract"]
    elif kind == "hypothesize":
        out["new_skill_name"] = row.get("name", "")
        out["evidence_role"] = _evidence_role_from_contract_blob(row.get("contract"))
        if row.get("name"):
            out["name"] = row["name"]
        if row.get("novel_protocol"):
            out["novel_protocol"] = list(row["novel_protocol"])
        if row.get("contract") is not None:
            out["contract"] = row["contract"]
        if row.get("source_failure_pattern_ids"):
            out["source_failure_pattern_ids"] = list(row["source_failure_pattern_ids"])
    return out


def _evidence_role_from_contract_blob(c: Any) -> str:
    if not isinstance(c, dict):
        return ""
    roles = c.get("expected_evidence_roles") or []
    if not roles:
        return ""
    return str(roles[0]).upper()


# ---------------------------------------------------------------------------
# Smoke output → adapter root walker
# ---------------------------------------------------------------------------


def _walk_smoke_proposals(smoke_mode_dir: Path) -> List[Dict[str, Any]]:
    """Walk a single mode's smoke output dir and return all proposal rows.

    Smoke layout::

        <smoke_mode_dir>/<domain>/<benchmark>/<sample>/proposals.jsonl
        <smoke_mode_dir>/<domain>/<benchmark>/_cycle/proposals.jsonl

    We treat *all* proposals.jsonl files as eligible — including
    ``_cycle/`` which is where the cross-sample
    ``Hypothesizer.cycle()`` aggregation runs (and is responsible
    for almost all hypothesis proposals on the 50-sample VTB run).
    """
    rows: List[Dict[str, Any]] = []
    for f in sorted(smoke_mode_dir.rglob("proposals.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                logger.warning("skip malformed row in %s: %s", f, e)
    return rows


def _detect_llm_set(smoke_mode_dir: Path) -> set:
    """Walk ``audit.jsonl`` files (the ArtifactStore audit log
    sink wired into ``crafter._llm_runtime.install_llm_hooks`` from
    ``reflect_per_episode_gpt54.py``) and collect the
    ``proposal_id`` of every successful LLM call. Used to flip the
    ``proposer`` enum to ``llm_crafter`` for those rows so the
    gate's bookkeeping (``by_proposer``) reflects reality.

    The audit rows the LLM hooks emit have ``event ∈ {llm_repair,
    llm_hypothesize, llm_diagnose}`` and ``proposal_id`` set on
    the ``…_ok`` variants. We accept any audit row that carries
    ``proposal_id`` regardless of event name so this stays robust
    to future event-naming changes.
    """
    out: set = set()
    for f in smoke_mode_dir.rglob("audit.jsonl"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            pid = ev.get("proposal_id")
            if pid:
                out.add(pid)
    return out


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------


@dataclass
class ModePromotion:
    label: str
    n_translated: int = 0
    n_proposals_seen_by_gate: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    by_decision: Dict[str, int] = field(default_factory=dict)
    by_verdict: Dict[str, int] = field(default_factory=dict)
    by_target_status: Dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    n_llm_proposer: int = 0

    @property
    def promote_rate(self) -> float:
        n = self.n_proposals_seen_by_gate
        if n == 0:
            return 0.0
        return float(self.by_decision.get("PROMOTE", 0)) / n


def _run_promotion_for_mode(
    *,
    mode_label: str,
    smoke_mode_dir: Path,
    output_dir: Path,
    bank_run: Path,
    corpus: str,
    source: str,
    domain: str,
    gate_mode: str,
    teacher_model: str,
    judge_model: str,
) -> ModePromotion:
    rep = ModePromotion(label=mode_label)

    rows = _walk_smoke_proposals(smoke_mode_dir)
    if not rows:
        logger.info("[%s] no proposals to translate — skipping gate run", mode_label)
        return rep

    # The reflect script writes its ArtifactStore audit log to a
    # ``tempfile.mkdtemp(...)`` root that gets torn down when the
    # process exits, so the per-proposal LLM trail is not on disk
    # for us to inspect after the fact. Fall back to the mode label
    # as ground truth: ``lane_a_llm`` ⇒ Hypothesizer is LLM-backed,
    # so all HypothesisProposals from this mode are LLM-minted; same
    # logic for ``lane_b_llm`` ⇒ Repairer ⇒ PatchProposals.
    # ``rule_only`` keeps every row at the deterministic
    # proposer enum (composer / hypothesizer / reflector).
    is_llm_repairer = "lane_b" in mode_label.lower()
    is_llm_hypoth   = "lane_a" in mode_label.lower() or "lane_b" in mode_label.lower()

    # Translate to gate schema.
    adapter_root = output_dir / mode_label
    pair_dir = adapter_root / corpus / source
    pair_dir.mkdir(parents=True, exist_ok=True)
    out_path = pair_dir / "proposals.jsonl"
    n_llm = 0
    with out_path.open("w") as f:
        for row in rows:
            type_name = row.get("type", "")
            is_llm = (
                (is_llm_repairer and type_name == "PatchProposal")
                or (is_llm_hypoth and type_name == "HypothesisProposal")
            )
            if is_llm:
                n_llm += 1
            translated = _translate_row(row, domain=domain, is_llm=is_llm)
            f.write(json.dumps(translated) + "\n")
    rep.n_translated = len(rows)
    rep.n_llm_proposer = n_llm
    logger.info(
        "[%s] translated %d rows (%d LLM-minted) → %s",
        mode_label, len(rows), n_llm, out_path,
    )

    # Run the gate.
    gate_out = adapter_root / "_promotion"
    gate_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "labeling_supplement" / "decide_promotion_gpt54.py"),
        "--proposals-run", str(adapter_root),
        "--bank-run", str(bank_run),
        "--no-actions",
        "--gate-mode", gate_mode,
        "--corpus", corpus,
        "--source", source,
        "--output-dir", str(gate_out),
        "--teacher-model", teacher_model,
        "--judge-model", judge_model,
        "-v",
    ]
    logger.info("[%s] $ %s", mode_label, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    rep.elapsed_s = time.time() - t0
    (gate_out / "_gate_stdout.log").write_text(proc.stdout)
    (gate_out / "_gate_stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        logger.error("[%s] gate exited with %d — see %s",
                     mode_label, proc.returncode, gate_out / "_gate_stderr.log")
        return rep

    summary_path = gate_out / "_run_summary.json"
    if not summary_path.exists():
        logger.error("[%s] gate ran but no _run_summary.json (check stderr)", mode_label)
        return rep
    summary = json.loads(summary_path.read_text())
    rep.n_proposals_seen_by_gate = int(summary.get("n_proposals", 0))
    rep.by_kind = dict(summary.get("by_kind", {}))
    rep.by_decision = dict(summary.get("by_decision", {}))
    rep.by_verdict = dict(summary.get("by_verdict", {}))
    rep.by_target_status = dict(summary.get("by_target_status", {}))
    return rep


# ---------------------------------------------------------------------------
# Cross-mode summary
# ---------------------------------------------------------------------------


def _emit_summary(
    *,
    output_dir: Path,
    reports: List[ModePromotion],
    bank_run: Path,
    gate_mode: str,
    smoke_dir: Path,
    smoke_summary_blob: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    blob = {
        "smoke_dir":  str(smoke_dir),
        "bank_run":   str(bank_run),
        "gate_mode":  gate_mode,
        "completed_at": time.time(),
        "modes": [
            {
                "label":               r.label,
                "n_translated":        r.n_translated,
                "n_llm_proposer":      r.n_llm_proposer,
                "n_proposals_seen":    r.n_proposals_seen_by_gate,
                "by_kind":             r.by_kind,
                "by_decision":         r.by_decision,
                "by_verdict":          r.by_verdict,
                "by_target_status":    r.by_target_status,
                "promote_rate":        r.promote_rate,
                "elapsed_s":           r.elapsed_s,
            } for r in reports
        ],
    }
    js = output_dir / "_promotion_summary.json"
    js.write_text(json.dumps(blob, indent=2))

    # Also emit a markdown table that can be appended/embedded in
    # ``_attribution_summary.md`` from the smoke runner.
    lines: List[str] = []
    lines.append("# Promotion-gate attribution\n")
    lines.append(f"- smoke_dir: `{smoke_dir}`")
    lines.append(f"- bank_run: `{bank_run}`")
    lines.append(f"- gate_mode: `{gate_mode}`\n")
    lines.append(
        "| mode | n_translated | n_llm | n_seen_by_gate | "
        "PROMOTE | REJECT | DEFER | promote-rate |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in reports:
        lines.append(
            f"| `{r.label}` | {r.n_translated} | {r.n_llm_proposer} | "
            f"{r.n_proposals_seen_by_gate} | "
            f"{r.by_decision.get('PROMOTE', 0)} | "
            f"{r.by_decision.get('REJECT', 0)} | "
            f"{r.by_decision.get('DEFER', 0)} | "
            f"{r.promote_rate * 100:.1f}% |"
        )
    lines.append("\n## by-kind breakdown")
    lines.append("| mode | " + " | ".join(_collect_kinds(reports)) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(_collect_kinds(reports))) + "|")
    for r in reports:
        cells = [
            str(r.by_kind.get(k, 0))
            for k in _collect_kinds(reports)
        ]
        lines.append(f"| `{r.label}` | " + " | ".join(cells) + " |")
    md = output_dir / "_promotion_summary.md"
    md.write_text("\n".join(lines) + "\n")
    return js, md


def _collect_kinds(reports: List[ModePromotion]) -> List[str]:
    seen: Counter = Counter()
    for r in reports:
        for k, n in r.by_kind.items():
            seen[k] += n
    return sorted(seen.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--smoke-dir", type=Path, required=True,
        help="The output_dir of run_smoke_attribution (contains "
             "<mode>/visual_reasoning/visual_toolbench/...).",
    )
    p.add_argument(
        "--bank-run", type=Path, required=True,
        help="A skill_bank_out/run_<ts> snapshot whose "
             "<corpus>/<source>/skill_bank.jsonl matches the seed "
             "bank used by the smoke run.",
    )
    p.add_argument(
        "--corpus", default="env_wrappers", choices=("gym_v", "env_wrappers"),
        help="Corpus bucket to fold the proposals into (must exist "
             "under --bank-run). Default env_wrappers.",
    )
    p.add_argument(
        "--source", default="twenty_forty_eight",
        help="Source name within the corpus. Default twenty_forty_eight.",
    )
    p.add_argument(
        "--domain", default="gymv",
        help="Source domain string for the adapter_plan and the "
             "evidence-role default (default gymv since the seed "
             "bank lives there).",
    )
    p.add_argument(
        "--gate-mode", default="offline-synthetic",
        choices=("offline-synthetic", "live", "offline-with-llm-judge", "permissive"),
        help="Promotion gate mode. Default offline-synthetic (no LLM).",
    )
    p.add_argument(
        "--modes", nargs="+", default=None,
        help="Subset of mode labels to run (default: every "
             "subdirectory of --smoke-dir that contains a "
             "visual_reasoning/visual_toolbench tree).",
    )
    p.add_argument(
        "--teacher-model", default="gpt-5.4",
        help="Teacher model identifier logged to the gate's _run_meta.",
    )
    p.add_argument(
        "--judge-model", default="Qwen/Qwen3.5-35B-A3B",
        help="Judge model identifier (only invoked if --gate-mode "
             "is offline-with-llm-judge).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write the per-mode adapter trees + gate outputs "
             "+ cross-mode summary.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _discover_modes(smoke_dir: Path) -> List[str]:
    out: List[str] = []
    for sub in sorted(smoke_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if (sub / "visual_reasoning" / "visual_toolbench").exists():
            out.append(sub.name)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.smoke_dir.exists():
        logger.error("smoke-dir does not exist: %s", args.smoke_dir)
        return 1
    if not (args.bank_run / args.corpus / args.source / "skill_bank.jsonl").exists():
        logger.error(
            "bank-run is missing %s/%s/skill_bank.jsonl under %s",
            args.corpus, args.source, args.bank_run,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = args.modes or _discover_modes(args.smoke_dir)
    if not modes:
        logger.error("no modes discovered under %s", args.smoke_dir)
        return 1
    logger.info("running promotion gate for modes: %s", modes)

    reports: List[ModePromotion] = []
    for label in modes:
        smoke_mode_dir = args.smoke_dir / label
        rep = _run_promotion_for_mode(
            mode_label=label,
            smoke_mode_dir=smoke_mode_dir,
            output_dir=args.output_dir,
            bank_run=args.bank_run,
            corpus=args.corpus,
            source=args.source,
            domain=args.domain,
            gate_mode=args.gate_mode,
            teacher_model=args.teacher_model,
            judge_model=args.judge_model,
        )
        reports.append(rep)
        logger.info(
            "[%s] gate done: n_seen=%d  PROMOTE=%d  REJECT=%d  DEFER=%d  rate=%.1f%%  elapsed=%.1fs",
            label,
            rep.n_proposals_seen_by_gate,
            rep.by_decision.get("PROMOTE", 0),
            rep.by_decision.get("REJECT", 0),
            rep.by_decision.get("DEFER", 0),
            rep.promote_rate * 100,
            rep.elapsed_s,
        )

    js, md = _emit_summary(
        output_dir=args.output_dir,
        reports=reports,
        bank_run=args.bank_run,
        gate_mode=args.gate_mode,
        smoke_dir=args.smoke_dir,
    )
    print()
    print("=== promotion attribution summary ===")
    print(f"output_dir: {args.output_dir}")
    print(f"json:       {js}")
    print(f"markdown:   {md}\n")
    for r in reports:
        print(
            f"  {r.label:<14} n_seen={r.n_proposals_seen_by_gate:<3}  "
            f"PROMOTE={r.by_decision.get('PROMOTE', 0):<3}  "
            f"REJECT={r.by_decision.get('REJECT', 0):<3}  "
            f"DEFER={r.by_decision.get('DEFER', 0):<3}  "
            f"rate={r.promote_rate * 100:.1f}%  "
            f"by_kind={r.by_kind}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
