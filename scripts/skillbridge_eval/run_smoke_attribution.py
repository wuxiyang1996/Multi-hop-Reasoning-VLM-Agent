"""Smoke-attribution runner — does Crafter+LLM contribute to the bank?

Per the user's smoke design (chat 2026-05-06), this script answers ONE
question: *if we turn the Crafter LLM hooks on for a transfer-target
benchmark, do the resulting proposals (a) materialise into DRAFT
SkillRecords that (b) survive Stage-0 of the canonical gate, and
(c) cover failure-mode dimensions the rule path alone does not?*

It does NOT exercise the actor — that requires a vLLM endpoint, real
EligibilityFilter wiring, and an active eval loop, all of which are
multi-day setups. The purpose here is to establish the cheapest
**attribution** signal for the upstream proposal-quality gradient
*before* paying for the full actor smoke.

Three conditions are run in sequence (all on the same VTB sample
slice for parity; conditions vary only in the Crafter knobs):

  +---+----------------+--------------+--------------+
  | # | label          | LLM repairer | LLM hypoth.  |
  +---+----------------+--------------+--------------+
  | A | rule_only      | off          | off          |
  | B | lane_a_llm     | off          | ON           |
  | C | lane_b_llm     | ON           | ON           |
  +---+----------------+--------------+--------------+

Each condition writes its proposals to
``<output_dir>/<label>/<benchmark>/.../proposals.jsonl`` (existing
``reflect_per_episode_gpt54.py`` output layout). After each run the
proposals are walked, materialised back into ``SkillRecord``s, and
fed through ``GateService._run_static`` for a survival check.

The top-level ``attribution_summary.{json,md}`` reports the three
attribution signals from the chat:

  1. **Counts** of each proposal kind / recovery_strategy by mode.
  2. **Stage-0 survival rate** per mode (= fraction of DRAFT skills
     that pass the static gate). Below ~70% means the LLM is producing
     malformed proposals — a clear "do not promote until fixed" signal.
  3. **Net new strategies** per mode (= recovery_strategy values the
     LLM modes produced that the rule mode did not). The cheapest
     measure of "is the LLM adding value the rules can't?".

Usage::

    cd Multi-hop-Reasoning-VLM-Agent
    python -m scripts.skillbridge_eval.run_smoke_attribution \
        --samples-root Cold-start-out-visual-reasoning \
        --benchmarks visual_toolbench \
        --max-samples 50 \
        --output-dir labeling_supplement/episode_reflections_out/_smoke_attribution_v1 \
        --llm-model gpt-5.4
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
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
for p in (REPO_ROOT, WORKSPACE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logger = logging.getLogger("skillbridge_eval.run_smoke_attribution")


# ---------------------------------------------------------------------
# Mode definitions (= flag combos for reflect_per_episode_gpt54.py)
# ---------------------------------------------------------------------


@dataclass
class Mode:
    label: str
    llm_repairer: bool
    llm_hypothesizer: bool
    enable_protocol_patching: bool
    description: str

    def reflect_argv(
        self,
        *,
        samples_root: Path,
        benchmarks: List[str],
        max_samples: int,
        hot_threshold: int,
        hyp_min_recurrences: int,
        llm_model: str,
        seed_bank: Optional[Path],
        match_skill_by_token: bool,
        binding_jaccard_min: float,
        sample_ids_file: Optional[Path],
        output_dir: Path,
    ) -> List[str]:
        argv = [
            sys.executable,
            str(REPO_ROOT / "labeling_supplement" / "reflect_per_episode_gpt54.py"),
            "--domain", "visual_reasoning",
            "--samples-root", str(samples_root),
            "--benchmarks", *benchmarks,
            "--max-samples-per-benchmark", str(max_samples),
            "--hot-pattern-threshold", str(hot_threshold),
            "--hypothesize-min-recurrences", str(hyp_min_recurrences),
            "--llm-model", llm_model,
            "--output-dir", str(output_dir),
            "-v",
        ]
        if self.enable_protocol_patching:
            argv.append("--enable-protocol-patching")
        if self.llm_repairer:
            argv.append("--llm-repairer")
        if self.llm_hypothesizer:
            argv.append("--llm-hypothesizer")
        if seed_bank is not None:
            argv += ["--seed-bank", str(seed_bank)]
        if match_skill_by_token:
            argv.append("--match-skill-by-token")
            argv += ["--binding-jaccard-min", str(binding_jaccard_min)]
        if sample_ids_file is not None:
            argv += ["--sample-ids-file", str(sample_ids_file)]
        return argv


MODES: Dict[str, Mode] = {
    "rule_only": Mode(
        label="rule_only",
        llm_repairer=False,
        llm_hypothesizer=False,
        enable_protocol_patching=False,
        description=(
            "Baseline. Pure deterministic Crafter rule path. No LLM "
            "calls. Lane-(a) only (no Repairer)."
        ),
    ),
    "lane_a_llm": Mode(
        label="lane_a_llm",
        llm_repairer=False,
        llm_hypothesizer=True,
        enable_protocol_patching=False,
        description=(
            "Lane-(a) + LLM Hypothesizer. Tests whether gpt-5.4 "
            "produces better novel-skill proposals than the rule "
            "VERIFY+COMMIT template."
        ),
    ),
    "lane_b_llm": Mode(
        label="lane_b_llm",
        llm_repairer=True,
        llm_hypothesizer=True,
        enable_protocol_patching=True,
        description=(
            "Lane-(b) + LLM Repairer + LLM Hypothesizer. The full "
            "stack: Repairer can mint PatchProposal records and the "
            "LLM rewrites both protocols and contracts. Requires "
            "--seed-bank (or --match-skill-by-token won't have a base "
            "to attach to)."
        ),
    ),
}


# ---------------------------------------------------------------------
# Stage-0 survival check (reimplements GateService._run_static logic
# inline so we don't have to spin up an Orchestrator).
# ---------------------------------------------------------------------


@dataclass
class Stage0Result:
    proposal_id: str
    skill_id: str
    proposal_type: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    recovery_strategy: Optional[str] = None
    feasible_domains: List[str] = field(default_factory=list)
    n_protocol_steps: int = 0


def _stage0_check(
    proposal_blob: Dict[str, Any],
    *,
    domains_set: set,
    evidence_roles_set: set,
) -> Stage0Result:
    """Replicate the survival rules of ``GateService._run_static``.

    We do NOT re-instantiate a SkillRecord here because the persisted
    JSON has already been through the lifecycle's static schema check
    (``__post_init__``). The remaining survival-relevant rules are:

      * non-empty protocol (unless the proposal is a Retire or Compose
        with explicit components),
      * non-empty ``expected_evidence_roles`` for non-action skills,
      * every ``feasible_domains`` entry in canonical DOMAINS,
      * lineage check for Patch / Generalize / Rewrite (base_skill_id
        non-empty).

    Stage-0's ``source_type`` mismatch check is omitted because the
    persisted proposal blob doesn't carry the materialised
    ``SkillRecord``'s source_type in the same row — that pairing
    happens in the live promotion path, not at proposal time.
    """
    failures: List[str] = []
    ptype = proposal_blob.get("type") or "Unknown"
    pid = proposal_blob.get("proposal_id") or ""
    rec_strat = proposal_blob.get("recovery_strategy")

    # The persisted proposal dict carries different protocol field
    # names depending on type (composed_protocol / abstracted_protocol
    # / novel_protocol / patched_protocol).
    PROTOCOL_FIELDS = (
        "composed_protocol", "abstracted_protocol",
        "novel_protocol", "patched_protocol",
    )
    proto_blob: List[Dict[str, Any]] = []
    for f in PROTOCOL_FIELDS:
        v = proposal_blob.get(f)
        if isinstance(v, list) and v:
            proto_blob = v
            break
    n_steps = len(proto_blob)

    contract_blob = (
        proposal_blob.get("contract")
        or proposal_blob.get("patched_contract")
        or {}
    )
    if not isinstance(contract_blob, dict):
        contract_blob = {}
    domains = list(proposal_blob.get("target_domains") or [])

    # 1. domains canonical
    for d in domains:
        if d not in domains_set:
            failures.append(f"unknown_domain={d!r}")

    # 2. protocol non-empty (Retire is exempt)
    if ptype != "RetireProposal" and n_steps == 0:
        failures.append("skill.protocol is empty")

    # 3. expected_evidence_roles non-empty for non-action skills.
    # The persisted proposal does not carry skill_type, so we
    # conservatively require ≥1 role for non-Retire types unless the
    # protocol is purely EXECUTE/COMMIT (= action-only). Mirror Stage-0's
    # "skill.skill_type.value != 'action'" carve-out by checking action verbs.
    if ptype != "RetireProposal":
        roles = list(contract_blob.get("expected_evidence_roles") or [])
        action_verbs = {(s.get("action") or "").upper() for s in proto_blob}
        is_pure_action = bool(action_verbs) and action_verbs.issubset({"EXECUTE", "COMMIT", "RETRY"})
        if not roles and not is_pure_action:
            failures.append("contract.expected_evidence_roles empty (G0)")
        for r in roles:
            if r not in evidence_roles_set:
                failures.append(f"unknown_evidence_role={r!r}")

    # 4. lineage (Patch/Generalize need base_skill_id; Compose needs components)
    if ptype in ("PatchProposal", "GeneralizeProposal", "RewriteProposal"):
        if not (proposal_blob.get("base_skill_id") or ""):
            failures.append("base_skill_id is empty")
    if ptype == "ComposeProposal":
        if not (proposal_blob.get("component_skill_ids") or []):
            failures.append("ComposeProposal.component_skill_ids is empty")

    return Stage0Result(
        proposal_id=str(pid),
        skill_id=str(proposal_blob.get("base_skill_id") or ""),
        proposal_type=str(ptype),
        passed=not failures,
        failures=failures,
        recovery_strategy=str(rec_strat) if rec_strat else None,
        feasible_domains=domains,
        n_protocol_steps=n_steps,
    )


# ---------------------------------------------------------------------
# Mode runner
# ---------------------------------------------------------------------


@dataclass
class ModeReport:
    label: str
    description: str
    elapsed_sec: float
    n_proposals: int
    n_passed_stage0: int
    n_failed_stage0: int
    by_kind: Counter = field(default_factory=Counter)
    by_proposer: Counter = field(default_factory=Counter)
    by_recovery_strategy: Counter = field(default_factory=Counter)
    failure_reasons: Counter = field(default_factory=Counter)
    stage0_pass_rate: float = 0.0
    # Diversity signal — # of distinct skill names + # of distinct
    # protocol-shape hashes the proposer minted. Rule path produces
    # boilerplate (n_distinct_*=1 regardless of n_proposals); LLM
    # path should produce > 1.
    n_distinct_names: int = 0
    n_distinct_protocols: int = 0
    avg_protocol_steps: float = 0.0
    sample_names: List[str] = field(default_factory=list)
    teacher_models: Counter = field(default_factory=Counter)
    cmd: List[str] = field(default_factory=list)
    output_dir: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "elapsed_sec": self.elapsed_sec,
            "n_proposals": self.n_proposals,
            "n_passed_stage0": self.n_passed_stage0,
            "n_failed_stage0": self.n_failed_stage0,
            "stage0_pass_rate": self.stage0_pass_rate,
            "n_distinct_names": self.n_distinct_names,
            "n_distinct_protocols": self.n_distinct_protocols,
            "avg_protocol_steps": self.avg_protocol_steps,
            "sample_names": self.sample_names,
            "teacher_models": dict(self.teacher_models),
            "by_kind": dict(self.by_kind),
            "by_proposer": dict(self.by_proposer),
            "by_recovery_strategy": dict(self.by_recovery_strategy),
            "stage0_failure_reasons": dict(self.failure_reasons),
            "cmd": self.cmd,
            "output_dir": self.output_dir,
        }


def _walk_proposals(out_root: Path) -> List[Dict[str, Any]]:
    """Walk ``proposals.jsonl`` files under a reflect-run output dir."""
    blobs: List[Dict[str, Any]] = []
    for p in out_root.rglob("proposals.jsonl"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    blobs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception as exc:                                       # noqa: BLE001
            logger.warning("could not read %s: %s", p, exc)
    return blobs


def _run_mode(
    *,
    mode: Mode,
    samples_root: Path,
    benchmarks: List[str],
    max_samples: int,
    hot_threshold: int,
    hyp_min_recurrences: int,
    llm_model: str,
    seed_bank: Optional[Path],
    match_skill_by_token: bool,
    binding_jaccard_min: float,
    sample_ids_file: Optional[Path],
    output_dir: Path,
    domains_set: set,
    evidence_roles_set: set,
) -> ModeReport:
    mode_dir = output_dir / mode.label
    mode_dir.mkdir(parents=True, exist_ok=True)

    cmd = mode.reflect_argv(
        samples_root=samples_root,
        benchmarks=benchmarks,
        max_samples=max_samples,
        hot_threshold=hot_threshold,
        hyp_min_recurrences=hyp_min_recurrences,
        llm_model=llm_model,
        seed_bank=seed_bank,
        match_skill_by_token=match_skill_by_token,
        binding_jaccard_min=binding_jaccard_min,
        sample_ids_file=sample_ids_file,
        output_dir=mode_dir,
    )
    logger.info("mode=%s -> %s", mode.label, " ".join(cmd))

    t0 = time.time()
    log_path = mode_dir / "_reflect_stdout.log"
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.warning("mode=%s reflect rc=%s (see %s)",
                       mode.label, proc.returncode, log_path)

    blobs = _walk_proposals(mode_dir)

    report = ModeReport(
        label=mode.label,
        description=mode.description,
        elapsed_sec=round(elapsed, 2),
        n_proposals=len(blobs),
        n_passed_stage0=0,
        n_failed_stage0=0,
        cmd=cmd,
        output_dir=str(mode_dir),
    )
    distinct_names: set = set()
    proto_shapes: set = set()
    total_steps = 0
    for blob in blobs:
        distinct_names.add(str(blob.get("name") or ""))
        # Protocol-shape hash: tuple of action verbs (ignoring payload).
        for f in ("composed_protocol", "abstracted_protocol",
                  "novel_protocol", "patched_protocol"):
            v = blob.get(f)
            if isinstance(v, list) and v:
                shape = tuple((str(s.get("action") or "")).upper() for s in v)
                proto_shapes.add(shape)
                total_steps += len(v)
                break
        teacher = blob.get("teacher_model")
        if teacher:
            report.teacher_models[str(teacher)] += 1
    report.n_distinct_names = len(distinct_names)
    report.n_distinct_protocols = len(proto_shapes)
    if blobs:
        report.avg_protocol_steps = round(total_steps / len(blobs), 2)
    report.sample_names = sorted(n for n in distinct_names if n)[:8]

    for blob in blobs:
        report.by_kind[str(blob.get("type") or "Unknown")] += 1
        # Proposer label mirrors reflect_per_episode_gpt54._infer_proposer.
        kind = blob.get("type") or ""
        proposer = {
            "ComposeProposal":     "composer",
            "GeneralizeProposal":  "generalizer",
            "HypothesisProposal":  "hypothesizer",
            "PatchProposal":       "reflector",
            "RetireProposal":      "reflector",
            "RewriteProposal":     "rewriter",
        }.get(kind, "unknown")
        report.by_proposer[proposer] += 1
        if blob.get("recovery_strategy"):
            report.by_recovery_strategy[str(blob["recovery_strategy"])] += 1
        result = _stage0_check(
            blob,
            domains_set=domains_set,
            evidence_roles_set=evidence_roles_set,
        )
        if result.passed:
            report.n_passed_stage0 += 1
        else:
            report.n_failed_stage0 += 1
            for f in result.failures:
                # Bucket "unknown_*" by family so the failure report
                # doesn't fragment on the value embedded in the message.
                if "unknown_domain" in f:
                    report.failure_reasons["unknown_domain"] += 1
                elif "unknown_evidence_role" in f:
                    report.failure_reasons["unknown_evidence_role"] += 1
                else:
                    report.failure_reasons[f] += 1
    if report.n_proposals:
        report.stage0_pass_rate = round(
            report.n_passed_stage0 / report.n_proposals, 4
        )
    (mode_dir / "_attribution.json").write_text(
        json.dumps(report.to_json(), indent=2, sort_keys=True)
    )
    return report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--samples-root", type=Path,
        default=REPO_ROOT / "Cold-start-out-visual-reasoning",
        help="Cold-start root holding per-benchmark sample dirs.",
    )
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["visual_toolbench"],
        help="Benchmarks to include (default: visual_toolbench only — "
             "cheapest smoke).",
    )
    p.add_argument(
        "--max-samples", type=int, default=50,
        help="Per-benchmark cap (default 50; ~10 min @ gpt-5.4).",
    )
    p.add_argument(
        "--sample-ids-file", type=Path, default=None,
        help="Optional sample-id manifest (e.g. "
             "cold_start/evaluation_dataset/pool/visual_toolbench.txt).",
    )
    p.add_argument(
        "--seed-bank", type=Path, default=None,
        help="Optional skill_bank.jsonl to seed as CANDIDATE before "
             "reflect runs. Required for the C/lane_b_llm condition's "
             "Repairer to fire (the patch path needs a base skill).",
    )
    p.add_argument(
        "--match-skill-by-token", action="store_true",
        help="Token-Jaccard nearest-neighbor binding (passed to reflect "
             "to attach base skill_ids to the failures).",
    )
    p.add_argument(
        "--binding-jaccard-min", type=float, default=0.05,
        help="Minimum Jaccard for token-binding (only used when "
             "--match-skill-by-token is set; default 0.05). Pair with "
             "--seed-bank or no skills will be available to bind to.",
    )
    p.add_argument(
        "--llm-model", default="gpt-5.4",
        help="LLM to use for the Repairer / Hypothesizer hooks "
             "(default gpt-5.4 via OpenRouter).",
    )
    p.add_argument(
        "--hot-pattern-threshold", type=int, default=2,
        help="Per-batch hot-pattern threshold for cycle() (default 2 "
             "— small for smoke; default in production is 3).",
    )
    p.add_argument(
        "--hypothesize-min-recurrences", type=int, default=2,
        help="Hypothesizer recurrence gate (default 2 — match smoke "
             "size; production default is 3).",
    )
    p.add_argument(
        "--modes", nargs="+", choices=list(MODES.keys()),
        default=list(MODES.keys()),
        help="Subset of modes to run (default: all three).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "labeling_supplement" / "episode_reflections_out"
                / f"_smoke_attribution_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Output dir; one subdir per mode + a top-level summary.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _emit_markdown(summary: Dict[str, Any], path: Path) -> None:
    md_lines = [
        "# Crafter smoke-attribution",
        "",
        f"- samples-root: `{summary['samples_root']}`",
        f"- benchmarks  : {', '.join(summary['benchmarks'])}",
        f"- max samples : {summary['max_samples']}",
        f"- llm model   : `{summary['llm_model']}`",
        f"- seed bank   : `{summary['seed_bank'] or '(none)'}`",
        f"- match token : {summary['match_skill_by_token']}",
        "",
        "## Per-mode rollup",
        "",
        "| Mode | n_prop | Stage-0 pass | distinct names | distinct protocol shapes | avg steps | by kind | by recovery_strategy |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in summary["per_mode"]:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(r["by_kind"].items())) or "—"
        recs = ", ".join(f"{k}={v}" for k, v in sorted(r["by_recovery_strategy"].items())) or "—"
        pass_str = (
            f"{r['n_passed_stage0']}/{r['n_proposals']} ({r['stage0_pass_rate']:.0%})"
            if r["n_proposals"] else "0/0 (—)"
        )
        md_lines.append(
            f"| `{r['label']}` | {r['n_proposals']} | {pass_str} "
            f"| {r['n_distinct_names']} | {r['n_distinct_protocols']} "
            f"| {r['avg_protocol_steps']} | {kinds} | {recs} |"
        )

    md_lines += [
        "",
        "## Sample skill names per mode (cheapest qualitative-diversity signal)",
        "",
    ]
    for r in summary["per_mode"]:
        names = ", ".join(f"`{n}`" for n in r["sample_names"]) or "_none_"
        md_lines.append(f"- **{r['label']}** ({r['n_distinct_names']} distinct): {names}")

    md_lines += [
        "",
        "## Net new recovery_strategy values added by LLM modes",
        "",
    ]
    rule = next(
        (r for r in summary["per_mode"] if r["label"] == "rule_only"),
        None,
    )
    rule_recs = set((rule or {}).get("by_recovery_strategy", {}).keys())
    for r in summary["per_mode"]:
        if r["label"] == "rule_only":
            continue
        new = set(r["by_recovery_strategy"].keys()) - rule_recs
        md_lines.append(
            f"- **{r['label']}**: " +
            (", ".join(sorted(new)) if new else "_none — same coverage as rule_only_")
        )

    md_lines += [
        "",
        "## Stage-0 failure reasons (any mode)",
        "",
    ]
    any_failures = False
    for r in summary["per_mode"]:
        for k, v in sorted(r["stage0_failure_reasons"].items()):
            md_lines.append(f"- `{r['label']}` · `{k}`: {v}")
            any_failures = True
    if not any_failures:
        md_lines.append("_none — all proposals survive Stage-0._")
    path.write_text("\n".join(md_lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    samples_root: Path = args.samples_root.resolve()
    if not samples_root.is_dir():
        logger.error("samples-root not a directory: %s", samples_root)
        return 2
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-import canonical DOMAINS / EVIDENCE_ROLES once for Stage-0.
    from common.enums import DOMAINS, EVIDENCE_ROLES
    domains_set = set(DOMAINS)
    evidence_roles_set = set(EVIDENCE_ROLES)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info("smoke attribution: %d mode(s), %d benchmark(s), %d sample(s) each",
                len(args.modes), len(args.benchmarks), args.max_samples)
    logger.info("  output_dir: %s", output_dir)

    reports: List[ModeReport] = []
    for mode_label in args.modes:
        mode = MODES[mode_label]
        rep = _run_mode(
            mode=mode,
            samples_root=samples_root,
            benchmarks=list(args.benchmarks),
            max_samples=args.max_samples,
            hot_threshold=args.hot_pattern_threshold,
            hyp_min_recurrences=args.hypothesize_min_recurrences,
            llm_model=args.llm_model,
            seed_bank=args.seed_bank,
            match_skill_by_token=args.match_skill_by_token,
            binding_jaccard_min=args.binding_jaccard_min,
            sample_ids_file=args.sample_ids_file,
            output_dir=output_dir,
            domains_set=domains_set,
            evidence_roles_set=evidence_roles_set,
        )
        reports.append(rep)
        logger.info(
            "  %-12s -> %d proposals (%d pass, %d fail @ Stage-0; %.1fs)",
            mode_label, rep.n_proposals, rep.n_passed_stage0,
            rep.n_failed_stage0, rep.elapsed_sec,
        )

    summary = {
        "schema": "smoke_attribution_v1",
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples_root": str(samples_root),
        "benchmarks": list(args.benchmarks),
        "max_samples": args.max_samples,
        "sample_ids_file": str(args.sample_ids_file) if args.sample_ids_file else None,
        "seed_bank": str(args.seed_bank) if args.seed_bank else None,
        "match_skill_by_token": bool(args.match_skill_by_token),
        "llm_model": args.llm_model,
        "hot_pattern_threshold": args.hot_pattern_threshold,
        "hypothesize_min_recurrences": args.hypothesize_min_recurrences,
        "per_mode": [r.to_json() for r in reports],
    }
    summary_path = output_dir / "_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    md_path = output_dir / "_attribution_summary.md"
    _emit_markdown(summary, md_path)

    print()
    print(f"=== smoke attribution summary ===")
    print(f"output_dir: {output_dir}")
    print(f"summary:    {summary_path}")
    print(f"markdown:   {md_path}")
    print()
    for r in reports:
        pr = (r.stage0_pass_rate * 100) if r.n_proposals else 0.0
        print(f"  {r.label:<12s}  n={r.n_proposals:<3d}  pass={pr:5.1f}%  "
              f"distinct_names={r.n_distinct_names}  "
              f"distinct_proto_shapes={r.n_distinct_protocols}  "
              f"avg_steps={r.avg_protocol_steps}  "
              f"kinds={dict(r.by_kind)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
