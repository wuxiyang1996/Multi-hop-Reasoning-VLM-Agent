#!/usr/bin/env python3
"""
Cognitive-signature-based skill clustering for cross-domain transfer.

Instead of LLM-based label merging, this extracts the cognitive verb
sequence from each skill's protocol steps, then clusters by signature.

Two-level hierarchy:
  Level 1: Cognitive loop type (reactive_execute, deliberate_select, ...)
  Level 2: Strategic intent (ATTACK, EVADE, CLEAR, ...) from skill_id

Output: mega_skill_clusters_v2.json + mega_skill_codebook_v2.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BANKS_DIR = ROOT / "frontier_data" / "output" / "per_task_banks"
OUT_DIR = ROOT / "frontier_data" / "output"

# ── Cognitive primitive mapping ──────────────────────────────────────

VERB_TO_PRIMITIVE = {}

_MAP = {
    "P": [  # Perceive
        "observe", "scan", "inspect", "identify", "locate", "detect",
        "recognize", "check", "monitor", "notice", "read", "note",
        "perceive", "look", "see", "watch", "view", "find",
    ],
    "R": [  # Retrieve (from memory / instruction)
        "retrieve", "recall", "bring", "fetch", "remember", "recollect",
    ],
    "E": [  # Evaluate / reason
        "evaluate", "assess", "compare", "determine", "estimate", "judge",
        "match", "eliminate", "exclude", "infer", "reason", "analyze",
        "weigh", "consider", "decide", "deduce",
    ],
    "S": [  # Select / filter
        "select", "choose", "keep", "pick", "filter", "narrow", "prefer",
    ],
    "X": [  # Execute / act
        "execute", "perform", "activate", "initiate", "trigger", "input",
        "move", "enter", "apply", "fire", "attack", "jump", "run",
        "place", "drop", "press", "click", "type", "drag", "scroll",
        "navigate", "advance", "proceed", "continue", "repeat",
    ],
    "C": [  # Confirm / verify
        "confirm", "verify", "validate", "wait", "maintain", "submit",
        "ensure", "assert", "stabilize", "hold",
    ],
    "T": [  # Transform state
        "remove", "achieve", "update", "assign", "set", "change",
        "clear", "swap", "merge", "transform", "modify", "replace",
    ],
    "O": [  # Output / respond
        "return", "respond", "output", "report", "answer", "produce",
    ],
}

for prim, verbs in _MAP.items():
    for v in verbs:
        VERB_TO_PRIMITIVE[v] = prim


def extract_signature(protocol, max_steps=4):
    """Extract cognitive primitive sequence from protocol steps."""
    if not isinstance(protocol, dict):
        return ""
    steps = protocol.get("steps", [])
    prims = []
    for step in steps[:max_steps]:
        if not isinstance(step, str):
            continue
        words = step.strip().split()
        if not words:
            continue
        verb = words[0].lower().rstrip(".,;:()")
        prim = VERB_TO_PRIMITIVE.get(verb, "?")
        if not prims or prims[-1] != prim:
            prims.append(prim)
    return "".join(prims)


def extract_intent(skill_id):
    """Extract strategic intent from VERB/VERB skill_id."""
    bare = skill_id.split(":")[-1] if ":" in skill_id else skill_id
    if "/" in bare:
        return bare.split("/", 1)[1].upper()
    if bare.startswith("archetype."):
        parts = bare.split(".")
        return parts[-1].upper() if len(parts) >= 3 else bare.upper()
    if bare.startswith("skill-"):
        return "GENERIC"
    return bare.upper()


# ── Cognitive loop family definitions ────────────────────────────────

LOOP_FAMILIES = {
    "reactive_execute": {
        "signatures": {"XC", "XCP", "XCPT", "XCT", "XPC", "XT", "XP",
                        "EXPT", "EXP", "EX", "EXC", "EXCT", "EXC?",
                        "X", "XC?T", "X?X"},
        "description": "Act first, then observe outcome. Reactive game loop.",
        "primary_domain": "GAME",
        "transferable_to": ["GAME"],
    },
    "deliberate_select": {
        "signatures": {"PSX", "PSXC", "PESX", "PSE", "PS?C", "PXSC", "P?SC",
                        "PSEC", "PS", "PSC", "PSX?", "PSCX", "PST", "PCSP",
                        "P?SX", "PES?"},
        "description": "Perceive options, select target, then execute. Shared by GAME (deliberate actions) and WEB (element interaction).",
        "primary_domain": "GAME+WEB",
        "transferable_to": ["GAME", "WEB"],
    },
    "inferential_reason": {
        "signatures": {"PES", "PESC", "PRES", "PRSE", "PRS", "PE", "PEC",
                        "PXES", "P?E", "PETC"},
        "description": "Observe evidence, evaluate hypotheses, select conclusion. Shared by VR (reasoning) and WEB (complex filtering).",
        "primary_domain": "VR+WEB",
        "transferable_to": ["VR", "WEB"],
    },
    "retrieve_match_act": {
        "signatures": {"RPSX", "RPX", "RPXC", "RPS", "RPS?", "RPES", "RPE",
                        "RPEC", "RSC", "RSCP", "RXSC", "R?ES", "RPEX",
                        "RXS", "RPS"},
        "description": "Recall instruction, match against observations, act. Instruction-following loop shared by WEB and VR.",
        "primary_domain": "WEB+VR",
        "transferable_to": ["WEB", "VR"],
    },
    "plan_transform": {
        "signatures": {"PTC", "PT", "PTP", "PTPC", "PXTC", "PTX", "PXET",
                        "P?CP", "PXCX", "PXC", "PX?C"},
        "description": "Analyze state, transform/update representation, confirm. State management shared by GAME (board state) and WEB (form fill).",
        "primary_domain": "GAME+WEB",
        "transferable_to": ["GAME", "WEB"],
    },
    "explore_monitor": {
        "signatures": {"XPX", "XPXC", "XPE", "X?P", "XPCX", "XP", "XPX",
                        "X?XP", "X?C"},
        "description": "Act, observe change, act again. Exploration loop.",
        "primary_domain": "GAME",
        "transferable_to": ["GAME"],
    },
    "filter_aggregate": {
        "signatures": {"PSO", "PSCO", "P?SO", "PEO", "PECO", "PSE"},
        "description": "Perceive items, filter/count, output result. Shared by VR (counting) and WEB (data extraction).",
        "primary_domain": "VR+WEB",
        "transferable_to": ["VR", "WEB"],
    },
    "sequence_chain": {
        "signatures": {"XCXC", "XCX", "XCXP", "CPC", "?C", "?CP", "?CX",
                        "?X?", "?C?", "C?PC", "C?", "C?C", "?CX?",
                        "?E?", "?E?S", "?S?", "?PS?"},
        "description": "Sequential step-by-step execution with confirmation.",
        "primary_domain": "GAME",
        "transferable_to": ["GAME"],
    },
}


def assign_loop_family(sig):
    """Assign a signature to the best-matching cognitive loop family."""
    for family, info in LOOP_FAMILIES.items():
        if sig in info["signatures"]:
            return family

    # Fuzzy match: find family with most character overlap
    best_family = None
    best_score = 0
    for family, info in LOOP_FAMILIES.items():
        for ref_sig in info["signatures"]:
            # Prefix match score
            common = 0
            for a, b in zip(sig, ref_sig):
                if a == b:
                    common += 1
                else:
                    break
            score = common / max(len(sig), len(ref_sig), 1)
            if score > best_score:
                best_score = score
                best_family = family
    if best_score >= 0.5:
        return best_family

    # Heuristic fallback based on primitive composition
    if not sig:
        return "sequence_chain"
    first = sig[0]
    if first == "X" or first == "E" and "X" in sig and sig.index("X") < len(sig) - 1:
        return "reactive_execute"
    if first == "R":
        return "retrieve_match_act"
    if first == "P":
        if "T" in sig:
            return "plan_transform"
        if "E" in sig and "S" in sig:
            return "inferential_reason"
        if "S" in sig and "X" in sig:
            return "deliberate_select"
        if "S" in sig:
            return "deliberate_select"
        if "X" in sig and "C" in sig:
            return "plan_transform"
        return "inferential_reason"
    if first == "?" or first == "C":
        if "E" in sig and "S" in sig:
            return "inferential_reason"
        return "sequence_chain"
    return "sequence_chain"


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Load all skill banks
    all_skills = []
    for task_dir in sorted(BANKS_DIR.iterdir()):
        bank_path = task_dir / "skill_bank.jsonl"
        if not bank_path.exists():
            continue
        task = task_dir.name
        with open(bank_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                sk = entry.get("skill", entry)
                all_skills.append((task, sk, entry))

    print(f"Loaded {len(all_skills)} skills from {BANKS_DIR}")
    print()

    # Extract signatures and assign families
    skill_records = []
    sig_counter = Counter()
    family_counter = Counter()

    for task, sk, entry in all_skills:
        sid = sk.get("skill_id", "")
        proto = sk.get("protocol", {})
        sig = extract_signature(proto)
        intent = extract_intent(sid)
        family = assign_loop_family(sig) if sig else "sequence_chain"

        # Determine domain
        if task.startswith("gymv_") or task.startswith("Temporal_"):
            domain = "GAME"
        elif task in ("candy_crush", "tetris", "twenty_forty_eight", "super_mario"):
            domain = "GAME"
        elif task in ("miniwob", "webshop"):
            domain = "WEB"
        else:
            domain = "VR"

        record = {
            "task": task,
            "skill_id": sid,
            "domain": domain,
            "signature": sig,
            "cognitive_family": family,
            "intent": intent,
            "full_key": f"{family}/{intent}",
        }
        skill_records.append(record)
        sig_counter[sig] += 1
        family_counter[family] += 1

    # Print family distribution
    print("=" * 70)
    print("COGNITIVE LOOP FAMILIES")
    print("=" * 70)
    for family, count in family_counter.most_common():
        info = LOOP_FAMILIES.get(family, {})
        desc = info.get("description", "")
        domain_dist = Counter()
        intent_dist = Counter()
        task_set = set()
        for r in skill_records:
            if r["cognitive_family"] == family:
                domain_dist[r["domain"]] += 1
                intent_dist[r["intent"]] += 1
                task_set.add(r["task"])

        doms = "+".join(sorted(domain_dist.keys()))
        dom_detail = ", ".join(f"{d}={c}" for d, c in domain_dist.most_common())
        n_way = len(domain_dist)
        cross = "★" if n_way >= 3 else ("●" if n_way >= 2 else " ")

        print(f"\n{cross} {family}  ({count} skills, {doms})")
        print(f"  {desc}")
        print(f"  Domains: {dom_detail}")
        print(f"  Tasks: {len(task_set)} unique")
        print(f"  Top intents: {', '.join(f'{i}({c})' for i, c in intent_dist.most_common(8))}")

    # Build output structure
    families_out = {}
    for family in sorted(set(r["cognitive_family"] for r in skill_records)):
        members = [r for r in skill_records if r["cognitive_family"] == family]
        domains = sorted(set(r["domain"] for r in members))
        tasks = sorted(set(r["task"] for r in members))

        # Sub-families by intent
        intent_groups = defaultdict(list)
        for r in members:
            intent_groups[r["intent"]].append(r)

        sub_families = {}
        for intent, group in sorted(intent_groups.items(), key=lambda x: -len(x[1])):
            sub_domains = sorted(set(r["domain"] for r in group))
            sub_tasks = sorted(set(r["task"] for r in group))
            sub_families[intent] = {
                "count": len(group),
                "domains": sub_domains,
                "tasks": sub_tasks,
                "skills": [
                    {"task": r["task"], "skill_id": r["skill_id"],
                     "domain": r["domain"], "signature": r["signature"]}
                    for r in group
                ],
            }

        families_out[family] = {
            "count": len(members),
            "domains": domains,
            "tasks": tasks,
            "description": LOOP_FAMILIES.get(family, {}).get("description", ""),
            "primary_domain": LOOP_FAMILIES.get(family, {}).get("primary_domain", ""),
            "sub_families": sub_families,
            "skills": [
                {"task": r["task"], "skill_id": r["skill_id"],
                 "domain": r["domain"], "signature": r["signature"],
                 "intent": r["intent"]}
                for r in members
            ],
        }

    output = {
        "total_skills": len(skill_records),
        "total_families": len(families_out),
        "total_sub_families": sum(
            len(f["sub_families"]) for f in families_out.values()
        ),
        "clustering_method": "cognitive_verb_signature",
        "cognitive_primitives": {
            "P": "Perceive (observe, scan, identify, locate, detect)",
            "R": "Retrieve (recall, fetch from memory/instruction)",
            "E": "Evaluate (assess, compare, match, eliminate, reason)",
            "S": "Select (choose, filter, pick)",
            "X": "Execute (perform, activate, move, input, fire)",
            "C": "Confirm (verify, wait, maintain, submit)",
            "T": "Transform (remove, update, assign, change)",
            "O": "Output (return, respond, report)",
        },
        "families": families_out,
    }

    out_path = OUT_DIR / "mega_skill_clusters_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nSaved → {out_path}")

    # Build codebook v2
    codebook = {}
    for family, info in families_out.items():
        codebook[family] = {
            "description": info["description"],
            "primary_domain": info["primary_domain"],
            "sub_families": list(info["sub_families"].keys()),
        }
    cb_path = OUT_DIR / "mega_skill_codebook_v2.json"
    with open(cb_path, "w") as f:
        json.dump(codebook, f, indent=2)
    print(f"Codebook → {cb_path}")

    # Cross-domain transfer summary
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN TRANSFER BRIDGES")
    print("=" * 70)

    # Level 1: families spanning multiple domains
    for family, info in sorted(families_out.items(), key=lambda x: -x[1]["count"]):
        domains = info["domains"]
        if len(domains) >= 2:
            dom_counts = Counter(r["domain"] for r in info["skills"])
            dom_str = ", ".join(f"{d}={c}" for d, c in dom_counts.most_common())
            transferable = LOOP_FAMILIES.get(family, {}).get("transferable_to", [])
            print(f"\n  {family} ({info['count']} skills, {'+'.join(domains)})")
            print(f"    Domain breakdown: {dom_str}")
            print(f"    Transferable to: {', '.join(transferable)}")

            # Level 2: sub-families spanning domains
            for intent, sub in sorted(
                info["sub_families"].items(), key=lambda x: -len(x[1]["domains"])
            ):
                if len(sub["domains"]) >= 2:
                    sub_doms = "+".join(sub["domains"])
                    tasks_short = ", ".join(sub["tasks"][:4])
                    print(f"    ├── {intent} [{sub['count']}sk, {sub_doms}]: {tasks_short}")

    # Transfer path summary
    print("\n" + "=" * 70)
    print("TRANSFER PATHS")
    print("=" * 70)
    paths = [
        ("GAME action → GAME action", "reactive_execute", "Same cognitive loop, different game context"),
        ("GAME deliberate → WEB", "deliberate_select", "PSXC signature shared: perceive→select→execute→confirm"),
        ("WEB reasoning → VR", "inferential_reason", "PES signature shared: perceive→evaluate→select"),
        ("WEB instruction → VR", "retrieve_match_act", "RPES signature shared: retrieve→perceive→evaluate→select"),
        ("GAME state → WEB form", "plan_transform", "PXC/PT signature shared: perceive→transform→confirm"),
    ]
    for path_name, family, reason in paths:
        info = families_out.get(family, {})
        count = info.get("count", 0)
        doms = "+".join(info.get("domains", []))
        print(f"  {path_name}")
        print(f"    via {family} ({count} skills, {doms})")
        print(f"    reason: {reason}")
        print()


if __name__ == "__main__":
    main()
