#!/usr/bin/env python3
"""Collect development-only TIR active-perception receipts for grounder V3.

V2 showed genuine crop headroom, but its 768-pixel overview made most target
interventions behaviorally inert and its endpoint confidence was almost always
0.95--1.00 even when wrong.  This development-only collector makes those two
grounding variables explicit while leaving the source program completely
unavailable to every neural call:

* the baseline gets a 384-pixel contextual overview;
* each candidate follows the same context-to-local 1/4/8 zoom schedule; and
* answer calls also return outcome-blind evidence-quality measurements.

Qualification and formal stages are deliberately rejected.  A separate
freezer may authorize them only after a fixed artifact passes the V3 held-out
development gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS  # noqa: E402


def _load_v2_collector():
    path = REPO / "scripts/collect_phase3_tir_visual_search_v2.py"
    spec = importlib.util.spec_from_file_location("phase3_tir_visual_search_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load TIR visual-search V2 collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_v2_collector()
BASE = V2.BASE
ANSWER_SLOTS = tuple("ABCDEF")
OVERVIEW_MAX_SIDE = 384
PROPOSAL_MODEL_OVERRIDE: str | None = None
ANSWER_MODEL_OVERRIDE: str | None = None
BASELINE_MODEL_OVERRIDE: str | None = None
ENDPOINT_MODEL_OVERRIDE: str | None = None
BASELINE_VERIFIER_MODEL_OVERRIDE: str | None = None
COLLECTION_ROLE = "development"
ENDPOINT_OVERVIEW_POLICY = "context"


def _clip_probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    output = float(value)
    if not 0.0 <= output <= 1.0:
        raise ValueError(f"{label} is outside [0,1]")
    return output


def _expand_box(
    box: Sequence[float], *, scale: float, dx: float = 0.0, dy: float = 0.0,
) -> list[float]:
    x, y, width, height = map(float, box)
    cx = x + width / 2.0 + dx * width
    cy = y + height / 2.0 + dy * height
    width = min(1.0, max(0.04, width * scale))
    height = min(1.0, max(0.04, height * scale))
    left = min(max(0.0, cx - width / 2.0), 1.0 - width)
    top = min(max(0.0, cy - height / 2.0), 1.0 - height)
    return [round(left, 6), round(top, 6), round(width, 6), round(height, 6)]


def expand_neural_anchor_v3(
    anchor: Sequence[float], *, planner_score: float,
    raw_effects: Mapping[str, float], hypothesis: str,
) -> dict[str, Any]:
    """Use one identical causal schedule for every opaque neural operand.

    Transition 1 preserves context, transition 4 reaches the proposed local
    target, and transitions 5--8 test neighboring evidence.  This makes the
    source-induced H1/H4/H8 types refer to actual evidence-acquisition timing,
    not merely three repeated reads of the same crop.
    """

    transforms = (
        (3.20, 0.00, 0.00),
        (2.20, 0.00, 0.00),
        (1.50, 0.00, 0.00),
        (1.00, 0.00, 0.00),
        (1.10, -0.65, 0.00),
        (1.10, 0.65, 0.00),
        (1.10, 0.00, -0.65),
        (1.10, 0.00, 0.65),
    )
    actions = [
        {
            "tool": "zoom_region",
            "normalized_box": _expand_box(
                anchor, scale=scale, dx=dx, dy=dy,
            ),
        }
        for scale, dx, dy in transforms
    ]
    body = {
        "schema_version": "phase3-tir-neural-anchor-program-v3",
        "actions": actions,
        "planner_score": float(planner_score),
        "raw_typed_effect_probabilities": {
            name: float(raw_effects[name]) for name in TYPED_EFFECTS
        },
        "target_hypothesis": str(hypothesis),
    }
    return body | {"candidate_id": stable_hash({
        "schema_version": body["schema_version"],
        "actions": actions,
    })}


def _propose_programs_v3(
    client: OpenAI, *, model: str, prompt: str, overview: bytes,
    routing: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = PROPOSAL_MODEL_OVERRIDE or model
    system = (
        "You are a target-native active-vision grounder. Do not answer the "
        "multiple-choice question and do not assume which option is correct. "
        "Propose exactly four distinct normalized regions that test four "
        "different plausible visual hypotheses, referents, or distractors. "
        "The regions must be meaningfully different; do not return four crops "
        "of the same object. Return JSON {\"anchors\":[{\"x\":0..1,\"y\":"
        "0..1,\"w\":0..1,\"h\":0..1,\"planner_score\":0..1,"
        "\"hypothesis\":string,\"typed_effect_probabilities\":{"
        "\"EFFECT_BY_TRANSITION_1\":0..1,\"EFFECT_BY_TRANSITION_4\":0..1,"
        "\"EFFECT_BY_TRANSITION_8\":0..1,"
        "\"EXECUTABLE_TRANSITION_PERSISTENCE\":0..1}}]}. Coordinates are "
        "fractions of the whole image. Effects estimate whether this region's "
        "context crop (H1), focused crop (H4), extended neighboring evidence "
        "(H8), or executable refinement persistence will resolve the question."
    )
    content = [
        {
            "type": "text",
            "text": "Question: " + prompt + "\nRouting: "
            + json.dumps(routing, ensure_ascii=False),
        },
        BASE._image_content(overview),
    ]

    def parse(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        anchors = payload.get("anchors") or ()
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            raise ValueError("neural anchor response omitted anchors")
        if len(anchors) != 4:
            raise ValueError("neural grounder did not return exactly four anchors")
        output = []
        keys = set()
        for row in anchors:
            box = [float(row.get(key, -1)) for key in ("x", "y", "w", "h")]
            x, y, width, height = box
            if (
                x < 0 or y < 0 or width < 0.04 or height < 0.04
                or x + width > 1.000001 or y + height > 1.000001
            ):
                raise ValueError(f"neural anchor is outside image: {box}")
            key = tuple(round(value, 5) for value in box)
            if key in keys:
                raise ValueError("neural grounder returned duplicate anchors")
            keys.add(key)
            score = _clip_probability(row.get("planner_score"), label="planner_score")
            effects = row.get("typed_effect_probabilities") or {}
            if set(effects) != set(TYPED_EFFECTS):
                raise ValueError("neural anchor typed-effect schema mismatch")
            checked = {
                name: _clip_probability(effects[name], label=name)
                for name in TYPED_EFFECTS
            }
            output.append(expand_neural_anchor_v3(
                box, planner_score=score, raw_effects=checked,
                hypothesis=str(row.get("hypothesis") or ""),
            ))
        return output

    usages = []
    errors = []
    for attempt in range(2):
        repair = ""
        if attempt:
            repair = (
                " Schema-repair retry: " + errors[-1]
                + ". Return all required fields and four distinct valid boxes."
            )
        try:
            payload, usage = BASE._json_call(
                client, model=model, system=system + repair,
                content=content, max_tokens=4000,
            )
            usages.append(usage)
            programs = parse(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt == 0:
                continue
            raise
        return programs, {
            "calls": usages,
            "schema_repair_attempts": attempt,
            "schema_errors": errors,
        }
    raise RuntimeError("unreachable V3 proposal repair state")


def _answer_v3(
    client: OpenAI, *, model: str, prompt: str, overview: bytes,
    evidence: bytes | None, evidence_receipts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = (
        (ENDPOINT_MODEL_OVERRIDE if evidence is not None else BASELINE_MODEL_OVERRIDE)
        or ANSWER_MODEL_OVERRIDE or model
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Solve this TIR multiple-choice visual-reasoning task. Return "
                "probability for every A--F slot, using near-zero mass for "
                "absent choices. The overview is deliberately low-resolution "
                "context. Question: " + prompt
            ),
        },
    ]
    if evidence is None or ENDPOINT_OVERVIEW_POLICY == "context":
        content.extend([
            {"type": "text", "text": "Low-resolution contextual overview:"},
            BASE._image_content(overview),
        ])
    else:
        content.append({
            "type": "text",
            "text": (
                "The overview is not repeated at this endpoint. Ground only on "
                "the evidence produced by the executed region program below."
            ),
        })
    if evidence is not None:
        compact = [
            {
                "transition": row["transition"],
                "normalized_box": row["action"]["normalized_box"],
                "nonredundant": row["effect"]["nonredundant"],
            }
            for row in evidence_receipts
        ]
        content.extend([
            {
                "type": "text",
                "text": (
                    "Executed region sequence: "
                    + json.dumps(compact, ensure_ascii=False)
                    + "\nAccumulated regional-evidence collage:"
                ),
            },
            BASE._image_content(evidence),
        ])
    schema = (
        "Return concise JSON {\"answer\":\"A-F\",\"probabilities\":{"
        "\"A\":number,...,\"F\":number},\"reason\":\"brief grounded reason\","
        "\"evidence_quality\":{\"referent_visible\":0..1,"
        "\"local_detail_sufficient\":0..1,\"question_coverage\":0..1,"
        "\"contradiction_risk\":0..1}}. Evidence-quality values assess only "
        "what is visibly supplied; they are not confidence that an answer key "
        "matches. Never claim evidence not supplied."
    )
    usages = []
    errors = []
    for attempt in range(2):
        repair = ""
        if attempt:
            repair = " Schema-repair retry: " + errors[-1]
        try:
            payload, usage = BASE._json_call(
                client, model=model, system=schema + repair,
                content=content, max_tokens=4000,
            )
            usages.append(usage)
            probabilities = BASE.normalized_probabilities(
                payload.get("probabilities") or {}
            )
            answer = str(payload.get("answer") or "").strip().upper()[:1]
            if answer not in ANSWER_SLOTS:
                raise ValueError("answer is not A--F")
            quality = payload.get("evidence_quality") or {}
            expected = {
                "referent_visible", "local_detail_sufficient",
                "question_coverage", "contradiction_risk",
            }
            if set(quality) != expected:
                raise ValueError("evidence_quality schema mismatch")
            checked = {
                key: _clip_probability(quality[key], label=key)
                for key in sorted(expected)
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt == 0:
                continue
            raise
        return {
            "answer": answer,
            "probabilities": {
                slot: float(value)
                for slot, value in zip(ANSWER_SLOTS, probabilities)
            },
            "reason": str(payload.get("reason") or ""),
            "evidence_quality": checked,
        }, {
            "calls": usages,
            "schema_repair_attempts": attempt,
            "schema_errors": errors,
        }
    raise RuntimeError("unreachable V3 answer repair state")


def _verify_baseline_v3(
    client: OpenAI, *, model: str, prompt: str, overview: bytes,
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently assess whether the overview supports a baseline answer.

    This call is target-native and outcome blind.  It receives neither a gold
    answer, a source artifact, candidate effects, nor any unexecuted endpoint.
    The verifier is deliberately not allowed to replace the answer; its three
    calibrated outputs only ground whether the controller should acquire more
    visual evidence.
    """

    system = (
        "You are an independent visual-evidence verifier. Assess whether the "
        "supplied low-resolution overview supports the proposed answer to the "
        "multiple-choice question. Do not trust model confidence or reasoning "
        "text. Return JSON {\"support_probability\":0..1,"
        "\"overview_sufficiency_probability\":0..1,"
        "\"contradiction_probability\":0..1,\"reason\":\"brief visible "
        "evidence audit\"}. Probabilities must describe visible support only; "
        "use low sufficiency when OCR, counting, identity, or local detail is "
        "not resolvable from the overview."
    )
    content = [
        {
            "type": "text",
            "text": (
                "Question: " + prompt + "\nProposed answer: "
                + str(baseline["answer"]) + "\nProposer reason (untrusted): "
                + str(baseline.get("reason") or "")
            ),
        },
        BASE._image_content(overview),
    ]
    usages = []
    errors = []
    for attempt in range(2):
        repair = ""
        if attempt:
            repair = " Schema-repair retry: " + errors[-1]
        try:
            payload, usage = BASE._json_call(
                client, model=model, system=system + repair,
                content=content, max_tokens=1200,
            )
            usages.append(usage)
            expected = {
                "support_probability", "overview_sufficiency_probability",
                "contradiction_probability",
            }
            verified = {
                key: _clip_probability(payload.get(key), label=key)
                for key in sorted(expected)
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt == 0:
                continue
            raise
        return verified | {"reason": str(payload.get("reason") or "")}, {
            "calls": usages,
            "schema_repair_attempts": attempt,
            "schema_errors": errors,
        }
    raise RuntimeError("unreachable V3 verifier repair state")


def augment_with_baseline_verifier(
    receipt: Mapping[str, Any], *, target_input: Mapping[str, Any],
    dataset_root: Path, config: Mapping[str, Any], api_key: str,
    verifier_model: str,
) -> dict[str, Any]:
    """Attach a content-bound verifier result to an existing V3 receipt."""

    image_path = dataset_root / str(target_input["image_1"])
    with Image.open(image_path) as handle:
        maximum = int(config["media"]["native_working_max_side"])
        handle.draft("RGB", (maximum, maximum))
        image = handle.convert("RGB")
        image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    overview_image = BASE._thumbnail(image, int(config["media"]["overview_max_side"]))
    overview = BASE._image_bytes(
        overview_image, max_side=0, quality=int(config["media"]["jpeg_quality"]),
    )
    overview_sha256 = hashlib.sha256(overview).hexdigest()
    if overview_sha256 != str(receipt["overview_sha256"]):
        raise ValueError("baseline verifier overview does not match receipt")
    client = OpenAI(
        api_key=api_key, base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    verification, usage = _verify_baseline_v3(
        client, model=verifier_model, prompt=str(target_input["prompt"]),
        overview=overview, baseline=receipt["baseline"],
    )
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    base_contract = str(body["collection_contract_sha256"])
    augmented_contract = stable_hash({
        "base_collection_contract_sha256": base_contract,
        "augmentation": "OUTCOME_BLIND_BASELINE_SUPPORT_VERIFIER_V1",
        "verifier_model": verifier_model,
    })
    body["base_collection_contract_sha256"] = base_contract
    body["collection_contract_sha256"] = augmented_contract
    body["baseline_verification"] = verification
    body["baseline_verification_usage"] = usage
    body["baseline_verification_contract"] = {
        "model": verifier_model,
        "overview_sha256": overview_sha256,
        "baseline_answer_sha256": stable_hash({
            "answer": receipt["baseline"]["answer"],
            "reason": receipt["baseline"].get("reason"),
        }),
        "gold_answer_exposed": False,
        "source_program_or_identity_exposed": False,
        "candidate_endpoint_exposed": False,
    }
    return body | {"receipt_sha256": stable_hash(body)}


def augmented_collection_contract(base_contract: str, verifier_model: str) -> str:
    return stable_hash({
        "base_collection_contract_sha256": str(base_contract),
        "augmentation": "OUTCOME_BLIND_BASELINE_SUPPORT_VERIFIER_V1",
        "verifier_model": str(verifier_model),
    })


def _collection_contract(config: Mapping[str, Any]) -> str:
    paths = (
        Path(__file__).resolve(),
        REPO / "scripts/collect_phase3_tir_visual_search_v2.py",
        REPO / "scripts/collect_phase3_tir_nonmaze.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
    )
    return stable_hash({
        "parent_split_config_sha256": config["config_sha256"],
        "code_sha256": {str(path): BASE.file_sha256(path) for path in paths},
        "role": "DEVELOPMENT_ONLY",
        "collection_role": COLLECTION_ROLE,
        "overview_max_side": OVERVIEW_MAX_SIDE,
        "proposal_model": PROPOSAL_MODEL_OVERRIDE or config["model"]["id"],
        "baseline_model": (
            BASELINE_MODEL_OVERRIDE or ANSWER_MODEL_OVERRIDE
            or config["model"]["id"]
        ),
        "endpoint_model": (
            ENDPOINT_MODEL_OVERRIDE or ANSWER_MODEL_OVERRIDE
            or config["model"]["id"]
        ),
        "baseline_verifier_model": (
            BASELINE_VERIFIER_MODEL_OVERRIDE or PROPOSAL_MODEL_OVERRIDE
            or config["model"]["id"]
        ),
        "candidate_schedule": "CONTEXT_H1_TO_LOCAL_H4_TO_NEIGHBORS_H8",
        "observed_effect": "STRICT_TARGET_NEURAL_EVIDENCE_QUALITY_V1",
        "endpoint_overview_policy": ENDPOINT_OVERVIEW_POLICY,
        "source_program_exposed": False,
    })


def _collect_sample_v3(*args, **kwargs) -> dict[str, Any]:
    # V2's execution path is reused so every action still goes through the
    # repository wrapper.  Only the target-native proposal/answer grounders are
    # replaced above.
    output = V2._collect_sample(*args, **kwargs)
    body = dict(output)
    body.pop("receipt_sha256", None)
    body["schema_version"] = "phase3-tir-visual-search-receipt-v3"
    body["development_only_protocol"] = COLLECTION_ROLE.startswith("development")
    body["evaluation_role"] = COLLECTION_ROLE
    body["overview_intervention_design"] = {
        "overview_max_side": OVERVIEW_MAX_SIDE,
        "schedule": "CONTEXT_H1_TO_LOCAL_H4_TO_NEIGHBORS_H8",
    }
    body = body | {"receipt_sha256": stable_hash(body)}
    return augment_with_baseline_verifier(
        body, target_input=kwargs["target_input"],
        dataset_root=Path(kwargs["dataset_root"]), config=kwargs["config"],
        api_key=str(kwargs["api_key"]),
        verifier_model=(
            BASELINE_VERIFIER_MODEL_OVERRIDE or PROPOSAL_MODEL_OVERRIDE
            or str(kwargs["config"]["model"]["id"])
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=(
            "development_train", "development_validation",
            "development_holdout", "qualification", "formal",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--model", default=None,
        help="Development-only target grounder override; recorded in contract.",
    )
    parser.add_argument("--proposal-model", default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--endpoint-model", default=None)
    parser.add_argument("--baseline-verifier-model", default=None)
    parser.add_argument(
        "--endpoint-overview-policy", choices=("context", "omit"),
        default="context",
    )
    parser.add_argument(
        "--authorization", type=Path, default=None,
        help="Required passed prior-stage report for qualification/formal.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    body = dict(config)
    claimed = str(body.pop("config_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise SystemExit("parent TIR split manifest hash mismatch")
    accepted_statuses = {
        "FROZEN_BEFORE_ANY_TIR_V2_TARGET_CALL",
        "FROZEN_BEFORE_NEW_DEVELOPMENT_HOLDOUT_QUALIFICATION_FORMAL",
        "FROZEN_CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY",
    }
    if config.get("status") not in accepted_statuses:
        raise SystemExit("unexpected parent split manifest")
    if config.get("status") != "FROZEN_BEFORE_ANY_TIR_V2_TARGET_CALL":
        for relative, expected in config["integrity"]["code_sha256"].items():
            if BASE.file_sha256(REPO / relative) != str(expected):
                raise SystemExit(f"TIR V3 frozen dependency changed: {relative}")
    dataset_file = args.dataset_root / "TIR-Bench.json"
    if BASE.file_sha256(dataset_file) != config["dataset"]["sha256"]:
        raise SystemExit("TIR dataset drift")
    Image.MAX_IMAGE_PIXELS = int(config["media"]["maximum_source_pixels"])
    # The split registry stays frozen.  Evaluation pools require a passed,
    # content-bound report from the immediately preceding stage.
    ids = list(map(str, config["splits"][args.stage]))
    forbidden = set(map(str, config["splits"]["qualification"])) | set(
        map(str, config["splits"]["formal"])
    )
    if args.stage.startswith("development") and set(ids) & forbidden:
        raise SystemExit("development stage overlaps a locked target pool")
    if args.stage in {"qualification", "formal"}:
        if args.authorization is None or not args.authorization.is_file():
            raise SystemExit(f"{args.stage} requires a prior-stage authorization")
        authorization = json.loads(args.authorization.read_text())
        authorization_body = dict(authorization)
        claimed_report = str(authorization_body.pop("report_sha256", ""))
        if not claimed_report or stable_hash(authorization_body) != claimed_report:
            raise SystemExit("authorization report hash mismatch")
        required_status = (
            "TIR_PHASE3_DEVELOPMENT_HOLDOUT_PASSED"
            if args.stage == "qualification"
            else "TIR_PHASE3_QUALIFICATION_PASSED"
        )
        if authorization.get("status") != required_status:
            raise SystemExit(f"{args.stage} prior-stage gate did not pass")
        if (
            authorization.get("phase3_tir_manifest_sha256")
            != config["config_sha256"]
        ):
            raise SystemExit("authorization belongs to another TIR manifest")
        if (
            authorization.get("grounder_artifact_sha256")
            != config["grounder"]["artifact_sha256"]
        ):
            raise SystemExit("authorization belongs to another grounder")
    rows = json.loads(dataset_file.read_text())
    index = {str(row["id"]): row for row in rows}
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    runtime_config = json.loads(json.dumps(config))
    runtime_config["media"]["overview_max_side"] = OVERVIEW_MAX_SIDE
    if args.model:
        runtime_config["model"]["id"] = str(args.model)
    global PROPOSAL_MODEL_OVERRIDE, ANSWER_MODEL_OVERRIDE
    global BASELINE_MODEL_OVERRIDE, ENDPOINT_MODEL_OVERRIDE, COLLECTION_ROLE
    global BASELINE_VERIFIER_MODEL_OVERRIDE
    global ENDPOINT_OVERVIEW_POLICY
    PROPOSAL_MODEL_OVERRIDE = (
        str(args.proposal_model) if args.proposal_model else None
    )
    ANSWER_MODEL_OVERRIDE = str(args.answer_model) if args.answer_model else None
    BASELINE_MODEL_OVERRIDE = (
        str(args.baseline_model) if args.baseline_model else None
    )
    ENDPOINT_MODEL_OVERRIDE = (
        str(args.endpoint_model) if args.endpoint_model else None
    )
    BASELINE_VERIFIER_MODEL_OVERRIDE = (
        str(args.baseline_verifier_model)
        if args.baseline_verifier_model else None
    )
    COLLECTION_ROLE = str(args.stage)
    ENDPOINT_OVERVIEW_POLICY = str(args.endpoint_overview_policy)
    base_contract = _collection_contract(runtime_config)
    verifier_model = (
        BASELINE_VERIFIER_MODEL_OVERRIDE or PROPOSAL_MODEL_OVERRIDE
        or runtime_config["model"]["id"]
    )
    contract = augmented_collection_contract(base_contract, verifier_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / f"{args.stage}_receipts.json"
    existing = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text())
        }
        if any(
            row.get("collection_contract_sha256") != contract
            for row in existing.values()
        ):
            raise SystemExit("TIR V3 development receipt contract mismatch")
    pending = [sample_id for sample_id in ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for sample_id in pending:
            row = index[sample_id]
            target_input = {key: value for key, value in row.items() if key != "answer"}
            futures[executor.submit(
                _collect_sample_v3, sample_id, target_input=target_input,
                gold_answer=str(row["answer"]), dataset_root=args.dataset_root,
                config=runtime_config, api_key=str(key),
                contract_sha256=base_contract,
            )] = sample_id
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                existing[sample_id] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            ordered = [existing[value] for value in ids if value in existing]
            receipts_path.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(ordered)}/{len(ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete TIR V3 development receipts; rerun: {missing}")
    print(json.dumps({
        "stage": args.stage,
        "receipts": len(ids),
        "collection_contract_sha256": contract,
        "receipts_file_sha256": BASE.file_sha256(receipts_path),
        "formal_or_qualification_id_opened": args.stage in {"qualification", "formal"},
        "output": str(receipts_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    # Patch only the imported execution module.  No source-side module is
    # imported into the neural proposal or answer prompts.
    V2._propose_programs = _propose_programs_v3
    V2.BASE._answer = _answer_v3
    raise SystemExit(main())
