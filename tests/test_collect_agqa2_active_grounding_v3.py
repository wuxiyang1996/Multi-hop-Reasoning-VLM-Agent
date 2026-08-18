import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches,
    _evaluation_protocol_core,
    _grounder_semantic_core,
    _operand_response_format,
    _provider_json_call,
    _query_response_format,
)
from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    parse_operand_receipt,
    recurrent_rescan_window,
    specific_object_grounded,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def test_frozen_v3_manifests_are_video_disjoint_and_content_hashed():
    manifests = []
    for split in ("development", "reserve"):
        path = REPO / f"configs/agqa2_active_grounding_v3_{split}_manifest.json"
        payload = json.loads(path.read_text())
        claimed = payload.pop("manifest_sha256")
        assert claimed == stable_hash(payload)
        manifests.append(payload)
    dev_videos = {row["video_id"] for row in manifests[0]["samples"]}
    reserve_videos = {row["video_id"] for row in manifests[1]["samples"]}
    assert len(dev_videos) == len(reserve_videos) == 9
    assert dev_videos.isdisjoint(reserve_videos)
    assert manifests[1]["prior_v3_raw_video_exposure"] is False


def test_preregistered_gates_are_not_weaker_than_v2():
    prereg = json.loads((
        REPO / "configs/agqa2_active_grounding_v3_preregistration.json"
    ).read_text())
    for split in ("development", "reserve"):
        gates = prereg[f"{split}_gates"]
        assert gates["minimum_decisive_accuracy"] >= 2 / 3
        assert gates["maximum_typed_vs_direct_losses"] == 0
        assert gates["minimum_typed_vs_direct_wins"] >= 1


def test_provider_schemas_avoid_known_incompatible_unique_items_keyword():
    assert "uniqueItems" not in json.dumps(_query_response_format())
    assert "uniqueItems" not in json.dumps(_operand_response_format(48))


def test_grounder_identity_excludes_dataset_level_selection_quota():
    base = json.loads((
        REPO / "configs/agqa2_active_grounding_v13_development.json"
    ).read_text())
    scaled = deepcopy(base)
    scaled["runtime_selection"].update({
        "candidate_count": 36,
        "per_predicted_route": 10,
    })
    sources = [SimpleNamespace(contract_sha256="source-contract")]
    assert stable_hash(_grounder_semantic_core(base, sources)) == stable_hash(
        _grounder_semantic_core(scaled, sources)
    )
    assert stable_hash(_evaluation_protocol_core(base)) != stable_hash(
        _evaluation_protocol_core(scaled)
    )


def test_grounder_identity_still_changes_for_acquisition_semantics():
    base = json.loads((
        REPO / "configs/agqa2_active_grounding_v13_development.json"
    ).read_text())
    changed = deepcopy(base)
    changed["acquisition"]["rescan_confidence_threshold"] += 0.01
    sources = [SimpleNamespace(contract_sha256="source-contract")]
    assert stable_hash(_grounder_semantic_core(base, sources)) != stable_hash(
        _grounder_semantic_core(changed, sources)
    )


def test_provider_call_retries_a_null_choices_transport_envelope():
    good = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"response":"yes"}'),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=2, completion_tokens=1,
            cost=0.001, model_extra={},
        ),
        model="test-model",
    )

    class Completions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(choices=None) if self.calls == 1 else good

    completions = Completions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    payload, usage = _provider_json_call(
        client,
        model={"id": "test-model"},
        system="test",
        content=[{"type": "text", "text": "test"}],
        max_tokens=10,
        response_format={"type": "json_object"},
    )
    assert completions.calls == 2
    assert payload == {"response": "yes"}
    assert usage["reported_cost_usd"] == 0.001


def test_rescan_window_uses_evidence_and_falls_back_to_full_video():
    observed = parse_operand_receipt({
        "operand_role": "A", "requested_operand": "opening a laptop",
        "observations": [{
            "occurrence_id": "O0", "label": "opening a laptop",
            "subject": "person", "predicate": "opening", "object": "laptop",
            "observability": "PARTIAL", "start_frame": 20, "end_frame": 22,
            "evidence_frames": [21], "confidence": 0.5, "uncertainties": [],
        }],
        "coverage": "PARTIAL", "uncertainties": [],
    }, expected_role="A", expected_operand="opening a laptop", frame_count=48)
    start, end = recurrent_rescan_window(
        observed, seconds=list(range(48)), duration=48.0,
    )
    assert (start, end) == (17.0, 25.0)

    unobserved = parse_operand_receipt({
        "operand_role": "A", "requested_operand": "opening a laptop",
        "observations": [{
            "occurrence_id": "O0", "label": "opening a laptop",
            "subject": "person", "predicate": "opening", "object": "laptop",
            "observability": "UNOBSERVED", "start_frame": None, "end_frame": None,
            "evidence_frames": [], "confidence": 0.1, "uncertainties": [],
        }],
        "coverage": "INSUFFICIENT", "uncertainties": [],
    }, expected_role="A", expected_operand="opening a laptop", frame_count=48)
    assert recurrent_rescan_window(
        unobserved, seconds=list(range(48)), duration=48.0,
    ) == (0.0, 48.0)


def test_answer_matching_keeps_free_object_exact_and_closed_sets_canonical():
    assert _answer_matches("After sitting down.", "after")
    assert _answer_matches("a chair", "chair")
    assert not _answer_matches("chair or bed", "chair")


def test_generic_object_does_not_anchor_a_recurrent_rescan_window():
    generic = parse_operand_receipt({
        "operand_role": "A", "requested_operand": "putting down unknown object",
        "observations": [{
            "occurrence_id": "O0", "label": "putting down", "subject": "person",
            "predicate": "putting down", "object": "unknown object",
            "observability": "OBSERVED", "start_frame": 3, "end_frame": 5,
            "evidence_frames": [3, 5], "confidence": 0.9, "uncertainties": [],
        }],
        "coverage": "PARTIAL", "uncertainties": [],
    }, expected_role="A", expected_operand="putting down unknown object", frame_count=8)
    assert not specific_object_grounded(generic)
    assert recurrent_rescan_window(
        generic, seconds=list(range(8)), duration=8.0,
        require_specific_object=True,
    ) == (0.0, 8.0)
