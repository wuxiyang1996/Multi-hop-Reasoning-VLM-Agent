from motif_transfer.contracts import stable_hash
from motif_transfer.vtb_treatments import compile_vtb_treatments


def _bundle(motif_id: str, prefix: str):
    candidate = {
        "motif_id": motif_id,
        "source_lineage": [f"{prefix}-game", f"{prefix}-episode"],
        "nodes": [
            {"node_id": f"{prefix}-a", "transition_receipt_ids": [f"{prefix}-t1"],
             "decision_signatures": [{"reward_sign": "ZERO"}]},
            {"node_id": f"{prefix}-b", "transition_receipt_ids": [f"{prefix}-t2"],
             "decision_signatures": [{"reward_sign": "POSITIVE"}]},
        ],
        "edges": [{"source": f"{prefix}-a", "target": f"{prefix}-b",
                   "replay_receipt_ids": [f"{prefix}-r1"], "untrusted_claim": "game words"}],
        "untrusted_description": "source description",
    }
    return {
        "candidate": candidate,
        "qualification": {"lifecycle": "SOURCE_SUPPORTED", "candidate_sha256": stable_hash(candidate)},
    }


def test_compiler_produces_structural_invariance_and_destructive_controls() -> None:
    treatments = compile_vtb_treatments(_bundle("m1", "x"), _bundle("m2", "y"))
    authentic = treatments["authentic_game_source"]
    renamed = treatments["renamed_game_source"]
    shuffled = treatments["shuffled_game_source"]
    generic = treatments["generic_reasoning"]
    assert authentic["source_receipt_ids"] == renamed["source_receipt_ids"]
    assert "x-a" not in str(renamed["payload"])
    assert len(authentic["payload"]["nodes"]) == len(renamed["payload"]["nodes"])
    assert shuffled["payload_sha256"] != authentic["payload_sha256"]
    assert generic["source_receipt_ids"] == []
    assert "x-t1" not in str(generic["payload"])


def test_compiler_rejects_unqualified_or_same_other_game() -> None:
    first = _bundle("m1", "x")
    other = _bundle("m2", "y")
    other["qualification"]["lifecycle"] = "CANDIDATE"
    try:
        compile_vtb_treatments(first, other)
    except ValueError as exc:
        assert "SOURCE_SUPPORTED" in str(exc)
    else:
        raise AssertionError("unqualified other-game control was accepted")
