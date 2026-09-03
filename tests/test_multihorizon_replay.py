from __future__ import annotations

from motif_transfer.multihorizon_replay import (
    analyze_multihorizon_rows,
    choose_lineage_snapshots,
    cumulative_returns,
    extract_policy_prefix,
)


def test_lineage_sampling_preserves_longest_span_without_outcomes():
    records = []
    for step in (0, 1, 2, 5):
        records.append({
            "treatment": "G_PLUS_S", "episode_seed": 1,
            "episode_id": "e", "step": step,
        })
    chosen = choose_lineage_snapshots(
        records, {1: "qualification"}, maximum_per_split=2
    )
    assert chosen["qualification"] == ["seed=1:step=0", "seed=1:step=1"]


def test_prompt_prefix_and_absorbing_cumulative_returns():
    assert extract_policy_prefix("SYSTEM\nGame state:\nstate") == "SYSTEM"
    assert cumulative_returns([1.0, 2.0]) == {
        "h1": 1.0, "h2": 3.0, "h4": 3.0, "h8": 3.0,
    }


def test_h8_gate_requires_both_estimands_and_both_blind_splits():
    rows = []
    split_by_seed = {0: "discovery", 1: "qualification", 2: "held_out"}
    for seed in split_by_seed:
        for mode in ("COMMON_G_MINUS_S_CONTINUATION", "FULL_TREATMENT_REGIME"):
            for treatment, value in (
                ("B", 0), ("G_MINUS_S", 1), ("G_PLUS_S", 3), ("G_PLUS_RANDOM", 2)
            ):
                rows.append({
                    "episode_seed": seed,
                    "episode_id": f"episode-{seed}",
                    "fork_step": 3,
                    "mode": mode,
                    "treatment": treatment,
                    "status": "INTERVENTION_OBSERVED",
                    "cumulative_returns": {f"h{h}": value for h in (1, 2, 4, 8)},
                })
    report = analyze_multihorizon_rows(rows, split_by_seed)
    assert report["gates"]["SOURCE_H8_VALUE_SUPPORTED"] is True
    # Break the held-out FULL_TREATMENT_REGIME source-content contrast:
    # G_PLUS_S must no longer beat G_MINUS_S at h8.
    rows[-2]["cumulative_returns"]["h8"] = 0
    report = analyze_multihorizon_rows(rows, split_by_seed)
    assert report["gates"]["HELDOUT_FULL_H8_VALUE"] is False
    assert report["gates"]["SOURCE_H8_VALUE_SUPPORTED"] is False


def test_h8_gate_fails_closed_on_an_incomplete_blind_cell():
    rows = []
    for split, seed in (("qualification", 1), ("held_out", 2)):
        for mode in ("COMMON_G_MINUS_S_CONTINUATION", "FULL_TREATMENT_REGIME"):
            for treatment, value in (
                ("B", 0), ("G_MINUS_S", 1), ("G_PLUS_S", 3),
                ("G_PLUS_RANDOM", 2),
            ):
                rows.append({
                    "episode_seed": seed,
                    "episode_id": f"episode-{seed}",
                    "fork_step": 4,
                    "split": split,
                    "mode": mode,
                    "treatment": treatment,
                    "status": "INTERVENTION_OBSERVED",
                    "cumulative_returns": {f"h{h}": value for h in (1, 2, 4, 8)},
                })
    rows.pop()
    report = analyze_multihorizon_rows(rows)
    assert report["gates"]["BLIND_CELLS_COMPLETE"] is False
    assert report["gates"]["SOURCE_H8_VALUE_SUPPORTED"] is False
