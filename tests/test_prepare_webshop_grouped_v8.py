from scripts.prepare_webshop_grouped_v8 import build_grouped_manifest


def _goal(asin: str) -> dict:
    return {"asin": asin, "instruction_text": asin}


def test_group_split_is_disjoint_by_asin_and_quarantines_diagnostics() -> None:
    diagnostic = ["webshop.1", "webshop.6", "webshop.13", "webshop.24",
                  "webshop.28", "webshop.31", "webshop.34", "webshop.49"]
    task_ids = diagnostic + ["webshop.2", "webshop.3", "webshop.4", "webshop.5"]
    asins = [*[f"d{index}" for index in range(8)], *[f"g{index}" for index in range(4)]]
    goals = {
        task_id: _goal(asins[index])
        for index, task_id in enumerate(task_ids)
    }
    frozen = {
        "roles": {"adaptation": task_ids[:5], "qualification": task_ids[5:10], "reserve": task_ids[10:]},
        "task_ids": task_ids,
        "goals": goals,
        "artifact_sha256": "frozen",
    }

    artifact = build_grouped_manifest(frozen)

    roles = artifact["groups_by_role"]
    flattened = [group for groups in roles.values() for group in groups]
    assert len(flattened) == len(set(flattened)) == 12
    assert set(roles["diagnostic"]) == {f"d{index}" for index in range(8)}
    assert {role: len(groups) for role, groups in roles.items()} == {
        "adaptation": 2,
        "calibration": 1,
        "confirmation": 1,
        "diagnostic": 8,
    }
