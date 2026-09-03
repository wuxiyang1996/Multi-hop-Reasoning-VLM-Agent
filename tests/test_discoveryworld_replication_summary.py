from __future__ import annotations

from scripts.summarize_discoveryworld_replication_v1 import frozen_runtime_hashes_match


def test_frozen_runtime_hashes_fail_closed_on_postfreeze_drift() -> None:
    protocol = {"integrity": {
        "matched_runner_sha256": "runner",
        "environment_wrapper_sha256": "environment",
        "target_policy_sha256": "policy",
        "transfer_selector_sha256": "selector",
    }}
    valid = {"task": {"runtime_hashes": {
        "runner": "runner", "environment": "environment",
        "target_policy": "policy", "transfer_selector": "selector",
    }}}
    assert frozen_runtime_hashes_match(protocol, valid)
    valid["task"]["runtime_hashes"]["transfer_selector"] = "post-freeze"
    assert not frozen_runtime_hashes_match(protocol, valid)
