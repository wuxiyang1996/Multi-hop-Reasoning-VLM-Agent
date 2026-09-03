#!/usr/bin/env python3
"""Activate the final untouched V74 single-binding replication."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_source_binding_formal_v15_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_source_binding_formal_v15_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_source_binding_formal_v15_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_source_binding_formal_v15.json"
activation.MANIFEST_STATUS = "FROZEN_V74_FORMAL_BEFORE_PROVIDER_OR_FORMAL_LABEL_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V74_SOURCE_BINDING_FORMAL"
activation.REPORT_VERSION = "QWEN235_V74"
activation.CLAIM_BOUNDARY = "360_UNTOUCHED_FORMAL_HOLDOUT_VIDEOS;FINAL_SINGLE_BINDING_NO_TIEBREAK_SOURCE_EXECUTOR_REPLICATION"
activation.SPLIT = "reserve"


if __name__ == "__main__":
    activation.main()
