#!/usr/bin/env python3
"""Activate fresh V70 source-triggered qualification data."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v10_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_source_triggered_qualification_v10_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v10_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v10.json"
activation.MANIFEST_STATUS = "FROZEN_V70_QUALIFICATION_BEFORE_PROVIDER_OR_OUTCOME_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V70_SOURCE_TRIGGERED_QUALIFICATION"
activation.REPORT_VERSION = "QWEN235_V70"
activation.CLAIM_BOUNDARY = "80_NEW_ROUTER_VALIDATION_VIDEOS;FRESH_QUALIFICATION_AFTER_ALL_KNOWN_OBJECT_CHOICE_AND_COMPOSITE_PROGRAM_ABSTENTION_FIXES;NO_FORMAL_CLAIM"


if __name__ == "__main__":
    activation.main()
