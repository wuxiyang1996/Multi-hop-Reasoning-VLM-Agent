#!/usr/bin/env python3
"""Activate fresh V69 source-triggered qualification data."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v9_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_source_triggered_qualification_v9_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v9_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v9.json"
activation.MANIFEST_STATUS = "FROZEN_V69_QUALIFICATION_BEFORE_PROVIDER_OR_OUTCOME_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V69_SOURCE_TRIGGERED_QUALIFICATION"
activation.REPORT_VERSION = "QWEN235_V69"
activation.CLAIM_BOUNDARY = "80_NEW_ROUTER_VALIDATION_VIDEOS;FRESH_QUALIFICATION_OF_V8_SOURCE_TRIGGERED_GLOBAL_REANSWER_AFTER_TYPED_APPLICABILITY_FIXES;NO_FORMAL_CLAIM"


if __name__ == "__main__": activation.main()
