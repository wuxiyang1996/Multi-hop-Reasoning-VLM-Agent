#!/usr/bin/env python3
"""Activate the pre-registered 160-video V71 qualification."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_source_triggered_powered_qualification_v12_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_source_triggered_powered_qualification_v12_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_source_triggered_powered_qualification_v12_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_source_triggered_powered_qualification_v12.json"
activation.MANIFEST_STATUS = "FROZEN_V71_QUALIFICATION_BEFORE_PROVIDER_OR_OUTCOME_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V71_SOURCE_TRIGGERED_POWERED_QUALIFICATION"
activation.REPORT_VERSION = "QWEN235_V71"
activation.CLAIM_BOUNDARY = "160_NEW_ROUTER_VALIDATION_VIDEOS;POWERED_QUALIFICATION_OF_UNCHANGED_SOURCE_TRIGGERED_QWEN235_TOOL_POLICY;NO_FORMAL_CLAIM"


if __name__ == "__main__":
    activation.main()
