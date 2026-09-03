#!/usr/bin/env python3
"""Activate fresh V68 grounding qualification data."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_router_grounding_qualification_v2_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_router_grounding_qualification_v2_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_router_grounding_qualification_v2_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_router_grounding_qualification_v2.json"
activation.MANIFEST_STATUS = "FROZEN_V68_QUALIFICATION_BEFORE_PROVIDER_OR_OUTCOME_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V68_ROUTER_AND_GROUNDER_QUALIFICATION"
activation.REPORT_VERSION = "QWEN235_V68"
activation.CLAIM_BOUNDARY = "80_NEW_ROUTER_VALIDATION_VIDEOS;FRESH_GROUNDER_QUALIFICATION_AFTER_V67_DEVELOPMENT;NO_FORMAL_CLAIM"


if __name__ == "__main__":
    activation.main()
