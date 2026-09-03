#!/usr/bin/env python3
"""Activate the pre-registered V72 source-executor qualification."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_source_executor_qualification_v13_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_source_executor_qualification_v13_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_source_executor_qualification_v13_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_source_executor_qualification_v13.json"
activation.MANIFEST_STATUS = "FROZEN_V72_QUALIFICATION_BEFORE_PROVIDER_OR_OUTCOME_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V72_SOURCE_EXECUTOR_QUALIFICATION"
activation.REPORT_VERSION = "QWEN235_V72"
activation.CLAIM_BOUNDARY = "120_NEW_ROUTER_VALIDATION_VIDEOS;FINAL_QUALIFICATION_OF_CORE_SOURCE_INDUCED_TYPED_EXECUTOR;NO_FORMAL_CLAIM"


if __name__ == "__main__":
    activation.main()
