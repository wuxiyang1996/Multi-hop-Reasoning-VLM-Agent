#!/usr/bin/env python3
"""Activate the untouched V73 formal source-executor cohort."""

import scripts.activate_agqa2_router_grounding_development_v1 as activation


activation.SELECTION = activation.REPO_ROOT / "configs/agqa2_source_executor_formal_v14_selection.json"
activation.DOWNLOAD = activation.REPO_ROOT / "runs/agqa2_source_executor_formal_v14_download/receipt.json"
activation.MANIFEST = activation.REPO_ROOT / "configs/agqa2_source_executor_formal_v14_manifest.json"
activation.CONFIG = activation.REPO_ROOT / "configs/agqa2_source_executor_formal_v14.json"
activation.MANIFEST_STATUS = "FROZEN_V73_FORMAL_BEFORE_PROVIDER_OR_FORMAL_LABEL_ACCESS"
activation.CONFIG_STATUS = "FROZEN_V73_SOURCE_EXECUTOR_FORMAL"
activation.REPORT_VERSION = "QWEN235_V73"
activation.CLAIM_BOUNDARY = "240_UNTOUCHED_FORMAL_HOLDOUT_VIDEOS;CORE_SOURCE_INDUCED_TYPED_EXECUTOR_CONFIRMATORY_EVALUATION"
activation.SPLIT = "reserve"


if __name__ == "__main__":
    activation.main()
