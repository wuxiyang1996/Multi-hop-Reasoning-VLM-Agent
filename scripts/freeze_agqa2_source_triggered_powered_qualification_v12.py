#!/usr/bin/env python3
"""Freeze 160-video powered qualification with the V70 mechanism unchanged."""

import scripts.freeze_agqa2_router_grounding_development_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_source_triggered_powered_qualification_v12_selection.json"
freezer.COUNT = 160
freezer.NONCE = "agqa2-source-triggered-powered-qualification-v12-fixed-negative-transfer-rate-gate"
freezer.STATUS = "FROZEN_V71_QUALIFICATION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS"
freezer.SPLIT_NAME = "official_train_router_validation_source_triggered_powered_qualification_v12"


if __name__ == "__main__":
    freezer.main()
