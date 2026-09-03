#!/usr/bin/env python3
"""Freeze fresh V69 qualification for the V8 source-triggered re-answer rule."""

import scripts.freeze_agqa2_router_grounding_development_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v9_selection.json"
freezer.NONCE = "agqa2-source-triggered-reanswer-v8-fresh-qualification-v9"
freezer.STATUS = "FROZEN_V69_QUALIFICATION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS"
freezer.SPLIT_NAME = "official_train_router_validation_fresh_source_triggered_qualification"


if __name__ == "__main__": freezer.main()
