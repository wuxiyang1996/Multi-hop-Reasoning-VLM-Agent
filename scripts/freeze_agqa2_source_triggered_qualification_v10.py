#!/usr/bin/env python3
"""Freeze fresh V70 qualification after the second-candidate ``some`` fix."""

import scripts.freeze_agqa2_router_grounding_development_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_source_triggered_qualification_v10_selection.json"
freezer.NONCE = "agqa2-source-triggered-reanswer-fresh-qualification-v10-after-some-second-candidate-fix"
freezer.STATUS = "FROZEN_V70_QUALIFICATION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS"
freezer.SPLIT_NAME = "official_train_router_validation_fresh_source_triggered_qualification_v10"


if __name__ == "__main__":
    freezer.main()
