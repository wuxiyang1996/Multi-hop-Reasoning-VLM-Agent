#!/usr/bin/env python3
"""Freeze the untouched 240-video formal source-executor cohort."""

import scripts.freeze_agqa2_router_heldout_formal_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_source_executor_formal_v14_selection.json"
freezer.COUNT = 240
freezer.NONCE = "agqa2-source-executor-untouched-formal-v14"
freezer.STATUS = "FROZEN_V73_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS"
freezer.SPLIT_NAME = "official_train_formal_holdout_source_executor_v14"


if __name__ == "__main__":
    freezer.main()
