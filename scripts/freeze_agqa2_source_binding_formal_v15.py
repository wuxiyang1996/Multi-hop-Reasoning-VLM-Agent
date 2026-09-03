#!/usr/bin/env python3
"""Freeze the final 360-video single-binding formal replication."""

import scripts.freeze_agqa2_router_heldout_formal_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_source_binding_formal_v15_selection.json"
freezer.COUNT = 266
freezer.NONCE = "agqa2-source-single-binding-no-tiebreak-final-formal-v15"
freezer.STATUS = "FROZEN_V74_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS"
freezer.SPLIT_NAME = "official_train_formal_holdout_source_binding_v15"


if __name__ == "__main__":
    freezer.main()
