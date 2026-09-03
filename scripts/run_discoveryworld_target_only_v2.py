#!/usr/bin/env python3
"""Run the V1 target-native policy with OpenAI JSON-mode compatibility.

OpenAI-compatible providers reject JSON mode unless the request text contains
the lowercase token ``json``.  The scientific prompt and parser are otherwise
unchanged; this wrapper only supplies that transport-level compatibility token.
"""

from __future__ import annotations

import scripts.run_discoveryworld_target_only_v1 as runner


runner.TARGET_ONLY_SYSTEM_PROMPT += "\nReturn one valid json object."


if __name__ == "__main__":
    runner.main()
