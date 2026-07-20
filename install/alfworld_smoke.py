"""Import, reset, and one-step smoke test for text-mode ALFWorld."""

from __future__ import annotations

import random
import sys


def main() -> int:
    try:
        from env_wrappers.alfworld_nl_wrapper import make_alfworld_env
    except Exception as exc:
        print(f"[FAIL] ALFWorld wrapper import failed: {exc}")
        return 1

    try:
        env = make_alfworld_env(split="eval_out_of_distribution")
        observation, info = env.reset()
    except Exception as exc:
        print(f"[FAIL] ALFWorld reset failed: {exc}")
        print("       If data are missing, run alfworld-download.")
        return 1

    print(f"[OK] reset: {observation.replace(chr(10), ' ')[:160]}")
    actions = list(info.get("action_names") or [])
    action = random.choice(actions) if actions else "look"
    try:
        next_observation, reward, terminated, truncated, _ = env.step(action)
    except Exception as exc:
        print(f"[FAIL] step({action!r}) failed: {exc}")
        return 1
    finally:
        env.close()

    print(
        f"[OK] step: action={action!r} reward={reward} "
        f"done={terminated or truncated} "
        f"obs={next_observation.replace(chr(10), ' ')[:120]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
