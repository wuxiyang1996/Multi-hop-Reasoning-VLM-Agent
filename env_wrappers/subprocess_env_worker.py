#!/usr/bin/env python
"""Subprocess env worker — runs a game env and serves it over stdin/stdout.

This script is meant to be executed in a child process so the env's
hard "one instance per process" constraints (Orak / nes-py NumPy 2.x
incompatibility, **stable-retro Genesis emulator singleton** for the
``gymv_*`` Temporal/* envs) cannot collide with the training process or
with sibling episodes.  Communication uses newline-delimited JSON over
stdin/stdout.

Protocol
--------
Request  (parent -> worker, one JSON line on stdin):
    {"cmd": "reset"}
    {"cmd": "step", "action": "<action string>"}
    {"cmd": "close"}
    {"cmd": "get_action_names"}

Response (worker -> parent, one JSON line on stdout):
    {"ok": true, ...payload...}
    {"ok": false, "error": "<message>"}

The worker exits cleanly when stdin is closed or a "close" command arrives.

CLI
---
``--env-kind {orak,gymv}`` selects the env factory:

* ``orak`` (default; back-compat): ``env_wrappers.orak_nl_wrapper.make_orak_env``
  — the original super_mario / nes-py path.
* ``gymv``: ``env_wrappers.gymv_temporal_nl_wrapper.make_gymv_temporal_env``
  — wraps stable-retro Genesis envs.  This branch fixes the
  "Cannot create multiple emulator instances per process" failure that
  used to drop 7/8 concurrent gymv episodes per GRPO step.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from typing import Any, List

os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# We communicate JSON over fd 1 (stdout).  Game envs may print debug
# output to stdout which would corrupt the protocol.  We keep a
# reference to the *real* stdout for our JSON messages and redirect
# sys.stdout to stderr so any game prints go there instead.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr  # game prints go to stderr


def _write(obj: dict) -> None:
    _REAL_STDOUT.write(json.dumps(obj, default=str) + "\n")
    _REAL_STDOUT.flush()


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect even stderr-routed stdout to devnull during noisy env calls."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--env-kind",
        choices=("orak", "gymv"),
        default="orak",
        help=(
            "Env factory to use. 'orak' (default) -> "
            "env_wrappers.orak_nl_wrapper.make_orak_env (super_mario, etc.). "
            "'gymv' -> env_wrappers.gymv_temporal_nl_wrapper."
            "make_gymv_temporal_env (stable-retro Genesis singleton workaround)."
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codebase_root = os.path.dirname(script_dir)
    orak_src = os.path.join(codebase_root, "..", "Orak", "src")
    for p in [codebase_root, orak_src]:
        rp = os.path.realpath(p)
        if os.path.isdir(rp) and rp not in sys.path:
            sys.path.insert(0, rp)

    if args.env_kind == "gymv":
        from env_wrappers.gymv_temporal_nl_wrapper import make_gymv_temporal_env
        with _suppress_stdout():
            env = make_gymv_temporal_env(args.game, max_steps=args.max_steps)
    else:
        from env_wrappers.orak_nl_wrapper import make_orak_env
        with _suppress_stdout():
            env = make_orak_env(args.game, max_steps=args.max_steps)

    # T2.16 (2026-05-05): stash the most recent frame across reset/step
    # so the new ``render`` cmd (vision-aware game schema) can return it
    # without burning extra emulator state.  Lives in a 1-slot list so
    # the closure captures by reference.
    _last_frame_holder: List[Any] = [None]

    def _strip_info(info: dict) -> dict:
        # Drop fields that aren't JSON-friendly or are redundant with `obs`.
        # ``state_natural_language`` is duplicated by the worker's ``obs``
        # field already.  ``raw_obs`` (gymv: ``RetroObs`` dataclass) and
        # ``image`` (numpy array) are not JSON-serialisable — episode_runner
        # never reads them at training time, only offline tools under
        # ``visual_grounding_tests/`` and ``cold_start/legacy/`` do.
        # T2.16: capture ``info["image"]`` (numpy frame) into the worker's
        # holder before stripping so ``cmd=render`` can serve it later.
        img = info.get("image", None)
        if img is not None:
            _last_frame_holder[0] = img
        for k in ("state_natural_language", "raw_obs", "image"):
            info.pop(k, None)
        return info

    _write({"ok": True, "status": "ready", "action_names": list(env.action_names)})

    # Read commands from the real stdin (fd 0).
    real_stdin = open(0, "r")

    for raw_line in real_stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _write({"ok": False, "error": f"bad json: {exc}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "reset":
                with _suppress_stdout():
                    obs, info = env.reset()
                _write({"ok": True, "obs": obs, "info": _strip_info(info)})

            elif cmd == "step":
                action = req.get("action", "")
                with _suppress_stdout():
                    obs, reward, terminated, truncated, info = env.step(action)
                _write({
                    "ok": True,
                    "obs": obs,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _strip_info(info),
                })

            elif cmd == "get_action_names":
                _write({"ok": True, "action_names": env.action_names})

            elif cmd == "render":
                # T2.16 (2026-05-05): return current frame as
                # base64-encoded PNG so the orchestrator's vision-aware
                # game-schema generator (35B 1-shot) can pass an actual
                # image to the multimodal judge.  Always returns ``ok``
                # — failures populate ``image_b64=None`` so callers can
                # silently fall back to text-only schema generation.
                #
                # Frame source priority:
                #   1. ``_last_frame_holder`` populated by ``_strip_info``
                #      from ``info["image"]`` (gymv path — most reliable),
                #   2. ``env.render()`` (orak / gaming-agent envs),
                #   3. fallback attrs (``last_frame``, ``rgb_frame``...).
                image_b64 = None
                error = None
                try:
                    arr = _last_frame_holder[0]
                    if arr is None and hasattr(env, "render"):
                        try:
                            with _suppress_stdout():
                                arr = env.render()
                        except Exception as exc:  # noqa: BLE001
                            error = f"render_failed:{type(exc).__name__}"
                    if arr is None:
                        # Some gymv wrappers stash the raw RGB array on
                        # the underlying env; try common attributes.
                        for attr in ("last_frame", "_last_frame",
                                     "current_frame", "rgb_frame"):
                            if hasattr(env, attr):
                                cand = getattr(env, attr)
                                if cand is not None:
                                    arr = cand
                                    break
                    if arr is not None:
                        import base64 as _b64
                        from io import BytesIO as _BIO
                        try:
                            from PIL import Image as _PILImage
                            import numpy as _np
                            if isinstance(arr, _np.ndarray):
                                if arr.dtype != _np.uint8:
                                    # Map float images in [0,1] → uint8
                                    if arr.dtype.kind == "f":
                                        arr = (
                                            _np.clip(arr, 0.0, 1.0) * 255.0
                                        ).astype(_np.uint8)
                                    else:
                                        arr = arr.astype(_np.uint8)
                                pil = _PILImage.fromarray(arr)
                            else:
                                pil = arr  # already PIL
                            buf = _BIO()
                            pil.save(buf, format="PNG", optimize=True)
                            image_b64 = _b64.b64encode(buf.getvalue()).decode(
                                "ascii"
                            )
                        except Exception as exc:  # noqa: BLE001
                            error = f"encode_failed:{type(exc).__name__}"
                    elif error is None:
                        error = "no_frame_available"
                except Exception as exc:  # noqa: BLE001
                    error = f"unexpected:{type(exc).__name__}"
                _write({
                    "ok": True,
                    "image_b64": image_b64,
                    "render_error": error,
                })

            elif cmd == "close":
                with _suppress_stdout():
                    env.close()
                _write({"ok": True, "status": "closed"})
                break

            else:
                _write({"ok": False, "error": f"unknown cmd: {cmd}"})

        except Exception:
            _write({"ok": False, "error": traceback.format_exc()})

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
