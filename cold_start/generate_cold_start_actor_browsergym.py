#!/usr/bin/env python
"""
Cold-start actor-agent rollouts for **BrowserGym** (gpt-5.5 vision pipeline).

Pipeline (one outer step):

  1. Pull a multimodal observation from a real BrowserGym env (Playwright +
     headless Chromium): PIL screenshot, AXTree,
     ``extra_element_properties``, focused element, URL, last action / error.
     Two target modes are supported:

       - ``--tasks <env_id>``  pre-registered BrowserGym task ids
                              (e.g. ``browsergym/miniwob.click-button``,
                              ``browsergym/webarena.42``,
                              ``browsergym/visualwebarena.0``,
                              ``browsergym/assistantbench.test.0``).
                              The task ships with its own goal / success
                              criterion / reward.
       - ``--urls <url>``      open-ended browsing on top of
                              ``browsergym/openended`` (no built-in
                              task — reward is always 0, useful for
                              data collection on arbitrary live pages).

  2. Visual schema (deterministic fallback) — :func:`browsergym_wrapper
     .heuristic.obs_to_schema` walks the AXTree into the canonical
     ``<state>...</state>`` block. Free, fast, always available.
  3. Visual schema (VLM, primary) — gpt-5.5 (vision) reads the **screenshot**
     plus the AXTree-as-text grounding context and emits the canonical schema
     via :mod:`vlm_wrapper.schema`. The screenshot is the primary input.
  4. Action selection — gpt-5.5 reads the schema + the BrowserGym candidate
     actions list (``click(bid)`` / ``fill(bid, "...")`` / ``scroll`` /
     ``go_back`` / ``noop``) and picks ONE action via OpenAI function
     calling. The function-call schema separates ``action_type`` / ``bid`` /
     ``text`` so the actor can also fill text into an input box.
  5. ``env.step(action_string)`` and an :class:`Experience` is appended to
     the :class:`Episode`. The schema, raw VLM output, action reasoning,
     reward, terminate / truncate flags, and image path are all preserved
     on the Experience for SFT/GRPO consumers.

Companion to ``cold_start/generate_cold_start_actor_gymv.py`` (gym-v) and
``cold_start/generate_cold_start_actor.py`` (env_wrappers) — same Episode/
Experience output format, but driven through BrowserGym's live screenshot +
AXTree obs API. Real Chromium is required: there is no offline / synthetic
mode. Install via ``install/install_browsergym.sh``.

Output layout (``<codebase_root>/Cold-start-out-browsergym/<safe_id>/``):

  - ``episode_NNN.json``       individual Episode (Episode.to_dict())
  - ``episode_buffer.json``    Episode_Buffer (loadable for trainer)
  - ``rollouts.jsonl``         append-only JSONL, one Episode per line
  - ``rollout_summary.json``   per-target stats
  - ``frames/<ep>/step_NNN.png``  rendered frames sent to the VLM (debug)

Usage::

    export OPENAI_API_KEY="sk-..."          # or OPENROUTER_API_KEY

    # Default: 1 episode against Google + Wikipedia (openended), 8 steps each
    python cold_start/generate_cold_start_actor_browsergym.py

    # Real benchmark tasks (one episode each)
    python cold_start/generate_cold_start_actor_browsergym.py \\
        --tasks browsergym/miniwob.click-button \\
                browsergym/miniwob.enter-text \\
                browsergym/assistantbench.test.0 \\
        --episodes 1 --max_steps 12 --save_frames -v

    # Discover the registered task IDs (no rollout, just a listing)
    python cold_start/generate_cold_start_actor_browsergym.py --list_tasks

    # Custom open-ended URLs
    python cold_start/generate_cold_start_actor_browsergym.py \\
        --urls https://en.wikipedia.org/wiki/Reinforcement_learning \\
        --episodes 2 --max_steps 12 --save_frames -v

    # Visible (non-headless) Chromium for debugging
    python cold_start/generate_cold_start_actor_browsergym.py --no_headless

    # Cheap baseline: skip the vision call (AXTree-walked heuristic schema only)
    python cold_start/generate_cold_start_actor_browsergym.py --no_vision

Suite infra requirements (set ONLY if the corresponding suite is in --tasks):

    miniwob          MINIWOB_URL=file:///path/to/miniwob-plusplus/html/miniwob/
    webarena         WA_HOMEPAGE / WA_SHOPPING / WA_REDDIT / WA_GITLAB /
                     WA_WIKIPEDIA / WA_MAP — see github.com/web-arena-x/webarena
    visualwebarena   VWA_HOMEPAGE / VWA_CLASSIFIEDS / VWA_SHOPPING /
                     VWA_REDDIT — see github.com/web-arena-x/visualwebarena
    assistantbench   no extra infra (loads from HuggingFace dataset cache)
    openended        no extra infra (any live URL works)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make the codebase + workspace root importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in [str(CODEBASE_ROOT), str(WORKSPACE_ROOT)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_api_keys_from_file() -> Optional[Path]:
    """Seed ``os.environ`` from a sibling ``api_keys.py`` if present.

    Looked-up locations (first hit wins):
      1. ``$COSPLAY_API_KEYS_FILE``
      2. ``cold_start/api_keys.py``
      3. ``<codebase_root>/api_keys.py``
      4. ``<codebase_root>/../api_keys.py``  (workspace root)

    The file should define any of:
      ``openrouter_api_key``, ``openai_api_key``, ``claude_api_key``
    Existing env vars are NOT overwritten.
    """
    import importlib.util

    candidates: List[Path] = []
    env_override = os.environ.get("COSPLAY_API_KEYS_FILE")
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend([
        SCRIPT_DIR / "api_keys.py",
        CODEBASE_ROOT / "api_keys.py",
        CODEBASE_ROOT.parent / "api_keys.py",
    ])

    for path in candidates:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        try:
            spec = importlib.util.spec_from_file_location("_cosplay_api_keys", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to load api_keys file {path}: {exc}", file=sys.stderr)
            continue

        mapping = {
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
            "claude_api_key": "ANTHROPIC_API_KEY",
        }
        for attr, env_name in mapping.items():
            val = getattr(mod, attr, None)
            if isinstance(val, str) and val.strip() and not os.environ.get(env_name):
                os.environ[env_name] = val.strip()
        return path
    return None


_API_KEYS_FILE_USED = _bootstrap_api_keys_from_file()


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

import openai

try:
    from common.models import BACKBONE_SFT_TEACHER_MODEL as _SFT_TEACHER_MODEL
except Exception:  # pragma: no cover — keep script runnable in isolation
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_MODEL = _SFT_TEACHER_MODEL  # gpt-5.5

try:
    from API_func import OPENROUTER_BASE, make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    make_openai_client = None
    effective_openai_model = None

logger = logging.getLogger("cold_start.actor_browsergym")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_URLS: List[str] = [
    "https://www.google.com",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
]

# Optional task suites to import. Each pulls in its own registered env ids
# (browsergym/miniwob.<task>, browsergym/webarena.<id>, …). Failures are
# logged once and the suite is dropped — most installs only have a subset.
_OPTIONAL_TASK_SUITE_MODULES: List[str] = [
    "browsergym.miniwob",
    "browsergym.webarena",
    "browsergym.visualwebarena",
    "browsergym.assistantbench",
    "browsergym.workarena",
]


# ---------------------------------------------------------------------------
# Target = (kind, payload, safe_id)
#
# kind == "task"   : payload is a registered BrowserGym env id
#                    (e.g. "browsergym/miniwob.click-button"). The task
#                    carries its own goal + reward function — we just call
#                    gym.make(env_id).
# kind == "url"    : payload is a live URL. We boot browsergym/openended
#                    with task_kwargs={"start_url": payload}.
# safe_id          : filesystem-safe slug used for the per-target output dir.
# ---------------------------------------------------------------------------

# How many outer steps per episode.
DEFAULT_MAX_STEPS = 8
# Default episode count per URL/target when ``--episodes`` is not given.
DEFAULT_EPISODES = 1
# Anti-noop: force a different action after this many consecutive steps
# whose URL+focused-bid is identical AND no error was raised.
_MAX_CONSECUTIVE_NOOPS = 2
# Anti-error: force a different action_type/bid after this many consecutive
# steps that hit ``last_action_error`` with the same action.
_MAX_CONSECUTIVE_ERRORS = 2
# Number of recent action results to surface in the action-selection prompt.
_HISTORY_WINDOW = 5
# Substrings (case-insensitive) on a node's text/role that mark it as a
# cookie / consent / GDPR accept button. The actor pre-empts the LLM and
# auto-clicks the first such bid to unblock benchmarks (esp. assistantbench
# starting on google.com behind a consent wall).
_CONSENT_ACCEPT_KEYWORDS = (
    "accept all", "accept cookies", "i agree", "i accept", "agree to all",
    "agree all", "agree", "got it", "allow all", "allow cookies",
    "tout accepter",                 # fr
    "alle akzeptieren", "akzeptieren", "akzeptiere",  # de
    "accetta tutto", "accetto",      # it
    "aceptar todo", "aceptar",       # es
    "aceitar tudo", "aceitar",       # pt
    "hyväksy kaikki", "hyväksy",     # fi
    "godta alle", "godta",           # no
    "godkänn alla", "godkänn",       # sv
    "acepteer alle", "acepteer",     # nl approx
    "全部承諾", "承諾", "全て承認",    # ja
    "全部同意", "同意",               # zh
    "동의", "모두 동의",               # ko
)
# Default token budgets (output cap).
_ACTION_MAX_TOKENS = 400
_SCHEMA_MAX_TOKENS = 4000
# Reasoning models burn output tokens on hidden thinking — give them more.
_SCHEMA_MAX_TOKENS_REASONING = 12000
# Cap on entities per schema (keeps the action prompt focused).
_DEFAULT_MAX_ENTITIES = 25
# Cap on candidate-action list emitted to the actor LLM.
_MAX_CANDIDATE_ACTIONS = 18

# Models that require ``max_completion_tokens`` (no ``temperature``).
_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:gpt-5(?:[\.\-]\w+)?|o[134](?:[\.\-]\w+)?)(?:$|[^\w])",
    re.IGNORECASE,
)


def _is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI-style reasoning models (gpt-5.x, o1/o3/o4)."""
    if not model:
        return False
    return bool(_REASONING_MODEL_RE.search(model))


def _url_safe(url: str) -> str:
    """Best-effort filesystem-safe slug for a URL."""
    s = (url or "").lower()
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    out: List[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in "-._/":
            out.append("_")
    slug = "".join(out).strip("_")[:120] or "page"
    return slug


def _task_safe(env_id: str) -> str:
    """Filesystem-safe slug for a BrowserGym task id (``browsergym/foo.42``)."""
    s = env_id.replace("browsergym/", "", 1).replace("/", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-._":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")[:140] or "task"


def _import_optional_task_suites() -> Tuple[List[str], List[str]]:
    """Import all known BrowserGym task suite packages, returning (ok, fail).

    Each successful import registers ``browsergym/<suite>.*`` env ids in
    Gymnasium's global registry.  Suites that fail to import are simply
    skipped — the user only needs the subset they're targeting via
    ``--tasks``.  ``browsergym.core`` (which registers ``openended``) is
    imported separately, eagerly, in ``main()``.
    """
    ok: List[str] = []
    fail: List[str] = []
    for mod_name in _OPTIONAL_TASK_SUITE_MODULES:
        try:
            __import__(mod_name)
            ok.append(mod_name)
        except Exception as exc:
            fail.append(f"{mod_name}: {type(exc).__name__}: {str(exc)[:120]}")
    return ok, fail


def _list_registered_task_ids() -> Dict[str, List[str]]:
    """Return registered BrowserGym task ids bucketed by suite prefix."""
    import gymnasium as gym
    buckets: Dict[str, List[str]] = {}
    for k in gym.envs.registry.keys():
        if not k.startswith("browsergym/"):
            continue
        prefix = k.split("/", 1)[1].split(".")[0] or "openended"
        buckets.setdefault(prefix, []).append(k)
    for v in buckets.values():
        v.sort()
    return buckets


def _suite_of(env_id: str) -> str:
    """``browsergym/miniwob.click-button`` -> ``"miniwob"``."""
    if not env_id.startswith("browsergym/"):
        return ""
    return env_id.split("/", 1)[1].split(".")[0]


# (suite, env_var, hint) — preflight check on ``--tasks``: if any task in
# the suite is requested and the env var is missing, fail fast with a clear
# install message instead of crashing inside env.reset() per-episode.
_SUITE_INFRA_REQUIREMENTS: List[Tuple[str, List[str], str]] = [
    ("miniwob",
     ["MINIWOB_URL"],
     "MiniWoB++ HTML pages are not bundled with the pip package. Clone "
     "https://github.com/Farama-Foundation/miniwob-plusplus.git (frozen "
     "commit 7fd85d71a4b60325c6585396ec4f48377d049838) and export "
     "MINIWOB_URL=file:///path/to/miniwob-plusplus/miniwob/html/miniwob/ . "
     "The launcher script auto-resolves common locations."),
    ("webarena",
     # Exact list from browsergym.webarena's __init__ assertion (7 vars,
     # all required even if a particular task only touches one site).
     ["WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT", "WA_GITLAB",
      "WA_WIKIPEDIA", "WA_MAP", "WA_HOMEPAGE"],
     "WebArena requires the self-hosted Shopping / Shopping-Admin / "
     "Reddit / GitLab / Wikipedia / Map / Homepage Docker images. "
     "See github.com/web-arena-x/webarena#setup-instructions."),
    ("visualwebarena",
     # Exact list from browsergym.visualwebarena's __init__ assertion.
     ["VWA_SHOPPING", "VWA_REDDIT", "VWA_WIKIPEDIA", "VWA_HOMEPAGE",
      "VWA_CLASSIFIEDS", "VWA_CLASSIFIEDS_RESET_TOKEN"],
     "VisualWebArena requires the self-hosted Shopping / Reddit / Wikipedia "
     "/ Homepage / Classifieds Docker images plus a reset token. "
     "See github.com/web-arena-x/visualwebarena."),
    # assistantbench loads from the HF dataset cache and points the browser at
    # public URLs (e.g. google.com); no extra env vars are needed.
    # openended just needs a --urls argument; nothing to check here.
]


def _preflight_task_infra(task_ids: List[str]) -> List[str]:
    """Return a list of error strings; empty list means infra is OK."""
    if not task_ids:
        return []
    needed = {_suite_of(t) for t in task_ids}
    errors: List[str] = []
    for suite, env_vars, hint in _SUITE_INFRA_REQUIREMENTS:
        if suite not in needed:
            continue
        missing = [v for v in env_vars if not os.environ.get(v)]
        if missing:
            errors.append(
                f"  [{suite}] missing env var(s): {', '.join(missing)}\n"
                f"    {hint}"
            )
    return errors


# ---------------------------------------------------------------------------
# Image / observation helpers
# ---------------------------------------------------------------------------

def _to_pil(image: Any):
    """Coerce ``obs['screenshot']`` (PIL/np/bytes) into a single PIL RGB."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def _save_frame(image: Any, path: Path) -> Optional[str]:
    pil = _to_pil(image)
    if pil is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pil.save(str(path), format="PNG")
        return str(path)
    except Exception:
        return None


def _flatten_axtree(obs: Dict[str, Any], max_chars: int = 3000) -> str:
    """Flatten ``obs['axtree_object']`` into a compact text dump."""
    axtree = obs.get("axtree_object")
    if axtree is None:
        return ""

    try:
        from browsergym.utils.obs import flatten_axtree_to_str  # type: ignore
        text = flatten_axtree_to_str(
            axtree, extra_properties=obs.get("extra_element_properties", {}),
        )
    except Exception:
        lines: List[str] = []
        for node in axtree.get("nodes", [])[:80]:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            bid = node.get("browsergym_id", "")
            if not (role and (name or bid)):
                continue
            lines.append(f"[{bid}] {role}: {name[:60]}".rstrip())
        text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _extract_goal(obs: Dict[str, Any]) -> str:
    goal = obs.get("goal", "") or ""
    if not goal:
        goal_obj = obs.get("goal_object", ()) or ()
        goal = " ".join(
            m.get("text", "") for m in goal_obj if m.get("type") == "text"
        )
    return goal.strip() or "Explore the page"


def _extract_goal_images(obs: Dict[str, Any]) -> List[Any]:
    """Pull goal-side images out of ``goal_object``.

    BrowserGym's VisualWebArena task fills ``goal_object`` with OpenAI-
    style multimodal content: a sequence of dicts including
    ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
    entries that describe the visual reference (e.g. "buy a chair like
    *this image*"). The base actor flow drops these on the floor when it
    flattens the goal to text — meaning the VLM never sees the reference
    image and the task is unsolvable. This helper reconstructs the PIL
    images from the data URIs so we can forward them as additional image
    parts in the user message.

    Returns an (possibly empty) list of PIL.Image objects in goal order.
    """
    goal_obj = obs.get("goal_object") or ()
    if not goal_obj:
        return []
    try:
        from PIL import Image
    except ImportError:
        return []
    import base64

    images: List[Any] = []
    for entry in goal_obj:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "image_url":
            continue
        url = (entry.get("image_url") or {}).get("url") or ""
        if not url.startswith("data:"):
            # Skip remote URLs — fetching them at agent step time would
            # add a lot of latency and is not needed for VWA which
            # always inlines as data URIs.
            continue
        try:
            _header, _, b64 = url.partition(",")
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            logger.debug("goal image decode failed: %s", exc)
            continue
        images.append(img)
    return images


def _render_som_screenshot(obs: Dict[str, Any]) -> Optional[Any]:
    """Return a Set-of-Marks-annotated PIL image of the current viewport.

    Uses BrowserGym's built-in ``overlay_som`` helper, which draws a
    dashed bounding box + black tag containing the bid for every
    interactable element flagged with ``set_of_marks=True`` in
    ``extra_element_properties``. With this overlay the VLM can read
    bids straight off the pixels rather than having to cross-reference
    the AXTree text — empirically the single biggest leverage point for
    web-VLM action accuracy (WebVoyager / SeeAct / Aria-UI all rely on
    SoM).

    Returns ``None`` if the screenshot or extra_element_properties are
    unavailable, so the caller can fall back to the plain screenshot.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return None
    extras = obs.get("extra_element_properties") or {}
    try:
        from browsergym.utils.obs import overlay_som  # type: ignore
    except Exception as exc:
        logger.debug("overlay_som unavailable: %s", exc)
        return None
    try:
        arr = np.asarray(screenshot)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            return None
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        som_img = overlay_som(arr, extras)
        return som_img.convert("RGB") if hasattr(som_img, "convert") else som_img
    except Exception as exc:
        logger.debug("overlay_som failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# OpenAI client / model routing
# ---------------------------------------------------------------------------

def _build_client_and_route(
    *, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """Return ``(client, routed_model)`` or ``(None, model)`` on failure."""
    client = None
    if make_openai_client is not None:
        try:
            client = make_openai_client(api_key=api_key, base_url=base_url)
        except Exception:
            client = None

    if client is None:
        kw: Dict[str, Any] = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        try:
            client = openai.OpenAI(**kw) if kw else openai.OpenAI()
        except Exception:
            return None, model

    if api_key or base_url:
        return client, model
    if effective_openai_model is not None:
        try:
            return client, effective_openai_model(model)
        except Exception:
            return client, model
    return client, model


_VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high")


def _maybe_disable_thinking_kwargs(model: Any, tool_choice: Any) -> Dict[str, Any]:
    """Return ``extra_body`` kwargs to disable Qwen ``thinking`` mode.

    Qwen3/3.5/3.6 multimodal flagships (e.g. ``qwen/qwen3.5-plus-20260420``,
    ``qwen/qwen3.6-plus``) ship with thinking-mode ON.  When a strict
    ``tool_choice={"type":"function",...}`` payload is sent, the upstream
    DashScope endpoint rejects it with HTTP 400
    (``InvalidParameter ... in thinking mode``).  Our actor pipeline
    relies on strict tool-choice to force structured action JSON, so the
    only safe option is to turn thinking OFF.

    We forward both supported parameter names so the same payload works
    whether the endpoint is local-vLLM or DashScope/OpenRouter:

      - DashScope / OpenRouter:    ``extra_body.enable_thinking = False``
      - vLLM-OpenAI-compat:        ``extra_body.chat_template_kwargs
                                     .enable_thinking = False``

    Each server silently ignores the parameter it does not recognise, so
    the payload is portable.  No-ops for non-Qwen models and for calls
    that don't set ``tool_choice``.
    """
    if not isinstance(model, str):
        return {}
    if "qwen" not in model.lower():
        return {}
    if tool_choice is None:
        return {}
    return {
        "extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[list] = None,
    tool_choice: Any = None,
    reasoning_effort: Optional[str] = None,
):
    """Cross-model chat-completion wrapper.

    Reasoning models (gpt-5.x, o1/o3/o4) reject ``max_tokens`` and
    ``temperature``; they require ``max_completion_tokens`` and burn part of
    that budget on hidden thinking tokens.  Detect reasoning models up front
    and route them through with a generous output cap; classic models keep
    the legacy path with a single fallback retry.

    ``reasoning_effort`` (one of ``minimal`` / ``low`` / ``medium`` / ``high``)
    is forwarded only for reasoning models; ignored otherwise.  Setting
    ``minimal`` suppresses hidden thinking tokens — the right default for
    cold-start data generation, where the SFT student never consumes the
    teacher's hidden chain anyway.
    """
    if _is_reasoning_model(model):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max(6000, max_tokens * 4),
        }
        if reasoning_effort:
            if reasoning_effort not in _VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort must be one of {_VALID_REASONING_EFFORTS}, "
                    f"got {reasoning_effort!r}"
                )
            kwargs["reasoning_effort"] = reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        kwargs.update(_maybe_disable_thinking_kwargs(model, tool_choice))
        return client.chat.completions.create(**kwargs)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    kwargs.update(_maybe_disable_thinking_kwargs(model, tool_choice))

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc)
        if not (
            "max_completion_tokens" in msg
            or ("max_tokens" in msg and "Unsupported" in msg)
            or ("temperature" in msg and "Unsupported" in msg)
        ):
            raise
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        kwargs["max_completion_tokens"] = max(6000, max_tokens * 5)
        return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Lazy imports for optional deps
# ---------------------------------------------------------------------------

def _import_browsergym_heuristic():
    from browsergym_wrapper.heuristic import obs_to_schema as bg_obs_to_schema
    return bg_obs_to_schema


def _import_browser_tools():
    """List-valid-actions handler from the browser tool registry."""
    try:
        from browsergym_wrapper.tools import build_browser_registry
        return build_browser_registry
    except Exception as exc:
        logger.debug("browsergym_wrapper.tools unavailable: %s", exc)
        return None


def _import_schema_helpers():
    try:
        from vlm_wrapper.schema import (
            build_system_prompt,
            build_user_message,
            parse_schema_output,
        )
        return {
            "build_system_prompt": build_system_prompt,
            "build_user_message": build_user_message,
            "parse_schema_output": parse_schema_output,
        }
    except Exception as exc:
        logger.debug("vlm_wrapper.schema unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Stage 1 — visual schema generation (gpt-5.5 vision)
# ---------------------------------------------------------------------------

# Pre-compiled patterns for the lenient parser.
_LENIENT_STATE_OPEN_RE = re.compile(r"<state\b[^>]*>", re.IGNORECASE)
_LENIENT_STATE_CLOSE_RE = re.compile(r"</state\s*>", re.IGNORECASE)
_LENIENT_FENCE_RE = re.compile(
    r"^\s*```(?:xml|html|text|state)?\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LENIENT_SECTION_RE = re.compile(
    r"<(entities|attributes|affordances|relations|state_flags|targets|actions)\b",
    re.IGNORECASE,
)


def _lenient_parse_schema(raw: str, strict_parser) -> Tuple[Optional[str], str]:
    """Salvage a ``<state>...</state>`` schema from messy VLM output.

    Returns ``(parsed_schema_or_None, recovery_kind)``.
    """
    if not raw:
        return None, ""

    parsed = strict_parser(raw)
    if parsed:
        return parsed, "strict"

    cleaned = _LENIENT_FENCE_RE.sub("", raw).strip()
    if cleaned and cleaned != raw:
        parsed = strict_parser(cleaned)
        if parsed:
            return parsed, "fenced"

    candidate = cleaned or raw

    open_m = _LENIENT_STATE_OPEN_RE.search(candidate)
    close_m = _LENIENT_STATE_CLOSE_RE.search(candidate)
    if open_m and not close_m:
        salvaged = candidate[open_m.start():].rstrip() + "\n</state>"
        parsed = strict_parser(salvaged)
        if parsed:
            return parsed, "truncated"

    if not open_m and _LENIENT_SECTION_RE.search(candidate):
        salvaged = "<state>\n" + candidate.strip() + "\n</state>"
        parsed = strict_parser(salvaged)
        if parsed:
            return parsed, "untagged"

    return None, ""


def generate_schema_from_image(
    *,
    pil_image,
    obs: Dict[str, Any],
    task_id: str,
    goal: str,
    step: int,
    candidate_actions: List[str],
    client: Any,
    routed_model: str,
    schema_helpers: Dict[str, Any],
    canonical_fallback: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = _SCHEMA_MAX_TOKENS,
    max_entities: int = _DEFAULT_MAX_ENTITIES,
    use_som: bool = True,
    goal_images: Optional[List[Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call gpt-5.5 (vision) on the screenshot to produce a ``<state>`` schema.

    The screenshot is the **primary** input; the AXTree-as-text rides along
    as grounding context (so the VLM can emit correct ``bid``s).  Returns a
    dict with the parsed ``schema`` (or ``None``), raw output, finish reason,
    and any exception captured.

    Visual enhancements:
      - When ``use_som`` is True (default) and BrowserGym's
        ``extra_element_properties`` provides Set-of-Marks bbox metadata,
        we send the SoM-overlayed screenshot (bids drawn on the pixels)
        rather than the raw screenshot. This is the canonical input
        format for VLM web agents.
      - When ``goal_images`` is non-empty (typically VisualWebArena's
        reference product images extracted from ``goal_object``), each
        image is appended as an additional ``image_url`` part so the VLM
        can ground the goal visually.

    On failure or no schema, falls back to ``canonical_fallback`` (the
    deterministic AXTree-walked schema from
    :mod:`browsergym_wrapper.heuristic`).
    """
    if pil_image is None or schema_helpers is None or client is None:
        return {
            "schema": canonical_fallback,
            "raw": "",
            "source": "fallback_canonical" if canonical_fallback else "no_image_or_client",
            "error": None,
        }

    system = schema_helpers["build_system_prompt"]("browser", max_entities=max_entities)

    # Prefer the Set-of-Marks overlay when available — it's strictly more
    # informative than the raw screenshot for bid-targeted actions.
    primary_image = pil_image
    som_used = False
    if use_som:
        som_img = _render_som_screenshot(obs)
        if som_img is not None:
            primary_image = som_img
            som_used = True

    extra_parts: List[str] = []
    url = obs.get("url", "") or ""
    if url:
        extra_parts.append(f"URL: {url}")
    last_action = obs.get("last_action", "") or ""
    if last_action:
        extra_parts.append(f"Last action: {last_action}")
    last_error = obs.get("last_action_error", "") or ""
    if last_error:
        extra_parts.append(f"Last action error: {last_error[:200]}")
    focused = obs.get("focused_element_bid", "") or ""
    if focused:
        extra_parts.append(f"Focused element bid: {focused}")
    if som_used:
        extra_parts.append(
            "The screenshot has SET-OF-MARKS overlays: every interactable "
            "element is wrapped in a dashed bounding box and labelled with "
            "its bid in a black tag. Read bids directly off the image."
        )
    if goal_images:
        extra_parts.append(
            f"NOTE: {len(goal_images)} GOAL reference image(s) follow the "
            "page screenshot. Use them to ground product / item /"
            " visual-similarity references in the goal text."
        )
    axtree_text = _flatten_axtree(obs, max_chars=3000)
    if axtree_text:
        extra_parts.append(
            "AXTree (for element bid grounding, truncated):\n" + axtree_text
        )
    if candidate_actions:
        extra_parts.append(
            "Candidate actions for this page (you MUST copy one of these "
            "verbatim into <actions>; do NOT rename or reformat):\n"
            + "\n".join(f"  - {a}" for a in candidate_actions[:_MAX_CANDIDATE_ACTIONS])
        )
    extra_context = "\n\n".join(extra_parts)

    user_content = schema_helpers["build_user_message"](
        primary_image,
        domain="browser",
        task_id=task_id,
        goal=goal,
        step=step,
        extra_context=extra_context,
    )
    # Append goal-side reference images (VWA tasks). build_user_message
    # returns ``[image, text]``; we extend to ``[image, text, goal_img_1,
    # goal_img_2, ...]`` so each goal image rides as an OpenAI-format
    # image_url part.
    if goal_images:
        from vlm_wrapper.schema import encode_image_b64  # local import: optional dep
        for gi in goal_images[:4]:  # cap to avoid blowing up the prompt
            try:
                b64 = encode_image_b64(gi)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                })
            except Exception as exc:
                logger.debug("goal image encode failed: %s", exc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    parsed: Optional[str] = None
    err: Optional[str] = None
    finish_reason: Optional[str] = None
    recovery: str = ""
    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if resp.choices:
            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
        parsed, recovery = _lenient_parse_schema(
            raw, schema_helpers["parse_schema_output"]
        )
    except Exception as exc:
        err = repr(exc)
        logger.warning("[schema-VLM] step %d failed: %s", step, exc)

    base_meta: Dict[str, Any] = {
        "raw": raw,
        "raw_full_len": len(raw),
        "finish_reason": finish_reason,
        "recovery": recovery,
        "error": err,
    }

    if parsed:
        if recovery and recovery != "strict":
            logger.info(
                "[schema-VLM] step %d salvaged via '%s' (finish_reason=%s, raw_len=%d)",
                step, recovery, finish_reason, len(raw),
            )
        return {"schema": parsed, "source": "vlm", **base_meta}

    if raw and finish_reason == "length":
        logger.warning(
            "[schema-VLM] step %d response truncated (finish_reason=length, "
            "raw_len=%d) — consider raising _SCHEMA_MAX_TOKENS",
            step, len(raw),
        )

    return {
        "schema": canonical_fallback,
        "source": "fallback_canonical" if canonical_fallback else "vlm_no_schema",
        **base_meta,
    }


# ---------------------------------------------------------------------------
# Action vocabulary construction
# ---------------------------------------------------------------------------

# Standard navigation actions BrowserGym always accepts.
_GLOBAL_BROWSER_ACTIONS: List[str] = [
    "scroll(0, 300)",
    "scroll(0, -300)",
    "go_back()",
    "go_forward()",
    "noop()",
]

# Validates the action string we intend to send to ``env.step(...)``.
_BROWSERGYM_ACTION_RE = re.compile(
    r"^\s*("
    r"click\([^)]*\)"
    r"|fill\([^,]+,\s*\".*\"\s*\)"
    r"|check\([^)]*\)"
    r"|press\([^)]*\)"
    r"|hover\([^)]*\)"
    r"|select_option\([^)]*\)"
    r"|focus\([^)]*\)"
    r"|clear\([^)]*\)"
    r"|scroll\(\s*-?\d+\s*,\s*-?\d+\s*\)"
    r"|go_back\(\s*\)"
    r"|go_forward\(\s*\)"
    r"|new_tab\(\s*\)"
    r"|tab_close\(\s*\)"
    r"|tab_focus\([^)]*\)"
    r"|goto\([^)]*\)"
    r"|noop\([^)]*\)"
    r")\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _list_clickable_bids(
    registry, max_results: int = _MAX_CANDIDATE_ACTIONS,
) -> List[Dict[str, Any]]:
    """Return ``list_valid_actions`` results from the browser tool registry."""
    if registry is None:
        return []
    try:
        result = registry.call("list_valid_actions", {})
    except Exception as exc:
        logger.debug("list_valid_actions failed: %s", exc)
        return []
    actions = result.get("actions") if isinstance(result, dict) else None
    return list(actions or [])[:max_results]


def _build_candidate_actions(
    *, obs: Dict[str, Any], registry,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build a list of candidate action strings + structured metadata.

    Combines:
      - ``list_valid_actions`` from :mod:`browsergym_wrapper.tools`
        (each interactive bid → ``click(bid)`` / ``fill(bid, "...")`` /
        ``check(bid)``) when the registry resolves.
      - A small set of standard navigation actions (always available).

    Returns ``(strings, meta_list)`` where ``meta_list`` carries the parsed
    role/name/bid for each interactive entry (used by the actor LLM prompt).
    """
    meta: List[Dict[str, Any]] = []
    strings: List[str] = []

    seen: set = set()
    raw_actions = _list_clickable_bids(registry)
    for entry in raw_actions:
        a = (entry or {}).get("action") if isinstance(entry, dict) else None
        if not a or a in seen:
            continue
        seen.add(a)
        strings.append(a)
        meta.append({
            "action": a,
            "role": (entry or {}).get("role"),
            "name": (entry or {}).get("name"),
        })

    for a in _GLOBAL_BROWSER_ACTIONS:
        if a not in seen:
            seen.add(a)
            strings.append(a)
            meta.append({"action": a, "role": "navigation", "name": None})

    return strings[:_MAX_CANDIDATE_ACTIONS], meta[:_MAX_CANDIDATE_ACTIONS]


# ---------------------------------------------------------------------------
# Stage 2 — schema-driven action selection (gpt-5.5)
# ---------------------------------------------------------------------------

_ACTOR_SYSTEM_PROMPT = (
    "You are an Actor Agent for the COS-PLAY web-agent pipeline, driving a "
    "BrowserGym environment.\n"
    "On every step you receive a structured ``<state>...</state>`` schema "
    "describing the visual state of the page (entities have ``bid`` ids "
    "matching the AXTree's set-of-marks), plus a list of candidate "
    "BrowserGym action strings.\n\n"
    "Your job:\n"
    "1. Reason briefly (≤3 sentences) about the schema: which entity matters, "
    "what is the current sub-goal, and why one action best advances it.\n"
    "2. Pick EXACTLY ONE action by calling the ``choose_action`` function. "
    "You may either:\n"
    "   - Echo a candidate string verbatim (STRONGLY preferred), OR\n"
    "   - Specify ``action_type`` + ``bid`` (and ``text`` for ``fill``) so the "
    "harness can construct a typed BrowserGym action.\n\n"
    "CRITICAL: BrowserGym requires bids to be QUOTED string literals. Always "
    "write ``click(\"a17\")`` and ``fill(\"a23\", \"hello\")`` — NEVER "
    "``click(a17)`` or ``click(12)``. The candidate strings are already "
    "correctly quoted; copy them verbatim.\n\n"
    "If recent action history shows an action had NO EFFECT (URL/state did "
    "not change and no error), choose a DIFFERENT action this turn.\n\n"
    "Always respond by calling the ``choose_action`` function."
)


def _build_action_tools(candidate_actions: List[str]) -> list:
    """OpenAI function-calling tool definition for browser action selection.

    Two ways to specify the action:
      1. ``action_string`` — a verbatim BrowserGym action.  Validated by
         ``_BROWSERGYM_ACTION_RE``; usually the best match for the
         ``candidate_actions`` list.
      2. ``action_type`` + ``bid`` (+ ``text`` / ``key``) — a structured
         description. Useful when the model wants to ``fill`` an input box
         with custom text not present in the candidate list.
    """
    enum_types = [
        "click", "fill", "check", "press", "hover", "select_option",
        "focus", "clear", "scroll_down", "scroll_up", "go_back",
        "go_forward", "new_tab", "tab_close", "noop",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": (
                    "Choose a single BrowserGym action for this turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Brief chain-of-thought (≤3 sentences) "
                                "grounded in the schema entities."
                            ),
                        },
                        "action_string": {
                            "type": "string",
                            "description": (
                                "Verbatim BrowserGym action string. Bids "
                                "MUST be quoted string literals, e.g. "
                                "'click(\"a17\")' or "
                                "'fill(\"a23\", \"hello\")'. NEVER write "
                                "'click(a17)' or 'click(12)' — they will be "
                                "parsed as int/identifier and rejected. "
                                "Prefer copying one of these candidates "
                                "verbatim: "
                                + ", ".join(candidate_actions[:_MAX_CANDIDATE_ACTIONS])
                            ),
                        },
                        "action_type": {
                            "type": "string",
                            "enum": enum_types,
                            "description": (
                                "Structured action type. Used when "
                                "action_string is missing or invalid."
                            ),
                        },
                        "bid": {
                            "type": "string",
                            "description": (
                                "Element bid for click/fill/check/press/"
                                "hover/select_option/focus/clear."
                            ),
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Text payload for fill (input value) or "
                                "select_option (option label)."
                            ),
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Key to press for the press action (e.g. "
                                "'Enter', 'Tab', 'ArrowDown')."
                            ),
                        },
                        "scroll_dy": {
                            "type": "integer",
                            "description": (
                                "Pixels to scroll vertically. Positive = "
                                "down. Used with action_type=scroll_down/up "
                                "if action_string is missing."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }
    ]


_BID_RE = re.compile(r"\(\s*([A-Za-z0-9_-]+)")


def _bid_from_action(action: str) -> Optional[str]:
    """Return the first arg of the action call, e.g. ``click(123)`` -> ``"123"``."""
    if not action:
        return None
    m = _BID_RE.search(action)
    return m.group(1) if m else None


def _format_history_block(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Recent action history (newest last):"]
    for entry in history[-_HISTORY_WINDOW:]:
        effect = (
            "ERROR" if entry.get("error")
            else ("NO EFFECT" if entry.get("noop") else "ok")
        )
        lines.append(
            f"  - {entry.get('action')!r} -> {effect}"
            + (f" (err: {entry.get('error_text','')[:80]})" if entry.get("error") else "")
        )
    recent = history[-_HISTORY_WINDOW:]
    noop_actions = sorted({e["action"] for e in recent if e.get("noop")})
    if noop_actions:
        lines.append(
            f"WARNING: Action(s) {noop_actions} had no effect. Pick a DIFFERENT action."
        )
    # Surface recently-errored bids so the model stops retrying the same dead
    # bid. Especially important for fill() where a stale bid will keep
    # erroring after the page re-renders.
    bad_bids = sorted({
        b for b in (
            _bid_from_action(e["action"]) for e in recent if e.get("error")
        ) if b is not None
    })
    if bad_bids:
        lines.append(
            f"AVOID these bids in the next action — they errored recently: {bad_bids}. "
            "Pick a DIFFERENT bid (re-read the schema for the current bid of the target element) "
            "or switch to a different action_type (e.g. click before fill)."
        )
    return "\n".join(lines) + "\n"


def _structured_to_action_string(args: Dict[str, Any]) -> Optional[str]:
    """Build a BrowserGym action string from structured function-call args.

    Used as a fallback when the ``action_string`` field is empty/invalid.
    """
    atype = (args.get("action_type") or "").lower().strip()
    bid = (args.get("bid") or "").strip()
    text = args.get("text", "")
    key = (args.get("key") or "").strip()
    dy = args.get("scroll_dy")
    if isinstance(dy, str):
        try:
            dy = int(dy)
        except Exception:
            dy = None

    # NOTE: BrowserGym's high-level action grammar (parsers.py) requires every
    # bid argument to be a STRING literal — bare numerics like ``click(12)`` get
    # parsed as int and ``get_elem_by_bid`` raises ValueError. We always emit
    # bids inside double quotes so they round-trip cleanly through
    # ``highlevel_action_parser`` → ``repr(arg)`` → ``exec()``.
    if atype == "click" and bid:
        return f'click("{bid}")'
    if atype == "check" and bid:
        return f'check("{bid}")'
    if atype == "focus" and bid:
        return f'focus("{bid}")'
    if atype == "clear" and bid:
        return f'clear("{bid}")'
    if atype == "hover" and bid:
        return f'hover("{bid}")'
    if atype == "fill" and bid:
        # Escape backslashes + double quotes for the action string.
        text_str = (text or "").replace("\\", "\\\\").replace("\"", "\\\"")
        return f'fill("{bid}", "{text_str}")'
    if atype == "select_option" and bid:
        text_str = (text or "").replace("\"", "\\\"")
        return f'select_option("{bid}", "{text_str}")'
    if atype == "press" and bid and key:
        key_str = key.replace("\"", "\\\"")
        return f'press("{bid}", "{key_str}")'
    if atype == "scroll_down":
        d = dy if isinstance(dy, int) else 300
        return f"scroll(0, {abs(d)})"
    if atype == "scroll_up":
        d = dy if isinstance(dy, int) else 300
        return f"scroll(0, -{abs(d)})"
    if atype == "go_back":
        return "go_back()"
    if atype == "go_forward":
        return "go_forward()"
    if atype == "new_tab":
        return "new_tab()"
    if atype == "tab_close":
        return "tab_close()"
    if atype == "noop":
        return "noop()"
    return None


# Functions whose first positional argument is a ``bid`` and MUST be a
# Python string literal in the action source (BrowserGym enforces this in
# ``get_elem_by_bid``). Used by ``_autoquote_bids`` and
# ``_validate_action_string``.
_BID_FIRST_ARG_FNS = frozenset({
    "click", "fill", "check", "uncheck", "hover", "focus",
    "clear", "press", "select_option",
})

# Match a function call whose first positional arg is a *bare* identifier
# (not already quoted) — e.g. ``click(12)``, ``fill(a23, "hi")``,
# ``select_option(e7, "Croatia")``. Captures (whitespace, fname, bare_arg,
# delimiter) so the autoquote shim can rewrap the bid in double quotes
# while leaving the rest of the call intact.
_BARE_BID_CALL_RE = re.compile(
    r"^(\s*)(" + "|".join(sorted(_BID_FIRST_ARG_FNS)) + r")"
    r"\(\s*([A-Za-z_][A-Za-z0-9_-]*|\d+)\s*([,)])"
)


def _autoquote_bids(action: str) -> str:
    """Defense-in-depth: rewrite ``click(12)`` → ``click("12")`` etc.

    Only touches the FIRST positional argument of a known bid-taking
    function call, and only when that argument is a bare identifier or
    bare integer (i.e. not already quoted). Idempotent: already-quoted
    bids round-trip unchanged. Useful when the LLM still emits unquoted
    bids despite the prompt telling it to quote them.
    """
    if not action:
        return action
    m = _BARE_BID_CALL_RE.match(action)
    if not m:
        return action
    return f'{m[1]}{m[2]}("{m[3]}"{m[4]}' + action[m.end():]


def _validate_action_string(action: str) -> bool:
    """Strict structural check via BrowserGym's own parser.

    Rejects strings that:
      - the high-level parser cannot parse at all, OR
      - call a bid-taking function with a non-string first argument.

    This is what catches ``click(12)`` *before* it reaches ``env.step()``.
    The earlier regex-only validator was permissive: any ``click([^)]*)``
    matched, including the broken-int-bid form.
    """
    if not action:
        return False
    s = action.strip()
    # Quick legacy regex pre-filter to weed out obviously malformed strings
    # (keeps the failure mode user-friendly when the parser is missing).
    if not _BROWSERGYM_ACTION_RE.match(s):
        return False
    # Real parse — single source of truth on whether ``env.step()`` will
    # accept the string.
    try:
        from browsergym.core.action.parsers import highlevel_action_parser
    except Exception:  # pragma: no cover — parser ships with browsergym.core
        return True
    try:
        calls = highlevel_action_parser.parse_string(s, parse_all=True).as_list()
    except Exception:
        return False
    if not calls:
        return False
    fname, fargs = calls[0]
    if fname in _BID_FIRST_ARG_FNS:
        if not fargs or not isinstance(fargs[0], str):
            return False
    return True


def _canonicalize_to_candidate(
    raw: str, candidates: List[str],
) -> Optional[str]:
    """Map ``raw`` LLM output back to one of the candidate strings.

    Mirrors ``_canonicalize_action`` in ``generate_cold_start_actor_gymv``
    but operates on full BrowserGym call strings rather than action names.
    Returns ``None`` if no plausible match is found.

    Strategy:
      1. exact match
      2. case-insensitive match
      3. numeric index ("3" or "3." -> candidates[2])
      4. canonical form (collapse whitespace, strip outer parens spaces)
    """
    if not raw or not candidates:
        return None
    s = raw.strip().strip("`").strip()
    if not s:
        return None
    if s in candidates:
        return s
    lower_map = {c.lower(): c for c in candidates}
    if s.lower() in lower_map:
        return lower_map[s.lower()]
    m = re.match(r"^\s*(\d+)\s*[\.\)\-:]?\s*$", s)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    # Whitespace-collapse compare.
    norm = re.sub(r"\s+", "", s)
    for c in candidates:
        if re.sub(r"\s+", "", c) == norm:
            return c
    return None


def select_action(
    *,
    schema_text: Optional[str],
    obs: Dict[str, Any],
    candidate_actions: List[str],
    candidate_meta: List[Dict[str, Any]],
    task: str,
    step: int,
    history: List[Dict[str, Any]],
    client: Any,
    routed_model: str,
    temperature: float = 0.4,
    max_tokens: int = _ACTION_MAX_TOKENS,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Call gpt-5.5 with the schema → ``(action, reasoning, raw, error)``."""
    if not candidate_actions:
        candidate_actions = list(_GLOBAL_BROWSER_ACTIONS)
        candidate_meta = [
            {"action": a, "role": "navigation", "name": None}
            for a in _GLOBAL_BROWSER_ACTIONS
        ]
    if client is None:
        return None, None, "", "no_client"

    history_block = _format_history_block(history)
    schema_block = (
        schema_text.strip() if schema_text else
        "(no schema available — fall back to the URL + AXTree text below)"
    )

    candidate_lines = []
    for i, m in enumerate(candidate_meta[:_MAX_CANDIDATE_ACTIONS], start=1):
        suffix = ""
        if m.get("role") and m.get("name"):
            suffix = f"  # {m['role']}: {str(m['name'])[:60]}"
        elif m.get("role"):
            suffix = f"  # {m['role']}"
        candidate_lines.append(f"  {i}. {m['action']}{suffix}")
    candidate_block = "Candidate actions:\n" + "\n".join(candidate_lines)

    user_parts = [
        f"Task: {task}",
        f"URL: {obs.get('url', '') or ''}",
        f"Step: {step}",
        "",
        "Structured state schema:",
        schema_block,
        "",
        candidate_block,
    ]
    if not schema_text:
        ax_text = _flatten_axtree(obs, max_chars=2000)
        if ax_text:
            user_parts.extend([
                "",
                "AXTree (since no schema was parsed):",
                ax_text,
            ])
    user_parts.extend([
        "",
        history_block.strip(),
        "",
        "Pick the BEST action and call ``choose_action``. Prefer to copy a "
        "candidate verbatim into ``action_string``; otherwise specify "
        "``action_type`` + ``bid`` (+ ``text``).",
    ])
    user_content = "\n".join(p for p in user_parts if p is not None)

    tools = _build_action_tools(candidate_actions)

    raw = ""
    err: Optional[str] = None
    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=[
                {"role": "system", "content": _ACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            reasoning_effort=reasoning_effort,
        )
        choice = resp.choices[0]
        msg = choice.message
        raw = msg.content or ""

        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            raw_args = (
                getattr(tc, "arguments", None)
                or getattr(getattr(tc, "function", None), "arguments", None)
                or "{}"
            )
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            reasoning = args.get("reasoning") or None
            action_string = (args.get("action_string") or "").strip()

            # 1. Verbatim action_string path. Try to snap to a candidate first
            #    (exact match / case-insensitive / index), then fall through to
            #    the autoquote shim + parser-strict validator.
            if action_string:
                snapped = _canonicalize_to_candidate(action_string, candidate_actions)
                if snapped:
                    return snapped, reasoning, raw or json.dumps(args), None
                fixed = _autoquote_bids(action_string)
                if _validate_action_string(fixed):
                    return fixed, reasoning, raw or json.dumps(args), None

            # 2. Structured fallback: build the call from action_type + bid.
            structured = _structured_to_action_string(args)
            if structured and _validate_action_string(structured):
                return structured, reasoning, raw or json.dumps(args), None

        # 3. No tool call — try to extract a verbatim action from the raw text.
        for cand in candidate_actions:
            if cand in raw:
                return cand, None, raw, None

    except Exception as exc:
        err = repr(exc)
        logger.warning("[action-LLM] step %d failed: %s", step, exc)

    return None, None, raw, err


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _is_noop(prev_obs: Dict[str, Any], next_obs: Dict[str, Any]) -> bool:
    """Best-effort no-op detection for a browser env.

    URL + focused element + axtree-node count all unchanged AND no
    ``last_action_error`` ⇒ no-op.
    """
    if (next_obs.get("last_action_error") or "").strip():
        return False
    if (prev_obs.get("url") or "") != (next_obs.get("url") or ""):
        return False
    if (prev_obs.get("focused_element_bid") or "") != (next_obs.get("focused_element_bid") or ""):
        return False

    def _node_count(o: Dict[str, Any]) -> int:
        ax = o.get("axtree_object")
        if not isinstance(ax, dict):
            return -1
        return len(ax.get("nodes") or [])

    return _node_count(prev_obs) == _node_count(next_obs)


# Actions that *navigate away* from the current page. MiniWoB tasks
# terminate the moment the iframe leaves the task URL, so an override
# escalating to ``go_back()`` would destroy any chance of reward. We
# therefore exclude these from the override candidate pool.
_DESTRUCTIVE_NAV_ACTIONS = frozenset({
    "go_back()", "go_forward()", "new_tab()", "tab_close()",
})


def _pick_different(action: str, candidates: List[str]) -> str:
    """Pick a candidate that is neither ``action`` nor a destructive nav."""
    alts = [
        a for a in candidates
        if a != action and a not in _DESTRUCTIVE_NAV_ACTIONS
    ]
    if alts:
        return random.choice(alts)
    # Fall back to a non-destructive default; ``noop()`` is always safe.
    return "noop()"


_SCHEMA_ENTITY_RE = re.compile(
    r"^(e\d+)\[[^]]*\blabel=([^,\]]+?)(?:,[^]]*\bbid=([A-Za-z0-9_-]+))?",
    re.MULTILINE,
)


def _detect_consent_button_bid(
    obs: Dict[str, Any], canonical_schema: Optional[str] = None,
) -> Optional[str]:
    """Return the bid of a cookie/consent ACCEPT button if visible, else None.

    Strategy (most reliable first):
      1. Parse the canonical/heuristic schema text — every entity line is
         ``e<n>[type=element, label=<role> '<name>', bid=<bid>, ...]``,
         which is exactly the bid BrowserGym expects in ``click(bid)``.
      2. Fall back to ``extra_element_properties`` + AXTree walking.

    Only bids whose label matches a localized ``accept all`` keyword are
    returned, ranked by keyword priority (``accept all`` beats plain
    ``accept``).
    """
    candidates: List[Tuple[int, str, str]] = []  # (rank, bid, label)

    # 1. Schema-text parse (the safest source — these bids are guaranteed
    #    valid because the heuristic only emits entities for elements present
    #    in extra_element_properties).
    if canonical_schema:
        for m in _SCHEMA_ENTITY_RE.finditer(canonical_schema):
            label = (m.group(2) or "").lower()
            bid = m.group(3)
            if not bid:
                continue
            for rank, kw in enumerate(_CONSENT_ACCEPT_KEYWORDS):
                if kw in label:
                    candidates.append((rank, bid, label))
                    break

    # 2. extra_element_properties fallback — scan node text for keywords.
    if not candidates:
        extras = obs.get("extra_element_properties") or {}
        if isinstance(extras, dict):
            for bid, props in extras.items():
                if not isinstance(props, dict):
                    continue
                text_parts = []
                for k in ("name", "ariaLabel", "innerText", "value", "title"):
                    v = props.get(k)
                    if isinstance(v, str):
                        text_parts.append(v)
                blob = " ".join(text_parts).strip().lower()
                if not blob:
                    continue
                # Only auto-click clickable elements.
                if not (props.get("clickable")
                        or props.get("role") in ("button", "link", "menuitem")):
                    continue
                for rank, kw in enumerate(_CONSENT_ACCEPT_KEYWORDS):
                    if kw in blob:
                        candidates.append((rank, str(bid), blob[:80]))
                        break

    if not candidates:
        return None

    # Prefer the strongest keyword match (lowest rank).
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _capture_initial_obs(
    *, kind: str, payload: str, headless: bool, seed: Optional[int] = None,
):
    """Boot a real BrowserGym env and reset it.

    Parameters
    ----------
    kind : "task" | "url"
        ``"task"``: ``payload`` is a registered BrowserGym env id
        (e.g. ``"browsergym/miniwob.click-button"``); the task ships with
        its own goal + reward function, so no ``task_kwargs`` are needed.
        ``"url"``: ``payload`` is a live URL; we boot
        ``browsergym/openended`` with ``task_kwargs={"start_url": ...}``.
    headless : bool
        Whether to run Chromium headless. ``False`` is useful for
        interactive debugging.
    seed : int, optional
        Forwarded to ``env.reset(seed=seed)`` so episodes are reproducible.

    Returns ``(env, obs, info)`` where ``env`` is the live BrowserEnv
    (Playwright + Chromium), ``obs`` is the observation dict from
    ``env.reset()`` (carries ``screenshot``, ``axtree_object``,
    ``extra_element_properties``, ``focused_element_bid``, ``url``,
    ``last_action`` / ``last_action_error`` …), and ``info`` is the
    Gymnasium info dict.

    Raises ``ImportError`` if BrowserGym / Playwright is not installed,
    or ``ValueError`` if the requested env id is not registered.
    """
    import gymnasium as gym
    import browsergym.core  # noqa: F401  -- registers openended

    if kind == "task":
        env_id = payload
        if env_id not in gym.envs.registry:
            raise ValueError(
                f"BrowserGym env id {env_id!r} is not registered. "
                f"Did you forget to import the suite (e.g. browsergym.miniwob)? "
                f"Use --list_tasks to see registered ids."
            )
        env = gym.make(env_id, headless=headless)
    elif kind == "url":
        env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": payload},
            headless=headless,
        )
    else:
        raise ValueError(f"Unknown target kind: {kind!r}")

    try:
        obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        obs, info = env.reset()
    return env, dict(obs), dict(info or {})


def _step_env(
    env: Any, action: str,
) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
    """Step the live BrowserGym env once with ``action`` (a high-level string)."""
    obs, reward, term, trunc, info = env.step(action)
    return dict(obs), float(reward or 0.0), bool(term), bool(trunc), dict(info or {})


def run_actor_episode(
    *,
    target_kind: str,
    target_payload: str,
    target_safe_id: str,
    headless: bool,
    max_steps: int,
    client: Any,
    routed_model: str,
    fallback_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    use_vision: bool,
    temperature_action: float,
    temperature_schema: float,
    max_entities: int,
    frames_dir: Optional[Path],
    seed: Optional[int],
    verbose: bool,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one BrowserGym episode end-to-end and return ``(Episode, stats)``."""
    bg_obs_to_schema = _import_browsergym_heuristic()
    build_browser_registry = _import_browser_tools()

    if seed is not None:
        random.seed(seed)

    env, obs, _info = _capture_initial_obs(
        kind=target_kind, payload=target_payload, headless=headless, seed=seed,
    )

    if target_kind == "task":
        task_id = target_payload
        task_human = (
            f"Solve the BrowserGym task {target_payload}. "
            + (_extract_goal(obs) or "")
        ).strip()
    else:
        task_id = f"browsergym/openended/{target_safe_id}"
        task_human = (
            f"Browse {target_payload} to satisfy the page goal. "
            + (_extract_goal(obs) or "")
        ).strip()
    task = task_human
    goal = _extract_goal(obs)

    experiences: List[Experience] = []
    history: List[Dict[str, Any]] = []
    consecutive_noops = 0
    last_noop_action: Optional[str] = None
    consecutive_errors = 0
    last_error_bid: Optional[str] = None
    bad_bids: List[str] = []        # bids that errored at any point this ep
    consent_dismissed = False  # only auto-click cookie accept once per episode
    schema_calls = 0
    schema_ok = 0
    action_llm_ok = 0
    action_llm_fail = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    t0 = time.time()
    try:
        for step in range(max_steps):
            # 1. Pull the screenshot for the VLM and (optionally) save it.
            pil = _to_pil(obs.get("screenshot"))
            img_path: Optional[str] = None
            if pil is not None and frames_dir is not None:
                img_path = _save_frame(pil, frames_dir / f"step_{step:03d}.png")

            # 2. Heuristic visual grounding (deterministic, AXTree-walked).
            try:
                canonical_schema = bg_obs_to_schema(
                    obs, step=step, task_id=task_id, max_entities=max_entities,
                )
            except Exception as exc:
                logger.debug("heuristic obs_to_schema failed: %s", exc)
                canonical_schema = None

            # 3. Candidate-action vocabulary from the browser tool registry.
            registry = None
            if build_browser_registry is not None:
                try:
                    registry = build_browser_registry(obs)
                except Exception as exc:
                    logger.debug("build_browser_registry failed: %s", exc)
            candidate_actions, candidate_meta = _build_candidate_actions(
                obs=obs, registry=registry,
            )

            # 4. Visual grounding (vision call): screenshot → schema.
            schema_text: Optional[str] = None
            schema_meta: Dict[str, Any] = {
                "schema": None, "raw": "", "source": "skipped", "error": None,
            }
            if (
                use_vision and pil is not None
                and schema_helpers is not None and client is not None
            ):
                schema_budget = (
                    _SCHEMA_MAX_TOKENS_REASONING
                    if _is_reasoning_model(routed_model)
                    else _SCHEMA_MAX_TOKENS
                )
                # VWA tasks ride goal images on goal_object — surface them
                # to the schema VLM so it can ground "this product / this
                # chair / this image" references in the goal text.
                goal_imgs = _extract_goal_images(obs)
                schema_meta = generate_schema_from_image(
                    pil_image=pil,
                    obs=obs,
                    task_id=task_id,
                    goal=goal,
                    step=step,
                    candidate_actions=candidate_actions,
                    client=client,
                    routed_model=routed_model,
                    schema_helpers=schema_helpers,
                    canonical_fallback=canonical_schema,
                    temperature=temperature_schema,
                    max_tokens=schema_budget,
                    max_entities=max_entities,
                    use_som=True,
                    goal_images=goal_imgs,
                    reasoning_effort=reasoning_effort,
                )
                schema_calls += 1
                if schema_meta.get("source") == "vlm":
                    schema_ok += 1
                schema_text = schema_meta.get("schema")
            else:
                schema_text = canonical_schema
                schema_meta = {
                    "schema": canonical_schema,
                    "raw": "",
                    "source": "canonical" if canonical_schema else "text_only",
                    "error": None,
                }

            # 4b. Auto-dismiss cookie / consent dialogs ONCE before handing
            #     control to the LLM. Especially valuable for AssistantBench
            #     which boots on google.com behind a localized consent wall.
            preempted_action: Optional[str] = None
            if not consent_dismissed:
                cb = _detect_consent_button_bid(obs, canonical_schema)
                if cb is not None:
                    preempted_action = f'click("{cb}")'
                    consent_dismissed = True
                    if verbose:
                        print(f"  step {step}: consent-dismiss override -> {preempted_action!r}")

            # 5. Action selection (text-only call: schema → action). Skipped
            #    when we're pre-empting with a consent click, but we still
            #    want a placeholder reasoning trace for the Experience record.
            if preempted_action is not None:
                action = preempted_action
                reasoning = ("Auto-clicking the cookie/consent accept button to "
                             "unblock the page (deterministic pre-LLM heuristic).")
                action_raw = ""
                action_err = None
                action_llm_ok += 1
            else:
                action, reasoning, action_raw, action_err = select_action(
                    schema_text=schema_text,
                    obs=obs,
                    candidate_actions=candidate_actions,
                    candidate_meta=candidate_meta,
                    task=task,
                    step=step,
                    history=history,
                    client=client,
                    routed_model=routed_model,
                    temperature=temperature_action,
                    reasoning_effort=reasoning_effort,
                )
                if action is not None:
                    action_llm_ok += 1
                else:
                    action_llm_fail += 1
                    action = candidate_actions[0] if candidate_actions else "noop()"
                    if last_noop_action == action and len(candidate_actions) > 1:
                        action = _pick_different(action, candidate_actions)

            # 6. Anti-NOOP override.
            if (
                consecutive_noops >= _MAX_CONSECUTIVE_NOOPS
                and action == last_noop_action
                and len(candidate_actions) > 1
            ):
                old_action = action
                action = _pick_different(action, candidate_actions)
                reasoning = (
                    (reasoning or "")
                    + f" [auto-override: '{old_action}' was no-op {consecutive_noops}x]"
                )
                if verbose:
                    print(f"  step {step}: anti-noop override {old_action!r} -> {action!r}")

            # 6b. Anti-ERROR override — if the chosen action targets a
            #     bid that has already errored at any point this episode,
            #     force a different bid. We deliberately avoid escalating
            #     to ``go_back()``: in MiniWoB it leaves the task page and
            #     terminates the episode with reward 0, hiding model bugs.
            chosen_bid = _bid_from_action(action)
            if (
                chosen_bid is not None
                and chosen_bid in bad_bids
                and len(candidate_actions) > 1
            ):
                old_action = action
                alt_pool = [
                    a for a in candidate_actions
                    if (_bid_from_action(a) or "") not in bad_bids
                    and a not in _DESTRUCTIVE_NAV_ACTIONS
                ]
                if not alt_pool:
                    # Stay on the page rather than navigating away.
                    alt_pool = ["scroll(0, 300)", "scroll(0, -300)", "noop()"]
                action = random.choice(alt_pool)
                reasoning = (
                    (reasoning or "")
                    + f" [auto-override: bid={chosen_bid} previously errored — "
                    + f"choosing {action!r} (bad_bids={bad_bids})]"
                )
                if verbose:
                    print(f"  step {step}: anti-error override {old_action!r} -> {action!r} (bad bid {chosen_bid})")

            # 6c. Last-mile defense: quote any unquoted bid that slipped
            #     through (no-op for already-quoted candidate strings).
            #     ``click(12)`` becomes ``click("12")`` automatically.
            normalized_action = _autoquote_bids(action)
            if normalized_action != action and verbose:
                print(f"  step {step}: autoquote {action!r} -> {normalized_action!r}")
            action = normalized_action

            # 7. Step the env.
            try:
                next_obs, reward, terminated, truncated, _next_info = _step_env(
                    env, action,
                )
            except Exception as exc:
                logger.error(
                    "[%s] step %d env.step(%r) failed: %s",
                    target_payload, step, action, exc,
                )
                if verbose:
                    traceback.print_exc()
                break
            total_reward += reward
            done = bool(terminated) or bool(truncated)
            error_text = (next_obs.get("last_action_error") or "").strip()

            is_noop = _is_noop(obs, next_obs)
            history.append({
                "action": action,
                "reward": reward,
                "noop": is_noop,
                "error": bool(error_text),
                "error_text": error_text,
            })

            if is_noop and action == last_noop_action:
                consecutive_noops += 1
            elif is_noop:
                consecutive_noops = 1
                last_noop_action = action
            else:
                consecutive_noops = 0
                last_noop_action = None

            had_error = bool(error_text)
            this_bid = _bid_from_action(action)
            if had_error:
                # Record this bid as bad so we never re-target it.
                if this_bid is not None and this_bid not in bad_bids:
                    bad_bids.append(this_bid)
                # Cap memory so the avoid-list doesn't grow unbounded.
                bad_bids = bad_bids[-12:]
                if this_bid is not None and this_bid == last_error_bid:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 1
                    last_error_bid = this_bid
            else:
                consecutive_errors = 0
                last_error_bid = None

            # 8. Build the Experience record (use compact text observations
            #    that summarise the page; raw obs is too large to serialize).
            obs_summary = (
                f"url={obs.get('url','')} "
                f"focused_bid={obs.get('focused_element_bid','') or 'null'} "
                f"goal={goal[:120]}"
            )
            next_obs_summary = (
                f"url={next_obs.get('url','')} "
                f"focused_bid={next_obs.get('focused_element_bid','') or 'null'} "
                f"err={(error_text or 'null')[:120]}"
            )
            exp = Experience(
                state=obs_summary,
                action=str(action),
                reward=reward,
                next_state=next_obs_summary,
                done=done,
                intentions=reasoning,
                tasks=task,
            )
            exp.idx = step
            exp.action_type = "primitive"
            exp.raw_state = obs_summary[:4000]
            exp.raw_next_state = next_obs_summary[:4000]
            exp.available_actions = list(candidate_actions)
            exp.interface = {
                "env_name": "browsergym",
                "game_name": target_safe_id,
                "target_kind": target_kind,
                "target_payload": target_payload,
                "url": obs.get("url", target_payload if target_kind == "url" else ""),
                "wrapper": (
                    target_payload if target_kind == "task" else "browsergym/openended"
                ),
            }

            # Stash schema + VLM outputs both on Experience.extras (in-memory)
            # AND on Experience.metadata (which Experience.to_dict serialises).
            extras: Dict[str, Any] = {
                "schema": schema_text,
                "schema_source": schema_meta.get("source"),
                "schema_error": schema_meta.get("error"),
                "schema_canonical": canonical_schema,
                "candidate_actions": list(candidate_actions),
                "candidate_meta": candidate_meta[:_MAX_CANDIDATE_ACTIONS],
                "is_noop": is_noop,
                "error_text": error_text or None,
                "url": obs.get("url"),
                "focused_element_bid": obs.get("focused_element_bid") or None,
            }
            if schema_meta.get("finish_reason") is not None:
                extras["schema_finish_reason"] = schema_meta.get("finish_reason")
            if schema_meta.get("recovery"):
                extras["schema_recovery"] = schema_meta.get("recovery")
            schema_raw = schema_meta.get("raw") or ""
            if schema_raw:
                extras["schema_raw_excerpt"] = schema_raw[:4000]
                extras["schema_raw_full_len"] = schema_meta.get(
                    "raw_full_len", len(schema_raw)
                )
            if action_raw:
                extras["action_raw"] = action_raw[:1000]
            if action_err:
                extras["action_error"] = action_err
            if img_path:
                extras["frame_path"] = img_path
            exp.extras = extras
            existing_meta = getattr(exp, "metadata", None) or {}
            if isinstance(existing_meta, dict):
                existing_meta = dict(existing_meta)
            else:
                existing_meta = {}
            existing_meta.update(extras)
            exp.metadata = existing_meta
            experiences.append(exp)

            # Dump a sidecar JSON next to each frame so the PNG is
            # self-describing on disk (action / reward / schema / url …).
            if img_path and frames_dir is not None:
                try:
                    sidecar = {
                        "step": step,
                        "url": obs.get("url"),
                        "next_url": next_obs.get("url"),
                        "action": action,
                        "action_raw": (action_raw or "")[:1000],
                        "action_error": action_err,
                        "reward": reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "is_noop": is_noop,
                        "schema_source": schema_meta.get("source"),
                        "schema_error": schema_meta.get("error"),
                        "schema": schema_text,
                        "candidate_actions": list(candidate_actions),
                        "focused_element_bid": obs.get("focused_element_bid"),
                        "frame_path": img_path,
                        "target_kind": target_kind,
                        "target_payload": target_payload,
                    }
                    sidecar_path = frames_dir / f"step_{step:03d}.json"
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar, f, indent=2, ensure_ascii=False, default=str)
                except Exception as exc:
                    logger.debug("frame sidecar write failed at step %d: %s", step, exc)

            if verbose:
                r_short = (
                    (reasoning[:80] + "...")
                    if reasoning and len(reasoning) > 80 else reasoning
                )
                tag = (
                    " [ERR]" if error_text else
                    (" [NOOP]" if is_noop else "")
                )
                print(
                    f"  step {step:>3}: action={action!r:<32} "
                    f"reward={reward:+.2f} cum={total_reward:+.2f}{tag} "
                    f"schema={schema_meta.get('source')} reason={r_short}"
                )

            obs = next_obs
            if done:
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    elapsed = time.time() - t0

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="browsergym",
        game_name=target_safe_id,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "target_kind": target_kind,
        "target_payload": target_payload,
        "target_safe_id": target_safe_id,
        "wrapper": (
            target_payload if target_kind == "task" else "browsergym/openended"
        ),
        "macro": False,
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "elapsed_seconds": round(elapsed, 2),
        "model": fallback_model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_browsergym",
        "use_vision": use_vision,
        "schema_calls": schema_calls,
        "schema_ok": schema_ok,
        "action_llm_ok": action_llm_ok,
        "action_llm_fail": action_llm_fail,
        "noop_steps": sum(1 for h in history if h["noop"]),
        "error_steps": sum(1 for h in history if h.get("error")),
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _count_existing_episodes(env_dir: Path) -> int:
    if not env_dir.exists():
        return 0
    return sum(
        1 for f in env_dir.glob("episode_*.json")
        if f.name != "episode_buffer.json"
    )


def _save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_target_rollouts(
    target_kind: str,
    target_payload: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all episodes for one BrowserGym target and persist outputs.

    ``target_kind`` is ``"task"`` (BrowserGym env id) or ``"url"`` (live URL
    on top of ``browsergym/openended``).
    """
    safe = (
        _task_safe(target_payload) if target_kind == "task"
        else _url_safe(target_payload)
    )
    target_dir = output_dir / safe
    target_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = target_dir / "rollouts.jsonl"

    target_episodes = args.episodes
    effective_max_steps = args.max_steps
    label = f"{target_kind}:{target_payload}"

    start_idx = 0
    if args.resume:
        start_idx = _count_existing_episodes(target_dir)
        if start_idx >= target_episodes:
            print(f"  [SKIP] {label}: {start_idx}/{target_episodes} episodes already done")
            return {
                "target_kind": target_kind,
                "target_payload": target_payload,
                "target_safe_id": safe,
                "skipped": True,
                "existing": start_idx,
                "target_episodes": target_episodes,
            }
        if start_idx > 0:
            print(f"  [RESUME] {label}: starting from episode {start_idx}")

    buffer = Episode_Buffer(buffer_size=target_episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t_target = time.time()

    for ep_idx in range(start_idx, target_episodes):
        print(f"\n  [{label}] Episode {ep_idx + 1}/{target_episodes}")
        try:
            frames_dir = (
                target_dir / "frames" / f"ep_{ep_idx:03d}"
                if args.save_frames else None
            )

            episode, stats = run_actor_episode(
                target_kind=target_kind,
                target_payload=target_payload,
                target_safe_id=safe,
                headless=not args.no_headless,
                max_steps=effective_max_steps,
                client=client,
                routed_model=routed_model,
                fallback_model=args.model,
                schema_helpers=schema_helpers,
                use_vision=not args.no_vision,
                temperature_action=args.temperature_action,
                temperature_schema=args.temperature_schema,
                max_entities=args.max_entities,
                frames_dir=frames_dir,
                seed=42 + ep_idx,
                verbose=args.verbose,
                reasoning_effort=getattr(args, "reasoning_effort", None),
            )
            stats["episode_index"] = ep_idx
            print(
                f"    steps={stats['steps']:>3} "
                f"reward={stats['total_reward']:+.2f} "
                f"schema_ok={stats['schema_ok']}/{stats['schema_calls']} "
                f"action_ok={stats['action_llm_ok']} (fail={stats['action_llm_fail']}) "
                f"noops={stats['noop_steps']} errs={stats['error_steps']}"
            )

            buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            with open(target_dir / f"episode_{ep_idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
            _save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as exc:
            print(f"    [ERROR] episode {ep_idx + 1} failed: {exc}")
            traceback.print_exc()
            all_stats.append({
                "target_kind": target_kind,
                "target_payload": target_payload,
                "episode_index": ep_idx,
                "error": str(exc),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed_target = time.time() - t_target
    buffer.save_to_json(str(target_dir / "episode_buffer.json"))
    print(f"\n  Saved {len(buffer)} episodes for {label} in {elapsed_target:.1f}s")

    # ── Watchdog (added 2026-05-03) ──────────────────────────────────────
    # Print per-task timing in a single grep-able line so a parent shell
    # can ``grep '\[WATCHDOG\]' run.log | awk ...`` without scraping the
    # whole rollout_summary.json. Compare against the OSWorld baseline
    # ~24 s/step (gpt-5.4 default reasoning) measured 2026-04-29 from
    # ``Cold-start-out-osworld/gpt5.4_3per_domain``. VWA's Playwright
    # observation path is ~3× lighter than OSWorld's KVM-VM step, so
    # >30 s/step on VWA almost always means we accidentally re-enabled
    # a steering / high-reasoning knob — exactly the inflation pattern
    # that pushed the May-3 ``full7-smoke`` to 95.6 s/step (see the
    # OSWorld `bucketA-smoke` / `full7-smoke` run-comparison table).
    _wd_total_steps = sum(
        s.get("steps", 0) or 0 for s in all_stats if "error" not in s
    )
    _wd_total_eps = sum(1 for s in all_stats if "error" not in s)
    if _wd_total_steps > 0 and elapsed_target > 0:
        _wd_per_step = elapsed_target / _wd_total_steps
        _wd_status = (
            "SLOW " if _wd_per_step > 30.0
            else "FAST " if _wd_per_step < 12.0
            else "ok   "
        )
        print(
            f"  [WATCHDOG {_wd_status}] {label}: "
            f"eps={_wd_total_eps} steps={_wd_total_steps} "
            f"elapsed={elapsed_target:.1f}s "
            f"sec_per_step={_wd_per_step:.1f}s "
            f"(reasoning_effort={getattr(args, 'reasoning_effort', None) or 'unset'})"
        )
        if _wd_per_step > 30.0:
            print(
                f"  [WATCHDOG WARN] {_wd_per_step:.1f}s/step > 30s red-line. "
                f"Likely culprits: reasoning_effort=high/medium, steering "
                f"modules enabled, or LLM provider routing latency. "
                f"Cross-check ``rollout_summary.json:elapsed_seconds`` and "
                f"compare with the May-3 OSWorld watchdog table in the "
                f"implementation_notes/ memo."
            )

    summary: Dict[str, Any] = {
        "target_kind": target_kind,
        "target_payload": target_payload,
        "target_safe_id": safe,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_browsergym",
        "wrapper": (
            target_payload if target_kind == "task" else "browsergym/openended"
        ),
        "target_episodes": target_episodes,
        "completed_episodes": len([s for s in all_stats if "error" not in s]),
        "use_vision": not args.no_vision,
        "max_steps": effective_max_steps,
        "elapsed_seconds": round(elapsed_target, 2),
        "episode_stats": all_stats,
    }
    rewards = [s["total_reward"] for s in all_stats if "error" not in s]
    steps_list = [s["steps"] for s in all_stats if "error" not in s]
    if rewards:
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
    if steps_list:
        summary["mean_steps"] = sum(steps_list) / len(steps_list)

    with open(target_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start actor-agent rollouts using gpt-5.5 visual grounding "
            "+ schema-driven action selection over BrowserGym openended."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help=(
            "BrowserGym task ids to run, e.g. "
            "'browsergym/miniwob.click-button browsergym/webarena.42'. "
            "Each task carries its own goal + reward function. Mutually "
            "additive with --urls."
        ),
    )
    parser.add_argument(
        "--urls", type=str, nargs="+", default=None,
        help=(
            "Open-ended BrowserGym URLs (uses ``browsergym/openended`` with "
            "``task_kwargs={'start_url': <url>}``; reward is always 0). "
            f"Default if neither --tasks nor --urls is given: "
            f"{' '.join(DEFAULT_URLS)}"
        ),
    )
    parser.add_argument(
        "--list_tasks", action="store_true",
        help="Probe registered BrowserGym task ids per suite and exit.",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Episodes per target URL (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max outer steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--max_entities", type=int, default=_DEFAULT_MAX_ENTITIES,
        help=f"Cap on entities per schema (default: {_DEFAULT_MAX_ENTITIES})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Backbone model for visual grounding + actor (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature_action", type=float, default=0.4,
        help="Sampling temperature for the action call (default: 0.4)",
    )
    parser.add_argument(
        "--temperature_schema", type=float, default=0.2,
        help="Sampling temperature for the visual schema call (default: 0.2)",
    )
    parser.add_argument(
        "--no_vision", action="store_true",
        help="Skip the vision call; use the deterministic AXTree-walked "
             "heuristic schema for action selection.",
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help=(
            "Persist the PNG frames sent to the VLM under "
            "<safe_id>/frames/ep_NNN/step_NNN.png plus a sidecar "
            "<step_NNN>.json carrying the action / reward / schema / "
            "url / candidate actions for that step. Off by default; "
            "the rollouts.jsonl + episode_NNN.json files always include "
            "a frame_path field, but the PNGs themselves are only "
            "written when this flag is set."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip episodes that already have an episode_NNN.json on disk.",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Override OPENAI_API_KEY / OPENROUTER_API_KEY for this run.",
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="Override the OpenAI base URL (e.g. for a custom proxy).",
    )
    parser.add_argument(
        "--no_headless", action="store_true",
        help=(
            "Render the browser visibly. By default the actor runs "
            "headless (Xvfb-backed Chromium); only pass --no_headless "
            "when you have a real X display and want to watch the run."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <codebase_root>/Cold-start-out-browsergym)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details (action, reward, schema source).",
    )
    parser.add_argument(
        "--reasoning_effort", "--reasoning-effort",
        type=str, default="low",
        choices=list(_VALID_REASONING_EFFORTS),
        help=(
            "OpenAI reasoning_effort knob for gpt-5.x / o1 / o3 / o4. "
            "One of {minimal, low, medium, high}. Default: 'low' "
            "(2026-05-03). Why ``low`` and not ``minimal``: OpenAI "
            "direct ``/v1/chat/completions`` rejects ``minimal`` for "
            "gpt-5.x with HTTP-400 (``Unsupported value: 'reasoning_"
            "effort' does not support 'minimal' with this model``); "
            "``low`` is accepted on direct + OpenRouter + silently "
            "dropped by the driver for non-OpenAI-reasoning models "
            "(Claude / Gemini / Qwen3-VL). For SFT cold-start data "
            "generation ``low`` is the right knob — the student never "
            "consumes the teacher's hidden thinking, so anything above "
            "``low`` is wasted budget on structured-extraction + "
            "constrained-action tasks. Reserve ``medium`` / ``high`` "
            "for the leaderboard chase where teacher answer correctness "
            "is the bottleneck (visual reasoning MCQ, hard multi-hop)."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Probe BrowserGym imports up front so we fail fast with a clear message
    # instead of crashing inside the first episode.
    try:
        import gymnasium  # noqa: F401
        import browsergym.core  # noqa: F401
    except Exception as exc:
        print(
            "[FATAL] BrowserGym + Playwright are required for the actor "
            "pipeline (no synthetic / offline mode is supported).\n"
            f"        ImportError: {exc}\n"
            "        Install via: bash install/install_browsergym.sh"
        )
        sys.exit(2)

    suite_ok, suite_fail = _import_optional_task_suites()

    if args.list_tasks:
        buckets = _list_registered_task_ids()
        total = sum(len(v) for v in buckets.values())
        print(f"Registered browsergym/* env ids: {total}")
        for prefix in sorted(buckets):
            ids = buckets[prefix]
            print(f"  {prefix:<18s} {len(ids):>5d} envs   e.g. {ids[0]}")
        if suite_fail:
            print()
            print("Optional suites that failed to import (skipped):")
            for f in suite_fail:
                print(f"  - {f}")
        return

    # Build the (kind, payload) target list. --tasks takes priority; --urls
    # is additive. If neither is given, fall back to DEFAULT_URLS so the
    # script still runs out of the box.
    raw_tasks: List[str] = list(args.tasks or [])
    raw_urls: List[str] = list(args.urls or [])
    if not raw_tasks and not raw_urls:
        raw_urls = list(DEFAULT_URLS)

    # Validate task ids against the live registry.
    bad_tasks: List[str] = []
    if raw_tasks:
        registered = set(gymnasium.envs.registry.keys())
        for t in raw_tasks:
            if t not in registered:
                bad_tasks.append(t)
    if bad_tasks:
        print("[ERROR] These --tasks env ids are not registered:")
        for t in bad_tasks:
            print(f"  - {t}")
        print(
            "Did you forget to install the task suite? "
            "Use --list_tasks to inspect what's available."
        )
        if suite_fail:
            print("Failed-to-import suites that may be needed:")
            for f in suite_fail:
                print(f"  - {f}")
        sys.exit(2)

    infra_errors = _preflight_task_infra(raw_tasks)
    if infra_errors:
        print("[ERROR] Task suite infrastructure is missing:")
        for e in infra_errors:
            print(e)
        print(
            "\nThe launcher script (run_coldstart_actor_browsergym.sh) "
            "auto-resolves MINIWOB_URL when miniwob-plusplus is checked out at "
            "/workspace/BrowserGym/miniwob-plusplus. Set the listed env vars "
            "and re-run, or drop the affected --tasks."
        )
        sys.exit(2)

    targets: List[Tuple[str, str]] = (
        [("task", t) for t in raw_tasks] + [("url", u) for u in raw_urls]
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else CODEBASE_ROOT / "Cold-start-out-browsergym"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    has_key = bool(
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not has_key and not args.no_vision:
        print("[WARNING] No API key set. Both VLM and actor calls will fail.")
        print("  Set: export OPENAI_API_KEY='sk-...' or OPENROUTER_API_KEY='sk-or-...'")

    client, routed_model = _build_client_and_route(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    if client is None:
        print(
            "[WARNING] No OpenAI/OpenRouter client could be built — "
            "actor will fall through to deterministic defaults."
        )

    schema_helpers = _import_schema_helpers()

    print("=" * 78)
    print("  Cold-Start Actor Agent — BrowserGym + gpt-5.5")
    print("=" * 78)
    if _API_KEYS_FILE_USED is not None:
        print(f"  API keys file:      {_API_KEYS_FILE_USED}")
    print(f"  Targets ({len(targets)}):")
    for kind, payload in targets:
        print(f"    [{kind:<4s}] {payload}")
    if suite_ok:
        print(f"  Suites loaded:      {', '.join(suite_ok)}")
    if suite_fail:
        print(f"  Suites unavailable: {len(suite_fail)} (use --list_tasks for details)")
    print(f"  Episodes (per target): {args.episodes}")
    print(f"  Max steps:          {args.max_steps}")
    print(f"  Max entities:       {args.max_entities}")
    print(f"  Model (configured): {args.model}")
    print(f"  Model (routed):     {routed_model}")
    print(f"  Vision schema:      {'OFF (--no_vision)' if args.no_vision else 'ON'}")
    print(f"  Headless:           {not args.no_headless}")
    if args.no_headless:
        print("    [WARN] --no_headless requested: a real X display is required. "
              "Headless is the recommended default.")
    print(f"  Save frames:        {args.save_frames}"
          + ("  (PNG + JSON sidecar per step)" if args.save_frames else ""))
    print(f"  Resume:             {args.resume}")
    print(f"  Output:             {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    target_summaries: List[Dict[str, Any]] = []
    for kind, payload in targets:
        print(f"\n{'━' * 78}")
        print(f"  TARGET ({kind}): {payload}")
        print(f"{'━' * 78}")
        try:
            summary = run_target_rollouts(
                kind, payload,
                args=args, output_dir=output_dir,
                client=client, routed_model=routed_model,
                schema_helpers=schema_helpers,
            )
        except Exception as exc:
            traceback.print_exc()
            summary = {
                "target_kind": kind, "target_payload": payload, "error": str(exc),
            }
        target_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_browsergym",
        "use_vision": not args.no_vision,
        "tasks": [p for k, p in targets if k == "task"],
        "urls": [p for k, p in targets if k == "url"],
        "targets": [{"kind": k, "payload": p} for k, p in targets],
        "episodes_per_target": args.episodes,
        "max_steps": args.max_steps,
        "max_entities": args.max_entities,
        "temperature_action": args.temperature_action,
        "temperature_schema": args.temperature_schema,
        "suites_loaded": suite_ok,
        "suites_unavailable": suite_fail,
        "elapsed_seconds": round(overall_elapsed, 2),
        "per_target_summaries": target_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ACTOR COLD-START (BROWSERGYM) — BATCH COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Targets processed: {len(target_summaries)}")
    completed = [
        s for s in target_summaries
        if not s.get("skipped") and "completed_episodes" in s
    ]
    total_eps = sum(s["completed_episodes"] for s in completed)
    print(f"  Total episodes:   {total_eps}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    print(f"  Output:           {output_dir}")
    print(f"  Master summary:   {master_path}")
    if completed:
        means = [s["mean_reward"] for s in completed if "mean_reward" in s]
        steps_means = [s["mean_steps"] for s in completed if "mean_steps" in s]
        if means:
            print(f"  Avg reward:       {sum(means) / len(means):.2f}")
        if steps_means:
            print(f"  Avg steps:        {sum(steps_means) / len(steps_means):.1f}")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<url_safe>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
