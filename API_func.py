# API calling functions for the agent — routes to GPT, Claude, Gemini, or vLLM.
# Keys are sourced (in order) from:
#   1. Process environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
#   2. ``keys.py`` checked into the repo root one level above this file
#      (``/.../vlm-robot/keys.py``), exposing ``openai = "sk-..."`` etc.
#   3. ``.env.example`` documentation (not auto-loaded).

import importlib.util as _importlib_util
import itertools as _itertools
import os
import pathlib as _pathlib
import time as _time_mod
import threading as _threading

import openai

try:  # Anthropic / Gemini SDKs are optional for users who only call OpenAI.
    from anthropic import Anthropic  # type: ignore
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]

try:
    from google import genai  # type: ignore
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


def _load_repo_keys_module():
    """Best-effort import of ``keys.py`` from the parent of this repo.

    The keys file lives at ``<vlm-robot>/keys.py`` (one directory above
    ``Multi-hop-Reasoning-VLM-Agent``). Returns the loaded module or ``None``.
    """
    here = _pathlib.Path(__file__).resolve()
    candidates = [
        here.parent.parent / "keys.py",   # /vlm-robot/keys.py
        here.parent / "keys.py",          # local override (rare)
    ]
    extra = os.environ.get("VLM_ROBOT_KEYS_FILE", "").strip()
    if extra:
        candidates.insert(0, _pathlib.Path(extra))
    for path in candidates:
        try:
            if not path.is_file():
                continue
            spec = _importlib_util.spec_from_file_location("_vlm_robot_keys", str(path))
            if spec is None or spec.loader is None:
                continue
            mod = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
        except Exception:
            continue
    return None


_REPO_KEYS = _load_repo_keys_module()


def _key_from_repo(*attrs: str) -> str:
    """Return the first non-empty attribute value from ``keys.py``."""
    if _REPO_KEYS is None:
        return ""
    for a in attrs:
        v = getattr(_REPO_KEYS, a, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


openai_api_key = (
    os.environ.get("OPENAI_API_KEY", "").strip()
    or _key_from_repo("openai", "OPENAI_API_KEY", "openai_api_key")
)
claude_api_key = (
    os.environ.get("ANTHROPIC_API_KEY", "").strip()
    or _key_from_repo("anthropic", "ANTHROPIC_API_KEY", "claude", "claude_api_key")
)
gemini_api_key = (
    os.environ.get("GEMINI_API_KEY", "").strip()
    or _key_from_repo("gemini", "GEMINI_API_KEY", "google", "gemini_api_key")
)
open_router_api_key = (
    os.environ.get("OPENROUTER_API_KEY", "").strip()
    or _key_from_repo("openrouter", "OPENROUTER_API_KEY", "open_router_api_key")
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

_vllm_url_cycle = None
_vllm_url_lock = _threading.Lock()
_VLLM_URLS: list[str] = []
_VLLM_URL_BY_MODEL: dict[str, str] = {}


def _parse_url_map(raw: str) -> dict[str, str]:
    """Parse a ``VLLM_BASE_URL_MAP`` entry of the form
    ``model_id_1=url_1,model_id_2=url_2,...`` into a dict.

    Tolerant of whitespace and trailing commas. Bad entries (no ``=`` or
    empty model id) are silently skipped — the caller's round-robin
    fallback handles unmapped models.
    """
    out: dict[str, str] = {}
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece or "=" not in piece:
            continue
        model, url = piece.split("=", 1)
        model = model.strip()
        url = url.strip()
        if model and url:
            out[model] = url
    return out


def _init_vllm_urls() -> None:
    """Lazily read ``VLLM_BASE_URLS`` (or ``VLLM_BASE_URL``) and the
    optional per-model ``VLLM_BASE_URL_MAP`` override.

    ``VLLM_BASE_URL_MAP`` lets you run multiple vLLM servers (e.g. a 9B
    actor on :8000 and a 35B-A3B teacher on :8001) and dispatch by
    ``model=`` argument so a single ``ask_model(...)`` call lands on the
    right endpoint. Format::

        VLLM_BASE_URL_MAP="Qwen/Qwen3.5-9B=http://localhost:8000/v1,\\
                           Qwen/Qwen3.5-35B-A3B=http://localhost:8001/v1"

    Models not in the map fall back to the ``VLLM_BASE_URLS`` round-robin
    pool, preserving prior single-endpoint behaviour.
    """
    global _vllm_url_cycle, _VLLM_URLS, _VLLM_URL_BY_MODEL
    raw = os.environ.get("VLLM_BASE_URLS", "")
    if raw:
        _VLLM_URLS = [u.strip() for u in raw.split(",") if u.strip()]
    else:
        _VLLM_URLS = [os.environ.get("VLLM_BASE_URL", VLLM_BASE_URL)]
    _vllm_url_cycle = _itertools.cycle(_VLLM_URLS)

    map_raw = os.environ.get("VLLM_BASE_URL_MAP", "")
    _VLLM_URL_BY_MODEL = _parse_url_map(map_raw) if map_raw else {}


def _next_vllm_url(model: str | None = None) -> str:
    """Return the next vLLM URL.

    If ``model`` is supplied and present in ``VLLM_BASE_URL_MAP``, returns
    that endpoint directly (deterministic, no round-robin pollution).
    Otherwise rotates through ``_VLLM_URLS`` for backwards compatibility.
    """
    with _vllm_url_lock:
        global _vllm_url_cycle
        if _vllm_url_cycle is None:
            _init_vllm_urls()
        if model and model in _VLLM_URL_BY_MODEL:
            return _VLLM_URL_BY_MODEL[model]
        return next(_vllm_url_cycle)


def _candidate_vllm_urls(model: str | None = None) -> list[str]:
    """Return URLs to try for a given ``model``, preferring the per-model
    map entry first then falling back to the round-robin pool. Used by
    :func:`ask_vllm` to honour the per-model dispatch contract while
    still surviving a dead mapped instance via the existing pool.
    """
    with _vllm_url_lock:
        global _vllm_url_cycle
        if _vllm_url_cycle is None:
            _init_vllm_urls()
        candidates: list[str] = []
        if model and model in _VLLM_URL_BY_MODEL:
            candidates.append(_VLLM_URL_BY_MODEL[model])
        for url in _VLLM_URLS:
            if url not in candidates:
                candidates.append(url)
        return candidates


_vllm_reachable: bool | None = None
_vllm_probe_ts: float = 0.0
_VLLM_PROBE_TTL_S = float(os.environ.get("VLLM_PROBE_TTL_S", "60"))


def _probe_vllm() -> bool:
    """TCP probe to check if any vLLM server is reachable.

    Result is cached for ``_VLLM_PROBE_TTL_S`` seconds so that a
    temporarily-dead instance doesn't permanently disable local inference.
    """
    global _vllm_reachable, _vllm_probe_ts
    now = _time_mod.time()
    if _vllm_reachable is not None and (now - _vllm_probe_ts) < _VLLM_PROBE_TTL_S:
        return _vllm_reachable

    with _vllm_url_lock:
        if not _VLLM_URLS:
            _init_vllm_urls()

    import socket
    # Probe both the round-robin pool and any per-model mapped endpoints.
    probe_urls: list[str] = list(_VLLM_URLS)
    for url in _VLLM_URL_BY_MODEL.values():
        if url not in probe_urls:
            probe_urls.append(url)
    for url in probe_urls:
        try:
            stripped = url.replace("http://", "").replace("https://", "").rstrip("/")
            host_port = stripped.split("/")[0]
            host, port_str = host_port.rsplit(":", 1)
            sock = socket.create_connection((host, int(port_str)), timeout=2)
            sock.close()
            _vllm_reachable = True
            _vllm_probe_ts = now
            return True
        except Exception:
            continue

    _vllm_reachable = False
    _vllm_probe_ts = now
    print(f"[API_func] vLLM at {probe_urls} unreachable — "
          "Qwen calls will be routed through OpenRouter.")
    return _vllm_reachable


def make_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    prefer: str = "auto",
) -> "openai.OpenAI | None":
    """Build a configured ``openai.OpenAI`` client routed for this repo.

    Routing precedence:
      1. Explicit ``api_key`` / ``base_url`` arguments win.
      2. ``prefer='openrouter'`` forces OpenRouter (requires ``OPENROUTER_API_KEY``).
      3. ``prefer='openai'`` forces direct OpenAI (requires ``OPENAI_API_KEY``).
      4. ``prefer='auto'`` (default): use OpenRouter if ``OPENROUTER_API_KEY``
         is set, else fall back to direct OpenAI.

    Returns ``None`` if no usable credentials were found, matching the pattern
    used by ``cold_start/generate_cold_start_orak.py``.
    """
    if api_key:
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    if prefer not in ("auto", "openai", "openrouter"):
        prefer = "auto"

    or_key = (open_router_api_key or "").strip()
    oai_key = (openai_api_key or "").strip()

    use_openrouter = (
        prefer == "openrouter"
        or (prefer == "auto" and bool(or_key))
    )
    if use_openrouter and or_key:
        return openai.OpenAI(
            api_key=or_key,
            base_url=base_url or OPENROUTER_BASE,
        )
    if oai_key:
        kwargs = {"api_key": oai_key}
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)
    if or_key:
        return openai.OpenAI(
            api_key=or_key,
            base_url=base_url or OPENROUTER_BASE,
        )
    return None


def effective_openai_model(model: str, *, prefer: str = "auto") -> str:
    """Return a model id with the ``openai/`` prefix when routing via OpenRouter.

    Mirrors the ``_effective_model`` helper used by the cold-start scripts so
    callers can reuse it instead of duplicating string logic.
    """
    or_key = (open_router_api_key or "").strip()
    if prefer == "openai":
        return model
    if prefer == "openrouter" or (prefer == "auto" and or_key):
        return model if "/" in model else f"openai/{model}"
    return model


def ask_openrouter(question, model="openai/gpt-4o-mini", temperature=0.7, max_tokens=2000):
    """
    Ask a question via OpenRouter (unified API for GPT, Claude, Gemini, etc.).
    Used by default for cold-start data gathering and ask_model when key is set.
    """
    if not (open_router_api_key and open_router_api_key.strip()):
        return f"Error: OPENROUTER_API_KEY not set. See .env.example for required API keys."
    try:
        client = openai.OpenAI(base_url=OPENROUTER_BASE, api_key=open_router_api_key.strip())
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error calling OpenRouter API: {str(e)}"


def ask_gpt(question, model="gpt-5.5", temperature=0.7, max_tokens=2000):
    """
    Ask a question to GPT models. Uses OpenRouter when open_router_api_key is set (default in this repo).
    """
    if open_router_api_key and open_router_api_key.strip():
        # Prefer OpenRouter so one key is used for cold-start, etc.
        openrouter_model = model if "/" in model else f"openai/{model}"
        return ask_openrouter(question, model=openrouter_model, temperature=temperature, max_tokens=max_tokens)
    openai.api_key = openai_api_key
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT API: {str(e)}"


def ask_claude(question, model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=2000):
    """
    Ask a question to Claude models using Anthropic API.
    
    Args:
        question (str): The question to ask
        model (str): The Claude model to use (default: "claude-3-5-sonnet-20241022")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    if Anthropic is None:
        return ("Error: anthropic SDK not installed. "
                "`pip install anthropic` to use Claude models.")
    try:
        client = Anthropic(api_key=claude_api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"


def ask_gemini(question, model="gemini-2.5-flash", temperature=0.7, max_tokens=2000):
    """
    Ask a question to Gemini models using Google Generative AI API.
    
    Args:
        question (str): The question to ask
        model (str): The Gemini model to use (default: "gemini-2.5-flash")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    if genai is None:
        return ("Error: google-genai SDK not installed. "
                "`pip install google-genai` to use Gemini models.")
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=question,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (Qwen3, QwQ, etc.)."""
    import re
    if not text or "<think>" not in text:
        return text
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def ask_vllm(question, model="Qwen/Qwen3-8B", temperature=0.7, max_tokens=2000):
    """
    Ask a question via a vLLM-served model using its OpenAI-compatible endpoint.
    Configure the endpoint via VLLM_BASE_URL env var (default: http://localhost:8000/v1).

    Automatically strips ``<think>`` tags from reasoning models (Qwen3, QwQ, etc.).

    Tries each available vLLM URL before falling back to OpenRouter, so a
    single dead instance doesn't disable all local inference.
    """
    if not _probe_vllm():
        return _ask_qwen_via_openrouter(
            question, model=model, temperature=temperature, max_tokens=max_tokens,
        )

    # Per-model URL dispatch (VLLM_BASE_URL_MAP wins for mapped models;
    # everything else round-robins through VLLM_BASE_URLS as before).
    candidate_urls = _candidate_vllm_urls(model)

    _max_retries = int(os.environ.get("VLLM_OPENAI_MAX_RETRIES", "3"))
    last_exc = None
    for url in candidate_urls:
        try:
            client = openai.OpenAI(
                base_url=url, api_key=VLLM_API_KEY, max_retries=max(0, _max_retries),
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content or ""
            return _strip_think_tags(raw)
        except Exception as e:
            last_exc = e
            continue

    # All vLLM URLs failed — invalidate probe cache so next call re-probes
    global _vllm_probe_ts
    _vllm_probe_ts = 0.0
    fallback = _ask_qwen_via_openrouter(
        question, model=model, temperature=temperature, max_tokens=max_tokens,
    )
    if not fallback.startswith("Error"):
        return fallback
    return (
        f"Error calling vLLM API (all {len(candidate_urls)} candidate URLs "
        f"failed for model={model!r}, last: {last_exc})"
    )


def _ask_qwen_via_openrouter(question, model="Qwen/Qwen3-8B", temperature=0.7, max_tokens=2000):
    """Route a Qwen model call through OpenRouter as a fallback.

    Handles Qwen3 reasoning-model quirks:
      - Appends ``/no_think`` if not already present so the full token
        budget goes to actual content rather than thinking.
      - Falls back to the ``reasoning`` response field when ``content``
        is empty (some OpenRouter providers put thinking there).
    """
    if not (open_router_api_key and open_router_api_key.strip()):
        return (f"Error: vLLM at {VLLM_BASE_URL} unreachable and no "
                "OpenRouter API key configured for Qwen fallback.")

    if "/no_think" not in question:
        question = question.rstrip() + "\n/no_think"

    or_model = model.lower()
    try:
        client = openai.OpenAI(
            base_url=OPENROUTER_BASE, api_key=open_router_api_key.strip(),
        )
        response = client.chat.completions.create(
            model=or_model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        if not content:
            reasoning = getattr(choice.message, "reasoning", None) or ""
            if reasoning:
                content = reasoning
        return _strip_think_tags(content)
    except Exception as e:
        return f"Error calling OpenRouter API (Qwen fallback): {str(e)}"


def ask_model(question, model=None, temperature=0.7, max_tokens=2000):
    """
    General function to ask any AI model a question.
    Automatically routes to the appropriate API based on the model name.

    Args:
        question (str): The question to ask
        model (str): The model to use. Can be:
            - Qwen actor / skill-bank: "Qwen/Qwen3.5-9B" (default — routed
              to vLLM via ``ask_vllm``).
            - Qwen control-plane: "Qwen/Qwen3.5-35B-A3B" (crafter /
              harness / orchestrator — also routed to vLLM).
            - GPT judge / SFT teacher: "gpt-5.5", "gpt-5.5-mini", etc.
            - Claude models: "claude-3-5-sonnet-20241022", ...
            - Gemini models: "gemini-2.5-pro", "gemini-2.0-flash", ...
            - If None, defaults to "Qwen/Qwen3.5-9B" (the actor backbone;
              see ``common/models.py`` ``BACKBONE_MODEL``).
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)

    Returns:
        str: The generated answer
    """
    # Default model if none specified — the actor backbone (Qwen/Qwen3.5-9B)
    # routes through ``ask_vllm`` via the "qwen" branch below.
    if model is None:
        model = "Qwen/Qwen3.5-9B"
    model_lower = model.lower()

    # GPT-style models: use ask_gpt (which uses OpenRouter when open_router_api_key is set)
    if "gpt" in model_lower or model_lower.startswith("o1"):
        return ask_gpt(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "claude" in model_lower:
        # Anthropic Claude models
        return ask_claude(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "gemini" in model_lower:
        # Google Gemini models
        return ask_gemini(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "qwen" in model_lower or "vllm" in model_lower:
        return ask_vllm(question, model=model, temperature=temperature, max_tokens=max_tokens)

    else:
        return f"Error: Unknown model '{model}'. Please specify a GPT, Claude, Gemini, or Qwen model."