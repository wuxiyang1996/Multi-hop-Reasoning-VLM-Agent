#!/usr/bin/env python3
"""Video-Holmes parameterized wrapper windows with matched adaptation forks."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import cv2
from imageio_ffmpeg import get_ffmpeg_exe
from openai import OpenAI
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import (  # noqa: E402
    ANSWER_SLOTS,
    normalized_probabilities,
    stable_hash,
)
from motif_transfer.candidate_transfer_experiment import (  # noqa: E402
    evaluate_candidate_adaptation,
)
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    VIDEO_INTERVENTION_TOOLS,
    build_video_registry,
    execute_video_intervention,
    route_question,
    video_tool_schemas,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_contract(config: Mapping[str, Any]) -> str:
    wrapper_root = Path(config["wrapper"]["root"])
    paths = (
        Path(__file__).resolve(),
        REPO / "src/motif_transfer/active_video_transfer.py",
        REPO / "src/motif_transfer/candidate_transfer_experiment.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        wrapper_root / "visual_reasoning_wrapper/tools_video.py",
        wrapper_root / "visual_reasoning_wrapper/question_router.py",
    )
    return stable_hash({
        "config": config,
        "code_sha256": {str(path): file_sha256(path) for path in paths},
    })


def _questions(dataset_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = dataset_root / "Benchmark" / f"{split}_Video-Holmes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{row['video ID']}.Q{int(row['Question ID'])}": row
        for row in rows
    }


def _video_path(dataset_root: Path, video_id: str) -> Path:
    root = dataset_root / "Benchmark"
    candidates = (
        root / "videos_cropped" / f"{video_id}.mp4",
        root / "videos" / "videos_cropped" / f"{video_id}.mp4",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(candidates[0])


def _proxy_frames(
    video_path: Path,
    *,
    frame_count: int,
    max_side: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {video_path}")
    duration = total_frames / fps
    frames = []
    seconds = []
    for index in range(frame_count):
        second = (index + 0.5) * duration / frame_count
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"failed to decode {video_path} at {second:.3f}s")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        frames.append(image)
        seconds.append(round(second, 3))
    capture.release()
    return frames, {
        "total_frames": total_frames,
        "source_fps": fps,
        "duration_seconds": duration,
        "proxy_sample_seconds": seconds,
    }


def _panel_bytes(
    frames: Sequence[Image.Image],
    *,
    labels: Sequence[str],
    frame_width: int,
    quality: int,
) -> bytes:
    if len(frames) != len(labels) or not frames:
        raise ValueError("panel frames and labels must be nonempty and aligned")
    columns = min(4, len(frames))
    rows = (len(frames) + columns - 1) // columns
    frame_height = round(frame_width * 9 / 16)
    canvas = Image.new("RGB", (columns * frame_width, rows * (frame_height + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (frame, label) in enumerate(zip(frames, labels)):
        image = frame.convert("RGB").copy()
        image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        x = (index % columns) * frame_width
        y = (index // columns) * (frame_height + 20)
        draw.text((x + 3, y + 2), label, fill="black")
        canvas.paste(image, (x, y + 20))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _overview_panels(
    frames: Sequence[Image.Image],
    *,
    seconds: Sequence[float],
    segment_count: int,
    frame_width: int,
    quality: int,
) -> list[bytes]:
    output = []
    for segment in range(segment_count):
        start = round(segment * len(frames) / segment_count)
        end = round((segment + 1) * len(frames) / segment_count)
        segment_frames = frames[start:end]
        labels = [f"P{index} {seconds[index]:.1f}s" for index in range(start, end)]
        output.append(_panel_bytes(
            segment_frames, labels=labels, frame_width=frame_width, quality=quality,
        ))
    return output


def _image_content(data: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        },
    }


def _make_audio_analyzer(
    client: OpenAI,
    *,
    model: str,
    video_path: Path,
    bitrate: str,
    max_tokens: int,
):
    """Bind wrapper audio windows to provider-neutral event descriptions."""

    cache: dict[tuple[float, float], dict[str, Any]] = {}

    def analyze(*, start_sec: float, end_sec: float) -> dict[str, Any]:
        key = (round(float(start_sec), 3), round(float(end_sec), 3))
        if key in cache:
            return dict(cache[key])
        with tempfile.TemporaryDirectory(prefix="video-holmes-audio-") as temp_dir:
            audio_path = Path(temp_dir) / "window.mp3"
            command = [
                get_ffmpeg_exe(), "-nostdin", "-v", "error",
                "-ss", f"{start_sec:.3f}",
                "-t", f"{end_sec - start_sec:.3f}",
                "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", bitrate,
                str(audio_path),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False,
            )
            if completed.returncode != 0 or not audio_path.is_file():
                raise RuntimeError(
                    "ffmpeg audio extraction failed: "
                    + completed.stderr.strip()[-500:]
                )
            audio_bytes = audio_path.read_bytes()
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        response = client.chat.completions.create(
            model=model,
            modalities=["text"],
            temperature=0,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe only audible evidence in this video "
                            "window: intelligible speech, non-speech sound "
                            "events, their temporal order or repetition, and "
                            "what is uncertain. Do not infer a film plot, "
                            "character motive, or answer any question. Be "
                            "concise and say when the window is silent."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "mp3"},
                    },
                ],
            }],
        )
        description = (response.choices[0].message.content or "").strip()
        if not description:
            raise RuntimeError("audio model returned an empty description")
        usage = response.usage
        payload = {
            "description": description,
            "backend": "target_native_audio_event_model",
            "model": str(response.model),
            "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "finish_reason": str(response.choices[0].finish_reason),
            "prompt_tokens": int(usage.prompt_tokens if usage else 0),
            "completion_tokens": int(usage.completion_tokens if usage else 0),
            "response_sha256": stable_hash(description),
        }
        cache[key] = payload
        return dict(payload)

    return analyze


def _json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("model returned no JSON content")
    payload = json.loads(raw)
    usage = response.usage
    return payload, {
        "model": str(response.model),
        "finish_reason": str(response.choices[0].finish_reason),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        "response_sha256": stable_hash(payload),
    }


def _answer(
    client: OpenAI,
    *,
    model: str,
    row: Mapping[str, Any],
    overview_panels: Sequence[bytes],
    evidence_panel: bytes | None,
    wrapper_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "Answer the Video-Holmes multiple-choice question from visible temporal "
        "and aligned audible evidence. Track causal order and distinguish "
        "observed evidence from hypotheses. Select the option best entailed by "
        "the film's diegetic evidence; do not replace an unexplained depicted "
        "event with a more ordinary real-world explanation unless the clip "
        "supports it. Contrast the two strongest options before committing. "
        "Return probability mass for all A--F choices. Question: "
        f"{row['Question']} Options: {json.dumps(row['Options'], ensure_ascii=False)}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(overview_panels):
        content.extend([
            {"type": "text", "text": f"Low-bandwidth temporal overview O{index}:"},
            _image_content(panel),
        ])
    if evidence_panel is not None and wrapper_receipt is not None:
        compact = {
            "tool": wrapper_receipt["tool"],
            "arguments": wrapper_receipt["arguments"],
            "proxy_frame_indices": wrapper_receipt["proxy_frame_indices"],
            "audio_evidence": wrapper_receipt["result"]["audio"],
        }
        content.extend([
            {
                "type": "text",
                "text": (
                    "Target-native wrapper intervention receipt: "
                    + json.dumps(compact, ensure_ascii=False)
                    + "\nThe audio description is target-native evidence for "
                    "the exact same time window. Focused high-resolution "
                    "visual re-observation:"
                ),
            },
            _image_content(evidence_panel),
        ])
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Return concise JSON {\"answer\":\"A-F\",\"probabilities\":"
            "{\"A\":number,...,\"F\":number},\"evidence\":[\"brief visible "
            "facts\"],\"reason\":\"brief\"}."
        ),
        content=content,
        max_tokens=1200,
    )
    probabilities = normalized_probabilities(payload.get("probabilities") or {})
    answer = str(payload.get("answer") or "").strip().upper()[:1]
    if answer not in ANSWER_SLOTS:
        answer = ANSWER_SLOTS[int(probabilities.argmax())]
    return {
        "answer": answer,
        "probabilities": {
            slot: float(value) for slot, value in zip(ANSWER_SLOTS, probabilities)
        },
        "evidence": [str(value) for value in payload.get("evidence", ())],
        "reason": str(payload.get("reason") or ""),
    }, usage


def _propose_actions(
    client: OpenAI,
    *,
    model: str,
    row: Mapping[str, Any],
    overview_panels: Sequence[bytes],
    duration_seconds: float,
    tool_schemas: Sequence[Mapping[str, Any]],
    routing: Mapping[str, Any],
    audio_overview: Sequence[Mapping[str, Any]],
    candidate_count: int,
    frames_per_candidate: int,
    maximum_window_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Propose exactly {candidate_count} distinct parameterized temporal "
            "evidence interventions. Each must use "
            "inspect_multimodal_window and test a "
            "different causal/temporal hypothesis. Windows must lie within "
            f"0..{duration_seconds:.3f}s, span at most "
            f"{maximum_window_fraction * duration_seconds:.3f}s, and request "
            f"n={frames_per_candidate}. Cite concrete visible proxy frame labels "
            "from the overview when choosing each window; uniform quartile "
            "partitioning is not a question-conditioned intervention. "
            "Do not answer the question or emit an answer letter. "
            f"Question routing receipt: {json.dumps(routing, ensure_ascii=False)} "
            "Question-independent low-bandwidth audio scout (use only to "
            "localize a focused test window): "
            f"{json.dumps(list(audio_overview), ensure_ascii=False)} "
            f"Available wrapper schemas: {json.dumps(tool_schemas, ensure_ascii=False)} "
            f"Question: {row['Question']} Options: "
            f"{json.dumps(row['Options'], ensure_ascii=False)}"
        ),
    }]
    for index, panel in enumerate(overview_panels):
        content.extend([
            {"type": "text", "text": f"Temporal overview O{index}:"},
            _image_content(panel),
        ])
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Return JSON {\"actions\":[{\"candidate_id\":\"C0\","
            "\"tool\":\"inspect_multimodal_window\","
            "\"arguments\":{\"n\":int,"
            "\"start_sec\":number,\"end_sec\":number},\"score\":number in "
            "[0,1],\"hypothesis\":\"what visible event this tests\"}]}."
        ),
        content=content,
        max_tokens=1800,
    )
    actions = list(payload.get("actions") or ())
    if len(actions) != candidate_count:
        raise ValueError("planner did not return the frozen candidate count")
    output = []
    seen_ids: set[str] = set()
    seen_windows: set[tuple[float, float]] = set()
    for index, action in enumerate(actions):
        proposed_candidate_id = str(action.get("candidate_id") or "")
        candidate_id = f"C{index}"
        tool = str(action.get("tool") or "")
        arguments = dict(action.get("arguments") or {})
        if tool not in VIDEO_INTERVENTION_TOOLS:
            raise ValueError(
                "planner emitted duplicate ID or unsupported wrapper tool: "
                + json.dumps(action, ensure_ascii=False)
            )
        start = max(0.0, float(arguments.get("start_sec", 0.0)))
        end = min(duration_seconds, float(arguments.get("end_sec", 0.0)))
        if end <= start:
            raise ValueError(
                "planner emitted an empty temporal window: "
                + json.dumps(action, ensure_ascii=False)
            )
        proposed_arguments = {
            "n": arguments.get("n"),
            "start_sec": start,
            "end_sec": end,
        }
        maximum_span = maximum_window_fraction * duration_seconds
        if end - start > maximum_span:
            center = (start + end) / 2.0
            start = max(0.0, center - maximum_span / 2.0)
            end = min(duration_seconds, start + maximum_span)
            start = max(0.0, end - maximum_span)
        window_key = (round(start, 3), round(end, 3))
        if window_key in seen_windows:
            raise ValueError("planner emitted duplicate temporal windows")
        score = float(action.get("score", 0.0))
        if not 0 <= score <= 1:
            raise ValueError("planner score is outside [0,1]")
        seen_ids.add(candidate_id)
        seen_windows.add(window_key)
        output.append({
            "candidate_id": candidate_id,
            "proposed_candidate_id": proposed_candidate_id,
            "tool": tool,
            "arguments": {
                "n": frames_per_candidate, "start_sec": start, "end_sec": end,
            },
            "proposed_arguments": proposed_arguments,
            "planner_score": score,
            "hypothesis": str(action.get("hypothesis") or ""),
        })
    return output, usage


def _descriptor(
    arguments: Mapping[str, Any], *, duration_seconds: float, proxy_count: int,
) -> list[float]:
    start = float(arguments["start_sec"]) / duration_seconds
    end = float(arguments["end_sec"]) / duration_seconds
    width = end - start
    center = (start + end) / 2
    n_fraction = float(arguments["n"]) / proxy_count
    return [1.0, start, end, width, n_fraction, center, abs(center - 0.5), 1.0 - width]


def _semantic_candidate_text(
    row: Mapping[str, Any],
    proposal: Mapping[str, Any],
    audio_overview: Sequence[Mapping[str, Any]],
) -> str:
    """Build an outcome-blind target-native candidate representation."""

    start = float(proposal["arguments"]["start_sec"])
    end = float(proposal["arguments"]["end_sec"])
    overlapping_audio = [
        str(segment["description"])
        for segment in audio_overview
        if float(segment["end_sec"]) > start
        and float(segment["start_sec"]) < end
    ]
    return json.dumps({
        "question": str(row["Question"]),
        "options": dict(row["Options"]),
        "candidate_hypothesis": str(proposal["hypothesis"]),
        "candidate_window": {"start_sec": start, "end_sec": end},
        "overlapping_question_independent_audio_scout": overlapping_audio,
    }, ensure_ascii=False, sort_keys=True)


def _semantic_candidate_embeddings(
    client: OpenAI,
    *,
    model: str,
    dimensions: int,
    texts: Sequence[str],
) -> tuple[list[list[float]], dict[str, Any]]:
    response = client.embeddings.create(
        model=model,
        input=list(texts),
        dimensions=dimensions,
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda row: int(row.index))
    embeddings = [list(map(float, row.embedding)) for row in ordered]
    if len(embeddings) != len(texts) or any(
        len(row) != dimensions for row in embeddings
    ):
        raise RuntimeError("semantic embedding response is misaligned")
    usage = response.usage
    return embeddings, {
        "model": str(response.model),
        "dimensions": dimensions,
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "input_sha256": stable_hash(list(texts)),
        "output_sha256": stable_hash(embeddings),
    }


def _collect_sample(
    sample_id: str,
    *,
    row: Mapping[str, Any],
    dataset_root: Path,
    config: Mapping[str, Any],
    api_key: str,
    audio_api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    video_id = str(row["video ID"])
    path = _video_path(dataset_root, video_id)
    media = config["media"]
    frames, video_meta = _proxy_frames(
        path,
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    seconds = video_meta["proxy_sample_seconds"]
    overviews = _overview_panels(
        frames,
        seconds=seconds,
        segment_count=int(media["overview_panel_count"]),
        frame_width=int(media["overview_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    audio_analyzer = _make_audio_analyzer(
        OpenAI(
            api_key=audio_api_key,
            base_url=str(config["audio"]["base_url"]),
            timeout=float(config["audio"]["timeout_seconds"]),
            max_retries=int(config["audio"]["max_retries"]),
        ),
        model=str(config["audio"]["model"]),
        video_path=path,
        bitrate=str(config["audio"]["mp3_bitrate"]),
        max_tokens=int(config["audio"]["max_tokens"]),
    )
    wrapper_root = Path(config["wrapper"]["root"])
    registry, proxy_fps = build_video_registry(
        frames,
        duration_seconds=float(video_meta["duration_seconds"]),
        wrapper_root=wrapper_root,
        audio_analyzer=audio_analyzer,
    )
    routing = route_question(
        str(row["Question"]), modality="video", wrapper_root=wrapper_root,
    ).as_dict()
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    model = str(config["model"]["id"])
    scout_count = int(config["audio"]["overview_segment_count"])
    duration = float(video_meta["duration_seconds"])
    audio_overview = []
    for segment in range(scout_count):
        start = segment * duration / scout_count
        end = (segment + 1) * duration / scout_count
        receipt = audio_analyzer(start_sec=start, end_sec=end)
        audio_overview.append({
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "description": receipt["description"],
            "model": receipt["model"],
            "audio_sha256": receipt["audio_sha256"],
            "response_sha256": receipt["response_sha256"],
        })
    proposals, proposal_usage = _propose_actions(
        client,
        model=model,
        row=row,
        overview_panels=overviews,
        duration_seconds=float(video_meta["duration_seconds"]),
        tool_schemas=video_tool_schemas(registry),
        routing=routing,
        audio_overview=audio_overview,
        candidate_count=int(config["interventions"]["candidate_count"]),
        frames_per_candidate=int(media["frames_per_candidate"]),
        maximum_window_fraction=float(config["interventions"][
            "maximum_window_fraction"
        ]),
    )
    embedding_config = config.get("semantic_embedding")
    embedding_receipt = None
    if embedding_config:
        embedding_key_name = str(embedding_config["api_key_name"])
        if embedding_key_name != "OPENAI_API_KEY":
            raise ValueError(
                "runner currently accepts only the separately supplied "
                "OPENAI_API_KEY for semantic embeddings"
            )
        semantic_texts = [
            _semantic_candidate_text(row, proposal, audio_overview)
            for proposal in proposals
        ]
        embeddings, embedding_receipt = _semantic_candidate_embeddings(
            OpenAI(
                api_key=audio_api_key,
                base_url=str(embedding_config["base_url"]),
                timeout=float(embedding_config["timeout_seconds"]),
                max_retries=int(embedding_config["max_retries"]),
            ),
            model=str(embedding_config["model"]),
            dimensions=int(embedding_config["dimensions"]),
            texts=semantic_texts,
        )
        for proposal, embedding in zip(proposals, embeddings):
            proposal["semantic_embedding"] = embedding
            proposal["semantic_text_sha256"] = stable_hash(
                semantic_texts[int(proposal["candidate_id"][1:])]
            )
    baseline, baseline_usage = _answer(
        client,
        model=model,
        row=row,
        overview_panels=overviews,
        evidence_panel=None,
        wrapper_receipt=None,
    )
    candidates = []
    for proposal in proposals:
        selected, wrapper_receipt = execute_video_intervention(
            registry,
            frames,
            tool=str(proposal["tool"]),
            arguments=proposal["arguments"],
        )
        selected_indices = wrapper_receipt["proxy_frame_indices"]
        evidence_panel = _panel_bytes(
            selected,
            labels=[f"P{index} {seconds[index]:.1f}s" for index in selected_indices],
            frame_width=int(media["evidence_frame_width"]),
            quality=int(media["jpeg_quality"]),
        )
        answer, usage = _answer(
            client,
            model=model,
            row=row,
            overview_panels=overviews,
            evidence_panel=evidence_panel,
            wrapper_receipt=wrapper_receipt,
        )
        candidates.append({
            **proposal,
            "descriptor": _descriptor(
                wrapper_receipt["arguments"],
                duration_seconds=float(video_meta["duration_seconds"]),
                proxy_count=len(frames),
            ) + list(map(float, proposal.get("semantic_embedding", ()))),
            "wrapper_receipt": wrapper_receipt,
            "evidence_sha256": hashlib.sha256(evidence_panel).hexdigest(),
            "answer": answer,
            "usage": usage,
        })
    video_meta["wrapper_proxy_fps"] = proxy_fps
    video_meta["overview_panel_sha256"] = [
        hashlib.sha256(panel).hexdigest() for panel in overviews
    ]
    return {
        "schema_version": 1,
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "family": str(row.get("Question Type") or ""),
        "video_id": video_id,
        "gold_answer": str(row["Answer"]),
        "video_path": str(path),
        "video_sha256": file_sha256(path),
        "video_meta": video_meta,
        "wrapper_routing": routing,
        "wrapper_tool_names": registry.tool_names(),
        "proposal_usage": proposal_usage,
        "audio_overview": audio_overview,
        "semantic_embedding_receipt": embedding_receipt,
        "baseline": {"answer": baseline, "usage": baseline_usage},
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--split", choices=("adaptation",), default="adaptation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_sha256 = _collection_contract(config)
    sample_ids = list(config["splits"]["adaptation"])
    questions = _questions(args.dataset_root, "train")
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in questions]
    if missing_ids:
        raise SystemExit(f"frozen sample IDs are missing: {missing_ids}")
    keys = runpy.run_path(str(args.keys))
    model_key_name = str(config["model"].get(
        "api_key_name", "OPENROUTER_API_KEY"
    ))
    key = keys.get(model_key_name)
    audio_key = keys.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(f"{model_key_name} is missing")
    if not audio_key:
        raise SystemExit("OPENAI_API_KEY is missing for audio grounding")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.json"
    existing: dict[str, Any] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        bad = [
            sample_id for sample_id, row in existing.items()
            if row.get("collection_contract_sha256") != contract_sha256
        ]
        if bad:
            raise SystemExit(f"receipt/config contract mismatch: {bad}")
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_sample,
                sample_id,
                row=questions[sample_id],
                dataset_root=args.dataset_root,
                config=config,
                api_key=str(key),
                audio_api_key=str(audio_key),
                contract_sha256=contract_sha256,
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                existing[sample_id] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            ordered = [existing[value] for value in sample_ids if value in existing]
            receipts_path.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(ordered)}/{len(sample_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in sample_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete receipts; rerun to resume: {missing}")
    receipts = [existing[sample_id] for sample_id in sample_ids]
    controlled_path = Path(config["source"]["controlled_v3_config"])
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"]["controlled_v3_config_content_sha256"]:
        raise SystemExit("controlled source config content hash mismatch")
    report, artifact = evaluate_candidate_adaptation(
        receipts, config=config, controlled_config=controlled,
    )
    artifact_path = args.output_dir / "target_grounder_candidate.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["receipts"] = {
        "path": str(receipts_path.resolve()), "sha256": file_sha256(receipts_path),
    }
    report["target_grounder_candidate"] = {
        "path": str(artifact_path.resolve()),
        "sha256": file_sha256(artifact_path),
        "content_sha256": artifact["artifact_sha256"],
    }
    report_path = args.output_dir / "adaptation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "baseline_accuracy": report["baseline_accuracy"],
        "cross_fitted_selector_accuracy": report["cross_fitted_selector_accuracy"],
        "oracle_candidate_accuracy": report["oracle_candidate_accuracy"],
        "gates": report["gates"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
