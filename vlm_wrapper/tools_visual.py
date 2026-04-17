"""Vision-model-backed tool implementations for visual understanding.

Provides tools that let the VLM delegate to specialised vision models
(OmniParser-v2, Florence-2, CLIP, OCR) via tool calling instead of
relying on its own pixel-level reasoning.  Each tool wraps a real
model invocation and returns structured results.

Two registries are available:

  * ``build_visual_registry(image)`` — tools that operate on a single frame.
  * Tools can also be merged with the video registry for temporal+visual
    understanding (see tools_video_visual.py).

Designed for multi-hop reasoning: the VLM sees a screenshot, calls
``detect_objects`` to get precise elements, then ``describe_region`` to
caption an ambiguous area, then ``spatial_query`` to verify a relation —
building a chain of grounded evidence before producing the final schema.

Usage::

    from vlm_wrapper.tools_visual import build_visual_registry

    registry = build_visual_registry(pil_image)
    # merge with video tools if needed:
    # combined = video_registry.merge(visual_registry)
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


# ── Tool definitions (OpenAI function-calling schema) ─────────────────

TOOL_DETECT_OBJECTS = ToolDef(
    name="detect_objects",
    description=(
        "Run an object/UI-element detector on the current frame. Returns "
        "a list of detected elements with bounding boxes, labels, types "
        "(icon/text), confidence scores, and interactability flags. "
        "Uses OmniParser-v2 (YOLO + OCR + Florence-2) for precise "
        "grounding. Much more accurate than eyeballing from pixels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum detection confidence (0-1). Default 0.05.",
            },
            "max_elements": {
                "type": "integer",
                "description": "Max elements to return. Default 30.",
            },
        },
        "required": [],
    },
    domain="visual",
)

TOOL_DESCRIBE_REGION = ToolDef(
    name="describe_region",
    description=(
        "Generate a natural-language caption for a specific rectangular "
        "region of the frame. Crops the region and runs Florence-2 for "
        "a semantic description. Use when you can see something but "
        "cannot identify what it is."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge in pixels."},
            "y": {"type": "integer", "description": "Top edge in pixels."},
            "w": {"type": "integer", "description": "Width in pixels."},
            "h": {"type": "integer", "description": "Height in pixels."},
        },
        "required": ["x", "y", "w", "h"],
    },
    domain="visual",
)

TOOL_VISUAL_SEARCH = ToolDef(
    name="visual_search",
    description=(
        "Search the frame for elements matching a text query. Runs the "
        "detector then filters results by text similarity to the query. "
        "Returns matching elements ranked by relevance. Example queries: "
        "'submit button', 'red icon', 'price label'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what to find.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results. Default 5.",
            },
        },
        "required": ["query"],
    },
    domain="visual",
)

TOOL_COUNT_OBJECTS = ToolDef(
    name="count_objects",
    description=(
        "Count elements of a specific type or matching a description in "
        "the current frame. Runs the detector and counts matches. "
        "Supports filtering by type ('icon', 'text', 'all') and/or "
        "a text query for more specific counting."
    ),
    parameters={
        "type": "object",
        "properties": {
            "element_type": {
                "type": "string",
                "enum": ["all", "icon", "text"],
                "description": "Filter by element type. Default 'all'.",
            },
            "query": {
                "type": "string",
                "description": "Optional text filter — only count elements whose label matches.",
            },
        },
        "required": [],
    },
    domain="visual",
)

TOOL_CLASSIFY_SCENE = ToolDef(
    name="classify_scene",
    description=(
        "Classify the current frame's scene type and content. Returns "
        "scene categories (game, browser, desktop, dialog, menu, etc.), "
        "a brief scene description, dominant visual properties, and "
        "counts of detected element types."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="visual",
)

TOOL_SPATIAL_QUERY = ToolDef(
    name="spatial_query",
    description=(
        "Query the spatial relationship between two detected elements "
        "using their ground-truth bounding boxes. Returns distance, "
        "direction, overlap, containment, and alignment information. "
        "More precise than estimating from the screenshot."
    ),
    parameters={
        "type": "object",
        "properties": {
            "element_a": {
                "type": "string",
                "description": "Label or index (e.g. '0', '3') of the first element.",
            },
            "element_b": {
                "type": "string",
                "description": "Label or index of the second element.",
            },
        },
        "required": ["element_a", "element_b"],
    },
    domain="visual",
)

TOOL_MEASURE_DISTANCE = ToolDef(
    name="measure_distance",
    description=(
        "Measure the pixel distance between two points or element "
        "centres. Returns Euclidean distance, horizontal/vertical "
        "offsets, and direction. Use for precise spatial reasoning."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x1": {"type": "integer", "description": "X of first point (pixels)."},
            "y1": {"type": "integer", "description": "Y of first point (pixels)."},
            "x2": {"type": "integer", "description": "X of second point (pixels)."},
            "y2": {"type": "integer", "description": "Y of second point (pixels)."},
        },
        "required": ["x1", "y1", "x2", "y2"],
    },
    domain="visual",
)

TOOL_EXTRACT_COLORS = ToolDef(
    name="extract_colors",
    description=(
        "Extract dominant colours from a region of the frame. Returns "
        "top-K colours as hex codes with their coverage percentages. "
        "Useful for identifying colour-coded elements, status indicators, "
        "or matching visual themes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge (pixels). Default: full frame."},
            "y": {"type": "integer", "description": "Top edge (pixels)."},
            "w": {"type": "integer", "description": "Width (pixels)."},
            "h": {"type": "integer", "description": "Height (pixels)."},
            "top_k": {"type": "integer", "description": "Number of colours. Default 5."},
        },
        "required": [],
    },
    domain="visual",
)

TOOL_READ_TEXT_REGION = ToolDef(
    name="read_text_region",
    description=(
        "Run OCR on a specific region (or the whole frame) and return "
        "structured text with bounding boxes. More precise than the "
        "video-level read_text_in_frame because it also returns "
        "line structure and reading order."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge (pixels). Omit for full frame."},
            "y": {"type": "integer", "description": "Top edge (pixels)."},
            "w": {"type": "integer", "description": "Width (pixels)."},
            "h": {"type": "integer", "description": "Height (pixels)."},
        },
        "required": [],
    },
    domain="visual",
)


# ── Internal state ────────────────────────────────────────────────────

@dataclass
class _DetectedElement:
    """Cached detection result for a single element."""
    index: int
    element_type: str
    label: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    interactable: bool
    confidence: float
    source: str


class _VisualState:
    """Holds the image and lazily-cached detection results."""

    def __init__(self, image: Image.Image):
        self.image = image.convert("RGB")
        self._detections: list[_DetectedElement] | None = None
        self._omniparser_available: bool | None = None

    def _check_omniparser(self) -> bool:
        if self._omniparser_available is not None:
            return self._omniparser_available
        try:
            from .grounding import parse_screen  # noqa: F401
            self._omniparser_available = True
        except ImportError:
            self._omniparser_available = False
            logger.warning(
                "OmniParser-v2 not available (missing ultralytics/transformers). "
                "detect_objects will fall back to OCR-only mode."
            )
        return self._omniparser_available

    def detect(
        self,
        confidence_threshold: float = 0.05,
        max_elements: int = 30,
    ) -> list[_DetectedElement]:
        """Run detection (cached after first call with default params)."""
        if self._detections is not None:
            filtered = [
                d for d in self._detections
                if d.confidence >= confidence_threshold
            ]
            return filtered[:max_elements]

        elements = self._run_detection(
            confidence_threshold=confidence_threshold,
            max_elements=max_elements,
        )
        self._detections = elements
        return elements

    def _run_detection(
        self,
        confidence_threshold: float,
        max_elements: int,
    ) -> list[_DetectedElement]:
        if self._check_omniparser():
            return self._detect_omniparser(confidence_threshold, max_elements)
        return self._detect_ocr_only()

    def _detect_omniparser(
        self,
        confidence_threshold: float,
        max_elements: int,
    ) -> list[_DetectedElement]:
        from .grounding import parse_screen

        screen_elements = parse_screen(
            self.image,
            box_threshold=confidence_threshold,
            max_elements=max_elements,
        )
        results = []
        for i, el in enumerate(screen_elements):
            results.append(_DetectedElement(
                index=i,
                element_type=el.element_type,
                label=el.label,
                bbox=(el.bbox.x, el.bbox.y, el.bbox.w, el.bbox.h),
                interactable=el.interactable,
                confidence=el.confidence,
                source=el.source,
            ))
        return results

    def _detect_ocr_only(self) -> list[_DetectedElement]:
        """Fallback: OCR-only detection when OmniParser is unavailable."""
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            raw = reader.readtext(np.array(self.image))
            results = []
            for i, (bbox_pts, text, conf) in enumerate(raw):
                flat = [int(p) for pt in bbox_pts for p in pt]
                x_min = min(flat[::2])
                y_min = min(flat[1::2])
                x_max = max(flat[::2])
                y_max = max(flat[1::2])
                results.append(_DetectedElement(
                    index=i,
                    element_type="text",
                    label=text,
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    interactable=False,
                    confidence=conf,
                    source="easyocr",
                ))
            return results
        except ImportError:
            logger.warning("No detection backend available")
            return []

    def find_by_ref(self, ref: str) -> _DetectedElement | None:
        """Find element by index string ('0', '3') or label substring."""
        dets = self.detect()
        if ref.isdigit():
            idx = int(ref)
            if 0 <= idx < len(dets):
                return dets[idx]
            return None
        ref_lower = ref.lower()
        for d in dets:
            if ref_lower in d.label.lower():
                return d
        return None


# ── Handler implementations ──────────────────────────────────────────

def _h_detect_objects(
    state: _VisualState,
    *,
    confidence_threshold: float = 0.05,
    max_elements: int = 30,
) -> dict:
    dets = state.detect(
        confidence_threshold=confidence_threshold,
        max_elements=max_elements,
    )
    elements = []
    for d in dets:
        elements.append({
            "index": d.index,
            "type": d.element_type,
            "label": d.label,
            "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
            "interactable": d.interactable,
            "confidence": round(d.confidence, 3),
            "source": d.source,
        })
    return {
        "elements": elements,
        "count": len(elements),
        "image_size": {"w": state.image.size[0], "h": state.image.size[1]},
    }


def _h_describe_region(
    state: _VisualState,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> dict:
    img_w, img_h = state.image.size
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    crop = state.image.crop((x, y, x + w, y + h))

    caption = _caption_crop(crop)
    ocr_text = _ocr_crop(crop)

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "caption": caption,
        "text_content": ocr_text,
    }


def _h_visual_search(
    state: _VisualState,
    *,
    query: str,
    max_results: int = 5,
) -> dict:
    dets = state.detect()
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored: list[tuple[float, _DetectedElement]] = []
    for d in dets:
        label_lower = d.label.lower()
        label_words = set(label_lower.split())

        score = 0.0
        if query_lower in label_lower:
            score += 1.0
        if query_lower == label_lower:
            score += 0.5
        word_overlap = len(query_words & label_words)
        if query_words:
            score += 0.4 * word_overlap / len(query_words)
        if d.element_type.lower() in query_lower:
            score += 0.2
        if d.interactable and any(
            kw in query_lower for kw in ("button", "link", "input", "click")
        ):
            score += 0.15

        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: -x[0])
    top = scored[:max_results]

    results = []
    for score, d in top:
        results.append({
            "index": d.index,
            "label": d.label,
            "type": d.element_type,
            "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
            "relevance": round(score, 3),
        })

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "total_elements": len(dets),
    }


def _h_count_objects(
    state: _VisualState,
    *,
    element_type: str = "all",
    query: str = "",
) -> dict:
    dets = state.detect()

    if element_type != "all":
        dets = [d for d in dets if d.element_type == element_type]

    if query:
        query_lower = query.lower()
        dets = [d for d in dets if query_lower in d.label.lower()]

    return {
        "count": len(dets),
        "element_type_filter": element_type,
        "query_filter": query or None,
        "labels": [d.label for d in dets[:20]],
    }


def _h_classify_scene(state: _VisualState) -> dict:
    dets = state.detect()
    img_w, img_h = state.image.size

    type_counts = {}
    for d in dets:
        type_counts[d.element_type] = type_counts.get(d.element_type, 0) + 1

    interactable_count = sum(1 for d in dets if d.interactable)
    text_count = type_counts.get("text", 0)
    icon_count = type_counts.get("icon", 0)

    scene_hints = []
    if icon_count > 10:
        scene_hints.append("icon-heavy (toolbar / app grid)")
    if text_count > 15:
        scene_hints.append("text-heavy (article / form)")
    if interactable_count > 8:
        scene_hints.append("highly interactive (form / game / control panel)")
    if len(dets) < 5:
        scene_hints.append("sparse (dialog / splash / loading)")

    label_text = " ".join(d.label.lower() for d in dets)
    category_clues = {
        "game": any(w in label_text for w in ("score", "level", "lives", "play", "game")),
        "form": any(w in label_text for w in ("submit", "name", "email", "password", "login")),
        "menu": any(w in label_text for w in ("file", "edit", "view", "settings", "menu")),
        "dialog": any(w in label_text for w in ("ok", "cancel", "close", "confirm", "yes", "no")),
        "browser": any(w in label_text for w in ("http", "www", "url", "search", "tab")),
    }
    categories = [k for k, v in category_clues.items() if v] or ["unknown"]

    arr = np.array(state.image)
    mean_brightness = float(arr.mean()) / 255.0
    color_variety = float(arr.reshape(-1, 3).std(axis=0).mean()) / 255.0

    return {
        "categories": categories,
        "scene_hints": scene_hints or ["no strong signals"],
        "element_counts": type_counts,
        "total_elements": len(dets),
        "interactable_elements": interactable_count,
        "image_size": {"w": img_w, "h": img_h},
        "visual_properties": {
            "brightness": round(mean_brightness, 3),
            "color_variety": round(color_variety, 3),
            "theme": "dark" if mean_brightness < 0.35 else "light" if mean_brightness > 0.65 else "medium",
        },
    }


def _h_spatial_query(
    state: _VisualState,
    *,
    element_a: str,
    element_b: str,
) -> dict:
    a = state.find_by_ref(element_a)
    b = state.find_by_ref(element_b)

    if a is None:
        return {"error": f"Element '{element_a}' not found. Run detect_objects first."}
    if b is None:
        return {"error": f"Element '{element_b}' not found. Run detect_objects first."}

    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    a_cx, a_cy = ax + aw // 2, ay + ah // 2
    b_cx, b_cy = bx + bw // 2, by + bh // 2

    dx, dy = b_cx - a_cx, b_cy - a_cy
    dist = math.sqrt(dx * dx + dy * dy)

    # Overlap / containment
    overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap_area = overlap_x * overlap_y
    a_area = max(1, aw * ah)
    b_area = max(1, bw * bh)

    contains_a_b = overlap_area / b_area > 0.85 if b_area > 0 else False
    contains_b_a = overlap_area / a_area > 0.85 if a_area > 0 else False
    overlaps = overlap_area > 0

    h_aligned = abs(a_cy - b_cy) < min(ah, bh) * 0.5
    v_aligned = abs(a_cx - b_cx) < min(aw, bw) * 0.5

    if abs(dx) > abs(dy) * 2:
        direction = "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx) * 2:
        direction = "below" if dy > 0 else "above"
    elif dx > 0 and dy > 0:
        direction = "below-right"
    elif dx > 0:
        direction = "above-right"
    elif dy > 0:
        direction = "below-left"
    else:
        direction = "above-left"

    return {
        "element_a": {"label": a.label, "bbox": {"x": ax, "y": ay, "w": aw, "h": ah}},
        "element_b": {"label": b.label, "bbox": {"x": bx, "y": by, "w": bw, "h": bh}},
        "center_distance_px": round(dist, 1),
        "horizontal_offset": dx,
        "vertical_offset": dy,
        "direction_a_to_b": direction,
        "overlaps": overlaps,
        "overlap_area_px": overlap_area,
        "a_contains_b": contains_a_b,
        "b_contains_a": contains_b_a,
        "horizontally_aligned": h_aligned,
        "vertically_aligned": v_aligned,
    }


def _h_measure_distance(
    state: _VisualState,
    *,
    x1: int, y1: int, x2: int, y2: int,
) -> dict:
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    img_w, img_h = state.image.size
    diag = math.sqrt(img_w ** 2 + img_h ** 2)

    if abs(dx) > abs(dy) * 2:
        direction = "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx) * 2:
        direction = "down" if dy > 0 else "up"
    else:
        direction = f"{'down' if dy > 0 else 'up'}-{'right' if dx > 0 else 'left'}"

    return {
        "euclidean_px": round(dist, 1),
        "horizontal_px": dx,
        "vertical_px": dy,
        "direction": direction,
        "relative_to_diagonal": round(dist / diag, 4) if diag > 0 else 0,
    }


def _h_extract_colors(
    state: _VisualState,
    *,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    top_k: int = 5,
) -> dict:
    if x is not None and y is not None and w is not None and h is not None:
        img_w, img_h = state.image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        region = state.image.crop((x, y, x + w, y + h))
    else:
        region = state.image
        x, y = 0, 0
        w, h = region.size

    small = region.resize((64, 64))
    arr = np.array(small).reshape(-1, 3)

    try:
        from sklearn.cluster import KMeans
        n_clusters = min(top_k, len(arr))
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        km.fit(arr)
        centres = km.cluster_centers_.astype(int)
        labels = km.labels_
        counts = np.bincount(labels, minlength=n_clusters)
        total = counts.sum()
        order = np.argsort(-counts)
    except ImportError:
        colours_unique, counts = np.unique(arr, axis=0, return_counts=True)
        order = np.argsort(-counts)[:top_k]
        centres = colours_unique[order]
        counts = counts[order]
        total = counts.sum()
        order = range(len(centres))

    colors = []
    for i in order[:top_k]:
        r, g, b = int(centres[i][0]), int(centres[i][1]), int(centres[i][2])
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        pct = float(counts[i]) / total if total > 0 else 0
        colors.append({"hex": hex_code, "rgb": [r, g, b], "coverage": round(pct, 3)})

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "colors": colors,
    }


def _h_read_text_region(
    state: _VisualState,
    *,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
) -> dict:
    if x is not None and y is not None and w is not None and h is not None:
        img_w, img_h = state.image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        region = state.image.crop((x, y, x + w, y + h))
    else:
        region = state.image
        x, y = 0, 0
        w, h = region.size

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(region))
        lines: list[dict] = []
        full_texts: list[str] = []
        for bbox_pts, text, conf in results:
            flat = [int(p) for pt in bbox_pts for p in pt]
            x_min, y_min = min(flat[::2]), min(flat[1::2])
            x_max, y_max = max(flat[::2]), max(flat[1::2])
            lines.append({
                "text": text,
                "confidence": round(conf, 3),
                "bbox": {"x": x_min + x, "y": y_min + y,
                         "w": x_max - x_min, "h": y_max - y_min},
            })
            full_texts.append(text)

        lines.sort(key=lambda l: (l["bbox"]["y"], l["bbox"]["x"]))
        return {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "lines": lines,
            "full_text": " ".join(full_texts),
            "count": len(lines),
            "engine": "easyocr",
        }
    except ImportError:
        pass

    try:
        import pytesseract
        text = pytesseract.image_to_string(region)
        return {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "full_text": text.strip(),
            "lines": [],
            "count": 0,
            "engine": "tesseract",
            "note": "Install easyocr for bbox-level results.",
        }
    except ImportError:
        pass

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "error": "No OCR engine available. Install easyocr or pytesseract.",
    }


# ── Vision model helpers ─────────────────────────────────────────────

def _caption_crop(crop: Image.Image) -> str:
    """Caption a cropped region using Florence-2 (if available)."""
    try:
        from .grounding import _load_caption
        import torch

        cap = _load_caption()
        model, processor = cap["model"], cap["processor"]
        device = model.device

        small = crop.resize((224, 224))
        prompt = "<CAPTION>"

        if device.type == "cuda":
            inputs = processor(
                images=[small], text=[prompt],
                return_tensors="pt", do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=[small], text=[prompt],
                return_tensors="pt",
            ).to(device=device)

        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50, num_beams=1, do_sample=False,
        )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()
    except (ImportError, Exception) as exc:
        logger.debug("Florence-2 captioning unavailable: %s", exc)
        return "(captioning model not available)"


def _ocr_crop(crop: Image.Image) -> str:
    """Run OCR on a crop and return concatenated text."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(crop))
        return " ".join(text for _, text, _ in results)
    except ImportError:
        pass
    try:
        import pytesseract
        return pytesseract.image_to_string(crop).strip()
    except ImportError:
        pass
    return ""


# ── Public: build registry ───────────────────────────────────────────

def build_visual_registry(image: Image.Image | np.ndarray) -> ToolRegistry:
    """Create a ToolRegistry with all vision-model-backed tools.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The frame / screenshot to analyse.

    Returns
    -------
    ToolRegistry
        Registry with tools: detect_objects, describe_region,
        visual_search, count_objects, classify_scene, spatial_query,
        measure_distance, extract_colors, read_text_region.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    state = _VisualState(image)
    reg = ToolRegistry(domain="visual")

    reg.register(TOOL_DETECT_OBJECTS, lambda **kw: _h_detect_objects(state, **kw))
    reg.register(TOOL_DESCRIBE_REGION, lambda **kw: _h_describe_region(state, **kw))
    reg.register(TOOL_VISUAL_SEARCH, lambda **kw: _h_visual_search(state, **kw))
    reg.register(TOOL_COUNT_OBJECTS, lambda **kw: _h_count_objects(state, **kw))
    reg.register(TOOL_CLASSIFY_SCENE, lambda **kw: _h_classify_scene(state))
    reg.register(TOOL_SPATIAL_QUERY, lambda **kw: _h_spatial_query(state, **kw))
    reg.register(TOOL_MEASURE_DISTANCE, lambda **kw: _h_measure_distance(state, **kw))
    reg.register(TOOL_EXTRACT_COLORS, lambda **kw: _h_extract_colors(state, **kw))
    reg.register(TOOL_READ_TEXT_REGION, lambda **kw: _h_read_text_region(state, **kw))

    return reg
