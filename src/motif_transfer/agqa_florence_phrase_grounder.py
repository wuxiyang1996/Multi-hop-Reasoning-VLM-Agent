"""Frozen Florence-2 referring-expression grounding for AGQA frames."""

from __future__ import annotations

from typing import Sequence

from PIL import Image

from .agqa_open_vocabulary_grounder import PhraseDetection


_MODEL = None
_PROCESSOR = None
_MODEL_ID = None


def ground_relation_phrases(
    frames: Sequence[Image.Image], *, frame_indices: Sequence[int],
    phrases: Sequence[str], model_id: str = "florence-community/Florence-2-base-ft",
) -> tuple[PhraseDetection, ...]:
    """Ground each answer-free phrase using Florence-2's native task token."""
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    global _MODEL, _PROCESSOR, _MODEL_ID
    if _MODEL is None:
        _PROCESSOR = AutoProcessor.from_pretrained(
            model_id, local_files_only=True, trust_remote_code=False)
        _MODEL = Florence2ForConditionalGeneration.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.bfloat16,
            trust_remote_code=False).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _MODEL_ID = model_id
    if _MODEL_ID != model_id:
        raise ValueError("one Florence process may use only one frozen model identity")
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    detections = []
    for frame_index in tuple(dict.fromkeys(int(value) for value in frame_indices)):
        image = frames[frame_index]
        for phrase in tuple(dict.fromkeys(str(value).strip().casefold()
                                          for value in phrases if str(value).strip())):
            inputs = _PROCESSOR(text=task + phrase, images=image, return_tensors="pt")
            inputs = {key: value.to(_MODEL.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = _MODEL.generate(
                    **inputs, max_new_tokens=256, num_beams=3, do_sample=False)
            text = _PROCESSOR.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = _PROCESSOR.post_process_generation(
                text, task=task, image_size=image.size).get(task, {})
            boxes = parsed.get("bboxes", ())
            for box in boxes:
                detections.append(PhraseDetection(
                    frame_index=frame_index, phrase=phrase, confidence=1.0,
                    bbox_xyxy=tuple(float(value) for value in box)))
    return tuple(detections)


__all__ = ["ground_relation_phrases"]
