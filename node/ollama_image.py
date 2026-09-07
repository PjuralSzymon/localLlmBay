"""Parse Ollama /api/generate image responses (Flux NDJSON or JSON)."""
from __future__ import annotations

import json
from typing import Any


IMAGE_RUNNER_HINTS = (
    "image runner not available",
    "image runner",
    "experimental image",
    "this model requires the image",
    "unsupported architecture",
    "unknown architecture",
    "flux runner",
    "no image runner",
    "image generation models are not currently supported",
)


def is_image_runner_error(status: int | None, body: str | None) -> bool:
    t = (body or "").lower()
    if not t:
        return False
    if any(h in t for h in IMAGE_RUNNER_HINTS):
        return True
    if status in (400, 500) and "image" in t and (
        "runner" in t or "not available" in t or "not currently supported" in t
    ):
        return True
    return False


def _b64_from_obj(obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw = obj.get("image")
    if isinstance(raw, str) and raw.strip():
        out.append(raw.strip())
    images = obj.get("images")
    if isinstance(images, list):
        for item in images:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                b = item.get("image") or item.get("b64") or item.get("data")
                if isinstance(b, str) and b.strip():
                    out.append(b.strip())
    return out


def parse_ollama_generate_image(payload: dict[str, Any] | str | None) -> dict[str, Any]:
    """Extract base64 images and token counts from a generate response.

    Accepts a parsed JSON object, NDJSON text, or mixed. Returns:
    images, prompt_eval_count, eval_count, error, done.
    """
    acc: dict[str, Any] = {}
    images: list[str] = []
    if payload is None:
        return {
            "images": [],
            "prompt_eval_count": 0,
            "eval_count": 0,
            "error": "",
            "done": False,
        }
    chunks: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        chunks.append(payload)
    else:
        text = str(payload)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                chunks.append(parsed)
            elif isinstance(parsed, list):
                chunks.extend(c for c in parsed if isinstance(c, dict))
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    chunks.append(chunk)
    for chunk in chunks:
        acc.update({k: v for k, v in chunk.items() if k not in ("image", "images")})
        for b64 in _b64_from_obj(chunk):
            if b64 not in images:
                images.append(b64)
    err = acc.get("error")
    err_s = str(err) if err else ""
    try:
        pt = int(acc.get("prompt_eval_count") or 0)
    except (TypeError, ValueError):
        pt = 0
    try:
        ct = int(acc.get("eval_count") or 0)
    except (TypeError, ValueError):
        ct = 0
    return {
        "images": images,
        "prompt_eval_count": pt,
        "eval_count": ct,
        "error": err_s,
        "done": bool(acc.get("done")),
        "raw": acc,
    }


def image_content_from_b64(images: list[str], *, mime: str = "image/png") -> str:
    """Markdown data-URL representation. Length is not billable."""
    parts = []
    for i, b64 in enumerate(images):
        alt = "generated image" if len(images) == 1 else f"generated image {i + 1}"
        parts.append(f"![{alt}](data:{mime};base64,{b64})")
    return "\n\n".join(parts)


def video_content_from_b64(videos: list[str], *, mime: str = "video/mp4") -> str:
    """Markdown data-URL representation. Length is not billable."""
    parts = []
    for i, b64 in enumerate(videos):
        alt = "generated video" if len(videos) == 1 else f"generated video {i + 1}"
        parts.append(f"![{alt}](data:{mime};base64,{b64})")
    return "\n\n".join(parts)
