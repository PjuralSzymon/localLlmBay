"""Hugging Face Diffusers runner for FLUX.2 Klein 4B.

Ollama 0.33.x rejects image models (HTTP 400). The Ollama tag
``x/flux2-klein:4b`` is a different blob format and cannot be loaded here.
This module downloads official Apache-2.0 weights and runs them with PyTorch.
"""
from __future__ import annotations

import base64
import inspect
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any

HF_REPO = os.getenv("FLUX_HF_ID", "black-forest-labs/FLUX.2-klein-4B")
HF_DIR = Path(os.getenv("FLUX_HF_DIR", "/app/models/hf/FLUX.2-klein-4B"))
STEPS = int(os.getenv("FLUX_HF_STEPS", "4"))
GUIDANCE = float(os.getenv("FLUX_HF_GUIDANCE", "1.0"))

_PIPE = None
_PIPE_LOCK = threading.Lock()


class FluxUnavailable(RuntimeError):
    """No CUDA, no Diffusers pipeline class, or weights missing."""


def is_image_ollama_tag(tag: str | None) -> bool:
    t = str(tag or "").strip().lower()
    return t.startswith("x/flux2-klein:") or t.startswith("flux2-klein:")


def pick_size(vram_mb: float | None) -> tuple[int, int]:
    """Square size that is likely to fit. 8 GB cards start at 768."""
    try:
        v = float(vram_mb) if vram_mb is not None else 0.0
    except (TypeError, ValueError):
        v = 0.0
    if v >= 12000:
        return 1024, 1024
    if v >= 7000:
        return 768, 768
    return 512, 512


def size_fallbacks(width: int, height: int) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    for pair in ((width, height), (768, 768), (512, 512), (384, 384)):
        if pair not in seen:
            seen.append(pair)
    return seen


def weights_ready(root: Path | None = None) -> bool:
    d = Path(root or HF_DIR)
    if (d / "model_index.json").is_file():
        return True
    return any(d.glob("*.safetensors")) or any(d.glob("**/*.safetensors"))


def ensure_weights(root: Path | None = None) -> Path:
    d = Path(root or HF_DIR)
    d.mkdir(parents=True, exist_ok=True)
    if weights_ready(d):
        return d
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise FluxUnavailable("huggingface_hub is not installed") from e
    snapshot_download(repo_id=HF_REPO, local_dir=str(d))
    if not weights_ready(d):
        raise FluxUnavailable(f"Hugging Face download finished but no weights in {d}")
    return d


def _cuda_ok() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def runtime_ready() -> bool:
    if not _cuda_ok():
        return False
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401
    except ImportError:
        return False
    return True


def _vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024.0 * 1024.0)
    except Exception:
        return None


def _decode_ref(raw: str):
    from PIL import Image

    s = (raw or "").strip()
    if not s:
        return None
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        blob = base64.b64decode(s, validate=False)
    except Exception:
        return None
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None


def _load_pipe():
    global _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE
        if not _cuda_ok():
            raise FluxUnavailable("CUDA is not available for Diffusers Flux")
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as e:
            raise FluxUnavailable("Diffusers Flux2KleinPipeline is not installed") from e
        path = ensure_weights()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        pipe = Flux2KleinPipeline.from_pretrained(str(path), torch_dtype=dtype)
        vram = _vram_mb() or 0.0
        if vram and vram < 12000:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        _PIPE = pipe
        return _PIPE


def generate(prompt: str, images: list[str] | None = None, strength: float | None = None) -> tuple[str, dict[str, Any]]:
    """Return PNG base64 and meta. Raises FluxUnavailable or RuntimeError."""
    import torch

    pipe = _load_pipe()
    vram = _vram_mb()
    w0, h0 = pick_size(vram)
    refs = []
    for item in images or []:
        if isinstance(item, str):
            im = _decode_ref(item)
            if im is not None:
                refs.append(im)
    last_err: Exception | None = None
    for width, height in size_fallbacks(w0, h0):
        kwargs: dict[str, Any] = {
            "prompt": prompt or "image",
            "height": height,
            "width": width,
            "guidance_scale": GUIDANCE,
            "num_inference_steps": STEPS,
        }
        try:
            gen = torch.Generator(device="cuda")
            kwargs["generator"] = gen
        except Exception:
            pass
        if refs:
            params = inspect.signature(pipe.__call__).parameters
            if "image" in params:
                kwargs["image"] = refs if len(refs) > 1 else refs[0]
            if strength is not None and "strength" in params:
                kwargs["strength"] = float(strength)
        try:
            out = pipe(**kwargs)
            pil = (out.images or [None])[0]
            if pil is None:
                raise RuntimeError("Flux pipeline returned no image")
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return b64, {"width": width, "height": height, "steps": STEPS, "runner": "diffusers"}
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            oom = "out of memory" in msg or type(e).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if not oom:
                raise
            continue
    raise RuntimeError(f"Flux Diffusers failed (including smaller sizes): {last_err}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--ensure"]:
        try:
            path = ensure_weights()
            print(f"flux_hf ready dir={path}")
            return 0
        except Exception as e:
            print(f"flux_hf ensure failed: {e}", file=sys.stderr)
            return 1
    print("usage: python -m flux_hf --ensure", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
