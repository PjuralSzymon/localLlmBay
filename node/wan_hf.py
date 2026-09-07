"""Hugging Face Diffusers runner for Wan 2.1 T2V 1.3B.

Ollama cannot generate video. The queue tag ``x/wan2.1-t2v:1.3b`` is a
catalog id only. This module downloads official Apache-2.0 Diffusers weights
and runs them with PyTorch.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

HF_REPO = os.getenv("WAN_HF_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
HF_DIR = Path(os.getenv("WAN_HF_DIR", "/app/models/hf/Wan2.1-T2V-1.3B"))
STEPS = int(os.getenv("WAN_HF_STEPS", "30"))
GUIDANCE = float(os.getenv("WAN_HF_GUIDANCE", "5.0"))
FLOW_SHIFT = float(os.getenv("WAN_HF_FLOW_SHIFT", "3.0"))
HEIGHT = int(os.getenv("WAN_HF_HEIGHT", "480"))
WIDTH = int(os.getenv("WAN_HF_WIDTH", "832"))
FRAMES = int(os.getenv("WAN_HF_FRAMES", "33"))
FPS = int(os.getenv("WAN_HF_FPS", "15"))

# Official Wan 2.1 negative prompt (480P T2V).
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen deformed limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

_PIPE = None
_PIPE_LOCK = threading.Lock()


class WanUnavailable(RuntimeError):
    """No CUDA, no Diffusers Wan pipeline, or weights missing."""


def is_video_ollama_tag(tag: str | None) -> bool:
    t = str(tag or "").strip().lower()
    return t.startswith("x/wan2.1-t2v:") or t.startswith("wan2.1-t2v:")


def _align_frames(n: int) -> int:
    """Wan VAE temporal compression is 4; num_frames must be 4k+1."""
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 33
    v = max(5, v)
    return ((v - 1) // 4) * 4 + 1


def size_fallbacks(
    height: int, width: int, frames: int
) -> list[tuple[int, int, int]]:
    """Smaller (H, W, frames) when VRAM OOMs. Prefer 480P, then fewer frames."""
    h0, w0, f0 = int(height), int(width), _align_frames(frames)
    seen: list[tuple[int, int, int]] = []
    for triple in (
        (h0, w0, f0),
        (480, 832, 33),
        (480, 832, 17),
        (320, 576, 17),
        (256, 448, 9),
    ):
        h, w, f = triple[0], triple[1], _align_frames(triple[2])
        pair = (h, w, f)
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
        raise WanUnavailable("huggingface_hub is not installed") from e
    snapshot_download(repo_id=HF_REPO, local_dir=str(d))
    if not weights_ready(d):
        raise WanUnavailable(f"Hugging Face download finished but no weights in {d}")
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
        from diffusers import AutoencoderKLWan, WanPipeline  # noqa: F401
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


def _load_pipe():
    global _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE
        if not _cuda_ok():
            raise WanUnavailable("CUDA is not available for Diffusers Wan")
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import (
                UniPCMultistepScheduler,
            )
        except ImportError as e:
            raise WanUnavailable("Diffusers WanPipeline is not installed") from e
        path = ensure_weights()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        vae = AutoencoderKLWan.from_pretrained(
            str(path), subfolder="vae", torch_dtype=torch.float32
        )
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=FLOW_SHIFT,
        )
        pipe = WanPipeline.from_pretrained(
            str(path), vae=vae, torch_dtype=dtype
        )
        pipe.scheduler = scheduler
        vram = _vram_mb() or 0.0
        if vram and vram < 16000:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _PIPE = pipe
        return _PIPE


def generate(prompt: str) -> tuple[str, dict[str, Any]]:
    """Return MP4 base64 and meta. Raises WanUnavailable or RuntimeError."""
    import torch
    from diffusers.utils import export_to_video

    pipe = _load_pipe()
    last_err: Exception | None = None
    for height, width, frames in size_fallbacks(HEIGHT, WIDTH, FRAMES):
        kwargs: dict[str, Any] = {
            "prompt": prompt or "video",
            "negative_prompt": NEGATIVE_PROMPT,
            "height": height,
            "width": width,
            "num_frames": frames,
            "guidance_scale": GUIDANCE,
            "num_inference_steps": STEPS,
        }
        try:
            kwargs["generator"] = torch.Generator(device="cuda")
        except Exception:
            pass
        try:
            out = pipe(**kwargs)
            frames_out = (out.frames or [None])[0]
            if frames_out is None:
                raise RuntimeError("Wan pipeline returned no frames")
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            try:
                export_to_video(frames_out, tmp.name, fps=FPS)
                raw = Path(tmp.name).read_bytes()
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            if not raw:
                raise RuntimeError("Wan export produced an empty MP4")
            b64 = base64.b64encode(raw).decode("ascii")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return b64, {
                "width": width,
                "height": height,
                "frames": frames,
                "steps": STEPS,
                "fps": FPS,
                "runner": "diffusers",
            }
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            oom = "out of memory" in msg or type(e).__name__ in (
                "OutOfMemoryError",
                "CUDAOutOfMemoryError",
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if not oom:
                raise
            continue
    raise RuntimeError(f"Wan Diffusers failed (including smaller sizes): {last_err}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--ensure"]:
        try:
            path = ensure_weights()
            print(f"wan_hf ready dir={path}")
            return 0
        except Exception as e:
            print(f"wan_hf ensure failed: {e}", file=sys.stderr)
            return 1
    print("usage: python -m wan_hf --ensure", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
