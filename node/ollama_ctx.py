"""Ollama context window for contribution nodes.

Granite loads 64k (CPU/GPU offload allowed). Other families stay at 16k unless
NODE_NUM_CTX says otherwise, still capped at PUBLIC_MAX_CTX.
"""
from __future__ import annotations

import os
import re

DEFAULT_NUM_CTX = 16384
GRANITE_NUM_CTX = 65536
PUBLIC_MAX_CTX = 65536
MIN_NUM_CTX = 4096
DEFAULT_NUM_PREDICT = 2048
WRITE_NUM_PREDICT = 8192
WALL_MARGIN = 256
INVALID_TOOL_RETRY_PREDICT = 2048
INVALID_TOOL_RETRY_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.3


def _tools_need_write_budget(tools: list | None) -> bool:
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        n = str((fn or {}).get("name") or t.get("name") or "").strip().lower()
        if n in ("write", "edit"):
            return True
    return False


def resolve_num_predict(job_num_predict: int | None = None, tools: list | None = None) -> int:
    """Wanted decode budget before leftover clamp. Write/edit JSON needs more than a Read call."""
    if job_num_predict is None:
        n = DEFAULT_NUM_PREDICT
    else:
        try:
            n = int(job_num_predict)
        except (TypeError, ValueError):
            n = DEFAULT_NUM_PREDICT
        if n <= 0:
            n = DEFAULT_NUM_PREDICT
    if _tools_need_write_budget(tools):
        return max(n, WRITE_NUM_PREDICT)
    return n


def clamp_num_predict(want: int | None, num_ctx: int | None, prompt_est: int | None) -> int:
    """Stop decode before the KV wall. llama.cpp min(num_predict, leftover) fills 16k.

    Honor sealed 2048. Keep 8192 only when leftover > 8192 + WALL_MARGIN. Always leave
    WALL_MARGIN unused so n_tokens never reaches num_ctx - 1.
    """
    try:
        n = int(want) if want is not None else DEFAULT_NUM_PREDICT
    except (TypeError, ValueError):
        n = DEFAULT_NUM_PREDICT
    if n <= 0:
        n = DEFAULT_NUM_PREDICT
    try:
        nctx = int(num_ctx or 0)
    except (TypeError, ValueError):
        nctx = 0
    try:
        est = int(prompt_est or 0)
    except (TypeError, ValueError):
        est = 0
    if nctx <= 0 or est <= 0:
        return n
    leftover = nctx - est
    if leftover <= WALL_MARGIN:
        return max(1, leftover - 32)
    return max(1, min(n, leftover - WALL_MARGIN))


def resolve_temperature(job_temp=None) -> float:
    """Stable tool calling. Honor an explicit infer temperature (think hop is 0.2)."""
    if job_temp is None:
        return DEFAULT_TEMPERATURE
    try:
        return float(job_temp)
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE


def parse_env_num_ctx(raw: str | None, default: int = DEFAULT_NUM_CTX) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(MIN_NUM_CTX, n)


def parse_model_context_length(show: dict | None) -> int | None:
    """Architecture max from Ollama POST /api/show (not the loaded 4096 slot)."""
    info = (show or {}).get("model_info") or {}
    best: int | None = None
    for key, val in info.items():
        k = str(key).lower()
        if not (k.endswith(".context_length") or k.endswith(".context_len")):
            continue
        try:
            n = int(val)
        except (TypeError, ValueError):
            continue
        if n < MIN_NUM_CTX:
            continue
        if best is None or n > best:
            best = n
    return best


def node_family_wants_granite_ctx(raw: str | None) -> bool:
    blob = (raw or "").lower()
    return any(p.strip() == "granite" for p in blob.replace(";", ",").split(","))


def recommend_num_ctx(*, vram_mb: float = 0, cpu_only: bool = False, family: str | None = None) -> int:
    """Largest public window this machine should load. Graph pack is 8k. Granite: 64k with offload."""
    fam = family if family is not None else os.getenv("NODE_FAMILY")
    blob = (fam or "").lower()
    if any(p.strip() == "graph" for p in blob.replace(";", ",").split(",")):
        return 8192
    if node_family_wants_granite_ctx(fam):
        return GRANITE_NUM_CTX
    if cpu_only:
        return MIN_NUM_CTX
    if vram_mb <= 0:
        return DEFAULT_NUM_CTX
    if vram_mb < 6 * 1024:
        return MIN_NUM_CTX
    return DEFAULT_NUM_CTX


def recommend_num_parallel(*, vram_mb: float = 0, cpu_only: bool = False) -> int:
    if cpu_only or vram_mb < 22 * 1024:
        return 1
    return 2


def size_b_from_tag(tag: str | None) -> float:
    raw = (tag or "").lower()
    m = re.search(r":(\d+(?:\.\d+)?)b\b", raw)
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return 8.0
    return 8.0


def estimate_load_mode(
    *,
    vram_mb: float = 0,
    cpu_only: bool = False,
    num_ctx: int = DEFAULT_NUM_CTX,
    tag: str | None = None,
    min_vram_gb: float = 0,
) -> str:
    """Guess full-GPU vs hybrid from VRAM vs weights + a rough KV size."""
    if cpu_only or vram_mb < 512:
        return "cpu"
    size_b = size_b_from_tag(tag)
    weights_gb = float(min_vram_gb) * 0.8 if min_vram_gb else size_b * 0.65
    kv_gb = (max(int(num_ctx or MIN_NUM_CTX), MIN_NUM_CTX) / 1024.0) * max(size_b, 1.0) * 0.08
    vram_gb = float(vram_mb) / 1024.0
    if vram_gb + 0.05 >= weights_gb + kv_gb:
        return "full_gpu"
    return "hybrid"


def resolve_num_ctx(
    *,
    env_val: str | None = None,
    model_max: int | None = None,
    default: int = DEFAULT_NUM_CTX,
    vram_mb: float = 0,
    cpu_only: bool = False,
    job_num_ctx: int | None = None,
    family: str | None = None,
    job_ctx_wins: bool = False,
) -> int:
    rec = recommend_num_ctx(vram_mb=vram_mb, cpu_only=cpu_only, family=family)
    env_wanted = min(parse_env_num_ctx(env_val, rec), PUBLIC_MAX_CTX, rec)
    wanted = env_wanted
    fam = str(family or "").strip().lower()
    if fam == "graph":
        job_ctx_wins = True
    if job_num_ctx is not None:
        try:
            jn = int(job_num_ctx)
        except (TypeError, ValueError):
            jn = 0
        if jn >= MIN_NUM_CTX:
            if job_ctx_wins:
                wanted = min(jn, PUBLIC_MAX_CTX)
            elif jn > env_wanted:
                # Honor a larger job window; never shrink the env slot (a 4k think
                # hop used to unload a 16k runner).
                wanted = min(jn, rec, PUBLIC_MAX_CTX)
    if model_max and model_max >= MIN_NUM_CTX:
        wanted = min(wanted, int(model_max))
    return max(MIN_NUM_CTX, wanted)
