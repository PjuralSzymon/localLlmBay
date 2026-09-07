"""Node-local family ladder, fit/speed grades, and hysteresis (no server imports).

Keep FAMILY_LADDERS / MODEL_META in sync with server/catalog.py.
"""
from __future__ import annotations

from typing import Any, Optional

FAMILY_LADDERS: dict[str, list[str]] = {
    "granite": [
        "granite4.2:30b",
        "granite4.2:8b-q5_K_M",
        "granite4.2:8b",
    ],
    "qwen-coder": [
        "qwen2.5-coder:32b",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:7b-instruct-q6_K",
        "qwen2.5-coder:7b-instruct-q5_K_M",
        "qwen2.5-coder:7b",
        "qwen2.5-coder:3b-instruct-q8_0",
        "qwen2.5-coder:3b",
    ],
    "ministral": [
        "ministral-3:14b",
        "ministral-3:8b",
        "ministral-3:3b-instruct-2512-q8_0",
        "ministral-3:3b",
    ],
    "qwen3.5": [
        "qwen3.5:35b",
        "qwen3.5:27b",
        "qwen3.5:9b",
        "qwen3.5:4b-q8_0",
        "qwen3.5:4b",
        "qwen3.5:2b",
    ],
    "deepseek": [
        "deepseek-r1:32b",
        "deepseek-r1:14b",
        "deepseek-r1:8b",
        "deepseek-r1:1.5b",
    ],
    "flux2": [
        "x/flux2-klein:9b",
        "x/flux2-klein:4b-fp8",
        "x/flux2-klein:4b",
    ],
    "wan": [
        "x/wan2.1-t2v:1.3b",
    ],
    "graph": [
        "granite3.3:8b",
        "deepseek-coder:6.7b",
        "qwen2.5-coder:7b",
    ],
    "gemma": [
        "gemma3:4b",
    ],
}

GRAPH_FAMILY = "graph"
GRAPH_ORCH_TAG = "granite3.3:8b"
GRAPH_SKETCH_TAG = "deepseek-coder:6.7b"
GRAPH_FINAL_TAG = "qwen2.5-coder:7b"
GRAPH_IMAGE_TAG = "x/flux2-klein:4b"
GRAPH_VIDEO_TAG = "x/wan2.1-t2v:1.3b"
GRAPH_DOCS_TAG = "gemma3:4b"
GRAPH_PACK: tuple[str, ...] = (GRAPH_ORCH_TAG, GRAPH_SKETCH_TAG, GRAPH_FINAL_TAG)
GRAPH_NUM_CTX = 16384
GRAPH_ROLE_TAGS: dict[str, str] = {
    "orchestrator": GRAPH_ORCH_TAG,
    "coder_sketch": GRAPH_SKETCH_TAG,
    "coder_architect": GRAPH_SKETCH_TAG,
    "coder_info": GRAPH_SKETCH_TAG,
    "coder_impl": GRAPH_FINAL_TAG,
    "coder_verify": GRAPH_FINAL_TAG,
    "coder_final": GRAPH_FINAL_TAG,
    "image_pass1": GRAPH_IMAGE_TAG,
    "image_pass2": GRAPH_IMAGE_TAG,
    "image_pass3": GRAPH_IMAGE_TAG,
    "video_pass1": GRAPH_VIDEO_TAG,
    "docs_writer": GRAPH_DOCS_TAG,
}
GRAPH_VARIANT_LABELS: dict[str, str] = {
    GRAPH_ORCH_TAG: "8b-orch",
    GRAPH_SKETCH_TAG: "6.7b-sketch",
    GRAPH_FINAL_TAG: "7b-final",
    GRAPH_IMAGE_TAG: "4b-fp4",
    GRAPH_VIDEO_TAG: "1.3b-t2v",
    GRAPH_DOCS_TAG: "4b-q4",
    "granite4.2:3b": "3b-orch",
    "deepseek-coder:1.3b-instruct-q5_K_M": "1.3b-sketch",
    "qwen2.5-coder:3b": "3b-final",
}

GRAPHICAL_FAMILIES: frozenset[str] = frozenset({"flux2"})
VIDEO_FAMILIES: frozenset[str] = frozenset({"wan"})

FAMILY_ALIASES: dict[str, str] = {
    "qwen": "qwen-coder",
    "qwen2.5": "qwen-coder",
    "qwen-2.5": "qwen-coder",
    "qwen2.5-coder": "qwen-coder",
    "qwencoder": "qwen-coder",
    "qwen-3.5": "qwen3.5",
    "qwen35": "qwen3.5",
    "ministral-3": "ministral",
    "ministral3": "ministral",
    "deepseek-r1": "deepseek",
    "deepseekr1": "deepseek",
    "r1": "deepseek",
    "flux": "flux2",
    "flux.2": "flux2",
    "flux-2": "flux2",
    "flux2-klein": "flux2",
    "flux2klein": "flux2",
    "wan": "wan",
    "wan2": "wan",
    "wan2.1": "wan",
    "wan-t2v": "wan",
    "wan2.1-t2v": "wan",
    "gemma": "gemma",
    "gemma3": "gemma",
    "graph": "graph",
    "graph-image": "graph",
    "graph-video": "graph",
    "graph-docs": "graph",
}

FAMILY_DEFAULTS: dict[str, str] = {
    "granite": "granite4.2:8b",
    "qwen-coder": "qwen2.5-coder:7b",
    "ministral": "ministral-3:8b",
    "qwen3.5": "qwen3.5:4b",
    "deepseek": "deepseek-r1:8b",
    "flux2": "x/flux2-klein:4b",
    "wan": "x/wan2.1-t2v:1.3b",
    "gemma": "gemma3:4b",
    "graph": "granite3.3:8b",
}

MODEL_META: dict[str, dict] = {
    "granite4.2:30b": {"min_vram_gb": 24, "min_ram_gb": 32, "cpu_ok": True, "size": "30b", "quant": "q4_k_m", "disk_gb": 18.0},
    "granite4.2:8b": {"min_vram_gb": 8, "min_ram_gb": 16, "cpu_ok": True, "size": "8b", "quant": "q4_k_m", "disk_gb": 5.3},
    "granite4.2:8b-q5_K_M": {"min_vram_gb": 12, "min_ram_gb": 16, "cpu_ok": True, "size": "8b", "quant": "q5_k_m", "disk_gb": 6.3},
    "granite4.2:3b": {"min_vram_gb": 0, "min_ram_gb": 8, "cpu_ok": True, "size": "3b", "quant": "q4_k_m", "disk_gb": 2.1},
    "granite3.3:8b": {"min_vram_gb": 6, "min_ram_gb": 16, "cpu_ok": True, "size": "8b", "quant": "q4_k_m", "disk_gb": 4.9},
    "qwen2.5-coder:32b": {"min_vram_gb": 24, "min_ram_gb": 32, "cpu_ok": True, "size": "32b", "quant": "q4_k_m", "disk_gb": 20.0},
    "qwen2.5-coder:14b": {"min_vram_gb": 12, "min_ram_gb": 32, "cpu_ok": False, "size": "14b", "quant": "q4_k_m", "disk_gb": 9.0},
    "qwen2.5-coder:7b-instruct-q6_K": {"min_vram_gb": 8, "min_ram_gb": 16, "cpu_ok": False, "size": "7b", "quant": "q6_k", "disk_gb": 6.3},
    "qwen2.5-coder:7b-instruct-q5_K_M": {"min_vram_gb": 7, "min_ram_gb": 16, "cpu_ok": False, "size": "7b", "quant": "q5_k_m", "disk_gb": 5.4},
    "qwen2.5-coder:7b": {"min_vram_gb": 6, "min_ram_gb": 16, "cpu_ok": True, "size": "7b", "quant": "q4_k_m", "disk_gb": 4.7},
    "qwen2.5-coder:3b-instruct-q8_0": {"min_vram_gb": 4, "min_ram_gb": 16, "cpu_ok": True, "size": "3b", "quant": "q8_0", "disk_gb": 3.3},
    "qwen2.5-coder:3b": {"min_vram_gb": 0, "min_ram_gb": 8, "cpu_ok": True, "size": "3b", "quant": "q4_k_m", "disk_gb": 1.9},
    "ministral-3:14b": {"min_vram_gb": 12, "min_ram_gb": 32, "cpu_ok": False, "size": "14b", "quant": "q4_k_m", "disk_gb": 9.1},
    "ministral-3:8b": {"min_vram_gb": 8, "min_ram_gb": 16, "cpu_ok": True, "size": "8b", "quant": "q4_k_m", "disk_gb": 6.0},
    "ministral-3:3b-instruct-2512-q8_0": {"min_vram_gb": 5, "min_ram_gb": 16, "cpu_ok": True, "size": "3b", "quant": "q8_0", "disk_gb": 4.5},
    "ministral-3:3b": {"min_vram_gb": 3, "min_ram_gb": 8, "cpu_ok": True, "size": "3b", "quant": "q4_k_m", "disk_gb": 3.0},
    "qwen3.5:35b": {"min_vram_gb": 24, "min_ram_gb": 48, "cpu_ok": False, "size": "35b", "quant": "q4_k_m", "disk_gb": 24.0},
    "qwen3.5:27b": {"min_vram_gb": 24, "min_ram_gb": 48, "cpu_ok": False, "size": "27b", "quant": "q4_k_m", "disk_gb": 17.0},
    "qwen3.5:9b": {"min_vram_gb": 8, "min_ram_gb": 16, "cpu_ok": False, "size": "9b", "quant": "q4_k_m", "disk_gb": 6.6},
    "qwen3.5:4b-q8_0": {"min_vram_gb": 6, "min_ram_gb": 16, "cpu_ok": False, "size": "4b", "quant": "q8_0", "disk_gb": 5.3},
    "qwen3.5:4b": {"min_vram_gb": 4, "min_ram_gb": 8, "cpu_ok": True, "size": "4b", "quant": "q4_k_m", "disk_gb": 3.4},
    "qwen3.5:2b": {"min_vram_gb": 0, "min_ram_gb": 8, "cpu_ok": True, "size": "2b", "quant": "q8_0", "disk_gb": 2.7},
    "deepseek-r1:32b": {"min_vram_gb": 24, "min_ram_gb": 32, "cpu_ok": True, "size": "32b", "quant": "q4_k_m", "disk_gb": 20.0},
    "deepseek-r1:14b": {"min_vram_gb": 12, "min_ram_gb": 32, "cpu_ok": False, "size": "14b", "quant": "q4_k_m", "disk_gb": 9.0},
    "deepseek-r1:8b": {"min_vram_gb": 8, "min_ram_gb": 16, "cpu_ok": True, "size": "8b", "quant": "q4_k_m", "disk_gb": 5.2},
    "deepseek-r1:1.5b": {"min_vram_gb": 0, "min_ram_gb": 8, "cpu_ok": True, "size": "1.5b", "quant": "q4_k_m", "disk_gb": 1.1},
    "deepseek-coder:1.3b-instruct-q5_K_M": {"min_vram_gb": 0, "min_ram_gb": 8, "cpu_ok": True, "size": "1.3b", "quant": "q5_k_m", "disk_gb": 1.0},
    "deepseek-coder:6.7b": {"min_vram_gb": 5, "min_ram_gb": 8, "cpu_ok": True, "size": "6.7b", "quant": "q4_k_m", "disk_gb": 3.8},
    "x/flux2-klein:9b": {"min_vram_gb": 12, "min_ram_gb": 16, "cpu_ok": False, "size": "9b", "quant": "fp4", "disk_gb": 12.0, "kind": "image"},
    "x/flux2-klein:4b-fp8": {"min_vram_gb": 10, "min_ram_gb": 16, "cpu_ok": False, "size": "4b", "quant": "fp8", "disk_gb": 9.5, "kind": "image"},
    "x/flux2-klein:4b": {"min_vram_gb": 6, "min_ram_gb": 16, "cpu_ok": False, "size": "4b", "quant": "fp4", "disk_gb": 8.0, "kind": "image"},
    "x/wan2.1-t2v:1.3b": {"min_vram_gb": 8, "min_ram_gb": 16, "cpu_ok": False, "size": "1.3b", "quant": "bf16", "disk_gb": 16.0, "kind": "video"},
    "gemma3:4b": {"min_vram_gb": 4, "min_ram_gb": 8, "cpu_ok": True, "size": "4b", "quant": "q4_k_m", "disk_gb": 3.3},
}

FIT_THRESHOLD = 50
HYBRID_FIT = 25
HYBRID_OVERHEAD_GB = 6.0
# nvidia-smi on an "8 GB" card often reports ~7.5-8.0 GiB. Do not treat that as 12 GB+.
VRAM_SLACK_GB = 0.75
UNKNOWN_GPU_MAX_MIN_VRAM_GB = 8.0
PROBE_FIT_MIN = 20
DEFAULT_TARGET_TPS = 8.0
DEFAULT_WALL_SLOW_SECONDS = 90.0
DEFAULT_PROMPT_EVAL_SLOW_SECONDS = 20.0
DEMOTE_SPEED = 40
PROMOTE_SPEED = 75
PROBE_GOOD_SPEED = 50
DEMOTE_STREAK = 2
PROMOTE_STREAK = 3
PROBE_AFTER_JOBS = 3
PROBE_RETRY_JOBS = 20
COOLDOWN_JOBS = 2
VERY_LOW_TPS = 2.0
HISTORY_N = 40

CTX_GRANITE_8B = (65536, 32768, 16384)
CTX_AGENT_FLOOR = (16384,)
CTX_SMALL = (16384, 8192, 4096)
CTX_GRAPH = (GRAPH_NUM_CTX,)


def _size_b(tag: str) -> float:
    meta = MODEL_META.get(tag) or {}
    raw = str(meta.get("size") or "8b").lower().replace("b", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 8.0


def ctx_windows_for_tag(tag: str) -> tuple[int, ...]:
    """Frozen windows for one Ollama tag. Image tags have no chat KV ladder."""
    meta = MODEL_META.get(tag) or {}
    if str(meta.get("kind") or "") in ("image", "video"):
        return ()
    if tag in ("granite4.2:8b", "granite4.2:8b-q5_K_M"):
        return CTX_GRANITE_8B
    if tag in GRAPH_PACK:
        return CTX_GRAPH
    if _size_b(tag) >= 7:
        return CTX_AGENT_FLOOR
    return CTX_SMALL


def family_rungs(family: str) -> list[tuple[str, int]]:
    """Size desc, then quant desc, then ctx desc. Flux: tag with num_ctx 0."""
    fam = resolve_family(family)
    out: list[tuple[str, int]] = []
    for tag in FAMILY_LADDERS.get(fam) or []:
        wins = ctx_windows_for_tag(tag)
        if not wins:
            out.append((tag, 0))
            continue
        for n in wins:
            out.append((tag, int(n)))
    return out


def estimate_kv_gb(tag: str, num_ctx: int | None) -> float:
    if not num_ctx:
        return 0.0
    n = max(int(num_ctx), 4096)
    return (n / 16384.0) * max(_size_b(tag), 1.0) * 0.5


def estimate_rung_gb(tag: str, num_ctx: int | None) -> float:
    return _weights_gb(tag) + estimate_kv_gb(tag, num_ctx)


def resolve_family(family: str | None) -> str:
    fam = (family or "").strip().lower()
    if fam in FAMILY_LADDERS:
        return fam
    return FAMILY_ALIASES.get(fam, fam or "granite")


def is_image_family(family: str | None) -> bool:
    return resolve_family(family) in GRAPHICAL_FAMILIES


def is_video_family(family: str | None) -> bool:
    return resolve_family(family) in VIDEO_FAMILIES


def is_image_tag(tag: str | None) -> bool:
    meta = MODEL_META.get(str(tag or "").strip()) or {}
    return str(meta.get("kind") or "") == "image"


def is_video_tag(tag: str | None) -> bool:
    meta = MODEL_META.get(str(tag or "").strip()) or {}
    return str(meta.get("kind") or "") == "video"


def clamp(n: float, lo: float, hi: float) -> float:
    return lo if n < lo else hi if n > hi else n


def _vram_gb(hardware: dict | None) -> float:
    hw = hardware or {}
    try:
        return float(hw.get("vram_mb_total") or 0) / 1024.0
    except (TypeError, ValueError):
        return 0.0


def _ram_gb(hardware: dict | None) -> float:
    hw = hardware or {}
    try:
        n = float(hw.get("ram_gb") or 0)
    except (TypeError, ValueError):
        n = 0.0
    # Setup assumes 16 GB when RAM is unknown; same here for CPU / KV-offload grades.
    return n if n > 0 else 16.0


def _cpu_only(hardware: dict | None) -> bool:
    hw = hardware or {}
    backend = str(hw.get("backend") or "cpu").lower()
    try:
        gpus = int(hw.get("gpu_count") or 0)
    except (TypeError, ValueError):
        gpus = 0
    if gpus > 0 and backend not in ("cpu",):
        return False
    if backend in ("cuda", "vulkan", "rocm", "metal"):
        return False
    return gpus <= 0 or backend == "cpu"


def gpu_resident_ok(tag: str, hardware: dict | None) -> bool:
    """True when this GPU can hold catalog weights, or leftover layers fit in RAM.

    Image tags stay GPU-resident (no RAM offload). Text tags may fill VRAM and
    put the rest on CPU/RAM.
    """
    meta = MODEL_META.get(tag) or {}
    try:
        min_v = float(meta.get("min_vram_gb") or 0)
    except (TypeError, ValueError):
        min_v = 0.0
    image = str(meta.get("kind") or "") in ("image", "video")
    if image:
        if hardware is None or _cpu_only(hardware):
            return False
        vram = _vram_gb(hardware)
        if vram <= 0:
            return False
        return vram + VRAM_SLACK_GB >= min_v
    if hardware is None:
        return (
            min_v <= UNKNOWN_GPU_MAX_MIN_VRAM_GB
            or _hybrid_fit(tag, UNKNOWN_GPU_MAX_MIN_VRAM_GB, 16.0) > 0
        )
    if _cpu_only(hardware):
        return True
    vram = _vram_gb(hardware)
    ram = _ram_gb(hardware)
    assume = vram if vram > 0 else UNKNOWN_GPU_MAX_MIN_VRAM_GB
    if assume + VRAM_SLACK_GB >= min_v:
        return True
    return _hybrid_fit(tag, assume, ram) > 0


def _weights_gb(tag: str) -> float:
    meta = MODEL_META.get(tag) or {}
    try:
        disk = float(meta.get("disk_gb") or 0)
    except (TypeError, ValueError):
        disk = 0.0
    if disk > 0:
        return disk
    try:
        return float(meta.get("min_vram_gb") or 0)
    except (TypeError, ValueError):
        return 0.0


def _hybrid_fit(tag: str, vram: float, ram: float) -> int:
    """Non-zero when RAM can hold the layers that do not fit in VRAM."""
    weights = _weights_gb(tag)
    need_ram = max(0.0, weights - max(0.0, vram)) + HYBRID_OVERHEAD_GB
    if ram >= need_ram:
        return HYBRID_FIT
    if ram >= need_ram * 0.75:
        return 15
    return 0


def _fit_grade_weights(tag: str, hardware: dict | None) -> int:
    """Fit from weights vs VRAM/RAM only (no KV window)."""
    meta = MODEL_META.get(tag) or {}
    min_v = float(meta.get("min_vram_gb") or 0)
    min_r = float(meta.get("min_ram_gb") or 0)
    cpu_ok = bool(meta.get("cpu_ok"))
    cpu = _cpu_only(hardware)
    ram = _ram_gb(hardware)
    vram = _vram_gb(hardware)
    image = str(meta.get("kind") or "") in ("image", "video")

    if image:
        if cpu or not gpu_resident_ok(tag, hardware):
            return 0
        if min_v <= 0:
            return 70
        if vram >= min_v * 1.3:
            return 100
        if vram >= min_v:
            return 70
        return 30

    if cpu:
        weights = _weights_gb(tag)
        need = max(min_r, weights + HYBRID_OVERHEAD_GB) if (cpu_ok or weights > 0) else min_r
        if not cpu_ok and ram < need:
            return 0
        if ram < min_r * 0.7 and ram < weights + 2:
            return 0
        if min_r <= 0 and weights <= 0:
            return 100
        if ram >= max(min_r, weights) * 1.3:
            return 100
        if ram >= max(min_r, weights):
            return 70
        if ram >= need * 0.75 or ram >= weights + HYBRID_OVERHEAD_GB:
            return HYBRID_FIT
        return 0

    if not gpu_resident_ok(tag, hardware):
        return 0

    assume = vram if vram > 0 else UNKNOWN_GPU_MAX_MIN_VRAM_GB
    if min_v <= 0:
        if ram >= min_r * 1.3 or min_r <= 0:
            return 100
        if ram >= min_r:
            return 70
        return 30 if ram >= min_r * 0.7 else 0

    if assume + VRAM_SLACK_GB >= min_v:
        if assume >= min_v * 1.3:
            return 100
        if assume >= min_v:
            return 70
        return 30
    return _hybrid_fit(tag, assume, ram)


def fit_grade(tag: str, hardware: dict | None, num_ctx: int | None = None) -> int:
    """0-100 hardware fit for a ladder tag, optionally at a context window.

    0 means weights+KV cannot load even as a GPU+CPU split.
    15-25 is leftover layers or KV sitting in RAM while the GPU stays full.
    Passing num_ctx uses a milder KV estimate so 8B@64k can hybrid on 8+16 GB.
    """
    grade = _fit_grade_weights(tag, hardware)
    if grade <= 0:
        return grade
    try:
        n = int(num_ctx or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return grade
    meta = MODEL_META.get(tag) or {}
    if str(meta.get("kind") or "") in ("image", "video"):
        return grade
    need = estimate_rung_gb(tag, n)
    pool = _vram_gb(hardware) + _ram_gb(hardware)
    if pool + 0.05 < need:
        return 0
    vram = _vram_gb(hardware)
    if vram > 0 and need > vram + 0.05 and grade > HYBRID_FIT:
        return HYBRID_FIT
    return grade


def decode_tps(eval_count: int | float | None, eval_duration_ns: int | float | None) -> Optional[float]:
    try:
        count = float(eval_count or 0)
        ns = float(eval_duration_ns or 0)
    except (TypeError, ValueError):
        return None
    if count <= 0 or ns <= 0:
        return None
    return count / (ns / 1e9)


def speed_grade(tps: float | None, target_tps: float = DEFAULT_TARGET_TPS) -> int:
    if tps is None or tps < 0 or target_tps <= 0:
        return 0
    return int(round(clamp(100.0 * float(tps) / float(target_tps), 0.0, 100.0)))


def short_variant(tag: str) -> str:
    graph_label = GRAPH_VARIANT_LABELS.get(tag)
    if graph_label:
        return graph_label
    meta = MODEL_META.get(tag) or {}
    size = str(meta.get("size") or "")
    quant = str(meta.get("quant") or "").lower()
    if "fp8" in quant:
        q = "fp8"
    elif "fp4" in quant:
        q = "fp4"
    elif "q8" in quant:
        q = "q8"
    elif "q6" in quant:
        q = "q6"
    elif "q5" in quant:
        q = "q5"
    elif "q4" in quant:
        q = "q4"
    else:
        q = ""
    if size and q:
        return f"{size}-{q}"
    if size:
        return size
    if ":" in tag:
        return tag.split(":", 1)[-1]
    return tag


def short_ctx_label(num_ctx: int | None) -> str:
    try:
        n = int(num_ctx or 0)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def _family_from_ladder(ladder: list[str]) -> str:
    for fam, tags in FAMILY_LADDERS.items():
        if ladder and ladder[0] in tags:
            return fam
    return "granite"


def pulled_rungs(family: str, pulled: list[str]) -> list[tuple[str, int]]:
    want = set(pulled or [])
    return [(tag, n) for tag, n in family_rungs(family) if tag in want]


def pick_initial_rung(
    family: str, pulled: list[str], hardware: dict | None
) -> tuple[Optional[str], Optional[int]]:
    """Heaviest (size, quant, ctx) rung with Fit > 0 among pulled tags."""
    rungs = pulled_rungs(family, pulled)
    for tag, nctx in rungs:
        if fit_grade(tag, hardware, nctx) > 0:
            return tag, nctx
    return None, None


def pick_initial_tag(ladder: list[str], pulled: list[str], hardware: dict | None) -> Optional[str]:
    family = _family_from_ladder(ladder)
    tag, _n = pick_initial_rung(family, pulled, hardware)
    return tag


def _ladder_index(ladder: list[str], tag: str | None) -> int:
    if not tag or tag not in ladder:
        return 10**9
    return ladder.index(tag)


def tags_fitting_vram(pulled: list[str], hardware: dict | None) -> list[str]:
    """Keep pulled tags that have at least one context window with Fit > 0."""
    out: list[str] = []
    for tag in pulled or []:
        wins = ctx_windows_for_tag(tag)
        if not wins:
            if fit_grade(tag, hardware) > 0:
                out.append(tag)
            continue
        if any(fit_grade(tag, hardware, n) > 0 for n in wins):
            out.append(tag)
    return out


def skip_rows_for_tags(tags: list[str], hardware: dict | None) -> list[dict[str, Any]]:
    """Human-readable skip list for tags that will not be pulled, loaded, or probed."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    unique = [t for t in (tags or []) if t]
    fitting = set(tags_fitting_vram(unique, hardware))
    vram = _vram_gb(hardware) if hardware else 0.0
    cpu = bool(hardware) and _cpu_only(hardware)
    for tag in unique:
        if tag in seen:
            continue
        seen.add(tag)
        if tag in fitting:
            continue
        meta = MODEL_META.get(tag) or {}
        try:
            min_v = float(meta.get("min_vram_gb") or 0)
        except (TypeError, ValueError):
            min_v = 0.0
        if cpu:
            reason = "Too big for this RAM"
        elif vram > 0:
            reason = f"Too big even as GPU + CPU split (this GPU has {vram:.0f} GB)"
        else:
            reason = "Too big even as GPU + CPU split"
        rows.append(
            {
                "tag": tag,
                "variant": short_variant(tag),
                "reason": reason,
                "min_vram_gb": min_v,
            }
        )
    return rows


def filter_models_for_vram(
    models_csv: str, vram_mb: float | None, ram_gb: float | None = None
) -> list[str]:
    """start.sh helper: keep NODE_MODELS tags this GPU+RAM can actually load."""
    tags = [t.strip() for t in str(models_csv or "").split(",") if t.strip()]
    try:
        n = float(vram_mb) if vram_mb is not None else 0.0
    except (TypeError, ValueError):
        n = 0.0
    try:
        ram = float(ram_gb) if ram_gb not in (None, "") else 16.0
    except (TypeError, ValueError):
        ram = 16.0
    if ram <= 0:
        ram = 16.0
    if n > 0:
        hw: dict | None = {
            "backend": "cuda",
            "gpu_count": 1,
            "vram_mb_total": n,
            "ram_gb": ram,
        }
    else:
        hw = None
    return tags_fitting_vram(tags, hw)


def history_has_oom_on_tag(history: list | None, tag: str) -> bool:
    """True if a past OOM was on this exact tag. Heavier-tag OOM does not pin a lighter leftover."""
    want = str(tag or "")
    for h in history or []:
        if not isinstance(h, dict) or not h.get("oom"):
            continue
        if str(h.get("tag") or "") == want:
            return True
    return False


def _rung_index(rungs: list[tuple[str, int]], tag: str | None, nctx: int | None) -> int:
    if not tag:
        return 10**9
    try:
        n = int(nctx or 0)
    except (TypeError, ValueError):
        n = 0
    pair = (tag, n)
    if pair in rungs:
        return rungs.index(pair)
    for i, (t, _w) in enumerate(rungs):
        if t == tag:
            return i
    return 10**9


def boot_reconcile(
    state: dict[str, Any],
    hardware: dict | None,
    pulled: list[str],
) -> dict[str, Any]:
    """Start at the heaviest fitting rung. Keep a working heavier probe. Unstick leftovers."""
    family = resolve_family(state.get("family") or "granite")
    raw_pulled = list(pulled or state.get("pulled") or [])
    pulled_list = tags_fitting_vram(raw_pulled, hardware)
    rungs = pulled_rungs(family, pulled_list)
    wanted_tag, wanted_ctx = pick_initial_rung(family, pulled_list, hardware)
    active = state.get("active_tag")
    try:
        active_ctx = int(state.get("active_num_ctx") or 0)
    except (TypeError, ValueError):
        active_ctx = 0
    out = dict(state)
    out["family"] = family
    out["pulled"] = pulled_list
    out["rebased"] = False
    if not wanted_tag:
        return out

    hist = list(state.get("history") or [])
    if wanted_tag and history_has_oom_on_tag(hist, wanted_tag):
        alt = None
        for tag, nctx in rungs:
            if fit_grade(tag, hardware, nctx) > 0 and not history_has_oom_on_tag(hist, tag):
                alt = (tag, nctx)
                break
        if alt:
            wanted_tag, wanted_ctx = alt
    want_i = _rung_index(rungs, wanted_tag, wanted_ctx)
    act_i = _rung_index(rungs, active, active_ctx)
    active_fit = fit_grade(str(active), hardware, active_ctx or None) if active else 0

    def _apply(tag: str, nctx: int | None, mood: str) -> dict[str, Any]:
        out["active_tag"] = tag
        out["active_num_ctx"] = nctx
        out["rebased"] = True
        out["last_step"] = None
        out["slow_streak"] = 0
        out["fast_streak"] = 0
        out["cooldown"] = 0
        out["demote_reason"] = None
        out["mood"] = mood
        return out

    if not active or act_i >= 10**9 or active_fit <= 0:
        if wanted_tag != active or wanted_ctx != active_ctx:
            return _apply(
                wanted_tag,
                wanted_ctx,
                f"Serving {short_variant(wanted_tag)} / {short_ctx_label(wanted_ctx)}",
            )
        return out

    if act_i > want_i:
        if str(state.get("demote_reason") or "") == "oom":
            return out
        return _apply(
            wanted_tag,
            wanted_ctx,
            f"Rebase - {short_variant(wanted_tag)} / {short_ctx_label(wanted_ctx)} is the start rung",
        )

    if act_i < want_i and history_has_oom_on_tag(hist, str(active)):
        return _apply(
            wanted_tag,
            wanted_ctx,
            f"Serving {short_variant(wanted_tag)} / {short_ctx_label(wanted_ctx)}",
        )

    return out


def next_lower(ladder: list[str], active: str) -> Optional[str]:
    if active not in ladder:
        return None
    i = ladder.index(active)
    if i + 1 < len(ladder):
        return ladder[i + 1]
    return None


def next_higher(ladder: list[str], active: str) -> Optional[str]:
    if active not in ladder:
        return None
    i = ladder.index(active)
    if i > 0:
        return ladder[i - 1]
    return None


def next_lower_rung(
    rungs: list[tuple[str, int]], tag: str | None, nctx: int | None
) -> Optional[tuple[str, int]]:
    i = _rung_index(rungs, tag, nctx)
    if i >= 10**9:
        return None
    if i + 1 < len(rungs):
        return rungs[i + 1]
    return None


def next_higher_rung(
    rungs: list[tuple[str, int]], tag: str | None, nctx: int | None
) -> Optional[tuple[str, int]]:
    i = _rung_index(rungs, tag, nctx)
    if i >= 10**9 or i <= 0:
        return None
    return rungs[i - 1]


def default_num_ctx(tag: str | None) -> int:
    wins = ctx_windows_for_tag(str(tag or ""))
    return int(wins[0]) if wins else 0


def empty_state(
    family: str,
    active_tag: str,
    pulled: list[str] | None = None,
    active_num_ctx: int | None = None,
) -> dict[str, Any]:
    nctx = active_num_ctx if active_num_ctx is not None else default_num_ctx(active_tag)
    variant = f"{short_variant(active_tag)} - {short_ctx_label(nctx)}".rstrip(" -") if active_tag else ""
    mood = f"Serving {variant}" if variant else "No model fits this GPU"
    return {
        "family": resolve_family(family),
        "active_tag": active_tag,
        "active_num_ctx": nctx,
        "pulled": list(pulled or []),
        "slow_streak": 0,
        "fast_streak": 0,
        "cooldown": 0,
        "last_step": None,
        "last_fit": None,
        "last_speed": None,
        "last_tps": None,
        "mood": mood,
        "history": [],
        "rung_stats": {},
        "good_on_rung": 0,
        "probe_wait": 0,
        "demote_reason": None,
        "rebased": False,
    }


def _rung_key(tag: str, nctx: int | None) -> str:
    return f"{tag}@{int(nctx or 0)}"


def _rung_stats(state: dict, tag: str, nctx: int | None = None) -> dict:
    stats = state.setdefault("rung_stats", {})
    key = _rung_key(tag, nctx) if nctx is not None else tag
    row = stats.setdefault(key, {"jobs": 0, "last_speed": None, "last_tps": None})
    return row


def _mood(
    active: str,
    stepped: str | None,
    speed: int | None,
    fit: int | None,
    num_ctx: int | None = None,
) -> str:
    variant = f"{short_variant(active)} - {short_ctx_label(num_ctx)}".rstrip(" -")
    if stepped == "demote":
        return f"Busy PC - stepped down to {variant}"
    if stepped == "promote":
        return f"Headroom - probing {variant}"
    if speed is not None and speed < DEMOTE_SPEED:
        return f"Busy PC - holding {variant}"
    if fit is not None and fit >= 70 and speed is not None and speed > PROMOTE_SPEED:
        return f"Headroom - {variant} is comfortable"
    return f"Serving {variant}"


THINK_HOP_MAX_PREDICT = 400


def is_think_hop_job(job: dict | None = None, timings: dict | None = None) -> bool:
    """True for the slim plan hop: no tools and num_predict ? 400."""
    job = job or {}
    tools = job.get("tools")
    if isinstance(tools, list) and tools:
        return False
    timings = timings or {}
    infer = job.get("infer") if isinstance(job.get("infer"), dict) else {}
    raw = timings.get("num_predict")
    if raw is None:
        raw = infer.get("num_predict")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return False
    return 0 < n <= THINK_HOP_MAX_PREDICT


def apply_adapt(
    state: dict[str, Any],
    *,
    hardware: dict | None,
    pulled: list[str],
    decode_tps_val: float | None,
    wall_s: float,
    oom: bool = False,
    target_tps: float = DEFAULT_TARGET_TPS,
    wall_slow_s: float = DEFAULT_WALL_SLOW_SECONDS,
    prompt_eval_s: float = 0.0,
    prompt_eval_slow_s: float = DEFAULT_PROMPT_EVAL_SLOW_SECONDS,
    ctx_overflow: bool = False,
    skip_slow: bool = False,
) -> dict[str, Any]:
    """Update streaks and maybe step one (tag, ctx) rung.

    OOM/load fail demotes one rung (smaller ctx on the same tag, then next tag).
    Context overflow does not demote - a smaller window makes overflow worse.
    Granite never slow-demotes. Probe-up uses the next heavier rung with Fit > 0.
    """
    family = resolve_family(state.get("family") or "granite")
    raw_pulled = list(pulled or [])
    pulled = tags_fitting_vram(raw_pulled, hardware)
    rungs = pulled_rungs(family, pulled)
    init_tag, init_ctx = pick_initial_rung(family, pulled, hardware)
    active = state.get("active_tag") or init_tag
    try:
        active_ctx = int(state.get("active_num_ctx") or 0)
    except (TypeError, ValueError):
        active_ctx = 0
    if not active_ctx:
        active_ctx = int(init_ctx or default_num_ctx(active) or 0)
    if _rung_index(rungs, active, active_ctx) >= 10**9 and init_tag:
        active, active_ctx = init_tag, int(init_ctx or 0)

    fit = fit_grade(active, hardware, active_ctx) if active else 0
    speed = speed_grade(decode_tps_val, target_tps)
    cooldown = int(state.get("cooldown") or 0)
    slow = int(state.get("slow_streak") or 0)
    fast = int(state.get("fast_streak") or 0)
    good_on_rung = int(state.get("good_on_rung") or 0)
    probe_wait = int(state.get("probe_wait") or 0)
    last_step = state.get("last_step")
    demote_reason = state.get("demote_reason")
    stepped: str | None = None
    new_active = active
    new_ctx = active_ctx

    can_demote = cooldown <= 0 or last_step == "demote"
    can_promote = cooldown <= 0 or last_step == "promote"

    if ctx_overflow or (skip_slow and not oom):
        stepped = None
        if stepped is None and cooldown > 0:
            cooldown -= 1
        if stepped is None and probe_wait > 0:
            probe_wait -= 1
    elif oom:
        nxt = next_lower_rung(rungs, active, active_ctx)
        if nxt:
            new_active, new_ctx = nxt
            stepped = "demote"
            demote_reason = "oom"
            slow = 0
            fast = 0
            good_on_rung = 0
            cooldown = COOLDOWN_JOBS
            last_step = "demote"
            if state.get("last_step") == "promote":
                probe_wait = PROBE_RETRY_JOBS
    else:
        very_slow = decode_tps_val is not None and decode_tps_val < VERY_LOW_TPS
        decode_ok = decode_tps_val is not None and speed >= DEMOTE_SPEED
        wall_slow = float(wall_s) > float(wall_slow_s)
        prompt_slow = float(prompt_eval_s or 0) > float(prompt_eval_slow_s)
        if very_slow and wall_slow:
            very_slow = False
        stall = (wall_slow or prompt_slow) and not decode_ok
        if speed < DEMOTE_SPEED or stall:
            slow += 1
            fast = 0
            good_on_rung = 0
        elif speed > PROMOTE_SPEED:
            fast += 1
            slow = 0
            good_on_rung += 1
        else:
            slow = 0
            if speed >= PROBE_GOOD_SPEED:
                good_on_rung += 1

        if family != "granite" and can_demote and (slow >= DEMOTE_STREAK or very_slow):
            nxt = next_lower_rung(rungs, active, active_ctx)
            if nxt:
                new_active, new_ctx = nxt
                stepped = "demote"
                demote_reason = "slow"
                slow = 0
                fast = 0
                good_on_rung = 0
                cooldown = COOLDOWN_JOBS
                last_step = "demote"
                if state.get("last_step") == "promote":
                    probe_wait = PROBE_RETRY_JOBS
        elif can_promote:
            nxt = next_higher_rung(rungs, active, active_ctx)
            nxt_ok = bool(nxt and nxt[0] in set(pulled))
            nxt_fit = fit_grade(nxt[0], hardware, nxt[1]) if nxt_ok else 0
            if nxt_ok and fast >= PROMOTE_STREAK and nxt_fit >= FIT_THRESHOLD:
                new_active, new_ctx = nxt
                stepped = "promote"
                demote_reason = None
                slow = 0
                fast = 0
                good_on_rung = 0
                cooldown = COOLDOWN_JOBS
                last_step = "promote"
            elif nxt_ok and probe_wait <= 0 and good_on_rung >= PROBE_AFTER_JOBS and nxt_fit >= PROBE_FIT_MIN:
                new_active, new_ctx = nxt
                stepped = "promote"
                demote_reason = None
                slow = 0
                fast = 0
                good_on_rung = 0
                cooldown = COOLDOWN_JOBS
                last_step = "promote"

        if stepped is None and cooldown > 0:
            cooldown -= 1
        if stepped is None and probe_wait > 0:
            probe_wait -= 1

    history = list(state.get("history") or [])
    history.append(
        {
            "tag": active,
            "num_ctx": active_ctx,
            "variant": short_variant(active),
            "decode_tps": round(decode_tps_val, 2) if decode_tps_val is not None else None,
            "wall_s": round(float(wall_s), 1),
            "prompt_eval_s": round(float(prompt_eval_s or 0), 2),
            "speed_grade": speed,
            "fit_grade": fit,
            "oom": bool(oom),
            "ctx_overflow": bool(ctx_overflow),
        }
    )
    history = history[-HISTORY_N:]

    row = _rung_stats(state, active, active_ctx)
    row["jobs"] = int(row.get("jobs") or 0) + 1
    row["last_speed"] = speed
    if decode_tps_val is not None:
        row["last_tps"] = round(decode_tps_val, 2)

    out = dict(state)
    out.update(
        {
            "family": family,
            "active_tag": new_active,
            "active_num_ctx": new_ctx,
            "pulled": list(pulled),
            "slow_streak": slow,
            "fast_streak": fast,
            "cooldown": cooldown,
            "last_step": last_step if stepped else (last_step if cooldown > 0 else None),
            "last_fit": fit_grade(new_active, hardware, new_ctx) if new_active else fit,
            "last_speed": speed,
            "last_tps": round(decode_tps_val, 2) if decode_tps_val is not None else None,
            "mood": _mood(new_active, stepped, speed, fit, new_ctx),
            "history": history,
            "rung_stats": state.get("rung_stats") or {},
            "good_on_rung": good_on_rung,
            "probe_wait": probe_wait,
            "stepped": stepped,
            "demote_reason": demote_reason,
            "last_prompt_eval_s": round(float(prompt_eval_s or 0), 2),
        }
    )
    return out


def snapshot_for_heartbeat(state: dict[str, Any], hardware: dict | None = None) -> dict[str, Any]:
    family = resolve_family(state.get("family") or "granite")
    pulled = set(state.get("pulled") or [])
    active = state.get("active_tag")
    try:
        active_ctx = int(state.get("active_num_ctx") or 0)
    except (TypeError, ValueError):
        active_ctx = 0
    rungs = []
    skipped_tags: list[str] = []
    seen_skip: set[str] = set()
    ladder_tags: list[str] = []
    seen_ladder: set[str] = set()
    for tag, _n in family_rungs(family):
        if tag not in seen_ladder:
            seen_ladder.add(tag)
            ladder_tags.append(tag)
    fitting = set(tags_fitting_vram(ladder_tags, hardware))
    for tag, nctx in family_rungs(family):
        fit = fit_grade(tag, hardware, nctx) if hardware is not None else None
        rs = (state.get("rung_stats") or {}).get(_rung_key(tag, nctx)) or {}
        skip_reason = None
        if tag not in fitting:
            cpu = bool(hardware) and _cpu_only(hardware)
            skip_reason = "Too big for this RAM" if cpu else "Too big even as GPU + CPU split"
            if tag not in seen_skip:
                seen_skip.add(tag)
                skipped_tags.append(tag)
        locked = tag not in pulled or (fit is not None and fit <= 0 and not (tag == active and nctx == active_ctx))
        rungs.append(
            {
                "tag": tag,
                "num_ctx": nctx,
                "variant": f"{short_variant(tag)} - {short_ctx_label(nctx)}".rstrip(" -"),
                "now": tag == active and int(nctx or 0) == int(active_ctx or 0),
                "pulled": tag in pulled,
                "locked": bool(locked),
                "fit": fit,
                "skip_reason": skip_reason,
                "last_speed": rs.get("last_speed"),
                "last_tps": rs.get("last_tps"),
                "jobs": int(rs.get("jobs") or 0),
            }
        )
    return {
        "family": family,
        "active_tag": active,
        "active_num_ctx": active_ctx or None,
        "fit": state.get("last_fit"),
        "speed": state.get("last_speed"),
        "tps": state.get("last_tps"),
        "mood": state.get("mood"),
        "history": list(state.get("history") or []),
        "rungs": rungs,
        "skipped": skip_rows_for_tags(skipped_tags, hardware),
        "num_ctx": state.get("last_num_ctx") or active_ctx,
        "parallel": state.get("num_parallel"),
        "last_hybrid": state.get("last_hybrid"),
        "last_load_mode": state.get("last_load_mode"),
        "last_prompt_eval_s": state.get("last_prompt_eval_s"),
    }


def is_context_overflow(status: int | None, body: str | None) -> bool:
    """True for Ollama exceed_context_size (including nested JSON)."""
    del status
    t = (body or "").lower()
    if not t:
        return False
    bits = (
        "exceed_context_size",
        "exceeds the available context size",
        "exceeds context",
        "prompt is too long",
        "prompt too long",
        "prompt_too_long",
        "n_prompt_tokens",
    )
    if any(b in t for b in bits):
        return True
    return "context size" in t and "exceed" in t


def is_load_error(status: int | None, body: str | None) -> bool:
    """True for VRAM/weight load failures. Context overflow is not a load error."""
    if is_context_overflow(status, body):
        return False
    t = (body or "").lower()
    oom_bits = (
        "out of memory",
        "oom",
        "unable to load",
        "cannot allocate",
        "cuda malloc",
        "not enough memory",
    )
    if status in (500, 502, 503) and any(s in t for s in oom_bits):
        return True
    if any(s in t for s in ("out of memory", "cuda malloc")):
        return True
    return False
