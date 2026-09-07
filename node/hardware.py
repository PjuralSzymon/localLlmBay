"""Host hardware probe for contribution nodes (GPU / CPU / RAM)."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")

_DRM_VENDORS = {
    "0x10de": "NVIDIA",
    "0x1002": "AMD",
    "0x1022": "AMD",
    "0x8086": "Intel",
}

_SKIP_VK_NAMES = ("llvmpipe", "swiftshader", "lavapipe")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_cores() -> int | None:
    try:
        return os.cpu_count() or None
    except Exception:
        return None


def _ram_gb() -> float | None:
    # Linux
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except Exception:
        pass
    # Windows (rare inside official Linux node image)
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / (1024**3), 1)
    except Exception:
        pass
    return None


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Parse nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits."""
    gpus: list[dict] = []
    for line in (text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            vram = int(float(parts[2]))
        except ValueError:
            continue
        gpus.append({"index": idx, "name": parts[1], "vram_mb": vram, "source": "nvidia"})
    return gpus


def parse_vulkaninfo_summary(text: str) -> list[dict]:
    """Pull deviceName rows out of `vulkaninfo --summary` (skip software rasterizers)."""
    gpus: list[dict] = []
    for i, m in enumerate(re.finditer(r"deviceName\s*=\s*(.+)", text or "", re.I)):
        name = m.group(1).strip()
        if not name or any(s in name.lower() for s in _SKIP_VK_NAMES):
            continue
        gpus.append({"index": i, "name": name, "vram_mb": 0, "source": "vulkan"})
    return gpus


def classify_backend(gpus: list[dict]) -> str:
    sources = {str(g.get("source") or "") for g in gpus}
    blob = " ".join(str(g.get("name") or "") for g in gpus).lower()
    if "nvidia" in sources or "nvidia" in blob:
        return "cuda"
    if "vulkan" in sources or "drm" in sources:
        return "vulkan"
    return "cpu"


def merge_gpu_lists(*groups: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for g in group:
            key = (g.get("name") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(g)
    return out


def _nvidia_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
    except Exception:
        return []
    return parse_nvidia_smi_csv(out)


def _vulkan_gpus() -> list[dict]:
    if not shutil.which("vulkaninfo"):
        return []
    try:
        out = subprocess.check_output(
            ["vulkaninfo", "--summary"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=12,
        )
    except Exception:
        return []
    return parse_vulkaninfo_summary(out)


def drm_gpus(drm_root: Path | None = None) -> list[dict]:
    """PCI GPUs from /sys/class/drm/cardN (skip connectors like card0-DP-1)."""
    root = drm_root or Path("/sys/class/drm")
    if not root.is_dir():
        return []
    gpus: list[dict] = []
    idx = 0
    for card in sorted(root.iterdir()):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        vendor_path = card / "device" / "vendor"
        if not vendor_path.is_file():
            continue
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        brand = _DRM_VENDORS.get(vendor)
        if not brand:
            continue
        vram_mb = 0
        mem_path = card / "device" / "mem_info_vram_total"
        if mem_path.is_file():
            try:
                vram_mb = int(mem_path.read_text(encoding="utf-8").strip()) // (1024 * 1024)
            except (OSError, ValueError):
                vram_mb = 0
        gpus.append(
            {
                "index": idx,
                "name": f"{brand} ({card.name})",
                "vram_mb": vram_mb,
                "source": "drm",
            }
        )
        idx += 1
    return gpus


def _ollama_version() -> str | None:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if r.status_code == 200:
            return str((r.json() or {}).get("version") or "") or None
    except Exception:
        pass
    return None


def probe_hardware() -> dict:
    """Return a compact hardware snapshot for heartbeat → project fleet summary."""
    gpus = merge_gpu_lists(_nvidia_gpus(), _vulkan_gpus(), drm_gpus())
    cpu = _cpu_cores()
    ram = _ram_gb()
    vram_total = sum(int(g.get("vram_mb") or 0) for g in gpus)
    # Rough index: VRAM MB dominates; CPU-only nodes still get a small score
    compute_index = vram_total + int((cpu or 0) * 100)
    backend = classify_backend(gpus)
    return {
        "collected_at": _utcnow_iso(),
        "hostname": platform.node() or None,
        "os": f"{platform.system()} {platform.release()}".strip(),
        "cpu_cores": cpu,
        "ram_gb": ram,
        "gpus": gpus,
        "gpu_count": len(gpus),
        "vram_mb_total": vram_total,
        "compute_index": compute_index,
        "backend": backend,
        "ollama_version": _ollama_version(),
    }


def summarize_label(hw: dict | None) -> str:
    if not hw:
        return "unknown"
    gpus = hw.get("gpus") or []
    if gpus:
        names = [g.get("name") or "GPU" for g in gpus]
        return ", ".join(names)
    cores = hw.get("cpu_cores")
    return f"CPU×{cores}" if cores else "CPU"
