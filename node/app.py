"""LocalLLMBay contribution node — official image worker."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify

from tool_parse import (
    file_tool_args_ok as _file_tool_args_ok,
    is_invalid_tool_json_error,
    normalize_file_tool_args as _normalize_file_tool_args,
    promote_content_tool_calls,
)

from adapt import (
    FAMILY_DEFAULTS,
    FAMILY_LADDERS,
    GRAPH_FAMILY,
    GRAPH_ORCH_TAG,
    GRAPH_PACK,
    GRAPH_ROLE_TAGS,
    GRAPHICAL_FAMILIES,
    VIDEO_FAMILIES,
    CTX_GRAPH,
    MODEL_META,
    apply_adapt,
    boot_reconcile,
    decode_tps,
    empty_state,
    family_rungs,
    fit_grade,
    is_context_overflow,
    is_image_family,
    is_video_family,
    is_load_error,
    is_think_hop_job,
    pick_initial_rung,
    resolve_family,
    snapshot_for_heartbeat,
    tags_fitting_vram,
)
from ollama_image import image_content_from_b64, video_content_from_b64
import flux_hf
import wan_hf
from ollama_ctx import (
    INVALID_TOOL_RETRY_PREDICT,
    INVALID_TOOL_RETRY_TIMEOUT,
    clamp_num_predict,
    estimate_load_mode,
    parse_model_context_length,
    resolve_num_ctx,
    resolve_num_predict,
    resolve_temperature,
)
from debug_log import (
    format_messages_transcript,
    last_user_log_field,
    last_user_text,
    plaintext_logs_enabled,
    scrub_secrets,
    summarize_assistant_output,
    to_log_json,
)
from ollama_messages import (
    finish_from_ollama,
    format_messages_wire,
    messages_for_ollama,
    ollama_tool_calls_to_openai,
    visible_content,
    wire_loss_report,
)
from tool_budget import (
    MAX_CTX_RETRIES,
    UNFIT_RUNG,
    next_body_trim_rung,
    next_ctx_rung,
    ollama_think_flag,
    prepare_ollama_payload,
    tool_name,
)
from wire_crypto import WireCryptoError, decode_payload, encode_payload

try:
    from hardware import probe_hardware, summarize_label
except ImportError as _hw_err:
    probe_hardware = None  # type: ignore
    summarize_label = None  # type: ignore
    _HARDWARE_IMPORT_ERROR = str(_hw_err)
else:
    _HARDWARE_IMPORT_ERROR = None

app = Flask(__name__)

API_BASE = os.getenv("API_BASE", "http://localhost:7000").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT = int(os.getenv("NODE_OLLAMA_TIMEOUT_SECONDS", "900"))
NODE_NAME = os.getenv("NODE_NAME", "unnamed-node")
NODE_MODELS = [m.strip() for m in os.getenv("NODE_MODELS", ",".join(GRAPH_PACK)).split(",") if m.strip()]
_raw_fams = [p.strip().lower() for p in (os.getenv("NODE_FAMILY") or "").split(",") if p.strip()]
NODE_FAMILIES: list[str] = []
for _raw in _raw_fams:
    _fam = resolve_family(_raw)
    if _fam in FAMILY_LADDERS and _fam not in NODE_FAMILIES:
        NODE_FAMILIES.append(_fam)
if not NODE_FAMILIES:
    first = NODE_MODELS[0] if NODE_MODELS else ""
    inferred = "granite"
    if first in FAMILY_LADDERS:
        inferred = first
    else:
        for fam, tags in FAMILY_LADDERS.items():
            if first in tags:
                inferred = fam
                break
    NODE_FAMILIES = [inferred]
NODE_FAMILY = NODE_FAMILIES[0]
if not NODE_MODELS:
    NODE_MODELS = []
    for fam in NODE_FAMILIES:
        NODE_MODELS.extend(list(FAMILY_LADDERS.get(fam) or []))
    if not NODE_MODELS:
        NODE_MODELS = list(GRAPH_PACK)
else:
    _wanted = set(NODE_MODELS)
    _ordered: list[str] = []
    _seen: set[str] = set()
    for fam in NODE_FAMILIES:
        for t in FAMILY_LADDERS.get(fam) or []:
            if t in _wanted and t not in _seen:
                _ordered.append(t)
                _seen.add(t)
    for t in NODE_MODELS:
        if t not in _seen:
            _ordered.append(t)
            _seen.add(t)
    NODE_MODELS = _ordered
NODE_TARGET_TPS = float(os.getenv("NODE_TARGET_TPS", "8"))
NODE_WALL_SLOW_SECONDS = float(os.getenv("NODE_WALL_SLOW_SECONDS", "90"))
PROJECT_ID = (os.getenv("PROJECT_ID") or "").strip() or None
PLAINTEXT_LOGS = plaintext_logs_enabled(API_BASE, os.getenv("QUEUE_PLAINTEXT_LOGS"))
_MODEL_CTX_CACHE: dict[str, int | None] = {}

SENDED_QUESTIONS = 0
ANSWERED_QUESTIONS = 0
POLL_COUNT = 0
START_TIME = datetime.now(timezone.utc)
LAST_ERROR = None
PHASE = "starting"  # starting|warming|idle|polling|processing|error
CURRENT_JOB_ID = None
ACTIVITY_DETAIL = "booting"
LAST_POLL_AT = None
LAST_EMPTY_POLL_AT = None
HARDWARE = None
HARDWARE_REFRESH_EVERY = 20  # heartbeats (~1–2 min)
_HEARTBEATS = 0
ADAPT_STATE = None
ADAPT_STATE_BY_FAMILY: dict = {}
_WARMED: tuple[str, int] | None = None

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

_PATHISH = re.compile(
    r"(?i)(?:[a-z]:\\[^\s\"']+|\\\\[^\s\"']+|/(?:home|Users|users|root)/[^\s\"']+|https?://[^\s\"']+)"
)


def _tools_canonical_json(tools: list | None) -> str:
    if not tools:
        return "[]"
    return json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tools_sha256(tools: list | None) -> str:
    return hashlib.sha256(_tools_canonical_json(tools).encode("utf-8")).hexdigest()[:16]


def _redact_tool_tree(obj):
    if isinstance(obj, str):
        return _PATHISH.sub("<redacted>", obj)
    if isinstance(obj, list):
        return [_redact_tool_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _redact_tool_tree(v) for k, v in obj.items()}
    return obj


def _tool_names_list(tools: list | None) -> str:
    names = []
    for t in tools or []:
        if not isinstance(t, dict):
            names.append("?")
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        names.append(str((fn or {}).get("name") or "?"))
    return ",".join(names)


def _adapt_path() -> Path:
    env = (os.getenv("NODE_ADAPT_PATH") or "").strip()
    if env:
        return Path(env)
    for p in (Path("/root/.ollama"), Path("/app/models"), Path("ollama_data"), Path(".")):
        try:
            if p.exists() and p.is_dir():
                return p / "adapt.json"
        except OSError:
            continue
    return Path("adapt.json")


def _graph_hop_tag(job: dict | None) -> str | None:
    """Sealed graph hop tag. Do not fall back to the adapt ladder."""
    if not isinstance(job, dict):
        return None
    tag = str(job.get("ollama_tag") or "").strip()
    if tag:
        return tag
    role = str(job.get("graph_role") or "").strip()
    if role:
        return GRAPH_ROLE_TAGS.get(role)
    return None


def advertised_families() -> list[str]:
    primary = NODE_FAMILY if NODE_FAMILY in FAMILY_LADDERS else "granite"
    out = [primary]
    for fam in NODE_FAMILIES:
        if fam in FAMILY_LADDERS and fam not in out:
            out.append(fam)
    return out


def pulled_for_family(family: str, hardware: dict | None) -> list[str]:
    ladder = list(FAMILY_LADDERS.get(family) or [])
    wanted = [t for t in NODE_MODELS if t in set(ladder)]
    if not wanted:
        wanted = list(ladder)
    return tags_fitting_vram(wanted, hardware)


def _hydrate_family_state(family: str, raw: dict | None, hardware: dict | None) -> dict:
    pulled = pulled_for_family(family, hardware)
    state = raw if isinstance(raw, dict) and raw.get("family") == family else None
    if not state:
        tag, nctx = pick_initial_rung(family, pulled, hardware)
        active = tag or (pulled[0] if pulled else None)
        return empty_state(family, active, pulled, active_num_ctx=nctx)
    state["pulled"] = pulled
    active = state.get("active_tag")
    if not pulled:
        state["active_tag"] = None
        state["active_num_ctx"] = None
        state["mood"] = "No model fits this GPU"
    elif active not in pulled:
        tag, nctx = pick_initial_rung(family, pulled, hardware)
        state["active_tag"] = tag or pulled[0]
        if nctx:
            state["active_num_ctx"] = nctx
    prev = state.get("active_tag")
    prev_ctx = state.get("active_num_ctx")
    state = boot_reconcile(state, hardware, pulled)
    if state.get("rebased"):
        _log(
            "worker",
            f"QUEUE.ADAPT_REBASE family={family} from={prev}@{prev_ctx} "
            f"to={state.get('active_tag')}@{state.get('active_num_ctx')} reason=heaviest_fit",
        )
    if state.get("active_tag"):
        state["last_fit"] = fit_grade(
            state["active_tag"], hardware, state.get("active_num_ctx")
        ) if hardware else state.get("last_fit")
    return state


def load_adapt_state(hardware: dict | None) -> dict:
    global ADAPT_STATE_BY_FAMILY
    path = _adapt_path()
    raw = None
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raw = None
    stored: dict = {}
    if isinstance(raw, dict) and int(raw.get("version") or 0) >= 2 and isinstance(raw.get("families"), dict):
        stored = {k: v for k, v in raw["families"].items() if isinstance(v, dict)}
    elif isinstance(raw, dict) and raw.get("family"):
        stored[str(raw.get("family"))] = raw
    by_family: dict = {}
    for fam in advertised_families():
        by_family[fam] = _hydrate_family_state(fam, stored.get(fam), hardware)
    ADAPT_STATE_BY_FAMILY = by_family
    return by_family.get(NODE_FAMILY) or next(iter(by_family.values()))


def save_adapt_state(state: dict | None = None) -> None:
    path = _adapt_path()
    blob = {"version": 2, "families": ADAPT_STATE_BY_FAMILY or {}}
    if state and not ADAPT_STATE_BY_FAMILY:
        blob = state
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        _log("worker", f"WARNING adapt persist failed: {e}")


def active_tag() -> str:
    if ADAPT_STATE and ADAPT_STATE.get("active_tag"):
        return str(ADAPT_STATE["active_tag"])
    pulled = pulled_for_family(NODE_FAMILY, HARDWARE)
    tag, _n = pick_initial_rung(NODE_FAMILY, pulled, HARDWARE)
    if tag:
        return tag
    if pulled:
        return pulled[0]
    return FAMILY_DEFAULTS.get(NODE_FAMILY) or "granite4.2:8b"


def active_num_ctx_for(family: str | None = None) -> int | None:
    fam = resolve_family(family) if family else NODE_FAMILY
    st = ADAPT_STATE_BY_FAMILY.get(fam) if ADAPT_STATE_BY_FAMILY else None
    if fam == NODE_FAMILY and not isinstance(st, dict):
        st = ADAPT_STATE
    if not isinstance(st, dict):
        return None
    try:
        n = int(st.get("active_num_ctx") or 0)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def active_tag_for(family: str | None) -> str:
    fam = resolve_family(family) if family else NODE_FAMILY
    st = ADAPT_STATE_BY_FAMILY.get(fam) if ADAPT_STATE_BY_FAMILY else None
    if isinstance(st, dict) and st.get("active_tag"):
        return str(st["active_tag"])
    pulled = pulled_for_family(fam, HARDWARE)
    tag, _n = pick_initial_rung(fam, pulled, HARDWARE)
    if tag:
        return tag
    if pulled:
        return pulled[0]
    if fam == NODE_FAMILY:
        return active_tag()
    return FAMILY_DEFAULTS.get(fam) or active_tag()


def _adapt_snapshot_payload() -> dict:
    primary = ADAPT_STATE or empty_state(NODE_FAMILY, active_tag(), NODE_MODELS)
    snap = snapshot_for_heartbeat(primary, HARDWARE)
    families = {}
    for fam, st in (ADAPT_STATE_BY_FAMILY or {}).items():
        if isinstance(st, dict):
            families[fam] = snapshot_for_heartbeat(st, HARDWARE)
    if families:
        snap["families"] = families
    return snap


def _log(module: str, msg: str):
    """Print a line tagged [node][<module>] for Logdy filtering."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [node][{module}] [{NODE_NAME}] {scrub_secrets(msg)}", flush=True)


def _dump(kind: str, msg: str):
    if not PLAINTEXT_LOGS:
        return
    _log("debug", f"QUEUE.PLAINTEXT.{kind} {msg}")


def _dump_block(kind: str, body: str, *, extra: str = ""):
    if not PLAINTEXT_LOGS:
        return
    tail = f" {extra}" if extra else ""
    _log("debug", f"QUEUE.PLAINTEXT.{kind} ===== begin{tail} =====")
    print(scrub_secrets(body).rstrip() + "\n", flush=True)
    _log("debug", f"QUEUE.PLAINTEXT.{kind} ===== end =====")


def _dump_json(kind: str, obj, *, extra: str = ""):
    _dump_block(kind, to_log_json(obj), extra=extra)


def _set_phase(phase: str, detail: str = "", job_id=None):
    global PHASE, ACTIVITY_DETAIL, CURRENT_JOB_ID
    PHASE = phase
    ACTIVITY_DETAIL = detail or phase
    if job_id is not None:
        CURRENT_JOB_ID = job_id
    elif phase != "processing":
        CURRENT_JOB_ID = None


def heartbeat(extra: dict | None = None):
    global LAST_ERROR, HARDWARE, _HEARTBEATS, ADAPT_STATE
    _HEARTBEATS += 1
    try:
        if probe_hardware:
            if HARDWARE is None:
                HARDWARE = probe_hardware()
                _log(
                    "worker",
                    f"hardware {summarize_label(HARDWARE)} compute_index={HARDWARE.get('compute_index')}",
                )
            elif _HEARTBEATS > 1 and _HEARTBEATS % HARDWARE_REFRESH_EVERY == 0:
                prev = summarize_label(HARDWARE)
                HARDWARE = probe_hardware()
                label = summarize_label(HARDWARE)
                if label != prev:
                    _log(
                        "worker",
                        f"hardware {label} compute_index={HARDWARE.get('compute_index')}",
                    )
        if ADAPT_STATE is None:
            ADAPT_STATE = load_adapt_state(HARDWARE)
        snap = _adapt_snapshot_payload()
        body = {
            "node_name": NODE_NAME,
            "models": advertised_families(),
            "active_tag": ADAPT_STATE.get("active_tag") if ADAPT_STATE else None,
            "adapt": snap,
            "phase": PHASE,
            "activity_detail": ACTIVITY_DETAIL,
            "polls_total": POLL_COUNT,
            "current_job_id": CURRENT_JOB_ID,
            "hardware": HARDWARE,
        }
        if PROJECT_ID:
            body["project_id"] = PROJECT_ID
        if LAST_POLL_AT:
            body["last_poll_at"] = LAST_POLL_AT
        if extra:
            body.update(extra)
        r = requests.post(
            f"{API_BASE}/nodes/heartbeat",
            headers=headers,
            json=body,
            timeout=30,
        )
        if r.status_code >= 400:
            LAST_ERROR = f"heartbeat {r.status_code}: {r.text[:200]}"
            _log("worker", f"ERROR heartbeat: {LAST_ERROR}")
            _set_phase("error", LAST_ERROR)
        else:
            data = r.json()
            # Idle ok every few seconds is noise. Keep errors, first beat, phase changes, and a heartbeat.
            noisy = PHASE in ("processing", "error", "warming", "starting")
            if noisy or _HEARTBEATS <= 1 or _HEARTBEATS % 20 == 0:
                _log(
                    "worker",
                    f"heartbeat ok node_id={data.get('node_id')} "
                    f"scope={'project:' + PROJECT_ID if PROJECT_ID else 'public'} "
                    f"phase={PHASE} beat={_HEARTBEATS}",
                )
    except Exception as e:
        LAST_ERROR = str(e)
        _log("worker", f"ERROR heartbeat: {e}")
        _set_phase("error", str(e))


def _dump_assistant_turn(job_id, *, finish, content, tool_calls, data=None, empty: bool = False) -> None:
    thinking = ""
    if isinstance(data, dict):
        msg = data.get("message") if isinstance(data.get("message"), dict) else {}
        thinking = str(msg.get("thinking") or data.get("thinking") or "")
    _dump(
        "CHAT_RESULT_META",
        f"job={job_id} finish={finish} empty={int(empty)} "
        f"content_chars={len(content or '')} thinking_chars={len(thinking)} "
        f"tool_calls={len(tool_calls) if tool_calls else 0}",
    )
    if content:
        _dump_block("ASSISTANT_CONTENT", content, extra=f"job={job_id} chars={len(content)}")
    if thinking:
        _dump_block("ASSISTANT_THINKING", thinking, extra=f"job={job_id} chars={len(thinking)}")
    if tool_calls:
        _dump_json("ASSISTANT_TOOL_CALLS", tool_calls, extra=f"job={job_id}")


def _compose_prompt(payload: dict) -> str:
    parts = []
    if payload.get("compact_summary"):
        parts.append("Summary of earlier context:\n" + str(payload["compact_summary"]))
    if payload.get("history"):
        parts.append("Recent history:\n" + str(payload["history"]))
    parts.append("Question:\n" + (payload.get("question") or payload.get("prompt") or ""))
    return "\n\n".join(parts)


def _tool_names_from_tools(tools: list | None) -> set[str]:
    names: set[str] = set()
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        if name:
            names.add(str(name))
    return names


def _model_max_ctx(model: str) -> int | None:
    if model in _MODEL_CTX_CACHE:
        return _MODEL_CTX_CACHE[model]
    n: int | None = None
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/show", json={"name": model}, timeout=10)
        if r.status_code == 200:
            n = parse_model_context_length(r.json())
    except Exception:
        n = None
    _MODEL_CTX_CACHE[model] = n
    return n


def _hw_bits() -> tuple[float, bool]:
    hw = HARDWARE or {}
    try:
        vram = float(hw.get("vram_mb_total") or 0)
    except (TypeError, ValueError):
        vram = 0.0
    backend = str(hw.get("backend") or "cpu").lower()
    try:
        gpus = int(hw.get("gpu_count") or 0)
    except (TypeError, ValueError):
        gpus = 0
    cpu_only = gpus <= 0 or backend == "cpu"
    if gpus > 0 and backend not in ("cpu",):
        cpu_only = False
    return vram, cpu_only


def _ollama_keep_alive() -> str:
    """Stay resident so llama.cpp warmup is a boot cost, not every job."""
    return (os.getenv("OLLAMA_KEEP_ALIVE") or "30m").strip() or "30m"


def _warmup_ollama_tag(tag: str | None, num_ctx: int | None = None) -> bool:
    """Load the runner (weights + empty llama.cpp warmup) once per tag/ctx."""
    global _WARMED
    tag = (tag or "").strip()
    if not tag:
        return False
    infer = {"num_predict": 1, "temperature": 0, "think": False}
    if num_ctx:
        infer["num_ctx"] = int(num_ctx)
    options = _ollama_options(tag, infer, force_num_ctx=num_ctx)
    try:
        nctx = int(options.get("num_ctx") or 0)
    except (TypeError, ValueError):
        nctx = 0
    key = (tag, nctx)
    if _WARMED == key:
        return True
    keep = _ollama_keep_alive()
    payload = {
        "model": tag,
        "prompt": ".",
        "stream": False,
        "keep_alive": keep,
        "think": False,
        "options": options,
    }
    _set_phase("warming", f"loading {tag}")
    _log(
        "ollama",
        f"QUEUE.MODEL_WARMUP start tag={tag} num_ctx={nctx} keep_alive={keep}",
    )
    t0 = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT
        )
        elapsed = time.time() - t0
        if int(getattr(response, "status_code", 0) or 0) != 200:
            body = (getattr(response, "text", None) or "")[:200].replace("\n", " ")
            _log(
                "ollama",
                f"QUEUE.MODEL_WARMUP fail tag={tag} status={response.status_code} "
                f"in={elapsed:.1f}s body={body}",
            )
            return False
        _WARMED = key
        data = {}
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            data = {}
        pe = int(data.get("prompt_eval_count") or 0)
        _log(
            "ollama",
            f"QUEUE.MODEL_WARMUP done tag={tag} num_ctx={nctx} in={elapsed:.1f}s prompt_eval={pe}",
        )
        return True
    except Exception as e:
        _log("ollama", f"QUEUE.MODEL_WARMUP fail tag={tag} error={e}")
        return False


def _warmup_walk_family(fam: str) -> None:
    """Try Fit>0 rungs from heaviest until generate 200. Skip tags that cannot load even as GPU+CPU."""
    global ADAPT_STATE, ADAPT_STATE_BY_FAMILY
    hw = HARDWARE
    pulled = pulled_for_family(fam, hw)
    ladder = list(FAMILY_LADDERS.get(fam) or [])
    wanted = [t for t in NODE_MODELS if t in set(ladder)] or ladder
    skipped = [t for t in wanted if t not in set(pulled)]
    if skipped:
        _log("worker", f"QUEUE.SKIP_FIT family={fam} tags={','.join(skipped)}")
    if fam == GRAPH_FAMILY and GRAPH_ORCH_TAG in set(pulled):
        st = (ADAPT_STATE_BY_FAMILY or {}).get(fam)
        nctx = int(CTX_GRAPH[0])
        if not isinstance(st, dict):
            st = empty_state(fam, GRAPH_ORCH_TAG, pulled, active_num_ctx=nctx)
        st["active_tag"] = GRAPH_ORCH_TAG
        st["active_num_ctx"] = nctx
        if _warmup_ollama_tag(GRAPH_ORCH_TAG, nctx):
            ADAPT_STATE_BY_FAMILY[fam] = st
            if fam == NODE_FAMILY:
                ADAPT_STATE = st
            save_adapt_state(st)
            return
        _log("worker", f"QUEUE.WARMUP_DEMOTE family={fam} from={GRAPH_ORCH_TAG}@{nctx}")
    candidates = [
        (t, n)
        for t, n in family_rungs(fam)
        if t in set(pulled) and (n <= 0 or fit_grade(t, hw, n) > 0)
    ]
    start_tag, start_ctx = pick_initial_rung(fam, pulled, hw)
    idx = 0
    if start_tag and (start_tag, start_ctx) in candidates:
        idx = candidates.index((start_tag, start_ctx))
    st = (ADAPT_STATE_BY_FAMILY or {}).get(fam)
    if not isinstance(st, dict):
        st = empty_state(fam, start_tag or (pulled[0] if pulled else None), pulled, active_num_ctx=start_ctx)
    for tag, nctx in candidates[idx:]:
        st["active_tag"] = tag
        st["active_num_ctx"] = nctx
        if _warmup_ollama_tag(tag, nctx):
            ADAPT_STATE_BY_FAMILY[fam] = st
            if fam == NODE_FAMILY:
                ADAPT_STATE = st
            save_adapt_state(st)
            return
        _log("worker", f"QUEUE.WARMUP_DEMOTE family={fam} from={tag}@{nctx}")
    ADAPT_STATE_BY_FAMILY[fam] = st
    if fam == NODE_FAMILY:
        ADAPT_STATE = st
    save_adapt_state(st)


def _warmup_active_model() -> None:
    for fam in advertised_families():
        if fam in GRAPHICAL_FAMILIES or fam in VIDEO_FAMILIES:
            continue
        _warmup_walk_family(fam)


def _family_for_model(model: str) -> str:
    if NODE_FAMILY == GRAPH_FAMILY and model in GRAPH_PACK:
        return GRAPH_FAMILY
    for fam, tags in FAMILY_LADDERS.items():
        if model in tags:
            return fam
    return NODE_FAMILY


def _infer_chat_format(infer: dict | None) -> str | None:
    """Forward sealed infer.format to Ollama /api/chat (graph hops use json)."""
    infer = infer if isinstance(infer, dict) else {}
    raw = infer.get("format")
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    return s or None


def _ollama_options(
    model: str,
    infer: dict | None = None,
    tools: list | None = None,
    force_num_ctx: int | None = None,
) -> dict:
    infer = infer if isinstance(infer, dict) else {}
    vram, cpu_only = _hw_bits()
    fam = _family_for_model(model) or NODE_FAMILY
    job_n = infer.get("num_ctx")
    graph_tag = model in GRAPH_PACK or fam == GRAPH_FAMILY
    if force_num_ctx:
        n = int(force_num_ctx)
    elif graph_tag and job_n is not None:
        try:
            n = int(job_n)
        except (TypeError, ValueError):
            n = int(CTX_GRAPH[0])
        from ollama_ctx import MIN_NUM_CTX, PUBLIC_MAX_CTX

        n = max(MIN_NUM_CTX, min(n, PUBLIC_MAX_CTX))
    else:
        n = resolve_num_ctx(
            env_val=os.getenv("NODE_NUM_CTX") or os.getenv("OLLAMA_CONTEXT_LENGTH"),
            model_max=_model_max_ctx(model),
            vram_mb=vram,
            cpu_only=cpu_only,
            job_num_ctx=job_n,
            family=fam,
            job_ctx_wins=graph_tag,
        )
        cap = active_num_ctx_for(fam)
        if cap and n > cap and not graph_tag:
            _log("ollama", f"QUEUE.CTX_CLAMP model={model} from={n} to={cap}")
            n = cap
    opts: dict = {
        "num_ctx": n,
        "num_predict": resolve_num_predict(infer.get("num_predict"), tools=tools),
        "temperature": resolve_temperature(infer.get("temperature")),
    }
    for key in ("top_p", "min_p", "repeat_penalty"):
        if key not in infer:
            continue
        try:
            opts[key] = float(infer[key])
        except (TypeError, ValueError):
            continue
    return opts


def _load_mode(model: str, num_ctx: int) -> str:
    vram, cpu_only = _hw_bits()
    meta = MODEL_META.get(model) or {}
    try:
        min_v = float(meta.get("min_vram_gb") or 0)
    except (TypeError, ValueError):
        min_v = 0.0
    return estimate_load_mode(
        vram_mb=vram,
        cpu_only=cpu_only,
        num_ctx=num_ctx,
        tag=model,
        min_vram_gb=min_v,
    )


def _timings_from_ollama(data: dict | None, *, status: int | None = None, err_body: str | None = None) -> dict:
    data = data or {}
    err = err_body if err_body is not None else str(data.get("error") or "")
    overflow = is_context_overflow(status, err) or is_context_overflow(status, str(data.get("error") or ""))
    oom = (not overflow) and (
        is_load_error(status, err) or is_load_error(status, str(data.get("error") or ""))
    )
    return {
        "eval_count": int(data.get("eval_count") or 0),
        "eval_duration": int(data.get("eval_duration") or 0),
        "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
        "prompt_eval_duration": int(data.get("prompt_eval_duration") or 0),
        "oom": oom,
        "ctx_overflow": overflow,
        "empty_completion": False,
        "ok": not oom and not overflow and status in (None, 200),
    }


def _stamp_infer_timings(timings: dict, options: dict, model: str) -> dict:
    out = dict(timings or {})
    out["num_ctx"] = options.get("num_ctx")
    out["num_predict"] = options.get("num_predict")
    try:
        nctx = int(options.get("num_ctx") or 0)
    except (TypeError, ValueError):
        nctx = 0
    out["load_mode"] = _load_mode(model, nctx)
    return out


def _opt(options: dict, key: str) -> str:
    val = options.get(key)
    return "-" if val is None else str(val)


def _log_inference_start(job_id, via: str, model: str, options: dict, extra: str = "") -> None:
    try:
        nctx = int(options.get("num_ctx") or 0)
    except (TypeError, ValueError):
        nctx = 0
    mode = _load_mode(model, nctx)
    fit = fit_grade(model, HARDWARE)
    _log(
        "ollama",
        f"QUEUE.INFERENCE_START job={job_id} via={via} model={model} "
        f"num_ctx={options.get('num_ctx')} num_predict={options.get('num_predict') if options.get('num_predict') is not None else '-'} "
        f"rung={model} fit={fit} load={mode} full_gpu={int(mode == 'full_gpu')} hybrid={int(mode == 'hybrid')}"
        + (f" {extra}" if extra else ""),
    )
    _log(
        "job",
        f"QUEUE.JOB_INFER job={job_id} phase=inferring via={via} tag={model} "
        f"num_ctx={_opt(options, 'num_ctx')} num_predict={_opt(options, 'num_predict')} "
        f"temp={_opt(options, 'temperature')} top_p={_opt(options, 'top_p')} "
        f"min_p={_opt(options, 'min_p')} repeat_penalty={_opt(options, 'repeat_penalty')} "
        f"load={mode}"
        + (f" {extra}" if extra else ""),
    )


def ask_ollama_generate(prompt: str, model: str, job_id=None, infer: dict | None = None) -> tuple[str, int, int, dict]:
    """Legacy /api/generate path. Returns answer, prompt_tokens, completion_tokens, timings."""
    options = _ollama_options(model, infer)
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": _ollama_keep_alive(), "options": options}
    try:
        _log_inference_start(job_id, "generate", model, options, extra=f"prompt_chars={len(prompt)}")
        _dump("GENERATE_META", f"job={job_id} model={model} chars={len(prompt)} options={to_log_json(options)}")
        _dump_block("GENERATE_PROMPT", prompt, extra=f"job={job_id} chars={len(prompt)}")
        response = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        if response.status_code != 200:
            err_body = (response.text or "")[:400].replace("\n", " ")
            _log(
                "ollama",
                f"QUEUE.INFERENCE_ERR job={job_id} status={response.status_code} body={err_body}",
            )
            return (
                f"Error from Ollama: {response.status_code}",
                max(1, len(prompt) // 4),
                0,
                _stamp_infer_timings(
                    _timings_from_ollama(None, status=response.status_code, err_body=err_body),
                    options,
                    model,
                ),
            )
        data = response.json()
        answer = data.get("response", "")
        pt = int(data.get("prompt_eval_count") or max(1, len(prompt) // 4))
        ct = int(data.get("eval_count") or max(1, len(answer) // 4))
        thinking = str(data.get("thinking") or "")
        _dump_block("GENERATE_ANSWER", answer or "(empty)", extra=f"job={job_id} chars={len(answer or '')}")
        if thinking:
            _dump_block("GENERATE_THINKING", thinking, extra=f"job={job_id} chars={len(thinking)}")
        return answer, pt, ct, _stamp_infer_timings(_timings_from_ollama(data, status=200), options, model)
    except Exception as e:
        _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} error={e}")
        return (
            f"Error communicating with Ollama: {e}",
            max(1, len(prompt) // 4),
            0,
            _stamp_infer_timings(_timings_from_ollama(None, status=500, err_body=str(e)), options, model),
        )


IMAGE_UNSUPPORTED_ANSWER = (
    "Error: this node could not generate an image. Ollama does not run Flux; "
    "the node uses Hugging Face Diffusers (FLUX.2 Klein 4B) on the GPU."
)
VIDEO_UNSUPPORTED_ANSWER = (
    "Error: this node could not generate a video. Ollama does not run Wan; "
    "the node uses Hugging Face Diffusers (Wan 2.1 T2V 1.3B) on the GPU."
)


def _unload_ollama_models() -> None:
    """Free GPU VRAM so Diffusers can load Flux after a text hop."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=15)
        models = (r.json() or {}).get("models") or []
    except Exception as e:
        _log("ollama", f"unload list skip error={e}")
        return
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if not name:
            continue
        try:
            requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": name, "prompt": "", "keep_alive": 0},
                timeout=60,
            )
            _log("ollama", f"unloaded model={name} for Flux Diffusers")
        except Exception as e:
            _log("ollama", f"unload model={name} error={e}")


def _image_unsupported_timings(options: dict, model: str, status: int | None, err_body: str) -> dict:
    out = _stamp_infer_timings(
        _timings_from_ollama(None, status=status, err_body=err_body),
        options,
        model,
    )
    out["image_unsupported"] = True
    return out


def ask_ollama_image_generate(
    prompt: str, model: str, job_id=None, infer: dict | None = None
) -> tuple[str, int, int, dict]:
    """Flux / graphical path: Hugging Face Diffusers, never bill PNG length."""
    infer = infer if isinstance(infer, dict) else {}
    images = infer.get("images")
    strength = infer.get("image_strength")
    try:
        strength_f = float(strength) if strength is not None else None
    except (TypeError, ValueError):
        strength_f = None
    opt_infer = {k: v for k, v in infer.items() if k not in ("images", "image_strength")}
    options = _ollama_options(model, opt_infer)
    _log_inference_start(job_id, "image-generate", model, options, extra=f"prompt_chars={len(prompt)}")
    _dump("IMAGE_GENERATE_META", f"job={job_id} model={model} chars={len(prompt)} runner=diffusers")
    _dump_block("IMAGE_GENERATE_PROMPT", prompt, extra=f"job={job_id} chars={len(prompt)}")
    if flux_hf.runtime_ready():
        _unload_ollama_models()
        try:
            b64, meta = flux_hf.generate(
                prompt,
                images=images if isinstance(images, list) else None,
                strength=strength_f,
            )
            content = image_content_from_b64([b64])
            pt = max(1, len(prompt) // 4)
            t = _stamp_infer_timings(_timings_from_ollama(None, status=200), options, model)
            t["hf_flux"] = True
            t.update({k: v for k, v in meta.items() if k in ("width", "height", "steps", "runner")})
            _dump("IMAGE_GENERATE_RESULT", f"job={job_id} runner=diffusers {to_log_json(meta)}")
            return content, pt, 0, t
        except flux_hf.FluxUnavailable as e:
            _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} flux_hf={e}")
            return (
                IMAGE_UNSUPPORTED_ANSWER,
                max(1, len(prompt) // 4),
                0,
                _image_unsupported_timings(options, model, 503, str(e)),
            )
        except Exception as e:
            _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} flux_hf={e}")
            return (
                f"Error: Flux Diffusers failed: {e}",
                max(1, len(prompt) // 4),
                0,
                _image_unsupported_timings(options, model, 500, str(e)),
            )
    _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} flux_hf runtime not ready")
    return (
        IMAGE_UNSUPPORTED_ANSWER,
        max(1, len(prompt) // 4),
        0,
        _image_unsupported_timings(options, model, 503, "diffusers runtime not ready"),
    )


def ask_ollama_video_generate(
    prompt: str, model: str, job_id=None, infer: dict | None = None
) -> tuple[str, int, int, dict]:
    """Wan / video path: Hugging Face Diffusers, never bill MP4 length."""
    infer = infer if isinstance(infer, dict) else {}
    options = _ollama_options(model, infer)
    _log_inference_start(job_id, "video-generate", model, options, extra=f"prompt_chars={len(prompt)}")
    _dump("VIDEO_GENERATE_META", f"job={job_id} model={model} chars={len(prompt)} runner=diffusers")
    _dump_block("VIDEO_GENERATE_PROMPT", prompt, extra=f"job={job_id} chars={len(prompt)}")
    if wan_hf.runtime_ready():
        _unload_ollama_models()
        try:
            b64, meta = wan_hf.generate(prompt)
            content = video_content_from_b64([b64])
            pt = max(1, len(prompt) // 4)
            t = _stamp_infer_timings(_timings_from_ollama(None, status=200), options, model)
            t["hf_wan"] = True
            t.update(
                {
                    k: v
                    for k, v in meta.items()
                    if k in ("width", "height", "frames", "steps", "fps", "runner")
                }
            )
            _dump("VIDEO_GENERATE_RESULT", f"job={job_id} runner=diffusers {to_log_json(meta)}")
            return content, pt, 0, t
        except wan_hf.WanUnavailable as e:
            _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} wan_hf={e}")
            return (
                VIDEO_UNSUPPORTED_ANSWER,
                max(1, len(prompt) // 4),
                0,
                _image_unsupported_timings(options, model, 503, str(e)),
            )
        except Exception as e:
            _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} wan_hf={e}")
            return (
                f"Error: Wan Diffusers failed: {e}",
                max(1, len(prompt) // 4),
                0,
                _image_unsupported_timings(options, model, 500, str(e)),
            )
    _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} wan_hf runtime not ready")
    return (
        VIDEO_UNSUPPORTED_ANSWER,
        max(1, len(prompt) // 4),
        0,
        _image_unsupported_timings(options, model, 503, "diffusers runtime not ready"),
    )


def _parse_chat_data(data: dict, tools: list | None, job_id) -> tuple:
    msg = data.get("message") or {}
    content = msg.get("content")
    if content is None:
        content = ""
    raw_calls = msg.get("tool_calls") or []
    openai_calls = ollama_tool_calls_to_openai(raw_calls) if raw_calls else None
    if openai_calls:
        openai_calls = [_normalize_file_tool_args(tc) for tc in openai_calls]
    vis = visible_content(content, None)
    need_promote = not openai_calls or (
        bool(tools)
        and vis
        and any(not _file_tool_args_ok(tc) for tc in openai_calls)
    )
    if need_promote and tools and vis:
        promoted = promote_content_tool_calls(
            vis,
            allowed_names=_tool_names_from_tools(tools),
            tools_requested=True,
        )
        if promoted:
            _log(
                "ollama",
                f"QUEUE.TOOL_PROMOTE job={job_id} count={len(promoted)} "
                f"names={','.join(tc['function']['name'] for tc in promoted)}",
            )
            openai_calls = promoted
            content = None
    finish = finish_from_ollama(data, has_tool_calls=bool(openai_calls))
    vis = visible_content(content, None)
    return content, openai_calls, finish, vis


def _overflow_fail(job_id, status, err_body, options, model):
    _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} status={status} body={err_body}")
    timings = _stamp_infer_timings(
        _timings_from_ollama(None, status=status, err_body=err_body),
        options,
        model,
    )
    timings["ctx_overflow"] = True
    timings["oom"] = False
    return ("Error: prompt_too_long", None, "stop", 1, 0, timings)


def _log_payload_trim(job_id, trim: dict, *, retry: bool = False):
    action = str(trim.get("rung") or trim.get("action") or "")
    if action in ("", "none", "passthrough"):
        return
    extra = " retry_overflow" if retry else ""
    _log(
        "ollama",
        f"QUEUE.TOOLS_TRIMMED job={job_id} action={action}{extra} "
        f"kept={','.join(trim.get('kept') or [])} "
        f"dropped={','.join(trim.get('dropped') or [])} "
        f"est={trim.get('est')} budget={trim.get('budget')} "
        f"keep_last={trim.get('keep_last')} cap_chars={trim.get('cap_chars')}",
    )


def _prompt_is_packed(prompt_eval_count, num_ctx, margin: int = 16) -> bool:
    try:
        pe = int(prompt_eval_count or 0)
        nctx = int(num_ctx or 0)
    except (TypeError, ValueError):
        return False
    if pe <= 0 or nctx <= 0:
        return False
    return pe >= nctx - margin


def _assemble_ndjson_chat(text: str) -> dict | None:
    acc: dict = {}
    msg: dict = {"role": "assistant", "content": ""}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("data:"):
            line = line[5:].strip() if line.startswith("data:") else line
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        acc.update({k: v for k, v in chunk.items() if k != "message"})
        part = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
        if part.get("content"):
            msg["content"] = (msg.get("content") or "") + str(part.get("content") or "")
        if part.get("tool_calls"):
            msg["tool_calls"] = part.get("tool_calls")
        if part.get("thinking"):
            msg["thinking"] = (msg.get("thinking") or "") + str(part.get("thinking") or "")
    if not acc and not (msg.get("content") or msg.get("tool_calls")):
        return None
    acc["message"] = msg
    return acc


def _read_chat_response(response) -> tuple[dict | None, str, int]:
    status = int(getattr(response, "status_code", 0) or 0)
    text = getattr(response, "text", None) or ""
    data = None
    if status == 200:
        try:
            data = response.json()
            if isinstance(data, dict):
                return data, text, status
        except Exception:
            data = None
        data = _assemble_ndjson_chat(text)
        return data, text, status
    try:
        data = response.json()
        if not isinstance(data, dict):
            data = None
    except Exception:
        data = _assemble_ndjson_chat(text)
    return data, text, status


def ask_ollama_chat(
    messages: list,
    model: str,
    job_id=None,
    tools: list | None = None,
    infer: dict | None = None,
) -> tuple[str | None, list | None, str, int, int, dict]:
    """
    Prefer /api/chat. Returns (content, tool_calls|None, finish_reason, pt, ct, timings).
    Stateless: claim → infer → answer. Promote bare tool JSON before returning.
    """
    infer = infer if isinstance(infer, dict) else {}
    options = _ollama_options(model, infer, tools)
    msgs_now, tools_now, trim = prepare_ollama_payload(
        messages,
        tools,
        num_ctx=options.get("num_ctx"),
        num_predict=options.get("num_predict"),
    )
    if str(trim.get("rung") or "") == UNFIT_RUNG or trim.get("overflow"):
        _log_payload_trim(job_id, trim)
        return _overflow_fail(job_id, 0, "est_exceeds_budget", options, model)
    want_predict = options.get("num_predict")
    capped_predict = clamp_num_predict(want_predict, options.get("num_ctx"), trim.get("est"))
    if capped_predict != want_predict:
        _log(
            "ollama",
            f"QUEUE.DECODE_CAP job={job_id} want={want_predict} num_predict={capped_predict} "
            f"est={trim.get('est')} num_ctx={options.get('num_ctx')}",
        )
    options["num_predict"] = capped_predict
    rung = str(trim.get("rung") or "passthrough")
    ollama_messages = messages_for_ollama(msgs_now, native=False)
    report = wire_loss_report(messages, ollama_messages)
    _log(
        "ollama",
        f"QUEUE.MESSAGES_TO_OLLAMA job={job_id} native=0 "
        f"inbound={format_messages_wire(messages)} mapped={format_messages_wire(ollama_messages)} "
        f"ok={int(report['ok'])} losses={len(report['losses'])}",
    )
    if report["losses"]:
        _log(
            "ollama",
            f"QUEUE.MESSAGES_WIRE_LOSS job={job_id} " + " | ".join(report["losses"][:12]),
        )
    # Omit OpenAI tool_choice — sealed on the job, not forwarded to Ollama.
    payload: dict = {
        "model": model,
        "messages": ollama_messages,
        "stream": bool(tools),
        "keep_alive": _ollama_keep_alive(),
        "options": options,
    }
    think = ollama_think_flag(infer, tools)
    if think is not None:
        payload["think"] = think
    fmt = _infer_chat_format(infer)
    if fmt:
        payload["format"] = fmt
    # Disable llama.cpp context-shift so overflow 400s instead of dropping the user task.
    payload["shift"] = False
    payload["options"]["shift"] = False

    _log_payload_trim(job_id, trim)
    if tools:
        payload["tools"] = tools_now
        in_sha = _tools_sha256(tools)
        out_sha = _tools_sha256(payload["tools"])
        _log(
            "ollama",
            f"QUEUE.TOOLS_TO_OLLAMA job={job_id} sha256={out_sha} n={len(payload['tools'])} "
            f"names={_tool_names_list(payload['tools'])} "
            f"passthrough={'ok' if in_sha == out_sha else 'trimmed'}",
        )
        _dump_json(
            "TOOLS_TO_OLLAMA",
            payload["tools"],
            extra=f"job={job_id} sha256={out_sha} n={len(payload['tools'])}",
        )

    roles = ",".join(str(m.get("role")) for m in ollama_messages[:30])
    _log_inference_start(
        job_id,
        "chat",
        model,
        payload["options"],
        extra=(
            f"msg_count={len(ollama_messages)} tools={len(payload.get('tools') or [])} "
            f"think={payload.get('think')!r} format={payload.get('format')!r} roles={roles}"
        ),
    )
    _dump(
        "CHAT_TO_OLLAMA_META",
        f"job={job_id} model={model} think={payload.get('think')!r} format={payload.get('format')!r} "
        f"shift={payload.get('shift')!r} stream={payload.get('stream')} keep_alive={payload.get('keep_alive')!r} "
        f"msg_count={len(ollama_messages)} tools={len(payload.get('tools') or [])} "
        f"options={to_log_json(payload.get('options'))}",
    )
    qtext = last_user_text(ollama_messages)
    if qtext:
        _dump_block("USER_QUESTION", qtext, extra=f"job={job_id} chars={len(qtext)}")
    _dump_block(
        "CHAT_MESSAGES",
        format_messages_transcript(ollama_messages),
        extra=f"job={job_id} msg_count={len(ollama_messages)}",
    )

    def _post(*, timeout=None):
        return requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT if timeout is None else timeout,
        )

    def _set_num_predict(n: int) -> None:
        options["num_predict"] = n
        payload["options"]["num_predict"] = n

    def _apply_rung(next_rung: str, *, retry: bool):
        nonlocal msgs_now, tools_now, trim, rung, ollama_messages
        msgs_now, tools_now, trim = prepare_ollama_payload(
            messages,
            tools,
            num_ctx=options.get("num_ctx"),
            num_predict=options.get("num_predict"),
            rung=next_rung,
        )
        rung = str(trim.get("rung") or next_rung)
        ollama_messages = messages_for_ollama(msgs_now, native=False)
        payload["messages"] = ollama_messages
        if tools:
            payload["tools"] = tools_now
        _log_payload_trim(job_id, trim, retry=retry)

    try:
        retries = 0
        shape_tried = False
        json_predict_cut = False
        response = _post()
        data, err_text, status = _read_chat_response(response)

        def _retry_rung(next_rung: str, reason: str, *, timeout=None) -> str | None:
            nonlocal retries, response, data, err_text, status
            retries += 1
            _log(
                "ollama",
                f"QUEUE.CTX_RETRY job={job_id} rung={next_rung} attempt={retries} {reason}",
            )
            _apply_rung(next_rung, retry=True)
            if str(trim.get("rung") or "") == UNFIT_RUNG or trim.get("overflow"):
                return "overflow"
            response = _post(timeout=timeout)
            data, err_text, status = _read_chat_response(response)
            return None

        def _repaired_invalid_tool_json():
            repaired = None
            blob = ""
            if isinstance(data, dict):
                msg = data.get("message") if isinstance(data.get("message"), dict) else {}
                blob = str(msg.get("content") or "")
                raw_calls = msg.get("tool_calls") or []
                if raw_calls:
                    repaired = ollama_tool_calls_to_openai(raw_calls)
            if not repaired:
                repaired = promote_content_tool_calls(
                    blob or err_text,
                    allowed_names=_tool_names_from_tools(payload.get("tools") or tools),
                    tools_requested=True,
                )
            if not repaired:
                return None
            repaired = [_normalize_file_tool_args(tc) for tc in repaired]
            _log(
                "ollama",
                f"QUEUE.TOOL_REPAIR job={job_id} count={len(repaired)} "
                f"from_invalid_tool_json",
            )
            timings = _stamp_infer_timings(
                _timings_from_ollama(data, status=200),
                payload["options"],
                model,
            )
            return (
                None,
                repaired,
                "tool_calls",
                1,
                max(1, len(json.dumps(repaired)) // 4),
                timings,
            )

        while True:
            if (
                status != 200
                and is_context_overflow(status, err_text)
                and retries < MAX_CTX_RETRIES
            ):
                nxt = next_ctx_rung(rung)
                if nxt:
                    if _retry_rung(nxt, f"status={status}") == "overflow":
                        return _overflow_fail(job_id, status, "est_exceeds_budget", options, model)
                    continue
            if (
                status == 200
                and data
                and _prompt_is_packed(
                    data.get("prompt_eval_count"), payload["options"].get("num_ctx")
                )
            ):
                nxt = next_body_trim_rung(rung)
                if nxt and retries < MAX_CTX_RETRIES:
                    if _retry_rung(nxt, "status=200 packed_prompt") == "overflow":
                        return _overflow_fail(job_id, 200, "packed_prompt_eval", options, model)
                    continue
                return _overflow_fail(job_id, 200, "packed_prompt_eval", options, model)
            if status != 200 and is_invalid_tool_json_error(status, err_text):
                got = _repaired_invalid_tool_json()
                if got:
                    return got
                nxt = next_body_trim_rung(rung)
                if retries < MAX_CTX_RETRIES and (nxt or not json_predict_cut):
                    json_predict_cut = True
                    _set_num_predict(INVALID_TOOL_RETRY_PREDICT)
                    reason = (
                        f"status={status} invalid_tool_json "
                        f"num_predict={INVALID_TOOL_RETRY_PREDICT}"
                    )
                    if nxt:
                        if _retry_rung(nxt, reason, timeout=INVALID_TOOL_RETRY_TIMEOUT) == "overflow":
                            return _overflow_fail(job_id, status, "est_exceeds_budget", options, model)
                    else:
                        retries += 1
                        _log(
                            "ollama",
                            f"QUEUE.CTX_RETRY job={job_id} rung={rung} attempt={retries} {reason}",
                        )
                        response = _post(timeout=INVALID_TOOL_RETRY_TIMEOUT)
                        data, err_text, status = _read_chat_response(response)
                    continue
                err_body = (err_text or "")[:400].replace("\n", " ")
                _log(
                    "ollama",
                    f"QUEUE.INFERENCE_ERR job={job_id} status={status} body={err_body}",
                )
                timings = _stamp_infer_timings(
                    _timings_from_ollama(None, status=status, err_body=err_body),
                    payload["options"],
                    model,
                )
                timings["truncated_tool_call"] = True
                timings["ok"] = False
                return ("Error: truncated_tool_call", None, "stop", 1, 0, timings)
            if (
                not shape_tried
                and tools
                and status in (400, 404, 500)
                and not is_context_overflow(status, err_text)
                and not is_invalid_tool_json_error(status, err_text)
            ):
                shape_tried = True
                native_messages = messages_for_ollama(msgs_now, native=True)
                _log(
                    "ollama",
                    f"QUEUE.CHAT_TOOLS_SHAPE_FALLBACK job={job_id} status={status} "
                    f"retry_ollama_native keep_tools sha256={_tools_sha256(payload.get('tools'))} "
                    f"mapped={format_messages_wire(native_messages)}",
                )
                payload["messages"] = native_messages
                response = _post()
                data, err_text, status = _read_chat_response(response)
                continue
            break

        if status != 200:
            overflow = is_context_overflow(status, err_text)
            err_body = (err_text or "")[:400].replace("\n", " ")
            if overflow:
                return _overflow_fail(job_id, status, err_body, options, model)
            _log(
                "ollama",
                f"QUEUE.INFERENCE_ERR job={job_id} status={status} body={err_body}",
            )
            return (
                f"Error from Ollama: {status}",
                None,
                "stop",
                1,
                0,
                _stamp_infer_timings(
                    _timings_from_ollama(None, status=status, err_body=err_body),
                    payload["options"],
                    model,
                ),
            )

        if not data:
            return (
                "Error from Ollama: empty",
                None,
                "stop",
                1,
                0,
                _stamp_infer_timings(
                    _timings_from_ollama(None, status=500, err_body="empty"),
                    payload["options"],
                    model,
                ),
            )

        content, openai_calls, finish, vis = _parse_chat_data(data, payload.get("tools") or tools, job_id)
        if not openai_calls and not vis and payload.get("think") is not False:
            _log("ollama", f"QUEUE.THINK_RETRY job={job_id} empty_visible retry_think_false")
            payload["think"] = False
            response = _post()
            data2, _err2, status2 = _read_chat_response(response)
            if status2 == 200 and data2:
                data = data2
                content, openai_calls, finish, vis = _parse_chat_data(
                    data, payload.get("tools") or tools, job_id
                )

        pt = int(
            data.get("prompt_eval_count")
            or max(1, sum(len(str(m.get("content") or "")) for m in ollama_messages) // 4)
        )
        basis = (content or "") + (json.dumps(openai_calls) if openai_calls else "")
        ct = int(data.get("eval_count") or max(1, len(basis) // 4))
        timings = _stamp_infer_timings(_timings_from_ollama(data, status=200), payload["options"], model)
        try:
            est = int(trim.get("est") or 0)
        except (TypeError, ValueError):
            est = 0
        if est and pt:
            _log(
                "ollama",
                f"QUEUE.CTX_EST job={job_id} est={est} prompt_eval={pt} "
                f"num_ctx={payload['options'].get('num_ctx')}",
            )
        if not openai_calls and not vis:
            timings["empty_completion"] = True
            timings["ok"] = False
            _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} empty_completion finish={finish} tokens_out={ct}")
            _dump_assistant_turn(job_id, finish=finish, content=content, tool_calls=None, data=data, empty=True)
            return ("Error: empty_completion", None, finish, pt, ct, timings)

        _dump_assistant_turn(job_id, finish=finish, content=content, tool_calls=openai_calls, data=data, empty=False)
        return (
            content if content else (None if openai_calls else ""),
            openai_calls,
            finish,
            pt,
            ct,
            timings,
        )
    except Exception as e:
        _log("ollama", f"QUEUE.INFERENCE_ERR job={job_id} error={e}")
        return (
            f"Error communicating with Ollama: {e}",
            None,
            "stop",
            1,
            0,
            _stamp_infer_timings(_timings_from_ollama(None, status=500, err_body=str(e)), payload["options"], model),
        )


def run_inference(job: dict) -> tuple[str | None, list | None, str, int, int, dict]:
    """
    Chat jobs → /api/chat; graphical → /api/generate image; legacy text → generate.
    One-shot only — no node-local conversation memory.
    Always runs the sealed ollama_tag on graph hops; otherwise the local active tag.
    """
    job_id = job.get("job_id") or job.get("id")
    fam = resolve_family(job.get("model_family") or job.get("model") or NODE_FAMILY)
    sealed = _graph_hop_tag(job)
    model = sealed or active_tag_for(fam)
    messages = job.get("messages")
    tools = job.get("tools") if isinstance(job.get("tools"), list) else None
    infer = job.get("infer") if isinstance(job.get("infer"), dict) else None
    _dump(
        "INFER_START",
        f"job={job_id} family={fam} tag={model} role={job.get('graph_role') or '-'} "
        f"mode={job.get('mode')} "
        f"infer={to_log_json(infer)} "
        f"msg_count={len(messages) if isinstance(messages, list) else 0} "
        f"tools={len(tools or [])}",
    )

    if is_image_family(fam) or str(job.get("mode") or "") == "image":
        prompt = job.get("prompt") or job.get("question") or _compose_prompt(job)
        answer, pt, ct, timings = ask_ollama_image_generate(prompt, model, job_id=job_id, infer=infer)
        return answer, None, "stop", pt, ct, timings

    if is_video_family(fam) or str(job.get("mode") or "") == "video":
        prompt = job.get("prompt") or job.get("question") or _compose_prompt(job)
        answer, pt, ct, timings = ask_ollama_video_generate(prompt, model, job_id=job_id, infer=infer)
        return answer, None, "stop", pt, ct, timings

    if isinstance(messages, list) and messages:
        content, tcs, finish, pt, ct, timings = ask_ollama_chat(
            messages, model, job_id=job_id, tools=tools, infer=infer
        )
        if isinstance(content, str) and content.startswith("Error") and (job.get("prompt") or job.get("question")):
            if tools or sealed:
                tag = "CHAT_KEEP_TOOLS_NO_GENERATE" if tools else "GRAPH_NO_GENERATE_FALLBACK"
                _log(
                    "ollama",
                    f"QUEUE.{tag} job={job_id} "
                    "chat error; not falling back to /api/generate",
                )
                return content, tcs, finish, pt, ct, timings
            _log("ollama", f"QUEUE.CHAT_TO_GENERATE_FALLBACK job={job_id}")
            prompt = job.get("prompt") or _compose_prompt(job)
            answer, pt2, ct2, timings2 = ask_ollama_generate(prompt, model, job_id=job_id, infer=infer)
            return answer, None, "stop", pt2, ct2, timings2
        return content, tcs, finish, pt, ct, timings

    prompt = job.get("prompt") or _compose_prompt(job)
    answer, pt, ct, timings = ask_ollama_generate(prompt, model, job_id=job_id, infer=infer)
    return answer, None, "stop", pt, ct, timings


def _open_claim(data: dict) -> dict | None:
    env = data.get("envelope")
    jid = str(data.get("job_id") or data.get("id") or "")
    if not env or not jid:
        _log("worker", "ERROR claim missing envelope")
        return None
    try:
        inner = decode_payload(str(env), api_token=API_TOKEN, job_id=jid, purpose="claim")
    except WireCryptoError as e:
        _log("worker", f"ERROR claim decrypt: {e}")
        return None
    opened = {k: v for k, v in data.items() if k != "envelope"}
    opened.update(inner)
    return opened


def get_question():
    global SENDED_QUESTIONS, LAST_ERROR, POLL_COUNT, LAST_POLL_AT, LAST_EMPTY_POLL_AT
    POLL_COUNT += 1
    LAST_POLL_AT = datetime.now(timezone.utc).isoformat()
    _set_phase("polling", f"poll #{POLL_COUNT} families={','.join(advertised_families())} tag={active_tag()}")
    try:
        models_param = ",".join(advertised_families())
        r = requests.get(
            f"{API_BASE}/getQuestion",
            headers=headers,
            params={"models": models_param},
            timeout=60,
        )
        if r.status_code == 200:
            raw = r.json()
            data = _open_claim(raw) if isinstance(raw, dict) else None
            if not data:
                LAST_ERROR = "claim envelope invalid"
                _log("worker", f"ERROR getQuestion: {LAST_ERROR}")
                _set_phase("error", LAST_ERROR)
                return None
            SENDED_QUESTIONS += 1
            jid = data.get("job_id") or data.get("id")
            msg_n = len(data.get("messages") or []) if isinstance(data.get("messages"), list) else 0
            claimed_msgs = data.get("messages") if isinstance(data.get("messages"), list) else []
            tool_n = len(data.get("tools") or []) if isinstance(data.get("tools"), list) else 0
            _log(
                "worker",
                f"QUEUE.CLAIMED job={jid} model={data.get('model')} mode={data.get('mode')} "
                f"family={data.get('model_family') or data.get('model')} "
                f"role={data.get('graph_role') or '-'} tag={data.get('ollama_tag') or '-'} "
                f"run={data.get('graph_run_id') or '-'} "
                f"msg_count={msg_n} tools={tool_n} "
                f"msg_wire={format_messages_wire(claimed_msgs)} "
                f"poll={POLL_COUNT} scope={'project:' + PROJECT_ID if PROJECT_ID else 'public'}",
            )
            _log(
                "job",
                f"QUEUE.JOB_CLAIM job={jid} phase=claimed "
                f"family={data.get('model_family') or data.get('model')} "
                f"role={data.get('graph_role') or '-'} tag={data.get('ollama_tag') or '-'} "
                f"run={data.get('graph_run_id') or '-'} mode={data.get('mode') or '-'} "
                f"msg_count={msg_n} tools={tool_n} "
                f"tool_names={_tool_names_list(data.get('tools') if isinstance(data.get('tools'), list) else []) or '-'} "
                f"msg_wire={format_messages_wire(claimed_msgs)} "
                f"{last_user_log_field(claimed_msgs, plaintext=PLAINTEXT_LOGS, fallback=str(data.get('question') or data.get('prompt') or ''))}",
            )
            qtext = last_user_text(claimed_msgs) or str(data.get("question") or data.get("prompt") or "")
            _dump(
                "CLAIMED_META",
                f"job={jid} model={data.get('model')} family={data.get('model_family')} "
                f"role={data.get('graph_role') or '-'} tag={data.get('ollama_tag') or '-'} "
                f"mode={data.get('mode')} infer={to_log_json(data.get('infer'))} "
                f"msg_count={msg_n} "
                f"tools={len(data.get('tools') or []) if isinstance(data.get('tools'), list) else 0}",
            )
            if qtext:
                _dump_block("USER_QUESTION", qtext, extra=f"job={jid} claimed chars={len(qtext)}")
            if claimed_msgs:
                _dump_block(
                    "CLAIMED_MESSAGES",
                    format_messages_transcript(claimed_msgs),
                    extra=f"job={jid} msg_count={msg_n}",
                )
            elif data.get("question") or data.get("prompt"):
                _dump_block("CLAIMED_LEGACY_PROMPT", _compose_prompt(data), extra=f"job={jid}")
            if isinstance(data.get("tools"), list) and data.get("tools"):
                _dump_json("CLAIMED_TOOLS", data.get("tools"), extra=f"job={jid}")
            return data
        if r.status_code == 404:
            LAST_EMPTY_POLL_AT = LAST_POLL_AT
            if POLL_COUNT == 1 or POLL_COUNT % 10 == 0:
                _log(
                    "worker",
                    f"QUEUE.POLL_EMPTY poll={POLL_COUNT} "
                    f"scope={'project:' + PROJECT_ID if PROJECT_ID else 'public'} "
                    f"models={models_param}",
                )
            _set_phase("idle", f"queue empty after poll #{POLL_COUNT}")
            return None
        LAST_ERROR = f"getQuestion {r.status_code}: {r.text[:200]}"
        _log("worker", f"ERROR getQuestion: {LAST_ERROR}")
        _set_phase("error", LAST_ERROR)
        return None
    except Exception as e:
        LAST_ERROR = str(e)
        _log("worker", f"ERROR getQuestion: {e}")
        _set_phase("error", str(e))
        return None


def send_answer(
    job_id: str,
    content: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    model: str | None = None,
    decode_tps_val: float | None = None,
    num_ctx: int | None = None,
    error_code: str | None = None,
):
    global ANSWERED_QUESTIONS, LAST_ERROR
    try:
        inner: dict = {
            "answer": content if content is not None else "",
            "finish_reason": finish_reason,
        }
        if error_code:
            inner["error_code"] = error_code
        if tool_calls:
            inner["tool_calls"] = tool_calls
            inner["assistant_message"] = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            inner["finish_reason"] = "tool_calls"
        body = {
            "id": job_id,
            "encoding": "v1",
            "envelope": encode_payload(
                inner,
                api_token=API_TOKEN,
                job_id=str(job_id),
                purpose="answer",
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model or active_tag(),
            "model_used": model or active_tag(),
        }
        if decode_tps_val is not None:
            try:
                body["decode_tps"] = round(float(decode_tps_val), 2)
            except (TypeError, ValueError):
                pass
        if num_ctx:
            try:
                body["num_ctx"] = int(num_ctx)
            except (TypeError, ValueError):
                pass
        _dump(
            "ANSWER_META",
            f"job={job_id} finish={inner['finish_reason']} error_code={error_code} "
            f"tokens_in={prompt_tokens} tokens_out={completion_tokens} "
            f"model={model} num_ctx={num_ctx} tps={decode_tps_val}",
        )
        if content:
            _dump_block("ANSWER_BODY", content, extra=f"job={job_id} chars={len(content)}")
        if tool_calls:
            _dump_json("ANSWER_TOOL_CALLS", tool_calls, extra=f"job={job_id}")
        _log(
            "worker",
            f"QUEUE.ANSWER_SENT job={job_id} finish={inner['finish_reason']} "
            f"answer_chars={len(content or '')} tool_calls={len(tool_calls) if tool_calls else 0} "
            f"tokens_in={prompt_tokens} tokens_out={completion_tokens} "
            f"summary={summarize_assistant_output(content, tool_calls, detail=PLAINTEXT_LOGS)}",
        )
        _log(
            "job",
            f"QUEUE.JOB_SEND job={job_id} phase=answering finish={inner['finish_reason']} "
            f"answer_chars={len(content or '')} tool_calls={len(tool_calls) if tool_calls else 0} "
            f"tokens_in={prompt_tokens} tokens_out={completion_tokens} "
            f"summary={summarize_assistant_output(content, tool_calls, detail=PLAINTEXT_LOGS)}",
        )
        r = requests.post(
            f"{API_BASE}/sendAnswer",
            headers=headers,
            json=body,
            timeout=60,
        )
        if r.status_code == 200 and "PROCESSED" in r.text:
            ANSWERED_QUESTIONS += 1
            _log(
                "worker",
                f"QUEUE.DONE job={job_id} answered_total={ANSWERED_QUESTIONS}",
            )
            _log(
                "job",
                f"QUEUE.JOB_DONE job={job_id} phase=done answered_total={ANSWERED_QUESTIONS}",
            )
        elif r.status_code == 200:
            LAST_ERROR = f"sendAnswer {r.status_code}: {r.text[:200]}"
            _log("worker", f"QUEUE.ANSWER_RESULT job={job_id} {r.text[:200]}")
            _log("job", f"QUEUE.JOB_FAIL job={job_id} phase=fail error=sendAnswer {LAST_ERROR}")
        else:
            LAST_ERROR = f"sendAnswer {r.status_code}: {r.text[:200]}"
            _log("worker", f"ERROR sendAnswer: {LAST_ERROR}")
            _log("job", f"QUEUE.JOB_FAIL job={job_id} phase=fail error=sendAnswer {LAST_ERROR}")
        return r.json() if r.content else {}
    except Exception as e:
        LAST_ERROR = str(e)
        _log("worker", f"ERROR sendAnswer: {e}")
        _log("job", f"QUEUE.JOB_FAIL job={job_id} phase=fail error=sendAnswer {e}")
        return {}


def main_loop():
    global ADAPT_STATE, ADAPT_STATE_BY_FAMILY, HARDWARE
    if _HARDWARE_IMPORT_ERROR:
        _log(
            "worker",
            f"WARNING hardware module missing ({_HARDWARE_IMPORT_ERROR}) — continuing without fleet probe",
        )
    if probe_hardware:
        try:
            HARDWARE = probe_hardware()
            if HARDWARE and summarize_label:
                _log(
                    "worker",
                    f"hardware {summarize_label(HARDWARE)} compute_index={HARDWARE.get('compute_index')}",
                )
        except Exception as e:
            _log("worker", f"WARNING hardware probe failed: {e}")
    ADAPT_STATE = load_adapt_state(HARDWARE)
    save_adapt_state(ADAPT_STATE)
    _set_phase("warming", f"loading {ADAPT_STATE.get('active_tag') or active_tag()}")
    _warmup_active_model()
    _log(
        "worker",
        f"QUEUE.NODE_START families={advertised_families()} active={ADAPT_STATE.get('active_tag')} "
        f"rebased={int(bool(ADAPT_STATE.get('rebased')))} "
        f"pulled={NODE_MODELS} api={API_BASE} scope={'project:' + PROJECT_ID if PROJECT_ID else 'public'} "
        f"plaintext_logs={PLAINTEXT_LOGS}",
    )
    _set_phase("idle", "started")
    while True:
        heartbeat()
        q = get_question()
        heartbeat()
        if q:
            job_id = q.get("job_id") or q.get("id")
            job_fam = resolve_family(q.get("model_family") or q.get("model") or NODE_FAMILY)
            graph_hop = bool(_graph_hop_tag(q))
            model = _graph_hop_tag(q) or active_tag_for(job_fam)
            _set_phase("processing", f"running {model}", job_id=str(job_id))
            heartbeat()
            t0 = time.time()
            content, tool_calls, finish, pt, ct, timings = run_inference(q)
            elapsed = time.time() - t0
            tps = decode_tps(timings.get("eval_count"), timings.get("eval_duration"))
            oom = bool(timings.get("oom"))
            ctx_overflow = bool(timings.get("ctx_overflow"))
            if isinstance(content, str) and content.startswith("Error"):
                oom = oom or is_load_error(500, content)
                ctx_overflow = ctx_overflow or is_context_overflow(500, content)
            if ctx_overflow:
                oom = False
            # Packed 500 / truncated tool JSON is a window problem, not a slow 8B.
            adapt_overflow = ctx_overflow or bool(timings.get("truncated_tool_call"))
            try:
                pe_ns = float(timings.get("prompt_eval_duration") or 0)
            except (TypeError, ValueError):
                pe_ns = 0.0
            prompt_eval_s = pe_ns / 1e9 if pe_ns else 0.0
            skip_slow = (
                is_think_hop_job(q, timings)
                or is_image_family(job_fam)
                or is_video_family(job_fam)
                or graph_hop
            )
            fam_state = (ADAPT_STATE_BY_FAMILY or {}).get(job_fam) or ADAPT_STATE or empty_state(
                job_fam, model, pulled_for_family(job_fam, HARDWARE)
            )
            if not graph_hop:
                fam_state = apply_adapt(
                    fam_state,
                    hardware=HARDWARE,
                    pulled=pulled_for_family(job_fam, HARDWARE),
                    decode_tps_val=tps,
                    wall_s=elapsed,
                    oom=oom,
                    ctx_overflow=adapt_overflow,
                    skip_slow=skip_slow,
                    target_tps=NODE_TARGET_TPS,
                    wall_slow_s=NODE_WALL_SLOW_SECONDS,
                    prompt_eval_s=prompt_eval_s,
                )
            fam_state["last_num_ctx"] = timings.get("num_ctx")
            fam_state["last_load_mode"] = timings.get("load_mode")
            fam_state["last_hybrid"] = timings.get("load_mode") == "hybrid"
            try:
                fam_state["num_parallel"] = int(os.getenv("OLLAMA_NUM_PARALLEL") or "1")
            except (TypeError, ValueError):
                fam_state["num_parallel"] = 1
            ADAPT_STATE_BY_FAMILY[job_fam] = fam_state
            if job_fam == NODE_FAMILY or not ADAPT_STATE:
                ADAPT_STATE = fam_state
            save_adapt_state()
            stepped = fam_state.get("stepped")
            err_code = (
                "prompt_too_long"
                if ctx_overflow
                else (
                    "empty_completion"
                    if timings.get("empty_completion")
                    else (
                        "truncated_tool_call"
                        if timings.get("truncated_tool_call")
                        else (
                            "image_unsupported"
                            if timings.get("image_unsupported")
                            else (
                                "ollama_error"
                                if isinstance(content, str) and content.startswith("Error")
                                else None
                            )
                        )
                    )
                )
            )
            _log(
                "ollama",
                f"QUEUE.INFERENCE_DONE job={job_id} in={elapsed:.1f}s finish={finish} "
                f"tokens_in={pt} tokens_out={ct} tool_calls={len(tool_calls) if tool_calls else 0} "
                f"family={job_fam} tag={model} tps={tps if tps is not None else '-'} "
                f"speed={fam_state.get('last_speed')} fit={fam_state.get('last_fit')} "
                f"step={stepped or 'stay'} load={timings.get('load_mode') or '-'} "
                f"num_ctx={timings.get('num_ctx') or '-'} prompt_eval_s={prompt_eval_s:.1f} "
                f"skip_slow={int(skip_slow)}",
            )
            _log(
                "job",
                f"QUEUE.JOB_RESULT job={job_id} phase=answered "
                f"role={q.get('graph_role') or '-'} tag={model} run={q.get('graph_run_id') or '-'} "
                f"status={'fail' if err_code else 'ok'} error={err_code or '-'} "
                f"finish={finish} tokens_in={pt} tokens_out={ct} "
                f"tps={tps if tps is not None else '-'} wall={elapsed:.1f}s "
                f"prompt_eval_s={prompt_eval_s:.1f} "
                f"summary={summarize_assistant_output(content, tool_calls, detail=PLAINTEXT_LOGS)}",
            )
            send_answer(
                str(job_id),
                content,
                pt,
                ct,
                tool_calls=tool_calls,
                finish_reason=finish,
                model=model,
                decode_tps_val=tps,
                num_ctx=timings.get("num_ctx"),
                error_code=err_code if err_code != "ollama_error" else None,
            )
            nxt = fam_state.get("active_tag")
            nxt_ctx = fam_state.get("active_num_ctx")
            if (
                nxt
                and not graph_hop
                and not is_image_family(job_fam)
                and not is_video_family(job_fam)
                and (nxt != model or nxt_ctx != timings.get("num_ctx"))
            ):
                _warmup_ollama_tag(nxt, nxt_ctx)
            _set_phase("idle", f"finished job {job_id}")
            heartbeat()
        time.sleep(3 + random.random() * 2)


@app.get("/status")
def status():
    return jsonify(
        {
            "node_name": NODE_NAME,
            "family": NODE_FAMILY,
            "families": advertised_families(),
            "models": list(NODE_MODELS) if NODE_FAMILY == GRAPH_FAMILY else advertised_families(),
            "active_tag": (ADAPT_STATE or {}).get("active_tag") or GRAPH_ORCH_TAG,
            "active_num_ctx": (ADAPT_STATE or {}).get("active_num_ctx"),
            "adapt": _adapt_snapshot_payload() if ADAPT_STATE or ADAPT_STATE_BY_FAMILY else snapshot_for_heartbeat(empty_state(NODE_FAMILY, active_tag(), NODE_MODELS), HARDWARE),
            "pulled": NODE_MODELS,
            "project_id": PROJECT_ID,
            "phase": PHASE,
            "activity_detail": ACTIVITY_DETAIL,
            "current_job_id": CURRENT_JOB_ID,
            "polls_total": POLL_COUNT,
            "sended_questions": SENDED_QUESTIONS,
            "answered_questions": ANSWERED_QUESTIONS,
            "claimed_jobs": SENDED_QUESTIONS,
            "uptime_seconds": (datetime.now(timezone.utc) - START_TIME).total_seconds(),
            "api_base": API_BASE,
            "last_error": LAST_ERROR,
            "last_poll_at": LAST_POLL_AT,
            "last_empty_poll_at": LAST_EMPTY_POLL_AT,
            "hardware": HARDWARE,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "phase": PHASE})


if __name__ == "__main__":
    if not API_TOKEN:
        _log("worker", "WARNING: API_TOKEN is empty")
    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
