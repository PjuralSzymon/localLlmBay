"""Fit the Ollama payload into num_ctx. The sealed job stays full."""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from typing import Any

CORE_TOOLS = frozenset(
    {
        "read",
        "write",
        "edit",
        "bash",
        "glob",
        "grep",
        "list",
        "webfetch",
    }
)

# OpenCode helpers — drop these before domain MCP (godot_*, etc.).
META_TOOLS = frozenset(
    {
        "skill",
        "task",
        "todowrite",
        "todo",
        "question",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    }
)

_VERBOSE_KEYS = frozenset(
    {"description", "title", "examples", "example", "markdownDescription", "markdown_description"}
)

TRUNCATED_SUFFIX = "\n…[truncated]"
STUB_TEXT = "[truncated old tool result]"

# Drop unused MCP before stubbing file bodies. Overflow retries walk this ladder.
CTX_RUNGS = (
    "passthrough",
    "compact",
    "drop_meta",
    "drop_extras",
    "cap_4_2000",
    "cap_2_400",
    "stub",
)
UNFIT_RUNG = "unfit"
MAX_CTX_RETRIES = 5
CODING_NUDGE = (
    "Those file contents are already in this thread. Call Edit with filePath plus "
    "oldString and newString, or Write with filePath and content. Do not Read the "
    "same path again; do not paste the patch as chat."
)
_META_DROP_RUNGS = frozenset({"drop_meta", "drop_extras", "cap_4_2000", "cap_2_400", "stub"})
_EXTRA_DROP_RUNGS = frozenset({"drop_extras", "cap_4_2000", "cap_2_400", "stub"})
# llama.cpp is denser than chars/4. Stay under 2.3 so the first POST leaves decode leftover.
CHARS_PER_TOKEN = 2.2


def estimate_tokens(obj: Any) -> int:
    try:
        blob = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = str(obj)
    if not blob:
        return 0
    return max(1, math.ceil(len(blob) / CHARS_PER_TOKEN))


def tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name") or tool.get("name") or "").strip()


def core_tool_names() -> set[str]:
    raw = os.getenv("NODE_CORE_TOOLS") or ""
    extra = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return {n.lower() for n in CORE_TOOLS} | extra


def names_in_messages(messages: list | None) -> set[str]:
    found: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                n = str(fn.get("name") or tc.get("name") or "").strip()
                if n:
                    found.add(n.lower())
        n = str(m.get("name") or m.get("tool_name") or "").strip()
        if n:
            found.add(n.lower())
    return found


def fit_target(num_ctx: int | None, num_predict: int | None = None) -> int:
    """Leave ~512 tokens of headroom. Do not reserve the full decode budget."""
    del num_predict
    try:
        nctx = int(num_ctx or 0)
    except (TypeError, ValueError):
        nctx = 16384
    if nctx < 1024:
        nctx = 16384
    return max(1024, nctx - 512)


def ctx_budget(num_ctx: int | None, num_predict: int | None = None) -> int:
    return fit_target(num_ctx, num_predict)


def _strip_verbose(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_verbose(v) for k, v in obj.items() if k not in _VERBOSE_KEYS}
    if isinstance(obj, list):
        return [_strip_verbose(x) for x in obj]
    return obj


def compact_tools(tools: list) -> list:
    return [_strip_verbose(deepcopy(t)) for t in tools]


def drop_meta_tools(tools: list, messages: list | None) -> tuple[list, list[str]]:
    """Drop OpenCode meta tools. Keep core, in-use, and everything else (including Godot)."""
    core = core_tool_names()
    in_use = names_in_messages(messages)
    keep: list = []
    dropped: list[str] = []
    for t in tools:
        n = tool_name(t).lower()
        is_meta = n in META_TOOLS or n.startswith("list_mcp_")
        if is_meta and n not in core and n not in in_use:
            dropped.append(tool_name(t))
        else:
            keep.append(t)
    if not keep and tools:
        keep = [tools[0]]
        dropped = [tool_name(t) for t in tools[1:]]
    return keep, dropped


def drop_extra_tools(tools: list, messages: list | None) -> tuple[list, list[str]]:
    """Keep core file tools plus MCP names already in tool_calls. Not mentioned families."""
    core = core_tool_names()
    in_use = names_in_messages(messages)
    keep: list = []
    dropped: list[str] = []
    for t in tools:
        n = tool_name(t).lower()
        if n in core or n in in_use:
            keep.append(t)
        else:
            dropped.append(tool_name(t))
    if not keep and tools:
        keep = [tools[0]]
        dropped = [tool_name(t) for t in tools[1:]]
    return keep, dropped


def ollama_think_flag(infer: dict | None, tools: list | None) -> bool | None:
    """Honor sealed infer.think (Granite execute uses true). Default off."""
    infer = infer if isinstance(infer, dict) else {}
    if "think" in infer:
        return bool(infer.get("think"))
    if tools:
        return False
    return False


def _msg_text(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def trim_tool_results(
    messages: list | None,
    *,
    keep_last: int = 4,
    cap_chars: int = 2000,
) -> list:
    """Cap older role=tool bodies. Last keep_last stay full. Does not drop rows or ids."""
    src = [deepcopy(m) if isinstance(m, dict) else m for m in (messages or [])]
    try:
        keep_n = max(0, int(keep_last))
    except (TypeError, ValueError):
        keep_n = 4
    try:
        cap = int(cap_chars)
    except (TypeError, ValueError):
        cap = 2000
    tool_idxs = [
        i
        for i, m in enumerate(src)
        if isinstance(m, dict) and str(m.get("role") or "").strip().lower() == "tool"
    ]
    protected = set(tool_idxs[-keep_n:]) if keep_n else set()
    for i in tool_idxs:
        if i in protected:
            continue
        msg = src[i]
        if not isinstance(msg, dict):
            continue
        text = _msg_text(msg)
        if cap <= 0:
            msg["content"] = STUB_TEXT
            continue
        if len(text) > cap:
            msg["content"] = text[:cap].rstrip() + TRUNCATED_SUFFIX
    return src


def next_ctx_rung(rung: str | None) -> str | None:
    a = (rung or "passthrough").strip().lower()
    if a in ("", "none"):
        a = "passthrough"
    try:
        i = CTX_RUNGS.index(a)
    except ValueError:
        return "compact"
    if i + 1 >= len(CTX_RUNGS):
        return None
    return CTX_RUNGS[i + 1]


def next_body_trim_rung(rung: str | None) -> str | None:
    """Skip schema-only rungs. Packed decode / invalid tool JSON need smaller file bodies."""
    a = (rung or "passthrough").strip().lower()
    if a in ("", "none", "passthrough", "compact", "drop_meta", "drop_extras"):
        return "cap_4_2000"
    return next_ctx_rung(a)


def tighten_after_overflow(action: str | None) -> str | None:
    """Next rung after Ollama exceed_context_size. Alias of next_ctx_rung."""
    return next_ctx_rung(action)


def _rung_trim_args(rung: str) -> tuple[int | None, int | None]:
    if rung == "cap_4_2000":
        return 4, 2000
    if rung == "cap_2_400":
        return 2, 400
    if rung == "stub":
        return 2, 0
    return None, None


def _path_from_tool_call(tc: dict) -> str:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    raw = fn.get("arguments") if isinstance(fn, dict) else None
    args: dict[str, Any] = {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            args = parsed
    elif isinstance(raw, dict):
        args = raw
    for key in ("filePath", "file_path", "path", "filepath"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def should_nudge_edit(messages: list | None) -> bool:
    """True when recent assistant calls only re-read files already in role=tool results."""
    reads: list[str] = []
    have_paths: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        if role == "tool":
            text = _msg_text(m)
            if text and text != STUB_TEXT:
                have_paths.add(str(m.get("tool_call_id") or ""))
            continue
        tcs = m.get("tool_calls")
        if role != "assistant" or not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str((fn or {}).get("name") or tc.get("name") or "").strip().lower()
            if name in ("edit", "write"):
                return False
            if name == "read":
                reads.append(_path_from_tool_call(tc) or str(tc.get("id") or ""))
    if len(reads) < 3:
        return False
    recent = [p for p in reads[-4:] if p]
    if len(recent) < 3:
        return False
    return len(set(recent)) <= 2


def with_coding_nudge(messages: list | None) -> list:
    src = list(messages or [])
    if not should_nudge_edit(src):
        return src
    return src + [{"role": "system", "content": CODING_NUDGE}]


def apply_ctx_rung(
    messages: list | None,
    tools: list | None,
    rung: str | None,
    *,
    num_ctx: int | None = None,
) -> tuple[list, list | None, dict[str, Any]]:
    """Apply one named rung. Returns (messages, tools, report). Never mutates inputs."""
    a = (rung or "passthrough").strip().lower()
    if a in ("", "none"):
        a = "passthrough"
    msgs: list = list(messages or [])
    tls = tools
    dropped: list[str] = []

    if tools and a != "passthrough":
        tls = compact_tools(tools)
    if tools and a in _META_DROP_RUNGS:
        tls, dropped = drop_meta_tools(tls if tls is not None else compact_tools(tools), messages)
    if tools and a in _EXTRA_DROP_RUNGS:
        extra_dropped: list[str] = []
        tls, extra_dropped = drop_extra_tools(
            tls if tls is not None else compact_tools(tools), messages
        )
        dropped = list(dropped) + extra_dropped

    keep_last, cap_chars = _rung_trim_args(a)
    if keep_last is not None:
        msgs = trim_tool_results(messages, keep_last=keep_last, cap_chars=int(cap_chars or 0))

    est = estimate_tokens(msgs) + estimate_tokens(tls or [])
    kept = [tool_name(t) for t in (tls or [])]
    return msgs, tls, {
        "action": a,
        "rung": a,
        "kept": kept,
        "dropped": dropped,
        "est": est,
        "budget": fit_target(num_ctx),
        "keep_last": keep_last,
        "cap_chars": cap_chars,
    }


def pick_first_rung(
    messages: list | None,
    tools: list | None,
    num_ctx: int | None,
    num_predict: int | None = None,
) -> str:
    """Lightest rung whose estimate fits. Never POST when every rung is over budget.

    Unused MCP is dropped on the first POST whenever that still fits — compact
    descriptions can look cheap while llama still packs 50 schemas. Named Godot
    families do not stay unless that exact tool was already called.
    """
    target = fit_target(num_ctx, num_predict)
    if not tools:
        candidates = ("passthrough", "cap_4_2000", "cap_2_400", "stub")
    else:
        _keep, unused = drop_extra_tools(list(tools), messages)
        del _keep
        if unused:
            candidates = ("drop_extras", "cap_4_2000", "cap_2_400", "stub")
        else:
            candidates = CTX_RUNGS
    for rung in candidates:
        _msgs, _tls, report = apply_ctx_rung(messages, tools, rung, num_ctx=num_ctx)
        try:
            est = int(report.get("est") or 0)
        except (TypeError, ValueError):
            est = 0
        if est <= target:
            return rung
    return UNFIT_RUNG


def prepare_ollama_payload(
    messages: list | None,
    tools: list | None,
    *,
    num_ctx: int | None,
    num_predict: int | None = None,
    rung: str | None = None,
) -> tuple[list, list | None, dict[str, Any]]:
    chosen = rung or pick_first_rung(messages, tools, num_ctx, num_predict)
    apply_name = "stub" if chosen == UNFIT_RUNG else chosen
    msgs, tls, report = apply_ctx_rung(messages, tools, apply_name, num_ctx=num_ctx)
    report = dict(report)
    if chosen == UNFIT_RUNG:
        report["rung"] = UNFIT_RUNG
        report["action"] = UNFIT_RUNG
        report["overflow"] = True
    nudged = with_coding_nudge(msgs)
    try:
        budget = int(report.get("budget") or 0)
    except (TypeError, ValueError):
        budget = 0
    if nudged is not msgs:
        nudged_est = estimate_tokens(nudged) + estimate_tokens(tls or [])
        if report.get("overflow") or nudged_est <= budget:
            msgs = nudged
            report["est"] = nudged_est
    try:
        est = int(report.get("est") or 0)
    except (TypeError, ValueError):
        est = 0
    if est > budget:
        report["overflow"] = True
    return msgs, tls, report


def budget_tools(
    messages: list | None,
    tools: list | None,
    *,
    num_ctx: int | None,
    num_predict: int | None = None,
    force: str | None = None,
) -> tuple[list | None, dict[str, Any]]:
    """Return (tools_for_ollama, report). force=drop_extras keeps core + in-use names only."""
    if not tools:
        return tools, {"action": "none", "kept": [], "dropped": [], "est": 0, "budget": 0, "rung": "none"}
    _msgs, tls, report = prepare_ollama_payload(
        messages, tools, num_ctx=num_ctx, num_predict=num_predict, rung=force
    )
    return tls, report
