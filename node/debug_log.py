"""Plaintext debug dumps and short job summaries for the contribution node.

Does not raise Ollama's log level. Opt out of full prompt dumps with
QUEUE_PLAINTEXT_LOGS=0. Forced off when API_BASE is the public LocalLLMBay host.
Job lifecycle lines (claim / infer / result) stay on; they only include
prompt snippets when plaintext dumps are enabled.
"""
from __future__ import annotations

import json
import re
from typing import Any

_TOKEN_RE = re.compile(r"(?i)\b(localllmbay_live_|neuronbay_live_)[A-Za-z0-9._\-]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+")
_B64_RE = re.compile(r"^(?:[A-Za-z0-9+/]{80,}={0,2})$")
_DATA_URL_RE = re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/=\s]{80,}")

_IMAGE_KEYS = frozenset(
    {"image", "images", "b64_json", "b64", "png", "jpeg", "jpg", "webp"}
)


def plaintext_logs_enabled(api_base: str, raw: str | None) -> bool:
    base = (api_base or "").lower()
    if "localllmbay.com" in base or "neuronbay.com" in base:
        return False
    val = (raw or "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


def scrub_secrets(text: str | None) -> str:
    s = text if isinstance(text, str) else str(text or "")
    s = _TOKEN_RE.sub(r"\1<redacted>", s)
    s = _BEARER_RE.sub("Bearer <redacted>", s)
    return s


def _looks_b64(s: str) -> bool:
    t = "".join(s.split())
    if len(t) < 80:
        return False
    return bool(_B64_RE.match(t))


def collapse_for_log(obj: Any, *, _depth: int = 0) -> Any:
    """Keep prompts; replace huge base64 / image blobs so dumps stay sendable."""
    if _depth > 24:
        return "<max-depth>"
    if isinstance(obj, bytes):
        return f"<bytes {len(obj)}>"
    if isinstance(obj, str):
        s = _DATA_URL_RE.sub(lambda m: f"<data-url {len(m.group(0))} chars>", obj)
        if _looks_b64(s):
            return f"<b64 {len(s)} chars>"
        return s
    if isinstance(obj, list):
        if obj and all(isinstance(x, str) and _looks_b64(x) for x in obj):
            return [f"<b64 {len(x)} chars>" for x in obj]
        return [collapse_for_log(x, _depth=_depth + 1) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            low = key.lower()
            if low in _IMAGE_KEYS and isinstance(v, str) and len(v) > 80:
                out[key] = f"<b64 {len(v)} chars>"
            elif low in _IMAGE_KEYS and isinstance(v, list):
                out[key] = collapse_for_log(v, _depth=_depth + 1)
            else:
                out[key] = collapse_for_log(v, _depth=_depth + 1)
        return out
    return obj


def to_log_json(obj: Any) -> str:
    try:
        return json.dumps(collapse_for_log(obj), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return scrub_secrets(repr(obj))


def _content_text(content: Any) -> str:
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
                if "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") in ("image_url", "image"):
                    parts.append("<image>")
                else:
                    parts.append(to_log_json(collapse_for_log(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        return to_log_json(collapse_for_log(content))
    return str(content)


def format_messages_transcript(messages: list | None) -> str:
    if not messages:
        return "(no messages)"
    parts: list[str] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            parts.append(f"--- {i} ? ---")
            parts.append(repr(m))
            continue
        role = str(m.get("role") or "?")
        text = _content_text(m.get("content"))
        thinking = m.get("thinking")
        parts.append(f"--- {i} {role} ({len(text)} chars) ---")
        if text:
            parts.append(text)
        elif m.get("tool_calls"):
            parts.append("(content empty; tool_calls below)")
        else:
            parts.append("(empty)")
        if thinking:
            t = str(thinking)
            parts.append(f"--- {i} {role} thinking ({len(t)} chars) ---")
            parts.append(t)
        if m.get("tool_calls"):
            parts.append("tool_calls=" + to_log_json(m.get("tool_calls")))
        if m.get("tool_call_id"):
            parts.append(f"tool_call_id={m.get('tool_call_id')} name={m.get('name') or ''}")
        extra = {k: v for k, v in m.items() if k not in ("role", "content", "thinking", "tool_calls", "tool_call_id", "name")}
        if extra:
            parts.append("extra=" + to_log_json(extra))
    return "\n".join(parts)


def last_user_text(messages: list | None) -> str:
    last = ""
    for m in messages or []:
        if isinstance(m, dict) and str(m.get("role") or "") == "user":
            last = _content_text(m.get("content"))
    return last


def text_snippet(text: str | None, limit: int = 180) -> str:
    s = scrub_secrets(str(text or "")).replace("\r", " ").replace("\n", " ")
    s = re.sub(r" {2,}", " ", s).strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def last_user_log_field(
    messages: list | None,
    *,
    plaintext: bool,
    fallback: str = "",
    limit: int = 180,
) -> str:
    raw = last_user_text(messages) or (fallback or "")
    n = len(raw or "")
    if not plaintext:
        return f"last_user_chars={n}"
    return f"last_user_chars={n} last_user={json.dumps(text_snippet(raw, limit), ensure_ascii=False)}"


def summarize_assistant_output(
    content: Any,
    tool_calls: Any = None,
    *,
    limit: int = 220,
    detail: bool = True,
) -> str:
    names: list[str] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str((fn or {}).get("name") or tc.get("name") or "").strip()
        if name:
            names.append(name)
    if names:
        return text_snippet("tool_calls=" + ",".join(names), limit)
    raw = content if isinstance(content, str) else _content_text(content)
    s = (raw or "").strip()
    if s.startswith("{") and "}" in s:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            action = str(obj.get("action") or "").strip()
            if action == "tool_call":
                if not detail:
                    return text_snippet(f"action=tool_call tool={obj.get('tool')}", limit)
                args = obj.get("arguments")
                arg_s = to_log_json(args) if args is not None else "{}"
                return text_snippet(f"action=tool_call tool={obj.get('tool')} args={arg_s}", limit)
            if action == "final":
                if not detail:
                    return "action=final"
                return text_snippet(f"action=final message={obj.get('message')}", limit)
            if action == "coding":
                if not detail:
                    return "action=coding"
                return text_snippet(f"action=coding objective={obj.get('objective')}", limit)
            if obj.get("implementation"):
                if not detail:
                    return "sketch"
                files = obj.get("files_changed") or obj.get("files") or ""
                return text_snippet(f"sketch files={files} impl={obj.get('implementation')}", limit)
    if not s:
        return "(empty)"
    if not detail:
        return f"chars={len(s)}"
    return text_snippet(s, limit)
