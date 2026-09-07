"""Hermes/OpenAI tool_call parsing shared by orchestrator and node.

Keep server/tool_parse.py and node/tool_parse.py byte-identical (same rule as wire_crypto).
"""
from __future__ import annotations

import json
import re
from typing import Any

def _tool_args_to_str(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args if args is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _loads_jsonish(text: str) -> Any:
    blob = (text or "").strip()
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        try:
            parsed, _end = json.JSONDecoder().raw_decode(blob)
            return parsed
        except json.JSONDecodeError:
            return None


def close_truncated_json(text: str) -> Any | None:
    """Valid JSON only. Do not invent closing braces for a cut tool command."""
    return _loads_jsonish(text)


def is_invalid_tool_json_error(status: int | None, body: str | None) -> bool:
    if int(status or 0) != 500:
        return False
    blob = (body or "").lower()
    return "invalid tool call arguments" in blob or "unexpected end of json input" in blob


def _hermes_parsed_to_candidate(name: str, parsed: Any) -> list[Any]:
    if isinstance(parsed, dict):
        if any(k in parsed for k in ("arguments", "parameters", "args")):
            if not str(parsed.get("name") or parsed.get("tool") or "").strip():
                parsed = {**parsed, "name": name}
            return [parsed]
        return [{"name": name, "arguments": parsed}]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def _function_xml_candidates(text: str) -> list[Any]:
    """Hermes/Qwen <function=name>…</function> — parameter tags or a JSON body.

    Models often emit ``<function=read>{"path": "a.py"}</function>``. Parameter-only
    parsing used to promote that as Read with empty args, so OpenCode never opened
    the file.
    """
    out: list[Any] = []
    for m in re.finditer(
        r"<function=([^\s>]+)>\s*(.*?)\s*</function>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        name = m.group(1).strip()
        if not name:
            continue
        inner = (m.group(2) or "").strip()
        params: dict[str, Any] = {}
        for pm in re.finditer(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            inner,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            params[pm.group(1).strip()] = pm.group(2).strip()
        if params:
            out.append({"name": name, "arguments": params})
            continue
        parsed = _loads_jsonish(inner)
        if parsed is None:
            continue
        out.extend(_hermes_parsed_to_candidate(name, parsed))
    if out:
        return out
    # num_predict often cuts </function>; still take a JSON body after the open tag.
    m = re.search(r"<function=([^\s>]+)>\s*", text, flags=re.IGNORECASE)
    if not m:
        return out
    name = m.group(1).strip()
    if not name:
        return out
    parsed = _loads_jsonish(text[m.end() :])
    if parsed is None:
        return out
    out.extend(_hermes_parsed_to_candidate(name, parsed))
    return out


_PATH_ALIASES = (
    "file_path",
    "filepath",
    "filePath",
    "file",
    "filename",
    "target",
    "relativePath",
    "relative_path",
    "uri",
    "src",
    "source",
    "dest",
    "destination",
    "output",
    "target_file",
    "source_file",
)
_PATH_CANON = ("path", "filePath", "file_path")
_CONTENT_ALIASES = ("contents", "body", "text")
_CONTENT_CANON = ("content", "contents")
_EDIT_OLD_KEYS = ("oldString", "old_string", "oldText", "old_text", "old")
_EDIT_NEW_KEYS = ("newString", "new_string", "newText", "new_text", "new")
_GREP_PATTERN_KEYS = ("pattern", "query", "search", "regex")
_GLOB_PATTERN_KEYS = ("pattern", "glob", "glob_pattern", "query")
_BASH_KEYS = ("command", "cmd", "script")
_FETCH_KEYS = ("url", "uri", "href")
_FILE_TOOLS = frozenset({"read", "write", "edit", "glob", "grep", "list"})


def _as_path_str(val: Any) -> str | None:
    if isinstance(val, str):
        s = val.strip()
        return s or None
    if isinstance(val, dict):
        return _first_str(val, _PATH_CANON + _PATH_ALIASES, strip=True)
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return _as_path_str(val[0])
    return None


def _first_str(
    args: dict[str, Any],
    keys: tuple[str, ...],
    *,
    strip: bool = False,
    allow_empty: bool = False,
) -> str | None:
    empty: str | None = None
    for key in keys:
        val = args.get(key)
        if isinstance(val, str):
            if strip:
                val = val.strip()
            if val:
                return val
            if allow_empty and empty is None:
                empty = val
            continue
        nested = _as_path_str(val) if key in _PATH_CANON + _PATH_ALIASES else None
        if nested:
            return nested
    return empty if allow_empty else None


def _tc_args(tc: dict[str, Any]) -> dict[str, Any] | None:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
    if not fn:
        return None
    raw = fn.get("arguments")
    if isinstance(raw, str):
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        args = dict(raw)
    else:
        return None
    return args if isinstance(args, dict) else None


def _fill_str_keys(
    args: dict[str, Any],
    dests: tuple[str, ...],
    sources: tuple[str, ...],
    *,
    strip: bool = False,
    allow_empty: bool = False,
) -> None:
    val = _first_str(args, sources, strip=strip, allow_empty=allow_empty)
    if val is None:
        return
    for dest in dests:
        cur = args.get(dest)
        if not (isinstance(cur, str) and (cur.strip() if strip else cur)):
            args[dest] = val


def _has_nonempty(args: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(isinstance(args.get(k), str) and str(args.get(k)).strip() for k in keys)


def _has_str_key(args: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(isinstance(args.get(k), str) for k in keys)


_LIST_BASH = re.compile(
    r"\b(ls|dir|find|grep|Get-ChildItem|gci)\b|\|\s*grep\b",
    re.IGNORECASE,
)
_KEEP_BASH = re.compile(
    r"\.(bat|cmd|sh|ps1)\b|\bnpm\b|\bnpx\b|\bng\b|\bpython\b|\bpytest\b|"
    r"\bdocker\b|\bgodot\b|\bpip\b|\bcargo\b",
    re.IGNORECASE,
)
_STAR_GLOB = re.compile(r"(?:\*\*/)?\*(?:\{\s*[\w.,]+\s*\}|\.\w+)")
DEFAULT_LIST_GLOB = "**/*.{gd,tscn,js,ts,py}"


def _can_target(name: str, allowed: set[str] | None) -> bool:
    if allowed is None:
        return True
    return str(name).strip().lower() in allowed


def _rewrite_fragile_tool(
    name: str,
    args: dict[str, Any],
    fn: dict[str, Any],
    allowed: set[str] | None,
) -> dict[str, Any]:
    """Unix ls|grep and empty Edit never reach OpenCode (PowerShell has no grep)."""
    if name == "bash":
        cmd = str(args.get("command") or args.get("cmd") or "")
        if (
            cmd.strip()
            and not _KEEP_BASH.search(cmd)
            and _LIST_BASH.search(cmd)
            and _can_target("glob", allowed)
        ):
            found = _STAR_GLOB.search(cmd.replace(" ", ""))
            if not found:
                found = _STAR_GLOB.search(cmd)
            pattern = found.group(0) if found else DEFAULT_LIST_GLOB
            fn["name"] = "glob"
            return {"pattern": pattern}
    if name == "edit":
        has_path = _has_nonempty(args, _PATH_CANON + _PATH_ALIASES)
        has_old = _has_nonempty(args, _EDIT_OLD_KEYS)
        has_body = _has_str_key(args, _CONTENT_CANON + _EDIT_NEW_KEYS)
        if has_path and not has_old and not has_body and _can_target("read", allowed):
            fn["name"] = "read"
            return args
        if not has_path and not has_old and not has_body and _can_target("glob", allowed):
            fn["name"] = "glob"
            return {"pattern": DEFAULT_LIST_GLOB}
    return args


def _normalize_file_tool_args(
    tc: dict[str, Any],
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    """Mirror slop onto OpenCode keys (filePath/content/oldString) and path/content tests."""
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
    if not fn:
        return tc
    name = str(fn.get("name") or "").strip().lower()
    args = _tc_args(tc)
    if args is None:
        return tc
    if name in _FILE_TOOLS:
        _fill_str_keys(args, _PATH_CANON, ("path",) + _PATH_ALIASES, strip=True)
        # OpenCode Edit requires oldString. A full-file body with no old is Write.
        if name == "edit" and not _has_nonempty(args, _EDIT_OLD_KEYS) and _has_str_key(
            args, _CONTENT_CANON + _EDIT_NEW_KEYS
        ):
            fn["name"] = "write"
            name = "write"
            _fill_str_keys(
                args,
                _CONTENT_CANON,
                ("content",) + _CONTENT_ALIASES + _EDIT_NEW_KEYS,
                strip=False,
                allow_empty=True,
            )
        if name == "write":
            _fill_str_keys(
                args, _CONTENT_CANON, ("content",) + _CONTENT_ALIASES, strip=False, allow_empty=True
            )
        elif name == "edit":
            _fill_str_keys(args, _EDIT_OLD_KEYS, _EDIT_OLD_KEYS, strip=False)
            _fill_str_keys(args, _EDIT_NEW_KEYS, _EDIT_NEW_KEYS, strip=False, allow_empty=True)
            if not _has_str_key(args, _EDIT_NEW_KEYS):
                _fill_str_keys(args, _EDIT_NEW_KEYS, _CONTENT_CANON, strip=False, allow_empty=True)
        elif name == "grep":
            _fill_str_keys(args, ("pattern",), _GREP_PATTERN_KEYS, strip=True)
        elif name == "glob":
            _fill_str_keys(args, ("pattern",), _GLOB_PATTERN_KEYS, strip=True)
    elif name == "bash":
        _fill_str_keys(args, ("command",), _BASH_KEYS, strip=False)
    elif name == "webfetch":
        _fill_str_keys(args, ("url",), _FETCH_KEYS, strip=True)
    name = str(fn.get("name") or "").strip().lower()
    args = _rewrite_fragile_tool(name, args, fn, allowed)
    fn["arguments"] = _tool_args_to_str(args)
    return tc


def apply_fallback_path(tc: dict[str, Any], path: str) -> dict[str, Any]:
    """Fill filePath/path when hop JSON omitted them but hop 2 listed the file."""
    path = str(path or "").strip()
    if not path:
        return tc
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
    if not fn:
        return tc
    args = _tc_args(tc)
    if args is None:
        args = {}
    if _has_nonempty(args, _PATH_CANON + _PATH_ALIASES):
        return tc
    args["filePath"] = path
    args["path"] = path
    fn["arguments"] = _tool_args_to_str(args)
    return _normalize_file_tool_args(tc)


def _file_tool_args_ok(tc: dict[str, Any]) -> bool:
    """False when OpenCode would SchemaError or no-op (empty path/pattern/content)."""
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
    if not fn:
        return True
    name = str(fn.get("name") or "").strip().lower()
    args = _tc_args(tc)
    if args is None:
        return False
    if name in ("grep", "glob"):
        return _has_nonempty(args, _GREP_PATTERN_KEYS + _GLOB_PATTERN_KEYS)
    if name == "write":
        return _has_nonempty(args, _PATH_CANON) and _has_str_key(args, _CONTENT_CANON)
    if name == "edit":
        return (
            _has_nonempty(args, _PATH_CANON)
            and _has_nonempty(args, _EDIT_OLD_KEYS)
            and _has_str_key(args, _EDIT_NEW_KEYS)
        )
    if name == "read":
        return _has_nonempty(args, _PATH_CANON)
    if name == "bash":
        return _has_nonempty(args, _BASH_KEYS)
    if name == "webfetch":
        return _has_nonempty(args, _FETCH_KEYS)
    if name == "question":
        return _has_nonempty(args, ("prompt", "question", "text"))
    if name == "skill":
        return _has_nonempty(args, ("name", "skill", "id"))
    if name == "task":
        return _has_nonempty(args, ("prompt", "description", "task"))
    if name == "todowrite":
        todos = args.get("todos")
        return isinstance(todos, list) and len(todos) > 0
    if name in _FILE_TOOLS:
        return _has_nonempty(args, _PATH_CANON)
    return True


def _dict_as_tool_call(obj: dict, idx: int = 0) -> dict[str, Any] | None:
    """Accept OpenAI-ish or bare {name, arguments|parameters} objects."""
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = str(fn.get("name") or "").strip()
        if not name:
            return None
        return {
            "id": str(obj.get("id") or f"call_{idx + 1}"),
            "type": "function",
            "function": {"name": name, "arguments": _tool_args_to_str(fn.get("arguments"))},
        }
    name = str(obj.get("name") or obj.get("tool") or "").strip()
    if not name:
        fn = obj.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
            if name:
                return {
                    "id": str(obj.get("id") or f"call_{idx + 1}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _tool_args_to_str(fn.get("arguments") or obj.get("arguments")),
                    },
                }
        return None
    has_args = any(k in obj for k in ("arguments", "parameters", "args"))
    if not has_args and not isinstance(obj.get("function"), dict):
        extra = {k: v for k, v in obj.items() if k not in ("name", "tool", "id", "type")}
        if not extra:
            return None
        return {
            "id": str(obj.get("id") or f"call_{idx + 1}"),
            "type": "function",
            "function": {"name": name, "arguments": _tool_args_to_str(extra)},
        }
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = obj.get("args")
    if args is None and isinstance(obj.get("function"), dict):
        args = obj["function"].get("arguments")
    return {
        "id": str(obj.get("id") or f"call_{idx + 1}"),
        "type": "function",
        "function": {"name": name, "arguments": _tool_args_to_str(args)},
    }

def promote_content_tool_calls(
    content: str | None,
    *,
    allowed_names: set[str] | None = None,
    tools_requested: bool = False,
) -> list[dict[str, Any]] | None:
    """
    Qwen/Ollama often emit bare tool JSON (or <tool_call> XML) in content instead of
    message.tool_calls. Promote into OpenAI tool_calls so OpenCode can run tools.
    Stateless — pure string transform for one job answer.
    """
    if not content or not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    candidates: list[Any] = []

    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())
    candidates.extend(_function_xml_candidates(text))

    if not candidates:
        try:
            candidates.append(json.loads(text))
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    candidates.append(json.loads(text[start : end + 1]))
                except json.JSONDecodeError:
                    pass
            a0, a1 = text.find("["), text.rfind("]")
            if a0 >= 0 and a1 > a0:
                try:
                    candidates.append(json.loads(text[a0 : a1 + 1]))
                except json.JSONDecodeError:
                    pass

    raw_objs: list[dict] = []
    for c in candidates:
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except json.JSONDecodeError:
                continue
        if isinstance(c, list):
            raw_objs.extend(x for x in c if isinstance(x, dict))
        elif isinstance(c, dict):
            inner = c.get("tool_calls")
            if isinstance(inner, list):
                raw_objs.extend(x for x in inner if isinstance(x, dict))
            else:
                raw_objs.append(c)

    if not raw_objs:
        return None

    lower_map = {n.lower(): n for n in (allowed_names or set())}
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(raw_objs):
        tc = _dict_as_tool_call(obj, i)
        if not tc:
            continue
        name = tc["function"]["name"]
        if lower_map:
            canon = lower_map.get(name.lower())
            if canon:
                tc["function"]["name"] = canon
            elif not tools_requested:
                continue
        out.append(_normalize_file_tool_args(tc))

    out = [tc for tc in out if _file_tool_args_ok(tc)]
    return out or None

normalize_file_tool_args = _normalize_file_tool_args
file_tool_args_ok = _file_tool_args_ok
