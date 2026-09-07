"""Map sealed OpenAI messages[] to Ollama /api/chat without dropping tool metadata.

OpenCode resends assistant.tool_calls[].id and role=tool.tool_call_id every turn.
Ollama needs tool_name plus object-shaped arguments. This module keeps both.
"""
from __future__ import annotations

import json
import re
from typing import Any


def flatten_content(content: Any) -> str:
    """OpenCode may send content as a list of parts; never use Python repr()."""
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
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") == "input_text":
                    parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _parse_args(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
    if args is None:
        return {}
    return args


def _fn_name(tc: dict) -> str:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    return str(fn.get("name") or tc.get("name") or "").strip()


def fingerprint_messages_wire(messages: list | None) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        tcs = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
        ids: list[str] = []
        names: list[str] = []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            ids.append(str(tc.get("id") or ""))
            names.append(_fn_name(tc))
        items.append(
            {
                "role": m.get("role"),
                "tool_call_id": m.get("tool_call_id"),
                "tool_name": m.get("tool_name") or m.get("name"),
                "content_chars": len(flatten_content(m.get("content"))),
                "tc_n": len(tcs),
                "tc_ids": ",".join(x for x in ids if x),
                "tc_names": ",".join(x for x in names if x),
            }
        )
    return {"msg_n": len(items), "items": items[:40]}


def format_messages_wire(messages: list | None) -> str:
    parts: list[str] = []
    for item in fingerprint_messages_wire(messages).get("items") or []:
        role = str(item.get("role") or "?")
        extra: list[str] = []
        if item.get("tc_ids"):
            extra.append(f"tc={item['tc_ids']}")
        if item.get("tc_names"):
            extra.append(f"fn={item['tc_names']}")
        if item.get("tool_call_id"):
            extra.append(f"id={item['tool_call_id']}")
        if item.get("tool_name"):
            extra.append(f"name={item['tool_name']}")
        parts.append(f"{role}({','.join(extra)})" if extra else role)
    return "|".join(parts[:40]) or "-"


def _index_tool_call_names(messages: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        tcs = m.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tid = str(tc.get("id") or "")
            name = _fn_name(tc)
            if tid and name:
                out[tid] = name
    return out


def messages_for_ollama(messages: list, *, native: bool = False) -> list[dict[str, Any]]:
    """
    OpenAI-shaped claim → Ollama /api/chat messages.

    Default keeps tool_call_id and tool_calls[].id (OpenAI/OpenCode) AND
    tool_name + object arguments (Ollama). native=True strips OpenAI extras
    for a retry if a given Ollama build 400s on unknown fields — tools[] stay.
    """
    id_to_name = _index_tool_call_names(messages)
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "user").strip().lower()
        item: dict[str, Any] = {"role": role}
        item["content"] = flatten_content(m.get("content"))

        tcs = m.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            ollama_tcs: list[dict[str, Any]] = []
            for i, tc in enumerate(tcs):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = str(fn.get("name") or tc.get("name") or "")
                args = fn.get("arguments")
                if args is None:
                    args = tc.get("arguments")
                entry_fn = {"name": name, "arguments": _parse_args(args)}
                if native:
                    ollama_tcs.append({"function": entry_fn})
                else:
                    ollama_tcs.append(
                        {
                            "id": str(tc.get("id") or f"call_{i + 1}"),
                            "type": str(tc.get("type") or "function"),
                            "function": entry_fn,
                        }
                    )
            if ollama_tcs:
                item["tool_calls"] = ollama_tcs

        if role == "tool":
            call_id = m.get("tool_call_id") or m.get("toolCallId")
            tool_name = m.get("tool_name") or m.get("name") or id_to_name.get(str(call_id or ""))
            if tool_name:
                item["tool_name"] = str(tool_name)
            if call_id and not native:
                item["tool_call_id"] = str(call_id)

        out.append(item)
    return out


def ollama_tool_calls_to_openai(raw_calls: list | None) -> list[dict[str, Any]]:
    """Ollama tool_calls (object args, optional id) → OpenAI (string args, stable id)."""
    tool_calls: list[dict[str, Any]] = []
    for i, tc in enumerate(raw_calls or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(fn.get("name") or tc.get("name") or "")
        args = fn.get("arguments")
        if args is None:
            args = tc.get("arguments")
        if isinstance(args, dict):
            args_s = json.dumps(args, ensure_ascii=False)
        elif isinstance(args, str):
            args_s = args
        else:
            args_s = str(args or "{}")
        tool_calls.append(
            {
                "id": str(tc.get("id") or f"call_{i + 1}"),
                "type": str(tc.get("type") or "function"),
                "function": {"name": name, "arguments": args_s},
            }
        )
    return tool_calls


def wire_loss_report(inbound: list, mapped: list) -> dict[str, Any]:
    """Compare OpenCode-shaped messages vs what would be posted to Ollama."""
    losses: list[str] = []
    in_fp = fingerprint_messages_wire(inbound)
    out_fp = fingerprint_messages_wire(mapped)
    in_items = in_fp.get("items") or []
    out_items = out_fp.get("items") or []
    if len(in_items) != len(out_items):
        losses.append(f"msg_count {len(in_items)} -> {len(out_items)}")
    n = min(len(in_items), len(out_items))
    for i in range(n):
        a, b = in_items[i], out_items[i]
        role = a.get("role")
        if role == "assistant" and a.get("tc_ids") and a.get("tc_ids") != b.get("tc_ids"):
            losses.append(f"msg[{i}] tool_calls.id lost: {a.get('tc_ids')!r} -> {b.get('tc_ids')!r}")
        if role == "assistant" and a.get("tc_names") and a.get("tc_names") != b.get("tc_names"):
            losses.append(f"msg[{i}] tool_calls.name lost: {a.get('tc_names')!r} -> {b.get('tc_names')!r}")
        if role == "tool" and a.get("tool_call_id") and a.get("tool_call_id") != b.get("tool_call_id"):
            losses.append(
                f"msg[{i}] tool_call_id lost: {a.get('tool_call_id')!r} -> {b.get('tool_call_id')!r}"
            )
        if role == "tool" and not b.get("tool_name"):
            losses.append(f"msg[{i}] tool_name missing for Ollama (id={a.get('tool_call_id')})")
        if a.get("content_chars") and not b.get("content_chars") and role != "assistant":
            losses.append(f"msg[{i}] content dropped ({a.get('content_chars')} chars)")
        # Python repr of a list looks like "[{'type':..."
        if isinstance(inbound[i].get("content") if i < len(inbound) else None, list):
            mapped_c = mapped[i].get("content") if i < len(mapped) else None
            if isinstance(mapped_c, str) and mapped_c.startswith("[{"):
                losses.append(f"msg[{i}] list content became Python repr")
    return {
        "ok": not losses,
        "losses": losses,
        "inbound_wire": format_messages_wire(inbound),
        "mapped_wire": format_messages_wire(mapped),
        "inbound": in_fp,
        "mapped": out_fp,
    }


_THINK_BLOCK = re.compile(r"<\s*think\s*>(.*?)</\s*think\s*>", re.IGNORECASE | re.DOTALL)


def visible_content(content: Any, thinking: Any = None) -> str:
    """Text outside <think> tags. Empty means think-only or blank."""
    raw = flatten_content(content)
    visible = _THINK_BLOCK.sub("", raw).strip()
    if visible:
        return visible
    del thinking
    return ""


def finish_from_ollama(data: dict | None, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    dr = str((data or {}).get("done_reason") or (data or {}).get("finish_reason") or "").lower()
    if dr in ("length", "max_tokens"):
        return "length"
    return "stop"
