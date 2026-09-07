"""AES-256-GCM envelopes for orchestrator ↔ node HTTP.

Jobs are already zip+AES at rest (JobCrypto). This module re-wraps the claim
and answer bodies so prompts are not plaintext on the node hop — including
local HTTP without TLS. Both sides already share the node's API token
(Bearer); a per-job key is derived with HKDF.

Ollama still needs readable text on the node. This is transport protection,
not end-to-end confidentiality against the volunteer operator.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VERSION = "v1"
NONCE_LEN = 12
KEY_LEN = 32
MAX_COMPRESSED = 20 * 1024 * 1024
_SALT = b"localllmbay-node-transport"


class WireCryptoError(ValueError):
    """Envelope missing, truncated, or not authentic for this token/job."""


def _hkdf_key(api_token: str, job_id: str, purpose: str) -> bytes:
    if not api_token or not job_id or not purpose:
        raise WireCryptoError("missing_transport_secret")
    info = f"localllmbay-wire|{VERSION}|{purpose}|{job_id}".encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=_SALT,
        info=info,
    ).derive(api_token.encode("utf-8"))


def _aad(purpose: str, job_id: str) -> bytes:
    return f"{VERSION}|{purpose}|{job_id}".encode("utf-8")


def encode_payload(
    obj: dict[str, Any],
    *,
    api_token: str,
    job_id: str,
    purpose: str,
    max_compressed: int = MAX_COMPRESSED,
) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    zipped = gzip.compress(raw, compresslevel=6)
    if len(zipped) > max_compressed:
        raise WireCryptoError("compressed_payload_too_large")
    nonce = os.urandom(NONCE_LEN)
    job_id = str(job_id)
    ct = AESGCM(_hkdf_key(api_token, job_id, purpose)).encrypt(
        nonce, zipped, _aad(purpose, job_id)
    )
    blob = nonce + ct
    return VERSION + "." + base64.urlsafe_b64encode(blob).decode("ascii")


def decode_payload(
    token: str,
    *,
    api_token: str,
    job_id: str,
    purpose: str,
) -> dict[str, Any]:
    if not isinstance(token, str) or "." not in token:
        raise WireCryptoError("malformed_envelope")
    ver, b64 = token.split(".", 1)
    if ver != VERSION:
        raise WireCryptoError("unsupported_envelope")
    try:
        blob = base64.urlsafe_b64decode(b64.encode("ascii"))
    except Exception as e:
        raise WireCryptoError("malformed_envelope") from e
    if len(blob) < NONCE_LEN + 16:
        raise WireCryptoError("malformed_envelope")
    nonce, ct = blob[:NONCE_LEN], blob[NONCE_LEN:]
    job_id = str(job_id)
    try:
        zipped = AESGCM(_hkdf_key(api_token, job_id, purpose)).decrypt(
            nonce, ct, _aad(purpose, job_id)
        )
    except Exception as e:
        raise WireCryptoError("decrypt_failed") from e
    try:
        data = json.loads(gzip.decompress(zipped).decode("utf-8"))
    except Exception as e:
        raise WireCryptoError("malformed_envelope") from e
    if not isinstance(data, dict):
        raise WireCryptoError("malformed_envelope")
    return data
