"""see_bench (M10): on-demand workbench part identification via a 2nd camera.

Narrow by design (the council's kill list): this is NOT an always-on
recognition pipeline. The bench camera is opened per request by the vision
sidecar (``GET /bench.jpg``) and released immediately — nothing streams, for
both privacy and VRAM. The frame is center-cropped, sent to moondream with an
electronics-biased prompt, and (best-effort) OCR'd with Tesseract; marking
tokens that look like part numbers are merged into the spoken answer
("looks like an AMS1117-3.3 voltage regulator").

Optional inventory: "log this part" appends {timestamp, label, ocr text} to
a ``bench_inventory`` table, and "do I have any 10k resistors" searches it
semantically with the same local embeddings stack facts use (plain substring
fallback when embeddings are off).

Everything external is injectable (frame fetch, vision model, OCR) so the
skill is fully testable without a camera, Ollama, or Tesseract installed.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

BENCH_PROMPT = (
    "This photo shows an item on an electronics workbench. Identify the "
    "electronic component or part in one short spoken sentence, and read any "
    "part markings you can see. Plain text."
)

#: Part-number-shaped tokens: uppercase alphanumerics with at least one digit.
_PART_TOKEN = re.compile(r"\b[A-Z0-9]{3,}(?:[-.][A-Z0-9.]+)*\b")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bench_inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    label TEXT NOT NULL,
    ocr_text TEXT NOT NULL DEFAULT '',
    embedding BLOB
);
"""


def extract_markings(ocr_text: str, limit: int = 3) -> list[str]:
    """Part-number-looking tokens from messy OCR output. Pure, unit-tested.

    A part number mixes letters and digits (AMS1117-3.3, STM32F103C8T6);
    all-digit tokens are dates/lot codes ("2024") and all-letter ones are
    words ("TAIWAN", "ROHS") — both rejected.
    """
    found: list[str] = []
    for token in _PART_TOKEN.findall(ocr_text.upper()):
        if (len(token) >= 4 and any(c.isdigit() for c in token)
                and any(c.isalpha() for c in token) and token not in found):
            found.append(token)
    return found[:limit]


def center_crop_jpeg(jpeg: bytes, keep: float = 0.7) -> bytes:
    """Middle ``keep`` fraction of the frame — the part sits under the camera."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg))
        w, h = img.size
        dx, dy = int(w * (1 - keep) / 2), int(h * (1 - keep) / 2)
        out = io.BytesIO()
        img.crop((dx, dy, w - dx, h - dy)).save(out, format="JPEG")
        return out.getvalue()
    except Exception:  # cropping is an optimization, never a failure
        return jpeg


def default_ocr(jpeg: bytes) -> str:
    """Tesseract OCR, best-effort: '' when the engine/wrapper is missing."""
    try:
        import pytesseract
        from PIL import Image

        # pytesseract defaults to whatever's on PATH — but a GUI-launched app
        # (run.command from Finder, run.bat) doesn't inherit the shell PATH, so
        # Homebrew/Windows installs go unseen and OCR silently returns "".
        # Probe the usual install locations for THIS OS as well.
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",   # Windows
            "/opt/homebrew/bin/tesseract",                     # macOS (Apple Silicon)
            "/usr/local/bin/tesseract",                        # macOS (Intel) / Linux
            "/usr/bin/tesseract",                              # Linux
        ):
            exe = Path(candidate)
            if exe.exists():
                pytesseract.pytesseract.tesseract_cmd = str(exe)
                break
        return pytesseract.image_to_string(Image.open(io.BytesIO(jpeg))) or ""
    except Exception:
        logger.debug("tesseract unavailable; skipping OCR", exc_info=True)
        return ""


class BenchInventory:
    """Sqlite log of identified parts with optional semantic recall."""

    def __init__(self, db_path: str | Path, embedder=None) -> None:
        self._db_path = str(db_path)
        self._embedder = embedder

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.executescript(_SCHEMA)
        return conn

    def add(self, label: str, ocr_text: str = "") -> None:
        blob = None
        if self._embedder is not None:
            vectors = self._embedder([f"{label} {ocr_text}".strip()])
            if vectors:
                blob = vectors[0].tobytes()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bench_inventory (ts, label, ocr_text, embedding)"
                " VALUES (?, ?, ?, ?)",
                (time.time(), label, ocr_text, blob),
            )

    def search(self, query: str, k: int = 3) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT label, ocr_text, embedding FROM bench_inventory").fetchall()
        if not rows:
            return []
        if self._embedder is not None:
            qvec = self._embedder([query])
            vectors = [r[2] for r in rows]
            if qvec and all(v is not None for v in vectors):
                import numpy as np

                from core.embeddings import rank_by_similarity

                order = rank_by_similarity(
                    qvec[0], [np.frombuffer(v, dtype=np.float32) for v in vectors])
                return [rows[i][0] for i in order[:k]]
        # No embeddings: plain word match over label + markings ("LEDs" finds
        # "blue LED" — trailing plural s is stripped from query words).
        words = [w.lower().rstrip("s") for w in query.split() if len(w) > 1]
        hits = [r[0] for r in rows
                if any(w and w in f"{r[0]} {r[1]}".lower() for w in words)]
        return hits[:k]


class BenchSkill(Skill):
    name = "see_bench"
    description = (
        "Identify the electronic part currently on the workbench camera "
        "(reads part markings too); can also log parts to the bench inventory."
    )

    patterns = [
        re.compile(r"\bwhat(?:'?s| is)\s+on\s+(?:my|the)\s+bench\b", re.IGNORECASE),
        re.compile(r"\bidentify\s+this\s+(?:part|component|chip)\b", re.IGNORECASE),
        re.compile(r"\bшто\s+е\s+ова\b", re.IGNORECASE),
        re.compile(r"\blog\s+this\s+part\b", re.IGNORECASE),
        re.compile(r"\bdo\s+i\s+have\s+any\s+(?P<query>.+)", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, inventory: BenchInventory,
                 fetch_frame=None, describe=None, ocr=None) -> None:
        self._settings = settings
        self._inventory = inventory
        self._fetch_frame = fetch_frame or self._default_fetch
        self._describe = describe or self._default_describe
        self._ocr = ocr or default_ocr

    # -- default (real) backends ---------------------------------------------

    async def _default_fetch(self) -> bytes:
        """One frame from the sidecar's per-request bench camera endpoint."""
        import httpx

        port = self._settings.vision.stream_port
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/bench.jpg")
            resp.raise_for_status()
            return resp.content

    async def _default_describe(self, image_b64: str) -> SkillResult:
        from skills.vision_skill import _describe

        return await _describe(self._settings, image_b64, BENCH_PROMPT)

    # -- the skill -------------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()

        # Inventory recall needs no camera — works even with the bench disabled.
        gd = request.match.groupdict() if request.match else {}
        query = (request.args.get("query") or gd.get("query") or "").strip(" ?.!")
        if query and "do i have" in text:
            hits = await asyncio.to_thread(self._inventory.search, query)
            if not hits:
                return SkillResult(f"Nothing in the bench inventory matches {query}.")
            return SkillResult("In the bench inventory: " + "; ".join(hits) + ".",
                               data={"hits": hits})

        if not self._settings.vision.bench.enabled:
            return SkillResult(
                "The bench camera is disabled — set vision.bench.enabled in "
                "config.yaml.", success=False)

        try:
            frame = await self._fetch_frame()
        except Exception:
            return SkillResult(
                "I couldn't reach the bench camera — is the vision sidecar "
                "running?", success=False)
        crop = center_crop_jpeg(frame)

        import base64

        described = await self._describe(base64.b64encode(crop).decode())
        if not described.success:
            return described
        ocr_text = await asyncio.to_thread(self._ocr, crop)
        markings = extract_markings(ocr_text)
        speech = described.speech.rstrip(".") + "."
        if markings:
            speech += f" Markings: {', '.join(markings)}."

        if "log this part" in text:
            await asyncio.to_thread(self._inventory.add, speech, ocr_text)
            return SkillResult("Logged. " + speech,
                               data={"markings": markings, "logged": True})
        return SkillResult(speech, data={"markings": markings})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search the bench inventory instead "
                                           "of looking at the camera.",
                        }
                    },
                    "required": [],
                },
            },
        }
