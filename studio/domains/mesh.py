"""Lightweight STL inspection for the 3D printability check — pure Python (no
trimesh/numpy), so it runs in tests and degrades nowhere.

Reads a binary OR ASCII STL, welds coincident vertices, and reports the triangle
count, the number of non-manifold (open) edges — an edge NOT shared by exactly
two triangles means the mesh isn't watertight, so it isn't print-ready — and the
bounding box in the model's units (mm for CadQuery output).
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple


def _read_binary(data: bytes) -> Optional[List[tuple]]:
    if len(data) < 84:
        return None
    n = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + n * 50:
        return None
    tris, off = [], 84
    for _ in range(n):
        v = struct.unpack("<9f", data[off + 12:off + 48])
        tris.append(((v[0], v[1], v[2]), (v[3], v[4], v[5]), (v[6], v[7], v[8])))
        off += 50
    return tris


def _read_ascii(data: bytes) -> Optional[List[tuple]]:
    try:
        text = data.decode("ascii", errors="ignore")
    except Exception:
        return None
    if "vertex" not in text:
        return None
    verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
    if len(verts) < 3 or len(verts) % 3 != 0:
        return None
    return [tuple(verts[i:i + 3]) for i in range(0, len(verts), 3)]


def inspect_stl(path) -> Optional[dict]:
    """``{triangles, open_edges, watertight, bbox}`` for an STL, or ``None`` if it
    can't be read. ``bbox`` is ``(dx, dy, dz)``."""
    try:
        data = Path(path).read_bytes()
    except Exception:
        return None
    tris = _read_binary(data)
    if tris is None:
        tris = _read_ascii(data)
    if not tris:
        return None

    index: dict = {}
    faces = []
    for tri in tris:
        face = []
        for v in tri:
            key = (round(v[0], 5), round(v[1], 5), round(v[2], 5))
            face.append(index.setdefault(key, len(index)))
        faces.append(face)
    edges: Counter = Counter()
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            edges[frozenset(e)] += 1
    open_edges = sum(1 for count in edges.values() if count != 2)

    pts = list(index.keys())
    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    bbox: Tuple[float, float, float] = (
        (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) if pts
        else (0.0, 0.0, 0.0))
    return {"triangles": len(faces), "open_edges": open_edges,
            "watertight": open_edges == 0, "bbox": bbox}
