#!/usr/bin/env python3
"""Smooth occluding contour extraction from subdivided polytopes.

Finds where n(p)·v = 0 (the occluding contour condition) on a
Catmull-Clark subdivided mesh, chains the zero-crossing points into
smooth curves, and renders print-quality PDF proofs with cubic Bezier
strokes.

Trust boundary: relies on polytope_numbers for polytope primitives
and subdivide for Catmull-Clark subdivision — both trusted.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pure_pdf import A4_PORTRAIT, PurePDF, mm_to_pts
from polytope_numbers import (
    EPS,
    Polytope,
    face_center,
    face_normal,
    make_polytope,
    normalize,
    polytope_edge_faces,
    rot_x,
    rot_z,
)
from subdivide import catmull_clark, normalize_to_sphere

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


# ═══════════════════════════════════════════════════════════════════════════
# Contour extraction — n·v = 0 on subdivided mesh
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ContourPoint:
    """A zero-crossing point on a mesh edge where n·v changes sign."""

    position: np.ndarray       # world-space position (3,)
    normal: np.ndarray         # interpolated normal at crossing
    edge_a: int                # vertex index at one end
    edge_b: int                # vertex index at other end
    t: float                   # interpolation parameter a→b


def _vertex_normals(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    """Compute area-weighted per-vertex normals from adjacent faces."""
    vn = np.zeros_like(vertices)
    for face in faces:
        fn = face_normal(vertices, face)
        fc = face_center(vertices, face)
        area = 0.0
        for i in range(len(face)):
            a = vertices[face[i]]
            b = vertices[face[(i + 1) % len(face)]]
            area += 0.5 * float(np.linalg.norm(np.cross(a - fc, b - fc)))
        for vid in face:
            vn[vid] += fn * area
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    return vn / norms


def _raw_edge_faces(faces: list[list[int]]) -> dict[tuple[int, int], list[int]]:
    """Build edge→adjacent-faces map from raw face index lists."""
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces):
        k = len(face)
        for i in range(k):
            a, b = face[i], face[(i + 1) % k]
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append(fi)
    return edge_map


def extract_contours(
    vertices: np.ndarray,
    faces: list[list[int]],
    camera_position: np.ndarray,
    *,
    project_to_sphere: bool = False,
    sphere_radius: float = 1.0,
) -> tuple[list[list[ContourPoint]], np.ndarray, np.ndarray]:
    """Extract occluding contour curves from a subdivided mesh.

    For each mesh edge, checks whether n·v (normal dot view-direction)
    changes sign between endpoints.  If so, linearly interpolates the
    zero-crossing point and chains adjacent crossings into closed loops.

    Parameters
    ----------
    vertices : np.ndarray (V×3)
        Vertex positions in world space (centered at origin).
    faces : list[list[int]]
        Face index lists.
    camera_position : np.ndarray (3,)
        Camera position in world space (used to compute view direction).
    project_to_sphere : bool
        If True, project contour points to sphere surface (for analytic
        comparison).
    sphere_radius : float
        Sphere radius for projection.

    Returns
    -------
    (list[list[ContourPoint]], np.ndarray, np.ndarray)
        Contour chains, all contour points (N×3), and per-vertex normals.
    """
    vn = _vertex_normals(vertices, faces)

    # n·v at each vertex
    ndotv = np.empty(len(vertices), dtype=float)
    for vi in range(len(vertices)):
        view_dir = camera_position - vertices[vi]
        vlen = float(np.linalg.norm(view_dir))
        if vlen < EPS:
            ndotv[vi] = 0.0
        else:
            ndotv[vi] = float(np.dot(vn[vi], view_dir / vlen))

    # Find zero-crossing edges and index them
    epf = _raw_edge_faces(faces)
    crossings: dict[tuple[int, int], ContourPoint] = {}
    crossing_index: dict[tuple[int, int], int] = {}
    next_idx = 0

    for edge in epf:
        a, b = edge
        fa, fb = ndotv[a], ndotv[b]
        if fa * fb >= 0.0:
            continue
        # Zero crossing: interpolate position along the edge
        t = abs(fa) / (abs(fa) + abs(fb))
        pos = vertices[a] + t * (vertices[b] - vertices[a])
        nrm = vn[a] + t * (vn[b] - vn[a])
        nrm = nrm / (float(np.linalg.norm(nrm)) + EPS)

        if project_to_sphere:
            r = float(np.linalg.norm(pos))
            if r > EPS:
                pos = pos * (sphere_radius / r)

        cp = ContourPoint(position=pos, normal=nrm, edge_a=a, edge_b=b, t=t)
        crossings[edge] = cp
        crossing_index[edge] = next_idx
        next_idx += 1

    if not crossings:
        return [], np.empty((0, 3)), vn

    # Build adjacency: two crossing points are adjacent if they belong
    # to the same face AND that face contains exactly two zero-crossing
    # edges.  This is the standard contour-chaining rule.
    crossing_adj: list[list[int]] = [[] for _ in range(next_idx)]

    for face in faces:
        x_edges: list[tuple[int, int]] = []
        k = len(face)
        for i in range(k):
            a, b = face[i], face[(i + 1) % k]
            key = (a, b) if a < b else (b, a)
            if key in crossing_index:
                x_edges.append(key)
        if len(x_edges) == 2:
            c0 = crossing_index[x_edges[0]]
            c1 = crossing_index[x_edges[1]]
            crossing_adj[c0].append(c1)
            crossing_adj[c1].append(c0)

    # Chain: walk the adjacency graph
    visited_crossing = [False] * next_idx
    chains: list[list[ContourPoint]] = []

    for start in range(next_idx):
        if visited_crossing[start]:
            continue
        if len(crossing_adj[start]) == 0:
            visited_crossing[start] = True
            continue
        # Walk forward from start
        chain_pts: list[ContourPoint] = []
        current = start
        prev = -1
        while True:
            visited_crossing[current] = True
            # Find the crossing edge for this index
            for e, idx in crossing_index.items():
                if idx == current:
                    chain_pts.append(crossings[e])
                    break
            candidates = [n for n in crossing_adj[current] if n != prev and not visited_crossing[n]]
            if len(candidates) == 0:
                break
            prev = current
            current = candidates[0]

        if len(chain_pts) >= 3:
            chains.append(chain_pts)

    all_pts = np.array([cp.position for ch in chains for cp in ch], dtype=float) if chains else np.empty((0, 3))
    return chains, all_pts, vn


# ═══════════════════════════════════════════════════════════════════════════
# Bezier fitting — polyline → smooth cubic Bezier curve
# ═══════════════════════════════════════════════════════════════════════════


def fit_bezier_chain(
    points: np.ndarray,
    closed: bool = True,
    segments: int = 24,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit a C¹ cubic Bezier spline to a cyclic polyline.

    Each segment (P_i → P_{i+1}) gets control handles:
        cp1 = P_i + T_i * h/3
        cp2 = P_{i+1} - T_{i+1} * h/3
    where T_i is the unit tangent at point i and h is chord length.

    Parameters
    ----------
    points : np.ndarray (N×3 or N×2)
        Cyclic point sequence.
    closed : bool
        If True, the curve closes back to points[0].
    segments : int
        Number of Bezier segments to emit (≥4).  If points has fewer
        vertices, we resample the closed loop uniformly.

    Returns
    -------
    list[tuple[ndarray, ndarray, ndarray, ndarray]]
        Each tuple is (P0, CP1, CP2, P1) — the four control points of a
        cubic Bezier segment.
    """
    n = len(points)
    if n < 2:
        return []

    # If we have many raw contour points, resample uniformly
    if n > segments:
        # Build cumulative chord-length parameterization
        chords = np.zeros(n, dtype=float)
        for i in range(1, n):
            chords[i] = chords[i - 1] + float(np.linalg.norm(points[i] - points[i - 1]))
        if closed:
            chords = np.append(chords, chords[-1] + float(np.linalg.norm(points[0] - points[-1])))
            total = chords[-1]
            indices = np.linspace(0, total, segments + 1)[:-1]
            resampled = np.empty((segments, points.shape[1]), dtype=float)
            for s in range(segments):
                t = indices[s]
                # Find enclosing chord segment
                j = np.searchsorted(chords, t, side='right') - 1
                j = max(0, min(n - 1, j))
                if j == n - 1:
                    resampled[s] = points[j]
                else:
                    local_t = (t - chords[j]) / max(chords[j + 1] - chords[j], EPS)
                    resampled[s] = points[j] + local_t * (points[j + 1] - points[j])
            points = resampled
            n = segments
            closed = True
            chords = np.zeros(n, dtype=float)
            for i in range(1, n):
                chords[i] = chords[i - 1] + float(np.linalg.norm(points[i] - points[i - 1]))

    # Compute tangents (central difference for interior, reflecting for boundary)
    tangents = np.empty_like(points)
    for i in range(n):
        if closed:
            prev = points[(i - 1) % n]
            nxt = points[(i + 1) % n]
            tangents[i] = nxt - prev
        else:
            if i == 0:
                tangents[i] = points[1] - points[0]
            elif i == n - 1:
                tangents[i] = points[-1] - points[-2]
            else:
                tangents[i] = points[i + 1] - points[i - 1]
    # Normalize tangents
    tnorm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tnorm[tnorm < EPS] = 1.0
    tangents = tangents / tnorm

    bez_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    count = n if closed else n - 1
    for i in range(count):
        j = (i + 1) % n if closed else i + 1
        h = float(np.linalg.norm(points[j] - points[i]))
        p0 = points[i]
        p1 = points[j]
        cp1 = p0 + tangents[i] * h / 3.0
        cp2 = p1 - tangents[j] * h / 3.0
        bez_segments.append((p0, cp1, cp2, p1))

    return bez_segments


# ═══════════════════════════════════════════════════════════════════════════
# Projection helpers
# ═══════════════════════════════════════════════════════════════════════════


def orthographic_project(points: np.ndarray, axes: tuple[int, int] = (0, 2)) -> np.ndarray:
    """Drop one axis for orthographic view. Default: drop Y → (X,Z) plane."""
    return points[:, axes]


def perspective_project(
    points: np.ndarray,
    camera_pos: np.ndarray,
    screen_dist: float = 5.0,
    axes: tuple[int, int] = (0, 2),
) -> np.ndarray:
    """Simple perspective projection: point projected along ray from camera."""
    result = np.empty((len(points), 2), dtype=float)
    for i, pt in enumerate(points):
        rel = pt - camera_pos
        z = rel[2]
        if abs(z) < EPS:
            z = EPS
        result[i, 0] = screen_dist * rel[axes[0]] / z
        result[i, 1] = screen_dist * rel[axes[1]] / z
    return result


def fit_points_to_rect(
    points: np.ndarray,
    rect: tuple[float, float, float, float],
    pad_fraction: float = 0.10,
) -> np.ndarray:
    """Scale and translate 2D points to fit within a rect."""
    x0, y0, width, height = rect
    mins = points[:, :2].min(axis=0)
    maxs = points[:, :2].max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-6)
    mins = mins - pad_fraction * span
    maxs = maxs + pad_fraction * span
    span = np.maximum(maxs - mins, 1.0e-6)
    scale = min(width / span[0], height / span[1])
    offset_x = x0 + 0.5 * (width - span[0] * scale)
    offset_y = y0 + 0.5 * (height - span[1] * scale)
    mapped = np.empty((len(points), 2), dtype=float)
    mapped[:, 0] = offset_x + (points[:, 0] - mins[0]) * scale
    mapped[:, 1] = offset_y + (points[:, 1] - mins[1]) * scale
    return mapped


# ═══════════════════════════════════════════════════════════════════════════
# PDF proof output
# ═══════════════════════════════════════════════════════════════════════════


def write_smooth_contour_pdf(
    path: Path,
    shape_name: str,
    subdiv_levels: int,
    vertices: np.ndarray,
    faces: list[list[int]],
    contour_chains: list[list[ContourPoint]],
    camera_position: np.ndarray,
) -> None:
    """Write a 4-panel PDF proof showing smooth contours on the subdivided mesh.

    Panels:
      - Isometric: subdivided wireframe + smooth contour (perspective)
      - Front: orthographic subdivided wireframe + contour
      - Top: orthographic subdivided wireframe + contour
      - Side: orthographic subdivided wireframe + contour
    """
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(12.0)
    gap = mm_to_pts(6.0)
    panel_w = 0.5 * (page_w - 2.0 * margin - gap)
    panel_h = 0.5 * (page_h - 2.0 * margin - gap - mm_to_pts(20.0))

    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"{shape_name} smooth contour — subdiv {subdiv_levels}",
             font="Helvetica-Bold", size=13.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             f"n·v = 0 on Catmull-Clark subdivided mesh | {sum(len(c) for c in contour_chains)} contour points in {len(contour_chains)} chain(s)",
             font="Helvetica", size=8.5, gray=0.28)

    # Project subdivided mesh vertices once per view
    epf = _raw_edge_faces(faces)
    edges_2d = list(epf.keys())

    panel_specs = [
        ("Isometric", "perspective", np.array([2.8, 2.0, 3.5])),
        ("Front", "orthographic", None),
        ("Top", "orthographic", None),
        ("Side", "orthographic", None),
    ]

    for index, (title, proj, cam_pos) in enumerate(panel_specs):
        col = index % 2
        row = index // 2
        rect = (margin + col * (panel_w + gap),
                margin + (1 - row) * (panel_h + gap),
                panel_w, panel_h)

        if proj == "perspective" and cam_pos is not None:
            v2d = perspective_project(vertices, cam_pos, screen_dist=4.0, axes=(0, 2))
            # Build world-space contour point array for projection
            all_cp = np.vstack([np.array([cp.position for cp in ch]) for ch in contour_chains]) if contour_chains else np.empty((0, 3))
            cp2d = perspective_project(all_cp, cam_pos, screen_dist=4.0, axes=(0, 2)) if len(all_cp) > 0 else np.empty((0, 2))
        else:
            axes = (0, 2) if title == "Front" else (0, 1) if title == "Top" else (1, 2)
            v2d = orthographic_project(vertices, axes=axes)
            all_cp = np.vstack([np.array([cp.position for cp in ch]) for ch in contour_chains]) if contour_chains else np.empty((0, 3))
            cp2d = orthographic_project(all_cp, axes=axes) if len(all_cp) > 0 else np.empty((0, 2))

        # Fit to panel rect
        v2d_mapped = fit_points_to_rect(v2d, rect)
        if len(cp2d) > 0:
            cp2d_mapped = fit_points_to_rect(cp2d, rect)

        # Draw panel title
        pdf.text(rect[0], rect[1] + rect[3] + mm_to_pts(2.2), title,
                 font="Helvetica-Bold", size=9.5, gray=0.0)

        # Draw subdivided wireframe (light)
        for ea, eb in edges_2d:
            pdf.save_state()
            pdf.line_width(0.18)
            pdf.stroke_gray(0.88)
            pdf.content.set_line_cap(1)
            pdf.content.move_to(float(v2d_mapped[ea, 0]), float(v2d_mapped[ea, 1]))
            pdf.content.line_to(float(v2d_mapped[eb, 0]), float(v2d_mapped[eb, 1]))
            pdf.content.stroke()
            pdf.restore_state()

        # Draw smooth contour as cubic Bezier (bold, dark)
        if len(cp2d) > 0:
            cp_offset = 0
            for chain in contour_chains:
                n = len(chain)
                if n < 2:
                    cp_offset += n
                    continue
                chain_pts_2d = cp2d_mapped[cp_offset:cp_offset + n]
                bez = fit_bezier_chain(chain_pts_2d, closed=True, segments=max(24, n))
                pdf.save_state()
                pdf.line_width(0.85)
                pdf.stroke_gray(0.0)
                pdf.content.set_line_cap(1)
                pdf.content.set_line_join(1)
                for seg in bez:
                    pdf.content.move_to(float(seg[0][0]), float(seg[0][1]))
                    pdf.content.curve_to(
                        float(seg[1][0]), float(seg[1][1]),
                        float(seg[2][0]), float(seg[2][1]),
                        float(seg[3][0]), float(seg[3][1]),
                    )
                pdf.content.stroke()
                pdf.restore_state()
                cp_offset += n

        # Panel border
        pdf.save_state()
        pdf.line_width(0.35)
        pdf.stroke_gray(0.82)
        pdf.content.rect(rect[0], rect[1], rect[2], rect[3])
        pdf.content.stroke()
        pdf.restore_state()

    footer_y = margin - mm_to_pts(2.0)
    pdf.text(
        margin, footer_y,
        f"subdiv levels={subdiv_levels} | vertices={len(vertices)} faces={len(faces)} | "
        f"contour chains={len(contour_chains)}",
        font="Helvetica", size=8.5, gray=0.25,
    )
    pdf.save(str(path))


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def smooth_contour_proof(
    shape: str,
    subdiv_levels: int = 3,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Run the full smooth contour pipeline for one shape.

    1. Build the polytope
    2. Subdivide with Catmull-Clark
    3. Project to sphere (for regular solids)
    4. Extract n·v = 0 contour curves
    5. Fit cubic Bezier splines
    6. Write print-quality PDF proof

    Returns the path to the generated PDF.
    """
    polytope = make_polytope(shape)
    vertices = polytope.vertices.copy()
    faces = [list(f) for f in polytope.faces]

    vertices, faces = catmull_clark(vertices, faces, levels=subdiv_levels)
    vertices = normalize_to_sphere(vertices, faces, radius=polytope.sphere_radius)

    camera_pos = np.array([0.0, 0.0, 5.0], dtype=float)

    chains, all_pts, vnormals = extract_contours(
        vertices, faces, camera_pos, project_to_sphere=False,
    )

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"smooth_{shape}_subdiv{subdiv_levels}"
    pdf_path = out_dir / f"{stem}.pdf"

    write_smooth_contour_pdf(
        pdf_path, shape, subdiv_levels,
        vertices, faces, chains, camera_pos,
    )

    return pdf_path, chains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cube", "octahedron", "icosahedron"),
                        default="cube")
    parser.add_argument("--subdiv", type=int, default=3,
                        help="Catmull-Clark subdivision levels (default 3)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path, chains = smooth_contour_proof(
        args.shape, args.subdiv, output_dir=args.output_dir,
    )
    print(f"shape       {args.shape}")
    print(f"subdiv      {args.subdiv}")
    print(f"contour chains  {len(chains)}")
    for i, ch in enumerate(chains):
        print(f"  chain {i}: {len(ch)} points")
    print(f"pdf         {pdf_path}")


if __name__ == "__main__":
    main()
