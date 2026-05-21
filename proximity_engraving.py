#!/usr/bin/env python3
"""Silhouette-proximity engraving — Piranesi-style form from polytope geometry.

Core principle: line density is proportional to proximity to the
silhouette boundary.  Within each visible face, strokes accumulate
near edges that border hidden faces, and thin out toward the face
center.  Stroke direction follows the projected face normal.

The result is not a grid — it's a density field driven by geometric
truth.  Every line placement is derived from the same numeric polytope
model the oracle trusts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pure_pdf import A4_PORTRAIT, PurePDF, mm_to_pts
from polytope_numbers import (
    Polytope,
    face_center,
    face_normal,
    make_polytope,
    normalize,
    polytope_edge_faces,
)
from subdivide import catmull_clark, normalize_to_sphere

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


# ═══════════════════════════════════════════════════════════════════════════
# Projection
# ═══════════════════════════════════════════════════════════════════════════

def project_onto_image(
    points: np.ndarray,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    image_size: tuple[float, float] = (800.0, 600.0),
    fov_y_deg: float = 35.0,
) -> np.ndarray:
    forward = normalize(camera_target - camera_pos)
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    aspect = image_size[0] / image_size[1]
    tan_half_fov = math.tan(math.radians(fov_y_deg) * 0.5)
    result = np.empty((len(points), 2), dtype=float)
    for i, pt in enumerate(points):
        rel = pt - camera_pos
        x = float(np.dot(rel, right))
        y = float(np.dot(rel, true_up))
        z = float(np.dot(rel, forward))
        if z < 1e-6:
            z = 1e-6
        ndc_x = x / (z * tan_half_fov * aspect)
        ndc_y = y / (z * tan_half_fov)
        result[i, 0] = (ndc_x + 1.0) * 0.5 * image_size[0]
        result[i, 1] = (1.0 - (ndc_y + 1.0) * 0.5) * image_size[1]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Silhouette-aware face visibility
# ═══════════════════════════════════════════════════════════════════════════

def classify_faces(
    polytope: Polytope,
    camera_pos: np.ndarray,
) -> tuple[set[int], set[int], set[int]]:
    """Classify faces as front-facing, back-facing, or silhouette candidates.

    Returns (visible_faces, hidden_faces, silhouette_edges).
    """
    verts = polytope.vertices
    front_faces: set[int] = set()
    back_faces: set[int] = set()

    for fi, face in enumerate(polytope.faces):
        fn = face_normal(verts, face)
        fc = face_center(verts, face)
        vd = camera_pos - fc
        if float(np.dot(fn, vd)) > 0:
            front_faces.add(fi)
        else:
            back_faces.add(fi)

    epf = polytope_edge_faces(polytope)
    silhouette_edges: set[int] = set()
    for edge, faces in epf.items():
        if len(faces) != 2:
            continue
        fv = sum(1 for f in faces if f in front_faces)
        if fv == 1:
            silhouette_edges.add(faces[0] if faces[0] in front_faces else faces[1])

    return front_faces, back_faces, silhouette_edges


# ═══════════════════════════════════════════════════════════════════════════
# Stroke generation
# ═══════════════════════════════════════════════════════════════════════════

def _signed_dist_to_silhouette_edge(
    point_2d: np.ndarray,
    face_poly_2d: np.ndarray,
    sil_edges: list[tuple[int, int]],
    face_vertex_indices: list[int],
) -> float:
    """Distance from a 2D point inside a face to the nearest silhouette edge.

    Returns signed distance: negative = inside silhouette boundary.
    """
    best = float("inf")
    for ea, eb in sil_edges:
        # Map local face vertex indices to positions in poly_2d
        try:
            ia = face_vertex_indices.index(ea)
            ib = face_vertex_indices.index(eb)
        except ValueError:
            continue
        a = face_poly_2d[ia]
        b = face_poly_2d[ib]
        # Perpendicular distance to line segment
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-12:
            continue
        ap = point_2d - a
        t = max(0.0, min(1.0, float(np.dot(ap, ab)) / ab_len_sq))
        closest = a + t * ab
        d = float(np.linalg.norm(point_2d - closest))
        best = min(best, d)
    return best if best != float("inf") else 0.0


def generate_strokes(
    polytope: Polytope,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    *,
    base_spacing: float = 5.0,
    min_spacing: float = 1.5,
    max_lines_per_face: int = 80,
) -> list[tuple[tuple[float, float], tuple[float, float], float, float]]:
    """Generate proximity-weighted engraving strokes.

    For each visible face, strokes follow the projected face normal.
    Density increases near silhouette edges (where the form turns away).
    """
    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)
    front_faces, back_faces, silhouette_face_edges = classify_faces(polytope, camera_pos)

    epf = polytope_edge_faces(polytope)
    # Find which edges of each front face are silhouette edges
    face_sil_edges: dict[int, list[tuple[int, int]]] = {}
    for edge, faces in epf.items():
        if len(faces) != 2:
            continue
        fv = sum(1 for f in faces if f in front_faces)
        if fv == 1:
            for f in faces:
                if f in front_faces:
                    face_sil_edges.setdefault(f, []).append(edge)

    all_strokes: list[tuple[tuple[float, float], tuple[float, float], float, float]] = []

    for fi in sorted(front_faces):
        face = polytope.faces[fi]
        n = len(face)
        poly_2d = np.array([projected[vid] for vid in face])
        fn = face_normal(verts, face)
        fc = face_center(verts, face)

        # Stroke direction: project face normal to 2D
        view_dir = camera_pos - fc
        stroke_dir_3d = normalize(np.cross(fn, view_dir))
        if float(np.linalg.norm(stroke_dir_3d)) < 1e-9:
            stroke_dir_3d = np.cross(fn, np.array([0.0, 1.0, 0.0]))

        p0 = project_onto_image(np.array([fc]), camera_pos, camera_target)[0]
        p1 = project_onto_image(np.array([fc + stroke_dir_3d * 0.01]), camera_pos, camera_target)[0]
        stroke_dir = p1 - p0
        slen = float(np.linalg.norm(stroke_dir))
        if slen < 1e-9:
            continue
        stroke_dir /= slen
        perp_dir = np.array([-stroke_dir[1], stroke_dir[0]])

        # Project bounds along perpendicular direction
        dots = np.dot(poly_2d, perp_dir)
        d_min, d_max = float(dots.min()), float(dots.max())
        span = max(d_max - d_min, 1e-9)

        sil_edges = face_sil_edges.get(fi, [])

        # Generate strokes with varying density
        n_lines = max(3, min(max_lines_per_face, int(span / min_spacing)))
        for i in range(n_lines):
            t = (i + 0.5) / n_lines
            sweep_pos = d_min + t * span

            # Compute proximity weight at this sweep position
            # by sampling a point in the middle of the sweep line
            mid_pt = poly_2d.mean(axis=0) + perp_dir * (sweep_pos - float(np.dot(poly_2d.mean(axis=0), perp_dir)))
            dist = _signed_dist_to_silhouette_edge(mid_pt, poly_2d, sil_edges, list(face))

            # Density weight: 1 / (dist + base_spacing)
            # Near silhouette: high density (thin spacing)
            # Far from silhouette: low density (wider spacing)
            weight = base_spacing / (dist + base_spacing * 0.5)
            weight = max(0.15, min(1.0, weight))

            # Skip low-weight strokes randomly
            if weight < 0.25 and (i % 3 != 0):
                continue

            # Find intersections of sweep line with polygon
            hits = _line_polygon_intersections(poly_2d, perp_dir, sweep_pos, stroke_dir)
            if hits is None:
                continue

            (x0, y0), (x1, y1) = hits

            # Line thickness and gray: heavier near silhouette
            thick = 0.08 + 0.40 * weight
            gray = 0.02 + 0.50 * (1.0 - weight)

            all_strokes.append(((x0, y0), (x1, y1), thick, gray))

    return all_strokes


def _line_polygon_intersections(
    poly_2d: np.ndarray,
    perp_dir: np.ndarray,
    offset: float,
    stroke_dir: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    n = len(poly_2d)
    hits: list[np.ndarray] = []
    for i in range(n):
        a = poly_2d[i]
        b = poly_2d[(i + 1) % n]
        da = float(np.dot(a, perp_dir)) - offset
        db = float(np.dot(b, perp_dir)) - offset
        if da * db > 0:
            continue
        if abs(da - db) < 1e-12:
            continue
        t = da / (da - db)
        pt = a + t * (b - a)
        hits.append(pt)
    if len(hits) == 2:
        d0 = float(np.dot(hits[0], stroke_dir))
        d1 = float(np.dot(hits[1], stroke_dir))
        if d0 > d1:
            hits[0], hits[1] = hits[1], hits[0]
        return ((float(hits[0][0]), float(hits[0][1])),
                (float(hits[1][0]), float(hits[1][1])))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PDF output
# ═══════════════════════════════════════════════════════════════════════════

def write_proximity_engraving_pdf(
    path: Path,
    shape_name: str,
    polytope: Polytope,
    strokes: list[tuple[tuple[float, float], tuple[float, float], float, float]],
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    *,
    image_size: tuple[float, float] = (800.0, 600.0),
) -> None:
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(18.0)

    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"{shape_name} — silhouette-proximity engraving",
             font="Helvetica-Bold", size=13.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             f"{len(strokes)} strokes | density ∝ 1/distance-to-silhouette"
             f" | direction = projected face normal",
             font="Helvetica", size=8.5, gray=0.28)

    content_rect = (margin, margin,
                    page_w - 2.0 * margin,
                    page_h - 2.0 * margin - mm_to_pts(14.0))
    x0, y0, cw, ch = content_rect
    scale = min(cw / image_size[0], ch / image_size[1])
    ox = x0 + (cw - image_size[0] * scale) * 0.5
    oy = y0 + (ch - image_size[1] * scale) * 0.5

    def mp(x: float, y: float) -> tuple[float, float]:
        return (ox + x * scale, oy + (image_size[1] - y) * scale)

    pdf.save_state()
    pdf.clip_rect(x0, y0, cw, ch)

    # Fill visible faces lightly so the solid reads
    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)
    front_faces, _, _ = classify_faces(polytope, camera_pos)
    for fi in front_faces:
        face = polytope.faces[fi]
        pts = [mp(float(projected[vid, 0]), float(projected[vid, 1])) for vid in face]
        if len(pts) < 3:
            continue
        pdf.save_state()
        pdf.content.set_fill_gray(0.97)
        pdf.content.set_stroke_gray(1.0)
        pdf.content.move_to(*pts[0])
        for p in pts[1:]:
            pdf.content.line_to(*p)
        pdf.content.close_path()
        pdf.content.fill()
        pdf.restore_state()

    # Draw strokes
    for (x0s, y0s), (x1s, y1s), thick, gray in strokes:
        x0p, y0p = mp(x0s, y0s)
        x1p, y1p = mp(x1s, y1s)
        pdf.save_state()
        pdf.line_width(max(0.06, thick * scale * 0.5))
        pdf.stroke_gray(gray)
        pdf.content.set_line_cap(1)
        pdf.content.move_to(x0p, y0p)
        pdf.content.line_to(x1p, y1p)
        pdf.content.stroke()
        pdf.restore_state()

    # Draw silhouette edges bold
    _, _, sil_face_set = classify_faces(polytope, camera_pos)
    epf = polytope_edge_faces(polytope)
    for edge, faces in epf.items():
        if len(faces) != 2:
            continue
        fv = sum(1 for f in faces if f in front_faces)
        if fv != 1:
            continue
        a, b = edge
        x0p, y0p = mp(float(projected[a, 0]), float(projected[a, 1]))
        x1p, y1p = mp(float(projected[b, 0]), float(projected[b, 1]))
        pdf.save_state()
        pdf.line_width(0.85)
        pdf.stroke_gray(0.02)
        pdf.content.set_line_cap(1)
        pdf.content.set_line_join(1)
        pdf.content.move_to(x0p, y0p)
        pdf.content.line_to(x1p, y1p)
        pdf.content.stroke()
        pdf.restore_state()

    # Visible face interior seams
    for edge, faces in epf.items():
        if len(faces) != 2:
            continue
        if all(f in front_faces for f in faces):
            a, b = edge
            x0p, y0p = mp(float(projected[a, 0]), float(projected[a, 1]))
            x1p, y1p = mp(float(projected[b, 0]), float(projected[b, 1]))
            pdf.save_state()
            pdf.line_width(0.35)
            pdf.stroke_gray(0.55)
            pdf.content.set_line_cap(1)
            pdf.content.move_to(x0p, y0p)
            pdf.content.line_to(x1p, y1p)
            pdf.content.stroke()
            pdf.restore_state()

    pdf.restore_state()

    footer_y = margin - mm_to_pts(2.0)
    pdf.text(margin, footer_y,
             "silhouette-proximity engraving: line density increases near "
             "the occluding contour — the form turns, the lines gather",
             font="Helvetica", size=8.0, gray=0.35)

    pdf.save_state()
    pdf.line_width(0.35)
    pdf.stroke_gray(0.82)
    pdf.content.rect(x0, y0, cw, ch)
    pdf.content.stroke()
    pdf.restore_state()

    pdf.save(str(path))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cube", "octahedron", "icosahedron"),
                        default="cube")
    parser.add_argument("--camera-x", type=float, default=2.8)
    parser.add_argument("--camera-y", type=float, default=2.0)
    parser.add_argument("--camera-z", type=float, default=3.5)
    parser.add_argument("--subdivide", action="store_true",
                        help="apply Catmull-Clark subdivision for smooth limit surface")
    parser.add_argument("--subdiv-levels", type=int, default=2,
                        help="subdivision levels (default 2)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polytope = make_polytope(args.shape)

    if args.subdivide:
        verts = polytope.vertices.copy()
        faces = [list(f) for f in polytope.faces]
        verts, faces = catmull_clark(verts, faces, levels=args.subdiv_levels)
        verts = normalize_to_sphere(verts, faces, radius=polytope.sphere_radius)
        from polytope_numbers import build_polytope
        polytope = build_polytope(polytope.name, verts, faces)

    camera_pos = np.array([args.camera_x, args.camera_y, args.camera_z], dtype=float)
    camera_target = np.zeros(3, dtype=float)

    strokes = generate_strokes(polytope, camera_pos, camera_target)

    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"proximity_engraving_{args.shape}"
    if args.subdivide:
        stem += f"_subdiv{args.subdiv_levels}"
    pdf_path = out_dir / f"{stem}.pdf"

    write_proximity_engraving_pdf(
        pdf_path, args.shape, polytope, strokes, camera_pos, camera_target,
    )

    print(f"shape     {args.shape}")
    print(f"strokes   {len(strokes)}")
    print(f"pdf       {pdf_path}")


if __name__ == "__main__":
    main()
