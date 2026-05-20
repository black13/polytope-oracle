#!/usr/bin/env python3
"""Facet engraving — Piranesi-style line drawing from polytope geometry.

Each planar facet has a uniform normal.  Within a facet, every stroke
is parallel — the engraving burin follows the same direction everywhere
on that face.  When the eye moves across an edge, the stroke direction
changes, revealing the 3D form through the pattern of direction shifts.

The line density within each facet is proportional to the facet area
and its obliqueness to the viewer — edge-on faces get fewer, denser
strokes; face-on faces get lighter treatment.

This produces an analytical engraving that is simultaneously:
  - geometric truth (every line direction = projected facet normal)
  - cartographic (line density encodes surface orientation)
  - Piranesi-like (accumulated parallel strokes define form)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pure_pdf import A4_PORTRAIT, A3_PORTRAIT, PurePDF, mm_to_pts
from polytope_numbers import (
    Polytope,
    face_center,
    face_normal,
    make_polytope,
    normalize,
    polytope_edge_faces,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


def facet_area(vertices: np.ndarray, face: list[int]) -> float:
    """Area of a planar convex polygon via triangle fan."""
    area = 0.0
    a = vertices[face[0]]
    for i in range(1, len(face) - 1):
        b = vertices[face[i]]
        c = vertices[face[i + 1]]
        area += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return area


def project_onto_image(
    points: np.ndarray,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    image_size: tuple[float, float] = (800.0, 600.0),
    fov_y_deg: float = 35.0,
) -> np.ndarray:
    """Perspective-project 3D points to 2D image plane.

    Returns (N×2) array of projected coordinates in image space.
    """
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


def facet_hatch_lines(
    vertices: np.ndarray,
    face: list[int],
    normal_3d: np.ndarray,
    projected_vertices: np.ndarray,
    camera_pos: np.ndarray,
    *,
    line_spacing: float = 3.0,
    min_lines: int = 3,
    max_lines: int = 40,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Generate parallel hatch lines within one facet.

    The hatch direction in 2D is the projection of the face normal.
    Lines span the facet polygon in the perpendicular direction.

    Parameters
    ----------
    vertices : (V×3) array
        3D vertex positions.
    face : list[int]
        Vertex indices for this face.
    normal_3d : (3,) array
        Face normal in world space.
    projected_vertices : (N×2) array
        All projected vertices (indexed by face indices).
    camera_pos : (3,) array
        Camera position (for projecting normal).
    line_spacing : float
        Approximate spacing between hatch lines in image pixels.
    min_lines, max_lines : int
        Range for number of lines per facet.

    Returns
    -------
    list of ((x0,y0), (x1,y1)) line segments in image space.
    """
    n = len(face)
    if n < 3:
        return []

    # 2D polygon vertices
    poly_2d = np.array([projected_vertices[vid] for vid in face])

    # Project face normal to 2D — this is the hatch direction
    fc = np.mean(vertices[list(face)], axis=0)
    view_dir = camera_pos - fc
    # The projected normal direction: project the 3D normal onto the
    # plane perpendicular to the view direction, then to the image.
    # Simpler: the 2D edge-on direction is where dot(normal, view_dir) → 0.
    # For parallel hatching within the facet, use the cross product of
    # the face normal with the view direction as the stroke direction.
    stroke_dir_3d = normalize(np.cross(normal_3d, view_dir))
    if float(np.linalg.norm(stroke_dir_3d)) < 1e-9:
        stroke_dir_3d = np.cross(normal_3d, np.array([0.0, 1.0, 0.0]))

    # Project stroke direction to 2D (as a direction, not a point)
    # Use the same camera projection for a small offset from the face center
    p0_2d = project_onto_image(
        np.array([fc]), camera_pos, np.zeros(3), (800.0, 600.0)
    )[0]
    p1_2d = project_onto_image(
        np.array([fc + stroke_dir_3d * 0.01]), camera_pos, np.zeros(3), (800.0, 600.0)
    )[0]
    stroke_dir_2d = p1_2d - p0_2d
    stroke_len = float(np.linalg.norm(stroke_dir_2d))
    if stroke_len < 1e-9:
        return []
    stroke_dir_2d /= stroke_len

    # Perpendicular direction for line placement
    perp_dir = np.array([-stroke_dir_2d[1], stroke_dir_2d[0]])

    # Project polygon bounds along perp_dir
    dots = np.dot(poly_2d, perp_dir)
    d_min, d_max = float(dots.min()), float(dots.max())
    span = max(d_max - d_min, 1e-9)

    # Number of lines proportional to facet area and obliqueness
    area = facet_area(vertices, face)
    obliquity = abs(float(np.dot(normal_3d, normalize(view_dir))))
    weight = area * (1.0 - obliquity + 0.3)
    n_lines = max(min_lines, min(max_lines, int(span / line_spacing * weight / area if area > 0 else min_lines)))

    # Generate lines
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n_lines):
        t = (i + 0.5) / n_lines  # center within band
        offset = d_min + t * span
        # Line goes through this offset along perp_dir, oriented along stroke_dir
        # Find where it intersects the polygon
        intersections = _line_polygon_intersection(
            poly_2d, perp_dir, offset, stroke_dir_2d,
        )
        if intersections is not None:
            lines.append(intersections)

    return lines


def _line_polygon_intersection(
    poly_2d: np.ndarray,
    perp_dir: np.ndarray,
    offset: float,
    stroke_dir: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find where a sweep line intersects a convex polygon.

    The sweep line is defined as {p | dot(p, perp_dir) = offset}.
    Returns the two intersection points with the polygon edges.
    """
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
        # Sort along stroke_dir so we get consistent orientation
        d0 = float(np.dot(hits[0], stroke_dir))
        d1 = float(np.dot(hits[1], stroke_dir))
        if d0 > d1:
            hits[0], hits[1] = hits[1], hits[0]
        return ((float(hits[0][0]), float(hits[0][1])),
                (float(hits[1][0]), float(hits[1][1])))
    return None


def engrave_facets(
    polytope: Polytope,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    *,
    line_spacing: float = 3.0,
) -> list[tuple[tuple[float, float], tuple[float, float], float, float]]:
    """Generate facet engraving lines for a polytope.

    Returns list of (start_pt, end_pt, thickness, gray) tuples.
    """
    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)

    all_lines: list[tuple[tuple[float, float], tuple[float, float], float, float]] = []

    for fi, face in enumerate(polytope.faces):
        normal_3d = face_normal(verts, face)
        fc = face_center(verts, face)
        view_dir = camera_pos - fc
        vlen = float(np.linalg.norm(view_dir))
        if vlen < 1e-9:
            continue
        view_dir_n = view_dir / vlen

        obliquity = abs(float(np.dot(normal_3d, view_dir_n)))
        if obliquity < 0.02:
            continue  # edge-on, skip

        lines = facet_hatch_lines(
            verts, face, normal_3d, projected, camera_pos,
            line_spacing=line_spacing,
        )

        # Line weight: darker when face is more oblique (edge-on)
        # Lighter when face is front-facing
        thickness = 0.12 + 0.35 * obliquity
        gray = 0.05 + 0.55 * obliquity

        for (a, b) in lines:
            all_lines.append((a, b, thickness, gray))

    return all_lines


def write_engraving_pdf(
    path: Path,
    shape_name: str,
    lines: list[tuple[tuple[float, float], tuple[float, float], float, float]],
    *,
    image_size: tuple[float, float] = (800.0, 600.0),
) -> None:
    """Write a print-quality PDF from facet engraving lines."""
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(18.0)

    pdf.text(
        margin, page_h - margin + mm_to_pts(1.0),
        f"{shape_name} — facet engraving",
        font="Helvetica-Bold", size=13.0, gray=0.0,
    )
    pdf.text(
        margin, page_h - margin - mm_to_pts(4.0),
        f"{len(lines)} parallel strokes | stroke direction = projected facet normal"
        f" | line weight ∝ obliquity",
        font="Helvetica", size=8.5, gray=0.28,
    )

    content_rect = (
        margin,
        margin,
        page_w - 2.0 * margin,
        page_h - 2.0 * margin - mm_to_pts(14.0),
    )

    # Scale image coordinates to page
    x0, y0, cw, ch = content_rect
    scale_x = cw / image_size[0]
    scale_y = ch / image_size[1]
    scale = min(scale_x, scale_y)
    offset_x = x0 + (cw - image_size[0] * scale) * 0.5
    offset_y = y0 + (ch - image_size[1] * scale) * 0.5

    def map_pt(x: float, y: float) -> tuple[float, float]:
        return (offset_x + x * scale, offset_y + (image_size[1] - y) * scale)

    pdf.save_state()
    pdf.clip_rect(x0, y0, cw, ch)

    # Draw all facet lines
    for a, b, thickness, gray in lines:
        x0p, y0p = map_pt(a[0], a[1])
        x1p, y1p = map_pt(b[0], b[1])

        pdf.save_state()
        pdf.line_width(thickness * scale * 0.5)
        pdf.stroke_gray(gray)
        pdf.content.set_line_cap(1)
        pdf.content.move_to(x0p, y0p)
        pdf.content.line_to(x1p, y1p)
        pdf.content.stroke()
        pdf.restore_state()

    pdf.restore_state()

    # Footer
    footer_y = margin - mm_to_pts(2.0)
    pdf.text(
        margin, footer_y,
        "facet engraving: within each face, all strokes are parallel "
        "(constant face normal) | direction changes at edges define form",
        font="Helvetica", size=8.0, gray=0.35,
    )

    # Content border
    pdf.save_state()
    pdf.line_width(0.35)
    pdf.stroke_gray(0.82)
    pdf.content.rect(x0, y0, cw, ch)
    pdf.content.stroke()
    pdf.restore_state()

    pdf.save(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cube", "octahedron", "icosahedron"),
                        default="cube")
    parser.add_argument("--line-spacing", type=float, default=3.0,
                        help="pixel spacing between hatch lines")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polytope = make_polytope(args.shape)

    # Camera position relative to polytope
    camera_pos = np.array([2.8, 2.0, 3.5], dtype=float)
    camera_target = np.zeros(3, dtype=float)

    lines = engrave_facets(
        polytope, camera_pos, camera_target,
        line_spacing=args.line_spacing,
    )

    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"engraving_{args.shape}"
    pdf_path = out_dir / f"{stem}.pdf"

    write_engraving_pdf(pdf_path, args.shape, lines)

    print(f"shape         {args.shape}")
    print(f"strokes       {len(lines)}")
    print(f"pdf           {pdf_path}")


if __name__ == "__main__":
    main()
