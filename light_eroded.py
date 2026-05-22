#!/usr/bin/env python3
"""Light-eroded engraving — illumination-driven line drawing.

Principle: the shape exists as a dark field.  Light removes darkness.
Where light strikes a face directly (n·l ≈ 1), lines are sparse and
thin — nearly erased.  Where light glances or misses (n·l ≈ 0 or
negative), lines are dense and dark — the form retains its mass.

This is the inverse of the silouhette-proximity kernel.  Instead of
density ∝ 1/distance-to-silhouette (view-dependent), density ∝
max(0, 1 − n·l) (light-dependent).  The result is an engraving that
reads as a shaded solid under a single light source, with every line
placement derived from the same numeric polytope model the oracle trusts.
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
from proximity_engraving import (
    classify_faces,
    project_onto_image,
    _line_polygon_intersections,
    _signed_dist_to_silhouette_edge,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


def light_eroded_strokes(
    polytope: Polytope,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    *,
    light_dir: np.ndarray = None,
    base_spacing: float = 4.0,
    min_spacing: float = 1.2,
    max_lines_per_face: int = 80,
) -> list[tuple[tuple[float, float], tuple[float, float], float, float]]:
    """Generate strokes where density = 1 − max(0, n·l).

    Faces that face the light (n·l ≈ 1) get few thin light strokes.
    Faces that face away (n·l ≈ 0) get dense dark strokes.
    Back-lit faces (n·l < 0) get maximum density.

    Parameters
    ----------
    light_dir : np.ndarray (3,)
        Direction TO the light source (e.g. [1, 3, 2] for a light
        coming from the upper-right-front).  Normalized internally.
    """
    if light_dir is None:
        light_dir = np.array([1.0, 3.0, 2.0], dtype=float)
    light_dir = normalize(light_dir)

    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)
    front_faces, _, _ = classify_faces(polytope, camera_pos)

    epf = polytope_edge_faces(polytope)
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
        poly_2d = np.array([projected[vid] for vid in face])
        fn = face_normal(verts, face)
        fc = face_center(verts, face)

        # Light intensity on this face: n·l clamped to [0, 1]
        ndotl = max(0.0, float(np.dot(fn, light_dir)))

        # Stroke direction: project face normal to 2D
        view_dir = camera_pos - fc
        sd3 = normalize(np.cross(fn, view_dir))
        if float(np.linalg.norm(sd3)) < 1e-9:
            sd3 = np.cross(fn, np.array([0.0, 1.0, 0.0]))
        p0 = project_onto_image(np.array([fc]), camera_pos, camera_target)[0]
        p1 = project_onto_image(np.array([fc + sd3 * 0.01]), camera_pos, camera_target)[0]
        stroke_dir = p1 - p0
        slen = float(np.linalg.norm(stroke_dir))
        if slen < 1e-9:
            continue
        stroke_dir /= slen
        perp_dir = np.array([-stroke_dir[1], stroke_dir[0]])

        dots = np.dot(poly_2d, perp_dir)
        d_min, d_max = float(dots.min()), float(dots.max())
        span = max(d_max - d_min, 1e-9)

        sil_edges = face_sil_edges.get(fi, [])

        # Light-eroded density: dark = 1 − n·l
        # Full light (n·l=1) → density 0 → skip most strokes
        # No light (n·l=0) → density 1 → maximum strokes
        light_density = 1.0 - ndotl

        # Also blend with silouhette proximity for edge emphasis
        n_lines = max(2, min(max_lines_per_face, int(span / min_spacing * max(0.15, light_density))))
        for i in range(n_lines):
            t = (i + 0.5) / n_lines
            sweep_pos = d_min + t * span

            mid_pt = poly_2d.mean(axis=0) + perp_dir * (sweep_pos - float(np.dot(poly_2d.mean(axis=0), perp_dir)))
            dist = _signed_dist_to_silhouette_edge(mid_pt, poly_2d, sil_edges, list(face))
            sil_weight = base_spacing / (dist + base_spacing * 0.5)
            sil_weight = max(0.1, min(1.0, sil_weight))

            # Combined weight: light erosion × silouhette proximity
            weight = light_density * 0.7 + sil_weight * 0.3
            weight = max(0.05, min(1.0, weight))

            if weight < 0.15 and (i % 4 != 0):
                continue

            hits = _line_polygon_intersections(poly_2d, perp_dir, sweep_pos, stroke_dir)
            if hits is None:
                continue
            (x0, y0), (x1, y1) = hits

            thick = 0.06 + 0.50 * weight
            gray = 0.02 + 0.55 * (1.0 - weight)

            all_strokes.append(((x0, y0), (x1, y1), thick, gray))

    return all_strokes


def write_light_eroded_pdf(
    path: Path,
    shape_name: str,
    polytope: Polytope,
    strokes: list,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    light_dir: np.ndarray,
    *,
    image_size: tuple[float, float] = (800.0, 600.0),
) -> None:
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(18.0)

    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"{shape_name} — light-eroded engraving",
             font="Helvetica-Bold", size=13.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             f"{len(strokes)} strokes | density ∝ 1−max(0,n·l)"
             f" | light dir = ({light_dir[0]:.1f}, {light_dir[1]:.1f}, {light_dir[2]:.1f})",
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

    # Background: dark field (the "shape is there")
    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)
    front_faces, _, _ = classify_faces(polytope, camera_pos)
    for fi in front_faces:
        face = polytope.faces[fi]
        pts = [mp(float(projected[vid, 0]), float(projected[vid, 1])) for vid in face]
        if len(pts) < 3:
            continue
        pdf.save_state()
        pdf.content.set_fill_gray(0.92)
        pdf.content.set_stroke_gray(1.0)
        pdf.content.move_to(*pts[0])
        for p in pts[1:]:
            pdf.content.line_to(*p)
        pdf.content.close_path()
        pdf.content.fill()
        pdf.restore_state()

    # Strokes: light removes darkness
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

    # Silhouette edges
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
        pdf.content.move_to(x0p, y0p)
        pdf.content.line_to(x1p, y1p)
        pdf.content.stroke()
        pdf.restore_state()

    pdf.restore_state()

    footer_y = margin - mm_to_pts(2.0)
    pdf.text(margin, footer_y,
             "light-eroded engraving: the shape exists as darkness; "
             "light removes lines. density ∝ 1−n·l",
             font="Helvetica", size=8.0, gray=0.35)

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
    parser.add_argument("--camera-x", type=float, default=2.8)
    parser.add_argument("--camera-y", type=float, default=2.0)
    parser.add_argument("--camera-z", type=float, default=3.5)
    parser.add_argument("--light-x", type=float, default=1.0)
    parser.add_argument("--light-y", type=float, default=3.0)
    parser.add_argument("--light-z", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polytope = make_polytope(args.shape)
    camera_pos = np.array([args.camera_x, args.camera_y, args.camera_z], dtype=float)
    camera_target = np.zeros(3, dtype=float)
    light_dir = np.array([args.light_x, args.light_y, args.light_z], dtype=float)

    strokes = light_eroded_strokes(polytope, camera_pos, camera_target, light_dir=light_dir)

    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"light_eroded_{args.shape}"
    pdf_path = out_dir / f"{stem}.pdf"

    write_light_eroded_pdf(
        pdf_path, args.shape, polytope, strokes, camera_pos, camera_target, light_dir,
    )

    print(f"shape       {args.shape}")
    print(f"light dir   ({args.light_x}, {args.light_y}, {args.light_z})")
    print(f"strokes     {len(strokes)}")
    print(f"pdf         {pdf_path}")


if __name__ == "__main__":
    main()
