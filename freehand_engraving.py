#!/usr/bin/env python3
"""Freehand engraving — simulated burin dynamics on polytope surfaces.

Instead of independent random strokes, this kernel traces continuous
paths across the visible surface.  A simulated hand moves from face
to face with inertia — stroke direction, pressure, and length vary
smoothly.  The result reads as a human engraver following the form.

Hand dynamics:
  - Position: follows the surface, crossing edges into adjacent faces
  - Direction: cross(face_normal, view_dir) with inertia smoothing
  - Pressure: responds to n·l (lighter where lit, heavier in shadow)
  - Speed: varies with curvature (slower around edges, faster across faces)
  - Path length: random walk with momentum — travels N steps before lifting
"""

from __future__ import annotations

import argparse
import math
import random
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
from proximity_engraving import classify_faces, project_onto_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


def adjacent_faces(polytope: Polytope) -> dict[int, set[int]]:
    """Build face adjacency: which faces share an edge."""
    epf = polytope_edge_faces(polytope)
    adj: dict[int, set[int]] = {}
    for edge, faces in epf.items():
        if len(faces) == 2:
            a, b = faces
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def random_point_on_face(
    vertices: np.ndarray, face: list[int],
) -> np.ndarray:
    """Random 3D point on a face polygon via triangle-fan barycentric."""
    n = len(face)
    if n < 3:
        return vertices[face[0]].copy()
    a = vertices[face[0]]
    total = 0.0
    areas = []
    for i in range(1, n - 1):
        b = vertices[face[i]]
        c = vertices[face[i + 1]]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        areas.append(area)
        total += area
    if total < 1e-12:
        return a.copy()
    r = random.random() * total
    cumulative = 0.0
    tri_idx = 0
    for i, area in enumerate(areas):
        cumulative += area
        if r <= cumulative:
            tri_idx = i
            break
    u = random.random()
    v = random.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    b = vertices[face[tri_idx + 1]]
    c = vertices[face[tri_idx + 2]]
    return a + u * (b - a) + v * (c - a)


def freehand_strokes(
    polytope: Polytope,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    *,
    light_pos: np.ndarray = None,
    num_paths: int = 60,
    steps_per_path: int = 40,
    step_size: float = 0.015,
    direction_inertia: float = 0.8,
    pressure_variation: float = 0.3,
    speed_variation: float = 0.2,
    seed: int = 42,
) -> list[tuple[tuple[float, float], tuple[float, float], float, float]]:
    """Generate strokes by tracing continuous paths across the surface.

    Each path is a random walk across visible faces.  The hand has
    inertia — direction changes are smoothed, pressure varies with
    n·l, and speed varies with local curvature.

    Parameters
    ----------
    num_paths : int
        Number of independent burin paths to trace.
    steps_per_path : int
        Steps per path before lifting the burin.
    step_size : float
        Step size in 3D world units.
    direction_inertia : float
        How much previous direction persists (0 = no memory, 1 = rigid).
    pressure_variation : float
        Random pressure modulation (0 = uniform, 0.3 = expressive).
    speed_variation : float
        Random speed modulation per step.
    """
    random.seed(seed)
    np.random.seed(seed)

    if light_pos is None:
        light_pos = np.array([1.0, 4.0, 3.0], dtype=float)

    verts = polytope.vertices
    front_faces, _, _ = classify_faces(polytope, camera_pos)
    front_list = sorted(front_faces)
    if not front_list:
        return []

    adj = adjacent_faces(polytope)

    all_strokes: list[tuple[tuple[float, float], tuple[float, float], float, float]] = []

    for _ in range(num_paths):
        # Start on a random visible face
        fi = random.choice(front_list)
        face = polytope.faces[fi]
        pt = random_point_on_face(verts, face)
        prev_dir = None  # smoothed direction from previous step
        prev_pressure = None

        for step in range(steps_per_path):
            fn = face_normal(verts, face)

            # Is this face still front-facing?
            view_dir = camera_pos - pt
            vlen = float(np.linalg.norm(view_dir))
            if vlen < 1e-9 or float(np.dot(fn, view_dir)) <= 0:
                break
            view_dir_n = view_dir / vlen

            # ── Light pressure ──
            to_light = light_pos - pt
            dist = float(np.linalg.norm(to_light))
            ndotl = 0.0
            if dist > 1e-9:
                light_dir = to_light / dist
                ndotl = max(0.0, float(np.dot(fn, light_dir)))
            base_pressure = 1.0 - ndotl
            if prev_pressure is None:
                pressure = base_pressure
            else:
                pressure = prev_pressure * 0.7 + base_pressure * 0.3
            pressure += (random.random() * 2.0 - 1.0) * pressure_variation
            pressure = max(0.05, min(1.0, pressure))
            prev_pressure = pressure

            # ── Stroke direction with inertia ──
            raw_dir = normalize(np.cross(fn, view_dir_n))
            if float(np.linalg.norm(raw_dir)) < 1e-9:
                raw_dir = np.cross(fn, np.array([0.0, 1.0, 0.0]))
                raw_dir = normalize(raw_dir)

            if prev_dir is None:
                stroke_dir = raw_dir
            else:
                # Blend previous direction with new target direction
                blend = 1.0 - direction_inertia
                stroke_dir = prev_dir * direction_inertia + raw_dir * blend
                stroke_dir = normalize(stroke_dir)
            prev_dir = stroke_dir

            # ── Step with speed variation ──
            speed = step_size * (1.0 + (random.random() * 2.0 - 1.0) * speed_variation)
            pt_prev = pt.copy()
            pt = pt_prev + stroke_dir * speed

            # Check if we crossed into an adjacent face
            # Find which face contains the new point
            new_fi = fi
            fc = face_center(verts, face)
            fn_check = face_normal(verts, face)
            edge_dist = abs(float(np.dot(fn_check, pt - fc)))
            if edge_dist > 1e-6:
                # Point drifted off the face plane — find adjacent face
                candidates = adj.get(fi, set()) & front_faces
                best_fi = fi
                best_dist = float("inf")
                for cfi in candidates:
                    cface = polytope.faces[cfi]
                    cfn = face_normal(verts, cface)
                    cfc = face_center(verts, cface)
                    d = abs(float(np.dot(cfn, pt - cfc)))
                    if d < best_dist:
                        best_dist = d
                        best_fi = cfi
                if best_fi != fi and best_dist < 0.1:
                    fi = best_fi
                    face = polytope.faces[fi]
                    # Recompute stroke direction on new face
                    fn2 = face_normal(verts, face)
                    vd2 = camera_pos - pt
                    vlen2 = float(np.linalg.norm(vd2))
                    if vlen2 > 1e-9:
                        vd2n = vd2 / vlen2
                        new_dir = normalize(np.cross(fn2, vd2n))
                        if float(np.linalg.norm(new_dir)) > 1e-9:
                            prev_dir = new_dir * 0.5 + prev_dir * 0.5
                            prev_dir = normalize(prev_dir)

            # ── Project to 2D ──
            pts_2d = project_onto_image(
                np.array([pt_prev, pt]), camera_pos, camera_target,
            )

            # ── Stroke properties from pressure ──
            thick = 0.06 + 0.55 * pressure
            gray = 0.02 + 0.50 * pressure

            all_strokes.append((
                (float(pts_2d[0, 0]), float(pts_2d[0, 1])),
                (float(pts_2d[1, 0]), float(pts_2d[1, 1])),
                thick, gray,
            ))

    return all_strokes


def write_freehand_pdf(
    path: Path,
    shape_name: str,
    polytope: Polytope,
    strokes: list,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    light_pos: np.ndarray,
    num_paths: int,
    *,
    image_size: tuple[float, float] = (800.0, 600.0),
) -> None:
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(18.0)

    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"{shape_name} — freehand burin engraving",
             font="Helvetica-Bold", size=13.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             f"{len(strokes)} strokes from {num_paths} paths"
             f" | burin dynamics with inertia + pressure variation",
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

    # Dark field
    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)
    front_faces, _, _ = classify_faces(polytope, camera_pos)
    for fi in front_faces:
        face = polytope.faces[fi]
        pts = [mp(float(projected[vid, 0]), float(projected[vid, 1])) for vid in face]
        if len(pts) < 3:
            continue
        pdf.save_state()
        pdf.content.set_fill_gray(0.93)
        pdf.content.set_stroke_gray(1.0)
        pdf.content.move_to(*pts[0])
        for p in pts[1:]:
            pdf.content.line_to(*p)
        pdf.content.close_path()
        pdf.content.fill()
        pdf.restore_state()

    # Strokes
    for (x0s, y0s), (x1s, y1s), thick, gray in strokes:
        x0p, y0p = mp(x0s, y0s)
        x1p, y1p = mp(x1s, y1s)
        pdf.save_state()
        pdf.line_width(max(0.05, thick * scale * 0.5))
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
             "freehand burin: continuous paths with inertia, variable pressure, "
             "face-crossing.  The hand follows the form.",
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
    parser.add_argument("--light-y", type=float, default=4.0)
    parser.add_argument("--light-z", type=float, default=3.0)
    parser.add_argument("--paths", type=int, default=60,
                        help="number of burin paths")
    parser.add_argument("--steps", type=int, default=40,
                        help="steps per path")
    parser.add_argument("--inertia", type=float, default=0.8,
                        help="direction inertia (0=no memory, 1=rigid)")
    parser.add_argument("--pressure-var", type=float, default=0.3,
                        help="pressure variation (0=uniform, 0.3=expressive)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polytope = make_polytope(args.shape)
    camera_pos = np.array([args.camera_x, args.camera_y, args.camera_z], dtype=float)
    camera_target = np.zeros(3, dtype=float)
    light_pos = np.array([args.light_x, args.light_y, args.light_z], dtype=float)

    strokes = freehand_strokes(
        polytope, camera_pos, camera_target,
        light_pos=light_pos,
        num_paths=args.paths, steps_per_path=args.steps,
        direction_inertia=args.inertia,
        pressure_variation=args.pressure_var,
        seed=args.seed,
    )

    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"freehand_{args.shape}_{args.paths}p_{args.steps}s"
    pdf_path = out_dir / f"{stem}.pdf"

    write_freehand_pdf(
        pdf_path, args.shape, polytope, strokes, camera_pos, camera_target,
        light_pos, args.paths,
    )

    print(f"shape       {args.shape}")
    print(f"paths       {args.paths}")
    print(f"strokes     {len(strokes)}")
    print(f"inertia     {args.inertia}")
    print(f"pdf         {pdf_path}")


if __name__ == "__main__":
    main()
