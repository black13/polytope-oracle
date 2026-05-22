#!/usr/bin/env python3
"""Monte Carlo line engraving — progressive refinement via random sampling.

Each stroke is a random sample on the visible surface.  The probability
of placing a stroke at a point is proportional to:
    P(stroke) ∝ 1 − max(0, n·l) / d²

where d is the distance from the point light source and n·l is the
Lambertian dot product.  Strokes near the light are rare and thin
(eroded by brightness).  Strokes in shadow are dense and dark.

The Monte Carlo approach means:
  - Lines are not a grid — they emerge from the probability field
  - Progressive refinement: increase samples for denser engraving
  - Natural variation: each run produces a different but valid drawing
  - The field adapts to both light position AND camera position
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
from proximity_engraving import (
    classify_faces,
    project_onto_image,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


def sample_point_on_face(
    vertices: np.ndarray,
    face: list[int],
) -> tuple[np.ndarray, float]:
    """Uniformly sample a random 3D point on a convex polygon face.

    Uses triangle-fan sampling: pick a random triangle within the
    fan, then a random barycentric coordinate within that triangle.

    Returns (point_3d, area_of_face).
    """
    n = len(face)
    a = vertices[face[0]]
    # Compute triangle areas for fan
    tri_areas = []
    total_area = 0.0
    for i in range(1, n - 1):
        b = vertices[face[i]]
        c = vertices[face[i + 1]]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        tri_areas.append(area)
        total_area += area

    if total_area < 1e-12:
        return a.copy(), 0.0

    # Pick triangle proportional to area
    r = random.random() * total_area
    cumulative = 0.0
    tri_idx = 0
    for i, area in enumerate(tri_areas):
        cumulative += area
        if r <= cumulative:
            tri_idx = i
            break

    # Barycentric coordinates within triangle
    u = random.random()
    v = random.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v

    b = vertices[face[tri_idx + 1]]
    c = vertices[face[tri_idx + 2]]
    pt = a + u * (b - a) + v * (c - a)
    return pt, total_area


def monte_carlo_strokes(
    polytope: Polytope,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    *,
    light_pos: np.ndarray = None,
    num_samples: int = 2000,
    stroke_length: float = 20.0,
    falloff_exponent: float = 2.0,
    density_curve: float = 1.0,
    angle_jitter: float = 0.0,
    length_jitter: float = 0.0,
    view_weight_strength: float = 0.7,
    seed: int = 42,
) -> list[tuple[tuple[float, float], tuple[float, float], float, float]]:
    """Generate strokes via Monte Carlo sampling.

    Parameters
    ----------
    light_pos : np.ndarray (3,)
        Position of the point light source.
    num_samples : int
        Number of random samples.  More = denser.
    stroke_length : float
        Nominal stroke length in image pixels.
    falloff_exponent : float
        Exponent for distance falloff.  2.0 = 1/r² (physical).
        1.0 = 1/r (softer falloff).  3.0 = 1/r³ (sharper falloff).
    density_curve : float
        Nonlinear mapping from light→probability.
        1.0 = linear.  >1 = more contrast (shadows darker, lit areas brighter).
        <1 = flatter (less variation between lit and shadow).
    angle_jitter : float
        Random rotation (degrees) added to each stroke direction.
        0 = perfectly aligned with projected normal.
        2−5 = subtle hand-drawn wobble.
    length_jitter : float
        Fractional random variation in stroke length.
        0 = uniform length.  0.3 = ±30% variation.
    view_weight_strength : float
        How much view obliquity affects stroke probability.
        0 = ignore view direction.  1.0 = maximum view sensitivity.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    if light_pos is None:
        light_pos = np.array([1.0, 4.0, 3.0], dtype=float)

    verts = polytope.vertices
    front_faces, _, _ = classify_faces(polytope, camera_pos)

    face_areas: dict[int, float] = {}
    total_visible_area = 0.0
    for fi in front_faces:
        _, area = sample_point_on_face(verts, polytope.faces[fi])
        face_areas[fi] = area
        total_visible_area += area

    if total_visible_area < 1e-12:
        return []

    all_strokes: list[tuple[tuple[float, float], tuple[float, float], float, float]] = []

    for _ in range(num_samples):
        # Importance-sample a face proportional to area
        r = random.random() * total_visible_area
        cumulative = 0.0
        chosen_fi = next(iter(front_faces))
        for fi in front_faces:
            cumulative += face_areas[fi]
            if r <= cumulative:
                chosen_fi = fi
                break

        face = polytope.faces[chosen_fi]
        pt_3d, _ = sample_point_on_face(verts, face)
        fn = face_normal(verts, face)

        # ── Light intensity: Lambert × 1/r^falloff_exponent ──
        to_light = light_pos - pt_3d
        dist = float(np.linalg.norm(to_light))
        if dist < 1e-9:
            light_intensity = 1.0
        else:
            light_dir = to_light / dist
            ndotl = max(0.0, float(np.dot(fn, light_dir)))
            light_intensity = ndotl / (dist ** falloff_exponent)

        # ── View obliquity ──
        view_dir = camera_pos - pt_3d
        vlen = float(np.linalg.norm(view_dir))
        if vlen < 1e-9:
            continue
        view_dir_n = view_dir / vlen
        ndotv = max(0.01, abs(float(np.dot(fn, view_dir_n))))
        view_weight = 1.0 - ndotv * view_weight_strength

        # ── Stroke probability with density curve ──
        raw_prob = max(0.0, min(1.0, (1.0 - light_intensity) * view_weight))
        stroke_prob = raw_prob ** (1.0 / max(0.1, density_curve))
        stroke_prob = max(0.01, min(1.0, stroke_prob))

        if random.random() > stroke_prob:
            continue

        # ── Stroke direction with angular jitter ──
        sd3 = normalize(np.cross(fn, view_dir_n))
        if float(np.linalg.norm(sd3)) < 1e-9:
            sd3 = np.cross(fn, np.array([0.0, 1.0, 0.0]))
        if angle_jitter > 0.0:
            jitter_rad = math.radians(angle_jitter * (random.random() * 2.0 - 1.0))
            cos_j, sin_j = math.cos(jitter_rad), math.sin(jitter_rad)
            axis = fn
            sd3 = (sd3 * cos_j + np.cross(axis, sd3) * sin_j +
                   axis * float(np.dot(axis, sd3)) * (1.0 - cos_j))
            sd3 = normalize(sd3)

        # ── Stroke length with random variation ──
        length_factor = 1.0
        if length_jitter > 0.0:
            length_factor = 1.0 + (random.random() * 2.0 - 1.0) * length_jitter
        half_len = stroke_length * stroke_prob * 0.5 * length_factor

        pt_a = pt_3d - sd3 * half_len * 0.02
        pt_b = pt_3d + sd3 * half_len * 0.02
        pts_2d = project_onto_image(
            np.array([pt_a, pt_b]), camera_pos, camera_target,
        )

        # ── Thickness and gray ──
        thickness = 0.06 + 0.50 * (1.0 - stroke_prob)
        gray = 0.02 + 0.52 * (1.0 - stroke_prob)

        all_strokes.append((
            (float(pts_2d[0, 0]), float(pts_2d[0, 1])),
            (float(pts_2d[1, 0]), float(pts_2d[1, 1])),
            thickness, gray,
        ))

    return all_strokes


def write_monte_carlo_pdf(
    path: Path,
    shape_name: str,
    polytope: Polytope,
    strokes: list,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    light_pos: np.ndarray,
    num_samples: int,
    *,
    image_size: tuple[float, float] = (800.0, 600.0),
) -> None:
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(18.0)

    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"{shape_name} — Monte Carlo light erosion",
             font="Helvetica-Bold", size=13.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             f"{len(strokes)} strokes from {num_samples} samples"
             f" | light at ({light_pos[0]:.1f},{light_pos[1]:.1f},{light_pos[2]:.1f})"
             f" | 1/r² falloff",
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

    # Dark field background
    verts = polytope.vertices
    projected = project_onto_image(verts, camera_pos, camera_target)
    front_faces, _, _ = classify_faces(polytope, camera_pos)
    for fi in front_faces:
        face = polytope.faces[fi]
        pts = [mp(float(projected[vid, 0]), float(projected[vid, 1])) for vid in face]
        if len(pts) < 3:
            continue
        pdf.save_state()
        pdf.content.set_fill_gray(0.94)
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
             "Monte Carlo line placement: each stroke is a random surface sample. "
             "Light erodes darkness via 1/r² falloff.",
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
    parser.add_argument("--samples", type=int, default=2000,
                        help="Monte Carlo samples (more = denser)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--falloff", type=float, default=2.0,
                        help="light falloff exponent (2.0 = 1/r^2 physical)")
    parser.add_argument("--density-curve", type=float, default=1.0,
                        help="nonlinear contrast (>1 = more contrast)")
    parser.add_argument("--angle-jitter", type=float, default=0.0,
                        help="random rotation degrees per stroke (2-5 = hand-drawn)")
    parser.add_argument("--length-jitter", type=float, default=0.0,
                        help="random length variation fraction (0.3 = +/-30%)")
    parser.add_argument("--view-weight", type=float, default=0.7,
                        help="view obliquity influence (0=none, 1=max)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polytope = make_polytope(args.shape)
    camera_pos = np.array([args.camera_x, args.camera_y, args.camera_z], dtype=float)
    camera_target = np.zeros(3, dtype=float)
    light_pos = np.array([args.light_x, args.light_y, args.light_z], dtype=float)

    strokes = monte_carlo_strokes(
        polytope, camera_pos, camera_target,
        light_pos=light_pos, num_samples=args.samples, seed=args.seed,
        falloff_exponent=args.falloff, density_curve=args.density_curve,
        angle_jitter=args.angle_jitter, length_jitter=args.length_jitter,
        view_weight_strength=args.view_weight,
    )

    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"montecarlo_{args.shape}_{args.samples}s"
    pdf_path = out_dir / f"{stem}.pdf"

    write_monte_carlo_pdf(
        pdf_path, args.shape, polytope, strokes, camera_pos, camera_target,
        light_pos, args.samples,
    )

    print(f"shape       {args.shape}")
    print(f"samples     {args.samples}")
    print(f"strokes     {len(strokes)}")
    print(f"seed        {args.seed}")
    print(f"pdf         {pdf_path}")


if __name__ == "__main__":
    main()
