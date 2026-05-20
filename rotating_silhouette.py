#!/usr/bin/env python3
"""Rotating silhouette accumulator — cartographic line drawing from polytopes.

Orbits a camera around a polytope at fixed distance, extracts the
silhouette contour (occluding boundary) at each azimuth step, and
accumulates all projected curves into a single print-quality PDF.

The result is a visual hull drawing: every line is a verified occluding
contour from some viewpoint.  The density of lines reveals surface
curvature — flatter regions produce fewer silhouette transitions,
curved regions produce denser boundary accumulation.

Analogy: Piranesi's engraved views accumulate boundary lines to convey
architectural mass.  Here the same principle is applied analytically
to pure geometric form.
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
from smooth_contour import (
    _vertex_normals,
    _raw_edge_faces,
    extract_contours,
    fit_bezier_chain,
    ContourPoint,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


def orbiting_camera_positions(
    distance: float = 5.0,
    azimuth_steps: int = 72,
    elevation_angle: float = 0.0,
) -> list[np.ndarray]:
    """Generate camera positions on a circle at fixed distance.

    Parameters
    ----------
    distance : float
        Distance from the origin (polytope center).
    azimuth_steps : int
        Number of equally spaced positions around the Y axis.
    elevation_angle : float
        Elevation in degrees above the equatorial plane.
        Positive = camera looks down on the object.
        0 = camera at equator (side view).

    Returns
    -------
    list[np.ndarray]
        Camera positions (3,) in world space.
    """
    elev_rad = math.radians(elevation_angle)
    cos_elev = math.cos(elev_rad)
    sin_elev = math.sin(elev_rad)
    positions = []
    for i in range(azimuth_steps):
        azimuth = 2.0 * math.pi * i / azimuth_steps
        x = distance * math.cos(azimuth) * cos_elev
        y = distance * sin_elev
        z = distance * math.sin(azimuth) * cos_elev
        positions.append(np.array([x, y, z], dtype=float))
    return positions


def accumulate_silhouettes(
    polytope: Polytope,
    camera_positions: list[np.ndarray],
    *,
    subdivide: bool = False,
    subdiv_levels: int = 2,
) -> tuple[list[list[np.ndarray]], np.ndarray, list[list[int]]]:
    """Accumulate 3D silhouette curves from many camera positions.

    For each camera position, extracts the occluding contour (n·v = 0)
    from the polytope surface.  Returns the accumulated 3D contour
    curves in object-local space.

    If subdivide is True, Catmull-Clark subdivision is applied first
    to produce smooth contours from the limit surface.

    Returns
    -------
    (list of contour chains as N×3 arrays, vertices, faces)
    """
    from subdivide import catmull_clark, normalize_to_sphere

    verts = polytope.vertices.copy()
    faces = [list(f) for f in polytope.faces]

    if subdivide:
        verts, faces = catmull_clark(verts, faces, levels=subdiv_levels)
        verts = normalize_to_sphere(verts, faces, radius=polytope.sphere_radius)

    all_chains: list[list[np.ndarray]] = []

    for cam_pos in camera_positions:
        chains, _, _ = extract_contours(
            verts, faces, cam_pos, project_to_sphere=False,
        )
        for chain in chains:
            pts = np.array([cp.position for cp in chain], dtype=float)
            if len(pts) >= 3:
                all_chains.append(pts)

    return all_chains, verts, faces


def perspective_project_chain(
    chain_pts: np.ndarray,
    camera_pos: np.ndarray,
    screen_dist: float = 4.0,
    screen_axes: tuple[int, int] = (0, 2),
) -> np.ndarray:
    """Perspective-project a 3D contour chain to 2D.

    Projects each point along the ray from camera_pos to the point,
    onto a screen plane at distance screen_dist.
    """
    result = np.empty((len(chain_pts), 2), dtype=float)
    for i, pt in enumerate(chain_pts):
        rel = pt - camera_pos
        # Use Z as depth axis, project onto X-Z plane
        z_val = rel[2]
        if abs(z_val) < 1e-12:
            z_val = 1e-12
        result[i, 0] = screen_dist * rel[screen_axes[0]] / z_val
        result[i, 1] = screen_dist * rel[screen_axes[1]] / z_val
    return result


def write_rotating_silhouette_pdf(
    path: Path,
    shape_name: str,
    all_chains: list[list[np.ndarray]],
    camera_positions: list[np.ndarray],
    *,
    azimuth_steps: int = 72,
    elevation_angle: float = 0.0,
) -> None:
    """Write a print-quality PDF showing accumulated silhouette curves.

    Each camera position contributes one silhouette chain.  Chains
    are projected to 2D via perspective projection from their
    respective camera, then rendered as thin gray Bezier curves.
    The accumulation produces a self-consistent cartographic drawing.

    A4 portrait by default.  Use A3 for dense accumulation.
    """
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(18.0)

    pdf.text(
        margin, page_h - margin + mm_to_pts(1.0),
        f"{shape_name} — rotating silhouette accumulation",
        font="Helvetica-Bold", size=13.0, gray=0.0,
    )
    pdf.text(
        margin, page_h - margin - mm_to_pts(4.0),
        f"{azimuth_steps} orbits × {len(all_chains)} accumulated contours"
        f" | elevation={elevation_angle}°"
        f" | visual hull drawing",
        font="Helvetica", size=8.5, gray=0.28,
    )

    content_rect = (
        margin,
        margin,
        page_w - 2.0 * margin,
        page_h - 2.0 * margin - mm_to_pts(16.0),
    )

    # Project all chains for all cameras and pool them for viewport fitting
    all_2d: list[np.ndarray] = []
    for i, chain_pts in enumerate(all_chains):
        cam_pos = camera_positions[i % len(camera_positions)]
        p2d = perspective_project_chain(chain_pts, cam_pos)
        all_2d.append(p2d)

    # Fit all accumulated 2D points to the content rect
    merged = np.vstack(all_2d) if all_2d else np.empty((0, 2))
    if len(merged) == 0:
        pdf.save(str(path))
        return

    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-6)
    pad = 0.08
    mins = mins - pad * span
    maxs = maxs + pad * span
    span = np.maximum(maxs - mins, 1.0e-6)
    x0, y0, cw, ch = content_rect
    scale = min(cw / span[0], ch / span[1])
    offset_x = x0 + 0.5 * (cw - span[0] * scale)
    offset_y = y0 + 0.5 * (ch - span[1] * scale)

    # Map function
    def map_pt(x: float, y: float) -> tuple[float, float]:
        return (
            offset_x + (x - mins[0]) * scale,
            offset_y + (y - mins[1]) * scale,
        )

    # Clip to content rect
    pdf.save_state()
    pdf.clip_rect(x0, y0, cw, ch)

    # Draw each chain as a thin Bezier curve
    for p2d in all_2d:
        if len(p2d) < 2:
            continue
        # Map to page space
        mapped = np.empty_like(p2d)
        mapped[:, 0] = offset_x + (p2d[:, 0] - mins[0]) * scale
        mapped[:, 1] = offset_y + (p2d[:, 1] - mins[1]) * scale

        # Fit cubic Bezier
        beziers = fit_bezier_chain(mapped, closed=True, segments=24)
        if not beziers:
            continue

        pdf.save_state()
        pdf.line_width(0.22)
        pdf.stroke_gray(0.15)
        pdf.content.set_line_cap(1)
        pdf.content.set_line_join(1)
        for seg in beziers:
            pdf.content.move_to(float(seg[0][0]), float(seg[0][1]))
            pdf.content.curve_to(
                float(seg[1][0]), float(seg[1][1]),
                float(seg[2][0]), float(seg[2][1]),
                float(seg[3][0]), float(seg[3][1]),
            )
        pdf.content.stroke()
        pdf.restore_state()

    pdf.restore_state()  # unclip

    # Footer
    footer_y = margin - mm_to_pts(2.0)
    pdf.text(
        margin, footer_y,
        f"visual hull from {azimuth_steps} azimuth steps | "
        f"each line = verified occluding contour n·v=0 | "
        f"line density ≈ surface curvature",
        font="Helvetica", size=8.0, gray=0.35,
    )

    # Light content border
    pdf.save_state()
    pdf.line_width(0.35)
    pdf.stroke_gray(0.82)
    pdf.content.rect(x0, y0, cw, ch)
    pdf.content.stroke()
    pdf.restore_state()

    pdf.save(str(path))


def rotating_silhouette_proof(
    shape: str,
    azimuth_steps: int = 72,
    elevation_angle: float = 0.0,
    distance: float = 5.0,
    *,
    subdivide: bool = False,
    subdiv_levels: int = 2,
    output_dir: Path | None = None,
) -> Path:
    """Full pipeline: orbit camera, accumulate silhouettes, write PDF."""
    polytope = make_polytope(shape)
    cam_positions = orbiting_camera_positions(
        distance=distance,
        azimuth_steps=azimuth_steps,
        elevation_angle=elevation_angle,
    )

    all_chains, verts, faces = accumulate_silhouettes(
        polytope, cam_positions,
        subdivide=subdivide, subdiv_levels=subdiv_levels,
    )

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rotating_{shape}_{azimuth_steps}az_{elevation_angle:.0f}el"
    if subdivide:
        stem += f"_subdiv{subdiv_levels}"
    pdf_path = out_dir / f"{stem}.pdf"

    write_rotating_silhouette_pdf(
        pdf_path, shape, all_chains, cam_positions,
        azimuth_steps=azimuth_steps, elevation_angle=elevation_angle,
    )

    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cube", "octahedron", "icosahedron"),
                        default="icosahedron")
    parser.add_argument("--azimuth-steps", type=int, default=72,
                        help="number of orbit positions (default 72 = every 5°)")
    parser.add_argument("--elevation", type=float, default=0.0,
                        help="camera elevation in degrees")
    parser.add_argument("--distance", type=float, default=5.0,
                        help="camera distance from origin")
    parser.add_argument("--subdivide", action="store_true",
                        help="apply Catmull-Clark subdivision for smooth contours")
    parser.add_argument("--subdiv-levels", type=int, default=2,
                        help="subdivision levels (default 2)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = rotating_silhouette_proof(
        args.shape,
        azimuth_steps=args.azimuth_steps,
        elevation_angle=args.elevation,
        distance=args.distance,
        subdivide=args.subdivide,
        subdiv_levels=args.subdiv_levels,
        output_dir=args.output_dir,
    )
    print(f"shape             {args.shape}")
    print(f"azimuth steps     {args.azimuth_steps}")
    print(f"elevation         {args.elevation}°")
    print(f"subdivide         {args.subdivide}")
    print(f"pdf               {pdf_path}")


if __name__ == "__main__":
    main()
