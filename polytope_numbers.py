#!/usr/bin/env python3
"""
Build and inspect convex polytope numbers in Python.

This script keeps the model numeric:

1. Construct a regular polytope from explicit vertex/face numbers.
2. Define a closed direction curve u(t) on the unit sphere.
3. Shoot rays from the polytope center along u(t).
4. Intersect each ray with the exterior polytope face it reaches first.
5. Export:
   - JSON with vertices, faces, curve points, and progressive face order
   - OBJ with the mesh, the curve on the polytope, and the face-order walk

The core representation is:

    boundary(u) = center + rho(u) * u

where u is a unit direction and rho(u) is the hit distance from the center
to the polytope boundary in that direction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from pure_pdf import A4_PORTRAIT, PurePDF, mm_to_pts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"
EPS = 1.0e-9


@dataclass
class Polytope:
    name: str
    vertices: np.ndarray
    faces: list[list[int]]
    center: np.ndarray
    sphere_radius: float


@dataclass
class FaceHit:
    face_id: int
    point: np.ndarray
    distance: float
    normal: np.ndarray


@dataclass
class FaceGroup:
    face_id: int
    sample_indices: list[int]


@dataclass
class ThreadCurve:
    curve_id: str
    directions: np.ndarray
    points: np.ndarray
    distances: np.ndarray
    face_ids: list[int]
    progressive_groups: list[FaceGroup]


@dataclass
class MeshBuffer:
    vertices: list[np.ndarray]
    triangles: list[tuple[int, int, int]]


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top(self) -> float:
        return self.y + self.height


@dataclass
class Viewport:
    rect: Rect
    min_x: float
    min_y: float
    scale: float
    offset_x: float
    offset_y: float

    def map(self, pts: np.ndarray) -> np.ndarray:
        x = self.rect.x + self.offset_x + (pts[..., 0] - self.min_x) * self.scale
        y = self.rect.y + self.offset_y + (pts[..., 1] - self.min_y) * self.scale
        return np.stack([x, y], axis=-1)


@dataclass
class ProjectionSpec:
    project: Callable[[np.ndarray], np.ndarray]
    rotation: np.ndarray
    screen_axes: tuple[int, int]
    depth_axis: int


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("cannot normalize near-zero vector")
    return v / n


def face_center(vertices: np.ndarray, face: Iterable[int]) -> np.ndarray:
    return vertices[list(face)].mean(axis=0)


def face_normal(vertices: np.ndarray, face: list[int]) -> np.ndarray:
    p0 = vertices[face[0]]
    normal = np.zeros(3, dtype=float)
    for i in range(1, len(face) - 1):
        normal += np.cross(vertices[face[i]] - p0, vertices[face[i + 1]] - p0)
    return normalize(normal)


def orient_faces_outward(vertices: np.ndarray, faces: list[list[int]],
                         center: np.ndarray) -> list[list[int]]:
    oriented: list[list[int]] = []
    for face in faces:
        candidate = list(face)
        normal = face_normal(vertices, candidate)
        to_face = face_center(vertices, candidate) - center
        if float(np.dot(normal, to_face)) < 0.0:
            candidate.reverse()
        oriented.append(candidate)
    return oriented


def build_polytope(name: str, vertices: np.ndarray,
                   faces: list[list[int]]) -> Polytope:
    center = vertices.mean(axis=0)
    centered_vertices = vertices - center
    center = np.zeros(3, dtype=float)
    radius = float(np.max(np.linalg.norm(centered_vertices, axis=1)))
    oriented_faces = orient_faces_outward(centered_vertices, faces, center)
    return Polytope(
        name=name,
        vertices=centered_vertices,
        faces=oriented_faces,
        center=center,
        sphere_radius=radius,
    )


def make_cube() -> Polytope:
    vertices = np.array([
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ], dtype=float) / math.sqrt(3.0)
    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return build_polytope("cube", vertices, faces)


def make_octahedron() -> Polytope:
    vertices = np.array([
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ], dtype=float)
    faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]
    return build_polytope("octahedron", vertices, faces)


def make_icosahedron() -> Polytope:
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    vertices = np.array([
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ], dtype=float)
    vertices /= np.linalg.norm(vertices[0])
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    return build_polytope("icosahedron", vertices, faces)


def make_polytope(shape: str) -> Polytope:
    if shape == "cube":
        return make_cube()
    if shape == "octahedron":
        return make_octahedron()
    if shape == "icosahedron":
        return make_icosahedron()
    raise ValueError(f"unsupported shape: {shape}")


def rescale_polytope(polytope: Polytope, radius: float) -> Polytope:
    if radius <= 0.0:
        raise ValueError("normalize radius must be positive")
    scale = radius / polytope.sphere_radius
    return Polytope(
        name=polytope.name,
        vertices=polytope.vertices * scale,
        faces=[list(face) for face in polytope.faces],
        center=polytope.center.copy(),
        sphere_radius=radius,
    )


def load_polytope_json(path: Path) -> Polytope:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "vertices" not in payload or "faces" not in payload:
        raise ValueError("polytope json must contain 'vertices' and 'faces'")
    name = payload.get("name", path.stem)
    vertices = np.array(payload["vertices"], dtype=float)
    faces = [list(map(int, face)) for face in payload["faces"]]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be an Nx3 array")
    if not faces:
        raise ValueError("faces must not be empty")
    return build_polytope(name, vertices, faces)


def point_in_convex_face(point: np.ndarray, vertices: np.ndarray,
                         face: list[int], normal: np.ndarray) -> bool:
    count = len(face)
    for i in range(count):
        a = vertices[face[i]]
        b = vertices[face[(i + 1) % count]]
        edge = b - a
        inward_test = float(np.dot(normal, np.cross(edge, point - a)))
        if inward_test < -1.0e-8:
            return False
    return True


def ray_face_intersection(origin: np.ndarray, direction: np.ndarray,
                          vertices: np.ndarray, face: list[int]) -> FaceHit | None:
    normal = face_normal(vertices, face)
    denom = float(np.dot(normal, direction))
    if denom <= EPS:
        return None

    anchor = vertices[face[0]]
    numer = float(np.dot(normal, anchor - origin))
    if numer <= EPS:
        return None

    distance = numer / denom
    if distance <= EPS:
        return None

    point = origin + distance * direction
    if not point_in_convex_face(point, vertices, face, normal):
        return None

    return FaceHit(face_id=-1, point=point, distance=distance, normal=normal)


def ray_polytope_intersection(polytope: Polytope, direction: np.ndarray) -> FaceHit:
    best: FaceHit | None = None
    for face_id, face in enumerate(polytope.faces):
        hit = ray_face_intersection(polytope.center, direction,
                                    polytope.vertices, face)
        if hit is None:
            continue
        hit.face_id = face_id
        if best is None or hit.distance < best.distance:
            best = hit

    if best is None:
        raise RuntimeError("ray did not hit any polytope face")
    return best


def sample_direction_curve(samples: int, amplitude: float,
                           frequency: int, phase: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    raw = np.stack([
        np.cos(t + phase),
        np.sin(t + phase),
        amplitude * np.sin(frequency * t),
    ], axis=1)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def family_rotation(axis_name: str) -> np.ndarray:
    if axis_name == "z":
        return np.eye(3, dtype=float)
    if axis_name == "x":
        return np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
    if axis_name == "y":
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
    raise ValueError(f"unsupported thread family axis: {axis_name}")


def group_progressive_faces(face_ids: list[int]) -> list[FaceGroup]:
    if not face_ids:
        return []

    groups: list[FaceGroup] = [FaceGroup(face_id=face_ids[0], sample_indices=[0])]
    for sample_index, face_id in enumerate(face_ids[1:], start=1):
        if face_id == groups[-1].face_id:
            groups[-1].sample_indices.append(sample_index)
        else:
            groups.append(FaceGroup(face_id=face_id, sample_indices=[sample_index]))

    if len(groups) > 1 and groups[0].face_id == groups[-1].face_id:
        groups[0].sample_indices = groups[-1].sample_indices + groups[0].sample_indices
        groups.pop()

    return groups


def front_back_labels(points: np.ndarray, center: np.ndarray,
                      view_direction: np.ndarray) -> list[str]:
    labels: list[str] = []
    for point in points:
        value = float(np.dot(point - center, view_direction))
        if value > 1.0e-8:
            labels.append("front")
        elif value < -1.0e-8:
            labels.append("back")
        else:
            labels.append("rim")
    return labels


def triangulate_face(face: list[int]) -> list[tuple[int, int, int]]:
    return [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]


def rot_x(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_projection_spec(rot_x_deg: float, rot_z_deg: float,
                         screen_axes: tuple[int, int] = (0, 2)) -> ProjectionSpec:
    rotation = rot_x(rot_x_deg) @ rot_z(rot_z_deg)
    axes = list(screen_axes)
    depth_axis = next(axis for axis in range(3) if axis not in axes)

    def project(points: np.ndarray) -> np.ndarray:
        return (points @ rotation.T)[..., axes]

    return ProjectionSpec(
        project=project,
        rotation=rotation,
        screen_axes=tuple(screen_axes),
        depth_axis=depth_axis,
    )


def make_projector(rot_x_deg: float, rot_z_deg: float,
                   screen_axes: tuple[int, int] = (0, 2)):
    return make_projection_spec(rot_x_deg, rot_z_deg, screen_axes).project


def compute_bounds(*panels: np.ndarray, pad: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate([panel.reshape(-1, 2) for panel in panels], axis=0)
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-6)
    return mins - pad * span, maxs + pad * span


def make_viewport(rect: Rect, *panels: np.ndarray, pad: float = 0.12) -> Viewport:
    mins, maxs = compute_bounds(*panels, pad=pad)
    span = np.maximum(maxs - mins, 1.0e-6)
    scale = min(rect.width / span[0], rect.height / span[1])
    offset_x = 0.5 * (rect.width - span[0] * scale)
    offset_y = 0.5 * (rect.height - span[1] * scale)
    return Viewport(
        rect=rect,
        min_x=float(mins[0]),
        min_y=float(mins[1]),
        scale=float(scale),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
    )


def polytope_edges(polytope: Polytope) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in polytope.faces:
        for i, start in enumerate(face):
            end = face[(i + 1) % len(face)]
            edges.add(tuple(sorted((start, end))))
    return sorted(edges)


def polytope_edge_faces(polytope: Polytope) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(polytope.faces):
        for i, start in enumerate(face):
            end = face[(i + 1) % len(face)]
            key = tuple(sorted((start, end)))
            edge_faces.setdefault(key, []).append(face_id)
    return edge_faces


def visible_face_ids(polytope: Polytope, projection: ProjectionSpec) -> set[int]:
    visible: set[int] = set()
    for face_id, face in enumerate(polytope.faces):
        normal = face_normal(polytope.vertices, face)
        transformed = projection.rotation @ normal
        if float(transformed[projection.depth_axis]) > 1.0e-8:
            visible.add(face_id)
    return visible


def rotation_minimizing_frames(points: np.ndarray, closed: bool) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    count = len(points)
    tangents = []
    for i in range(count):
        prev_point = points[(i - 1) % count] if closed else points[max(i - 1, 0)]
        next_point = points[(i + 1) % count] if closed else points[min(i + 1, count - 1)]
        tangents.append(normalize(next_point - prev_point))

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(reference, tangents[0]))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    normal = normalize(np.cross(tangents[0], reference))
    binormal = normalize(np.cross(tangents[0], normal))

    frames = [(tangents[0], normal.copy(), binormal.copy())]
    for i in range(1, count):
        tangent = tangents[i]
        prev_tangent, prev_normal, _ = frames[-1]
        axis = np.cross(prev_tangent, tangent)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= EPS:
            normal = prev_normal
        else:
            axis /= axis_norm
            angle = math.acos(float(np.clip(np.dot(prev_tangent, tangent), -1.0, 1.0)))
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            normal = (
                prev_normal * cos_a +
                np.cross(axis, prev_normal) * sin_a +
                axis * float(np.dot(axis, prev_normal)) * (1.0 - cos_a)
            )
            normal = normalize(normal - tangent * float(np.dot(tangent, normal)))
        binormal = normalize(np.cross(tangent, normal))
        frames.append((tangent, normal.copy(), binormal.copy()))

    return frames


def build_tube(points: np.ndarray, radius: float, sides: int = 12,
               closed: bool = True) -> MeshBuffer:
    if len(points) < 2:
        return MeshBuffer(vertices=[], triangles=[])

    frames = rotation_minimizing_frames(points, closed=closed)
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []

    for point, (_, normal, binormal) in zip(points, frames):
        for j in range(sides):
            theta = 2.0 * math.pi * j / sides
            offset = radius * (math.cos(theta) * normal + math.sin(theta) * binormal)
            vertices.append(point + offset)

    ring_count = len(points)
    segment_count = ring_count if closed else ring_count - 1
    for ring in range(segment_count):
        next_ring = (ring + 1) % ring_count
        for side in range(sides):
            next_side = (side + 1) % sides
            a = ring * sides + side
            b = ring * sides + next_side
            c = next_ring * sides + next_side
            d = next_ring * sides + side
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    return MeshBuffer(vertices=vertices, triangles=triangles)


def pdf_polyline(pdf: PurePDF, pts: np.ndarray, width: float, gray: float) -> None:
    if len(pts) < 2:
        return
    pdf.save_state()
    pdf.line_width(width)
    pdf.stroke_gray(gray)
    pdf.content.set_line_cap(1)
    pdf.content.set_line_join(1)
    pdf.content.polyline([(float(p[0]), float(p[1])) for p in pts])
    pdf.content.stroke()
    pdf.restore_state()


def visible_thread_runs(thread: ThreadCurve, visible_faces: set[int]) -> list[np.ndarray]:
    visible_mask = [face_id in visible_faces for face_id in thread.face_ids]
    if not any(visible_mask):
        return []

    runs: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for point, visible in zip(thread.points, visible_mask):
        if visible:
            current.append(point)
            continue
        if len(current) >= 2:
            runs.append(np.array(current, dtype=float))
        current = []

    if len(current) >= 2:
        runs.append(np.array(current, dtype=float))

    if not runs:
        return []

    wraps = visible_mask[0] and visible_mask[-1]
    if all(visible_mask):
        runs[0] = np.vstack([runs[0], runs[0][0]])
        return runs

    if wraps and len(runs) >= 2:
        merged = np.vstack([runs[-1], runs[0]])
        runs = [merged] + runs[1:-1]
    return runs


def draw_pdf_panel(pdf: PurePDF, rect: Rect, title: str, polytope: Polytope,
                   thread_curves: list[ThreadCurve], projection: ProjectionSpec,
                   edge_width_pt: float, thread_width_pt: float) -> None:
    title_y = rect.top - mm_to_pts(5.5)
    pdf.text(rect.x, title_y, title, font="Helvetica-Bold", size=10.0, gray=0.0)

    content_rect = Rect(
        x=rect.x,
        y=rect.y,
        width=rect.width,
        height=max(rect.height - mm_to_pts(9.0), mm_to_pts(8.0)),
    )

    visible_faces = visible_face_ids(polytope, projection)

    edge_segments = []
    for (start, end), faces in polytope_edge_faces(polytope).items():
        if any(face_id in visible_faces for face_id in faces):
            edge_segments.append(projection.project(polytope.vertices[[start, end]]))

    thread_panels = []
    for thread in thread_curves:
        for run in visible_thread_runs(thread, visible_faces):
            thread_panels.append(projection.project(run))

    viewport = make_viewport(content_rect, *edge_segments, *thread_panels, pad=0.10)
    mapped_edges = [viewport.map(segment) for segment in edge_segments]
    mapped_threads = [viewport.map(panel) for panel in thread_panels]

    pdf.save_state()
    pdf.clip_rect(content_rect.x, content_rect.y, content_rect.width, content_rect.height)
    for thread in mapped_threads:
        pdf_polyline(pdf, thread, width=thread_width_pt, gray=0.08)
    for edge in mapped_edges:
        pdf_polyline(pdf, edge, width=edge_width_pt, gray=0.72)
    pdf.restore_state()

    # light border so the panel reads as a proof frame, not page noise
    pdf.save_state()
    pdf.line_width(0.35)
    pdf.stroke_gray(0.82)
    pdf.content.rect(content_rect.x, content_rect.y, content_rect.width, content_rect.height)
    pdf.content.stroke()
    pdf.restore_state()


def write_pdf(path: Path, polytope: Polytope, thread_curves: list[ThreadCurve],
              edge_width_pt: float = 0.28, thread_width_pt: float = 0.20,
              layout: str = "quad") -> None:
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")

    margin = mm_to_pts(12.0)
    if layout == "isometric":
        pdf.text(
            margin,
            page_h - margin + mm_to_pts(2.0),
            f"{polytope.name} thin thread proof",
            font="Helvetica-Bold",
            size=13.0,
            gray=0.0,
        )
        pdf.text(
            margin,
            page_h - margin - mm_to_pts(3.5),
            "Large isometric view | source is numeric model",
            font="Helvetica",
            size=8.5,
            gray=0.25,
        )
        rect = Rect(
            x=margin,
            y=margin,
            width=page_w - 2.0 * margin,
            height=page_h - 2.0 * margin - mm_to_pts(11.0),
        )
        draw_pdf_panel(
            pdf,
            rect,
            "Isometric",
            polytope,
            thread_curves,
            make_projection_spec(32.0, -36.0),
            edge_width_pt=edge_width_pt,
            thread_width_pt=thread_width_pt,
        )
    else:
        gap = mm_to_pts(6.0)
        title_h = mm_to_pts(11.0)
        panel_w = 0.5 * (page_w - 2.0 * margin - gap)
        panel_h = 0.5 * (page_h - 2.0 * margin - gap - title_h)

        pdf.text(
            margin,
            page_h - margin + mm_to_pts(2.0),
            f"{polytope.name} thin thread proof",
            font="Helvetica-Bold",
            size=13.0,
            gray=0.0,
        )
        pdf.text(
            margin,
            page_h - margin - mm_to_pts(3.5),
            f"{len(thread_curves)} thread curves | source is numeric model",
            font="Helvetica",
            size=8.5,
            gray=0.25,
        )

        panels = [
            ("Isometric", make_projection_spec(32.0, -36.0)),
            ("Top / XY", make_projection_spec(0.0, 0.0, screen_axes=(0, 1))),
            ("Front / XZ", make_projection_spec(0.0, 0.0, screen_axes=(0, 2))),
            ("Side / YZ", make_projection_spec(0.0, 90.0, screen_axes=(0, 2))),
        ]

        for index, (title, projection) in enumerate(panels):
            col = index % 2
            row = index // 2
            rect = Rect(
                x=margin + col * (panel_w + gap),
                y=margin + (1 - row) * (panel_h + gap),
                width=panel_w,
                height=panel_h,
            )
            draw_pdf_panel(
                pdf,
                rect,
                title,
                polytope,
                thread_curves,
                projection,
                edge_width_pt=edge_width_pt,
                thread_width_pt=thread_width_pt,
            )
    pdf.save(str(path))


def write_mtl(path: Path) -> None:
    path.write_text(
        "\n".join([
            "newmtl polytope_body",
            "Ka 0.15 0.15 0.15",
            "Kd 0.78 0.80 0.84",
            "Ks 0.05 0.05 0.05",
            "Ns 16.0",
            "",
            "newmtl boundary_curve",
            "Ka 0.20 0.02 0.02",
            "Kd 0.92 0.18 0.18",
            "Ks 0.10 0.10 0.10",
            "Ns 24.0",
            "",
            "newmtl face_walk",
            "Ka 0.02 0.08 0.20",
            "Kd 0.18 0.50 0.92",
            "Ks 0.10 0.10 0.10",
            "Ns 24.0",
            "",
        ]),
        encoding="utf-8",
    )


def write_obj(path: Path, polytope: Polytope, thread_curves: list[ThreadCurve],
              walk_groups: list[FaceGroup], thread_radius: float,
              walk_radius: float) -> None:
    if not thread_curves:
        raise ValueError("expected at least one thread curve")

    face_reprs = np.array([
        face_center(polytope.vertices, polytope.faces[group.face_id])
        for group in walk_groups
    ], dtype=float)
    thread_tubes = [
        (thread.curve_id,
         build_tube(thread.points, radius=thread_radius * polytope.sphere_radius,
                    sides=10, closed=True))
        for thread in thread_curves
    ]
    walk_tube = build_tube(face_reprs, radius=walk_radius * polytope.sphere_radius,
                           sides=8, closed=len(face_reprs) >= 3)
    mtl_path = path.with_suffix(".mtl")
    write_mtl(mtl_path)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# polytope numbers export\n")
        handle.write(f"mtllib {mtl_path.name}\n")
        handle.write(f"o {polytope.name}\n")
        handle.write("usemtl polytope_body\n")
        for vertex in polytope.vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in polytope.faces:
            for tri in triangulate_face(face):
                handle.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

        next_vertex_index = len(polytope.vertices) + 1
        handle.write("usemtl boundary_curve\n")
        for curve_id, thread_tube in thread_tubes:
            handle.write(f"o {curve_id}\n")
            curve_start = next_vertex_index
            for vertex in thread_tube.vertices:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for tri in thread_tube.triangles:
                handle.write(
                    f"f {curve_start + tri[0]} {curve_start + tri[1]} {curve_start + tri[2]}\n"
                )
            next_vertex_index += len(thread_tube.vertices)

        handle.write("o face_walk\n")
        handle.write("usemtl face_walk\n")
        face_walk_start = next_vertex_index
        for vertex in walk_tube.vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for tri in walk_tube.triangles:
            handle.write(
                f"f {face_walk_start + tri[0]} {face_walk_start + tri[1]} {face_walk_start + tri[2]}\n"
            )

        for thread in thread_curves:
            handle.write(f"# {thread.curve_id}_faces")
            for face_id in thread.face_ids:
                handle.write(f" {face_id}")
            handle.write("\n")


def write_json(path: Path, polytope: Polytope, thread_curves: list[ThreadCurve],
               view_direction: np.ndarray) -> None:
    face_data = []
    for face_id, face in enumerate(polytope.faces):
        face_data.append({
            "face_id": face_id,
            "vertex_indices": face,
            "center": face_center(polytope.vertices, face).tolist(),
            "normal": face_normal(polytope.vertices, face).tolist(),
        })

    all_distances = np.concatenate([thread.distances for thread in thread_curves])
    radial_error = polytope.sphere_radius - all_distances
    thread_data = []
    for thread in thread_curves:
        progressive_data = []
        for group in thread.progressive_groups:
            progressive_data.append({
                "face_id": group.face_id,
                "sample_indices": group.sample_indices,
                "face_center": face_center(
                    polytope.vertices, polytope.faces[group.face_id]
                ).tolist(),
            })

        thread_data.append({
            "curve_id": thread.curve_id,
            "sample_count": int(len(thread.directions)),
            "directions": thread.directions.tolist(),
            "points": thread.points.tolist(),
            "distances": thread.distances.tolist(),
            "front_back": front_back_labels(
                thread.points, polytope.center, view_direction
            ),
            "face_ids": thread.face_ids,
            "progressive_faces": progressive_data,
        })

    payload = {
        "shape": polytope.name,
        "center": polytope.center.tolist(),
        "bounding_sphere_radius": polytope.sphere_radius,
        "vertex_count": int(len(polytope.vertices)),
        "face_count": int(len(polytope.faces)),
        "vertices": polytope.vertices.tolist(),
        "faces": face_data,
        "thread_curves": thread_data,
        "radius_stats": {
            "min_distance": float(np.min(all_distances)),
            "max_distance": float(np.max(all_distances)),
            "min_gap_to_sphere": float(np.min(radial_error)),
            "max_gap_to_sphere": float(np.max(radial_error)),
            "mean_gap_to_sphere": float(np.mean(radial_error)),
        },
        "view_direction": view_direction.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cube", "octahedron", "icosahedron"),
                        default="cube")
    parser.add_argument("--polytope-json", type=Path, default=None,
                        help="load a custom polytope from JSON with vertices/faces")
    parser.add_argument("--samples", type=int, default=180,
                        help="number of samples around the closed sphere curve")
    parser.add_argument("--amplitude", type=float, default=0.35,
                        help="z wobble amplitude for the sphere direction curve")
    parser.add_argument("--frequency", type=int, default=3,
                        help="z wobble frequency for the sphere direction curve")
    parser.add_argument("--phase", type=float, default=0.17,
                        help="phase offset in radians")
    parser.add_argument("--thread-count", type=int, default=6,
                        help="number of thin thread curves per axis family")
    parser.add_argument("--thread-families", type=str, default="xyz",
                        help="axis families to thread over, chosen from x, y, z")
    parser.add_argument("--thread-radius", type=float, default=0.006,
                        help="tube radius as a fraction of the polytope sphere radius")
    parser.add_argument("--walk-radius", type=float, default=0.0035,
                        help="face-walk tube radius as a fraction of the polytope sphere radius")
    parser.add_argument("--pdf-thread-width-pt", type=float, default=0.20,
                        help="thread stroke width in the proof PDF, in points")
    parser.add_argument("--pdf-edge-width-pt", type=float, default=0.28,
                        help="edge stroke width in the proof PDF, in points")
    parser.add_argument("--pdf-layout", choices=("quad", "isometric"),
                        default="quad",
                        help="proof PDF layout")
    parser.add_argument("--normalize-radius", type=float, default=None,
                        help="rescale the polytope so its outer vertex radius matches this value")
    parser.add_argument("--view-dir", type=float, nargs=3,
                        default=(0.0, 0.0, 1.0),
                        metavar=("VX", "VY", "VZ"))
    parser.add_argument("--json", type=Path, default=None,
                        help="path for JSON output")
    parser.add_argument("--obj", type=Path, default=None,
                        help="path for OBJ output")
    parser.add_argument("--pdf", type=Path, default=None,
                        help="path for proof PDF output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polytope = (
        load_polytope_json(args.polytope_json)
        if args.polytope_json is not None
        else make_polytope(args.shape)
    )
    if args.normalize_radius is not None:
        polytope = rescale_polytope(polytope, args.normalize_radius)

    if args.samples < 3:
        raise SystemExit("--samples must be at least 3")
    if args.thread_count < 1:
        raise SystemExit("--thread-count must be at least 1")
    if args.thread_radius <= 0.0 or args.walk_radius <= 0.0:
        raise SystemExit("thread and walk radii must be positive")
    if args.pdf_thread_width_pt <= 0.0 or args.pdf_edge_width_pt <= 0.0:
        raise SystemExit("PDF stroke widths must be positive")

    view_direction = normalize(np.array(args.view_dir, dtype=float))
    thread_families = []
    for axis_name in args.thread_families:
        if axis_name not in "xyz":
            raise SystemExit("--thread-families must use only x, y, z")
        if axis_name not in thread_families:
            thread_families.append(axis_name)

    thread_curves: list[ThreadCurve] = []
    for axis_name in thread_families:
        rotation = family_rotation(axis_name)
        for thread_index in range(args.thread_count):
            local_phase = args.phase + (2.0 * math.pi * thread_index / args.thread_count)
            directions = sample_direction_curve(
                samples=args.samples,
                amplitude=args.amplitude,
                frequency=args.frequency,
                phase=local_phase,
            ) @ rotation.T
            hits = [ray_polytope_intersection(polytope, direction) for direction in directions]
            points = np.array([hit.point for hit in hits], dtype=float)
            distances = np.array([hit.distance for hit in hits], dtype=float)
            face_ids = [hit.face_id for hit in hits]
            thread_curves.append(ThreadCurve(
                curve_id=f"thread_{axis_name}_{thread_index:02d}",
                directions=directions,
                points=points,
                distances=distances,
                face_ids=face_ids,
                progressive_groups=group_progressive_faces(face_ids),
            ))

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_base = polytope.name.replace(" ", "_")
    stem = f"polytope_{stem_base}_{args.samples}s_t{args.thread_count}{''.join(thread_families)}"
    json_path = args.json or (output_dir / f"{stem}.json")
    obj_path = args.obj or (output_dir / f"{stem}.obj")
    pdf_path = args.pdf or (output_dir / f"{stem}.pdf")

    write_json(json_path, polytope, thread_curves, view_direction)
    write_obj(obj_path, polytope, thread_curves, thread_curves[0].progressive_groups,
              args.thread_radius, args.walk_radius)
    write_pdf(pdf_path, polytope, thread_curves,
              edge_width_pt=args.pdf_edge_width_pt,
              thread_width_pt=args.pdf_thread_width_pt,
              layout=args.pdf_layout)

    all_distances = np.concatenate([thread.distances for thread in thread_curves])

    print(f"shape {polytope.name}")
    print(f"vertex_count {len(polytope.vertices)}")
    print(f"face_count {len(polytope.faces)}")
    print(f"thread_curve_count {len(thread_curves)}")
    print("primary_progressive_face_ids",
          " ".join(str(group.face_id) for group in thread_curves[0].progressive_groups))
    print(f"min_distance {float(np.min(all_distances)):.6f}")
    print(f"max_distance {float(np.max(all_distances)):.6f}")
    print(f"json {json_path}")
    print(f"obj {obj_path}")
    print(f"pdf {pdf_path}")


if __name__ == "__main__":
    main()
