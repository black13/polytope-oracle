#!/usr/bin/env python3
"""Validate trusted convex polytope definitions and write proof outputs.

This script treats the numeric polytope model as the source of truth and
derives every output from those same numbers:

- numbers JSON
- validation JSON
- OBJ mesh
- PLY mesh
- PDF proof from a trusted local camera
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pure_pdf import A4_PORTRAIT, PurePDF, mm_to_pts
from polytope_numbers import (
    EPS,
    Polytope,
    build_polytope,
    face_center,
    face_normal,
    make_polytope,
    normalize,
    point_in_convex_face,
    polytope_edge_faces,
    ray_polytope_intersection,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"


@dataclass(frozen=True)
class ExpectedPolytope:
    vertices: int
    edges: int
    faces: int
    euler: int = 2


@dataclass(frozen=True)
class KnownPolytopeCase:
    name: str
    description: str
    builder: Callable[[], Polytope]
    expected: ExpectedPolytope
    inscribed: bool = True
    convex: bool = True
    closed: bool = True


@dataclass(frozen=True)
class TrustedCamera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    projection: str = "perspective"
    fov_y_degrees: float = 35.0
    orthographic_height: float | None = None
    image_width: int = 800
    image_height: int = 600

    @property
    def aspect_ratio(self) -> float:
        return float(self.image_width) / float(self.image_height)

    @property
    def position_vec(self) -> np.ndarray:
        return np.array(self.position, dtype=float)

    @property
    def target_vec(self) -> np.ndarray:
        return np.array(self.target, dtype=float)

    @property
    def up_vec(self) -> np.ndarray:
        return np.array(self.up, dtype=float)

    @property
    def forward(self) -> np.ndarray:
        return normalize(self.target_vec - self.position_vec)

    @property
    def right(self) -> np.ndarray:
        return normalize(np.cross(self.forward, self.up_vec))

    @property
    def true_up(self) -> np.ndarray:
        return normalize(np.cross(self.right, self.forward))

    @property
    def tan_half_fov_y(self) -> float:
        return math.tan(math.radians(self.fov_y_degrees) * 0.5)

    def view_direction_at(self, point: np.ndarray) -> np.ndarray:
        if self.projection == "orthographic":
            return -self.forward
        return normalize(self.position_vec - point)

    def is_front_facing(self, point: np.ndarray, normal: np.ndarray) -> bool:
        return float(np.dot(normal, self.view_direction_at(point))) > 0.0

    def project_point(self, point: np.ndarray) -> tuple[float, float, float]:
        rel = point - self.position_vec
        x = float(np.dot(rel, self.right))
        y = float(np.dot(rel, self.true_up))
        z = float(np.dot(rel, self.forward))
        if z <= EPS:
            raise ValueError("point lies behind the camera")

        if self.projection == "orthographic":
            if self.orthographic_height is None or self.orthographic_height <= 0.0:
                raise ValueError("orthographic camera requires orthographic_height")
            half_h = 0.5 * self.orthographic_height
            ndc_x = x / (half_h * self.aspect_ratio)
            ndc_y = y / half_h
        else:
            ndc_x = x / (z * self.tan_half_fov_y * self.aspect_ratio)
            ndc_y = y / (z * self.tan_half_fov_y)
        return (ndc_x, ndc_y, z)


@dataclass
class ValidationResult:
    name: str
    description: str
    passed: bool
    expected: dict[str, int] | None
    metrics: dict[str, int | float]
    checks: dict[str, bool]
    notes: list[str]


def make_tetrahedron() -> Polytope:
    vertices = np.array([
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ], dtype=float) / math.sqrt(3.0)
    faces = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]
    return build_polytope("tetrahedron", vertices, faces)


def make_ngonal_prism(sides: int, *, height: float = 1.2) -> Polytope:
    if sides < 3:
        raise ValueError("prism requires at least 3 sides")
    z = 0.5 * height
    top = []
    bottom = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        x = math.cos(angle)
        y = math.sin(angle)
        top.append((x, y, z))
        bottom.append((x, y, -z))
    vertices = np.array(top + bottom, dtype=float)
    top_face = list(range(sides))
    bottom_face = list(range(2 * sides - 1, sides - 1, -1))
    side_faces = []
    for index in range(sides):
        nxt = (index + 1) % sides
        side_faces.append([index, nxt, sides + nxt, sides + index])
    return build_polytope(
        f"prism_{sides}",
        vertices,
        [top_face, bottom_face, *side_faces],
    )


def known_cases() -> dict[str, KnownPolytopeCase]:
    return {
        "tetrahedron": KnownPolytopeCase(
            name="tetrahedron",
            description="Regular tetrahedron with exact closed convex topology.",
            builder=make_tetrahedron,
            expected=ExpectedPolytope(vertices=4, edges=6, faces=4),
        ),
        "cube": KnownPolytopeCase(
            name="cube",
            description="Regular cube with 6 quad faces.",
            builder=lambda: make_polytope("cube"),
            expected=ExpectedPolytope(vertices=8, edges=12, faces=6),
        ),
        "octahedron": KnownPolytopeCase(
            name="octahedron",
            description="Regular octahedron with 8 triangular faces.",
            builder=lambda: make_polytope("octahedron"),
            expected=ExpectedPolytope(vertices=6, edges=12, faces=8),
        ),
        "icosahedron": KnownPolytopeCase(
            name="icosahedron",
            description="Regular icosahedron with 20 triangular faces.",
            builder=lambda: make_polytope("icosahedron"),
            expected=ExpectedPolytope(vertices=12, edges=30, faces=20),
        ),
        "prism_4": KnownPolytopeCase(
            name="prism_4",
            description="4-gonal prism, checked against prism formulas.",
            builder=lambda: make_ngonal_prism(4),
            expected=ExpectedPolytope(vertices=8, edges=12, faces=6),
        ),
        "prism_6": KnownPolytopeCase(
            name="prism_6",
            description="6-gonal prism, checked against prism formulas.",
            builder=lambda: make_ngonal_prism(6),
            expected=ExpectedPolytope(vertices=12, edges=18, faces=8),
        ),
        "prism_8": KnownPolytopeCase(
            name="prism_8",
            description="8-gonal prism, checked against prism formulas.",
            builder=lambda: make_ngonal_prism(8),
            expected=ExpectedPolytope(vertices=16, edges=24, faces=10),
        ),
    }


def fibonacci_directions(count: int) -> np.ndarray:
    directions = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * ((index + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * index + 0.37
        directions.append((radius * math.cos(theta), y, radius * math.sin(theta)))
    return np.array(directions, dtype=float)


def project_points(camera: TrustedCamera, points: np.ndarray) -> np.ndarray:
    projected = [camera.project_point(point) for point in points]
    return np.array(projected, dtype=float)


def fit_points_to_rect(points: np.ndarray, rect: tuple[float, float, float, float],
                       pad_fraction: float = 0.10) -> np.ndarray:
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


def classify_edges_for_camera(polytope: Polytope, camera: TrustedCamera) -> dict[tuple[int, int], str]:
    edge_faces = polytope_edge_faces(polytope)
    face_front: dict[int, bool] = {}
    for face_id, face in enumerate(polytope.faces):
        face_front[face_id] = camera.is_front_facing(
            face_center(polytope.vertices, face),
            face_normal(polytope.vertices, face),
        )

    kinds: dict[tuple[int, int], str] = {}
    for edge, faces in edge_faces.items():
        if len(faces) == 1:
            kinds[edge] = "border"
            continue
        front_count = sum(1 for face_id in faces if face_front[face_id])
        if front_count == 1:
            kinds[edge] = "silhouette"
        elif front_count == 2:
            kinds[edge] = "visible"
        else:
            kinds[edge] = "hidden"
    return kinds


def check_graph_connected(polytope: Polytope, edge_faces: dict[tuple[int, int], list[int]]) -> bool:
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(polytope.vertices))}
    for a, b in edge_faces:
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited: set[int] = set()
    stack = [0]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(sorted(adjacency[current] - visited))
    return len(visited) == len(polytope.vertices)


def validate_polytope(polytope: Polytope, *, description: str,
                      expected: ExpectedPolytope | None,
                      inscribed: bool, convex: bool, closed: bool,
                      ray_samples: int) -> ValidationResult:
    """Validate a single convex polytope against 20 discrete-geometry checks.

    This is the **core trusted validation** function.  Every check is derived
    from the same numeric polytope model (vertices, faces, center, bounding-
    sphere radius) — no external geometry, no approximations beyond float64
    arithmetic.

    Checks performed (in order)
    ---------------------------
    **1. Sanity (2 checks)**
        * ``face_count_positive``, ``vertex_count_positive`` — non-empty mesh.

    **2. Face validity (4 checks)**
        * ``face_size_at_least_three`` — every face is a polygon (not a digon
          or line segment).
        * ``distinct_vertices_per_face`` — no face repeats a vertex.
        * ``face_indices_valid`` — every vertex index in every face is
          within ``[0, vertex_count)``.
        * ``all_vertices_used`` — every vertex appears in at least one face
          (no orphan vertices).

    **3. Vertex uniqueness (1 check)**
        * ``no_duplicate_vertex_positions`` — no two vertices occupy the same
          position (within 12-decimal tolerance).  Duplicate positions would
          create zero-length edges.

    **4. Topology (4 checks)**
        * ``closed_edge_incidence`` — every edge is shared by exactly two
          faces (a closed manifold).  Skipped if ``closed=False``.
        * ``graph_connected`` — the vertex-edge graph is a single connected
          component (DFS traversal).
        * ``expected_vertex_count``, ``expected_edge_count``,
          ``expected_face_count``, ``expected_euler`` — match the known
          analytic counts.  Skipped if ``expected=None``.

    **5. Combinatorial invariants (2 checks)**
        * ``face_edge_sum_matches`` — sum of face sizes equals 2× edge count
          (handshake lemma, holds for closed manifolds).
        * ``vertex_valence_sum_matches`` — sum of vertex valences equals
          2× edge count.

    **6. Geometry (3 checks)**
        * ``faces_planar`` — every face's vertices lie in a common plane
          within ``1e-7`` (distance from face plane).
        * ``outward_normals`` — every face normal points outward from the
          center (``dot(normal, face_center - polytope_center) > 0``).
        * ``convex_halfspace`` — every vertex lies on or behind every face
          plane (``dot(normal, vertex - anchor) ≤ 0`` for outward normals).
          Skipped if ``convex=False``.  A positive value indicates a vertex
          outside the half-space defined by that face.

    **7. Sphere containment (2 checks)**
        * ``bounded_by_sphere`` — no vertex exceeds the bounding sphere
          radius (within ``1e-8``).  The bounding sphere is computed as the
          maximum vertex distance from center.
        * ``inscribed_vertex_radius`` — all vertices lie on the bounding
          sphere surface within ``1e-7``.  Skipped if ``inscribed=False``
          (e.g. prisms are not inscribed because top/bottom face vertices
          are closer to the center than side vertices).

    **8. Ray-casting (1 check)**
        * ``center_rays_hit_boundary`` — ``ray_samples`` Fibonacci-distributed
          unit directions are cast from the polytope center.  Every ray must
          hit exactly one face, and the hit point must lie within that face
          (``point_in_convex_face``).  Rays that hit near edges/vertices may
          be *ambiguous* (two faces at near-identical distances) — these are
          counted separately and do not cause a failure.

    Parameters
    ----------
    polytope : Polytope
        The polytope to validate (vertices centered at origin, faces oriented
        outward).
    description : str
        Human-readable description of this case for reporting.
    expected : ExpectedPolytope | None
        Known vertex/edge/face counts for this shape.  If ``None``, the
        expected-count checks are skipped.
    inscribed : bool
        If ``True``, every vertex must lie exactly on the bounding sphere
        surface.  Set ``False`` for shapes like prisms whose face vertices
        are at varying distances from the center.
    convex : bool
        If ``True``, the convex half-space check runs.  Set ``False`` for
        non-convex shapes.
    closed : bool
        If ``True``, every edge must belong to exactly two faces.  Set
        ``False`` for open meshes.
    ray_samples : int
        Number of center-ray directions to test (Fibonacci lattice).

    Returns
    -------
    ValidationResult
        Dataclass with ``passed`` (AND of all checks), ``checks`` dict
        (named boolean flags), ``metrics`` dict (numeric values), ``notes``
        (warnings), and ``expected`` (reported expected counts).
    """
    notes: list[str] = []
    checks: dict[str, bool] = {}

    vertex_count = len(polytope.vertices)
    face_count = len(polytope.faces)
    edge_faces = polytope_edge_faces(polytope)
    edge_count = len(edge_faces)

    checks["face_count_positive"] = face_count > 0
    checks["vertex_count_positive"] = vertex_count > 0

    valid_face_sizes = all(len(face) >= 3 for face in polytope.faces)
    checks["face_size_at_least_three"] = valid_face_sizes
    distinct_vertices_per_face = all(len(face) == len(set(face)) for face in polytope.faces)
    checks["distinct_vertices_per_face"] = distinct_vertices_per_face
    face_indices_valid = all(
        0 <= vertex_id < vertex_count
        for face in polytope.faces
        for vertex_id in face
    )
    checks["face_indices_valid"] = face_indices_valid

    used_vertices = {vertex_id for face in polytope.faces for vertex_id in face}
    checks["all_vertices_used"] = len(used_vertices) == vertex_count

    # Duplicate vertex positions would create zero-length edges and break
    # the winged-edge builder's assumption that every edge connects two
    # distinct vertices.  We round to 12 decimal places (≈ 1e-12) before
    # comparison because float64 arithmetic on the same input coordinates
    # should produce identical bits, but a small tolerance guards against
    # accidental near-duplicates from ill-conditioned geometric constructions.
    duplicate_positions = False
    seen_positions: set[tuple[float, float, float]] = set()
    for vertex in polytope.vertices:
        key = tuple(round(float(component), 12) for component in vertex)
        if key in seen_positions:
            duplicate_positions = True
            break
        seen_positions.add(key)
    checks["no_duplicate_vertex_positions"] = not duplicate_positions

    if closed:
        checks["closed_edge_incidence"] = all(len(faces) == 2 for faces in edge_faces.values())
    else:
        checks["closed_edge_incidence"] = True

    checks["graph_connected"] = check_graph_connected(polytope, edge_faces)

    euler = vertex_count - edge_count + face_count
    if expected is not None:
        checks["expected_vertex_count"] = vertex_count == expected.vertices
        checks["expected_edge_count"] = edge_count == expected.edges
        checks["expected_face_count"] = face_count == expected.faces
        checks["expected_euler"] = euler == expected.euler
    else:
        checks["expected_vertex_count"] = True
        checks["expected_edge_count"] = True
        checks["expected_face_count"] = True
        checks["expected_euler"] = True

    face_edge_sum = sum(len(face) for face in polytope.faces)
    checks["face_edge_sum_matches"] = face_edge_sum == 2 * edge_count if closed else True

    valences = [0] * vertex_count
    for a, b in edge_faces:
        valences[a] += 1
        valences[b] += 1
    checks["vertex_valence_sum_matches"] = sum(valences) == 2 * edge_count

    max_planarity_error = 0.0
    outward_normals = True
    max_convex_violation = 0.0
    for face in polytope.faces:
        normal = face_normal(polytope.vertices, face)
        anchor = polytope.vertices[face[0]]
        center_dot = float(np.dot(normal, face_center(polytope.vertices, face) - polytope.center))
        if center_dot <= 0.0:
            outward_normals = False
        for vertex_id in face:
            point = polytope.vertices[vertex_id]
            planarity_error = abs(float(np.dot(normal, point - anchor)))
            max_planarity_error = max(max_planarity_error, planarity_error)
        if convex:
            for point in polytope.vertices:
                violation = float(np.dot(normal, point - anchor))
                max_convex_violation = max(max_convex_violation, violation)
    checks["faces_planar"] = max_planarity_error <= 1.0e-7
    checks["outward_normals"] = outward_normals
    checks["convex_halfspace"] = max_convex_violation <= 1.0e-7 if convex else True

    radii = np.linalg.norm(polytope.vertices - polytope.center, axis=1)
    max_radius = float(np.max(radii))
    min_radius = float(np.min(radii))
    checks["bounded_by_sphere"] = max_radius <= polytope.sphere_radius + 1.0e-8
    max_sphere_error = float(np.max(np.abs(radii - polytope.sphere_radius)))
    checks["inscribed_vertex_radius"] = max_sphere_error <= 1.0e-7 if inscribed else True

    ambiguous_rays = 0
    ray_failures = 0
    directions = fibonacci_directions(ray_samples)
    for direction in directions:
        try:
            hit = ray_polytope_intersection(polytope, direction)
        except RuntimeError:
            ray_failures += 1
            continue
        face = polytope.faces[hit.face_id]
        anchor = polytope.vertices[face[0]]
        plane_error = abs(float(np.dot(hit.normal, hit.point - anchor)))
        if plane_error > 1.0e-6 or not point_in_convex_face(hit.point, polytope.vertices, face, hit.normal):
            ray_failures += 1
            continue

        distances = []
        for candidate_face in polytope.faces:
            candidate_normal = face_normal(polytope.vertices, candidate_face)
            denom = float(np.dot(candidate_normal, direction))
            if denom <= EPS:
                continue
            numer = float(np.dot(candidate_normal, polytope.vertices[candidate_face[0]] - polytope.center))
            if numer <= EPS:
                continue
            distance = numer / denom
            if distance <= EPS:
                continue
            point = polytope.center + distance * direction
            if point_in_convex_face(point, polytope.vertices, candidate_face, candidate_normal):
                distances.append(distance)
        distances.sort()
        if len(distances) >= 2 and abs(distances[1] - distances[0]) <= 1.0e-7:
            ambiguous_rays += 1
    checks["center_rays_hit_boundary"] = ray_failures == 0

    metrics: dict[str, int | float] = {
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "face_count": face_count,
        "euler_characteristic": euler,
        "min_valence": min(valences),
        "max_valence": max(valences),
        "min_radius": min_radius,
        "max_radius": max_radius,
        "bounding_sphere_radius": float(polytope.sphere_radius),
        "max_sphere_error": max_sphere_error,
        "max_planarity_error": max_planarity_error,
        "max_convex_violation": max_convex_violation,
        "ray_samples": ray_samples,
        "ray_failures": ray_failures,
        "ambiguous_rays": ambiguous_rays,
    }

    if ambiguous_rays > 0:
        notes.append(
            f"{ambiguous_rays} sampled center rays landed on edge/vertex-near cases; "
            "that is expected on a finite directional sample."
        )
    if not checks["inscribed_vertex_radius"]:
        notes.append("This shape is not exactly inscribed in its bounding sphere.")

    passed = all(checks.values())
    return ValidationResult(
        name=polytope.name,
        description=description,
        passed=passed,
        expected=asdict(expected) if expected is not None else None,
        metrics=metrics,
        checks=checks,
        notes=notes,
    )


def write_numbers_json(path: Path, polytope: Polytope) -> None:
    payload = {
        "name": polytope.name,
        "center": polytope.center.tolist(),
        "bounding_sphere_radius": float(polytope.sphere_radius),
        "vertex_count": len(polytope.vertices),
        "face_count": len(polytope.faces),
        "edge_count": len(polytope_edge_faces(polytope)),
        "vertices": polytope.vertices.tolist(),
        "faces": [list(face) for face in polytope.faces],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_validation_json(path: Path, result: ValidationResult) -> None:
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def write_obj_mesh(path: Path, polytope: Polytope) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# trusted polytope mesh\n")
        handle.write(f"o {polytope.name}\n")
        for vertex in polytope.vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in polytope.faces:
            indices = " ".join(str(vertex_id + 1) for vertex_id in face)
            handle.write(f"f {indices}\n")


def write_ply_mesh(path: Path, polytope: Polytope) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"comment trusted polytope mesh {polytope.name}\n")
        handle.write(f"element vertex {len(polytope.vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(polytope.faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in polytope.vertices:
            handle.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in polytope.faces:
            indices = " ".join(str(vertex_id) for vertex_id in face)
            handle.write(f"{len(face)} {indices}\n")


def pdf_polyline(pdf: PurePDF, points: list[tuple[float, float]], width: float,
                 gray: float, dashed: bool = False) -> None:
    if len(points) < 2:
        return
    pdf.save_state()
    pdf.line_width(width)
    pdf.stroke_gray(gray)
    pdf.content.set_line_cap(1)
    pdf.content.set_line_join(1)
    if dashed:
        pdf.content.set_dash([3.0, 2.5], phase=0.0)
    pdf.content.polyline(points)
    pdf.content.stroke()
    pdf.restore_state()


def _face_fill_gray(normal: np.ndarray) -> float:
    """Map a unit face normal to a fill gray so top / side / front faces
    are visually distinct.  The scheme is:

        +Y (top)      → 0.88   darker, reads as the "roof" of the solid
        -Y (bottom)   → 0.98   near-white, usually back-facing anyway
        +Z (front)    → 0.94   medium-light
        -Z (back)     → 0.97   light (back faces are usually hidden)
        +X / -X (side) → 0.91  slightly darker, side walls

    In the isometric view you can tell at a glance which faces point where.
    """
    nx, ny, nz = abs(float(normal[0])), float(normal[1]), abs(float(normal[2]))
    if ny > 0.6:
        return 0.88  # top — darkest, most prominent
    if ny < -0.6:
        return 0.98  # bottom — near white
    if nz > nx and nz > abs(ny):
        return 0.94  # front / back — medium
    return 0.91  # side — slightly darker


def draw_panel(pdf: PurePDF, polytope: Polytope, camera: TrustedCamera,
               title: str, rect: tuple[float, float, float, float]) -> None:
    """Draw one proof panel: face fills + classified edges.

    Front-facing faces are filled with a gray keyed to the face normal
    direction (top darker, sides medium, front lighter) so the 3D
    orientation reads instantly.  Edges are drawn on top following the
    same classification convention (silhouette bold, visible medium,
    hidden dashed).
    """
    x0, y0, width, height = rect
    pdf.text(x0, y0 + height + mm_to_pts(2.2), title, font="Helvetica-Bold", size=9.5, gray=0.0)

    projected = project_points(camera, polytope.vertices)
    mapped = fit_points_to_rect(projected, rect)
    edge_kinds = classify_edges_for_camera(polytope, camera)

    # --- face fills: color-coded by normal direction so the solid reads
    #     as a true 3D object with distinguishable top / side / front. ----
    for face_id, face in enumerate(polytope.faces):
        center_world = face_center(polytope.vertices, face)
        normal_world = face_normal(polytope.vertices, face)
        if not camera.is_front_facing(center_world, normal_world):
            continue
        pts = [tuple(mapped[vid]) for vid in face]
        if len(pts) < 3:
            continue
        fill = _face_fill_gray(normal_world)
        pdf.save_state()
        pdf.content.set_fill_gray(fill)
        pdf.content.set_stroke_gray(1.0)
        pdf.content.move_to(*pts[0])
        for p in pts[1:]:
            pdf.content.line_to(*p)
        pdf.content.close_path()
        pdf.content.fill()
        pdf.restore_state()

    # --- edge strokes on top of fills -------------------------------------
    for edge, kind in edge_kinds.items():
        pts = [tuple(mapped[index]) for index in edge]
        if kind == "hidden":
            pdf_polyline(pdf, pts, width=0.45, gray=0.82, dashed=True)
        elif kind == "visible":
            pdf_polyline(pdf, pts, width=0.58, gray=0.45)
        elif kind == "silhouette":
            pdf_polyline(pdf, pts, width=0.95, gray=0.02)
        else:
            pdf_polyline(pdf, pts, width=0.62, gray=0.18)

    # --- light panel border -----------------------------------------------
    pdf.save_state()
    pdf.line_width(0.35)
    pdf.stroke_gray(0.82)
    pdf.content.rect(x0, y0, width, height)
    pdf.content.stroke()
    pdf.restore_state()


def write_pdf_proof(path: Path, polytope: Polytope, result: ValidationResult) -> None:
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(12.0)
    gap = mm_to_pts(6.0)
    panel_w = 0.5 * (page_w - 2.0 * margin - gap)
    panel_h = 0.5 * (page_h - 2.0 * margin - gap - mm_to_pts(16.0))

    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"{polytope.name} validation proof",
             font="Helvetica-Bold", size=14.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             result.description,
             font="Helvetica", size=8.8, gray=0.28)
    pdf.text(page_w - margin - mm_to_pts(40.0), page_h - margin + mm_to_pts(1.0),
             "PASS" if result.passed else "FAIL",
             font="Helvetica-Bold", size=12.0, gray=0.0 if result.passed else 0.0)

    cameras = [
        (
            "Isometric Perspective",
            TrustedCamera(
                position=(2.8, 2.0, 3.5),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
                projection="perspective",
                fov_y_degrees=35.0,
            ),
        ),
        (
            "Front Orthographic",
            TrustedCamera(
                position=(0.0, 0.0, 5.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
                projection="orthographic",
                orthographic_height=2.8 * polytope.sphere_radius,
            ),
        ),
        (
            "Top Orthographic",
            TrustedCamera(
                position=(0.0, 5.0, 0.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, -1.0),
                projection="orthographic",
                orthographic_height=2.8 * polytope.sphere_radius,
            ),
        ),
        (
            "Side Orthographic",
            TrustedCamera(
                position=(5.0, 0.0, 0.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
                projection="orthographic",
                orthographic_height=2.8 * polytope.sphere_radius,
            ),
        ),
    ]

    for index, (title, camera) in enumerate(cameras):
        col = index % 2
        row = index // 2
        rect = (
            margin + col * (panel_w + gap),
            margin + (1 - row) * (panel_h + gap),
            panel_w,
            panel_h,
        )
        draw_panel(pdf, polytope, camera, title, rect)

    footer_y = margin - mm_to_pts(2.0)
    counts = (
        f"V={result.metrics['vertex_count']}  "
        f"E={result.metrics['edge_count']}  "
        f"F={result.metrics['face_count']}  "
        f"Euler={result.metrics['euler_characteristic']}"
    )
    pdf.text(margin, footer_y + mm_to_pts(3.5), counts, font="Helvetica-Bold", size=9.5, gray=0.0)
    pdf.text(
        margin,
        footer_y,
        f"planarity err={result.metrics['max_planarity_error']:.2e} | "
        f"convex violation={result.metrics['max_convex_violation']:.2e} | "
        f"ray failures={result.metrics['ray_failures']}",
        font="Helvetica",
        size=8.5,
        gray=0.25,
    )
    pdf.save(str(path))


def export_case(output_dir: Path, polytope: Polytope, result: ValidationResult) -> dict[str, str]:
    stem = polytope.name.replace(" ", "_")
    numbers_path = output_dir / f"{stem}_numbers.json"
    validation_path = output_dir / f"{stem}_validation.json"
    obj_path = output_dir / f"{stem}.obj"
    ply_path = output_dir / f"{stem}.ply"
    pdf_path = output_dir / f"{stem}.pdf"

    write_numbers_json(numbers_path, polytope)
    write_validation_json(validation_path, result)
    write_obj_mesh(obj_path, polytope)
    write_ply_mesh(ply_path, polytope)
    write_pdf_proof(pdf_path, polytope, result)

    return {
        "numbers_json": str(numbers_path),
        "validation_json": str(validation_path),
        "obj": str(obj_path),
        "ply": str(ply_path),
        "pdf": str(pdf_path),
    }


def write_manifest(path: Path, summaries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, default="all",
                        help="known case name, or 'all'")
    parser.add_argument("--polytope-json", type=Path, default=None,
                        help="validate a custom polytope JSON from this repo's numeric format")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ray-samples", type=int, default=180,
                        help="number of center rays used in convex hit checks")
    return parser.parse_args()


def resolve_cases(args: argparse.Namespace) -> list[tuple[str, Polytope, str, ExpectedPolytope | None, bool, bool, bool]]:
    if args.polytope_json is not None:
        payload = json.loads(args.polytope_json.read_text(encoding="utf-8"))
        vertices = np.array(payload["vertices"], dtype=float)
        if payload.get("faces") and isinstance(payload["faces"][0], dict):
            faces = [list(map(int, face["vertex_indices"])) for face in payload["faces"]]
        else:
            faces = [list(map(int, face)) for face in payload["faces"]]
        polytope = build_polytope(payload.get("name", args.polytope_json.stem), vertices, faces)
        return [
            (
                polytope.name,
                polytope,
                f"Custom polytope loaded from {args.polytope_json.name}",
                None,
                False,
                True,
                True,
            )
        ]

    corpus = known_cases()
    if args.case == "all":
        names = list(corpus.keys())
    else:
        if args.case not in corpus:
            raise SystemExit(f"unknown case: {args.case}. choose from: {', '.join(sorted(corpus))}")
        names = [args.case]

    cases = []
    for name in names:
        case = corpus[name]
        cases.append((
            case.name,
            case.builder(),
            case.description,
            case.expected,
            case.inscribed,
            case.convex,
            case.closed,
        ))
    return cases


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for name, polytope, description, expected, inscribed, convex, closed in resolve_cases(args):
        result = validate_polytope(
            polytope,
            description=description,
            expected=expected,
            inscribed=inscribed,
            convex=convex,
            closed=closed,
            ray_samples=args.ray_samples,
        )
        outputs = export_case(args.output_dir, polytope, result)
        summary = {
            "name": name,
            "passed": result.passed,
            "validation_json": outputs["validation_json"],
            "numbers_json": outputs["numbers_json"],
            "obj": outputs["obj"],
            "ply": outputs["ply"],
            "pdf": outputs["pdf"],
        }
        summaries.append(summary)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {name}")
        print(f"  V={result.metrics['vertex_count']} E={result.metrics['edge_count']} F={result.metrics['face_count']} Euler={result.metrics['euler_characteristic']}")
        print(f"  pdf {outputs['pdf']}")
        print(f"  obj {outputs['obj']}")
        print(f"  ply {outputs['ply']}")
        print(f"  validation {outputs['validation_json']}")

    manifest = args.output_dir / "manifest.json"
    write_manifest(manifest, summaries)
    print(f"manifest {manifest}")


if __name__ == "__main__":
    main()
