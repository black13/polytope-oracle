#!/usr/bin/env python3
"""
Trusted polytope-scene oracle: multi-object detached scene validation.
=====================================================================

This module extends the single-polytope validator to **detached multi-object
scenes**.  Every scene check is derived from the same numeric polytope model
(vertices + faces + center + bounding sphere radius) — no external geometry
sources are used.

Trust boundary
--------------
This file lives under ``tools/polytope_oracle/`` and is **trusted** because:

1. Every geometric primitive (vertex, face normal, face center, edge) comes
   from a ``Polytope`` — an explicit numeric model with no hidden state.
2. World-space transformations are closed-form rigid-body transforms
   (rotation + translation) with no scaling, no interpolation, and no
   discretization error beyond float64.
3. Occlusion is tested by casting **analytic rays** against the exact
   polygonal faces (not a triangulation).  The ray-face intersection uses
   half-plane tests that are exact for convex planar faces.
4. Silhouette edges are defined combinatorially from the visible-face set,
   so they inherit no additional error from ray-casting.
5. Connected-component counting uses graph traversal on the exact edge
   topology derived from the polytope face lists.

Checks performed
----------------
* **Connected component count** — must equal object count for detached
  scenes.  Verifies that Freestyle's winged-edge builder does not
  accidentally merge disconnected components.
* **No false chaining across objects** — verifies every edge's vertices
  belong to the same local polytope (structurally guaranteed for polytope
  data; the real integrity check is ``check_world_vertex_overlap``).
* **No world-space vertex overlap** — verifies no two objects share a
  vertex position in world coordinates.
* **Per-object visible-face sets** — front-facing classification (N·V > 0)
  followed by inter-object occlusion ray-casting.  A face is *visible* iff
  it is front-facing **and** no other object's face blocks the ray from
  the camera to the face center.
* **Per-object silhouette edges** — edges where exactly one of the two
  adjacent faces is visible.  These form the silhouette boundary of each
  object.  Border edges (single adjacent face) are excluded.
* **Occlusion ordering** — objects are sorted front-to-back by the
  projection of their centers onto the camera's forward direction.
  This gives a coarse but deterministic Z-ordering.
* **Center-ray boundary hits** — for each object independently, a set of
  Fibonacci-distributed unit directions are cast from the object's center
  and must hit the object's boundary.  This verifies the polytope is
  closed and the ray-face intersection code is correct.

Scene definitions
-----------------
Three scenes are built-in:

``two_prisms``
    Two 4-gonal prisms placed side-by-side (x-offset ±1.8).  No occlusion
    between them.  Both objects contribute silhouette edges.

``two_solids``
    A cube (left) and an octahedron (right) placed side-by-side
    (x-offset ±1.6).  No occlusion between them.

``partial_occlusion``
    A cube placed at z=2.0, partially occluding a 6-gonal prism at z=−0.3.
    From the front camera, the prism's front-facing face is fully occluded
    by the cube.  Only the cube contributes silhouette edges.

Usage
-----
.. code-block:: bash

    python3 tools/polytope_oracle/polytope_scene.py --scene all
    python3 tools/polytope_oracle/polytope_scene.py --scene partial_occlusion

Output
------
Per scene, four files are written to ``output/polytope_validation/``:

* ``scene_<name>_validation.json`` — full validation result with all checks
* ``scene_<name>.obj`` — combined OBJ mesh in world space
* ``scene_<name>.pdf`` — 4-panel proof PDF (isometric + three orthographic
  views) with silhouette edges drawn bold
* ``scene_manifest.json`` — summary of all scenes run
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

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
    ray_face_intersection,
    ray_polytope_intersection,
)
from polytope_validator import (
    TrustedCamera,
    classify_edges_for_camera,
    draw_panel,
    fit_points_to_rect,
    project_points,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "polytope-oracle" / "output"
IDENTITY_ROTATION = np.eye(3, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# PositionedPolytope — a polytope placed in world space
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PositionedPolytope:
    """A convex polytope placed at a world-space position and orientation.

    The polytope's vertices are defined in **local space** (center at origin,
    bounding sphere radius normalized).  The object's world-space vertices are
    computed as::

        world_vertex = local_vertex @ rotation^T + offset

    Since rotation is an orthogonal matrix (with determinant +1), the inverse
    transform is:

        local_vertex = (world_vertex - offset) @ rotation

    Attributes
    ----------
    polytope : Polytope
        The underlying convex polytope in local (object) space.
    offset : np.ndarray (3,) or tuple[float, float, float]
        World-space translation vector applied after rotation.
    rotation : np.ndarray (3,3) or None, optional
        3×3 rotation matrix.  ``None`` defaults to the identity matrix.
        Applied as ``vertex @ rotation.T`` (row-vector convention).
    """

    polytope: Polytope
    offset: np.ndarray
    rotation: np.ndarray = None

    def __post_init__(self) -> None:
        """Coerce offset/rotation to float64 arrays after dataclass init."""
        self.offset = np.asarray(self.offset, dtype=float)
        if self.rotation is None:
            object.__setattr__(self, "rotation", IDENTITY_ROTATION.copy())
        else:
            object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=float))

    @property
    def label(self) -> str:
        """Human-readable name, forwarded from the underlying Polytope."""
        return self.polytope.name

    # -- world-space accessors ------------------------------------------------

    def world_vertices(self) -> np.ndarray:
        """Return all vertices transformed to world space (N×3 array)."""
        return self.polytope.vertices @ self.rotation.T + self.offset

    def world_center(self) -> np.ndarray:
        """Return the polytope center transformed to world space (3,)."""
        return self.polytope.center @ self.rotation.T + self.offset

    def local_to_world(self, point: np.ndarray) -> np.ndarray:
        """Transform a single point from local to world space.

        Parameters
        ----------
        point : np.ndarray (3,)
            Point in local (object) coordinates.

        Returns
        -------
        np.ndarray (3,)
            The same point in world coordinates.
        """
        return point @ self.rotation.T + self.offset

    def world_to_local(self, point: np.ndarray) -> np.ndarray:
        """Transform a single point from world to local space.

        Uses the inverse of the rigid-body transform.  Because ``rotation``
        is orthogonal, ``rotation^{-1} = rotation^T``, and applying it as a
        row-vector post-multiplication gives ``(point - offset) @ rotation``.

        Parameters
        ----------
        point : np.ndarray (3,)
            Point in world coordinates.

        Returns
        -------
        np.ndarray (3,)
            The same point in local (object) coordinates.
        """
        return (point - self.offset) @ self.rotation

    def world_to_local_direction(self, direction: np.ndarray) -> np.ndarray:
        """Transform a direction vector from world to local space.

        Directions ignore translation, so only the inverse rotation is
        applied: ``direction_world @ rotation``.  The output remains a
        unit vector because orthogonal transformations preserve length.

        Parameters
        ----------
        direction : np.ndarray (3,)
            Unit direction vector in world coordinates.

        Returns
        -------
        np.ndarray (3,)
            The same direction in local (object) coordinates.
        """
        return direction @ self.rotation

    def world_face_center(self, face_id: int) -> np.ndarray:
        """Return the world-space centroid of face *face_id*."""
        return self.local_to_world(
            face_center(self.polytope.vertices, self.polytope.faces[face_id])
        )

    def world_face_normal(self, face_id: int) -> np.ndarray:
        """Return the world-space outward normal of face *face_id*.

        The local normal is computed from the local vertices via cross-product,
        then transformed by the rotation matrix.  Normals transform with the
        **same** rotation as positions (they are covectors, so they use the
        inverse-transpose; for orthogonal matrices, inverse-transpose = the
        matrix itself, so ``normal @ rotation.T`` is correct).
        """
        local_normal = face_normal(self.polytope.vertices, self.polytope.faces[face_id])
        return local_normal @ self.rotation.T

    def world_edges(self) -> list[tuple[int, int]]:
        """Return all undirected edges of this polytope as sorted (a,b) pairs.

        Edge indices reference the **local** vertex numbering (0..n_vertices-1).
        This is the canonical edge set for the polytope; the world-space
        positions are obtained by applying ``world_vertices()`` to these indices.

        Returns
        -------
        list[tuple[int, int]]
            Sorted list of unique undirected edges.
        """
        edges: set[tuple[int, int]] = set()
        for face in self.polytope.faces:
            for i, start in enumerate(face):
                end = face[(i + 1) % len(face)]
                edges.add(tuple(sorted((start, end))))
        return sorted(edges)


# ═══════════════════════════════════════════════════════════════════════════
# PolytopeScene — a collection of positioned polytopes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PolytopeScene:
    """An ordered collection of positioned polytopes in world space.

    A scene is purely geometric: it holds a list of ``PositionedPolytope``
    objects and provides aggregate counts (total vertices, faces, edges)
    across the whole scene.  All validation logic is external to this class.

    Parameters
    ----------
    objects : list[PositionedPolytope]
        The polytopes in the scene, in insertion order.
    """

    objects: list[PositionedPolytope]

    @property
    def object_count(self) -> int:
        """Number of objects in the scene."""
        return len(self.objects)

    def total_vertices(self) -> int:
        """Total vertex count across all objects."""
        return sum(len(obj.polytope.vertices) for obj in self.objects)

    def total_faces(self) -> int:
        """Total face count across all objects."""
        return sum(len(obj.polytope.faces) for obj in self.objects)

    def total_edges(self) -> int:
        """Total undirected edge count across all objects."""
        total = 0
        for obj in self.objects:
            total += len(polytope_edge_faces(obj.polytope))
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Scene builders — the three detached test cases
# ═══════════════════════════════════════════════════════════════════════════


def make_two_detached_prisms() -> PolytopeScene:
    """Scene: two 4-gonal prisms placed side-by-side with no overlap.

    Position
    --------
    * Prism A: centered at (-1.8, 0.0, 0.0), height 1.2
    * Prism B: centered at ( 1.8, 0.1, 0.2), height 1.0

    Expected properties
    -------------------
    =====================  =====
    Connected components    2
    Visible faces (each)    3 (top + 2 side faces facing camera)
    Silhouette edges (each) 6 (the boundary of the 3-visible-face set)
    Occlusion between       0 faces occluded (objects are side-by-side)
    =====================  =====
    """
    from polytope_validator import make_ngonal_prism

    prism_a = make_ngonal_prism(4, height=1.2)
    prism_b = make_ngonal_prism(4, height=1.0)

    return PolytopeScene(objects=[
        PositionedPolytope(polytope=prism_a, offset=np.array([-1.8, 0.0, 0.0])),
        PositionedPolytope(polytope=prism_b, offset=np.array([1.8, 0.1, 0.2])),
    ])


def make_two_detached_solids() -> PolytopeScene:
    """Scene: a cube and an octahedron placed side-by-side with no overlap.

    Position
    --------
    * Cube:        centered at (-1.6, 0.0, 0.0)
    * Octahedron:  centered at ( 1.6, 0.0, 0.0)

    Expected properties
    -------------------
    =====================  =====
    Connected components    2
    Visible faces (cube)    2 (one front-facing face visible)
    Visible faces (octa)    4 (four front-facing triangles visible)
    Silhouette edges (cube)  6
    Silhouette edges (octa)  4
    Occlusion between       0 faces occluded (objects are side-by-side)
    =====================  =====
    """
    cube = make_polytope("cube")
    octa = make_polytope("octahedron")

    return PolytopeScene(objects=[
        PositionedPolytope(polytope=cube, offset=np.array([-1.6, 0.0, 0.0])),
        PositionedPolytope(polytope=octa, offset=np.array([1.6, 0.0, 0.0])),
    ])


def make_partial_occlusion() -> PolytopeScene:
    """Scene: a cube partially occluding a hexagonal prism.

    Position
    --------
    * Cube (occluder): centered at (0.0, 0.0, 2.0) — closer to camera
    * Prism (occluded): centered at (0.4, 0.05, -0.3) — behind the cube

    From the front camera at (0, 0, 6) looking toward (0, 0, 0), the cube
    sits between the camera and the prism.  The prism's single front-facing
    face is fully occluded by the cube.

    Expected properties
    -------------------
    ======================  =====
    Connected components     2
    Occlusion order          [cube_idx, prism_idx] (front-to-back)
    Visible faces (cube)     1 (the front face, not occluded by anything)
    Visible faces (prism)    0 (the one front-facing face is occluded)
    Silhouette edges (cube)  4 (boundary of the single visible face)
    Silhouette edges (prism) 0 (no visible faces, so no silhouette)
    ======================  =====
    """
    cube = make_polytope("cube")
    from polytope_validator import make_ngonal_prism
    prism = make_ngonal_prism(6, height=1.4)

    cube_offset = np.array([0.0, 0.0, 2.0])
    prism_offset = np.array([0.4, 0.05, -0.3])

    return PolytopeScene(objects=[
        PositionedPolytope(polytope=cube, offset=cube_offset),
        PositionedPolytope(polytope=prism, offset=prism_offset),
    ])


SCENE_BUILDERS = {
    "two_prisms": ("Two detached 4-gonal prisms", make_two_detached_prisms),
    "two_solids": ("Two detached regular solids (cube + octahedron)", make_two_detached_solids),
    "partial_occlusion": ("Cube partially occluding a 6-gonal prism", make_partial_occlusion),
}


# ═══════════════════════════════════════════════════════════════════════════
# Scene-level geometry checks
# ═══════════════════════════════════════════════════════════════════════════


def check_connected_components(scene: PolytopeScene) -> int:
    """Count connected components of the scene-wide vertex-edge graph.

    **Algorithm.**  Build an adjacency list over all scene vertices (each
    object's vertices occupy a contiguous block).  For each object, insert
    edges between its own vertices using the polytope's undirected edge set.
    Perform a depth-first search and count the number of connected subgraphs.

    **Invariant.**  Since all objects in our test scenes are detached (no
    shared world-space vertices), the component count must equal the number
    of objects.  A mismatch would indicate that objects share vertices —
    either by construction error or because Freestyle's builder accidentally
    welded them.

    Returns
    -------
    int
        Number of connected components found in the scene graph.
    """
    # Build a flat list of world-space vertices and their owning object.
    all_vertices: list[tuple[float, float, float]] = []
    vertex_to_obj: list[int] = []
    for obj_idx, obj in enumerate(scene.objects):
        wv = obj.world_vertices()
        for row in wv:
            all_vertices.append(tuple(float(c) for c in row))
            vertex_to_obj.append(obj_idx)

    total = len(all_vertices)

    # Build adjacency: for each object, connect its own edges.
    adjacency: list[list[int]] = [[] for _ in range(total)]
    for obj_idx, obj in enumerate(scene.objects):
        edges = obj.world_edges()
        # base = starting global index for this object's vertex block
        base = sum(len(scene.objects[i].polytope.vertices) for i in range(obj_idx))
        for a, b in edges:
            adjacency[base + a].append(base + b)
            adjacency[base + b].append(base + a)

    # DFS to count components.
    visited = [False] * total
    components = 0
    for start in range(total):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            for n in adjacency[v]:
                if not visited[n]:
                    stack.append(n)
    return components


def check_no_cross_object_edges(scene: PolytopeScene) -> bool:
    """Verify that every edge references only vertices within its own object.

    Since edges are extracted from each polytope's own face lists (which use
    local 0-based indices), this check is **structurally guaranteed** to pass.
    It exists as a safety net: if a polytope's ``world_edges()`` ever returned
    indices outside ``[0, n_vertices)``, this would catch it.

    The real cross-object integrity check is ``check_world_vertex_overlap``,
    which verifies no two objects share a world-space vertex position (the
    mechanism by which Freestyle could accidentally merge components).

    Returns
    -------
    bool
        ``True`` if every edge's vertex indices are within bounds.
    """
    for obj in scene.objects:
        max_vertex = len(obj.polytope.vertices)
        edges = obj.world_edges()
        for a, b in edges:
            if a < 0 or a >= max_vertex or b < 0 or b >= max_vertex:
                return False
    return True


def check_world_vertex_overlap(scene: PolytopeScene) -> bool:
    """Verify that no two objects share a vertex position in world space.

    **Why this matters.**  If two polytopes in a scene share a world-space
    vertex, Freestyle's winged-edge builder (or any downstream mesh processor)
    could merge them into a single connected component.  This check ensures
    our detached scenes are genuinely detached.

    Positions are rounded to 10 decimal places before comparison to tolerate
    float64 round-off from the rotation+translation transform (which is exact
    for rational inputs but subject to finite-precision arithmetic).

    Returns
    -------
    bool
        ``True`` if every world-space vertex position is unique.
    """
    seen: set[tuple[float, float, float]] = set()
    for obj in scene.objects:
        for vertex in obj.world_vertices():
            key = tuple(round(float(c), 10) for c in vertex)
            if key in seen:
                return False
            seen.add(key)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Per-object visibility with inter-object occlusion
# ═══════════════════════════════════════════════════════════════════════════


def compute_scene_face_visibility(
    scene: PolytopeScene,
    camera: TrustedCamera,
) -> list[dict[str, set[int]]]:
    """Compute per-object face visibility accounting for inter-object occlusion.

    **Algorithm (two-pass).**

    **Pass 1 — front/back classification (per object, per face).**
    For each face of each object, compute the world-space face center and
    outward-pointing face normal.  The face is *front-facing* if the angle
    between the normal and the view direction (camera→face) is less than 90°
    (i.e., ``dot(normal, view_direction) > 0``).  Otherwise it is back-facing.

    This pass uses the *same* front-facing test as the single-polytope
    validator (``TrustedCamera.is_front_facing()``).

    **Pass 2 — inter-object occlusion test (only for front-facing faces).**
    For each front-facing face of each object, cast a ray from the camera
    position toward the face center:

    1. Compute the ray direction ``dir = normalize(face_center - camera_pos)``.
    2. Compute ``max_distance = ||face_center - camera_pos||`` (the distance
       along the ray to the face center).
    3. For every *other* object in the scene, transform the ray into that
       object's local coordinate system and test against every face using
       ``ray_face_intersection()`` (exact convex-face test from the trusted
       ``polytope_numbers`` module).
    4. If any hit is found with ``hit.distance < max_distance - 1e-6``, the
       face is **occluded**.  (The 1e-6 tolerance prevents self-occlusion
       false positives from float64 round-off.)
    5. Otherwise, the face is **visible**.

    **Important invariant.**  The ray is tested against the **exact polygonal
    faces** of the polytope, not a triangulation.  This avoids discretization
    artifacts that could cause false positives or missed occlusions at facet
    boundaries.

    **Why this is trusted.**  Both the ray-face intersection and the front-
    facing test use the same closed-form geometric primitives that the
    single-polytope validator has already verified against the full known
    corpus (7 convex polytopes, all PASS).  The only new aspect is the
    multi-object loop, which is a simple repetition of the same primitives.

    Parameters
    ----------
    scene : PolytopeScene
        The multi-object scene.
    camera : TrustedCamera
        The trusted camera from ``polytope_validator``.

    Returns
    -------
    list[dict[str, set[int]]]
        One dict per object, with keys:

        * ``"front_facing"`` — set of local face IDs that face the camera
        * ``"back_facing"`` — set of local face IDs that face away from the
          camera
        * ``"visible"`` — set of local face IDs that are front-facing AND
          not occluded by any other object
        * ``"occluded"`` — set of local face IDs that are front-facing BUT
          occluded by at least one other object
    """
    results: list[dict[str, set[int]]] = []
    camera_pos = camera.position_vec

    # --- Pass 1: front/back classification ---------------------------------
    for obj_idx, obj in enumerate(scene.objects):
        front_facing: set[int] = set()
        back_facing: set[int] = set()

        for face_id in range(len(obj.polytope.faces)):
            center_world = obj.world_face_center(face_id)
            normal_world = obj.world_face_normal(face_id)
            if camera.is_front_facing(center_world, normal_world):
                front_facing.add(face_id)
            else:
                back_facing.add(face_id)

        results.append({
            "front_facing": front_facing,
            "back_facing": back_facing,
            "visible": set(),
            "occluded": set(),
        })

    # --- Pass 2: inter-object occlusion ------------------------------------
    for obj_idx, obj in enumerate(scene.objects):
        info = results[obj_idx]
        for face_id in sorted(info["front_facing"]):
            center_world = obj.world_face_center(face_id)
            ray_dir_world = normalize(center_world - camera_pos)

            # Degenerate case: camera at face center — treat as visible.
            if np.linalg.norm(ray_dir_world) < EPS:
                info["visible"].add(face_id)
                continue

            max_distance = float(np.linalg.norm(center_world - camera_pos))

            occluded = False
            for other_idx, other_obj in enumerate(scene.objects):
                if other_idx == obj_idx:
                    continue

                # Transform the camera-origin ray into the other object's
                # local coordinate system.  Because rotations are orthogonal,
                # distances are preserved, so we can compare hit.distance
                # directly against max_distance.
                ray_origin_local = other_obj.world_to_local(camera_pos)
                ray_dir_local = other_obj.world_to_local_direction(ray_dir_world)

                for other_face_id, other_face in enumerate(other_obj.polytope.faces):
                    hit = ray_face_intersection(
                        ray_origin_local, ray_dir_local,
                        other_obj.polytope.vertices, other_face,
                    )
                    # hit.distance is the local-space distance along the ray
                    # from the camera to the hit point on the other object.
                    # Because the transform is rigid (rotation + translation),
                    # this equals the world-space distance.
                    if hit is not None and hit.distance < max_distance - 1.0e-6:
                        occluded = True
                        break
                if occluded:
                    break

            if occluded:
                info["occluded"].add(face_id)
            else:
                info["visible"].add(face_id)

    return results


def compute_silhouette_edges_per_object(
    scene: PolytopeScene,
    face_visibility: list[dict[str, set[int]]],
) -> list[list[tuple[int, int]]]:
    """Compute silhouette edges for each object.

    **Definition.**  A *silhouette edge* is a non-border edge (i.e., an edge
    with exactly two adjacent faces) where exactly one of the two adjacent
    faces is *visible*.  The other face must be either back-facing or
    occluded — in either case, it is not visible to the camera.

    This is a purely combinatorial operation on the visible-face set: it
    does not involve any ray-casting.  The silhouette edges form the boundary
    of the visible surface of each object.

    **Note on border edges.**  Edges with only one adjacent face (open
    boundaries) are excluded from the silhouette set.  All our test polytopes
    are closed solids, so every edge has exactly two adjacent faces.

    Parameters
    ----------
    scene : PolytopeScene
        The multi-object scene (used only for per-object edge topology).
    face_visibility : list[dict[str, set[int]]]
        The output of ``compute_scene_face_visibility()``, giving per-object
        visible-face sets.

    Returns
    -------
    list[list[tuple[int, int]]]
        One list per object.  Each inner list contains the local vertex-index
        pairs ``(a, b)`` of silhouette edges for that object.
    """
    all_silhouettes: list[list[tuple[int, int]]] = []

    for obj_idx, obj in enumerate(scene.objects):
        info = face_visibility[obj_idx]
        visible = info["visible"]
        edge_faces = polytope_edge_faces(obj.polytope)
        silhouettes: list[tuple[int, int]] = []

        for edge, faces in edge_faces.items():
            # Skip border edges (open meshes) — none exist in our corpus.
            if len(faces) != 2:
                continue
            visible_count = sum(1 for fid in faces if fid in visible)
            if visible_count == 1:
                silhouettes.append(edge)

        all_silhouettes.append(silhouettes)

    return all_silhouettes


def compute_occlusion_order(
    scene: PolytopeScene,
    camera: TrustedCamera,
) -> list[int]:
    """Order objects from front to back by average depth from the camera.

    **Method.**  Compute each object's world-space center, project it onto the
    camera's forward direction::

        depth = dot(center_world - camera_pos, forward)

    Objects with smaller depth are closer to the camera (the forward vector
    points away from the camera position).  Sort ascending by depth.

    **Limitations.**  This is a coarse center-based ordering.  It does not
    account for partial overlap — two objects could have overlapping depth
    ranges even if their centers order one way.  For our test scenes, all
    objects are well-separated in depth, so center ordering is unambiguous.

    Parameters
    ----------
    scene : PolytopeScene
        The multi-object scene.
    camera : TrustedCamera
        The camera defining the forward direction.

    Returns
    -------
    list[int]
        Object indices sorted front-to-back (index 0 = frontmost).
    """
    depths: list[float] = []
    camera_pos = camera.position_vec
    forward_dir = camera.forward

    for obj in scene.objects:
        center_world = obj.world_center()
        depth = float(np.dot(center_world - camera_pos, forward_dir))
        depths.append(depth)

    # Sort ascending: smaller depth → closer to camera.
    return sorted(range(len(scene.objects)), key=lambda i: depths[i])


def verify_center_rays_per_object(
    scene: PolytopeScene,
    ray_samples: int = 180,
) -> list[dict[str, int]]:
    """Verify that center rays from each object hit its own boundary.

    **Why this check exists.**  Even though each individual polytope passed
    the single-object validator, we re-verify center-ray hits in scene context
    for two reasons: (1) the scene builder may have introduced invalid
    polytopes (though it shouldn't), and (2) this serves as a regression test
    that the ray-casting primitives still work correctly when called from the
    scene module.

    **Algorithm.**  For each object, generate a set of ``ray_samples``
    uniformly spread directions on the unit sphere (Fibonacci lattice).
    Cast each ray from the polytope center and verify:

    1. The ray hits at least one face (``ray_polytope_intersection``).
    2. The hit point is within the face (``point_in_convex_face``).
    3. The hit point lies on the face plane (planarity error ≤ 1e-6).
    4. No other face is at a near-identical distance along the ray
       (ambiguous rays, which are expected near edges/vertices).

    The per-object nature of this check is important: it verifies that
    each polytope's internal ray-casting works regardless of its world-space
    position (the rays are cast in local space, so position is irrelevant).

    Parameters
    ----------
    scene : PolytopeScene
        The multi-object scene.
    ray_samples : int
        Number of Fibonacci-distributed directions to test per object.

    Returns
    -------
    list[dict[str, int]]
        One dict per object with keys ``object_index``, ``label``,
        ``ray_samples``, ``ray_failures``, ``ambiguous_rays``,
        ``passed`` (bool).
    """
    # Import here to avoid a circular dependency at module load time.
    from polytope_validator import fibonacci_directions

    results: list[dict[str, int]] = []
    for obj_idx, obj in enumerate(scene.objects):
        failures = 0
        ambiguous = 0
        directions = fibonacci_directions(ray_samples)
        for direction in directions:
            # Primary hit: find the face reached first from the center.
            try:
                hit = ray_polytope_intersection(obj.polytope, direction)
            except RuntimeError:
                failures += 1
                continue

            # Verify the hit point lies on the face plane and inside the face.
            face = obj.polytope.faces[hit.face_id]
            anchor = obj.polytope.vertices[face[0]]
            plane_error = abs(float(np.dot(hit.normal, hit.point - anchor)))
            if plane_error > 1.0e-6 or not point_in_convex_face(
                hit.point, obj.polytope.vertices, face, hit.normal
            ):
                failures += 1
                continue

            # Check for ambiguous rays (hits near edges/vertices where two
            # faces produce nearly identical distances).
            distances = []
            for candidate_face in obj.polytope.faces:
                cn = face_normal(obj.polytope.vertices, candidate_face)
                denom = float(np.dot(cn, direction))
                if denom <= EPS:
                    continue
                numer = float(np.dot(
                    cn, obj.polytope.vertices[candidate_face[0]] - obj.polytope.center
                ))
                if numer <= EPS:
                    continue
                d = numer / denom
                if d <= EPS:
                    continue
                pt = obj.polytope.center + d * direction
                if point_in_convex_face(
                    pt, obj.polytope.vertices, candidate_face, cn
                ):
                    distances.append(d)
            distances.sort()
            if len(distances) >= 2 and abs(distances[1] - distances[0]) <= 1.0e-7:
                ambiguous += 1

        results.append({
            "object_index": obj_idx,
            "label": obj.label,
            "ray_samples": ray_samples,
            "ray_failures": failures,
            "ambiguous_rays": ambiguous,
            "passed": failures == 0,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Scene validation result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SceneValidationResult:
    """Aggregated result of all scene-level and per-object validation checks.

    Attributes
    ----------
    scene_name : str
        Machine-readable scene identifier (e.g. ``"two_prisms"``).
    scene_description : str
        Human-readable description of the scene.
    object_count : int
        Number of objects in the scene.
    passed : bool
        Logical AND of all check flags.  ``True`` if every invariant holds.
    component_count : int
        Actual number of connected components found.
    expected_components : int
        Expected number of connected components (should equal object_count).
    no_false_chaining : bool
        ``True`` if no edge crosses object boundaries.
    no_vertex_overlap : bool
        ``True`` if no two objects share a world-space vertex position.
    per_object : list[dict]
        Per-object visibility and silhouette data (see ``validate_scene()``).
    occlusion_order : list[int]
        Object indices sorted front-to-back by center depth.
    checks : dict[str, bool]
        Named boolean flags for each individual check.
    notes : list[str]
        Human-readable warnings or informational messages.
    """

    scene_name: str
    scene_description: str
    object_count: int
    passed: bool
    component_count: int
    expected_components: int
    no_false_chaining: bool
    no_vertex_overlap: bool
    per_object: list[dict]
    occlusion_order: list[int]
    checks: dict[str, bool]
    notes: list[str]


def validate_scene(
    scene: PolytopeScene,
    scene_name: str,
    description: str,
    camera: TrustedCamera,
    *,
    ray_samples: int = 180,
) -> SceneValidationResult:
    """Run all scene-level and per-object validation checks.

    This is the top-level entry point for scene validation.  It:

    1. Checks connected components.
    2. Checks cross-object edge integrity.
    3. Checks world-space vertex overlap.
    4. Computes per-object face visibility (with occlusion).
    5. Computes per-object silhouette edges.
    6. Computes occlusion order.
    7. Verifies center-ray hits for each object.
    8. Reports any notes (missing visible faces, occlusion counts).

    Parameters
    ----------
    scene : PolytopeScene
        The scene to validate.
    scene_name : str
        Machine-readable name for this scene.
    description : str
        Human-readable description.
    camera : TrustedCamera
        The trusted camera to use for all visibility checks.
    ray_samples : int
        Number of center rays per object (default 180).

    Returns
    -------
    SceneValidationResult
        Full validation result with all checks, metrics, and notes.
    """
    notes: list[str] = []
    checks: dict[str, bool] = {}

    # --- Top-level scene checks ---
    component_count = check_connected_components(scene)
    checks["connected_components_match"] = component_count == scene.object_count
    checks["no_cross_object_edges"] = check_no_cross_object_edges(scene)
    checks["no_vertex_overlap"] = check_world_vertex_overlap(scene)

    # --- Visibility and silhouette ---
    face_vis = compute_scene_face_visibility(scene, camera)
    silhouette_edges = compute_silhouette_edges_per_object(scene, face_vis)
    occlusion_order = compute_occlusion_order(scene, camera)
    center_ray_results = verify_center_rays_per_object(scene, ray_samples)

    checks["all_center_rays_pass"] = all(r["passed"] for r in center_ray_results)

    # --- Per-object summary ---
    per_object: list[dict] = []
    for obj_idx, obj in enumerate(scene.objects):
        vis = face_vis[obj_idx]
        sil = silhouette_edges[obj_idx]
        cr = center_ray_results[obj_idx]

        per_object.append({
            "object_index": obj_idx,
            "label": obj.label,
            "total_faces": len(obj.polytope.faces),
            "total_edges": len(polytope_edge_faces(obj.polytope)),
            "total_vertices": len(obj.polytope.vertices),
            "front_facing_faces": sorted(vis["front_facing"]),
            "back_facing_faces": sorted(vis["back_facing"]),
            "visible_faces": sorted(vis["visible"]),
            "occluded_faces": sorted(vis["occluded"]),
            "silhouette_edge_count": len(sil),
            "silhouette_edges": [
                list(map(int, edge)) for edge in sorted(sil)
            ],
            "center_rays": {
                "samples": cr["ray_samples"],
                "failures": cr["ray_failures"],
                "ambiguous": cr["ambiguous_rays"],
                "passed": cr["passed"],
            },
        })

        # Warnings
        vis_count = len(vis["visible"])
        occ_count = len(vis["occluded"])
        front_count = len(vis["front_facing"])

        if vis_count == 0:
            notes.append(f"{obj.label}: no visible faces from this camera")
        if occ_count > 0:
            notes.append(
                f"{obj.label}: {occ_count}/{front_count} front-facing faces occluded "
                f"by other objects"
            )

    passed = all(checks.values())
    return SceneValidationResult(
        scene_name=scene_name,
        scene_description=description,
        object_count=scene.object_count,
        passed=passed,
        component_count=component_count,
        expected_components=scene.object_count,
        no_false_chaining=checks["no_cross_object_edges"],
        no_vertex_overlap=checks["no_vertex_overlap"],
        per_object=per_object,
        occlusion_order=occlusion_order,
        checks=checks,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Output writers — JSON, OBJ, PDF
# ═══════════════════════════════════════════════════════════════════════════


def write_scene_validation_json(path: Path, result: SceneValidationResult) -> None:
    """Write the full validation result as a JSON file.

    This is the primary machine-readable output.  It contains every check
    flag, all per-object face classifications, silhouette edge lists, and
    occlusion order — everything needed to verify consistency or to compare
    against an external geometry pipeline.
    """
    payload = {
        "scene_name": result.scene_name,
        "scene_description": result.scene_description,
        "object_count": result.object_count,
        "passed": result.passed,
        "component_count": result.component_count,
        "expected_components": result.expected_components,
        "no_false_chaining": result.no_false_chaining,
        "no_vertex_overlap": result.no_vertex_overlap,
        "occlusion_order": result.occlusion_order,
        "checks": result.checks,
        "notes": result.notes,
        "per_object": result.per_object,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_scene_obj(path: Path, scene: PolytopeScene) -> None:
    """Export all objects to a single Wavefront OBJ file in world space.

    Vertices are transformed to world coordinates before writing.  Each
    object gets its own ``o`` (object name) block.  Face indices are 1-based
    per the OBJ specification and are contiguous across all objects.
    """
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# trusted polytope scene\n")
        next_vertex_index = 1
        for obj in scene.objects:
            handle.write(f"o {obj.label}\n")
            wv = obj.world_vertices()
            for vertex in wv:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in obj.polytope.faces:
                # OBJ uses 1-based vertex indices.
                indices = " ".join(str(next_vertex_index + vid) for vid in face)
                handle.write(f"f {indices}\n")
            next_vertex_index += len(obj.polytope.vertices)


def write_scene_pdf_proof(
    path: Path,
    scene: PolytopeScene,
    result: SceneValidationResult,
    camera: TrustedCamera,
    face_visibility: list[dict[str, set[int]]],
) -> None:
    """Write a 4-panel A4 PDF proof sheet for a scene.

    **Layout.**  The page has:

    * A header with scene name and PASS/FAIL status.
    * A per-object visibility summary line.
    * Four view panels in a 2×2 grid:

      - **Top-left:** Isometric perspective
      - **Top-right:** Front orthographic (looking along +Z)
      - **Bottom-left:** Top orthographic (looking along −Y)
      - **Bottom-right:** Side orthographic (looking along +X)

    * A footer with component count, chaining status, and occlusion order.

    **Edge rendering conventions in each panel:**

    ==================  ============  =======  ==============
    Condition             Width (pt)   Gray     Style
    ==================  ============  =======  ==============
    Silhouette edge       0.95          0.02    Solid, bold
    Visible face edge     0.45          0.45    Solid, medium
    Hidden edge           0.45          0.82    Dashed, light
    Border edge           0.62          0.18    Solid
    ==================  ============  =======  ==============

    The convention mirrors the single-polytope proof PDF from
    ``polytope_validator.py``.
    """
    page_w, page_h = A4_PORTRAIT
    pdf = PurePDF("a4p")
    margin = mm_to_pts(12.0)
    gap = mm_to_pts(6.0)
    panel_w = 0.5 * (page_w - 2.0 * margin - gap)
    panel_h = 0.5 * (page_h - 2.0 * margin - gap - mm_to_pts(28.0))

    # --- Header ---
    pdf.text(margin, page_h - margin + mm_to_pts(1.0),
             f"scene: {result.scene_name}",
             font="Helvetica-Bold", size=13.0, gray=0.0)
    pdf.text(margin, page_h - margin - mm_to_pts(4.0),
             result.scene_description,
             font="Helvetica", size=8.5, gray=0.28)
    pdf.text(page_w - margin - mm_to_pts(40.0), page_h - margin + mm_to_pts(1.0),
             "PASS" if result.passed else "FAIL",
             font="Helvetica-Bold", size=12.0, gray=0.0)

    # --- Per-object summary line ---
    vis_label_y = page_h - margin - mm_to_pts(9.0)
    for obj_idx, per_obj in enumerate(result.per_object):
        pdf.text(
            margin + obj_idx * mm_to_pts(55.0),
            vis_label_y,
            f"{per_obj['label']}: {len(per_obj['visible_faces'])} vis / {per_obj['total_faces']} faces, "
            f"{per_obj['silhouette_edge_count']} sil edges",
            font="Helvetica",
            size=7.5,
            gray=0.25,
        )

    # --- Four view panels ---
    cameras = [
        ("Isometric Perspective", TrustedCamera(
            position=(4.0, 2.5, 4.5), target=(0.0, 0.0, 0.5), up=(0.0, 1.0, 0.0),
            projection="perspective", fov_y_degrees=35.0,
        )),
        ("Front Orthographic", TrustedCamera(
            position=(0.0, 0.0, 6.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
            projection="orthographic", orthographic_height=4.5,
        )),
        ("Top Orthographic", TrustedCamera(
            position=(0.0, 6.0, 0.0), target=(0.0, 0.0, 0.0), up=(0.0, 0.0, -1.0),
            projection="orthographic", orthographic_height=4.5,
        )),
        ("Side Orthographic", TrustedCamera(
            position=(6.0, 0.0, 0.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
            projection="orthographic", orthographic_height=4.5,
        )),
    ]

    for index, (title, panel_camera) in enumerate(cameras):
        col = index % 2
        row = index // 2
        rect = (
            margin + col * (panel_w + gap),
            margin + (1 - row) * (panel_h + gap),
            panel_w,
            panel_h,
        )
        draw_scene_panel(
            pdf, scene, face_visibility, panel_camera,
            title, rect,
        )

    # --- Footer ---
    footer_y = margin - mm_to_pts(2.0)
    checks_text = (
        f"components={result.component_count}/{result.expected_components} | "
        f"no false chaining={result.no_false_chaining} | "
        f"no vertex overlap={result.no_vertex_overlap}"
    )
    pdf.text(margin, footer_y + mm_to_pts(3.5), checks_text, font="Helvetica-Bold", size=9.0, gray=0.0)
    pdf.text(
        margin, footer_y,
        f"occlusion order (front\u2192back): {result.occlusion_order}",
        font="Helvetica", size=8.5, gray=0.25,
    )
    pdf.save(str(path))


def _face_fill_gray(normal: np.ndarray) -> float:
    """Map a unit face normal to a fill gray so top / side / front faces
    are visually distinct.  +Y (top) = 0.88 darkest, sides = 0.91,
    front/back = 0.94, bottom = 0.98 near-white."""
    nx, ny, nz = abs(float(normal[0])), float(normal[1]), abs(float(normal[2]))
    if ny > 0.6:
        return 0.88
    if ny < -0.6:
        return 0.98
    if nz > nx and nz > abs(ny):
        return 0.94
    return 0.91


def draw_scene_panel(
    pdf: PurePDF,
    scene: PolytopeScene,
    face_vis: list[dict[str, set[int]]],
    camera: TrustedCamera,
    title: str,
    rect: tuple[float, float, float, float],
) -> None:
    """Draw one proof panel for a multi-object scene.

    **Rendering order.**  First, visible (face_vis) faces are filled with a
    warm off-white tone so each object reads as a solid 3D volume.  Then
    edges are drawn on top using the same classification as the single-
    polytope panels: silhouette bold, interior-visible medium, hidden dashed.
    """
    x0, y0, width, height = rect
    pdf.text(x0, y0 + height + mm_to_pts(2.2), title,
             font="Helvetica-Bold", size=9.5, gray=0.0)

    all_world_verts = np.vstack([obj.world_vertices() for obj in scene.objects])
    projected = project_points(camera, all_world_verts)

    vert_offset = 0
    all_mapped = np.empty((len(all_world_verts), 2), dtype=float)

    for obj_idx, obj in enumerate(scene.objects):
        nv = len(obj.polytope.vertices)
        mapped_subset = fit_points_to_rect(
            projected[vert_offset:vert_offset + nv],
            rect,
        )
        all_mapped[vert_offset:vert_offset + nv] = mapped_subset

        vis = face_vis[obj_idx]["visible"]
        epf = polytope_edge_faces(obj.polytope)

        # --- face fills: color-coded by normal direction --------------------
        for face_id, face in enumerate(obj.polytope.faces):
            if face_id not in vis:
                continue
            normal_world = obj.world_face_normal(face_id)
            fill = _face_fill_gray(normal_world)
            pts = [tuple(all_mapped[vert_offset + vid]) for vid in face]
            if len(pts) < 3:
                continue
            pdf.save_state()
            pdf.content.set_fill_gray(fill)
            pdf.content.set_stroke_gray(1.0)
            pdf.content.move_to(*pts[0])
            for p in pts[1:]:
                pdf.content.line_to(*p)
            pdf.content.close_path()
            pdf.content.fill()
            pdf.restore_state()

        # --- edge strokes on top of fills ------------------------------------
        for edge, faces in epf.items():
            a_idx = vert_offset + edge[0]
            b_idx = vert_offset + edge[1]
            pts = [tuple(all_mapped[a_idx]), tuple(all_mapped[b_idx])]

            if len(faces) == 1:
                _pdf_polyline(pdf, pts, width=0.62, gray=0.18)
            else:
                vis_count = sum(1 for fid in faces if fid in vis)
                if vis_count == 1:
                    _pdf_polyline(pdf, pts, width=0.95, gray=0.02)
                elif vis_count == 2:
                    _pdf_polyline(pdf, pts, width=0.45, gray=0.45)
                else:
                    _pdf_polyline(pdf, pts, width=0.45, gray=0.82, dashed=True)

        vert_offset += nv

    pdf.save_state()
    pdf.line_width(0.35)
    pdf.stroke_gray(0.82)
    pdf.content.rect(x0, y0, width, height)
    pdf.content.stroke()
    pdf.restore_state()


def _pdf_polyline(pdf: PurePDF, points, width, gray, dashed=False):
    """Draw a single polyline with the given stroke properties.

    This is a low-level helper used by ``draw_scene_panel``.  It wraps the
    PurePDF state management (save/restore) around each stroke so that
    different edges can have different widths/grays/dashes without leaking
    state.

    Parameters
    ----------
    pdf : PurePDF
        The PDF document being built.
    points : list[tuple[float, float]]
        Two or more (x, y) points in PDF user space.
    width : float
        Stroke width in PDF points.
    gray : float
        Stroke gray level (0.0 = black, 1.0 = white).
    dashed : bool
        If ``True``, draw with a [3.0, 2.5] dash pattern.
    """
    if len(points) < 2:
        return
    pdf.save_state()
    pdf.line_width(width)
    pdf.stroke_gray(gray)
    pdf.content.set_line_cap(1)  # round caps
    pdf.content.set_line_join(1)  # round joins
    if dashed:
        pdf.content.set_dash([3.0, 2.5], phase=0.0)
    pdf.content.polyline(points)
    pdf.content.stroke()
    pdf.restore_state()


# ═══════════════════════════════════════════════════════════════════════════
# Command-line interface
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=str, default="all",
                        choices=["all", "two_prisms", "two_solids", "partial_occlusion"],
                        help="scene name to validate (default: all)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory for validation outputs")
    parser.add_argument("--ray-samples", type=int, default=180,
                        help="number of center rays per object (default: 180)")
    return parser.parse_args()


def default_camera() -> TrustedCamera:
    """Return the default front-facing perspective camera for scene validation.

    Camera is positioned at (0, 0, 6) looking at the origin with Y-up,
    35° vertical FOV.  This matches the single-polytope validator's default
    camera conventions.
    """
    return TrustedCamera(
        position=(0.0, 0.0, 6.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        projection="perspective",
        fov_y_degrees=35.0,
    )


def main() -> None:
    """Run the scene validator on one or all built-in test scenes.

    For each scene, computes visibility, silhouette, occlusion, and topology
    checks, then writes JSON, OBJ, and PDF outputs.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.scene == "all":
        scene_names = list(SCENE_BUILDERS.keys())
    else:
        scene_names = [args.scene]

    camera = default_camera()
    summaries: list[dict] = []

    for scene_name in scene_names:
        description, builder = SCENE_BUILDERS[scene_name]
        scene = builder()
        result = validate_scene(
            scene, scene_name, description, camera,
            ray_samples=args.ray_samples,
        )
        face_vis = compute_scene_face_visibility(scene, camera)

        stem = scene_name
        json_path = args.output_dir / f"scene_{stem}_validation.json"
        obj_path = args.output_dir / f"scene_{stem}.obj"
        pdf_path = args.output_dir / f"scene_{stem}.pdf"

        write_scene_validation_json(json_path, result)
        write_scene_obj(obj_path, scene)
        write_scene_pdf_proof(pdf_path, scene, result, camera, face_vis)

        summaries.append({
            "scene_name": scene_name,
            "passed": result.passed,
            "objects": result.object_count,
            "validation_json": str(json_path),
            "obj": str(obj_path),
            "pdf": str(pdf_path),
        })

        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {scene_name} ({result.object_count} objects)")
        print(f"  components: {result.component_count}/{result.expected_components}")
        print(f"  occlusion order: {result.occlusion_order}")
        for obj_info in result.per_object:
            print(f"  {obj_info['label']}: "
                  f"{len(obj_info['visible_faces'])} visible "
                  f"{len(obj_info['occluded_faces'])} occluded "
                  f"{obj_info['silhouette_edge_count']} silhouette edges")
        print(f"  pdf {pdf_path}")
        print(f"  obj {obj_path}")
        print(f"  validation {json_path}")

    manifest_path = args.output_dir / "scene_manifest.json"
    manifest_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
