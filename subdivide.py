#!/usr/bin/env python3
"""Catmull-Clark subdivision for arbitrary polygonal meshes.

Pure Python — no trimesh, no OpenSubdiv.  Operates on the same
numeric model as the polytope oracle (vertices as float triplets,
faces as index lists).

Used to produce smooth limit-surface approximations from convex
polytope primitives so occluding contour extraction (n·v = 0
root-finding) can be verified against analytic expectations.
"""

from __future__ import annotations

import numpy as np

EPS = 1.0e-12


def catmull_clark(
    vertices: np.ndarray,
    faces: list[list[int]],
    levels: int = 1,
) -> tuple[np.ndarray, list[list[int]]]:
    """Apply Catmull-Clark subdivision *levels* times.

    Parameters
    ----------
    vertices : np.ndarray (V×3)
        Vertex positions.
    faces : list[list[int]]
        Face lists (indices into vertices).  Supports arbitrary n-gons.
    levels : int
        Number of subdivision steps (default 1).

    Returns
    -------
    (np.ndarray, list[list[int]])
        Subdivided vertices and faces.
    """
    for _ in range(levels):
        vertices, faces = _subdivide_once(vertices, faces)
    return vertices, faces


def _subdivide_once(
    vertices: np.ndarray,
    faces: list[list[int]],
) -> tuple[np.ndarray, list[list[int]]]:
    """Single Catmull-Clark subdivision pass."""
    nv = len(vertices)
    nf = len(faces)

    # --- edge data structures --------------------------------------------
    # canonical_edge(a,b) where a < b
    edge_key = {}
    edge_faces = {}
    edge_verts = {}
    edge_midpoints = {}
    next_edge_id = 0
    for fi, face in enumerate(faces):
        k = len(face)
        for i in range(k):
            a, b = face[i], face[(i + 1) % k]
            key = (a, b) if a < b else (b, a)
            if key not in edge_key:
                edge_key[key] = next_edge_id
                edge_faces[key] = []
                edge_verts[key] = (a, b)
                edge_midpoints[key] = (
                    (vertices[a] + vertices[b]) * 0.5
                )
                next_edge_id += 1
            edge_faces[key].append(fi)

    # --- face points: centroid of each face -------------------------------
    face_points = np.empty((nf, 3), dtype=float)
    for fi, face in enumerate(faces):
        face_points[fi] = vertices[face].mean(axis=0)

    # --- edge points: average of edge endpoints + adjacent face points ---
    edge_points = {}
    for key, eid in edge_key.items():
        a, b = edge_verts[key]
        adj_faces = edge_faces[key]
        mid = edge_midpoints[key]
        if len(adj_faces) == 2:
            fp_sum = face_points[adj_faces[0]] + face_points[adj_faces[1]]
            edge_points[key] = (mid * 2.0 + fp_sum) * 0.25
        else:
            # Boundary edge: just the midpoint
            edge_points[key] = mid

    # --- vertex adjacency for new vertex positions ------------------------
    vertex_adj_faces = {i: [] for i in range(nv)}
    vertex_adj_edges = {i: [] for i in range(nv)}
    for fi, face in enumerate(faces):
        k = len(face)
        for i in range(k):
            a = face[i]
            b = face[(i + 1) % k]
            vertex_adj_faces[a].append(fi)
            key = (a, b) if a < b else (b, a)
            vertex_adj_edges[a].append(key)

    # --- new vertex positions ---------------------------------------------
    new_vertices: list[np.ndarray] = []

    # Original vertices relocated
    for vi in range(nv):
        adj_f = vertex_adj_faces[vi]
        adj_e = vertex_adj_edges[vi]
        n = len(adj_f)
        if n == 0:
            new_vertices.append(vertices[vi].copy())
            continue
        f_avg = sum(face_points[fi] for fi in adj_f) / n
        r_avg = sum(edge_midpoints[key] for key in adj_e) / n
        p = vertices[vi]
        new_vertices.append((f_avg + 2.0 * r_avg + (n - 3.0) * p) / n)

    # Face points become vertices
    for fi in range(nf):
        new_vertices.append(face_points[fi])

    # Edge points become vertices
    edge_vertex_index = {}
    for key in edge_key:
        idx = len(new_vertices)
        edge_vertex_index[key] = idx
        new_vertices.append(edge_points[key])

    new_vertex_array = np.array(new_vertices, dtype=float)

    # --- build new faces --------------------------------------------------
    # Face point index offset: original vertices come first, then face points
    fp_offset = nv

    new_faces: list[list[int]] = []
    for fi, face in enumerate(faces):
        k = len(face)
        fp_idx = fp_offset + fi
        for i in range(k):
            a = face[i]
            b = face[(i + 1) % k]
            key = (a, b) if a < b else (b, a)
            ep_idx = edge_vertex_index[key]
            # Each original edge + face point forms a quad:
            #   original_vertex_a → edge_point → face_point → next_edge_point
            # But the Catmull-Clark face for a k-gon produces k quads:
            #   (vi, e_i, fp, e_{i-1})
            prev_key = (face[i - 1], a) if face[i - 1] < a else (a, face[i - 1])
            prev_ep = edge_vertex_index[prev_key]
            new_faces.append([a, ep_idx, fp_idx, prev_ep])

    return new_vertex_array, new_faces


def normalize_to_sphere(
    vertices: np.ndarray,
    faces: list[list[int]],
    radius: float = 1.0,
) -> np.ndarray:
    """Project vertices onto a sphere of given radius.

    Used after subdivision to make a polytope approach the sphere.
    """
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return vertices * (radius / norms)
