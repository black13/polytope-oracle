"""Curvature field proof — strokes follow d1 instead of cross(n,v).

Generates two comparison PDFs for the same icosahedron:
  1. Old: stroke direction = cross(face_normal, view_dir)
  2. New: stroke direction = project(d1) into image
       where d1 = max principal curvature direction

The difference: d1 follows the surface's natural grain.
On a sphere (subdivided polytope), d1 wraps concentrically
around the curvature center.  cross(n,v) follows the view-
dependent grain.  One is surface-intrinsic, the other is
view-dependent.

This proves the curve field concept before porting to C++.
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from pure_pdf import A4_PORTRAIT, PurePDF, mm_to_pts
from polytope_numbers import make_polytope, face_normal, face_center, normalize
from polytope_validator import TrustedCamera
from subdivide import catmull_clark, normalize_to_sphere
from proximity_engraving import classify_faces, project_onto_image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'output')


def estimate_curvature(vertices, faces):
    """Estimate principal curvatures K, k1, k2, d1, d2 per vertex.

    Uses the Rusinkiewicz method: fit a second-order surface to
    the one-ring neighborhood of each vertex, then compute the
    shape operator eigenvalues/vectors.

    For a subdivided polytope approaching a sphere, k1 ≈ k2 ≈ 1/R
    and d1, d2 are any pair of orthogonal tangent vectors.
    """
    nv = len(vertices)
    vn = np.zeros_like(vertices)
    for face in faces:
        fn = face_normal(vertices, face)
        for vid in face:
            vn[vid] += fn
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    vn = vn / norms

    # Build per-vertex one-ring neighbors from faces
    neighbors = {i: set() for i in range(nv)}
    for face in faces:
        k = len(face)
        for i in range(k):
            a, b = face[i], face[(i + 1) % k]
            neighbors[a].add(b)
            neighbors[b].add(a)
    neighbors = {i: list(nb) for i, nb in neighbors.items()}

    # For each vertex, estimate shape operator in tangent plane
    k1_arr = np.zeros(nv)
    k2_arr = np.zeros(nv)
    d1_arr = np.zeros((nv, 3))
    d2_arr = np.zeros((nv, 3))

    for vi in range(nv):
        n = vn[vi]
        # Build tangent basis
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tu = normalize(np.cross(n, ref))
        tv = normalize(np.cross(n, tu))

        nbs = neighbors[vi]
        if len(nbs) < 3:
            k1_arr[vi] = 1.0
            k2_arr[vi] = 1.0
            d1_arr[vi] = tu
            d2_arr[vi] = tv
            continue

        # Accumulate normal differences in tangent coordinates
        A = np.zeros((3, 3))
        b = np.zeros(3)
        count = 0
        for ni in nbs:
            delta = vertices[ni] - vertices[vi]
            # Project onto tangent plane
            t_delta = delta - n * float(np.dot(delta, n))
            x = float(np.dot(t_delta, tu))
            y = float(np.dot(t_delta, tv))
            if x * x + y * y < 1e-16:
                continue
            # Normal difference along this edge
            dn = vn[ni] - vn[vi]
            nd = float(np.dot(dn, t_delta)) / (x * x + y * y + 1e-12)
            row = np.array([x * x, 2 * x * y, y * y])
            for ri in range(3):
                b[ri] += row[ri] * nd
                for ci in range(3):
                    A[ri, ci] += row[ri] * row[ci]
            count += 1

        if count < 3:
            k1_arr[vi] = 1.0
            k2_arr[vi] = 1.0
            d1_arr[vi] = tu
            d2_arr[vi] = tv
            continue

        try:
            abc = np.linalg.solve(A, b)
            a, b_val, c = abc[0], abc[1], abc[2]
        except np.linalg.LinAlgError:
            k1_arr[vi] = 1.0
            k2_arr[vi] = 1.0
            d1_arr[vi] = tu
            d2_arr[vi] = tv
            continue

        # Eigenvalues of [a, b; b, c]
        trace = a + c
        det = a * c - b_val * b_val
        disc = math.sqrt(max(0.0, trace * trace - 4.0 * det))
        k1 = 0.5 * (trace + disc)
        k2 = 0.5 * (trace - disc)

        # Eigenvector for k1: solve (a - k1) * x + b * y = 0
        if abs(b_val) > 1e-12:
            e1_2d = np.array([b_val, k1 - a])
        elif a >= c:
            e1_2d = np.array([1.0, 0.0])
        else:
            e1_2d = np.array([0.0, 1.0])
        e1_2d = e1_2d / np.linalg.norm(e1_2d)

        d1 = tu * e1_2d[0] + tv * e1_2d[1]
        d2 = normalize(np.cross(n, d1))

        k1_arr[vi] = k1
        k2_arr[vi] = k2
        d1_arr[vi] = d1
        d2_arr[vi] = d2

    K = k1_arr * k2_arr  # Gaussian
    return K, k1_arr, k2_arr, d1_arr, d2_arr, vn


def main():
    shape = 'icosahedron'
    subdiv = 3

    # Build and subdivide
    poly = make_polytope(shape)
    verts = poly.vertices.copy()
    faces = [list(f) for f in poly.faces]
    verts, faces = catmull_clark(verts, faces, levels=subdiv)
    verts = normalize_to_sphere(verts, faces, radius=poly.sphere_radius)

    # Curvature
    K, k1, k2, d1, d2, vn = estimate_curvature(verts, faces)
    print(f"Subdivided {shape}: {len(verts)} verts, {len(faces)} faces")
    print(f"  K range:  [{K.min():.4f}, {K.max():.4f}]")
    print(f"  k1 range: [{k1.min():.4f}, {k1.max():.4f}]")
    print(f"  k2 range: [{k2.min():.4f}, {k2.max():.4f}]")

    # Camera
    cam_pos = np.array([2.8, 2.0, 3.5])
    cam_tgt = np.zeros(3)
    front, _, _ = classify_faces(poly, cam_pos)

    # Generate strokes using d1 as direction (old: cross(n,v))
    projected = project_onto_image(verts, cam_pos, cam_tgt)
    strokes_old = []
    strokes_d1 = []

    for fi in front:
        face = faces[fi]
        fn = face_normal(verts, face)
        fc = face_center(verts, face)

        # Sky polygon in image
        poly_2d = np.array([projected[vid] for vid in face])

        # Old direction: cross(fn, view_dir)
        vd = normalize(cam_pos - fc)
        old_dir_3d = normalize(np.cross(fn, vd))
        if np.linalg.norm(old_dir_3d) < 1e-9:
            old_dir_3d = np.cross(fn, np.array([0, 1, 0]))

        # New direction: project average d1 of face vertices
        avg_d1 = np.zeros(3)
        for vid in face:
            avg_d1 += d1[vid]
        avg_d1 = normalize(avg_d1)
        if np.linalg.norm(avg_d1) < 1e-9:
            avg_d1 = old_dir_3d

        # Project directions to 2D
        po = project_onto_image(np.array([fc]), cam_pos, cam_tgt)[0]
        p1_old = project_onto_image(np.array([fc + old_dir_3d * 0.01]), cam_pos, cam_tgt)[0]
        p1_new = project_onto_image(np.array([fc + avg_d1 * 0.01]), cam_pos, cam_tgt)[0]
        dir_2d_old = p1_old - po
        dir_2d_new = p1_new - po
        dir_2d_old /= max(np.linalg.norm(dir_2d_old), 1e-9)
        dir_2d_new /= max(np.linalg.norm(dir_2d_new), 1e-9)

        # Stroke endpoints clipped to face polygon in 2D.
        # Sutherland-Hodgman: clip segment to convex polygon.
        def clip_segment_to_poly(x1, y1, x2, y2, poly):
            seg = np.array([[x1, y1], [x2, y2]])
            nv = len(poly)
            for i in range(nv):
                a = poly[i]
                b = poly[(i+1) % nv]
                dx = b[0] - a[0]; dy = b[1] - a[1]
                nx, ny = -dy, dx  # inward normal (assuming CCW)
                new_seg = []
                for j in range(len(seg)):
                    p_cur = seg[j]
                    p_prev = seg[(j-1) % len(seg)]
                    d_cur = nx*(p_cur[0]-a[0]) + ny*(p_cur[1]-a[1])
                    d_prev = nx*(p_prev[0]-a[0]) + ny*(p_prev[1]-a[1])
                    if d_cur >= -1e-10:
                        new_seg.append(p_cur)
                    if (d_cur >= -1e-10) != (d_prev >= -1e-10):
                        denom = d_prev - d_cur
                        if abs(denom) > 1e-12:
                            t = d_prev / denom
                            new_seg.append(p_prev + t * (p_cur - p_prev))
                seg = np.array(new_seg) if new_seg else np.empty((0,2))
                if len(seg) < 2:
                    return None
            return (float(seg[0,0]), float(seg[0,1])), (float(seg[1,0]), float(seg[1,1]))

        # Clipped strokes
        half = 12.0
        for name, dr, out_list in [('old', dir_2d_old, strokes_old),
                                   ('d1', dir_2d_new, strokes_d1)]:
            a = po - dr * half
            b = po + dr * half
            clipped = clip_segment_to_poly(a[0], a[1], b[0], b[1], poly_2d)
            if clipped:
                (ax, ay), (bx, by) = clipped
                out_list.append(((float(ax), float(ay)),
                                 (float(bx), float(by)),
                                 0.6, 0.05))

    # Write comparison PDF
    pdf = PurePDF('a4p')
    pw, ph = A4_PORTRAIT
    margin = mm_to_pts(18)

    pdf.text(margin, ph - margin,
             f'{shape} subdiv {subdiv} — stroke direction comparison',
             font='Helvetica-Bold', size=12.0, gray=0.0)

    for idx, (strokes, title) in enumerate(
        [(strokes_old, 'cross(n, v) — view-dependent'),
         (strokes_d1, 'project(d1) — surface-intrinsic')]):

        rect = (margin, margin + mm_to_pts(12),
                pw - 2 * margin, (ph - 2 * margin - mm_to_pts(40)) / 2)
        if idx == 0:
            rect = (margin, margin + rect[3] + mm_to_pts(8),
                    rect[2], rect[3])

        x0, y0, cw, ch = rect
        pdf.text(x0, y0 + ch + mm_to_pts(2), title,
                 font='Helvetica-Bold', size=9.5, gray=0.0)

        # Fit face vertices
        face_pts_2d = np.vstack([np.array([projected[vid] for vid in f]) for f in [faces[fi] for fi in front]])

        mins = face_pts_2d.min(axis=0); maxs = face_pts_2d.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pad = 0.08; mins = mins - pad*span; maxs = maxs + pad*span
        span = np.maximum(maxs - mins, 1e-6)
        sc = min(cw / span[0], ch / span[1])
        ox = x0 + (cw - span[0]*sc)/2 - mins[0]*sc
        oy = y0 + (ch - span[1]*sc)/2 - mins[1]*sc
        def mp(x, y):
            return (ox + x * sc, oy + y * sc)

        pdf.save_state()
        pdf.clip_rect(x0, y0, cw, ch)

        # Draw face edges lightly
        for fi in front:
            face = faces[fi]
            for i in range(len(face)):
                a, b = face[i], face[(i+1) % len(face)]
                pdf.line_width(0.15)
                pdf.stroke_gray(0.85)
                pdf.content.move_to(mp(float(projected[a,0]), float(projected[a,1]))[0],
                                    mp(float(projected[a,0]), float(projected[a,1]))[1])
                pdf.content.line_to(mp(float(projected[b,0]), float(projected[b,1]))[0],
                                    mp(float(projected[b,0]), float(projected[b,1]))[1])
                pdf.content.stroke()

        # Draw strokes
        for (x0s, y0s), (x1s, y1s), thick, gray in strokes:
            xp0, yp0 = mp(x0s, y0s)
            xp1, yp1 = mp(x1s, y1s)
            pdf.line_width(thick)
            pdf.stroke_gray(gray)
            pdf.content.set_line_cap(1)
            pdf.content.move_to(xp0, yp0)
            pdf.content.line_to(xp1, yp1)
            pdf.content.stroke()

        pdf.restore_state()

    output_path = os.path.join(OUT, f'curvature_field_{shape}_subdiv{subdiv}.pdf')
    os.makedirs(OUT, exist_ok=True)
    pdf.save(output_path)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
