# Polytope Oracle — Prove, Validate, Extend

## Trust Boundary

### Trusted (polytope-oracle/)

- `polytope_numbers.py` — numeric polytope model + ray intersection
- `polytope_validator.py` — 20-check single-polytope validator + proof PDFs
- `polytope_scene.py` — detached multi-object scene validator
- `pure_pdf.py` — zero-dependency PDF generator
- `polytope_numbers_example.json` — sample polytope JSON format

### Validated (pyfreestyle code proven against oracle)

- `pyfreestyle/geometry_truth.py` — front-facing, visibility, silhouette on polytopes
- `pyfreestyle/mesh.py` — edge topology, face normals, face centers
- `pyfreestyle/pdf.py` (face_visibility) — QI=0 classification

### Untested

- `pyfreestyle/view_map_builder.py` — ViewMap pipeline
- `pyfreestyle/features.py` — crease, suggestive contour, ridge/valley
- `pyfreestyle/chaining.py` — edge chain computation
- `pyfreestyle/winged_edge.py` — curvature estimation
- `pyfreestyle/camera.py` — projection accuracy vs oracle camera
- `examples/icosahedron_visible.py` — smooth icosphere pipeline
- `examples/icosahedron_ground_truth.py` — smooth icosphere pipeline

### Separate domain (smooth derivatives)

- `pyfreestyle/curvature.py`
- `pyfreestyle/steerable.py`

## How to run

```bash
# Single polytope validation (7 known shapes)
python3 polytope-oracle/polytope_validator.py --case all

# Detached multi-object scene validation (3 scenes)
python3 polytope-oracle/polytope_scene.py --scene all

# Oracle-vs-Freestyle comparison (3 scenes)
python3 tools/freestyle_comparison.py --scene all
```

## Validation corpus

| Shape | V | E | F | Euler | Status |
|-------|---|---|---|-------|--------|
| tetrahedron | 4 | 6 | 4 | 2 | PASS |
| cube | 8 | 12 | 6 | 2 | PASS |
| octahedron | 6 | 12 | 8 | 2 | PASS |
| icosahedron | 12 | 30 | 20 | 2 | PASS |
| prism_4 | 8 | 12 | 6 | 2 | PASS |
| prism_6 | 12 | 18 | 8 | 2 | PASS |
| prism_8 | 16 | 24 | 10 | 2 | PASS |

| Scene | Objects | Components | Occlusion | Silhouette | Status |
|-------|---------|------------|-----------|------------|--------|
| two_prisms | 2 | 2/2 | none | 6+6 edges | PASS |
| two_solids | 2 | 2/2 | none | 6+4 edges | PASS |
| partial_occlusion | 2 | 2/2 | 1 face occluded | 4+0 edges | PASS |

## Next extensions

A. ViewMap pipeline (chaining, T-vertices, QI per ViewEdge)
B. Feature edge classifiers (crease, suggestive contours, ridges/valleys)
C. Smooth mesh domain (curvature on icosphere subdivisions)
D. Full pipeline end-to-end (mesh → winged edge → features → viewmap → style → PDF)
