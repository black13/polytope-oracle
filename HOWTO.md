# Polytope Oracle — HOWTO

## What this is

A collection of tools that prove geometry is correct before trusting any
rendering pipeline. Four repos, four roles:

```
polytope-oracle/     Python — trusted numeric validator (source of truth)
contours/            C++   — Bénard contour extraction + Freestyle libraries
python-freestyle/    Python — Freestyle port under test
freestyle-qt5/       C++   — Qt5 viewer for interactive exploration
```

The oracle validates everything bottom-up. Every check, every line in every
PDF, every edge classification is derived from the same 12 vertices, 20
faces, and 1 camera position of an icosahedron.

---

## Quick start

```bash
# Clone the oracle
git clone git@github.com:black13/polytope-oracle.git
cd polytope-oracle

# You need numpy
# /Users/jjosburn/.conda/envs/dick-und-jane-repl/bin/python3

# Run the 7-shape validator
python3 polytope_validator.py --case all

# Run the 3-scene detached object validator
python3 polytope_scene.py --scene all

# Generate a silouhette-proximity engraving (the good one)
python3 proximity_engraving.py --shape icosahedron --subdivide --subdiv-levels 3

# Generate a rotating silouhette accumulation (visual hull)
python3 rotating_silhouette.py --shape cube --azimuth-steps 180 --elevation 15 --subdivide

# Smooth contour extraction (n·v = 0 on subdiv limit surface)
python3 smooth_contour.py --shape icosahedron --subdiv 4
```

Output goes to `output/`. Each script writes a PDF you can open directly.

---

## What each file does

### `polytope_numbers.py` — the numeric model

Defines the `Polytope` dataclass:
```python
Polytope(
    name="icosahedron",
    vertices=np.array([...]),   # 12×3 float array
    faces=[[0,11,5], ...],      # 20 triangular faces
    center=np.array([0,0,0]),   # origin
    sphere_radius=1.0,          # bounding sphere
)
```

Every operation starts from these numbers. Ray-face intersection is analytic:
```
t = n·(anchor - origin) / n·direction
```
Then check if the hit point is inside the convex polygon via signed halfplane
tests around the perimeter. No triangulation. No interpolation.

Built-in shapes: `cube`, `octahedron`, `icosahedron`.
Custom shapes: load from JSON with `--polytope-json path.json`.

### `polytope_validator.py` — the 20-check validator

Proves a single polytope is a well-formed convex solid. Checks everything
from face-index validity to ray-cast boundary hits. Writes proof PDFs with
face fills color-coded by normal direction.

```bash
python3 polytope_validator.py --case cube
```

### `polytope_scene.py` — detached multi-object scenes

Places multiple polytopes in world space and validates:
- Connected component count (must equal object count)
- No false chaining across objects
- Per-object visible/occluded face sets (with inter-object occlusion)
- Silhouette edges per object
- Occlusion ordering (front-to-back by depth)

```bash
python3 polytope_scene.py --scene partial_occlusion
```

### `proximity_engraving.py` — the keeper

Generates Piranesi-style engraving where line density = 1 / distance-to-
silhouette. Within each visible face, strokes follow the projected face
normal. Lines gather near the occluding contour and thin out toward the
face center.

```bash
python3 proximity_engraving.py --shape icosahedron --subdivide --subdiv-levels 4
# 8400+ strokes, print-quality. Open output/proximity_engraving_icosahedron_subdiv4.pdf
```

### `rotating_silhouette.py` — visual hull

Orbits a camera around the object at fixed distance, extracts the silhouette
at each azimuth step, accumulates all projected contours into a single
drawing. Line density reveals surface curvature.

```bash
python3 rotating_silhouette.py --shape cube --azimuth-steps 180 --elevation 15 --subdivide
```

### `smooth_contour.py` — n·v = 0 extraction

Finds occluding contours on the Catmull-Clark limit surface by detecting
n·v sign changes along subdivided edges, chains the zero-crossing points
through faces, and renders them as cubic Bezier curves.

```bash
python3 smooth_contour.py --shape cube --subdiv 4
```

### `subdivide.py` — pure Python Catmull-Clark

No dependencies beyond numpy. Subdivides arbitrary polygonal meshes.
Used by proximity_engraving, rotating_silhouette, and smooth_contour.

### `pure_pdf.py` — zero-dependency PDF generator

Builds valid PDF-1.4 files from raw operator strings. No external libraries.
Every proof PDF is constructed byte-by-byte with PostScript operators
emitted as text, zlib-compressed, and written with a proper xref table.

---

## C++ verification flow

The `contours/` repo contains a pure C++11 port of the original Freestyle
NPR rendering engine. The verification flow:

```bash
# 1. Generate a PLY from the Python oracle
cd polytope-oracle
python3 -c "
from polytope_numbers import make_polytope, face_normal
import numpy as np
ico = make_polytope('icosahedron')
...write PLY...
"

# 2. Run the C++ Freestyle renderer on it
cd ../contours/build
./freestyle_render /tmp/oracle_ico.ply --json

# 3. Compare JSON output against Python oracle expectations
```

The C++ binary outputs:
```json
{
  "faces": 20,
  "edges": 30,
  "front_facing_faces": 8,
  "silhouette_edges": 6,
  "border_edges": 0,
  "crease_edges": 0,
  "ridge_edges": 0,
  "valley_edges": 0,
  "suggestive_edges": 0
}
```

Verified MATCH on icosahedron and cube.

---

## The Bénard contour pipeline

`contours/build/contours_cli` implements the Bénard et al. 2014 smooth
occluding contour extraction:

```bash
cd contours/build
./contours_cli icosphere -subd 3 -cam 2.8 2.0 3.5 -o /tmp/contour
# Output: refined PLY with contour-consistent topology
```

Key stats from the run:
- ZC (zero-crossings): 58 n·v sign changes detected
- CUSPS: 1 contour singularity
- flips: 149 edge flips to align mesh with contour
- Output: initial + refined PLY files

---

## The Qt5 viewer

`freestyle-qt5/build-fresh/freestyle_qt5_viewer` is a macOS binary that:
- Opens a window with a 3DS model browser
- Runs the Freestyle pipeline on the selected model
- Displays wireframe + feature lines with keyboard orbit/zoom

```bash
cd freestyle-qt5/build-fresh
./freestyle_qt5_viewer
# Arrow keys: orbit, +/-: zoom, R: reset view
```

---

## PDF conventions

Every proof PDF uses the same rendering rules:

| Element | Width | Gray | Style |
|---------|-------|------|-------|
| Silhouette edge | 0.95 pt | 0.02 (near-black) | Solid, bold |
| Visible face seam | 0.35 pt | 0.55 | Solid, light |
| Hidden edge | 0.45 pt | 0.82 | Dashed |
| Border edge | 0.62 pt | 0.18 | Solid |
| Face fill (top) | — | 0.88 | Filled polygon |
| Face fill (side) | — | 0.91 | Filled polygon |
| Face fill (front) | — | 0.94 | Filled polygon |
| Proximity stroke | varies | varies | Line segment |

---

## How to add a new shape

1. Add a builder function in `polytope_numbers.py` that returns a `Polytope`
2. Add a `KnownPolytopeCase` entry in `polytope_validator.py` with the
   expected vertex/edge/face counts and Euler characteristic
3. Run `python3 polytope_validator.py --case yourshape`
4. If it passes, generate PLY and run the C++ pipeline for verification

Example for a new regular polyhedron:
```python
def make_dodecahedron() -> Polytope:
    phi = (1 + math.sqrt(5)) / 2
    vertices = np.array([
        (±1, ±1, ±1), (0, ±1/phi, ±phi), (±1/phi, ±phi, 0), (±phi, 0, ±1/phi)
    ])
    faces = [...]  # 12 pentagonal faces
    return build_polytope("dodecahedron", vertices, faces)
```

---

## Interpreter

All Python scripts use this conda environment:
```
/Users/jjosburn/.conda/envs/dick-und-jane-repl/bin/python3
```

VS Code is pre-configured (`.vscode/launch.json`, `.vscode/tasks.json`).
Press F5 to debug any script, Ctrl+Shift+B for task list.

---

## Trust summary

| Component | Status |
|-----------|--------|
| Python oracle (polytope-oracle/) | ✅ Trusted — 7/7 single, 3/3 scene PASS |
| Python Freestyle geometry_truth | ✅ Verified — 3/3 MATCH on polytope scenes |
| C++ Freestyle topology (faces/edges) | ✅ Verified — matches oracle |
| C++ Freestyle front-facing | ✅ Verified — computes n·v directly |
| C++ Freestyle silouhette | ✅ Verified — matches oracle |
| Bénard contour pipeline | ✅ Compiles, runs, produces refined PLY |
| Qt5 viewer | ✅ Builds on macOS, displays 3DS models |
