```mermaid
flowchart TB
    subgraph INPUT["Geometry Input"]
        PLY["PLY / Polytope OBJ\n(vertices + faces)"]
    end

    subgraph BENARD["Bénard Layer (OpenSubdiv)"]
        SUBD["Catmull-Clark subdivision\n→ smooth limit surface"]
    end

    subgraph WE["Winged-Edge (Freestyle)"]
        BUILD["build winged-edge\n(vertices, edges, faces, adjacency)"]
        FACE["face normals / centers"]
        CURVE["curvature estimation\nK, k1, k2, d1, d2\nper vertex"]
    end

    subgraph VIS["Visibility"]
        FB["front/back culling\nn·v > 0 → visible faces"]
        SIL["silhouette edge detection\n1 front face + 1 back face per edge"]
    end

    subgraph FIELD["Tone Field + Curve Field"]
        TONE["tone field\nstroke density\n∝ 1/distance_to_silhouette"]
        CURV["curve field\nstroke direction\n= project(d1) into image\n(max principal curvature)"]
    end

    subgraph RENDER["Rendering Passes"]
        MC["Monte Carlo strokes\n(light-eroded dust)"]
        FILL["face fill zigzag\n(Bezier curves)"]
        EDGE["silhouette strokes\n(scratchy Perlin)"]
        SHADOW["ground shadow\n(ray occlusion)"]
        PENCIL["pencil texture\n(Sousa & Buchanan\ntaper + grain)"]
    end

    subgraph OUTPUT["Output"]
        PDF["PDF (A4, print-ready)"]
        SVG["SVG (browser)"]
    end

    PLY --> SUBD
    SUBD --> BUILD
    BUILD --> FACE
    BUILD --> CURVE
    FACE --> FB
    FB --> SIL
    FB --> VISIBLE["visible face set"]

    CURVE --> CURV
    SIL --> TONE
    VISIBLE --> TONE

    TONE --> MC
    TONE --> FILL
    CURV --> MC
    CURV --> FILL
    SIL --> EDGE
    VISIBLE --> SHADOW

    MC --> PENCIL
    FILL --> PENCIL
    EDGE --> PENCIL
    SHADOW --> PENCIL

    PENCIL --> PDF
    PENCIL --> SVG

    style TONE fill:#f9f,stroke:#333
    style CURV fill:#9f9,stroke:#333
    style CURVE fill:#ff9,stroke:#333
```

**Tone field** (pink): stroke density driven by silhouette proximity — lines gather where the form turns away.  
**Curve field** (green): stroke direction driven by d1 — lines follow the surface's natural grain.  

The missing link (yellow): curvature estimation on the limit surface. `freestyle/winged_edge/Curvature.cpp` exists but only computes per-face neighbor normal variance. Needs upgrading to full K, k1, k2, d1, d2.
