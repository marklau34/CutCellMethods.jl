# Brief: Cut-cell geometry from segmented triangle patches

Sep 28, 2026 · @Mark

## Context and goal

Build a standalone Julia module, `CutCellGeometry`, that turns a triangulated hull surface into cut-cell geometry on the staggered MAC grid of the planing-hull solver. Its outputs are volume fractions, face apertures, and per-patch embedded-boundary faces for every control-volume family.

The accuracy target is the transom and chine corners. Planar patches must come out exact to roundoff, curved patches second order (O(h²)), and discrete closure must hold to machine precision in every control volume.

The method is to segment the surface into smooth patches at feature edges, fit one plane per patch in each cut control volume, and clip the control volume with a small convex-polyhedron clipper. Single-SDF, marching-cubes, and single-plane methods are rejected because they round the corner off, which moves transom separation and corrupts the trim moment.

## Scope and non-goals

In scope: mesh validation and patch segmentation, triangle binning, inside/outside classification, per-control-volume plane fits, the convex clipper, the face-consistency pass, pressure control volumes and the three staggered velocity families (U, V, W), VTK export and diagnostics, and a threaded CPU implementation. The hull moves (heave, trim, drop tests), so geometry must be rebuildable every time step from a rigid-body pose.

Out of scope, but the API must not block them later:

- Small-cell stabilization (cell merging, state redistribution).
- Split cells for features thinner than a cell. These are detected and flagged, not handled.
- Exact clipping against the triangle soup, and high-order implicit quadrature.
- GPU kernels. Design data layouts for them, but do not write them yet.
- Free-surface cutting. This module handles the body only.

## Inputs, outputs and data structures

Convention: embedded-boundary normals point out of the solid, into the fluid. Apertures and volume fractions are fluid fractions in \[0, 1\].

| Item | Type / shape | Notes |
| --- | --- | --- |
| Mesh vertices | `3×N Float64` | Body frame; watertight, outward-oriented |
| Mesh triangles | `3×M Int32` | Validated at load: manifold edges, consistent orientation |
| Grid | face-coordinate vectors `xf, yf, zf` | Rectilinear; uniform is a special case |
| Pose | rotation + translation | Applied to vertices on every rebuild |
| Feature angle θ\_f | `Float64`, default 30° | Dihedral threshold for patch segmentation |
| Tolerances | struct | Relative to local spacing h; see the clipper section |
| `vfrac` per family (P, U, V, W) | dense array | Fluid volume fraction |
| `ax, ay, az` per family | dense arrays on that family's faces | Fluid aperture of each face |
| `status` per family | dense `UInt8` array | fluid / solid / cut / feature / unsupported |
| Cut list per family | `StructArray` | CV index, fluid centroid, closure residual, correction magnitude |
| Boundary-face list | `StructArray` | CV index, patch id, area, unit normal, centroid |

Key internal types:

- `Patch`: triangle ids, area, planarity (max distance from best-fit plane), optional global plane, and a convexity label for each neighbouring patch across a shared feature edge.
- `ConvexPoly`: a fixed-capacity, isbits polyhedron with at most 32 vertices and 16 faces. Faces are vertex-index loops, each tagged either with a grid-face id (±x, ±y, ±z) or with an embedded patch id. The tags drive aperture and boundary-face accumulation.

## Pipeline

Steps 1–2 run once per geometry. Steps 3–9 run on every rebuild.

1. **Preprocess the mesh.** Validate that it is watertight and consistently oriented. Compute triangle normals, areas and centroids, and build edge adjacency.
2. **Segment into patches.** Mark edges with a dihedral angle above θ\_f as feature edges, then flood-fill (union-find) across non-feature edges to give each triangle a patch id. Label each patch pair sharing a feature edge as convex or concave, and store the pairs with no shared edge as non-adjacent. If a patch is planar within tolerance, give it one global plane; the transom is the main case.
3. **Apply the pose.** Transform the vertices, and the global planes, into the grid frame.
4. **Bin the triangles.** Map each triangle's bounding box, expanded by a margin δ, to a range of pressure cells. Fill CSR lists with a two-pass count, prefix-sum and fill. A velocity CV takes the deduplicated union of the two pressure-cell bins it straddles.
5. **Find the cut CVs.** A CV is cut if any candidate triangle, clipped to the CV box with Sutherland–Hodgman, has nonzero area. Record which patches are present.
6. **Fit the planes.** For each cut CV and each patch present, clip that patch's triangles to the CV. Take the plane normal from the area-weighted sum N = Σ Aᵢ nᵢ and its point from the area-weighted centroid. If the patch area is below ε\_area·h² (a sliver), refit over the CV enlarged by δ; if it is still below, drop that patch from the CV. Planar patches use their global plane.
7. **Choose the Boolean rule.** With one patch, the solid is the CV clipped by that patch's solid half-space. With several patches whose pairs are all labelled convex, the solid is the intersection of their solid half-spaces, and the fluid is its complement in the CV. With pairs all labelled concave, the fluid is the intersection of the fluid half-spaces. Mark mixed labels, or any non-adjacent pair (a thin feature), as `unsupported` and pass them to the fallback (decision D3).
8. **Classify the uncut CVs.** Run a parity sweep along x grid lines through CV centres, counting ray–triangle crossings. Degenerate hits on triangle edges or vertices use a consistent half-open rule, so every crossing is counted exactly once.
9. **Run the face pass, then the cell pass.** See the face-consistency section.

## Convex clipper

`clip!(poly::ConvexPoly, n, d, keep, tag)` keeps the side of the plane n·x = d chosen by `keep` and tags the new cap face with `tag`.

1. Compute the signed distances sᵢ = n·xᵢ − d, and snap |sᵢ| < tol to 0, with tol = 1e-10·h.
2. If every sᵢ is on the kept side, return unchanged. If every sᵢ is on the discarded side, return empty.
3. Insert a vertex, by linear interpolation, on each edge whose endpoints have strictly opposite signs. Trim each face loop to its kept part.
4. Build the cap from the inserted and on-plane vertices, ordered by angle around their centroid in the plane. Drop faces with area below 1e-14·h².

Measures: compute face area vectors with Newell's method, where p\_f is any point on face f:

```latex
V = \frac{1}{3}\sum_f \mathbf{p}_f\cdot\mathbf{A}_f, \qquad \sum_f \mathbf{A}_f = \mathbf{0}
```

Summing the area vectors by tag gives each grid face's solid footprint and each patch's embedded area vector.

Unit tests, all of which must pass before the clipper goes further:

- Axis-aligned cut: the volume fraction is exact.
- Corner cut through three edge midpoints: the volume is 1/48 of the cube (a corner tetrahedron with legs h/2).
- Two convex planes forming a wedge, checked against the analytic volume.
- Planes passing exactly through a vertex, along an edge, and coincident with a face.
- 10⁴ random planes checked against Monte Carlo volume, with Σ A\_f = 0 to 1e-14·h².
- Zero allocations per call, and type stability under `@code_warntype`.

## Face consistency and closure

Every grid face gets exactly one aperture, computed once and used by both neighbouring CVs. Without this, the two sides disagree by O(h²) and closure breaks. The work runs in three passes:

1. **Pass A (CV-parallel).** Fit the planes for each cut CV (pipeline step 6).
2. **Pass B (face-parallel).** For each face touching a cut CV, fit the planes over the union box of its two CVs, using the same rules as step 6. This is symmetric by construction. Clip the face square with those planes under the step-7 Boolean rule to get the aperture. A face between two uncut CVs takes 0 or 1 from classification.
3. **Pass C (CV-parallel).** Clip each cut CV with its own Pass-A planes to get the fluid volume, the fluid centroid, and the per-patch embedded area vectors S\_p. Then enforce closure with the Pass-B apertures:

```latex
\mathbf{A}_b = -\sum_{f \in \text{grid faces}} \mathbf{n}_f\, a_f\, |f|, \qquad \mathbf{A}_{b,p} = \mathbf{S}_p + \frac{|\mathbf{S}_p|}{\sum_q |\mathbf{S}_q|}\Big(\mathbf{A}_b - \sum_q \mathbf{S}_q\Big)
```

Store the correction magnitude |A\_b − Σ S\_q| as a diagnostic; it should be O(h³) or smaller. The volume fraction comes from the CV's own clip, since only face areas enter closure.

Acceptance: the closure residual recomputed from the stored quantities is at most 1e-12·h² in every CV of every family, and the results are bitwise identical for 1, 4 and 16 threads.

## Parallelism and performance

Passes A, B and C are embarrassingly parallel. Run them over compact work lists, not the full grid: build the lists of cut CVs and affected faces with a parallel flag followed by a prefix-sum compaction.

- **Threading.** Split each list into chunks, one task per chunk, each owning its own `ConvexPoly` and triangle scratch buffers. Do not index scratch by `threadid()`, which is unsafe with dynamic scheduling. Benchmark `Polyester.@batch` against `Threads.@threads` and keep whichever wins.
- **Allocation.** Hot loops must not allocate. Check with `@allocated` in tests, keep code type-stable, and use `StaticArrays` for vertices and planes.
- **GPU readiness.** Keep scratch types isbits with fixed capacity, keep outputs as `StructArray`s, and make structs Adapt.jl-compatible. There must be no `Dict` or `Vector{Vector}` on the per-CV path.
- **Rebuild cost.** Segmentation is pose-invariant and runs once. A rebuild is pose transform, binning, and passes A–C. The target is a full rebuild in well under the time of one pressure solve on the production grid; the exact number is set at gate G6.
- **Determinism.** No atomics in reductions. Per-face work is symmetric, so results do not depend on the thread count.

## Validation and visualization

Every test body is meshed finer than the grid, so geometry error comes from the method rather than the mesh.

| Test | Pass criterion |
| --- | --- |
| Box, grid-aligned and rotated | Volume and area exact to roundoff (all patches planar) |
| Sphere | Volume and area errors converge at O(h²) |
| Wedge / prism with a flat transom and deadrise | Volume exact; corner-line distance at roundoff |
| Curved-bottom hull with a planar transom | Corner-line distance converges at O(h²) |
| Totals against the STL | Solid volume and per-patch area match the mesh's own divergence-theorem values |
| Grid-offset sweep (≥ 1000 random sub-cell shifts and small rotations) | No NaNs, no unsupported CVs on non-thin bodies, bounded volume error |
| Closure | ≤ 1e-12·h² in every CV of every family |
| Thread-count invariance | Bitwise identical for 1, 4 and 16 threads |

The corner-line check works as follows. In each CV, extract the edges shared by two different patch faces, join them into a polyline, and report its maximum distance to the mesh's feature edges.

Visualization:

- **WriteVTK exports.** Write embedded faces as polydata, carrying patch id, normal and CV index. Optionally write the fluid pieces as `VTK_POLYHEDRON` cells. Write `vfrac`, `status`, patch count, closure residual and correction magnitude as rectilinear-grid cell data, and the apertures on the face grids.
- **`debug_cv(geom, family, I)`.** A GLMakie single-CV view showing the cube wireframe, the clipped input triangles, the fitted planes, the resulting polyhedron with its faces exploded, and each vertex labelled with its classification against every plane.

## Phases, gates and decisions

Work in six phases. Do not start a phase until the previous gate passes, and stop at each decision point for review.

&#91;embedded content: roadmap · 6 phases, 6 gates, 4 decisions\]

The decision points are: D1, the feature angle θ\_f and whether the transom gets a global-plane override; D2, the sliver refit margin δ; D3, the fallback for unsupported CVs (octree subdivision of the CV, or flag and skip); and D4, which outputs the solver consumes, confirmed before building the velocity families.

## Assumptions and open questions

- **Grid.** The brief assumes a rectilinear grid given by face-coordinate vectors. If the grid is always uniform, simplify the binning and tolerances accordingly.
- **Mesh source.** If the mesh is tessellated in-repo from the NURBS patches (BasicBSpline.jl), patch ids come for free from the NURBS patch boundaries, and dihedral segmentation becomes a fallback for external STL files only.
- **Consumer.** The solver is moving to sharp-interface direct forcing. Confirm which outputs it consumes (apertures, volume fractions, boundary-face centroids and normals, fluid centroids) before phase P5, because this sets what the velocity families must provide.
- **Defaults to tune.** θ\_f = 30°, δ = 0.25h, ε\_area = 1e-6, tol = 1e-10·h.
