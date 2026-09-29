# Brief: 2D cut cells by exact boundary walking

Sep 28, 2026 · @Mark

## Context and goal

Build a Julia module, `CutCells2D`, that computes exact cut-cell geometry for 2D Cartesian grids from closed polylines. For each cut cell it produces fluid areas, edge apertures, and boundary segments, with every separate fluid region in a cell reported on its own.

The method clips the polylines to each cell box and walks the cell boundary between entry and exit points to close each fluid loop. The result is exact for the polygonal geometry, including sharp corners and trailing edges thinner than a cell. Neighbouring cells agree on shared edges by construction, so closure needs no correction pass.

Marching squares and per-cell line fits are rejected for body geometry. Marching squares cuts off corners and can lose a thin trailing edge entirely, which moves the Kutta point. Line fits recover corners but still need a face-consistency pass and cannot represent split cells.

## Scope and non-goals

In scope: one or more closed polylines (for example, multi-element airfoils), cell-centred grids and optional staggered families (U, V), uniform, rectilinear, and AMR leaf cells, rebuild from a rigid-body pose each step, split cells with several fluid regions, VTK export and diagnostics, and a threaded CPU implementation.

Out of scope, but the API must not block them later:

- Small-cell stabilization (merging, state redistribution). This module only reports regions and their sizes.
- Curved boundary segments. Curves are represented by a polyline finer than the grid; exact spline intersection may come later.
- Level-set or free-surface interfaces, which stay with marching squares.
- GPU kernels. Design the data layouts for them, but do not write them yet.
- 3D, which is covered by the separate 3D brief.

## Inputs, outputs and data structures

Conventions: each body polyline is closed and oriented counter-clockwise, so the solid lies on its left and the fluid on its right. Boundary normals point into the fluid. Areas and apertures are fluid fractions in \[0, 1\].

| Item | Type / shape | Notes |
| --- | --- | --- |
| Polylines | `Vector` of `2×N Float64` | Closed, non-self-intersecting, and non-overlapping with each other; validated at load |
| Grid | face-coordinate vectors `xf, yf`, or a list of AMR leaf boxes | Every cell is an axis-aligned box |
| Pose | rotation + translation | Applied to vertices on every rebuild |
| `afrac` per family | dense array | Total fluid area fraction of the cell |
| `ax, ay` per family | dense arrays on edges | Fluid aperture of each edge |
| `status` per family | `UInt8` | fluid / solid / cut / split / invalid |
| Region list | `StructArray` | Cell index, region id, area, centroid, and the edge-aperture pieces that region owns |
| Boundary-segment list | `StructArray` | Cell index, region id, polyline id, length, unit normal, midpoint, and whether the segment contains a polyline vertex |

Key internal types:

- `Crossing`: polyline id, segment index, parameter t along the segment, grid-line id, coordinate along the line, and direction (entering or leaving the solid).
- `CellWork`: fixed-capacity, isbits scratch space for one cell, holding at most 64 crossings, 256 loop vertices, and 8 regions. Exceeding capacity raises a flagged error, never a silent truncation.

## Algorithm

Every quantity is built from crossings of polyline segments with grid lines, so it comes out identical in every cell that touches that line.

1. **Apply the pose** to all vertices.
2. **Compute crossings per grid line.** Intersect each segment with every x-line and y-line its bounding box spans. The crossing coordinate depends only on the segment and the line's value. Store the crossings per line in CSR form, sorted along the line, and tag each as entering or leaving the solid.
3. **Classify the grid nodes.** Walk each line's sorted crossings and flip between fluid and solid at each one. Every cell sharing a node then agrees on it.
4. **Bin segments into cells** by bounding box, using a two-pass count, prefix-sum and fill.
5. **Handle uncut cells.** A cell with no crossings on its edges and no segments inside is fluid or solid according to its corner nodes. A cell with segments but no edge crossings contains a whole island polyline. Its fluid area is the cell minus the island's area, and the whole island is its boundary.
6. **Walk each cut cell.** Gather the crossings on its four edges in counter-clockwise order from the lower-left corner. Start from an unused fluid arc of the cell boundary and walk counter-clockwise. At a crossing where the body polyline leaves the cell, turn onto the polyline and follow it backwards, keeping the fluid on the left, to its next crossing. Then continue along the cell edges. When the loop closes, it is one fluid region. Repeat until every fluid arc is used.
7. **Measure each region.** Area and centroid come from the shoelace formula on the loop. The polyline pieces used become boundary segments, with normals on the right of the polyline's forward direction, pointing into the fluid. The cell-edge arcs used are the region's share of each edge aperture. If an island also lies in the cell, subtract it from the region that contains one of its vertices.
8. **Compute edge apertures** once per edge from that line's sorted crossings and the node classification. Assert that they equal the sum of the arcs used by the walks.

&#91;embedded content: the boundary walk · one cell, two fluid regions\]

Only cells that a thin section spans completely become split cells. The cell containing a trailing-edge tip is a single region wrapping around an exact solid spike.

## Robustness

All degenerate cases are resolved by one symbolic perturbation. Treat every x-line as if shifted by +ε and every y-line by +ε², for an infinitesimal ε > 0. No polyline vertex then ever lies exactly on a grid line, and no segment lies along one. Implement this as a tie-break rule applied to exact comparisons, never as a numerical offset.

| Case | Resolution |
| --- | --- |
| Vertex exactly on a grid line | Treated as lying on the far side of the shifted line, so the two adjacent segments either both cross it or neither does |
| Segment lying along a grid line | Lies entirely on one side of the shifted line, so it produces no crossings on that line |
| Crossing exactly at a grid node | Belongs to one of the two lines' edges by the same rule, so node classification stays consistent |
| Two crossings at the same coordinate on an edge | Ordered by the perturbation, then by segment index |
| Loop with zero area (a crossing pair that touches the edge and returns) | Kept in the walk and dropped from the region list if its area is below 1e-14·h² |

Further requirements:

- **Sign tests.** Crossing direction and side-of-line tests use a robust orientation predicate. Start with plain `Float64`, and add an adaptive-precision fallback only if the offset sweep in the validation section finds failures (decision D2).
- **Load-time validation.** Reject a polyline that is open, self-intersecting, or overlapping another, and reorient it to counter-clockwise if needed.
- **Walk safety.** Assert that each walk visits every crossing on the cell's edges exactly once. On failure, mark the cell `invalid` in the status array and continue; never loop forever.

## Consistency, closure and AMR

No face-consistency pass is needed. Two cells sharing an edge read the same crossings from the same line list and the same node classifications, so their apertures are bitwise identical. Closure then holds for each region r:

```latex
\sum_{e \in r} \mathbf{n}_e\, \ell_{e,r} + \sum_{s \in r} \mathbf{n}_s\, L_s = \mathbf{0}
```

Here ℓ\_{e,r} is the length of the fluid arc of edge e used by region r, and L\_s is the length of boundary segment s. Report the residual for every region; it should be at roundoff, at most 1e-13·h.

For split cells, each region is its own control volume. The flux through an edge is divided between regions by their arcs, so the solver never averages across a thin body.

For AMR, every leaf is an axis-aligned box, and crossings are stored per line, keyed by axis and coordinate. Compute crossings only on lines that are edges of at least one leaf. Each leaf edge reads the crossings within its own extent, and a line that crosses a coarse cell's interior is ignored for that cell. Because a crossing depends only on the segment and the line's value, the aperture of a coarse edge equals the sum of its two fine neighbours' apertures exactly. Hanging nodes are classified by the same line walk. Test this bitwise at every 2:1 interface.

## Parallelism and performance

Every stage is parallel, and all shared state is written by exactly one task.

- **Crossings.** Parallel over segments with a two-pass count, prefix-sum and fill into the per-line CSR, then a parallel sort within each line.
- **Node classification.** Parallel over lines; each line walk is sequential and cheap.
- **Cell walks and edge apertures.** Parallel over compact lists of cut cells and cut edges, built with a flag followed by prefix-sum compaction. Each task owns its own `CellWork` scratch, split into chunks rather than indexed by `threadid()`.
- **Allocation.** No allocation in hot loops, checked with `@allocated`. Keep the code type-stable and use `StaticArrays` for points.
- **GPU readiness.** Keep types isbits and outputs as `StructArray`s, and make structs Adapt.jl-compatible. The per-line CSR layout maps directly onto a GPU implementation later.
- **Determinism.** Results must be bitwise identical for any thread count. Region ids are assigned in walk order starting from the lower-left corner, so they are deterministic too.

Rebuild cost is dominated by the crossings stage and scales with the number of segments plus the number of cut cells. For a rigid-body rebuild every time step, set the target at gate G5 against the solver's time-step cost.

## Validation and visualization

The method is exact for polygons, so most tests pass at roundoff rather than at a convergence rate.

| Test | Pass criterion |
| --- | --- |
| Any polygon (square, rotated square, 64-gon) | Total fluid area equals domain minus polygon area to 1e-13 relative |
| Axis-aligned square with vertices on grid nodes and edges along grid lines | Exact areas, no `invalid` cells |
| Thin plate spanning many cells | Every spanned cell has exactly two regions, and their areas are exact |
| Airfoil with a sharp trailing edge | Split cells upstream of the tip, a single region at the tip, and exact areas throughout |
| Island smaller than a cell, and two bodies in one cell | Correct subtraction and region count |
| Offset sweep (≥ 10⁴ random sub-cell shifts and rotations) | No `invalid` cells, and area error at roundoff |
| Closure per region | Residual at most 1e-13·h |
| AMR 2:1 interfaces | Coarse aperture equals the sum of the fine apertures, bitwise |
| Thread-count invariance | Bitwise identical for 1, 4 and 16 threads |

As a diagnostic rather than a gate, run marching squares on the SDF of the same airfoil and plot area and aperture differences near the trailing edge. This documents what the exact method buys.

Visualization:

- **WriteVTK exports.** Write boundary segments as polydata, carrying region id and normal. Write region loops as polygons, carrying area, region id and a split flag. Write `afrac`, `status` and the closure residual as grid cell data, and the apertures on the edges.
- **`debug_cell(geom, I)`.** A Makie single-cell view showing the cell box, the polyline, the crossings numbered in walk order and marked as entering or leaving, and each region's loop in its own colour with direction arrows.

## Phases, gates and decisions

Work in five phases. Do not start a phase until the previous gate passes, and stop at each decision point for review.

&#91;embedded content: roadmap · 5 phases, 5 gates, 4 decisions\]

The decision points are: D1, whether bodies arrive as polylines or through SDF extraction, decided before P1; D2, whether the offset sweep calls for adaptive-precision predicates; D3, the AMR package interface; and D4, whether staggered families are needed at all.

## Assumptions and open questions

- **Geometry source.** This method needs polylines, but the aircraft solver's geometry lives in an SDF library. Either define 2D bodies directly as polylines (for example, airfoil coordinates), or add an SDF-to-polyline extraction step at a resolution finer than the grid that preserves sharp features. This is decision D1.
- **Polyline resolution.** A curved surface's chord error is about κ·Δs²/8. The proposed default is a segment length Δs ≤ h/4 near the body, to be confirmed against the airfoil case.
- **First consumer.** The brief assumes the first user is the cell-centred finite-volume airfoil case on AMR. Staggered U and V families are only needed for a 2D MAC testbed of the planing solver (decision D4).
- **AMR interface.** The brief assumes the AMR package can provide a list of leaf boxes and a neighbour lookup across faces, including 2:1 faces. Confirm the API before phase P4 (decision D3).
