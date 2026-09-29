# Brief: 2D exact cut cells on AMR meshes

Sep 29, 2026 · @Mark

## Context and goal

`PolylineClippingCutCell` computes exact cut-cell geometry on a uniform `CartesianGrid{2}` from closed polylines. It clips the polylines to each cell and walks the cell boundary to close each fluid region. The uniform build is complete and is the reference for this brief: `src/polyline_clipping/` and its tests. Its rules carry over unchanged unless this brief says otherwise.

This brief extends the method to CartesianMeshes' `AdaptiveMesh{2}`, the quadtree that IBM.jl's cell-centred finite-volume solver runs on.

The reason is IBM's cut 2:1 faces. IBM uses the marching-squares tree cache, where each leaf builds its apertures from its own four corner values. At a 2:1 face, the coarse leaf interpolates between 2 corners and its two fine neighbours use 3, so the apertures disagree by O(h²). IBM therefore refuses any cut mortar: `refine_band!` puts every cut cell in a uniform band, and the `Semidiscretization` throws if `count_cut_mortars > 0`.

The goal is apertures that agree exactly across every face, including 2:1 faces. The crossings on a grid line depend only on the segment and the line's value, so a coarse edge and its two fine halves read the same crossings. Their open lengths then agree by construction, the same way shared edges agree on the uniform grid. IBM can then allow cut cells at a level change, and it gets sharp trailing edges and split cells exactly.

## Scope and non-goals

In scope:
- `allocate_cache(mesh::AdaptiveMesh{2}, PolylineClippingCutCell(); backend)` and `update_cache!(cache, body::Mesh{2}, mesh::AdaptiveMesh{2})`.
- Every output of the uniform cache, per leaf: cell records, regions of split cells, arcs, boundary segments, islands, snapped slivers, status and flags.
- Per-face apertures that are single-valued across conforming faces and across the two sub-faces of each 2:1 face.
- Readers, VTK export and `generate_mesh`/`interface_mesh` on the tree.
- The same execution model as the uniform cache: combinatorics on the host in `Float64` with exact predicates, arithmetic in KernelAbstractions kernels. The same determinism, allocation and GPU gates apply.
- A moving body under a fixed mesh. The topology is cached as it is today; IBM's body is stationary at present, but the uniform cache supports motion and this should not regress.

Out of scope:
- The IBM.jl changes: swapping the method, allowing cut mortars, and the mortar flux. They are listed under follow-on work because they set what this cache must provide.
- Rectilinear or locally stretched roots. `AdaptiveMesh` has a uniform base grid; anisotropic `d` is supported, as on the uniform cache.
- `FullConnect` balance. The method only needs 2:1 across faces (`FaceConnect`, the default); corner neighbours are never consulted.
- Remapping a cache across a refine or coarsen. A new mesh generation means allocating a new cache (decision D2).
- As before: small-cell stabilization, curved segments, the Makie `debug_cell` view, and the marching-squares comparison.

## Terms

- **Leaf**: an `AdaptiveMesh` cell, addressed by its index `i` in Morton order. Leaf indices change on every refine or coarsen.
- **Fine lattice**: the uniform lattice at `L*`, the deepest level present in the mesh, with spacing `d/2^L*`. Every leaf corner is a node of it. `L*` is used rather than `max_level`, whose default can make the lattice impractically large.
- **Line**: a fine-lattice line `(axis, k)`. A coarse leaf's edge lies on a fine line.
- **Face**: the unit that fluxes and apertures live on. A conforming `Interface` is one face. A `Mortar` (one coarse edge against two fine edges) is two faces, one per fine leaf. A `BoundaryFace` is one face. Each face is an interval of fine-lattice nodes on one line, with a leaf on each side.
- **Hanging node**: the midpoint of a coarse edge at a mortar. It is a fine-lattice node and is classified like any other node.

```
   fine leaf B  |
   -------------+ hanging node      coarse leaf C's right edge = face 1 + face 2
   fine leaf A  |                    each crossing belongs to exactly one face
                | coarse leaf C      A and C share face 1; B and C share face 2
```

## Inputs, outputs and data structures

Conventions are unchanged. Each body loop is closed and counter-clockwise, so the solid is on its left. Boundary normals point into the fluid. Areas and apertures are fluid fractions in [0, 1].

| Item | Type / shape | Notes |
| --- | --- | --- |
| Body | MeshLibrary `Mesh{2}` of `Line` | As today; validated the same way |
| Mesh | `AdaptiveMesh{2,T}` | Face-balanced 2:1; any base `n`; periodic allowed, since periodic faces are never cut (see body clearance) |
| `cells` | `StructArray{CutCellData{2,T,4,1}}`, length `nleaves` | Leaf order, the same layout as the marching-squares tree cache, so every reader works and IBM can swap methods |
| `info` | `StructArray{PolylineCellInfo{T}}`, length `nleaves` | Status, region count, flags, slots, residual |
| `faces` | `StructArray{PolylineFace{T}}`, one per face | Open length, open-part centroid offset from the face's lower node, both leaves. Order: interfaces, then two sub-faces per mortar, then boundary faces, as in `FaceList` |
| `leaf_faces` | per leaf and direction, one or two face ids | Two exactly when the leaf is the coarse side of a mortar |
| `regions`, `rinfo`, `arcs`, `bsegs` | as today | Keyed by leaf index. An arc names its **face** in place of `(dir, k)`, so every arc has exactly one neighbour leaf |
| Generation stamp | `(generation(mesh), nleaves)` | Checked on every update |

## Algorithm

Steps 1–2 run at allocation, once per mesh generation. Steps 3–9 run on every update. Steps 3–7 run on the host.

1. **Lattice and faces.**
   - Find `L*`.
   - The fine-lattice line values are `Float64(x0 + T(k) * (d / 2^L*))`: the same expression `cell_nodes(mesh, c)` uses, so they are bitwise equal to every leaf's corners at every level.
   - Build the face table from `FaceList(mesh)`. Each face gets its line and its node interval `[j0, j1)`, from the leaves' integer coordinates shifted to `L*`. No floating point is involved.
   - Build `leaf_faces`.
2. **Seed** every leaf as no body, as the uniform cache does.
3. **Block.**
   - Bin the body's vertices to fine cells with the exact `_locate`.
   - Map each fine cell to its leaf with `locate(mesh, TreeCell(L*, coord))`. That is an integer binary search; never use `find_leaf(point)`, which floors in floating point.
   - The block is the leaves the body can touch plus one leaf of padding, found by face neighbours. Throw if a block leaf has a boundary face; this is the tree form of today's one-clear-cell rule.
4. **Lines and crossings.**
   - The block's lines are every line that carries a face of a block leaf.
   - Compute each segment's crossings with these lines over the **whole** block extent, including stretches that cross the interior of a coarse leaf. Those interior crossings are only used to classify nodes; no leaf's walk reads them.
   - Store the crossings in a per-line CSR, sorted by (fine interval, then the exact comparator).
   - A crossing's **offset** is measured from the lower node of the face that owns it, with the uniform formula. Both leaves sharing the face read the same number.
5. **Nodes.** Walk each block line's crossings from the block boundary, which is fluid, and classify every leaf corner and hanging node on it. Classify each node from both its x-line and its y-line; a disagreement sets `STATUS_CONFLICT`, as today.
6. **Cut list.** A leaf is cut if one of its faces carries a crossing or it holds a vertex. Otherwise it is fluid or solid from its corners. Then assign the output slots and find the island targets, as today.
7. **Walk kernel**, one workitem per cut leaf.
   - A leaf's perimeter is its four edges, and each edge is one or two faces. Because the line CSR is sorted by fine interval, the crossings along a whole coarse edge are one contiguous slice.
   - The walk is today's walk with one change: an arc that runs across a hanging node is split there into one arc per face. The hanging node must then be fluid, and a disagreement fails the walk.
   - The walk works in leaf-local coordinates. It adds each face's lower-node offset to the crossing offsets; this rounding only affects areas, never apertures.
   - Tolerances use the leaf's own size: `drop_area·h²` and `closure_tol·h`.
8. **Face kernel**, one workitem per face beside a cut leaf.
   - Match each arc to its neighbour's region (`nbr_region`), close any sliver's arcs on both sides, and write the face's open length and centroid, summed in a fixed order.
   - A face with no crossings is fully open or fully closed, according to its end nodes.
9. **Finalize kernel** over the cut leaves and their face neighbours.
   - Write `cells` and `info` from the face table, the walk-free totals and the walls. Write the region records of split leaves.
   - A leaf's face fraction in direction `dir` is the sum of its one or two faces' open lengths, lower face first, divided by `full_face_area(get_elem_size(mesh, leaf), dir)`. That is the size IBM's `face_area_of` multiplies back by (decision D3).

## Consistency at 2:1 faces

What is exact, by construction:
- Each face's open length and centroid are computed once and copied by both leaves.
- A coarse leaf's open length on an edge is `fl(ℓ₁ + ℓ₂)` of its two stored faces, in a fixed order.
- A coarse leaf's arcs equal its fine neighbours' arcs bitwise, in face-local offsets, and `nbr_region` is symmetric across every face.

What is exact only to roundoff:
- `CutCellData` stores fractions. Turning a fraction back into an area, `face_area_of(m, dir, h)`, costs up to an ulp or two, so `f_c·A_c` equals `f₁·A₁ + f₂·A₂` only to a few ulps.
- IBM's mortar flux already uses the fine side's apertures, and the coarse cell's wall area comes from its own fractions. The free-stream mismatch in a coarse cut cell is then at roundoff, not the O(h²) marching squares gives.
- A consumer that needs the sub-face lengths bitwise reads them from `faces`, through a reader added for the purpose (decision D3).

Per-region closure, `Σ_e n_e ℓ_{e,r} − Σ_s n_s L_s = 0`, is reported per leaf and per region as today. The limit is `closure_tol·h` with the leaf's own `h`.

## Robustness

The uniform rules apply unchanged, with the fine lattice in place of the grid:
- One symbolic perturbation, with every x-line shifted by +ε and every y-line by +ε², evaluated exactly with `ExactPredicates.orient`.
- Side, ownership and ordering are all decided against fine-lattice nodes. A crossing's owning face follows from its fine interval.
- Validation: a clockwise loop, nesting, an open chain, self-intersection or touching loops throws. The mesh is never repaired.
- Only 64 crossings fit a leaf's perimeter. Over that, the leaf is `INVALID` with `OVERFLOW`, keeps its walk-free totals, and the update warns. A coarse leaf at a cut mortar carries more crossings than a fine one, so report the maximum in the stats.

Specific to the tree:
- **Never make a decision from `get_elem_bounds(...).hi`, `cell_grid`, `find_leaf(point)` or the tree `write_vtk` corners.** These form `lo + d` or floor in floating point, and are not bitwise on the lattice. Use `cell_nodes` or the fine-lattice values.
- **Stale meshes.** `update_cache!` throws if the mesh's generation or leaf count differs from the stamp, or if the mesh and the cache are on different backends.
- **Corner neighbours.** Under `FaceConnect`, leaves that touch only at a corner may differ by two levels. Nothing in the method reads a corner neighbour: nodes come from line walks and regions are matched across faces. Keep it that way.

## Parallelism and performance

This follows the uniform cache. The host does the combinatorics serially and in a fixed order, and the kernels do the rest, so results are bitwise identical for any thread count. On a GPU, the host outputs are uploaded every update. Allocating a new cache for each mesh generation includes building the face table on the host.

IBM re-adapts every few steps (`AMRCallback(; interval=5)`) and rebuilds its geometry each time. So the cost that matters is allocate plus one update, against IBM's cost between remeshes. Set the target at gate G4.

## Validation

| Test | Pass criterion |
| --- | --- |
| Uniform equivalence: an `AdaptiveMesh` refined uniformly to level L against the uniform cache on the matching `CartesianGrid` | Every per-cell output, face aperture, region, arc and segment bitwise equal |
| Lattice | Fine-lattice values equal `cell_nodes` at every leaf and level, bitwise; the faces tile every leaf edge exactly |
| Crossings and nodes | Crossings equal brute force; every corner and hanging-node state equals exact point-in-polygon; x and y walks agree |
| Polygons (square, rotated square, 64-gon) on random refinement patterns | Total fluid area equals domain minus polygons to 1e-13 relative |
| Cut mortars, deliberately placed through the body | Coarse arcs equal fine arcs in face-local offsets, bitwise; `nbr_region` symmetric; `|f_c·A_c − Σ f_i·A_i| ≤ 4 ulp·A_c` |
| Sharp trailing edge crossing a mortar, and a thin plate spanning coarse and fine leaves | Split leaves on both sides with exact areas, and a single region at the tip |
| Islands on coarse and fine leaves, and an island straddling a mortar | Correct subtraction and region count |
| Offset and rotation sweep (≥ 10⁴ cases on random AMR meshes) | No `INVALID` leaves; area error at roundoff |
| Closure per region | ≤ 1e-13·h with the leaf's own h |
| Stale generation, wrong backend, body at the boundary | Each throws |
| Determinism, GPU, allocation | 1/4/16 threads bitwise; GPU matches the host as on the uniform cache; bounded allocation on a second update of the same generation; `@inferred` |

The uniform-equivalence test is the main guard. It keeps the tree path and the uniform path in step while they share code.

Visualization: `write_cache_vtk(prefix, cache, mesh)` writes leaf data through `CartesianMeshes.write_vtk(fname, mesh; cell_data)`, with status, region count, flags, residual and level. Walls, region loops and faces are written as today. The tree VTK corners are not lattice-exact; that is fine for viewing only.

## Phases, gates and decisions

Work in four phases. Do not start a phase until the previous gate passes, and stop at each decision point for review.

| Phase | Content | Gate |
| --- | --- | --- |
| P1 | Settle the D1 code structure; fine lattice, face table, `leaf_faces`, generation stamp; `allocate_cache` on the tree, seeded as no body | The lattice test; a fresh cache reads as no body; the stale-mesh checks |
| P2 | Block, per-line crossings, node and hanging-node classification, vertex binning, cut list, uncut leaves and faces | Crossings and nodes against brute force; uniform equivalence of every status and uncut aperture |
| P3 | Walk with face-split arcs, face kernel, finalize, islands, snap | The rest of the validation table except performance. **Review**, which settles D3 |
| P4 | Readers by leaf index, VTK, `generate_mesh`/`interface_mesh`, threads, GPU, allocation, a benchmark on IBM's NACA 0012 case (G4) | Determinism and GPU rows; timing against IBM's remesh interval |

Decisions:
- **D1, code structure (before P1).** Recommendation: a separate `PolylineClippingTreeCache` that shares the predicates, topology and walk core with the uniform cache. The walk core would read the perimeter as a list of faces rather than four dense edges. The uniform cache keeps its dense per-block edges, which are faster on a uniform grid. The alternative, running the uniform grid as a level-0 tree, gives one code path but loses the dense layout.
- **D2, a new mesh generation.** Recommendation: throw and require a new cache, as the marching-squares tree cache does. IBM's `reset_cache!` already allocates anew after every adapt. The alternative, resizing in place, saves an allocation per remesh but needs the face table rebuilt inside `update_cache!`.
- **D3, what consumers read at mortars (at the P3 review).** Recommendation: both the leaf fractions in `cells`, for every existing reader, and a sub-face reader such as `face_apertures(cache, i, dir)` that returns each sub-face's open length and centroid bitwise. IBM's mortar flux could then use the coarse side's own view of each sub-face.
- **D4, split cells in IBM.** IBM avoids multi-valued cells by refining them (its `AmbiguityIndicator`, flagged through `ambiguous`). The method reports them either way. Whether IBM keeps refining them or treats each region as its own control volume is IBM's choice, and does not change this cache.

## Follow-on work in IBM.jl (not this brief)

- Set `CUTCELL_METHOD` to `PolylineClippingCutCell()`, passing the body as a `Mesh{2}` of lines: airfoil coordinates, or an extraction from the SDF.
- Drop the `count_cut_mortars` throw, and make `refine_band!` optional.
- Check free-stream preservation in coarse cut cells at mortars. The mismatch should be at roundoff.
- Decide D4.

## Assumptions and open questions

- **Depth.** The fine lattice has `n·2^L*` lines per axis. That is cheap as 1D vectors, but it means `L*` must be the deepest level present, not `max_level`.
- **Body source.** IBM's bodies are SDFs today. The method needs polylines finer than the finest leaf near the body, with segment length Δs ≤ h/4 as the uniform brief proposed.
- **Mesh on device.** `AdaptiveMesh` adapts to an isbits struct, and `leaf`, `locate` and `face_neighbors` are kernel-safe. The face table and `leaf_faces` are built on the host and uploaded, like the other host outputs.
