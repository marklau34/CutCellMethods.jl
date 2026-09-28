# CutCellMethods.jl

Cut-cell geometry on Cartesian grids: per-cell volume fractions, face apertures, centroids and
interface facets, plus the reconstructed surface. The bodies being cut are
[SDFLibrary.jl](https://github.com/tidalflight/SDFLibrary.jl) geometries, or any callable
`x -> phi` that is negative inside.

## The API is two calls

```julia
cut_cell_moments(method, domain, geo, cell)   # one cell's geometry
generate_mesh(geo, domain, method)            # the reconstructed surface
```

Nothing is allocated or kept between calls. To get a field over the domain, allocate the arrays you
want and fill them yourself, keeping only the fields your scheme uses.

`method` is one of three stateless tags, all `<: AbstractCutCellMethod`:

| tag | dimension | reconstruction | `cut_cell_moments` returns |
|---|---|---|---|
| `MarchingSquaresCutCell()` | 2D | nodal, from the cell's 4 corner values | `CutCellData{2,T,4,1}` |
| `MarchingCubesCutCell()` | 3D | nodal, from the cell's 8 corner values | `CutCellData{3,T,6,2}` |
| `PLICCutCell()` | 2D and 3D | one plane per cell, fitted at its centroid | `PLICCutCellData{D,T,2D}` |

`domain` is a `CartesianGrid`, or an `AdaptiveMesh` for the nodal methods. For the nodal methods,
`geo` is an `AbstractSDFGeometry`, a callable `x -> phi`, or, on a grid, an array of nodal values
already sampled on it with `SDFLibrary.sample_sdf(geo, grid)`. PLIC needs an `AbstractSDFGeometry`,
since it reads the normal from `get_sdf`. `cell` is a `CartesianIndex{D}` on a grid, or a
`TreeCell` or leaf index on a mesh.

```julia
using SDFLibrary, CutCellMethods, CartesianMeshes, StaticArrays

geo  = SDFCircle(center=SVector(0.0, 0.0), radius=0.5)
MS   = MarchingSquaresCutCell()
grid = CartesianIsoGrid(mincorner=(-1, -1), maxcorner=(1, 1), sz=0.01)

vol = Array{Float64}(undef, Tuple(grid.n))
ap  = Array{SVector{4,Float64}}(undef, Tuple(grid.n))
for ci in CartesianIndices(Tuple(grid.n))
    m = cut_cell_moments(MS, grid, geo, ci)
    vol[ci] = m.volume_fraction
    ap[ci]  = m.face_fraction
end

surface = generate_mesh(geo, grid, MS)        # Mesh{2} of Line elements
```

`cut_cell_moments` is `@inline`, isbits in and out, and allocation-free, so it is safe to call from
your own `@batch` loop or KernelAbstractions kernel.

## Caches: a whole grid, kept between updates

When a consumer wants every cell and refreshes them as the body moves, allocate a cache once and
update it in place:

```julia
cache = allocate_cache(grid, MarchingSquaresCutCell(); backend=CPU())   # or CUDABackend()
update_cache!(cache, geo, grid)          # grid may move between updates; its `n` may not

cache.cells.volume_fraction              # one field, as a plain array
cache.cells[ci]                          # one cell's whole CutCellData
```

Every cache holds its method's per-cell struct for every cell as a `StructArray` in `cells`, and the
stored cells are bitwise what `cut_cell_moments(method, grid, geo, ci)` returns. A new cache is
seeded as no body: every cell outside, all fluid, every face open. `update_cache!` synchronizes
before it returns. Beyond `cells`:

| method | extra field | holds |
|---|---|---|
| `MarchingSquaresCutCell`, `MarchingCubesCutCell` | `phi` | the field at every node; `generate_mesh(cache.phi, grid, method)` draws the same reconstruction |
| `PLICCutCell` | `face_fraction` | each cell's **resolved** face fractions, `SVector{2D}` per cell; read these, not the one-sided `cells.face_area` |

The full struct costs about 200 bytes a cell in 3D at `Float64`. When only a few fields are wanted
over a large grid, calling `cut_cell_moments` from your own kernel and keeping just those fields is
lighter.

## Exports are deliberately narrow

Only the method tags, `cut_cell_moments`, the two return types and `generate_mesh` (re-exported
from MeshLibrary.jl) are exported. Everything else is qualified:

| call | meaning |
|---|---|
| `CutCellMethods.face_area_of(cell, dir, cellsize)` | open measure, raw units (length in 2D, area in 3D) |
| `CutCellMethods.face_fraction_of(cell, dir, cellsize)` | open fraction, dimensionless `[0,1]` |
| `CutCellMethods.is_face_open(cell, dir)`, `is_cell_open(cell)` | connectivity, **not** `kind` |
| `CutCellMethods.is_cut`, `is_inside`, `is_outside`, `is_ambiguous` | classification |
| `CutCellMethods.volume_rule`, `face_rule(cell, dir, domain, idx)`, `interface_rule` | one-point quadrature rules |
| `CutCellMethods.face_centroid(cell, dir, domain, idx)` | an open face's centroid in global coordinates, from the stored in-plane offsets |
| `CutCellMethods.interface_area(cell, cellsize)`, `interface_normal`, `interface_normal_area` | the interface facet, formed from the face fractions |
| `CutCellMethods.closure_residual(m, cellsize)`, `closure_residuals`, `cut_cell_report` | verification |
| `CutCellMethods.calc_volume(geo, grid)`, `calc_volume_simple` | the enclosed volume, from the PLIC fit |
| `CutCellMethods.cell_indices(mesh, grid)`, `vertex_normals(geo, grid, MC)` | what a marched mesh cannot hold |
| `CutCellMethods.CELL_INSIDE`, `CELL_OUTSIDE`, `CELL_CUT` | the `kind` encoding |

The `face_*` readers work on either return type, so one call site serves both reconstructions.
`cellsize` is the cell's own extent: `grid.d`, or `get_elem_size(mesh, cell)` on a tree.

## The traps

These go wrong silently.

- **PLIC apertures are one-sided.** `PLICCutCellData.face_area` is this cell's own plane evaluated
  on its own faces, so two cells sharing a face report different numbers wherever their fits
  disagree. Resolve an interior face as the mean of its two slots,
  `(a[ci].face_area[2c] + a[ci + e_c].face_area[2c - 1]) / 2`, which is bitwise single-valued
  because `+` is commutative. The nodal methods need no resolution: both cells interpolate the same
  corner values, so a shared face agrees bitwise by construction.
- **`volume_fraction` is the fluid fraction** — the part of the cell outside the body. The solid
  fraction `calc_volume` integrates is its complement. `kind` follows the fluid reading:
  `CELL_INSIDE` at `0`, `CELL_OUTSIDE` at `1`, `CELL_CUT` strictly between.
- **Two sampling routes agree only to roundoff.** Per-corner sampling (pass `geo`) and a nodal
  block (pass `SDFLibrary.sample_sdf(geo, grid)`) differ by about an ulp per corner. Each is
  watertight within itself; use one route for both the moments and the surface.
- **Bitwise claims hold within one route and one backend.** `closure_residual` is exactly zero and
  shared faces agree bitwise on one backend. Across CPU and GPU, compare integers exactly and floats
  with a tight `isapprox`.
- **PLIC requires an isotropic grid, and the per-cell form does not check.**
  `generate_mesh(geo, grid, PLICCutCell())` throws on an anisotropic grid; `cut_cell_moments` with
  `PLICCutCell()` assumes it, so it stays callable from a kernel.
- **Faces are indexed by direction:** `1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`, `5 = -z`, `6 = +z`.
  `CartesianMeshes.direction_axis` and `direction_sign` decode them.
- **The two normal senses are opposite, on purpose.** Open-face normals point out of the fluid
  region; `interface_normal` points out of the body. The divergence theorem over the fluid region
  is `sum_k A_k n_k - A_int n_int == 0`, which is what `closure_residual` returns. A flux balance
  negates the interface normal at the point of use.
- **`vertex_normals` marches a second time**, and the march reserves about 0.5 GB at `N = 200`.
  For a mesh and normals every step, keep your own `MarchingCubes.MC` and call `MarchingCubes.march`.
- **`ambiguous` cells** (a saddle, or an ambiguous cube interior) have a lumped `interface_normal`:
  right for conservation, wrong for a surface flux. Count them and refine them away.

## Which reconstruction to use

- **A cut-cell flux scheme:** a nodal method. Shared faces are single-valued, the closure identity
  is exact per cell, and the volume, apertures and interface come from one construction.
- **A VOF solver's interface:** `PLICCutCell`. It is what `calc_volume` integrates, and its surface
  is a field of disconnected per-cell shards, which is what a VOF solver believes.
- **A picture of the body:** `generate_mesh` with a nodal method. It is watertight and wound
  outwards, so it can be fed back to `SDFMesh` as a geometry.

Nodal reconstructions chamfer sharp features to within one cell. Either resolve the feature or give
the geometry a `smoothing` comparable to the cell size.

## Reference

`dev/dev_cut_cell_api.jl` drives every call above and prints `true` for each invariant. Run it from
an environment with both packages, for example `julia --project=. dev/dev_cut_cell_api.jl`.
