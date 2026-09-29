# =====================================
# The marching-squares cache: every cell's moments over a whole grid, kept between updates
#
# The field is sampled once per grid NODE into `phi`, and the moments kernel reads that nodal block:
# a node is shared by four cells, a quarter of the field queries a per-corner sampling would make.
# Each node is sampled at `get_node(grid, ni)` -- the expression `cell_nodes` pairs the value with --
# so a corner shared by two cells is exactly one stored number, and the watertightness argument of
# `marching_squares.jl` holds for the cache as it does per cell.
#
# The two kernels below are dimension-generic; `marching_cubes/cache.jl` is the same construction one
# dimension up, on these same kernels.

"""
    MarchingSquaresCutCellCache

What [`allocate_cache`](@ref)`(grid, MarchingSquaresCutCell())` builds and
[`update_cache!`](@ref) refreshes:

- `cells` -- every cell's [`CutCellData`](@ref)`{2,T,4}`, a `StructArray` sized `grid.n`.
  `cells.volume_fraction`, `cells.face_fraction`, `cells.kind`, ... are plain arrays of one field;
  `cells[ci]` is cell `ci`'s whole struct. A shared face is single-valued by construction: cell
  `ci`'s slot `2c` and cell `ci + e_c`'s slot `2c-1` hold bitwise the same fraction.
- `phi` -- the field at every grid node, sized `grid.n .+ 1`: the corner values `cells` was built
  from. `generate_mesh(cache, grid)` draws the contour of this very
  reconstruction.

On an `AdaptiveMesh` -- what `allocate_cache(mesh, MarchingSquaresCutCell())` builds -- the same
type holds the tree's layout instead: `phi` and `cells` are linear lists over the leaves, in leaf
order, each leaf's four corner values as an `SVector{4}` in [`MS_NODE_BITS`](@ref) order and its
moments. `generate_mesh(cache, mesh)` marches that layout just as it does the grid's.
"""
struct MarchingSquaresCutCellCache{P<:AbstractArray,C<:AbstractArray{<:CutCellData}} <: AbstractCutCellCache
    phi::P
    cells::C
end

Adapt.@adapt_structure MarchingSquaresCutCellCache
KernelAbstractions.get_backend(cache::MarchingSquaresCutCellCache) = get_backend(cache.phi)
@inline face_fractions(cache::MarchingSquaresCutCellCache) = cache.cells.face_fraction

function allocate_cache(grid::CartesianGrid{2,T}, ::MarchingSquaresCutCell;
                        backend=KernelAbstractions.CPU()) where {T}
    cache = MarchingSquaresCutCellCache(
        KernelAbstractions.allocate(backend, T, Tuple(grid.n) .+ 1),
        _allocate_cells(backend, CutCellData{2,T,4,1}, Tuple(grid.n)))
    return update_cache!(cache, EmptyGeo(one(T)), grid)
end

update_cache!(cache::MarchingSquaresCutCellCache{<:AbstractMatrix}, geo, grid::CartesianGrid{2}) =
    _update_nodal_cache!(cache, geo, grid)

# The shared body of the two nodal caches' `update_cache!`: sample every node, then build every
# cell from those samples. Two launches on one backend run in order, so the second reads a finished
# `phi` without a synchronize between them.
function _update_nodal_cache!(cache, geo, grid::CartesianGrid{D}) where {D}
    _check_cache_size(cache.cells, grid)
    T = eltype(cache.phi)
    g = convert(CartesianGrid{D,T}, grid)
    backend = get_backend(cache)
    _sample_nodes_kernel!(backend, 64)(cache.phi, g, geo; ndrange=size(cache.phi))
    _nodal_cells_kernel!(backend, 64)(cache.cells, cache.phi, g; ndrange=size(cache.cells))
    KernelAbstractions.synchronize(backend)
    return cache
end

# One thread per grid NODE, at `get_node` -- the same expression `cell_nodes` places a corner with.
@kernel function _sample_nodes_kernel!(phi, grid::CartesianGrid, geo)
    ni = @index(Global, Cartesian)
    @inbounds phi[ni] = eltype(phi)(sdf_value(geo, get_node(grid, ni)))
end

# One thread per cell: the primitive on the cell's corners, read out of the shared nodal block and
# stored whole. `cell_nodes`/`cell_values` dispatch on the grid's dimension, so this is marching
# squares on a `CartesianGrid{2}` and marching cubes on a `CartesianGrid{3}`.
@kernel function _nodal_cells_kernel!(cells, @Const(phi), grid::CartesianGrid)
    ci = @index(Global, Cartesian)
    @inbounds cells[ci] = cut_cell_moments(cell_nodes(grid, ci), cell_values(phi, grid, ci), grid.d)
end

# =====================================
# On a tree: the same cache, laid out on the leaves
#
# A tree has no single nodal block -- a hanging node lies on a coarse leaf's edge, not at its
# corners -- so each leaf keeps its own four corner values, `phi[i]`, beside its moments, `cells[i]`,
# both linear over the leaves in leaf order. Corners are sampled at `cell_nodes(mesh, c)`, bitwise
# exact across levels, so two leaves sharing a corner store one number for it. One kernel samples a
# leaf's corners and builds its moments from exactly those, the same per-cell construction the grid
# cache uses, so a leaf and the grid cell it coincides with are one cell bit for bit.

"""
    allocate_cache(mesh::AdaptiveMesh{2}, ::MarchingSquaresCutCell; backend=get_backend(mesh))

A marching-squares cache over `mesh`'s leaves, in `mesh`'s element type, seeded as no body: `phi`
and `cells` one entry per leaf, in leaf order (see [`MarchingSquaresCutCellCache`](@ref)).

`backend` defaults to the tree's own, which is the one the leaf kernel needs it on: a device cache
is updated against the tree adapted to that device (`adapt(CuArray, mesh)`). The cache is sized for
the tree's leaves as they are; refine or coarsen it and allocate again.
"""
function allocate_cache(mesh::AdaptiveMesh{2,T}, ::MarchingSquaresCutCell;
                        backend=KernelAbstractions.get_backend(mesh)) where {T}
    n = nleaves(mesh)
    cache = MarchingSquaresCutCellCache(KernelAbstractions.allocate(backend, SVector{4,T}, n),
                                        _allocate_cells(backend, CutCellData{2,T,4,1}, (n,)))
    return update_cache!(cache, EmptyGeo(one(T)), mesh)
end

"""
    update_cache!(cache::MarchingSquaresCutCellCache, geo, mesh::AdaptiveMesh{2}) -> cache

Every leaf of `cache` from `geo` -- anything SDFLibrary.jl's `sdf_value` accepts -- over `mesh`: its
corners sampled into `phi`, its moments built from them into `cells`. `mesh` must have the leaves the
cache was allocated for, and live on the cache's backend. Blocks until the work has finished.
"""
function update_cache!(cache::MarchingSquaresCutCellCache{<:AbstractVector{<:SVector{4}}}, geo,
                       mesh::AdaptiveMesh{2})
    _check_leaf_size(cache, mesh)
    backend = get_backend(cache)
    typeof(KernelAbstractions.get_backend(mesh)) == typeof(backend) || throw(ArgumentError(
        "the tree lives on $(KernelAbstractions.get_backend(mesh)) but the cache on $backend; " *
        "update a device cache against the tree adapted to that device (`adapt(CuArray, mesh)`)"))
    n = nleaves(mesh)
    n > 0 && _leaf_cells_kernel!(backend, DEFAULT_WORKGROUP)(cache.phi, cache.cells, mesh, geo; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return cache
end

# One thread per leaf: its corners, then its moments from exactly those corners.
@kernel function _leaf_cells_kernel!(phi, cells, mesh::AdaptiveMesh, geo)
    i = @index(Global)
    if i <= nleaves(mesh)
        c = leaf(mesh, i)
        vals = cell_values(geo, mesh, c)
        @inbounds phi[i] = vals
        @inbounds cells[i] = cut_cell_moments(cell_nodes(mesh, c), vals, get_elem_size(mesh, c))
    end
end

# A leaf-layout cache is one entry per leaf, in leaf order, in both its lists.
@noinline function _check_leaf_size(cache::MarchingSquaresCutCellCache, mesh::AdaptiveMesh)
    n = nleaves(mesh)
    np, nc = length(cache.phi), length(cache.cells)
    (np == n && nc == n) || throw(DimensionMismatch(
        "the cache holds $np corner values and $nc cells, but `mesh` has $n leaves; on an " *
        "`AdaptiveMesh` both are one entry per leaf, in leaf order"))
    return nothing
end
