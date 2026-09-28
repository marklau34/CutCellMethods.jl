# =====================================
# The marching-squares cache: every cell's moments over a whole grid, kept between updates
#
# The field is sampled once per grid NODE into `phi`, and the moments kernel reads that nodal block.
# A node is shared by the four cells around it, so that is a quarter of the field queries a
# per-corner sampling makes. Each node is sampled at `get_node(grid, ni)` -- the expression
# `cell_nodes` pairs the value with -- so a corner shared by two cells is exactly one stored number,
# and the watertightness argument of `marching_squares.jl` holds for the cache as it does per cell.
#
# The two kernels below are dimension-generic; `marching_cubes/cache.jl` is the same construction
# one dimension up, on these same kernels.

"""
    MarchingSquaresCutCellCache

What [`allocate_cache`](@ref)`(grid, MarchingSquaresCutCell())` builds and
[`update_cache!`](@ref) refreshes:

- `cells` -- every cell's [`CutCellData`](@ref)`{2,T,4}`, a `StructArray` sized `grid.n`.
  `cells.volume_fraction`, `cells.face_fraction`, `cells.kind`, ... are plain arrays of one field;
  `cells[ci]` is cell `ci`'s whole struct. A shared face is single-valued by construction: cell
  `ci`'s slot `2c` and cell `ci + e_c`'s slot `2c-1` hold bitwise the same fraction.
- `phi` -- the field at every grid node, sized `grid.n .+ 1`: the corner values `cells` was built
  from. `generate_mesh(cache.phi, grid, MarchingSquaresCutCell())` draws the contour of this very
  reconstruction.
"""
struct MarchingSquaresCutCellCache{P<:AbstractArray,C<:AbstractArray} <: AbstractCutCellCache
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

update_cache!(cache::MarchingSquaresCutCellCache, geo, grid::CartesianGrid{2}) =
    _update_nodal_cache!(cache, geo, grid, MarchingSquaresCutCell())

# The shared body of the two nodal caches' `update_cache!`: sample every node, then build every
# cell from those samples. Two launches on one backend run in order, so the second reads a finished
# `phi` without a synchronize between them.
function _update_nodal_cache!(cache, geo, grid::CartesianGrid{D}, method) where {D}
    _check_cache_size(cache.cells, grid)
    T = eltype(cache.phi)
    g = convert(CartesianGrid{D,T}, grid)
    backend = get_backend(cache)
    _sample_nodes_kernel!(backend, 64)(cache.phi, g, geo; ndrange=size(cache.phi))
    _nodal_cells_kernel!(backend, 64)(cache.cells, cache.phi, g, method; ndrange=size(cache.cells))
    KernelAbstractions.synchronize(backend)
    return cache
end

# One thread per grid NODE, at `get_node` -- the same expression `cell_nodes` places a corner with.
@kernel function _sample_nodes_kernel!(phi, grid::CartesianGrid, geo)
    ni = @index(Global, Cartesian)
    @inbounds phi[ni] = eltype(phi)(sdf_value(geo, get_node(grid, ni)))
end

# One thread per cell: the tagged per-cell entry point on the shared nodal block, stored whole.
@kernel function _nodal_cells_kernel!(cells, @Const(phi), grid::CartesianGrid,
                                      method::AbstractCutCellMethod)
    ci = @index(Global, Cartesian)
    @inbounds cells[ci] = cut_cell_moments(method, grid, phi, ci)
end
