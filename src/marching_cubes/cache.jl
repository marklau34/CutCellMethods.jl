# =====================================
# The marching-cubes cache: `marching_squares/cache.jl` one dimension up
#
# Same layout, same node-sampled `phi`, and the same two kernels -- `_update_nodal_cache!` is
# dimension-generic, and `cut_cell_moments(MarchingCubesCutCell(), grid, phi, ci)` is the 3D
# method it reaches.

"""
    MarchingCubesCutCellCache

What [`allocate_cache`](@ref)`(grid, MarchingCubesCutCell())` builds and [`update_cache!`](@ref)
refreshes:

- `cells` -- every cell's [`CutCellData`](@ref)`{3,T,6}`, a `StructArray` sized `grid.n`.
  `cells.volume_fraction`, `cells.face_fraction`, `cells.kind`, ... are plain arrays of one field;
  `cells[ci]` is cell `ci`'s whole struct. A shared face is single-valued by construction.
- `phi` -- the field at every grid node, sized `grid.n .+ 1`: the corner values `cells` was built
  from. `generate_mesh(cache.phi, grid, MarchingCubesCutCell())` marches the same numbers.

About 200 bytes a cell at `Float64` -- 1.6 GB over `200^3`. Where only a few fields are wanted,
`cut_cell_moments(MarchingCubesCutCell(), grid, geo, ci)` in the consumer's own kernel is the
lighter route.
"""
struct MarchingCubesCutCellCache{P<:AbstractArray,C<:AbstractArray} <: AbstractCutCellCache
    phi::P
    cells::C
end

Adapt.@adapt_structure MarchingCubesCutCellCache
KernelAbstractions.get_backend(cache::MarchingCubesCutCellCache) = get_backend(cache.phi)
@inline face_fractions(cache::MarchingCubesCutCellCache) = cache.cells.face_fraction

function allocate_cache(grid::CartesianGrid{3,T}, ::MarchingCubesCutCell;
                        backend=KernelAbstractions.CPU()) where {T}
    cache = MarchingCubesCutCellCache(
        KernelAbstractions.allocate(backend, T, Tuple(grid.n) .+ 1),
        _allocate_cells(backend, CutCellData{3,T,6,2}, Tuple(grid.n)))
    return update_cache!(cache, EmptyGeo(one(T)), grid)
end

update_cache!(cache::MarchingCubesCutCellCache, geo, grid::CartesianGrid{3}) =
    _update_nodal_cache!(cache, geo, grid, MarchingCubesCutCell())
