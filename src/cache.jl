# =====================================
# Cut-cell storage a consumer keeps between updates
#
# `cut_cell_moments(method, domain, geo, cell)` answers one cell and keeps nothing. A cache is the
# other half: storage allocated once for a grid, refreshed whenever the body moves, and read by
# whoever needs it. Every method's cache holds that method's own per-cell struct for every cell of
# the grid, as a `StructArray` -- so a consumer after one field reads it as a plain array
# (`cache.cells.volume_fraction`), and one after a whole cell reads the struct (`cache.cells[ci]`)
# through the accessor layer. What else a cache holds depends on what its method needs; see each
# method's own `cache.jl`.
#
# Everything is launched through KernelAbstractions over the cache's own backend, so one call
# serves CPU and GPU. `update_cache!` synchronizes before it returns: the cache exists to be read,
# and a caller reading it straight after has to see the finished numbers.

"""
    allocate_cache(grid::CartesianGrid, method::AbstractCutCellMethod; backend=CPU()) -> cache

Storage for `method`'s cut cells over every cell of `grid`, on `backend` and in `grid`'s element
type. Refresh it with [`update_cache!`](@ref) and read it directly; each method's cache type
documents its fields.

Seeded as **no body** -- every cell `CELL_OUTSIDE`, all fluid, every face fully open -- by one update
against `SDFLibrary.EmptyGeo`, so a cache read before its first update describes an empty domain
rather than uninitialized memory.
"""
function allocate_cache end

"""
    update_cache!(cache, geo, grid::CartesianGrid) -> cache

Recompute every cell of `cache` from the body `geo` over `grid`, then return `cache`.

`grid` must have the `n` the cache was allocated for, but its origin may move: a body moving
through a fixed grid is the grid shifted into the body's own frame. It is converted to the cache's
element type once, so a `Float64` grid does not run a `Float32` cache's reconstruction in double.

Blocks until the work has finished, so the cache is readable on return on any backend.
"""
function update_cache! end

"""
    face_fractions(cache) -> AbstractArray{SVector{2D,T}}
    volume_fractions(cache) -> AbstractArray{T}
    kinds(cache) -> AbstractArray{Int8}

The three per-cell fields every cache holds whatever its method, so a consumer reads any cache the
same way. Each is sized `grid.n`:

- `face_fractions` -- each cell's open fraction of its `2D` faces, direction-indexed
  (`1 = -x`, `2 = +x`, ...). Single-valued: an interior face appears under both of its cells with
  bitwise the same value. For PLIC this is the **resolved** field, not the one-sided `face_area`.
- `volume_fractions` -- each cell's **fluid** fraction.
- `kinds` -- `CELL_INSIDE` / `CELL_OUTSIDE` / `CELL_CUT`.

These return the cache's own storage, not copies, so writing through them edits the cache -- which
is how a consumer post-processes in place (snapping slivers to dry, say). A cell edited that way no
longer agrees with the rest of its stored struct. `@inline` and allocation-free, so they are legal
in a kernel.
"""
function face_fractions end

"""
    volume_fractions(cache) -> AbstractArray{T}

Each cell's fluid fraction, as the cache stores it. See [`face_fractions`](@ref).
"""
@inline volume_fractions(cache::AbstractCutCellCache) = cache.cells.volume_fraction

"""
    kinds(cache) -> AbstractArray{Int8}

Each cell's `CELL_INSIDE` / `CELL_OUTSIDE` / `CELL_CUT`, as the cache stores it. See
[`face_fractions`](@ref).
"""
@inline kinds(cache::AbstractCutCellCache) = cache.cells.kind

# A `StructArray{S}` of `dims` cells on `backend`: every field of `S` gets its own column, allocated
# there. Built from explicit columns rather than `StructArray{S}(undef, dims)`, which only allocates
# on the host.
function _allocate_cells(backend, ::Type{S}, dims::Dims) where {S}
    cols = ntuple(i -> KernelAbstractions.allocate(backend, fieldtype(S, i), dims), fieldcount(S))
    return StructArray{S}(NamedTuple{fieldnames(S)}(cols))
end

# A cache is allocated for one grid's `n`; only the origin may change between updates.
@noinline function _check_cache_size(cells::AbstractArray, grid::CartesianGrid)
    size(cells) == Tuple(grid.n) || throw(DimensionMismatch(
        "the cache holds $(size(cells)) cells, but `grid` has $(Tuple(grid.n)); a cache is " *
        "allocated for one grid's `n`, and only its origin may move between updates"))
    return nothing
end
