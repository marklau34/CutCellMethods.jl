# =====================================
# The PLIC cache: every cell's fit over a whole grid, and the face fractions resolved from it
#
# Two passes, and it has to be two. The fit pass is one `cut_cell_moments` per cell, stored whole,
# and its `face_area` is ONE-SIDED -- the cell's own plane on its own faces, see `apertures.jl`'s
# header. The face pass then makes every face single-valued, which needs a cell to read its
# neighbour's fit, and so a finished first pass to read it from.
#
# The resolution is the mean of a face's two one-sided fractions, `apertures.jl`'s one line, taken
# per cell over both of that cell's faces on every axis. Each interior face is therefore resolved
# twice, once from each side, and the two copies are bitwise equal: both sides form the same sum of
# the same two numbers in the same order, over the same `full_face_area`. A domain-boundary face has
# no second candidate and keeps its one cell's fraction.

"""
    PLICCutCellCache

What [`allocate_cache`](@ref)`(grid, PLICCutCell())` builds and [`update_cache!`](@ref) refreshes:

- `cells` -- every cell's [`PLICCutCellData`](@ref)`{D,T,2D}`, a `StructArray` sized `grid.n`:
  `cells.volume_fraction`, `cells.kind`, `cells.normal`, ... as plain arrays, `cells[ci]` as one
  cell's struct. Its `face_area` is the fit's own **one-sided** measure, kept as the struct defines
  it; a flux reads `face_fraction` instead.
- `face_fraction` -- per cell, the **resolved** open fraction of each of its `2D` faces, an array of
  `SVector{2D,T}` sized `grid.n` in the direction-indexed layout (`1 = -x`, `2 = +x`, ...). An
  interior face appears under both of its cells, and the two copies are bitwise equal.

`grid` must be isotropic, which [`allocate_cache`](@ref) and [`update_cache!`](@ref) both check
on the host -- the per-cell fit assumes it and cannot check from inside a kernel.
"""
struct PLICCutCellCache{C<:AbstractArray,F<:AbstractArray} <: AbstractCutCellCache
    cells::C
    face_fraction::F
end

Adapt.@adapt_structure PLICCutCellCache
KernelAbstractions.get_backend(cache::PLICCutCellCache) = get_backend(cache.face_fraction)
@inline face_fractions(cache::PLICCutCellCache) = cache.face_fraction

function allocate_cache(grid::CartesianGrid{D,T}, ::PLICCutCell;
                        backend=KernelAbstractions.CPU()) where {D,T}
    _require_isotropic(grid)
    cache = PLICCutCellCache(
        _allocate_cells(backend, PLICCutCellData{D,T,2D}, Tuple(grid.n)),
        KernelAbstractions.allocate(backend, SVector{2D,T}, Tuple(grid.n)))
    return update_cache!(cache, EmptyGeo(one(T)), grid)
end

function update_cache!(cache::PLICCutCellCache, geo::AbstractSDFGeometry,
                       grid::CartesianGrid{D}) where {D}
    _check_cache_size(cache.cells, grid)
    _require_isotropic(grid)
    T = eltype(eltype(cache.face_fraction))
    g = convert(CartesianGrid{D,T}, grid)
    backend = get_backend(cache)
    _plic_cells_kernel!(backend, 64)(cache.cells, g, geo; ndrange=size(cache.cells))
    _plic_face_fraction_kernel!(backend, 64)(cache.face_fraction, cache.cells.face_area, g.d;
                                             ndrange=size(cache.cells))
    KernelAbstractions.synchronize(backend)
    return cache
end

# One thread per cell: one fit at its centroid, stored whole.
@kernel function _plic_cells_kernel!(cells, grid::CartesianGrid, geo)
    ci = @index(Global, Cartesian)
    @inbounds cells[ci] = cut_cell_moments(PLICCutCell(), grid, geo, ci)
end

# One thread per cell, writing all `2D` of its own resolved face fractions. A pure stencil over the
# one-sided areas the fit pass stored -- no field query, no plane.
@kernel function _plic_face_fraction_kernel!(face_fraction, @Const(face_area), cellsize)
    ci = @index(Global, Cartesian)
    @inbounds face_fraction[ci] = _resolved_face_fractions(face_area, ci, cellsize)
end

@inline function _resolved_face_fractions(face_area::AbstractArray{<:Any,D}, ci::CartesianIndex{D},
                                          cellsize::SVector{D,T}) where {D,T}
    own = @inbounds face_area[ci]
    n = size(face_area)
    return SVector{2D,T}(ntuple(k -> _resolved_slot(face_area, ci, own, k, n, cellsize), Val(2D)))
end

# Slot `k` of cell `ci`: its low face along axis `c` for odd `k`, its high face for even. On the low
# face this cell is the positive side and the cell below the negative; on the high face the senses
# swap. An absent neighbour is clamped onto `ci` rather than branched around -- read, then dropped
# by the `has` flag -- so the only divergence is on the two end planes of each axis.
@inline function _resolved_slot(face_area, ci::CartesianIndex{D}, own, k::Int, n,
                                cellsize) where {D}
    c = (k + 1) >> 1
    e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, Val(D)))
    full = full_face_area(cellsize, k)
    if isodd(k)
        has = ci[c] >= 2
        below = @inbounds face_area[has ? ci - e : ci][k + 1]
        return _resolve(below / full, has, own[k] / full, true)
    else
        has = ci[c] <= n[c] - 1
        above = @inbounds face_area[has ? ci + e : ci][k - 1]
        return _resolve(own[k] / full, true, above / full, has)
    end
end

# The mean of a face's two one-sided candidates, negative side first, or the single candidate a
# domain-boundary face has.
@inline _resolve(a_neg::T, has_neg::Bool, a_pos::T, has_pos::Bool) where {T} =
    has_neg && has_pos ? (a_neg + a_pos) / 2 : (has_neg ? a_neg : a_pos)
