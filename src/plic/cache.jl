# =====================================
# The PLIC cache: every cell's fit over a whole grid, and the face fractions resolved from it
#
# Two passes. The fit pass is one `cell_plane` per cell: its plane into
# `normals`/`intercepts`/`is_valid`, its fluid fraction and classification into `cells`. The face
# pass then makes every face single-valued, which needs a cell to read its neighbour's plane, and so
# a finished first pass to read it from.
#
# Resolution is the mean of a face's two one-sided fractions (`apertures.jl`), taken per cell over
# both of that cell's faces on every axis. The one-sided values are not stored: the face pass
# evaluates both of a face's candidates from the two cells' stored planes (`_face_area`), so no
# one-sided field outlives the update. Each interior face is resolved twice, once from each side,
# and the two copies are bitwise equal -- both sides form the same sum of the same two numbers in
# the same order, over the same `full_face_area`. A domain-boundary face keeps its one cell's
# fraction.

"""
    PLICCutCellCache

What [`allocate_cache`](@ref)`(grid, PLICCutCell())` builds and [`update_cache!`](@ref) refreshes,
and how a consumer reads PLIC data:

- `cells` -- every cell's [`CutCellData`](@ref)`{D,T,2D,D-1}`, a `StructArray` sized `grid.n`, the
  same per-cell struct every other cache holds: `cells.volume_fraction`, `cells.face_fraction`,
  `cells.kind`, ... as plain arrays, `cells[ci]` as one cell's struct.
  - `volume_fraction` is the plane's fluid fraction, the complement of [`cell_plane`](@ref)'s solid
    `fraction`, and `kind` its three-way reading (`CELL_INSIDE` at `0`, `CELL_OUTSIDE` at `1`,
    `CELL_CUT` strictly between).
  - `face_fraction` is the resolved open fraction of each of the cell's `2D` faces, direction-indexed
    (`1 = -x`, `2 = +x`, ...). An interior face appears under both of its cells, bitwise equal.
  - `centroid`, `face_centroid_local` and `interface_centroid` are `NaN`: a centroid plane fit has
    none of them to report.
  - `ambiguous` is `false`.

  The interface's vector area is formed from `face_fraction` on read, as for every `CutCellData`
  ([`interface_normal_area`](@ref)), so the closure identity holds exactly here too.
- `normals` -- every cell's fitted unit normal, an array of `SVector{D,T}` sized `grid.n`.
- `intercepts` -- the matching intercept `a` of each plane, in that cell's unit-cell coordinates
  ([`cell_plane`](@ref)), an array of `T` sized `grid.n`.
- `is_valid` -- `false` where the centroid fit is degenerate, an array of `Bool` sized `grid.n`. The
  plane there is zeroed so it stays finite; branch on the flag, not on `iszero`.

The one-sided apertures a cell's own plane gives are not kept.

`grid` must be isotropic, which [`allocate_cache`](@ref) and [`update_cache!`](@ref) both check on
the host -- the per-cell fit assumes it and cannot check from inside a kernel.
"""
struct PLICCutCellCache{C<:AbstractArray{<:CutCellData},N<:AbstractArray{<:SVector},
                        I<:AbstractArray{<:Real},V<:AbstractArray{Bool}} <: AbstractCutCellCache
    cells::C
    normals::N
    intercepts::I
    is_valid::V
end

Adapt.@adapt_structure PLICCutCellCache
KernelAbstractions.get_backend(cache::PLICCutCellCache) = get_backend(cache.intercepts)
@inline face_fractions(cache::PLICCutCellCache) = cache.cells.face_fraction

function allocate_cache(grid::CartesianGrid{D,T}, ::PLICCutCell;
                        backend=KernelAbstractions.CPU()) where {D,T}
    _require_isotropic(grid)
    n = Tuple(grid.n)
    cells = _allocate_cells(backend, CutCellData{D,T,2D,D-1}, n)
    # What a centroid plane fit has no value for, written once here; no update touches them.
    fill!(cells.ambiguous, false)
    fill!(cells.centroid, fill(T(NaN), SVector{D,T}))
    fill!(cells.face_centroid_local, fill(fill(T(NaN), SVector{D-1,T}), SVector{2D,SVector{D-1,T}}))
    fill!(cells.interface_centroid, fill(T(NaN), SVector{D,T}))
    cache = PLICCutCellCache(cells,
                             KernelAbstractions.allocate(backend, SVector{D,T}, n),
                             KernelAbstractions.allocate(backend, T, n),
                             KernelAbstractions.allocate(backend, Bool, n))
    return update_cache!(cache, EmptyGeo(one(T)), grid)
end

function update_cache!(cache::PLICCutCellCache, geo::AbstractSDFGeometry,
                       grid::CartesianGrid{D}) where {D}
    _check_cache_size(cache.cells, grid)
    _require_isotropic(grid)
    T = eltype(cache.intercepts)
    g = convert(CartesianGrid{D,T}, grid)
    backend = get_backend(cache)
    cells = cache.cells
    _plic_fit_kernel!(backend, 64)(cells.volume_fraction, cells.kind, cache.normals,
                                   cache.intercepts, cache.is_valid, g, geo; ndrange=size(cells))
    _plic_face_fraction_kernel!(backend, 64)(cells.face_fraction, cache.normals, cache.intercepts,
                                             cache.is_valid, cells.volume_fraction, g.d;
                                             ndrange=size(cells))
    KernelAbstractions.synchronize(backend)
    return cache
end

# One thread per cell: one `cell_plane` at its centroid, into the plane arrays and the fraction and
# kind columns of `cells`. `cell_plane` decides a degenerate fit, and zeroes its plane.
@kernel function _plic_fit_kernel!(volume_fraction, kind, normals, intercepts, is_valid,
                                   grid::CartesianGrid, geo)
    ci = @index(Global, Cartesian)
    plane = cell_plane(geo, grid, ci)
    # `plane.fraction` measures the solid -- `calc_volume`'s integrand and the sense a VOF caller
    # expects -- while what is stored is the fluid, matching every other cache's `volume_fraction`.
    # This is the only place the two meet.
    frac = one(eltype(volume_fraction)) - plane.fraction
    @inbounds begin
        normals[ci] = plane.normal
        intercepts[ci] = plane.intercept
        is_valid[ci] = plane.is_valid
        volume_fraction[ci] = frac
        # Classified from the **stored** fraction rather than from the raw solid one, so that
        # `kind == CELL_CUT` and `generate_mesh`'s `0 < volume_fraction < 1` stay exactly one set.
        kind[ci] = iszero(frac) ? CELL_INSIDE : isone(frac) ? CELL_OUTSIDE : CELL_CUT
    end
end

# One thread per cell, writing all `2D` of its own resolved face fractions: a pure stencil over the
# planes the fit pass stored, writing only `face_fraction`, which no thread reads -- so the fit
# pass's arrays are read-only here.
@kernel function _plic_face_fraction_kernel!(face_fraction, @Const(normals), @Const(intercepts),
                                             @Const(is_valid), @Const(volume_fraction), cellsize)
    ci = @index(Global, Cartesian)
    fit = (normals, intercepts, volume_fraction, is_valid)
    @inbounds face_fraction[ci] = _resolved_face_fractions(fit, ci, cellsize)
end

@inline function _resolved_face_fractions(fit, ci::CartesianIndex{D},
                                          cellsize::SVector{D,T}) where {D,T}
    n = size(first(fit))
    return SVector{2D,T}(ntuple(k -> _resolved_slot(fit, ci, k, n, cellsize), Val(2D)))
end

# Cell `ci`'s own one-sided open measure of its face `k`, from its stored plane.
@inline function _stored_face_area(fit, ci, k::Int, cellsize)
    normals, intercepts, volume_fraction, is_valid = fit
    return @inbounds _face_area(normals[ci], intercepts[ci], volume_fraction[ci], is_valid[ci],
                                cellsize, k)
end

# Slot `k` of cell `ci`: its low face along axis `c` for odd `k`, its high face for even. On the low
# face this cell is the positive side and the cell below the negative, whose candidate is its own
# high face `k + 1`; on the high face the senses swap. An absent neighbour is clamped onto `ci`
# rather than branched around -- evaluated, then dropped by the `has` flag -- so the only divergence
# is on the two end planes of each axis.
@inline function _resolved_slot(fit, ci::CartesianIndex{D}, k::Int, n, cellsize) where {D}
    c = (k + 1) >> 1
    e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, Val(D)))
    full = full_face_area(cellsize, k)
    own = _stored_face_area(fit, ci, k, cellsize)
    if isodd(k)
        has = ci[c] >= 2
        below = _stored_face_area(fit, has ? ci - e : ci, k + 1, cellsize)
        return _resolve(below / full, has, own / full, true)
    else
        has = ci[c] <= n[c] - 1
        above = _stored_face_area(fit, has ? ci + e : ci, k - 1, cellsize)
        return _resolve(own / full, true, above / full, has)
    end
end

# The mean of a face's two one-sided candidates, negative side first, or the single candidate a
# domain-boundary face has.
@inline _resolve(a_neg::T, has_neg::Bool, a_pos::T, has_pos::Bool) where {T} =
    has_neg && has_pos ? (a_neg + a_pos) / 2 : (has_neg ? a_neg : a_pos)
