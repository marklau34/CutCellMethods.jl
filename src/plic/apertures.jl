# =====================================
# Cut-cell apertures from one cell's plane
#
# How open each Cartesian face of a cell is, given the plane `plic.jl` fitted there. Pure function
# of one cell's fit -- `(normal, intercept, volume_fraction, is_valid)` plus a cell size. Nothing
# owns storage or looks at a neighbour: the PLIC cache's face pass (`cache.jl`) calls these on the
# planes it stored, for a cell and for its neighbours.
#
# ---------------------------------------------------------------------------
# The direction-indexed layout
#
# A cell's faces are an `SVector{2D}` indexed by DIRECTION, in `CartesianMeshes`' numbering
# (`1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`, `5 = -z`, `6 = +z`) -- the numbering
# `CutCellData.face_fraction` uses, and `direction_axis`/`direction_sign` decode it.
#
# The face between cells `ci` and `ci + e_c` has two names: `ci`'s slot `2c` and `ci + e_c`'s slot
# `2c-1`, and evaluated from the two cells' own planes those slots hold different numbers (see
# below). A boundary face has one name: axis `c`'s low boundary face is cell `1`'s slot `2c-1`, its
# high one cell `n[c]`'s slot `2c`.
#
# ---------------------------------------------------------------------------
# One-sided, and what the cache does about it
#
# Each plane is fitted from its own cell's centroid alone, so the two cells flanking a face each
# have their own candidate for how open it is, and near a sharp feature they genuinely disagree.
# `_face_area` evaluates a cell's own candidate and stops there.
#
# **Resolving that is not optional** for anything that fluxes: a scheme reading two different
# apertures for one face manufactures mass at it. The cache stores the mean, for the face between
# `ci` and `ci + e_c`:
#
#     open = (own(ci)[2c] + own(ci + e_c)[2c-1]) / 2
#
# and both cells sharing that face compute the same bits: floating-point addition is commutative,
# and each candidate is evaluated by the one compiled `_face_area`, so the face is single-valued
# with no accumulation-order contract to honour.
#
# Averaging is the symmetric, no-neighbour-privileged choice; a cut-cell projection's no-penetration
# exactness holds for any well-defined per-face aperture, however derived. Boundary faces need no
# resolution at all.
#
# ---------------------------------------------------------------------------
# Relationship to the nodal methods
#
# The PLIC cache stores the resolved fractions in a `CutCellData`, as every other cache does, but
# the two reconstructions differ:
#
# - the nodal marching-squares/-cubes case (phi at the cell's corners) makes a shared face
#   single-valued for free -- both neighbours interpolate the same corner values -- but samples a
#   field that is not linear between two corners near a sharp feature or a mesh vertex.
# - this file uses the centroid plane fit, one accurate sample per cell and exactly linear along
#   its own plane, but cell-local, with the one-sidedness above.
#
# Both are dimension-generic (2D and 3D).

"""
    _cell_wet(n, a, frac, valid, s, ::Val{c}, ::Val{p}, len) -> T

Wet measure of one cell's share of one face, from its own stored plane fit -- the 2D (wet length)
case; the 3D (wet area) method follows.

Works in the cell's own unit-cell coordinates `ξ ∈ [0,1]^D`, where the fit reads
`φ̃(ξ) = n̂ ⋅ ξ - a`, positive in the fluid. The open fraction is invariant under scaling `φ̃`, so no
conversion back to physical units is needed.

`s` selects which of the cell's two faces along axis `c` is measured (`1` high side, `0` low side);
the in-plane axis `p` runs `0 -> 1` across the face.

`valid` is the cell's fit flag; `false` reads `frac` instead, since the zeroed plane stored for a
degenerate cell would read `φ̃ ≡ 0` as fully wet -- backwards for a cell deep inside the body.
`frac` is the **fluid** fraction, so that branch exists for `frac < 1/2`, not `> 1/2`.
"""
@inline function _cell_wet(n::SVector{2,T}, a::T, frac::T, valid::Bool, s::T, ::Val{c}, ::Val{p},
                           len::T) where {T,c,p}
    valid || return frac < T(0.5) ? zero(T) : len
    base = n[c] * s - a
    return _open_fraction(base, base + n[p]) * len
end

"""
3D counterpart of [`_cell_wet`](@ref): wet AREA rather than wet length, from the same unit-cell fit.

On face `ξ_c = s` the fit is the 2D plane `n̂[p] ξ_p + n̂[q] ξ_q = a - n̂[c] s`, so the solid share of
the face is the 2D PLIC forward problem, `CartesianMeshes.get_volume_fraction`, and the wet area is
its complement scaled by `U * V`.

A face parallel to the plane (`n̂[p] == n̂[q] == 0`) is guarded first: `get_volume_fraction` would
return `NaN` there when the offset is exactly zero. `φ̃` is then the constant `base` across the
face, read with the same `>= 0` convention `_open_fraction` uses.
"""
@inline function _cell_wet(n::SVector{3,T}, a::T, frac::T, valid::Bool, s::T, ::Val{c}, ::Val{p},
                           ::Val{q}, U::T, V::T) where {T,c,p,q}
    valid || return frac < T(0.5) ? zero(T) : U * V
    base = n[c] * s - a
    iszero(n[p]) && iszero(n[q]) && return base >= zero(T) ? U * V : zero(T)
    solid = CartesianMeshes.get_volume_fraction(SVector{2,T}(n[p], n[q]), -base)
    return U * V * (one(T) - solid)
end

# One cell's own fit evaluated on its own face `k`, in the direction-indexed layout above and in raw
# physical units: a length in 2D, an area in 3D. The leading arguments are the fields `cell_plane`
# produces, which is what the cache stores per cell. One-sided: an interior face has a second
# candidate from the cell on its other side, and the cache resolves the two. A face is measured
# along its own in-plane axes, not the axis it is normal to -- an x-face across y (and z). In 3D the
# in-plane axes run in increasing order skipping `c`: `(2,3)`, `(1,3)`, `(1,2)`.
#
# `@noinline` is what keeps a shared face single-valued. The face pass evaluates each face twice,
# once in each of its two cells' threads, and the two must agree bitwise. `get_volume_fraction`
# (CartesianMeshes) is `@inline @fastmath`, so inlined into two different call sites it may be
# contracted and reassociated differently and land an ulp apart. Compiled once, the same inputs
# give the same bits wherever the call comes from.
@noinline function _face_area(n::SVector{2,T}, a::T, frac::T, valid::Bool, cellsize::SVector{2,T},
                              k::Int) where {T}
    s = isodd(k) ? zero(T) : one(T)
    return k <= 2 ? _cell_wet(n, a, frac, valid, s, Val(1), Val(2), @inbounds(cellsize[2])) :
                    _cell_wet(n, a, frac, valid, s, Val(2), Val(1), @inbounds(cellsize[1]))
end

@noinline function _face_area(n::SVector{3,T}, a::T, frac::T, valid::Bool, cellsize::SVector{3,T},
                              k::Int) where {T}
    s = isodd(k) ? zero(T) : one(T)
    dx, dy, dz = @inbounds (cellsize[1], cellsize[2], cellsize[3])
    return k <= 2 ? _cell_wet(n, a, frac, valid, s, Val(1), Val(2), Val(3), dy, dz) :
           k <= 4 ? _cell_wet(n, a, frac, valid, s, Val(2), Val(1), Val(3), dx, dz) :
                    _cell_wet(n, a, frac, valid, s, Val(3), Val(1), Val(2), dx, dy)
end
