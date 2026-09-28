# =====================================
# Cut-cell apertures, one cell at a time
#
# How open each Cartesian **face** of a cell is, given the plane `plic.jl` fitted there. Everything
# here is a pure function of **one** cell's fit -- `(normal, intercept, volume_fraction, is_valid)`
# plus a cell size. Nothing owns storage, reads a grid-wide array, looks at a neighbour, or is a
# kernel: a consumer allocates whatever field it wants, fills it with `cut_cell_moments`, and reads
# it back through these.
#
# ---------------------------------------------------------------------------
# The direction-indexed layout, stated once
#
# A cell's `face_area` is an `SVector{2D}` indexed by DIRECTION, in `CartesianMeshes`' numbering
# (`1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`, `5 = -z`, `6 = +z`) -- the same numbering
# `CutCellData.face_fraction` uses, so a consumer moving between the two reconstructions carries
# one convention, and `direction_axis`/`direction_sign` decode both.
#
# So the face between cells `ci` and `ci + e_c` has two names: `ci`'s slot `2c` and `ci + e_c`'s
# slot `2c-1`, and in a field built from `cut_cell_moments` those two slots hold **different
# numbers** (see below). A boundary face has one name: axis `c`'s low boundary face is cell `1`'s
# slot `2c-1`, its high one cell `n[c]`'s slot `2c`.
#
# ---------------------------------------------------------------------------
# One-sided, and what to do about it
#
# Each plane is fitted from its own cell's centroid alone, so the two cells flanking a face each
# have their own candidate for how open it is, and near a sharp feature they genuinely disagree.
# `cell_face_areas` hands back a cell's own candidate and stops there.
#
# **Resolving that is the consumer's job, and it is not optional** for anything that fluxes: a
# scheme reading two different apertures for one face manufactures mass at it. It is one line --
# for the face between `ci` and `ci + e_c`, over a field `a` of `PLICCutCellData`:
#
#     open = (a[ci].face_area[2c] + a[ci + e_c].face_area[2c-1]) / 2
#
# and there is deliberately no function wrapping it, because there is nothing left to get wrong:
# floating-point addition is **commutative**, so the two cells sharing that face compute the same
# bits whichever order they name each other in. The face is single-valued for free, with no
# accumulation-order contract to honour.
#
# Averaging is a choice, not a structural requirement -- a cut-cell projection's no-penetration
# exactness holds for *any* well-defined per-face aperture, however derived. It is simply the
# symmetric, no-neighbour-privileged option. Boundary faces need no resolution at all.
#
# ---------------------------------------------------------------------------
# Relationship to `CutCellData` (`src/cut_cell.jl`)
#
# `CutCellData.face_fraction` is the same physical quantity as `face_area` here, in the same
# per-cell direction-indexed layout. Where both apply a consumer should prefer the moment layer: it
# gets the open volume, the interface facet and the closure identity in one construction, and has no
# face to resolve. They are kept separate because they reconstruct differently and neither subsumes
# the other:
#
# - the moment layer walks the **nodal** marching-squares/-cubes case (phi at the cell's corners),
#   which makes a shared face single-valued for free -- both neighbours interpolate the same corner
#   values -- but samples a field that is genuinely not linear between two corners near a sharp
#   feature or a mesh vertex.
# - this file uses the **centroid** plane fit, which is one accurate sample per cell and exactly
#   linear along its own plane, but is cell-local, with the consequence above.
#
# Both are dimension-generic (2D and 3D).

"""
    face_area_of(d::PLICCutCellData, dir, cellsize) -> T

One direction's open measure out of a cell's data, in raw physical units -- a length in 2D, an area
in 3D. A broadcastable scalar function: `face_area_of.(cells, dir, Ref(grid.d))` fuses into a
surrounding broadcast with no temporary.

**`cellsize` is accepted and ignored here**: this reconstruction stores the area outright. It is in
the signature so that one call site reads either reconstruction, and the [`CutCellData`](@ref)
method -- which stores the fraction -- genuinely needs it.

Whether the number is one-sided or resolved depends on where `d` came from; see
[`PLICCutCellData`](@ref)`.face_area`.
"""
@inline face_area_of(d::PLICCutCellData{D,T}, dir::Integer, cellsize::SVector{D,T}) where {D,T} =
    @inbounds d.face_area[dir]

"""
    face_fraction_of(d::PLICCutCellData, dir, cellsize) -> T

One direction's open **fraction**, a dimensionless `[0,1]` -- what a flux weight or a Poisson
coefficient wants. Divided out here by that face's own full measure, since this reconstruction
stores the area; see [`face_area_of`](@ref) for why both take `cellsize` whichever half they store.
"""
@inline face_fraction_of(d::PLICCutCellData{D,T}, dir::Integer, cellsize::SVector{D,T}) where {D,T} =
    @inbounds d.face_area[dir] / full_face_area(cellsize, dir)

"""
    is_face_open(d::PLICCutCellData, dir) -> Bool

Whether one named face is open at all. Answered on the stored area here and on the stored fraction
in the [`CutCellData`](@ref) method; both are `iszero` tests and agree exactly.
"""
@inline is_face_open(d::PLICCutCellData, dir::Integer) = @inbounds !iszero(d.face_area[dir])

"""
    _cell_wet(n, a, frac, valid, s, ::Val{c}, ::Val{p}, len) -> T

Wet measure of one cell's share of one face, from that cell's own stored plane fit -- the 2D (wet
length) case; the 3D (wet area) method follows.

Everything is in the cell's own **unit-cell** coordinates `ξ ∈ [0,1]^D`, where the fit reads
`φ̃(ξ) = n̂ ⋅ ξ - a`, positive in the fluid. `φ̃` is the true fitted signed distance up to the factor
`grid.d[1]`, and the open fraction of a face is invariant under scaling `φ̃` -- `_open_fraction`
forms only the ratio `d1/(d1-d2)`, and `CartesianMeshes.get_volume_fraction` normalizes the plane
itself -- so that factor drops out and no conversion back to physical units is needed.

`s` selects which of the cell's two faces along axis `c` is measured: `1` for its high side, `0`
for its low side. The in-plane axis `p` runs `0 -> 1` across the face.

`valid` is the cell's [`PLICCutCellData`](@ref) `is_valid` flag, and `false` takes the degenerate
branch, which reads `frac` instead: the plane stored for such a cell is zeroed, and `φ̃ ≡ 0` would
read as fully wet, which is exactly backwards for a cell deep inside the body. `frac` is the
**fluid** fraction, so the mostly-solid cell that branch exists for is `frac < 1/2`, not `> 1/2`.
"""
@inline function _cell_wet(n::SVector{2,T}, a::T, frac::T, valid::Bool, s::T, ::Val{c}, ::Val{p},
                           len::T) where {T,c,p}
    valid || return frac < T(0.5) ? zero(T) : len
    base = n[c] * s - a
    return _open_fraction(base, base + n[p]) * len
end

"""
3D counterpart of [`_cell_wet`](@ref): wet AREA rather than wet length, from the same unit-cell fit.

On face `ξ_c = s` the fit is the 2D plane `n̂[p] ξ_p + n̂[q] ξ_q = a - n̂[c] s`, so the solid share
of the face is exactly the 2D PLIC forward problem, `CartesianMeshes.get_volume_fraction`, and the
wet area is its complement scaled by `U * V`.

A face parallel to the plane (`n̂[p] == n̂[q] == 0`) is guarded first: `get_volume_fraction`
normalizes by `|n̂[p]| + |n̂[q]|` and returns `NaN` there when the offset is exactly zero. `φ̃` is
then the constant `base` across the face, read with the same `>= 0` convention `_open_fraction`
uses.
"""
@inline function _cell_wet(n::SVector{3,T}, a::T, frac::T, valid::Bool, s::T, ::Val{c}, ::Val{p},
                           ::Val{q}, U::T, V::T) where {T,c,p,q}
    valid || return frac < T(0.5) ? zero(T) : U * V
    base = n[c] * s - a
    iszero(n[p]) && iszero(n[q]) && return base >= zero(T) ? U * V : zero(T)
    solid = CartesianMeshes.get_volume_fraction(SVector{2,T}(n[p], n[q]), -base)
    return U * V * (one(T) - solid)
end

"""
    cell_face_areas(normal, intercept, volume_fraction, is_valid, cellsize) -> SVector{2D,T}

**One** cell's own fit evaluated on **its own** `2D` Cartesian faces, in the direction-indexed
layout [`PLICCutCellData`](@ref)`.face_area` uses (`1 = -x`, `2 = +x`, `3 = -y`, ...) and in raw
physical units: a length in 2D, an area in 3D. Slot `2c-1` is the cell's low face along axis `c`,
slot `2c` its high face.

The four leading arguments are the fields [`cell_plane`](@ref) produces, however the caller chose to
store them -- this takes a *fit*, not a cache, so it serves a consumer keeping its own storage
exactly as it serves [`cut_cell_moments`](@ref).

**These are one-sided, and that is the whole point of the function.** An interior face has a second
candidate from the cell on its other side, and resolving the two is the caller's -- see this file's
header. On a domain-boundary face there is no second candidate and this is already final.
"""
@inline function cell_face_areas(n::SVector{2,T}, a::T, frac::T, valid::Bool,
                                 cellsize::SVector{2,T}) where {T}
    # A face is measured along its own IN-PLANE axis, not the axis it is normal to: an x-face is
    # measured across y. Silently wrong on a square cell if inverted, hence the names.
    len_x = @inbounds cellsize[2]
    len_y = @inbounds cellsize[1]
    return SVector{4,T}(_cell_wet(n, a, frac, valid, zero(T), Val(1), Val(2), len_x),
                        _cell_wet(n, a, frac, valid, one(T), Val(1), Val(2), len_x),
                        _cell_wet(n, a, frac, valid, zero(T), Val(2), Val(1), len_y),
                        _cell_wet(n, a, frac, valid, one(T), Val(2), Val(1), len_y))
end

# 3D: wet AREA per face. In-plane axes in increasing order skipping `c` -- `(2,3)`, `(1,3)`, `(1,2)`
# -- the pairing the direction numbering implies, so the slots line up with `face_area`.
@inline function cell_face_areas(n::SVector{3,T}, a::T, frac::T, valid::Bool,
                                 cellsize::SVector{3,T}) where {T}
    dx, dy, dz = @inbounds (cellsize[1], cellsize[2], cellsize[3])
    return SVector{6,T}(_cell_wet(n, a, frac, valid, zero(T), Val(1), Val(2), Val(3), dy, dz),
                        _cell_wet(n, a, frac, valid, one(T), Val(1), Val(2), Val(3), dy, dz),
                        _cell_wet(n, a, frac, valid, zero(T), Val(2), Val(1), Val(3), dx, dz),
                        _cell_wet(n, a, frac, valid, one(T), Val(2), Val(1), Val(3), dx, dz),
                        _cell_wet(n, a, frac, valid, zero(T), Val(3), Val(1), Val(2), dx, dy),
                        _cell_wet(n, a, frac, valid, one(T), Val(3), Val(1), Val(2), dx, dy))
end


"""
    is_cell_open(d::PLICCutCellData) -> Bool

Whether the cell can be fluxed through at all. Answered here on `kind` -- true for anything not
wholly inside the body -- where the [`CutCellData`](@ref) method answers on the face fractions;
the two differ only for a cut cell whose faces all happen to be closed.
"""
@inline is_cell_open(d::PLICCutCellData) = d.kind != CELL_INSIDE
