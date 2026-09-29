# =====================================
# What one cut cell knows about itself
#
# `CutCellData` is the output of a single geometric construction: volume fraction, outside-volume
# centroid, per-direction open-face fraction and centroid, and the interface facet's centroid -- all
# read off one reconstructed polygon, with the interface's vector area implied by the face fractions
# rather than stored beside them. Deriving them independently is the classic free-stream
# preservation failure of a cut-cell finite volume scheme. `isbits` and fixed-size, so it passes
# into a kernel by value and lives in a plain device array.
#
# ---------------------------------------------------------------------------
# The sign convention
#
# The field is negative inside the body (`get_sdf`'s convention), so a cell's inside region is
# `{phi < 0}` and its outside region -- the one every moment here measures -- is `{phi >= 0}`. This
# is the opposite region to the one [`calc_volume`](@ref) measures. A consumer wanting moments of
# the inside should negate the field: the construction is a pure function of the corner values, so
# `x -> -phi(x)` gives the complement with no separate code path.
#
# The open-face normals and the interface normal point in opposite senses, on purpose. A face's
# `n_k` points out of the outside region, what a flux balance over that volume integrates against;
# the interface's `n_int` points out of the body, agreeing with the field's own normal, with
# `cell_interface`'s segment orientation, and with the winding of the contour `generate_mesh`
# stitches. The divergence theorem over the outside region is therefore
#
#     sum_k A_k n_k  -  A_int n_int  =  0
#
# and that subtraction is `closure_residual`. A consumer assembling a flux balance negates
# `interface_normal` at the point of use -- one negation, in exchange for a surface normal that
# means what a reader expects everywhere else.
#
# ---------------------------------------------------------------------------
# Where `interface_area` and `interface_normal` come from
#
# The closure identity above, rearranged: `A_int n_int = sum_k A_k n_k`. For a linear reconstruction
# of the interface inside a cell that is an algebraic identity, not an approximation -- the closing
# edge of a polygon has exactly minus the vector area of all the others, and `n_int`'s opposite
# sense absorbs the minus. So the interface's vector area is not stored: `interface_normal_area`
# forms it on read from the stored face fractions, one subtraction of axis-aligned open-face areas
# per axis, and `closure_residual` sums the same `face_area_of` products. Both sides of the identity
# are one expression on the same stored numbers, which is what makes the closure hold to the last
# bit rather than merely to roundoff.
#
# `test/test_moments.jl` checks it against the interface segment's normal computed directly from its
# endpoints, the non-circular version of the claim.

"""
    CutCellData{D,T,NF,DF}

The complete geometric description of one cell's outside portion -- the part where `phi >= 0`.
`NF == 2D` is the number of Cartesian faces, indexed by `CartesianMeshes`' direction numbering
(`1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`, ...), and `DF == D - 1` the number of in-plane coordinates a
face centroid is stored with. Both are fixed by `D` -- the concrete types are `CutCellData{2,T,4,1}`
and `CutCellData{3,T,6,2}` -- and are parameters only because a field type cannot compute them.

- `kind` -- [`CELL_INSIDE`](@ref), `CELL_OUTSIDE` or `CELL_CUT`.
- `ambiguous` -- the reconstruction is multi-valued here: the marching-squares case is one of the
  two diagonal ones (in 3D, also a saddle face or an ambiguous cube interior), so the connectivity
  chosen is a guess. `interface_normal` is then a length-weighted lumped normal and must not be
  trusted for a surface flux; count these cells and refine them away rather than trust either
  resolution.
- `volume_fraction` -- the outside measure as a ratio of the whole cell.
- `centroid` -- the outside-volume centroid, not the Cartesian cell centre. A finite volume unknown
  placed at the Cartesian centre of a cell that is 20% outside is an `O(h)` error in exactly the
  cells where the geometry is doing the most work.
- `face_fraction` -- per direction, the open fraction of that Cartesian face.
- `face_centroid_local` -- per direction, the centroid of the face's open part, where a cut face's
  flux belongs, in the face's own in-plane coordinates: `DF` offsets from the cell's lower lattice
  corner along the face's in-plane axes, in increasing axis order. The coordinate along the face
  normal is not stored -- it is the face plane, which the lattice already knows. Read the point in
  global coordinates through [`face_centroid`](@ref), which adds the cell's position back.
- `interface_centroid` -- the centroid of the reconstructed interface. Its vector area
  `A_int n_int`, with `n_int` pointing out of the body, is [`interface_normal_area`](@ref), formed
  from the face fractions on read. That sense is opposite to the open faces'; see this file's
  header, and [`closure_residual`](@ref) for what it costs.

## Fractions, not measures

Neither the outside volume nor any face's area is stored, the interface's included. Each is a
fraction times a cell measure, so every dimensional reader multiplies at the point of use:
[`face_area_of`](@ref), [`face_vector_area`](@ref), the interface readers and
[`closure_residual`](@ref) each take a trailing `cellsize` for exactly that, the same expression in
the same order, so a value read back is bit-identical to the one it was built from.

The fraction is the better half to keep: it is the primitive [`cell_boundary_walk`](@ref) produces,
it is what the hot consumers want (a flux weight and a Poisson coefficient are both dimensionless),
a relative tolerance against it needs no scale, and `iszero` on it answers [`is_cell_open`](@ref)
without one either.

The interface's vector area follows from the open faces by the closure identity, so
[`interface_normal_area`](@ref) re-forms it from the stored fractions -- one subtraction of
[`face_area_of`](@ref) products per axis -- and it comes back bit-identical to what a stored copy
would hold. Its scalar area and unit normal, [`interface_area`](@ref) and [`interface_normal`](@ref),
are two views of that one vector: `interface_area * interface_normal` does not reproduce it bitwise
(normalising and re-scaling is two roundings), and that `O(eps)` gap is enough to break the closure,
so use `interface_normal_area` wherever the product is what you want.

## Face centroids are local

A face centroid is stored as offsets within the face, not as a point: `DF` numbers rather than `D`
(the normal coordinate carries no information), about a fifth of the struct, and an offset within a
cell keeps full relative precision wherever the cell sits, where a single-precision global
coordinate near `|x| = 1000` resolves only to about `6e-5`.

Reading one back needs the cell's corners, which [`face_centroid`](@ref) takes as the corner array
itself, or as a tree and leaf index. It adds the offsets to the lower lattice corner and takes the
normal coordinate from the lattice corner on the face's own side. Two cells sharing a face have
lower corners that differ only along that face's normal, so they add the same offsets to the same
in-plane coordinates and report the same point bit for bit -- the property a flux through the face
needs.
"""
struct CutCellData{D,T,NF,DF}
    kind::Int8
    ambiguous::Bool
    volume_fraction::T
    centroid::SVector{D,T}
    face_fraction::SVector{NF,T}
    face_centroid_local::SVector{NF,SVector{DF,T}}
    interface_centroid::SVector{D,T}
end

@inline Base.eltype(::CutCellData{D,T}) where {D,T} = T
@inline Base.eltype(::Type{<:CutCellData{D,T}}) where {D,T} = T

"""
    is_cut(m)
    is_inside(m)
    is_outside(m)
    is_ambiguous(m)

Classification predicates. `is_outside` means *entirely* outside the body (an uncut cell) and
`is_inside` entirely within it; a cut cell is neither.
"""
@inline is_cut(m::CutCellData) = m.kind == CELL_CUT
@inline is_inside(m::CutCellData) = m.kind == CELL_INSIDE
@inline is_outside(m::CutCellData) = m.kind == CELL_OUTSIDE
@inline is_ambiguous(m::CutCellData) = m.ambiguous

"""
    face_centroid(m, direction, nodes)
    face_centroid(m, direction, mesh::AdaptiveMesh{2}, i::Integer)

The centroid of the open part of Cartesian face `direction`, in global coordinates: `m`'s stored
in-plane offsets added to the cell's lower lattice corner, with the coordinate along the face
normal taken from the lattice corner on the face's side. `m` must be the moments of that cell.

The `nodes` form takes the cell's corner coordinates in [`MS_NODE_BITS`](@ref) or
[`MC_NODE_BITS`](@ref) order, the array the construction itself took -- [`cell_nodes`](@ref) on a
grid. The tree form takes leaf `i`, and finds its corners the same way.

Built from lattice corners throughout -- never as `lower corner + h`, which is not bitwise the
neighbour's lower corner -- so two cells sharing a face return the same point bit for bit. On a
fully covered face the point is the face's centre, a finite placeholder under zero weight.
"""
@inline function face_centroid(m::CutCellData{D,T}, direction::Integer,
                               nodes::SVector{N,SVector{D,T}}) where {D,T,N}
    ax = direction_axis(direction)
    lo = @inbounds nodes[1]
    plane = direction_sign(direction) > 0 ? @inbounds(_upper_corner(nodes)[ax]) : @inbounds(lo[ax])
    off = @inbounds m.face_centroid_local[direction]
    # In-plane axes in increasing order, skipping the normal: axis `a` is offset slot `a` below the
    # normal and `a - 1` above it -- the order the construction stored them in.
    return SVector{D,T}(ntuple(a -> a == ax ? plane : @inbounds(lo[a] + off[a < ax ? a : a - 1]),
                               Val(D)))
end

@inline face_centroid(m::CutCellData{2}, direction::Integer, mesh::AdaptiveMesh{2}, i::Integer) =
    face_centroid(m, direction, cell_nodes(mesh, i))

# The all-ones corner of a cell's corner array: `(1,1)` is corner 3 of `MS_NODE_BITS`, `(1,1,1)`
# corner 7 of `MC_NODE_BITS`. Its coordinates are the cell's upper lattice planes.
@inline _upper_corner(nodes::SVector{4}) = @inbounds nodes[3]
@inline _upper_corner(nodes::SVector{8}) = @inbounds nodes[7]

"""
    interface_area(m, cellsize)
    interface_normal(m, cellsize)

The interface facet's scalar area and outward-from-the-**body** unit normal, derived from
[`interface_normal_area`](@ref). `interface_normal` is exactly zero when the area is, rather than
`NaN` from a zero-length normalisation.

Derived rather than stored so they cannot disagree with `interface_normal_area`; use that wherever
the product of the two is what you want, since it is the exact quantity and this pair is not.

In an **ambiguous** cell (two interface segments) the vector area is their vector sum, so
`interface_area` is the length of that resultant and is *shorter* than the true interface length --
correct for conservation, wrong for a surface flux.
"""
@inline interface_area(m::CutCellData{D,T}, cellsize::SVector{D,T}) where {D,T} =
    sqrt(sum(abs2, interface_normal_area(m, cellsize)))

@inline function interface_normal(m::CutCellData{D,T}, cellsize::SVector{D,T}) where {D,T}
    v = interface_normal_area(m, cellsize)
    a = sqrt(sum(abs2, v))
    return a > zero(T) ? v / a : zero(SVector{D,T})
end

"""
    face_vector_area(m, direction, cellsize)

The vector area `A_k n_k` of one open Cartesian face: its open area times the outward unit normal.
Axis-aligned, so it is a signed scalar in one slot.
"""
@inline function face_vector_area(m::CutCellData{D,T}, direction::Integer,
                                  cellsize::SVector{D,T}) where {D,T}
    ax = direction_axis(direction)
    s = T(direction_sign(direction)) * face_area_of(m, direction, cellsize)
    return SVector{D,T}(ntuple(a -> a == ax ? s : zero(T), Val(D)))
end

"""
    interface_normal_area(m, cellsize)

The vector area `A_int n_int` of the interface facet: `-sum_k face_vector_area(m, k, cellsize)`,
formed per axis as `face_area_of(m, 2a, cellsize) - face_area_of(m, 2a - 1, cellsize)`. See this
file's header for why that is the definition rather than a coincidence.

Not stored: formed here from the face fractions with the expression the construction used, so it is
bit-identical to a stored copy, and [`closure_residual`](@ref), which sums the same `face_area_of`
products, is **exactly** zero rather than zero to roundoff. `cellsize` is this cell's own extent, as
for the open faces.
"""
@inline interface_normal_area(m::CutCellData{D,T}, cellsize::SVector{D,T}) where {D,T} =
    SVector{D,T}(ntuple(a -> face_area_of(m, 2a, cellsize) - face_area_of(m, 2a - 1, cellsize),
                        Val(D)))

"""
    full_face_area(cellsize, direction) -> T

The measure of a **fully open** Cartesian face normal to `direction`: in 2D the cell's length along
the *other* axis, in general the product of the sizes along every axis but `direction`'s. The scale
a [`CutCellData`](@ref) face fraction is multiplied by to get a physical area, and the divisor a
consumer holding an area divides by to get a fraction back.

Direction-aware, so it is right on a stretched cell where a single `grid.d[1]^(D-1)` would not be.
Unexported -- spell it `CutCellMethods.full_face_area`, since a consumer package is likely to have
its own grid-level function of that name.

Every open-face area -- and through them [`interface_normal_area`](@ref) -- is formed through this
one function, which is what keeps both sides of the per-cell closure the same expression.
"""
@inline full_face_area(cellsize::SVector{2,T}, direction::Integer) where {T} =
    @inbounds cellsize[3 - direction_axis(direction)]

@inline function full_face_area(cellsize::SVector{D,T}, direction::Integer) where {D,T}
    ax = direction_axis(direction)
    p = one(T)
    for a in 1:D
        a == ax || (p *= @inbounds cellsize[a])
    end
    return p
end

# =====================================
# The shared cut-cell accessor layer
#
# `face_area_of`/`is_cell_open` read one `CutCellData`, whichever method or cache it came from.
# `CutCellData` keeps `face_fraction` and forms an area on read: the fraction was built against
# each face's own full measure (`full_face_area` uses the in-plane cell size), so it is correct on
# a stretched cell.
#
# Everything here is per cell and takes the cell's own extent, which on an `AdaptiveMesh` varies by
# level -- which is why the scale is a per-call argument.

"""
    face_area_of(m::CutCellData, dir, cellsize) -> T

One direction's open measure out of a cell's moments, in raw physical units -- a length in 2D, an
area in 3D.

Formed as `face_fraction * full_face_area(cellsize, dir)` rather than read from storage -- see
[`CutCellData`](@ref) for why the fraction is the half that is kept. `cellsize` is the cell's own
extent: `grid.d` for every cell of a uniform grid, varying by level on an `AdaptiveMesh`.
"""
@inline face_area_of(m::CutCellData{D,T}, dir::Integer, cellsize::SVector{D,T}) where {D,T} =
    @inbounds m.face_fraction[dir] * full_face_area(cellsize, dir)

"""
    is_cell_open(m::CutCellData) -> Bool

Whether any of the cell's faces is open at all -- the connectivity question a solver asks before
fluxing through a cell, and deliberately **not** the geometric classification [`is_inside`](@ref)
gives. Answered on `face_fraction`, so it needs no cell size. A broadcastable scalar function:
`is_cell_open.(moments)` fuses into a surrounding broadcast with no temporary.
"""
@inline is_cell_open(m::CutCellData) = !all(iszero, m.face_fraction)

# =====================================
# Diagnostics over a field of cut cells

"""
    cut_cell_report(moments) -> NamedTuple

Counts of cells by kind, how many are ambiguous, and the smallest non-zero volume fraction.

That last one is the small-cell problem as a single number: for an explicit time-marching consumer
it caps the stable step, since that scales with the cell's outside volume.

`moments` is typically a cache's `cells`, on any backend: a device one is copied to the host first.
"""
function cut_cell_report(moments::AbstractArray{<:CutCellData})
    # `adapt`, not `Array`: a cache's `cells` is a `StructArray`, and `Array` of a device one
    # would gather it element by element through scalar indexing.
    mom = Adapt.adapt(Array, moments)
    T = eltype(eltype(mom))
    alpha_min = one(T)
    for m in mom
        if is_cut(m) && m.volume_fraction > zero(T)
            alpha_min = min(alpha_min, m.volume_fraction)
        end
    end
    return (n = length(mom),
            cut = count(is_cut, mom),
            inside = count(is_inside, mom),
            outside = count(is_outside, mom),
            ambiguous = count(is_ambiguous, mom),
            min_volume_fraction = alpha_min)
end

"""
    closure_residual(m, cellsize) -> SVector{D}

`sum_k A_k n_k - A_int n_int` for one cell: the discrete divergence theorem applied to its outside
region. **Must be exactly zero**, on every cell at every level.

The minus is the sign convention, not a correction: the face normals `n_k` point out of the outside
region while `n_int` points out of the *body*, which over the outside region is inward. See this
file's header.

`cellsize` is this cell's own extent, which turns the stored fractions into areas on both sides.
Exactness holds because both sides are the same [`face_area_of`](@ref) products:
[`interface_normal_area`](@ref) subtracts them per axis, and this sums them signed, which is the
same subtraction. Summed in direction order, so a consumer gathering its own face terms in the same
order reproduces it bit for bit.

Contracting it with a constant vector field gives the free-stream error of a finite volume scheme
built on these moments, which is why it has to be exactly zero rather than merely small.
"""
@inline function closure_residual(m::CutCellData{D,T,NF},
                                  cellsize::SVector{D,T}) where {D,T,NF}
    acc = zero(SVector{D,T})
    for dir in 1:NF
        acc += face_vector_area(m, dir, cellsize)
    end
    return acc - interface_normal_area(m, cellsize)
end

function Base.show(io::IO, m::CutCellData{D,T}) where {D,T}
    tag = is_inside(m) ? "inside" : is_outside(m) ? "outside" : "cut"
    print(io, "CutCellData{", D, ",", T, "}(", tag,
          m.ambiguous ? ", ambiguous" : "", ", alpha = ", m.volume_fraction, ")")
end

# Fractions and centroids only -- no areas, no interface normal, no closure line: all of them need
# this cell's size, which the moments do not carry. A consumer wanting them has the layout in hand
# and can call `face_area_of`, `interface_area` or `closure_residual`. Face centroids print as their
# stored in-plane offsets, for the same reason: the global point needs the cell's corners.
function Base.show(io::IO, ::MIME"text/plain", m::CutCellData{D,T,NF}) where {D,T,NF}
    tag = is_inside(m) ? "inside" : is_outside(m) ? "outside" : "cut"
    println(io, "CutCellData{", D, ",", T, "} -- ", tag,
            m.ambiguous ? " (AMBIGUOUS: multi-valued cut)" : "")
    println(io, "  volume frac : ", m.volume_fraction)
    println(io, "  centroid    : ", m.centroid)
    for dir in 1:NF
        println(io, "  face ", dir, "      : fraction ", m.face_fraction[dir],
                "  at in-plane offset ", m.face_centroid_local[dir])
    end
    print(io, "  interface   : at ", m.interface_centroid)
end
