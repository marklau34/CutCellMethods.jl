# =====================================
# What one cut cell knows about itself
#
# `CutCellData` is the output of a *single* geometric construction: volume fraction,
# outside-volume centroid, per-direction open-face fraction and centroid, and the interface facet's
# centroid -- all read off one reconstructed polygon rather than derived independently, with the
# interface's vector area implied by the face fractions rather than stored beside them. Deriving
# them independently is the classic free-stream preservation failure of a cut-cell finite volume
# scheme, so the struct exists partly to make that awkward. It is isbits and
# fixed-size, so it passes into a kernel by value and lives in a plain device array.
#
# ---------------------------------------------------------------------------
# The sign convention, stated once because getting it backwards is silent.
#
# The field is **negative inside the body** (`get_sdf`'s convention), so a cell's *inside* region is
# `{phi < 0}` and its *outside* region -- the one every moment here measures -- is `{phi >= 0}`.
# Note this is the opposite region to the one [`calc_volume`](@ref) measures. A consumer
# wanting moments of the inside should negate the field: the construction is a pure function of the
# corner values, so `x -> -phi(x)` gives the complement with no separate code path.
#
# **The open-face normals and the interface normal point in opposite senses, on purpose.** A face's
# `n_k` points out of the *outside region*, which is what a flux balance over that volume integrates
# against; the interface's `n_int` points out of the *body*, agreeing with the field's own normal,
# with `cell_interface`'s segment orientation, and with the winding of the contour `generate_mesh`
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
# The closure identity above, rearranged: `A_int n_int = sum_k A_k n_k`. For a *linear*
# reconstruction of the interface inside a cell that is an algebraic identity, not an approximation
# -- the closing edge of a polygon has exactly minus the vector area of all the others, and
# `n_int`'s opposite sense absorbs the minus. So the interface's vector area is not stored at all:
# `interface_normal_area` forms it on read from the stored face fractions, one subtraction of
# axis-aligned open-face areas per axis with no summation error, and `closure_residual` sums the very
# same `face_area_of` products. Both sides of the identity are one expression on the same stored
# numbers, which is what makes the closure hold to the last bit rather than merely to roundoff.
#
# `test/test_moments.jl` checks it against the interface segment's normal computed directly from its
# endpoints, which is the non-circular version of the claim.

"""
    CutCellData{D,T,NF,DF}

The complete geometric description of one cell's **outside** portion -- the part where `phi >= 0`,
outside the body. `NF == 2D` is the number of Cartesian faces, indexed by `CartesianMeshes`'
direction numbering (`1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`, ...), and `DF == D - 1` the number of
in-plane coordinates a face centroid is stored with. Both are fixed by `D` -- the concrete types are
`CutCellData{2,T,4,1}` and `CutCellData{3,T,6,2}` -- and are parameters only because a field type
cannot compute them.

- `kind` -- [`CELL_INSIDE`](@ref), `CELL_OUTSIDE` or `CELL_CUT`.
- `ambiguous` -- the reconstruction is **multi-valued** here: the marching-squares case is one of
  the two diagonal ones (in 3D, also a saddle face or an ambiguous cube interior), so the
  connectivity chosen is a guess. `interface_normal` is then a length-weighted lumped normal and
  must not be trusted for a surface flux; count these cells and refine them away rather than trust
  either resolution.
- `volume_fraction` -- the outside measure as a ratio of the whole cell.
- `centroid` -- the **outside-volume** centroid, not the Cartesian cell centre. A finite volume
  unknown placed at the Cartesian centre of a cell that is 20% outside is an `O(h)` error in exactly
  the cells where the geometry is doing the most work.
- `face_fraction` -- per direction, the open fraction of that Cartesian face.
- `face_centroid_local` -- per direction, the centroid of the face's open part, which is where a
  cut face's flux belongs, **in the face's own in-plane coordinates**: `DF` offsets from the cell's
  lower lattice corner along the face's in-plane axes, in increasing axis order. The coordinate
  along the face normal is not stored -- it is the face plane, which the lattice already knows.
  Read the point in global coordinates through [`face_centroid`](@ref) or [`face_rule`](@ref),
  which add the cell's position back.
- `interface_centroid` -- the centroid of the reconstructed interface. Its **vector area**
  `A_int n_int`, with `n_int` pointing **out of the body**, is [`interface_normal_area`](@ref),
  formed from the face fractions on read. That sense is opposite to the open faces'; see this
  file's header, and [`closure_residual`](@ref) for what it costs.

## The measures are fractions, and the caller supplies the scale

**Neither the outside volume nor any face's area is stored**, the interface's included. Each is a
fraction times a cell measure, so every dimensional reader multiplies at the point of use:
[`volume_rule`](@ref), [`face_rule`](@ref), [`face_area_of`](@ref), [`face_vector_area`](@ref), the
interface readers and [`closure_residual`](@ref) each take a trailing `cellsize` for exactly that.
It is the same expression the construction used, in the same order, so a value read back is
**bit-identical** to the one it was built from.

The fraction is the better half to keep: it is the primitive [`cell_boundary_walk`](@ref) produces,
it is what the hot consumers want (a flux weight and a Poisson coefficient are both dimensionless),
a relative tolerance against it needs no scale, and `iszero` on it answers [`is_cell_open`](@ref)
without one either.

The interface's vector area follows from the open faces by the closure identity, so
[`interface_normal_area`](@ref) re-forms it from the stored fractions -- one subtraction of
[`face_area_of`](@ref) products per axis, which is how the construction formed it -- and it comes
back bit-identical to what a stored copy would hold. Its scalar area and unit normal,
[`interface_area`](@ref) and [`interface_normal`](@ref), are two views of that one vector:
`interface_area * interface_normal` does not reproduce it bitwise (normalising and re-scaling is
two roundings), and that `O(eps)` gap is enough to break the closure, so use `interface_normal_area`
wherever the product is what you want.

## Face centroids are local, and the caller supplies the position

A face centroid is stored as offsets within the face, not as a point, for two reasons. It is `DF`
numbers rather than `D` -- the normal coordinate carries no information -- which is about a fifth
of the struct in either dimension. And an offset within a cell keeps full relative precision
wherever the cell sits, where a single-precision global coordinate near `|x| = 1000` resolves only
to about `6e-5`.

Reading one back needs the cell's corners, which [`face_centroid`](@ref) takes as the domain and
cell (or as the corner array itself). It adds the offsets to the lower lattice corner and takes the
normal coordinate from the lattice corner on the face's own side. Two cells sharing a face have
lower corners that differ only along that face's normal, so they add the same offsets to the same
in-plane coordinates and report the **same point bit for bit** -- the property a flux through the
face needs.

Read these through [`volume_rule`](@ref), [`face_rule`](@ref) and [`interface_rule`](@ref) where you
can: those are what a higher-order rule widens without touching the caller.
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

@inline Base.ndims(::CutCellData{D}) where {D} = D
@inline Base.eltype(::CutCellData{D,T}) where {D,T} = T
@inline Base.ndims(::Type{<:CutCellData{D}}) where {D} = D
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
    volume_rule(m, cellsize)

The one-point quadrature rule over the cell's outside portion: the outside centroid carrying the
outside volume `m.volume_fraction * prod(cellsize)`.

`cellsize` is this cell's own extent -- `grid.d` on a uniform grid, `get_elem_size(mesh, leaf)` on
an `AdaptiveMesh`, where it differs by level. See [`CutCellData`](@ref) for why the volume is
formed here rather than stored.
"""
@inline volume_rule(m::CutCellData{D,T}, cellsize::SVector{D,T}) where {D,T} =
    QuadratureRule(m.centroid, m.volume_fraction * prod(cellsize))

"""
    face_centroid(m, direction, grid::CartesianGrid, ci)
    face_centroid(m, direction, mesh::AdaptiveMesh, cell)
    face_centroid(m, direction, nodes)

The centroid of the open part of Cartesian face `direction`, in global coordinates: `m`'s stored
in-plane offsets added to the cell's lower lattice corner, with the coordinate along the face
normal taken from the lattice corner on the face's side. `m` must be the moments of that cell.

The domain forms take the cell the way [`cut_cell_moments`](@ref) does -- a `CartesianIndex` on a
grid, a `TreeCell` or leaf index on a tree. The `nodes` form takes the cell's corner coordinates in
[`MS_NODE_BITS`](@ref) or [`MC_NODE_BITS`](@ref) order, the array the construction itself took.

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

@inline face_centroid(m::CutCellData{D}, direction::Integer, grid::CartesianGrid{D},
                      ci::CartesianIndex{D}) where {D} =
    face_centroid(m, direction, cell_nodes(grid, ci))
@inline face_centroid(m::CutCellData{D}, direction::Integer, mesh::AdaptiveMesh{D},
                      c::TreeCell{D}) where {D} =
    face_centroid(m, direction, cell_nodes(mesh, c))
@inline face_centroid(m::CutCellData{D}, direction::Integer, mesh::AdaptiveMesh{D},
                      i::Integer) where {D} =
    face_centroid(m, direction, cell_nodes(mesh, i))

# The all-ones corner of a cell's corner array: `(1,1)` is corner 3 of `MS_NODE_BITS`, `(1,1,1)`
# corner 7 of `MC_NODE_BITS`. Its coordinates are the cell's upper lattice planes.
@inline _upper_corner(nodes::SVector{4}) = @inbounds nodes[3]
@inline _upper_corner(nodes::SVector{8}) = @inbounds nodes[7]

"""
    face_rule(m, direction, grid::CartesianGrid, ci)
    face_rule(m, direction, mesh::AdaptiveMesh, cell)
    face_rule(m, direction, nodes, cellsize)

The one-point quadrature rule over the open part of Cartesian face `direction`: the open-face
centroid [`face_centroid`](@ref) carrying the open area [`face_area_of`](@ref). Zero-weight on a
fully covered face.

The domain forms find the cell's corners and size themselves; the `nodes` form takes both, for a
caller holding a bare cell (see [`face_centroid`](@ref) for the corner order).
"""
@inline face_rule(m::CutCellData{D,T}, direction::Integer, nodes::SVector{N,SVector{D,T}},
                  cellsize::SVector{D,T}) where {D,T,N} =
    QuadratureRule(face_centroid(m, direction, nodes), face_area_of(m, direction, cellsize))

@inline face_rule(m::CutCellData{D}, direction::Integer, grid::CartesianGrid{D},
                  ci::CartesianIndex{D}) where {D} =
    face_rule(m, direction, cell_nodes(grid, ci), grid.d)
@inline face_rule(m::CutCellData{D}, direction::Integer, mesh::AdaptiveMesh{D},
                  c::TreeCell{D}) where {D} =
    face_rule(m, direction, cell_nodes(mesh, c), get_elem_size(mesh, c))
@inline face_rule(m::CutCellData{D}, direction::Integer, mesh::AdaptiveMesh{D},
                  i::Integer) where {D} =
    face_rule(m, direction, mesh, leaf(mesh, i))

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
    interface_rule(m, cellsize)

The one-point surface rule over the interface facet: its centroid carrying the interface area, with
the outward-from-the-body unit normal. Zero-weight in an uncut cell. A flux balance over the
outside region wants the opposite sense and should negate it.
"""
@inline interface_rule(m::CutCellData{D,T}, cellsize::SVector{D,T}) where {D,T} =
    SurfaceQuadratureRule(m.interface_centroid, interface_area(m, cellsize),
                          interface_normal(m, cellsize))

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
# `face_area_of`/`face_fraction_of`/`is_face_open`/`is_cell_open` are the names every cut-cell
# reconstruction answers to, so a consumer written against one reads the other. `plic/apertures.jl`
# carries the `PLICCutCellData` methods; these are the `CutCellData` half.
#
# The two store **different halves** of the same pair -- `CutCellData` keeps `face_fraction` and
# forms an area on read, `PLICCutCellData` keeps `face_area` and forms a fraction on read -- and this
# layer is where that stops mattering. Each is the primitive its own construction produced, so
# neither side stores a rounded round-trip of the other. Two things still genuinely differ, and both
# favour this side:
#
#  * the fraction here was built against each face's **own** full measure (`full_face_area` uses the
#    in-plane cell size), so it is correct on a stretched cell, where the PLIC reconstruction assumes
#    an isotropic grid -- which its own entry points require anyway.
#  * a shared face is single-valued by construction (both cells interpolate the same corner values),
#    where `cut_cell_moments(PLICCutCell(), ...)` hands back the un-averaged one-sided candidate and
#    leaves the resolution to the caller. These accessors cannot paper over that: read a one-sided
#    `PLICCutCellData` through `face_area_of` and you get the one-sided number.
#
# Everything here is per cell and takes the cell's own extent, which on an `AdaptiveMesh` varies by
# level.

# One cell's extent, from whatever the moments were laid out over. The `AdaptiveMesh` form is not
# constant across the domain, which is why the scale is a per-call argument.
@inline _cellsize_at(grid::CartesianGrid, ci) = grid.d
@inline _cellsize_at(mesh::AdaptiveMesh, i) = get_elem_size(mesh, leaf(mesh, i))

"""
    face_area_of(m::CutCellData, dir, cellsize) -> T

One direction's open measure out of a cell's moments, in raw physical units -- a length in 2D, an
area in 3D. The [`CutCellData`](@ref) counterpart of the [`PLICCutCellData`](@ref) method.

Formed as `face_fraction * full_face_area(cellsize, dir)` rather than read from storage -- see
[`CutCellData`](@ref) for why the fraction is the half that is kept. `cellsize` is the cell's own
extent: `grid.d` for every cell of a uniform grid, varying by level on an `AdaptiveMesh`. The
`PLICCutCellData` method takes it and ignores it, so one call site serves either reconstruction.
"""
@inline face_area_of(m::CutCellData{D,T}, dir::Integer, cellsize::SVector{D,T}) where {D,T} =
    @inbounds m.face_fraction[dir] * full_face_area(cellsize, dir)

"""
    face_fraction_of(m::CutCellData, dir, cellsize) -> T

One direction's open **fraction**, a dimensionless `[0,1]` -- what a flux weight or a Poisson
coefficient wants. Read straight out of `face_fraction`, which makes it **anisotropy-correct**: the
stored fraction was normalized by that face's own full measure.

`cellsize` is **ignored** here and needed by the [`PLICCutCellData`](@ref) method, exactly the
reverse of [`face_area_of`](@ref) -- between the two functions the caller never has to know which
side stores which half.
"""
@inline face_fraction_of(m::CutCellData{D,T}, dir::Integer, cellsize::SVector{D,T}) where {D,T} =
    @inbounds m.face_fraction[dir]

"""
    is_face_open(m::CutCellData, dir) -> Bool

Whether one named face of a cell is open at all, for **either** reconstruction and with **no cell
size**: `iszero` on the fraction and on the area agree exactly, since the scale between them is
never zero. The per-face counterpart of [`is_cell_open`](@ref).
"""
@inline is_face_open(m::CutCellData, dir::Integer) = @inbounds !iszero(m.face_fraction[dir])

"""
    is_cell_open(m::CutCellData) -> Bool

Whether any of the cell's faces is open at all -- the connectivity question a solver asks before
fluxing through a cell, and deliberately **not** the geometric classification [`is_inside`](@ref)
gives. Answered on `face_fraction`, so it needs no cell size. A broadcastable scalar function:
`is_cell_open.(moments)` fuses into a surrounding broadcast with no temporary.

The [`PLICCutCellData`](@ref) method answers the same question on `kind`; the two differ only for a
cut cell whose faces all happen to be closed.
"""
@inline is_cell_open(m::CutCellData) = !all(iszero, m.face_fraction)

# =====================================
# Diagnostics over a field of cut cells

@kernel function closure_residual_kernel!(out, mom, layout)
    i = @index(Global)
    if i <= length(mom)
        @inbounds out[i] = closure_residual(mom[i], _cellsize_at(layout, i))
    end
end

"""
    closure_residuals(moments, layout)

[`closure_residual`](@ref) for every cell of `moments`, as an array on the same backend.
**Every entry must be exactly zero**, at every level and on either backend.

It is zero by construction -- `cut_cell_moments` defines the interface's vector area as minus the
sum of the open faces' -- so this is a regression guard on that construction rather than a
measurement.
The non-circular check is comparing the interface normal against the reconstructed segment's own,
which `test/test_moments.jl` does separately.

`moments` is an array of [`CutCellData`](@ref), however it was filled. `layout` is the
`CartesianGrid` or `AdaptiveMesh` it was built over, indexed the same way; it supplies each cell's
size, which on an `AdaptiveMesh` differs by level, so passing a different domain measures nothing.
"""
function closure_residuals(moments::AbstractArray{<:CutCellData{D,T}}, layout) where {D,T}
    dev = KernelAbstractions.get_backend(moments)
    n = length(moments)
    out = KernelAbstractions.allocate(dev, SVector{D,T}, n)
    closure_residual_kernel!(dev, DEFAULT_WORKGROUP)(out, moments, layout; ndrange=n)
    KernelAbstractions.synchronize(dev) # as above: the caller reads `out`
    return out
end

"""
    cut_cell_report(moments) -> NamedTuple

Counts of cells by kind, how many are ambiguous, and the smallest non-zero volume fraction.

That last one is the small-cell problem as a single number: for an explicit time-marching consumer
it caps the stable step, since that scales with the cell's outside volume.
"""
function cut_cell_report(moments::AbstractArray{<:CutCellData})
    mom = Array(moments)
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
# and can call `face_area_of`, `interface_rule` or `closure_residual`. Face centroids print as their
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
