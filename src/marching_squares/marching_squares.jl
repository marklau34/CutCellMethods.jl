# =====================================
# Marching squares: the `sdf == 0` contour of a 2D field, one cell at a time.
#
# The reconstruction is **nodal / shared-vertex**: `phi` is sampled at a cell's *corners*, and each
# edge's zero crossing is interpolated between its two corner values. Two cells sharing a face read
# the same corner values at the same points, so they get the same crossing -- watertight by
# construction, with no communication, cache or consistency pass, which lets a consumer recompute one
# cell's geometry on demand at any level.
#
# A per-cell fit -- evaluate the distance at the cell centre and clip against the resulting line,
# what `PLICCutCell` (and [`calc_volume`](@ref)) does -- cannot offer that: the two fits are anchored
# `O(h)` apart, so cells flanking a face compute different apertures for it, and a flux leaving one
# through an opening of one size and entering the other through an opening of another manufactures
# mass at the face.
#
# ---------------------------------------------------------------------------
# Watertight *to the last bit*, which takes two deliberate choices.
#
# 1. An edge's open fraction is always anchored on its **outside** endpoint,
#    `s = phi_out / (phi_out - phi_in)`. Anchoring on the first endpoint in this cell's own
#    traversal order is algebraically the same number but a different floating-point expression,
#    and the two cells sharing an edge traverse it in opposite directions. Spent in
#    `_open_fraction` and `_open_segment`, further down this file.
#
# 2. Corner *coordinates* come from the integer lattice, `x0 + (index + bit) * h`, never from
#    arithmetic on a cell's own lower corner: cell A's `lo_A + h` is not bitwise cell B's `lo_B`. It
#    survives a level jump on an `AdaptiveMesh` too, since `h/2` is exact and `(2c + 2) * (h/2)`
#    rounds to the same value as `(c + 1) * h`.
#
# Neither costs anything, and together they let the contour be stitched by exact coordinate
# equality -- no merge tolerance anywhere in this directory.
#
# ---------------------------------------------------------------------------
# What is where.
#
# This file: the method tag, the corner conventions everything else is written against, the
# sampling layer (`cell_nodes`, `cell_values`, where choice 2 is spent), and the per-cell
# construction: `cell_boundary_walk` (per-edge open fractions and segments), `cell_gap_segments`
# (the interface segments bridging them) and `cell_interface` composing the two. `moments.jl`: the
# geometric moments, built on that same walk. `cache.jl`: every cell's moments over a grid or tree,
# kept between updates. `surface.jl`: the stitched contour, `generate_mesh`, walking this same
# `cell_interface` over a whole domain. `src/cut_cell.jl`: the `CutCellData` the moments come back
# as, with its accessors and diagnostics.
#
# Corner order is **counter-clockwise from the lower corner**, which is what the boundary walk
# further down needs:
#
#     4 ---3---- 3            edge 1: node 1 -> 2, lies on y = y_lo, direction 3
#     |          |            edge 2: node 2 -> 3, lies on x = x_hi, direction 2
#     4          2            edge 3: node 3 -> 4, lies on y = y_hi, direction 4
#     |          |            edge 4: node 4 -> 1, lies on x = x_lo, direction 1
#     1 ---1---- 2
#
# Note this is *not* `CartesianMeshes.element_nodes_global`'s ordering, which is the VTK_PIXEL bit
# pattern -- that one is not a boundary traversal, and walking it would trace a bow tie.

struct MarchingSquaresCutCell <: AbstractCutCellMethod
end

"""
    MS_NODE_BITS

The per-axis offsets of a cell's four corners, counter-clockwise from the lower corner:
`((0,0), (1,0), (1,1), (0,1))`. The order [`cell_nodes`](@ref), [`cell_values`](@ref) and
[`cell_boundary_walk`](@ref) use, and the order any array of four corner values handed to them must
be in.
"""
const MS_NODE_BITS = ((0, 0), (1, 0), (1, 1), (0, 1))

"""
    MS_EDGE_DIRECTION
    MS_DIRECTION_EDGE

The map between the four edges of the counter-clockwise walk and `CartesianMeshes`' face direction
numbering (`1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`), and its inverse: edge `k` lies on face direction
`MS_EDGE_DIRECTION[k]`, and face direction `dir` is edge `MS_DIRECTION_EDGE[dir]`.

Exposed because [`cell_boundary_walk`](@ref) reports per *edge* while flux bookkeeping wants per
*direction*.
"""
const MS_EDGE_DIRECTION = (3, 2, 4, 1)
const MS_DIRECTION_EDGE = (4, 2, 1, 3)

# =====================================
# Corner coordinates and corner values
#
# Both come from the **integer lattice** rather than from arithmetic on a cell's lower corner --
# choice 2 of this file's header, and the reason a corner shared by two cells is the same
# floating-point number for both.

"""
    cell_nodes(grid::CartesianGrid{2}, idx::CartesianIndex{2})
    cell_nodes(mesh::AdaptiveMesh{2}, cell::TreeCell{2})
    cell_nodes(mesh::AdaptiveMesh{2}, i::Integer)

The four corner coordinates of one cell -- cell `idx` of a grid, or `cell`/leaf `i` of an adaptive
mesh -- counter-clockwise from its lower corner ([`MS_NODE_BITS`](@ref) order).

Built off the integer lattice, `CartesianMeshes.get_node` on a grid and `x0 + (coord + bit) * h` on
a tree, so the shared corner of two neighbouring cells is one expression on one pair of integers
and the two land on bitwise the same point -- including across a level jump. See this file's
header.

Exposed because it is the contract worth testing directly.
"""
@inline function cell_nodes(grid::CartesianGrid{2,T}, idx::CartesianIndex{2}) where {T}
    return SVector{4,SVector{2,T}}(ntuple(Val(4)) do k
        b = MS_NODE_BITS[k]
        get_node(grid, CartesianIndex(idx[1] + b[1], idx[2] + b[2]))
    end)
end

@inline function cell_nodes(mesh::AdaptiveMesh{2,T}, c::TreeCell{2}) where {T}
    h = get_elem_size(mesh, c)
    x0 = mesh.base.x0
    return SVector{4,SVector{2,T}}(ntuple(Val(4)) do k
        b = MS_NODE_BITS[k]
        SVector{2,T}(x0[1] + T(c.coord[1] + b[1]) * h[1],
                     x0[2] + T(c.coord[2] + b[2]) * h[2])
    end)
end

@inline cell_nodes(mesh::AdaptiveMesh{2}, i::Integer) = cell_nodes(mesh, leaf(mesh, i))

"""
    cell_values(vals::AbstractMatrix, idx::CartesianIndex{2}, T)
    cell_values(vals::AbstractMatrix, grid::CartesianGrid{2}, idx::CartesianIndex{2})
    cell_values(field, mesh::AdaptiveMesh{2}, cell::TreeCell{2})

The four corner values of a cell, in the same [`MS_NODE_BITS`](@ref) order [`cell_nodes`](@ref)
returns.

From a nodal array, `vals` is indexed as a node block (`size(vals) == nodegrid_size(grid)`), so cell
`(i, j)` reads nodes `(i, j)`, `(i+1, j)`, `(i+1, j+1)`, `(i, j+1)`, converted to `T` -- or, given
the grid, to the grid's own element type. This is how a grid cache reads its own `phi`, which it
samples at `CartesianMeshes.get_node`, the expression `cell_nodes` places a corner with.

A tree has no single nodal block, so there the corners are sampled at `cell_nodes` through
SDFLibrary.jl's `sdf_value`: `field` may be an `AbstractSDFGeometry` or a callable `x -> phi`. That
samples one leaf without touching its neighbours and without allocating, which is what lets the
tree kernels run one thread per leaf.
"""
@inline function cell_values(vals::AbstractMatrix, idx::CartesianIndex{2}, ::Type{T}) where {T}
    i, j = Tuple(idx)
    return @inbounds SVector{4,T}(T(vals[i, j]), T(vals[i+1, j]),
                                  T(vals[i+1, j+1]), T(vals[i, j+1]))
end

@inline cell_values(vals::AbstractArray{<:Real,2}, ::CartesianGrid{2,T},
                    idx::CartesianIndex{2}) where {T<:Real} = cell_values(vals, idx, T)

@inline function cell_values(field, mesh::AdaptiveMesh{2,T}, c::TreeCell{2}) where {T}
    nodes = cell_nodes(mesh, c)
    return SVector{4,T}(ntuple(k -> T(sdf_value(field, @inbounds nodes[k])), Val(4)))
end

# =====================================
# The marching squares construction itself: corner values in, contour segments out.
#
# One cell at a time -- no domain, no field. Two steps, both exposed because both are useful alone:
#
#   1. `cell_boundary_walk` -- walk the four edges counter-clockwise and report, per edge, how much
#      of it is outside and where that open part runs.
#   2. `cell_gap_segments` -- the gaps between consecutive open parts *are* the interface.
#
# `cell_interface` is the two composed, what `generate_mesh` (`surface.jl`) stitches into a contour;
# `moments.jl` calls the two steps separately since it wants the apertures from the first as well as
# the segments from the second. Closing the outside polygon this way -- rather than fitting an
# interface line and intersecting it with the cell -- makes the contour and the apertures agree by
# construction instead of by luck.
#
# The interpolation is anchored on the **outside** endpoint of an edge, never the first endpoint in
# this cell's own traversal order (choice 1 of the watertightness argument above); `_open_fraction`
# and `_open_segment` below are where it is spent.
#
# Everything here is allocation-free, mutation-free and fixed-size, so it is safe to call inside a
# GPU kernel.

@inline _ms_next(k::Int) = k == 4 ? 1 : k + 1
@inline _ms_prev(k::Int) = k == 1 ? 4 : k - 1

"""
    _open_fraction(phi_a, phi_b) -> T

The outside fraction of an edge whose endpoints carry `phi_a` and `phi_b`, from linear
interpolation of `phi` along it.

Anchored on the outside endpoint (`phi_out / (phi_out - phi_in)`), so the expression does not depend
on which end the caller called `a` -- which is what makes two cells sharing an edge agree bitwise.
See this file's header.
"""
@inline function _open_fraction(phi_a::T, phi_b::T) where {T}
    fa = phi_a >= zero(T)
    fb = phi_b >= zero(T)
    fa && fb && return one(T)
    (!fa) && (!fb) && return zero(T)
    return fa ? phi_a / (phi_a - phi_b) : phi_b / (phi_b - phi_a)
end

"""
    _open_segment(pa, pb, phi_a, phi_b, frac) -> (SVector{2}, SVector{2})

The endpoints of the open (outside) part of the edge `pa -> pb`, in the traversal direction, given
the fraction [`_open_fraction`](@ref) already produced. The crossing is written as
`x_out + s * (x_in - x_out)`, anchored on the outside end for the same bitwise reason. Only ever
called on an edge with at least one outside endpoint.
"""
@inline function _open_segment(pa::SVector{2,T}, pb::SVector{2,T},
                               phi_a::T, phi_b::T, frac::T) where {T}
    if phi_a >= zero(T) && phi_b >= zero(T)
        return pa, pb
    elseif phi_a >= zero(T)
        return pa, pa + frac * (pb - pa)
    else
        return pb + frac * (pa - pb), pb
    end
end

"""
    cell_boundary_walk(nodes, phi) -> (fracs, present, seg_a, seg_b)

Walk one cell's four edges counter-clockwise and report, per edge `k`:

- `fracs[k]` -- the outside fraction of that edge, i.e. the aperture of the Cartesian face it lies
  on (`1` for a wholly outside edge, `0` for a wholly inside one). Edge `k` is face direction
  `MS_EDGE_DIRECTION[k]`.
- `present[k]` -- whether the edge has an open part at all.
- `seg_a[k] -> seg_b[k]` -- that open part's endpoints, in traversal order. Meaningless where
  `present[k]` is false.

`nodes` and `phi` are the corner coordinates and corner values in [`MS_NODE_BITS`](@ref) order.

This is the whole geometric content of marching squares in 2D: [`cell_interface`](@ref) turns it
into the contour, and the moment layer turns the same output into apertures and volumes. Both read
*this* function rather than reimplementing the outside-anchored interpolation, since a second copy
is exactly the kind of thing that drifts and quietly stops being watertight.

Allocation-free, branch-light and free of mutation, so it is safe to call inside a GPU kernel.
"""
@inline function cell_boundary_walk(nodes::SVector{4,SVector{2,T}}, phi::SVector{4,T}) where {T}
    fracs = SVector{4,T}(ntuple(k -> _open_fraction(@inbounds(phi[k]),
                                                    @inbounds(phi[_ms_next(k)])), Val(4)))
    present = SVector{4,Bool}(ntuple(k -> @inbounds(phi[k]) >= zero(T) ||
                                          @inbounds(phi[_ms_next(k)]) >= zero(T), Val(4)))
    segs = ntuple(k -> _open_segment(@inbounds(nodes[k]), @inbounds(nodes[_ms_next(k)]),
                                     @inbounds(phi[k]), @inbounds(phi[_ms_next(k)]),
                                     @inbounds(fracs[k])), Val(4))
    seg_a = SVector{4,SVector{2,T}}(ntuple(k -> @inbounds(segs[k][1]), Val(4)))
    seg_b = SVector{4,SVector{2,T}}(ntuple(k -> @inbounds(segs[k][2]), Val(4)))
    return fracs, present, seg_a, seg_b
end

# The next edge of the walk that has an open part, cyclically. With at most four edges this is a
# short unrolled scan rather than anything cleverer.
@inline function _next_present(present::SVector{4,Bool}, k::Int)
    j = _ms_next(k)
    @inbounds present[j] && return j
    j = _ms_next(j)
    @inbounds present[j] && return j
    j = _ms_next(j)
    @inbounds present[j] && return j
    return k
end

"""
    cell_gap_segments(present, seg_a, seg_b) -> (n, a, b)

The interface segments implied by a [`cell_boundary_walk`](@ref): `n` in `0:2` segments, the `s`-th
running from `a[s]` to `b[s]`.

These are the gaps between consecutive open parts of the cell boundary: where one open segment ends
and the next begins somewhere else, the reconstructed interface bridges them. Closing the outside
polygon this way -- rather than fitting the interface and intersecting it with the cell -- makes the
reconstruction agree with the apertures by construction.

`n` is at most 2, since sign changes around a four-corner walk are even and at most four. `n == 2` is
the ambiguous saddle: two opposite corners outside, and marching squares cannot tell which pair of
crossings connects. The answer here is naive gap-bridging (outside connected through the middle); a
caller that cares should count those cells and refine them away rather than trust either resolution.

**Orientation is the walk's, not the public one.** Each gap runs counter-clockwise around the
outside polygon, continuing the boundary traversal, since [`cut_cell_moments`](@ref) closes that
polygon with these segments and sums it with an orientation-sensitive shoelace. In that direction the
implied normal `SVector(v[2], -v[1])` points *into* the body, opposite [`interface_normal`](@ref) and
the stitched contour; [`cell_interface`](@ref) swaps the endpoints and is what a consumer wanting the
contour should call.

Fixed-size and mutation-free so it stays kernel-safe; a zero-length gap (consecutive open segments
meeting at a shared corner) is skipped.
"""
@inline function cell_gap_segments(present::SVector{4,Bool},
                                   seg_a::SVector{4,SVector{2,T}},
                                   seg_b::SVector{4,SVector{2,T}}) where {T}
    z = zero(SVector{2,T})
    n = 0
    a1 = z; b1 = z; a2 = z; b2 = z
    for k in 1:4
        if @inbounds present[k]
            j = _next_present(present, k)
            # Counter-clockwise, continuing the boundary walk: from where edge `k`'s open part
            # ended to where the next one begins. **Do not reverse these** -- `cut_cell_moments`
            # feeds them into an orientation-sensitive shoelace sum. The public contour uses the
            # opposite orientation, and `cell_interface` is where the two meet.
            a = @inbounds seg_b[k]
            b = @inbounds seg_a[j]
            if a != b
                n += 1
                if n == 1
                    a1 = a; b1 = b
                elseif n == 2
                    a2 = a; b2 = b
                end
            end
        end
    end
    return min(n, 2), SVector{2,SVector{2,T}}(a1, a2), SVector{2,SVector{2,T}}(b1, b2)
end

"""
    cell_interface(nodes, phi) -> (n, a, b)

The reconstructed zero contour inside one cell: `n` segments (`0`, `1`, or `2`), the `s`-th running
from `a[s]` to `b[s]`.

    n, a, b = cell_interface(nodes, phi)
    for s in 1:n
        # segment from a[s] to b[s]
    end

The per-cell form of what [`generate_mesh`](@ref) stitches into a whole mesh; reach for it to
reconstruct one cell without building anything.

Endpoints are exact across cells: a segment reaching a cell face ends at exactly the point where the
neighbouring cell's segment begins, bit for bit, so the contour is a genuinely closed polyline,
joinable with no merge tolerance. Two segments failing to meet means a broken construction rather
than a floating-point near-miss.

Segments are oriented so that `SVector(v[2], -v[1])`, with `v = b - a`, points **out of the body**
-- the same sense the field's own normal points, [`interface_normal`](@ref) reports, and the
stitched contour winds. A flux balance over the outside region wants the opposite and negates it at
the point of use.
"""
@inline function cell_interface(nodes::SVector{4,SVector{2,T}}, phi::SVector{4,T}) where {T}
    _, present, seg_a, seg_b = cell_boundary_walk(nodes, phi)
    n, a, b = cell_gap_segments(present, seg_a, seg_b)
    # Endpoints swapped, which is the whole of the orientation convention: `cell_gap_segments` runs
    # counter-clockwise around the outside polygon because `cut_cell_moments`' shoelace needs it to,
    # and in that orientation the implied normal points into the body, where everything public --
    # this function, `interface_normal`, the stitched contour -- points out of it.
    return n, b, a
end
