# =====================================
# What the polyline-clipping cache records beyond `CutCellData`
#
# `cells` holds the shared `CutCellData` -- the cell's totals -- so every reader in this package
# works unchanged. What only this method knows lives beside it: a per-cell `info`, dense per-edge
# apertures, the regions of the cells a thin body splits, the fluid arcs each region owns on the
# cell's edges, and the boundary segments.

# A cell's status. `kind` in `CutCellData` stays INSIDE / OUTSIDE / CUT; this is the finer split.
const PL_FLUID   = 0x00   # uncut, all fluid
const PL_SOLID   = 0x01   # uncut, all solid
const PL_CUT     = 0x02   # cut, one fluid region
const PL_SPLIT   = 0x03   # cut, two or more fluid regions
const PL_INVALID = 0x04   # the walk failed; the cell record is the walk-free totals

# Diagnostic bits in `PolylineCellInfo.flags`.
const PL_FLAG_OVERFLOW          = 0x01  # more crossings on the cell's edges than a walk tracks
const PL_FLAG_WALK_FAIL         = 0x02  # the walk did not use every crossing exactly once
const PL_FLAG_STATUS_CONFLICT   = 0x04  # the x- and y-line walks disagree at one of its nodes
const PL_FLAG_ISLAND            = 0x08  # a whole loop lies inside the cell
const PL_FLAG_SNAPPED           = 0x10  # a sliver's open edge was closed here
const PL_FLAG_CLOSURE           = 0x20  # a region's closure over its segments exceeds closure_tol
const PL_FLAG_APERTURE_MISMATCH = 0x40  # the regions' arcs do not sum to the edge apertures
const PL_FLAG_WALK_MISMATCH     = 0x80  # the regions' areas do not sum to the walk-free total

# The most crossings one cell's four edges may carry: the walk marks the ones it has used in the
# bits of one `UInt64`.
const PL_MAX_CROSS = 64

"""
    PolylineCellInfo{T}

What [`PolylineClippingCutCell`](@ref)'s cache knows about a cell beyond its `CutCellData`:

- `status` -- `PL_FLUID`, `PL_SOLID`, `PL_CUT`, `PL_SPLIT` or `PL_INVALID`.
- `nregion` -- fluid regions in the cell: `0` solid, `1` fluid or singly cut, `2+` split.
- `flags` -- diagnostic bits (`PL_FLAG_*`).
- `slot` -- the cell's index in this update's cut list, `0` if uncut.
- `rslot` -- the first of its region records in `cache.regions`, `0` unless it is split.
- `residual` -- a singly cut cell's closure over its actual boundary segments.
"""
struct PolylineCellInfo{T}
    status::UInt8
    nregion::Int16
    flags::UInt8
    slot::Int32
    rslot::Int32
    residual::SVector{2,T}
end

"""
    PolylineEdge{T}

One grid edge's aperture: its open `fraction` and the centroid of its open part, as an `offset`
along the edge from its lower node. Computed once per edge and copied into both cells sharing it.
"""
struct PolylineEdge{T}
    fraction::T
    offset::T
end

"""
    PolylineRegionInfo{T}

What goes with one region record of a split cell, beside its `CutCellData` in `cache.regions`:

- `cell` -- the cell it is a region of, and `region` its id there (walk order from the cell's lower
  left corner).
- `wall` -- `sum_s n_s L_s` over its actual boundary segments, normals into the fluid.
- `residual` -- `sum_e n_e l_e - wall`: its closure, at roundoff for an exact polygon.
- `flags` -- diagnostic bits.
"""
struct PolylineRegionInfo{T}
    cell::CartesianIndex{2}
    region::Int16
    wall::SVector{2,T}
    residual::SVector{2,T}
    flags::UInt8
end

"""
    PolylineArc{T}

One fluid interval of one of a cut cell's edges, and the region that owns it:

- `dir` -- the edge, by direction (`1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`); `k` -- which fluid
  interval along it, counted from its lower node. The neighbour across the edge counts the same
  intervals in the same order, which is how `nbr_region` is found.
- `s0`, `s1` -- the interval, as offsets along the edge from its lower node.
- `region` -- the region owning it in this cell; `nbr_region` -- in the neighbour.
- `closed` -- the interval was a sliver's and has been closed on both sides.
"""
struct PolylineArc{T}
    region::Int16
    dir::Int8
    k::Int16
    s0::T
    s1::T
    nbr_region::Int16
    closed::Bool
end

# The walk's working sums for one region of one cut cell, in the cell's local frame: twice its
# area, `sum (p + q) (p x q)` over its boundary, and its rank among the cell's kept regions (`0` for
# a sliver, dropped).
struct PolylineRegionAcc{T}
    area2::T
    mom::SVector{2,T}
    rank::Int16
end

# What the walk hands on for one cut cell: the walk-free area sums, how many regions it walked and
# kept, its flags, and whether it succeeded.
struct PolylineSlotResult{T}
    area2::T
    mom::SVector{2,T}
    nraw::Int16
    nregion::Int16
    flags::UInt8
    ok::Bool
end

"""
    PolylineBoundarySeg{T}

The piece of one `Line` element inside one cut cell:

- `region` -- the region whose wall it is; `loop` -- the polyline; `elem` -- the mesh element.
- `length`, `normal` (unit, into the fluid), `midpoint` (global).
- `has_vertex` -- the piece ends at a mesh vertex inside the cell rather than at crossings only.
"""
struct PolylineBoundarySeg{T}
    region::Int16
    loop::Int32
    elem::Int32
    length::T
    normal::SVector{2,T}
    midpoint::SVector{2,T}
    has_vertex::Bool
end
