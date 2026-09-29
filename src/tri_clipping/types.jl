# =====================================
# What the tri-clipping cache records beyond `CutCellData`
#
# `cells` holds the shared `CutCellData` so every reader in this package works unchanged. What only
# this method knows -- how many patches met in a cell, which Boolean rule cut it, what went wrong,
# and how far the closure had to move the clipped interface -- lives in a parallel `info` array, and
# the interface split by patch lives in `bfaces`.

# At most this many patches (fit groups) cut one cell. With the six box faces that bounds a clipped
# polytope at 14 faces, and a simple polytope at 2F - 4 = 24 vertices, inside `ConvexPoly`'s limits.
const TRI_K_MAX = 8
# Local patch-pair labels recorded per cell: every pair of `TRI_K_MAX` patches, with room to spare.
const TRI_MAX_PAIRS = 32

# How two patches meet along an edge, as bits so a pair meeting several ways keeps all of them.
const PAIR_CONVEX  = 0x01   # the solid is the intersection of the two half-spaces
const PAIR_CONCAVE = 0x02   # the fluid is
const PAIR_TANGENT = 0x04   # one smooth surface split in two; fitted as one

# The Boolean rule a cut cell was clipped with.
const RULE_NONE     = 0x00  # uncut
const RULE_SINGLE   = 0x01  # one fit group: the fluid half-space of its plane
const RULE_CONVEX   = 0x02  # solid = intersection of the groups' solid half-spaces
const RULE_CONCAVE  = 0x03  # fluid = intersection of the groups' fluid half-spaces
const RULE_FALLBACK = 0x04  # unsupported: one combined plane, flagged
const RULE_MIXED    = 0x05  # convex and concave creases together: the planes' arrangement, labelled

# At most this many fit groups go through the mixed rule, whose arrangement has 2^k pieces. Six
# covers a hull's bow-stem tip, where six CAD surfaces can meet in one cell; a cell pays for the
# pieces of its own k only, so the cap costs the other cells nothing but scratch.
const TRI_K_MIXED = 6
const TRI_NLEAF = 1 << TRI_K_MIXED

# Diagnostic bits in `TriClipCellInfo.flags`.
const FLAG_MULTI_PATCH      = 0x01  # two or more fit groups: a crease
const FLAG_UNSUPPORTED      = 0x02  # no Boolean rule fits; cut by the fallback plane
const FLAG_SPLIT            = 0x04  # the solid crosses the cell and splits its fluid in two (exact; not `ambiguous`)
const FLAG_CLOSURE_FALLBACK = 0x08  # the clipped interface had no area to distribute the closure over
const FLAG_CORR_LARGE       = 0x10  # the closure correction flipped or dominated a patch's facet
const FLAG_OVERFLOW         = 0x20  # a fixed capacity was exceeded
const FLAG_CHAIN_FAIL       = 0x40  # a clip's cap face did not close into one loop
const FLAG_STATUS_CONFLICT  = 0x80  # an uncut cell disagrees with an uncut neighbour

"""
    TriClipCellInfo{T}

What [`TriClippingCutCell`](@ref)'s cache knows about a cell beyond its `CutCellData`, one per
cell in `cache.info`:

- `npatch` -- fit groups that cut the cell; `0` in an uncut cell.
- `rule` -- the Boolean rule it was clipped with (`RULE_SINGLE`, `RULE_CONVEX`, ...).
- `flags` -- diagnostic bits (`FLAG_MULTI_PATCH`, `FLAG_UNSUPPORTED`, `FLAG_SPLIT`, ...). Only
  `FLAG_UNSUPPORTED` makes the cell's `CutCellData` `ambiguous`: a fallback plane stood in for the
  surface there. A `FLAG_SPLIT` cell is exact -- its fluid is two pockets joined outside the cell,
  its interface two facets, each reported in `boundary_faces`.
- `correction` -- `|A_int - sum_p S_p|`: how far the closure moved the clipped interface to meet the
  resolved face fractions. `O(h^3)` on a curved patch, zero to roundoff on a planar one.
"""
struct TriClipCellInfo{T}
    npatch::Int8
    rule::UInt8
    flags::UInt8
    correction::T
end

"""
    TriBoundaryFace{T}

One patch's share of a cut cell's interface, in `cache.bfaces`:

- `patch` -- the patch, as an index into the cache's patch table (`0` in an unused slot).
- `area` -- its area.
- `normal` -- its unit normal, out of the body.
- `centroid` -- its centroid, in global coordinates.

`area * normal` summed over a cell's faces is that cell's `interface_normal_area` to roundoff.
"""
struct TriBoundaryFace{T}
    patch::Int32
    area::T
    normal::SVector{3,T}
    centroid::SVector{3,T}
end
