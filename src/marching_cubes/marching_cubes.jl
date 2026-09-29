# =====================================
# Marching cubes: the `sdf == 0` surface of a 3D field, one cell at a time.
#
# This file holds the `MarchingCubesCutCell` tag, the per-cell corner/edge/face conventions, and the
# per-cell Lewiner dispatch; `moments.jl` builds cut-cell moments on them, and `surface.jl` marches
# `MarchingCubes.jl` over the same nodal samples for the whole-grid surface.
#
# =====================================
# Nodal (shared-vertex) reconstruction
#
# The distance is sampled once per grid **node**; each surface vertex is placed on the grid **edge**
# between two nodes of opposite sign, by linear interpolation of the two nodal values. Every cell
# around that edge reads the same two numbers, so the surface is watertight and the apertures two
# cells compute for a shared face are bitwise equal.
#
# Contrast `PLICCutCell`'s per-cell tangent plane through a cell centroid: exact for a plane and
# needs no neighbour, but adjacent facets fit independently and don't meet at a shared face, so the
# surface is a field of disconnected shards. That is why a field's normal is never read here.
#
# Linear interpolation along an edge is exact where the surface is planar and second-order where it
# curves. Measured on a unit sphere at `h = 0.1`: every vertex sits within `1.1e-3` of the true
# surface.

"""
    MarchingCubesCutCell()

The nodal reconstruction of a 3D field's `sdf == 0` surface by marching cubes over the field's
values at the grid's nodes. A stateless tag, one of the `AbstractCutCellMethod`s:

    method = MarchingCubesCutCell()

    cache = update_cache!(allocate_cache(grid, method), geo, grid)
    cache.cells                                       # every cell's moments
    surface = generate_mesh(cache, grid)              # the isosurface, as a mesh

The 2D counterpart is [`MarchingSquaresCutCell`](@ref), the same shared-vertex construction one
dimension down.

The cache builds each cell from its own eight corner values with [`cut_cell_moments`](@ref)'s 3D
primitive, needing no neighbour or previous pass; the surface marches the same corner values.

Axes may differ from each other: nothing here converts a distance into unit-cell coordinates, so
there is no isotropy requirement, though isotropic cells give the best-shaped facets.

Case resolution is `MarchingCubes.jl`'s Lewiner et al. (2003) variant, which resolves ambiguous cube
configurations consistently -- what makes the watertightness above hold.

Sign conventions line up with no flipping: the field is negative inside, and feeding that straight
in winds every facet so its normal points **outwards**. A node landing exactly on the surface is
nudged to `+eps` before classification ([`nudge_zeros`](@ref)), so an exact zero counts as outside
and never produces a degenerate facet.
"""
struct MarchingCubesCutCell <: AbstractCutCellMethod
end

# =====================================
# The 3D corner, edge and face conventions everything else in this directory is written against.
#
# `marching_squares.jl`'s sampling layer one dimension up: corner *coordinates* and corner *values*
# of one cell, built so a corner shared by two cells is the same floating-point number for both. Read
# that file's header for the watertightness argument.
#
# ---------------------------------------------------------------------------
# The corner order is `MarchingCubes.jl`'s and must stay so.
#
# `MC_NODE_BITS` is exactly the order `MarchingCubes.lut_entry` classifies corners in, so the case
# index this package computes indexes that package's Lewiner tables directly. Reordering it would
# silently select the wrong tiling for every ambiguous cube. `test/test_mc_moments.jl` pins the order
# against `lut_entry` itself.
#
# The bottom four corners of that order are `MS_NODE_BITS`, so the `z = 0` face of a cube reads as a
# marching-squares cell unpermuted.
#
# ---------------------------------------------------------------------------
# Choice 3 of the watertightness argument, new in 3D.
#
# A shared face has **four** corners and an orientation-sensitive shoelace sum over them, so the
# corner order of a face has to be a property of the *lattice* rather than of whichever cell is
# asking:
#
#   3. A face's corners are listed with its two in-plane axes in increasing order (`p < q`, skipping
#      the face normal's axis), starting at the face's `(0,0)` corner, in `MS_NODE_BITS` order from
#      there. `MC_FACE_NODES` is that rule tabulated. Cell A's `+x` face and cell B's `-x` face are
#      then the same four lattice nodes in the same sequence, so their shoelaces are the same
#      additions in the same order.
#
# Corollary (the 3D instance of choice 2): a face's coordinate *along its own normal* must be read
# off one of that face's corners, never formed as `origin[ax] + cellsize[ax]`, which is not bitwise
# the neighbour's `origin[ax]`. `moments.jl` takes it from `nodes[...][ax]`.

"""
    MC_NODE_BITS

The per-axis offsets of a cell's eight corners: counter-clockwise around the `z = 0` face, then
counter-clockwise around the `z = 1` face.

    ((0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1))

**`MarchingCubes.jl`'s own corner numbering**, which lets [`cell_case`](@ref) index its Lewiner
tables directly. The first four entries are [`MS_NODE_BITS`](@ref), so a cube's bottom face reads as
a marching-squares cell unpermuted.

The order [`cell_nodes`](@ref) and [`cell_values`](@ref) return, and that any array of eight corner
values handed to this directory must be in.
"""
const MC_NODE_BITS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))

# The corner index carrying a given bit pattern -- the inverse of `MC_NODE_BITS`. Used only to build
# the tables below at load time, so the linear scan costs nothing and the tables stay derived from
# the one order rather than hand-transcribed beside it.
function _mc_node_index(b::NTuple{3,Int})
    for k in 1:8
        MC_NODE_BITS[k] === b && return k
    end
    error("no cube corner at bits $b")
end

"""
    MC_EDGE_NODES

The twelve cube edges as pairs of [`MC_NODE_BITS`](@ref) corner indices, in `MarchingCubes.jl`'s
edge numbering -- the numbering its tiling tables emit:

| edges | direction | |
|:------|:----------|:--|
| 1-4   | around `z = 0` | `(1,2) (2,3) (4,3) (1,4)` |
| 5-8   | around `z = 1` | `(5,6) (6,7) (8,7) (5,8)` |
| 9-12  | the four verticals | `(1,5) (2,6) (3,7) (4,8)` |

Note the tilings emit these codes **1-based**, with `13` meaning the interior vertex (slot 13 of
[`edge_crossings`](@ref)), while `MarchingCubes.jl`'s *test* tables spell edges **0-based**.
The two bases sit a few lines apart in this file's Lewiner dispatch; they are not interchangeable.
"""
const MC_EDGE_NODES = ((1, 2), (2, 3), (4, 3), (1, 4),
                       (5, 6), (6, 7), (8, 7), (5, 8),
                       (1, 5), (2, 6), (3, 7), (4, 8))

"""
    MC_FACE_AXES

Per face direction (`1 = -x`, `2 = +x`, `3 = -y`, ...), the two in-plane axes `(p, q)` with
`p < q`, skipping the axis the face is normal to. Increasing order is choice 3 of the
watertightness argument -- see the corner-convention notes above [`MC_NODE_BITS`](@ref).
"""
const MC_FACE_AXES = ntuple(6) do dir
    ax = direction_axis(dir)
    Tuple(a for a in 1:3 if a != ax)
end

"""
    MC_FACE_NODES

Per face direction, the four [`MC_NODE_BITS`](@ref) corner indices of that face, ordered by choice 3
of the watertightness argument: in-plane axes `(p, q)` increasing, `MS_NODE_BITS` pattern in those
two, starting at the face's `(0,0)` corner.

    dir 1 (-x): (1, 4, 8, 5)      dir 4 (+y): (4, 3, 7, 8)
    dir 2 (+x): (2, 3, 7, 6)      dir 5 (-z): (1, 2, 3, 4)
    dir 3 (-y): (1, 2, 6, 5)      dir 6 (+z): (5, 6, 7, 8)

Derived from [`MC_NODE_BITS`](@ref) rather than transcribed, so the two cannot drift. The `-z` face
comes out as `(1,2,3,4)` -- the bottom four corners in [`MS_NODE_BITS`](@ref) order, which is the
consistency this ordering was chosen to preserve.

Feeding these four to [`cell_boundary_walk`](@ref) in the face's own `(p, q)` frame is the whole of
the 3D face clip; `moments.jl` does exactly that, six times.
"""
const MC_FACE_NODES = ntuple(6) do dir
    ax = direction_axis(dir)
    b_ax = direction_sign(dir) > 0 ? 1 : 0
    # Only the lower in-plane axis needs naming: an axis that is neither `ax` nor `p` is `q` by
    # elimination, and takes the second bit.
    p = first(MC_FACE_AXES[dir])
    ntuple(4) do k
        bp, bq = MS_NODE_BITS[k]
        bits = ntuple(a -> a == ax ? b_ax : (a == p ? bp : bq), 3)
        _mc_node_index(bits)
    end
end

# =====================================
# Corner coordinates and corner values
#
# Built from the **integer lattice**, which is choice 2 of `marching_squares.jl`'s watertightness
# argument. Identical in form to the 2D methods next door, with eight corners instead of four.

"""
    cell_nodes(grid::CartesianGrid{3}, idx::CartesianIndex{3})

The eight corner coordinates of one 3D cell, in [`MC_NODE_BITS`](@ref) order. The 3D method of the
name [`cell_nodes(grid::CartesianGrid{2}, idx)`](@ref) carries in 2D, built the same way off the
integer lattice, so the shared corner of two neighbouring cells is bitwise the same point.
"""
@inline function cell_nodes(grid::CartesianGrid{3,T}, idx::CartesianIndex{3}) where {T}
    return SVector{8,SVector{3,T}}(ntuple(Val(8)) do k
        b = MC_NODE_BITS[k]
        get_node(grid, CartesianIndex(idx[1] + b[1], idx[2] + b[2], idx[3] + b[3]))
    end)
end

"""
    cell_values(vals::AbstractArray{<:Real,3}, idx::CartesianIndex{3}, T)
    cell_values(vals::AbstractArray{<:Real,3}, grid::CartesianGrid{3}, idx::CartesianIndex{3})

The eight corner values of a 3D cell read out of the nodal block `vals`, in the same
[`MC_NODE_BITS`](@ref) order [`cell_nodes`](@ref) returns -- the 3D counterparts of the 2D methods
next door.

These are *raw* corner values. The degenerate-value nudge the reconstruction needs is applied once
inside [`cut_cell_moments`](@ref) rather than here -- see [`nudge_zeros`](@ref).
"""
@inline function cell_values(vals::AbstractArray{<:Real,3}, idx::CartesianIndex{3},
                             ::Type{T}) where {T}
    i, j, k = Tuple(idx)
    return @inbounds SVector{8,T}(T(vals[i, j, k]), T(vals[i+1, j, k]),
                                  T(vals[i+1, j+1, k]), T(vals[i, j+1, k]),
                                  T(vals[i, j, k+1]), T(vals[i+1, j, k+1]),
                                  T(vals[i+1, j+1, k+1]), T(vals[i, j+1, k+1]))
end

@inline cell_values(vals::AbstractArray{<:Real,3}, ::CartesianGrid{3,T},
                    idx::CartesianIndex{3}) where {T<:Real} = cell_values(vals, idx, T)

"""
    nudge_zeros(phi) -> SVector

`phi` with every value closer to zero than `eps(T)` replaced by `+eps(T)`.

Needed because a corner sitting exactly on the surface otherwise produces degenerate zero-length
edges and zero-area facets, and the Lewiner ambiguity tests divide by differences such a corner can
make exactly zero. Nudging *outward* agrees with this package's `phi >= 0 is outside` convention.

Applied once, at the top of [`cut_cell_moments`](@ref), then used for the case index, the six face
clips and every edge crossing alike -- applying it inconsistently would tear the reconstruction.

A **pure function of each single value**: two cells sharing a corner nudge it identically without
consulting each other.
"""
@inline function nudge_zeros(phi::SVector{N,T}) where {N,T}
    e = eps(T)
    return SVector{N,T}(ntuple(k -> (@inbounds(abs(phi[k])) < e ? e : @inbounds(phi[k])), Val(N)))
end

# =====================================
# The marching cubes construction, one cell at a time: corner values in, interface triangles out.
#
# The counterpart of the marching-squares walk one dimension up: no domain, no field, just eight
# corner values and the facets they imply. Unlike 2D, the crossings on the cube's twelve edges do not
# determine how the facets connect -- that is the marching cubes case problem, resolved by the
# Lewiner tables.
#
# ---------------------------------------------------------------------------
# Whose tables, and why not our own
#
# Case resolution is `MarchingCubes.jl`'s -- Lewiner et al. (2003) -- read **per cell** rather than
# through its whole-grid `march`: one cell's facets are rebuilt from its own eight corner values
# alone, with no grid, neighbour or previous pass, which is what the moment layer needs.
#
# Reading that package's tables rather than writing our own decider is deliberate: get the face test
# wrong and neighbouring cells disagree about which pair of crossings connects, tearing the surface;
# get the interior test wrong and a cube grows a spurious handle. `test/test_mc_moments.jl` pins this
# dispatch against `MarchingCubes.march` on random fields.
#
# ---------------------------------------------------------------------------
# The one thing *not* taken from that package: where a vertex sits on its edge
#
# `MarchingCubes.jl` anchors edge interpolation on the edge's low node; this package anchors on the
# **outside** node (choice 1 of `marching_squares.jl`'s watertightness argument). Both are
# neighbour-consistent and watertight, differing by an ulp.
#
# It has to be ours because `moments.jl` closes a polyhedron out of six face clips (from
# `cell_boundary_walk`, outside-anchored) and these triangles, and integrates over it -- mixing the
# two anchorings would leave `O(eps)` gaps along every cut edge. `_edge_crossing` reproduces
# `_open_segment`'s crossing bit for bit.
#
# The case index and every ambiguity test read *signs* and corner values, never positions.

# One cube edge's crossing point, bitwise equal to the one `_open_segment` places on that edge.
#
# Both anchor on the outside end, so two cells sharing an edge agree. The expression is symmetric
# under swapping the endpoints -- the outside end is always the base -- so it doesn't matter which
# corner is called `a`, and `MC_EDGE_NODES`'s orientation carries no arithmetic weight.
@inline function _edge_crossing(pa::SVector{3,T}, pb::SVector{3,T}, fa::T, fb::T) where {T}
    s = _open_fraction(fa, fb)
    return fa >= zero(T) ? pa + s * (pb - pa) : pb + s * (pa - pb)
end

"""
    edge_crossings(nodes, phi) -> (crossed, points)

Where the reconstructed surface meets each of the cube's twelve edges.

- `crossed[e]` -- whether edge `e` ([`MC_EDGE_NODES`](@ref)) has a sign change at all.
- `points[e]` -- the crossing on edge `e`, meaningless where `crossed[e]` is false.
- `points[13]` -- the cube's **interior vertex**, the mean of every crossing on the cube, which is
  where `MarchingCubes.jl` puts the extra vertex cases 6.1.2, 7.3, 10.2, 12.2, 13.3 and 13.4 need.
  A tiling's edge code, `13` included, therefore indexes this vector directly.

Thirteen slots rather than twelve plus a flag: the interior vertex costs a dozen adds on a cell that
is already being reconstructed, and paying that unconditionally removes a branch from every consumer.

Crossings are placed so that they are bitwise the points [`cell_boundary_walk`](@ref) puts on the
same edges, which is what lets `moments.jl` close a polyhedron out of the two together.
"""
@inline function edge_crossings(nodes::SVector{8,SVector{3,T}}, phi::SVector{8,T}) where {T}
    crossed = SVector{12,Bool}(ntuple(Val(12)) do e
        a, b = MC_EDGE_NODES[e]
        @inbounds (phi[a] >= zero(T)) != (phi[b] >= zero(T))
    end)
    pts = SVector{12,SVector{3,T}}(ntuple(Val(12)) do e
        a, b = MC_EDGE_NODES[e]
        @inbounds crossed[e] ? _edge_crossing(nodes[a], nodes[b], phi[a], phi[b]) :
                               zero(SVector{3,T})
    end)
    # The interior vertex, averaged over the crossings that exist. `n == 0` cannot reach a tiling
    # that asks for it (an uncut cube emits no triangles), but the guard keeps it a point rather
    # than a NaN for a caller that prints it.
    acc = zero(SVector{3,T})
    n = 0
    for e in 1:12
        if @inbounds crossed[e]
            acc += @inbounds pts[e]
            n += 1
        end
    end
    interior = n > 0 ? acc / T(n) : zero(SVector{3,T})
    return crossed, SVector{13,SVector{3,T}}(ntuple(e -> e <= 12 ? @inbounds(pts[e]) : interior,
                                                    Val(13)))
end

"""
    cell_case(phi) -> Int

The Lewiner lookup index of a cube with corner values `phi`, in `1:256`: bit `p` (zero-based) is set
when corner `p + 1` is **outside** the body, in [`MC_NODE_BITS`](@ref) order, plus one. Exactly what
`MarchingCubes.lut_entry` computes, so it indexes `MarchingCubes.cases` and every tiling table
directly.

`phi` is expected to have been through [`nudge_zeros`](@ref) already, which is what makes `>= 0` and
`> 0` the same test here; [`cut_cell_moments`](@ref) does that once, at the top.
"""
@inline function cell_case(phi::SVector{8,T}) where {T}
    lut = 0
    for p in 0:7
        @inbounds phi[p+1] >= zero(T) && (lut |= (1 << p))
    end
    return lut + 1
end

# =====================================
# The two ambiguity tests, ported from `MarchingCubes.jl` onto `SVector` corner values
#
# Ported rather than called: that package's versions take an `MVector{8}` scratch buffer off an `MC`
# object, a whole-grid structure this file doesn't have. Arithmetic is transcribed unchanged;
# `test/test_mc_moments.jl` compares the resulting triangulation against `march` case by case.
#
# Edge indices in the *test* tables are 0-based; edge codes in the *tiling* tables are 1-based. Both
# appear in this file, a few lines apart.

"""
    MC_TEST_FACE_QUADS

The six cube faces as corner quadruples in `MarchingCubes.jl`'s *face* numbering -- the numbering
its `test3`/`test6`/`test7`/`test10`/`test12`/`test13` tables name faces in, which is its own and
**not** `CartesianMeshes`' direction numbering. Each quadruple is one face's four corners in cyclic
order, as `A, B, C, D` of the asymptotic decider below.

Kept as a table so [`_test_face`](@ref) and [`MC_FACE_TEST_CODE`](@ref) read the same one: the
mapping between the two face numberings is derived from it rather than transcribed beside it.
"""
const MC_TEST_FACE_QUADS = ((1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4),
                            (4, 8, 5, 1), (1, 4, 3, 2), (5, 8, 7, 6))

"""
    MC_FACE_TEST_CODE

Per `CartesianMeshes` face direction (`1 = -x`, `2 = +x`, ...), the `MarchingCubes.jl` face code
naming that same face -- `(4, 2, 1, 3, 5, 6)`, derived from [`MC_TEST_FACE_QUADS`](@ref) rather
than written down.

Needed because `moments.jl` resolves an ambiguous face with the same decider the Lewiner tables
use, and has to ask about the face by *its* name. See that file's header.
"""
const MC_FACE_TEST_CODE = ntuple(6) do dir
    want = Set(MC_FACE_NODES[dir])
    something(findfirst(q -> Set(q) == want, MC_TEST_FACE_QUADS))
end

# Whether the two components meeting at ambiguous face `face` connect through it -- the asymptotic
# decider, reading the sign of the bilinear saddle value on that face.
#
# `face` is signed: a negative code inverts the result, how the tables spell "this configuration
# wants the complementary answer".
#
# **Reads only that face's four corner values**, so `moments.jl` can safely use it to resolve a
# shared face: the two cells touching a face give it the same four numbers, spelled with different
# codes (cell A's `+x` is cell B's `-x`) that order the quadruple oppositely -- so `A*C - B*D` and
# `A` both flip sign, but on a saddle the two flips cancel and the answer is the same.
# `test/test_mc_moments.jl` asserts that agreement directly.
@inline function _test_face(cb::SVector{8,T}, face::Integer) where {T}
    q = @inbounds MC_TEST_FACE_QUADS[abs(face)]
    @inbounds A, B, C, D = cb[q[1]], cb[q[2]], cb[q[3]], cb[q[4]]
    # `face` and `A` invert signs together, which is why the product carries both.
    return abs(A * C - B * D) < eps(T) ? face >= 0 : face * A * (A * C - B * D) >= zero(T)
end

"""
    face_outside_connected(phi, dir) -> Bool

Whether the **outside** region on cube face `dir` is connected *through* that face. `true` joins the
two outside corners and cuts off the two inside ones; `false` does the opposite.

Only meaningful on an ambiguous (saddle) face -- one whose four corner values alternate in sign, so
its four edge crossings can be joined into chords two different ways. It is the asymptotic decider,
the *same* decision the Lewiner tables make when they resolve that face, which is why a face clip
built on it and the facets built from those tables agree. It reads only that face's four corner
values, so the two cells sharing a face reach it independently and agree. `moments.jl` is the
caller; see its header for why the pairing has to come from here rather than from the facets.
"""
@inline face_outside_connected(phi::SVector{8}, dir::Integer) =
    _test_face(phi, @inbounds MC_FACE_TEST_CODE[dir])

# Whether the components should connect through the *interior* of the cube. The long branch picks
# the four values `At..Dt` the decision is read off, which differ by case family; `s` is the sign
# the table attaches to the configuration.
@inline function _test_interior(case::Int, cb::SVector{8,T}, cfg::Int, subcfg::Int,
                                s::Integer) where {T}
    At = Bt = Ct = Dt = zero(T)
    @inbounds if case == 5 || case == 11
        a = (cb[5] - cb[1]) * (cb[7] - cb[3]) - (cb[8] - cb[4]) * (cb[6] - cb[2])
        b = (cb[3] * (cb[5] - cb[1]) + cb[1] * (cb[7] - cb[3]) -
             cb[2] * (cb[8] - cb[4]) - cb[4] * (cb[6] - cb[2]))
        t = -b / (2a)
        (t < zero(T) || t > one(T)) && return s > 0
        At = cb[1] + (cb[5] - cb[1]) * t
        Bt = cb[4] + (cb[8] - cb[4]) * t
        Ct = cb[3] + (cb[7] - cb[3]) * t
        Dt = cb[2] + (cb[6] - cb[2]) * t
    else # case == 7 || 8 || 13 || 14 -- a reference edge of the triangulation, 0-based
        edge = if case == 7
            Int(MarchingCubes.test6[cfg][3])
        elseif case == 8
            Int(MarchingCubes.test7[cfg][5])
        elseif case == 13
            Int(MarchingCubes.test12[cfg][4])
        else # case == 14
            Int(MarchingCubes.tiling13_5_1[cfg][subcfg][1]) - 1
        end
        if edge == 0
            t = cb[1] / (cb[1] - cb[2])
            Bt = cb[4] + (cb[3] - cb[4]) * t; Ct = cb[8] + (cb[7] - cb[8]) * t; Dt = cb[5] + (cb[6] - cb[5]) * t
        elseif edge == 1
            t = cb[2] / (cb[2] - cb[3])
            Bt = cb[1] + (cb[4] - cb[1]) * t; Ct = cb[5] + (cb[8] - cb[5]) * t; Dt = cb[6] + (cb[7] - cb[6]) * t
        elseif edge == 2
            t = cb[3] / (cb[3] - cb[4])
            Bt = cb[2] + (cb[1] - cb[2]) * t; Ct = cb[6] + (cb[5] - cb[6]) * t; Dt = cb[7] + (cb[8] - cb[7]) * t
        elseif edge == 3
            t = cb[4] / (cb[4] - cb[1])
            Bt = cb[3] + (cb[2] - cb[3]) * t; Ct = cb[7] + (cb[6] - cb[7]) * t; Dt = cb[8] + (cb[5] - cb[8]) * t
        elseif edge == 4
            t = cb[5] / (cb[5] - cb[6])
            Bt = cb[8] + (cb[7] - cb[8]) * t; Ct = cb[4] + (cb[3] - cb[4]) * t; Dt = cb[1] + (cb[2] - cb[1]) * t
        elseif edge == 5
            t = cb[6] / (cb[6] - cb[7])
            Bt = cb[5] + (cb[8] - cb[5]) * t; Ct = cb[1] + (cb[4] - cb[1]) * t; Dt = cb[2] + (cb[3] - cb[2]) * t
        elseif edge == 6
            t = cb[7] / (cb[7] - cb[8])
            Bt = cb[6] + (cb[5] - cb[6]) * t; Ct = cb[2] + (cb[1] - cb[2]) * t; Dt = cb[3] + (cb[4] - cb[3]) * t
        elseif edge == 7
            t = cb[8] / (cb[8] - cb[5])
            Bt = cb[7] + (cb[6] - cb[7]) * t; Ct = cb[3] + (cb[2] - cb[3]) * t; Dt = cb[4] + (cb[1] - cb[4]) * t
        elseif edge == 8
            t = cb[1] / (cb[1] - cb[5])
            Bt = cb[4] + (cb[8] - cb[4]) * t; Ct = cb[3] + (cb[7] - cb[3]) * t; Dt = cb[2] + (cb[6] - cb[2]) * t
        elseif edge == 9
            t = cb[2] / (cb[2] - cb[6])
            Bt = cb[1] + (cb[5] - cb[1]) * t; Ct = cb[4] + (cb[8] - cb[4]) * t; Dt = cb[3] + (cb[7] - cb[3]) * t
        elseif edge == 10
            t = cb[3] / (cb[3] - cb[7])
            Bt = cb[2] + (cb[6] - cb[2]) * t; Ct = cb[1] + (cb[5] - cb[1]) * t; Dt = cb[4] + (cb[8] - cb[4]) * t
        else # edge == 11
            t = cb[4] / (cb[4] - cb[8])
            Bt = cb[3] + (cb[7] - cb[3]) * t; Ct = cb[2] + (cb[6] - cb[2]) * t; Dt = cb[1] + (cb[5] - cb[1]) * t
        end
    end

    test = 0
    At >= zero(T) && (test += 1)
    Bt >= zero(T) && (test += 2)
    Ct >= zero(T) && (test += 4)
    Dt >= zero(T) && (test += 8)

    if test == 6 || test == 8 || test == 9 || test == 12 || (test >= 0 && test <= 4)
        return s > 0
    elseif test == 5
        return (At * Ct - Bt * Dt < eps(T)) ? s > 0 : s < 0
    elseif test == 10
        return (At * Ct - Bt * Dt >= eps(T)) ? s > 0 : s < 0
    else # test == 7 || test == 11 || test >= 13
        return s < 0
    end
end

# One tiling table entry unpacked into the fixed-size triangle buffer. `tile` is a flat run of edge
# codes, three per triangle; `n` is how many of them this configuration actually uses, which the
# table's own length over-provides for. Unused slots are zeroed rather than left undefined so a
# consumer that reads past `n` gets an obvious answer rather than a plausible one.
@inline function _tris_from(tile::NTuple{M,Int8}, n::Int) where {M}
    z = zero(SVector{3,Int8})
    return SVector{12,SVector{3,Int8}}(ntuple(Val(12)) do t
        if t <= n
            j = 3t
            SVector{3,Int8}(tile[j-2], tile[j-1], tile[j])
        else
            z
        end
    end)
end

"""
    cell_triangles(phi) -> (n, codes)

The interface triangulation of one cube, as `n` triangles of **edge codes**: `codes[t]` is a triple
in `1:13`, indexing [`edge_crossings`](@ref)' thirteen points (`13` being the interior vertex). `n`
is at most 12 (case 13.4), and 0 for an uncut cube.

This is `MarchingCubes.march`'s per-cube dispatch over the Lewiner tables, lifted out of its grid
loop -- see the "whose tables" notes above it for why that is the shipping form rather than a
convenience.

Triangles are wound so that `cross(p2 - p1, p3 - p1)` points **out of the body**, matching
the field's own normal and every other surface this package draws; that falls out of feeding a
negative-inside field to these tables. `phi` is expected to have been through
[`nudge_zeros`](@ref); see [`cell_case`](@ref).
"""
function cell_triangles(phi::SVector{8,T}) where {T}
    lut = cell_case(phi)
    @inbounds entry = MarchingCubes.cases[lut]
    case = Int(entry[1])
    cfg = Int(entry[2])
    subcfg = 1
    MC = MarchingCubes

    if case == 1
        return 0, _tris_from((Int8(0), Int8(0), Int8(0)), 0)
    elseif case == 2
        return 1, _tris_from(MC.tiling1[cfg], 1)
    elseif case == 3
        return 2, _tris_from(MC.tiling2[cfg], 2)
    elseif case == 4
        return _test_face(phi, MC.test3[cfg]) ? (4, _tris_from(MC.tiling3_2[cfg], 4)) :
                                                (2, _tris_from(MC.tiling3_1[cfg], 2))
    elseif case == 5
        return _test_interior(case, phi, cfg, subcfg, MC.test4[cfg]) ?
               (2, _tris_from(MC.tiling4_1[cfg], 2)) : (6, _tris_from(MC.tiling4_2[cfg], 6))
    elseif case == 6
        return 3, _tris_from(MC.tiling5[cfg], 3)
    elseif case == 7
        if _test_face(phi, MC.test6[cfg][1])
            return 5, _tris_from(MC.tiling6_2[cfg], 5)
        elseif _test_interior(case, phi, cfg, subcfg, MC.test6[cfg][2])
            return 3, _tris_from(MC.tiling6_1_1[cfg], 3)
        else
            return 9, _tris_from(MC.tiling6_1_2[cfg], 9)
        end
    elseif case == 8
        _test_face(phi, MC.test7[cfg][1]) && (subcfg += 1)
        _test_face(phi, MC.test7[cfg][2]) && (subcfg += 2)
        _test_face(phi, MC.test7[cfg][3]) && (subcfg += 4)
        if subcfg == 1
            return 3, _tris_from(MC.tiling7_1[cfg], 3)
        elseif subcfg == 2 || subcfg == 3
            return 5, _tris_from(MC.tiling7_2[cfg][subcfg-1], 5)
        elseif subcfg == 4
            return 9, _tris_from(MC.tiling7_3[cfg][1], 9)
        elseif subcfg == 5
            return 5, _tris_from(MC.tiling7_2[cfg][3], 5)
        elseif subcfg == 6 || subcfg == 7
            return 9, _tris_from(MC.tiling7_3[cfg][subcfg-4], 9)
        else # subcfg == 8
            return _test_interior(case, phi, cfg, subcfg, MC.test7[cfg][4]) ?
                   (9, _tris_from(MC.tiling7_4_2[cfg], 9)) :
                   (5, _tris_from(MC.tiling7_4_1[cfg], 5))
        end
    elseif case == 9
        return 2, _tris_from(MC.tiling8[cfg], 2)
    elseif case == 10
        return 4, _tris_from(MC.tiling9[cfg], 4)
    elseif case == 11
        if _test_face(phi, MC.test10[cfg][1])
            return _test_face(phi, MC.test10[cfg][2]) ?
                   (4, _tris_from(MC.tiling10_1_1_[cfg], 4)) :
                   (8, _tris_from(MC.tiling10_2[cfg], 8))
        elseif _test_face(phi, MC.test10[cfg][2])
            return 8, _tris_from(MC.tiling10_2_[cfg], 8)
        elseif _test_interior(case, phi, cfg, subcfg, MC.test10[cfg][3])
            return 4, _tris_from(MC.tiling10_1_1[cfg], 4)
        else
            return 8, _tris_from(MC.tiling10_1_2[cfg], 8)
        end
    elseif case == 12
        return 4, _tris_from(MC.tiling11[cfg], 4)
    elseif case == 13
        if _test_face(phi, MC.test12[cfg][1])
            return _test_face(phi, MC.test12[cfg][2]) ?
                   (4, _tris_from(MC.tiling12_1_1_[cfg], 4)) :
                   (8, _tris_from(MC.tiling12_2[cfg], 8))
        elseif _test_face(phi, MC.test12[cfg][2])
            return 8, _tris_from(MC.tiling12_2_[cfg], 8)
        elseif _test_interior(case, phi, cfg, subcfg, MC.test12[cfg][3])
            return 4, _tris_from(MC.tiling12_1_1[cfg], 4)
        else
            return 8, _tris_from(MC.tiling12_1_2[cfg], 8)
        end
    elseif case == 14
        _test_face(phi, MC.test13[cfg][1]) && (subcfg += 1)
        _test_face(phi, MC.test13[cfg][2]) && (subcfg += 2)
        _test_face(phi, MC.test13[cfg][3]) && (subcfg += 4)
        _test_face(phi, MC.test13[cfg][4]) && (subcfg += 8)
        _test_face(phi, MC.test13[cfg][5]) && (subcfg += 16)
        _test_face(phi, MC.test13[cfg][6]) && (subcfg += 32)
        sc = Int(MC.subcfg13[subcfg])
        if sc == 0
            return 4, _tris_from(MC.tiling13_1[cfg], 4)
        elseif sc <= 6
            return 6, _tris_from(MC.tiling13_2[cfg][sc], 6)
        elseif sc <= 18
            return 10, _tris_from(MC.tiling13_3[cfg][sc-6], 10)
        elseif sc <= 22
            return 12, _tris_from(MC.tiling13_4[cfg][sc-18], 12)
        elseif sc <= 26
            sub = sc - 22
            # `[6]`, not `[7]` (Lewiner's reference indexing): keeping this dispatch and the meshed
            # surface as one reconstruction matters more. No behavioural difference -- `_test_interior`
            # only consumes `sign(s)`, and entries 6 and 7 of `test13` are both positive here.
            return _test_interior(case, phi, cfg, sub, MC.test13[cfg][6]) ?
                   (6, _tris_from(MC.tiling13_5_1[cfg][sub], 6)) :
                   (10, _tris_from(MC.tiling13_5_2[cfg][sub], 10))
        elseif sc <= 38
            return 10, _tris_from(MC.tiling13_3_[cfg][sc-26], 10)
        elseif sc <= 44
            return 6, _tris_from(MC.tiling13_2_[cfg][sc-38], 6)
        else # sc == 45
            return 4, _tris_from(MC.tiling13_1_[cfg], 4)
        end
    else # case == 15
        return 4, _tris_from(MC.tiling14[cfg], 4)
    end
end
