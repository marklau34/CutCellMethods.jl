# =====================================
# Predicates: every combinatorial decision the cut makes, exactly, on the host
#
# Degenerate cases -- a vertex exactly on a grid line, an element lying along one, a crossing
# exactly at a grid node -- are resolved by one symbolic perturbation of the grid: every x-line is
# treated as shifted by `+ε` and every y-line by `+ε²`, for an infinitesimal `ε > 0`. No vertex then
# lies on a line and no crossing lands on a node. The shift is never applied numerically; each
# test below is the exact limit `ε -> 0+` of the perturbed question, evaluated with exact
# orientation predicates (ExactPredicates.jl) in `Float64`. Because every decision is a fact about
# the same perturbed arrangement, they cannot contradict one another -- which is what a walk that
# must pair every crossing it meets needs.
#
# The rules, stated once:
#
#  * **Side.** A point is above x-line `X` iff `p.x > X`; a point on the line is below it, since the
#    line sits at `X + ε`. Likewise for y-lines. An element crosses a line iff its endpoints are on
#    opposite sides, so an element along a line never crosses it, and the two elements at a vertex
#    on a line both cross it or neither does.
#  * **Cell.** A vertex is in cell `i` along an axis iff `line[i] < p ≤ line[i + 1]` (`_locate`).
#  * **Ownership.** Which node interval of a line a crossing falls in is decided against each node
#    by `orient(a, b, (X + ε, Y + ε²))` (`_orient_node`), never by comparing an interpolated
#    coordinate. One value per (element, node) pair answers both the x-line and the y-line question
#    at that node, so the two line families agree on which of the node's four cells an element
#    passes through.
#  * **Order.** Crossings on one edge are ordered by `_below_on_line`, from orientations of the
#    elements' endpoints, never by their interpolated coordinates -- which flip at a thin trailing
#    edge, where two elements sharing the tip cross a line within an ulp of each other.
#
# Line values are the grid's own lattice, `Float64(get_node(grid, ·))` in the cache's element type
# (`PolylineLattice`), and every test uses those numbers, so a `Float32` cache's topology agrees with
# its lattice.

"""
    _orient(a, b, c) -> Int

`+1` if `c` lies left of the directed line `a -> b`, `-1` if right, `0` if on it -- exactly.
"""
@inline _orient(a::SVector{2,Float64}, b::SVector{2,Float64}, c::SVector{2,Float64}) =
    Int(ExactPredicates.orient((a[1], a[2]), (b[1], b[2]), (c[1], c[2])))

"""
    PolylineLattice

The grid's line values in `Float64`: x-line `i` at `xs[i]`, y-line `j` at `ys[j]`, for `i` in
`1:n[1]+1` and `j` in `1:n[2]+1`. Taken off `get_node` in the grid's own element type, the
expression the cells' corners are placed with, so every decision is made against the lattice the
cells are.
"""
struct PolylineLattice
    xs::Vector{Float64}
    ys::Vector{Float64}
end

PolylineLattice(g::CartesianGrid{2}) = _pl_lattice!(PolylineLattice(Float64[], Float64[]), g)

# The lattice of `g`, into `L`'s own vectors.
function _pl_lattice!(L::PolylineLattice, g::CartesianGrid{2})
    resize!(L.xs, g.n[1] + 1)
    resize!(L.ys, g.n[2] + 1)
    for i in eachindex(L.xs)
        L.xs[i] = Float64(get_node(g, CartesianIndex(i, 1))[1])
    end
    for j in eachindex(L.ys)
        L.ys[j] = Float64(get_node(g, CartesianIndex(1, j))[2])
    end
    return L
end

@inline _lines(L::PolylineLattice, axis::Integer) = axis == 1 ? L.xs : L.ys

"""
    _locate(lines, v) -> Int

The cell `i` with `lines[i] < v ≤ lines[i + 1]`: `0` below the first line, `length(lines)` above
the last. A floor guess corrected by exact comparison against the line values, so a vertex exactly
on a line always lands below it, whatever the division rounds to.
"""
function _locate(lines::AbstractVector{Float64}, v::Float64)
    n = length(lines) - 1
    v <= lines[1] && return 0
    v > lines[n + 1] && return n + 1
    i = clamp(floor(Int, (v - lines[1]) / (lines[n + 1] - lines[1]) * n) + 1, 1, n)
    while v <= lines[i]
        i -= 1
    end
    while v > lines[i + 1]
        i += 1
    end
    return i
end

"""
    _crossed_lines(lines, u, w) -> UnitRange{Int}

The lines an element with endpoint coordinates `u`, `w` along this axis crosses: those `L` with
exactly one of `u > L`, `w > L`. As the side rule has it, that is `min(u, w) ≤ L < max(u, w)`.
"""
@inline function _crossed_lines(lines::AbstractVector{Float64}, u::Float64, w::Float64)
    lo, hi = minmax(u, w)
    return (_locate(lines, lo) + 1):_locate(lines, hi)
end

"""
    _orient_node(a, b, X, Y) -> Int

`orient(a, b, (X + ε, Y + ε²))` in the limit `ε -> 0+`: never zero for an element of nonzero
length. When the node lies exactly on the element's line the perturbation decides, and expanding
`orient` in `ε` gives `-Δy ε + Δx ε²`: the sign of `-Δy`, else of `Δx`. (The same rule as
`_edge_tiebreak` in `tri_clipping/kernels.jl`, one dimension down.)
"""
@inline function _orient_node(a::SVector{2,Float64}, b::SVector{2,Float64}, X::Float64, Y::Float64)
    o = _orient(a, b, SVector(X, Y))
    o != 0 && return o
    b[2] != a[2] && return b[2] > a[2] ? -1 : 1
    return b[1] > a[1] ? 1 : -1
end

"""
    _xcross_above(a, b, X, Y) -> Bool

Whether element `a -> b`, which crosses x-line `X`, meets it above node `(X, Y)`. The element runs
across the line, so `Δx ≠ 0`; the crossing is above the node exactly when the node is below the
element's line, which is its right for an element running `+x` and its left for one running `-x`.
"""
@inline _xcross_above(a::SVector{2,Float64}, b::SVector{2,Float64}, X::Float64, Y::Float64) =
    (b[1] > a[1] ? -1 : 1) * _orient_node(a, b, X, Y) > 0

"""
    _ycross_right(a, b, X, Y) -> Bool

Whether element `a -> b`, which crosses y-line `Y`, meets it right of node `(X, Y)`: the node is
then left of the element's line for an element running `+y`, right of it for one running `-y`.
"""
@inline _ycross_right(a::SVector{2,Float64}, b::SVector{2,Float64}, X::Float64, Y::Float64) =
    (b[2] > a[2] ? 1 : -1) * _orient_node(a, b, X, Y) > 0

"""
    _cross_interval(a, b, axis, k, L::PolylineLattice) -> Int

The node interval of line `k` (an x-line for `axis == 1`, a y-line for `axis == 2`) that element
`a -> b` crosses it in: interval `j` runs from node `j` to node `j + 1` along the line. A guess from
the interpolated coordinate, corrected node by node with the exact ownership test -- which is
monotone along the line, so the correction terminates on the right interval.
"""
function _cross_interval(a::SVector{2,Float64}, b::SVector{2,Float64}, axis::Int, k::Int,
                         L::PolylineLattice)
    line = _lines(L, axis)[k]
    nodes = _lines(L, 3 - axis)
    m = length(nodes) - 1
    j = clamp(_locate(nodes, _cross_coordinate(a, b, axis, line)), 1, m)
    while j > 1 && !_past_node(a, b, axis, line, nodes[j])
        j -= 1
    end
    while j < m && _past_node(a, b, axis, line, nodes[j + 1])
        j += 1
    end
    return j
end

# Whether element `a -> b` crosses its line (of `axis`, at `line`) past the node at `node` along
# it: above the node on an x-line, right of it on a y-line.
@inline _past_node(a, b, axis::Int, line::Float64, node::Float64) =
    axis == 1 ? _xcross_above(a, b, line, node) : _ycross_right(a, b, node, line)

"""
    _cross_coordinate(a, b, axis, line) -> Float64

The coordinate along the line where element `a -> b` crosses it, interpolated from the endpoint
nearer the line, so that a vertex lying on the line gives back its own coordinate exactly.
"""
@inline function _cross_coordinate(a::SVector{2,Float64}, b::SVector{2,Float64}, axis::Int,
                                   line::Float64)
    p = 3 - axis
    e, f = abs(a[axis] - line) <= abs(b[axis] - line) ? (a, b) : (b, a)
    return e[p] + (line - e[axis]) * (f[p] - e[p]) / (f[axis] - e[axis])
end

"""
    _cross_offset(a, b, axis, line, lo, hi) -> Float64

The crossing's offset along its edge from the edge's lower node at `lo`, the edge running to `hi`:
interpolated in node-local coordinates from the endpoint nearer the line, and clamped into the
edge, which the exact ownership test chose.
"""
@inline function _cross_offset(a::SVector{2,Float64}, b::SVector{2,Float64}, axis::Int,
                               line::Float64, lo::Float64, hi::Float64)
    p = 3 - axis
    e, f = abs(a[axis] - line) <= abs(b[axis] - line) ? (a, b) : (b, a)
    s = (e[p] - lo) + (line - e[axis]) * (f[p] - e[p]) / (f[axis] - e[axis])
    return clamp(s, 0.0, hi - lo)
end

"""
    _line_frame(a, b, axis) -> (lo, hi)

Element `a -> b`, which crosses a line of `axis`, in that line's frame `(u, v)` -- `u` across the
line, `v` along it -- with `lo` the endpoint below the line (`u ≤` the line) and `hi` the one
above.
"""
@inline function _line_frame(a::SVector{2,Float64}, b::SVector{2,Float64}, axis::Int)
    p = 3 - axis
    fa = SVector(a[axis], a[p])
    fb = SVector(b[axis], b[p])
    return fa[1] <= fb[1] ? (fa, fb) : (fb, fa)
end

"""
    _below_on_line(lo1, hi1, lo2, hi2) -> Bool

Whether element 1 crosses a line below element 2 -- at the smaller coordinate along it -- given
both in the line's frame (`_line_frame`) and both crossing it. Needs the elements not to cross or
touch except at a shared endpoint, which the mesh validation guarantees; they are then one above the
other over their whole common span, so the answer is the same for every line both cross.

Decided by orientations, not by interpolated coordinates:

- a shared endpoint (two elements meeting at a trailing-edge tip, say) -- the other two endpoints
  against each other;
- otherwise -- the element that starts later, by its lower endpoint against the other element,
  which spans it.

Zero is touching or overlap and cannot occur on a validated mesh.
"""
function _below_on_line(lo1::SVector{2,Float64}, hi1::SVector{2,Float64},
                        lo2::SVector{2,Float64}, hi2::SVector{2,Float64})
    # `s < 0`: element 1 is below. Each case is "is element 1's point right of element 2's
    # +u-directed line", or the negation when the roles are swapped.
    s = lo1 == lo2 ? _orient(lo1, hi2, hi1) :
        hi1 == hi2 ? _orient(lo2, hi1, lo1) :
        lo1[1] >= lo2[1] ? _orient(lo2, hi2, lo1) :
        -_orient(lo1, hi1, lo2)
    s == 0 && throw(ArgumentError(
        "two elements touch or overlap where they cross a grid line; the mesh should have been " *
        "rejected by validation"))
    return s < 0
end
