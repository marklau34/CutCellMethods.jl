# =====================================
# The polylines' topology: loops, and the checks everything downstream assumes
#
# Built on the host from `mesh.elements` and cached by the cache between updates, keyed by a
# fingerprint of the connectivity: a body moving rigidly keeps its loops, so a moving airfoil pays
# for them once. The geometric checks -- winding, intersections, nesting -- depend on the
# coordinates too, and run with the topology or, under `validate = :always`, on every update.
#
# Validation is strict and never repairs. The walk needs each element's successor and predecessor,
# so every node must start exactly one element and end exactly one; it needs the solid on the left,
# so every loop must wind counter-clockwise; and it pairs crossings on the assumption that no two
# elements cross or touch, so they must not, and no loop may lie inside another (a hole in a body is
# out of scope). Each failure is an `ArgumentError` naming the loop and element, so the fix can be
# made in the mesh, where the element normals the caller sees live too.

"""
    PolylineTopology

What [`PolylineClippingCutCell`](@ref)'s cache knows about a mesh's connectivity, built by
[`build_polyline_topology`](@ref):

- `elem_next`, `elem_prev` -- each element's successor and predecessor along its loop: element `e`
  runs from node `con[1]` to `con[2]`, and `elem_next[e]` starts at `con[2]`.
- `elem_loop` -- each element's loop; `loop_first` and `loop_nelem` -- each loop's first element (in
  mesh order) and its length. Loops are numbered by their first element.
- `loop_name` -- the name of the element set holding a loop's first element, or `"loop k"`.
"""
struct PolylineTopology
    fingerprint::UInt
    nnode::Int
    nelem::Int
    elem_next::Vector{Int32}
    elem_prev::Vector{Int32}
    elem_loop::Vector{Int32}
    loop_first::Vector{Int32}
    loop_nelem::Vector{Int32}
    loop_name::Vector{String}
end

nloops(topo::PolylineTopology) = length(topo.loop_first)

# What identifies a mesh's topology between updates: a hash of its connectivity and its sets'
# contents, by content rather than identity, as `_topology_fingerprint` does for triangles. The
# coordinates are deliberately left out.
function _pl_fingerprint(mesh::Mesh)
    h = hash(length(mesh.nodes), hash(length(mesh.elements), hash(:poly_topology)))
    for e in mesh.elements
        h = hash(e.con, h)
    end
    for set in mesh.elemset
        h = hash(set.name, h)
        for i in set.elems
            h = hash(i, h)
        end
    end
    return h
end

function _pl_lines(mesh::Mesh{2})
    eltype(mesh.elements) <: Line || throw(ArgumentError(
        "PolylineClippingCutCell needs a mesh of `Line` elements, got $(eltype(mesh.elements))"))
    return SVector{2,Int32}[SVector{2,Int32}(e.con) for e in mesh.elements]
end
_pl_lines(::Mesh{D}) where {D} = throw(ArgumentError(
    "PolylineClippingCutCell needs a 2D mesh of `Line` elements, got a $(D)D one"))

"""
    build_polyline_topology(mesh) -> PolylineTopology

Find `mesh`'s loops and check its connectivity: every element joins two distinct nodes, and every
node that is used starts exactly one element and ends exactly one. Throws an `ArgumentError` naming
the first node or element that breaks this -- a polyline that is open, branches, or has an element
running the wrong way. The geometric checks are the `_pl_check_*` functions below.
"""
function build_polyline_topology(mesh::Mesh)
    lines = _pl_lines(mesh)
    nn = length(mesh.nodes)
    ne = length(lines)
    starts = zeros(Int32, nn)
    ends = zeros(Int32, nn)
    for (e, c) in enumerate(lines)
        (1 <= c[1] <= nn && 1 <= c[2] <= nn) || throw(ArgumentError(
            "element $e references a node outside 1:$nn"))
        c[1] == c[2] && throw(ArgumentError(
            "element $e starts and ends at node $(c[1]): remove collapsed elements before cutting"))
        starts[c[1]] == 0 || throw(ArgumentError(
            "node $(c[1]) starts two elements, $(starts[c[1]]) and $e: the polyline branches there, " *
            "or one of them runs the wrong way"))
        ends[c[2]] == 0 || throw(ArgumentError(
            "node $(c[2]) ends two elements, $(ends[c[2]]) and $e: the polyline branches there, " *
            "or one of them runs the wrong way"))
        starts[c[1]] = e
        ends[c[2]] = e
    end
    elem_next = zeros(Int32, ne)
    elem_prev = zeros(Int32, ne)
    for (e, c) in enumerate(lines)
        nxt = starts[c[2]]
        nxt == 0 && throw(ArgumentError(
            "the polyline is open: element $e ends at node $(c[2]), where no element starts. Every " *
            "polyline must be closed -- a duplicate node left unmerged also opens it"))
        prv = ends[c[1]]
        prv == 0 && throw(ArgumentError(
            "the polyline is open: element $e starts at node $(c[1]), where no element ends. Every " *
            "polyline must be closed -- a duplicate node left unmerged also opens it"))
        elem_next[e] = nxt
        elem_prev[e] = prv
    end

    # Every element has one successor and one predecessor, so following successors from any element
    # comes back to it: the loops are the cycles of `elem_next`.
    elem_loop = zeros(Int32, ne)
    loop_first = Int32[]
    loop_nelem = Int32[]
    for e0 in 1:ne
        elem_loop[e0] == 0 || continue
        push!(loop_first, e0)
        l = Int32(length(loop_first))
        e = e0
        n = 0
        while elem_loop[e] == 0
            elem_loop[e] = l
            n += 1
            e = elem_next[e]
        end
        push!(loop_nelem, n)
    end

    set_of = Dict{Int,String}()
    for set in mesh.elemset, i in set.elems
        get!(set_of, Int(i), set.name)
    end
    loop_name = [get(set_of, Int(loop_first[l]), "loop $l") for l in eachindex(loop_first)]
    return PolylineTopology(_pl_fingerprint(mesh), nn, ne, elem_next, elem_prev, elem_loop,
                        loop_first, loop_nelem, loop_name)
end

# =====================================
# The geometric half of the mesh checks, against the coordinates as they are now
#
# An update throws an `ArgumentError` naming the loop and element if:
#
# - an element has zero length (its two nodes coincide);
# - a loop is wound clockwise, or encloses no area -- the solid must be on the left of every element;
# - two elements cross or touch anywhere but at the node two neighbours share, or two neighbours
#   fold back along each other;
# - a loop lies inside another.
#
# Every test is exact, so a mesh that passes has no near-miss the cut could misread. The first two
# are linear and as cheap as reading the coordinates, so an update runs them every time
# (`_pl_check_loops`); the intersection and nesting checks (`_pl_check_intersections`,
# `_pl_check_nesting`) run when the connectivity changes, or every time under `validate = :always`.

"""
    _pl_check_loops(topo, X, lines) -> Vector{Float64}

The linear half of the geometric checks: no zero-length element, and every loop counter-clockwise
with positive area. Returns the loop areas.
"""
function _pl_check_loops(topo::PolylineTopology, X::AbstractVector{SVector{2,Float64}},
                         lines::AbstractVector{SVector{2,Int32}})
    for (e, c) in enumerate(lines)
        X[c[1]] == X[c[2]] && throw(ArgumentError(
            "element $e (loop $(topo.elem_loop[e])) has zero length: nodes $(c[1]) and $(c[2]) " *
            "coincide at $(Tuple(X[c[1]]))"))
    end
    area = _pl_loop_areas(topo, X, lines)
    for l in eachindex(area)
        area[l] > 0 || throw(ArgumentError(
            "loop $l (\"$(topo.loop_name[l])\", from element $(topo.loop_first[l])) " *
            (area[l] < 0 ? "is wound clockwise (area $(area[l]))" : "encloses no area") *
            ": each loop must run counter-clockwise, with the solid on the left of its elements. " *
            "Reverse its elements (`reverse.(mesh.elements)`; MeshLibrary's `flip_normals!` does " *
            "not handle `Line`)"))
    end
    return area
end

# Each loop's signed area by the shoelace formula, about its first node for precision.
function _pl_loop_areas(topo::PolylineTopology, X, lines)
    area = zeros(nloops(topo))
    for l in 1:nloops(topo)
        e0 = topo.loop_first[l]
        o = X[lines[e0][1]]
        a2 = 0.0
        e = e0
        while true
            p = X[lines[e][1]] - o
            q = X[lines[e][2]] - o
            a2 += p[1] * q[2] - p[2] * q[1]
            e = topo.elem_next[e]
            e == e0 && break
        end
        area[l] = a2 / 2
    end
    return area
end

# Whether the closed segments p1-p2 and q1-q2 share any point, exactly.
function _segments_touch(p1, p2, q1, q2)
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    o1 * o2 < 0 && o3 * o4 < 0 && return true
    o1 == 0 && _in_box(p1, p2, q1) && return true
    o2 == 0 && _in_box(p1, p2, q2) && return true
    o3 == 0 && _in_box(q1, q2, p1) && return true
    o4 == 0 && _in_box(q1, q2, p2) && return true
    return false
end

# For `r` collinear with `p`-`q`: whether it lies on the segment.
@inline _in_box(p, q, r) = min(p[1], q[1]) <= r[1] <= max(p[1], q[1]) &&
                           min(p[2], q[2]) <= r[2] <= max(p[2], q[2])

# Neighbours `e -> f` meeting at node `v`: they may share `v` and nothing else, so they must not be
# collinear and pointing the same way out of `v` (a fold-back). The sign of the dot product of two
# exactly collinear differences is exact.
function _folds_back(a, v, c)
    _orient(a, v, c) == 0 || return false
    return (a[1] - v[1]) * (c[1] - v[1]) + (a[2] - v[2]) * (c[2] - v[2]) > 0
end

# All pairs of elements, through a uniform binning of their bounding boxes; each pair is tested
# once, in the lowest bin both overlap.
function _pl_check_intersections(topo::PolylineTopology, X, lines)
    ne = length(lines)
    lo = reduce((u, v) -> min.(u, v), X[c[i]] for c in lines for i in 1:2)
    hi = reduce((u, v) -> max.(u, v), X[c[i]] for c in lines for i in 1:2)
    nb = max(1, ceil(Int, sqrt(ne)))
    ext = max.(hi - lo, eps())
    bin(x) = SVector{2,Int}(clamp.(floor.(Int, (x - lo) ./ ext .* nb) .+ 1, 1, nb))
    blo = Vector{SVector{2,Int}}(undef, ne)
    bhi = Vector{SVector{2,Int}}(undef, ne)
    for (e, c) in enumerate(lines)
        blo[e] = bin(min.(X[c[1]], X[c[2]]))
        bhi[e] = bin(max.(X[c[1]], X[c[2]]))
    end
    members = [Int32[] for _ in 1:nb, _ in 1:nb]
    for e in 1:ne, j in blo[e][2]:bhi[e][2], i in blo[e][1]:bhi[e][1]
        push!(members[i, j], e)
    end
    for j in 1:nb, i in 1:nb
        m = members[i, j]
        for x in eachindex(m), y in (x + 1):lastindex(m)
            e, f = Int(m[x]), Int(m[y])
            max(blo[e][1], blo[f][1]) == i && max(blo[e][2], blo[f][2]) == j || continue
            _pl_check_pair(topo, X, lines, e, f)
        end
    end
    return nothing
end

function _pl_check_pair(topo::PolylineTopology, X, lines, e::Int, f::Int)
    ce, cf = lines[e], lines[f]
    if topo.elem_next[e] == f || topo.elem_next[f] == e
        # Neighbours: they share one node, and must not fold back along each other there. (Two
        # elements that are each other's neighbour both ways are a two-element loop, which encloses
        # no area and was rejected before this.)
        g, h = topo.elem_next[e] == f ? (e, f) : (f, e)
        a, v, c = X[lines[g][1]], X[lines[g][2]], X[lines[h][2]]
        _folds_back(a, v, c) && throw(ArgumentError(
            "elements $g and $h (loop $(topo.elem_loop[g])) fold back along each other at node " *
            "$(lines[g][2]), $(Tuple(v)): the polyline overlaps itself there"))
        return nothing
    end
    _segments_touch(X[ce[1]], X[ce[2]], X[cf[1]], X[cf[2]]) || return nothing
    le, lf = topo.elem_loop[e], topo.elem_loop[f]
    what = le == lf ? "loop $le intersects itself" : "loops $le and $lf touch or overlap"
    throw(ArgumentError(
        "$what: elements $e and $f share a point. Polylines must be simple and must not touch " *
        "one another"))
end

# No loop inside another: one node of each loop, cast against every other loop along +x. The loops
# neither cross nor touch (checked first), so one node answers for the whole loop.
function _pl_check_nesting(topo::PolylineTopology, X, lines)
    nl = nloops(topo)
    nl < 2 && return nothing
    for l in 1:nl
        p = X[lines[topo.loop_first[l]][1]]
        for m in 1:nl
            m == l && continue
            _pl_inside_loop(topo, X, lines, m, p) && throw(ArgumentError(
                "loop $l (\"$(topo.loop_name[l])\") lies inside loop $m (\"$(topo.loop_name[m])\"): " *
                "bodies may not be nested, and a hole in a body is not supported"))
        end
    end
    return nothing
end

# Whether `p`, which is on no element of loop `m`, lies inside it: the parity of the loop's
# crossings of the ray from `p` along +x, with the side rule's `y > p.y` so that a node at `p`'s
# height counts once. A crossing is right of `p` when `p` is left of an element running +y, or right
# of one running -y.
function _pl_inside_loop(topo::PolylineTopology, X, lines, m::Int, p::SVector{2,Float64})
    e0 = topo.loop_first[m]
    e = e0
    inside = false
    while true
        a, b = X[lines[e][1]], X[lines[e][2]]
        if (a[2] > p[2]) != (b[2] > p[2])
            o = _orient(a, b, p)
            o == 0 && throw(ArgumentError("a node of one loop lies on element $e of loop $m"))
            (b[2] > a[2] ? o : -o) > 0 && (inside = !inside)
        end
        e = topo.elem_next[e]
        e == e0 && break
    end
    return inside
end
