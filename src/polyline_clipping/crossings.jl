# =====================================
# Crossings: the elements against the grid lines, and everything decided from them
#
# On the host, in `Float64`, serial and in element order, so every list below comes out the same
# whatever the thread count. All of it is confined to the *block*: the cells holding a vertex, plus
# one cell of padding on every side. An element only crosses the lines strictly between its
# endpoints' cells, so the block's two outermost lines on each side are crossed by nothing, and every
# node on the block's boundary is fluid -- which is where every line walk starts.
#
# What is built, per update:
#
#  * the crossings, in one CSR over the block's **edges** -- x-edges (x-line `i`, node interval `j`)
#    first, then y-edges (interval `i`, y-line `j`) -- each edge's run sorted along it by the exact
#    comparator (`_below_on_line`), with offsets from the edge's lower node forced non-decreasing in
#    that order;
#  * the vertex bins: for each block cell, the elements *ending* at a node inside it. Every node ends
#    exactly one element, so this is a binning of the nodes that also names the element into them;
#  * the node states, from walking every line of the block through its crossings, x-lines and
#    y-lines separately so each checks the other;
#  * the cell states, and the cut list: a cell is cut when one of its edges carries a crossing or it
#    holds a vertex, and otherwise all fluid or all solid as its corners say.
#
# The tags. A crossing records which way its element runs across the line: `+1` when towards
# increasing `x` (an x-line) or `y` (a y-line). With the solid on the left of every element, walking
# *up* an x-line a `+1` crossing enters the solid; walking *along* a y-line (`+x`) a `+1` crossing
# leaves it.

"""
    PolylineBlock

The cells a polyline body can touch, inclusive: `lo` to `hi`, with one cell of fluid padding on
every side. Empty when `empty`.
"""
struct PolylineBlock
    lo::SVector{2,Int}
    hi::SVector{2,Int}
    empty::Bool
end
PolylineBlock() = PolylineBlock(SVector(1, 1), SVector(0, 0), true)

@inline _pl_dims(b::PolylineBlock) = b.empty ? SVector(0, 0) : b.hi - b.lo .+ 1
@inline _pl_nxedges(b::PolylineBlock) = (d = _pl_dims(b); (d[1] + 1) * d[2])
@inline _pl_nedges(b::PolylineBlock) = (d = _pl_dims(b); (d[1] + 1) * d[2] + d[1] * (d[2] + 1))

# Block-local linear indices. Cells and nodes are `x` fastest; x-edges are (line, interval) and
# y-edges (interval, line), each `x` fastest, y-edges after all the x-edges.
@inline _pl_cell_lin(d, il, jl) = il + d[1] * (jl - 1)
@inline _pl_node_lin(d, il, jl) = il + (d[1] + 1) * (jl - 1)
@inline _pl_xedge_lin(d, il, jl) = il + (d[1] + 1) * (jl - 1)
@inline _pl_yedge_lin(d, il, jl) = (d[1] + 1) * d[2] + il + d[1] * (jl - 1)

"""
    _pl_block(X, lines, L) -> PolylineBlock

The block for the elements' nodes as they are now. Throws if it does not fit inside the grid: the
body, grown by a cell, must lie inside the domain, so that every line walk starts in fluid and
every domain-boundary face is open.
"""
function _pl_block(X::Vector{SVector{2,Float64}}, lines::Vector{SVector{2,Int32}}, L::PolylineLattice)
    isempty(lines) && return PolylineBlock()
    lo = SVector(typemax(Int), typemax(Int))
    hi = SVector(typemin(Int), typemin(Int))
    for c in lines
        p = X[c[2]]      # every used node ends exactly one element
        ij = SVector(_locate(L.xs, p[1]), _locate(L.ys, p[2]))
        lo = min.(lo, ij)
        hi = max.(hi, ij)
    end
    n = SVector(length(L.xs) - 1, length(L.ys) - 1)
    blo = lo .- 1
    bhi = hi .+ 1
    if any(blo .< 1) || any(bhi .> n)
        throw(ArgumentError(
            "the body reaches the edge of the grid: its nodes lie in cells $(Tuple(lo)) to " *
            "$(Tuple(hi)) of a $(Tuple(n)) grid, but a cut needs at least one clear cell between " *
            "the body and every side"))
    end
    return PolylineBlock(blo, bhi, false)
end

"""
    PolylineCrossings

The host's per-update record of the elements against the block's lines -- see this file's header.
Grow-only buffers, reused across updates.
"""
mutable struct PolylineCrossings
    edge_start::Vector{Int32}    # CSR over the block's edges: edge m's run is edge_start[m]:edge_start[m+1]-1
    elem::Vector{Int32}          # the crossing element
    offset::Vector{Float64}      # along the edge from its lower node
    dir::Vector{Int8}            # +1: the element runs towards increasing x (x-line) / y (y-line)
    vbin_start::Vector{Int32}    # CSR over the block's cells
    vbin_elem::Vector{Int32}     # elements ending at a node inside the cell
    node_solid::Vector{Bool}     # the block's nodes, from the x-line walks
    status::Vector{UInt8}        # the block's cells: PL_FLUID / PL_SOLID / PL_CUT
    flags::Vector{UInt8}
    slot::Vector{Int32}          # the block's cells: index in the cut list, 0 if uncut
    cut_list::Vector{CartesianIndex{2}}
    nconflict::Int               # line walks that failed, or nodes the two families disagree on
    # per cut slot, CSR offsets into the walk's outputs, and the islands each holds
    bseg_start::Vector{Int32}
    region_start::Vector{Int32}
    arc_start::Vector{Int32}
    island_start::Vector{Int32}
    island_loop::Vector{Int32}
    island_kind::Vector{Int8}    # 0: the region owning element `target`'s piece; 1: owning right-edge arc `target`
    island_target::Vector{Int32}
    loop_ncross::Vector{Int32}
    loop_island::Vector{Bool}
    fin_list::Vector{CartesianIndex{2}}   # the cut cells and their four neighbours
    mark::Vector{Bool}
    isl::Vector{Tuple{Int,Int}}
    cand::Vector{Int32}
    # what comes back from the walk, per cut slot
    nregion::Vector{Int16}
    rslot::Vector{Int32}
    sflags::Vector{UInt8}
    sok::Vector{Bool}
    # scratch for the counting sorts
    tmp_edge::Vector{Int32}
    tmp_elem::Vector{Int32}
    tmp_offset::Vector{Float64}
    tmp_dir::Vector{Int8}
    perm::Vector{Int}
end

PolylineCrossings() = PolylineCrossings(Int32[], Int32[], Float64[], Int8[], Int32[], Int32[], Bool[],
                                        UInt8[], UInt8[], Int32[], CartesianIndex{2}[], 0,
                                        Int32[], Int32[], Int32[], Int32[], Int32[], Int8[], Int32[],
                                        Int32[], Bool[], CartesianIndex{2}[],
                                        Bool[], Tuple{Int,Int}[], Int32[],
                                        Int16[], Int32[], UInt8[], Bool[],
                                        Int32[], Int32[], Float64[], Int8[], Int[])

@inline _pl_ncross(C::PolylineCrossings, m::Integer) = C.edge_start[m + 1] - C.edge_start[m]

"""
    _pl_crossings!(C, X, lines, L, b)

Every element's crossings with the block's lines, into `C`'s edge CSR: the side rule picks the
lines (`_crossed_lines`), the exact ownership test the node interval (`_cross_interval`), and the
crossings on each edge are then sorted along it by the exact comparator.
"""
function _pl_crossings!(C::PolylineCrossings, X::Vector{SVector{2,Float64}},
                        lines::Vector{SVector{2,Int32}}, L::PolylineLattice, b::PolylineBlock)
    d = _pl_dims(b)
    ne = _pl_nedges(b)
    empty!(C.tmp_edge); empty!(C.tmp_elem); empty!(C.tmp_offset); empty!(C.tmp_dir)
    for (e, c) in enumerate(lines)
        pa, pb = X[c[1]], X[c[2]]
        for axis in 1:2
            ls = _lines(L, axis)
            nodes = _lines(L, 3 - axis)
            for k in _crossed_lines(ls, pa[axis], pb[axis])
                j = _cross_interval(pa, pb, axis, k, L)
                m = axis == 1 ? _pl_xedge_lin(d, k - b.lo[1] + 1, j - b.lo[2] + 1) :
                                _pl_yedge_lin(d, j - b.lo[1] + 1, k - b.lo[2] + 1)
                push!(C.tmp_edge, m)
                push!(C.tmp_elem, e)
                push!(C.tmp_offset, _cross_offset(pa, pb, axis, ls[k], nodes[j], nodes[j + 1]))
                push!(C.tmp_dir, pb[axis] > pa[axis] ? Int8(1) : Int8(-1))
            end
        end
    end

    # A stable counting sort by edge keeps element order within each edge, which the comparator
    # sort below then only reorders by geometry -- so the result does not depend on it anyway.
    nc = length(C.tmp_edge)
    resize!(C.edge_start, ne + 1)
    fill!(C.edge_start, 0)
    for m in C.tmp_edge
        C.edge_start[m + 1] += 1
    end
    C.edge_start[1] = 1
    for m in 1:ne
        C.edge_start[m + 1] += C.edge_start[m]
    end
    resize!(C.elem, nc); resize!(C.offset, nc); resize!(C.dir, nc)
    next = C.perm
    resize!(next, ne)
    for m in 1:ne
        next[m] = C.edge_start[m]
    end
    for t in 1:nc
        m = C.tmp_edge[t]
        s = next[m]
        next[m] += 1
        C.elem[s] = C.tmp_elem[t]
        C.offset[s] = C.tmp_offset[t]
        C.dir[s] = C.tmp_dir[t]
    end

    for m in 1:ne
        r = C.edge_start[m]:(C.edge_start[m + 1] - 1)
        length(r) > 1 || continue
        axis = m <= _pl_nxedges(b) ? 1 : 2
        _pl_sort_edge!(C, r, X, lines, axis)
    end
    return nothing
end

# One edge's crossings, sorted along it by the exact comparator, then their offsets forced
# non-decreasing in that order: an interpolated offset may be an ulp out of order where two
# elements cross within an ulp of each other, and an arc must never have negative length.
# Insertion sort in place: an edge carries a handful of crossings, and it allocates nothing.
function _pl_sort_edge!(C::PolylineCrossings, r::UnitRange{Int}, X, lines, axis::Int)
    frame(e) = _line_frame(X[lines[e][1]], X[lines[e][2]], axis)
    for i in (first(r) + 1):last(r)
        e, o, dr = C.elem[i], C.offset[i], C.dir[i]
        fe = frame(e)
        j = i - 1
        while j >= first(r) && _below_on_line(fe..., frame(C.elem[j])...)
            C.elem[j + 1], C.offset[j + 1], C.dir[j + 1] = C.elem[j], C.offset[j], C.dir[j]
            j -= 1
        end
        C.elem[j + 1], C.offset[j + 1], C.dir[j + 1] = e, o, dr
    end
    for s in (first(r) + 1):last(r)
        C.offset[s] = max(C.offset[s], C.offset[s - 1])
    end
    return nothing
end

"""
    _pl_vertex_bins!(C, X, lines, L, b)

For each block cell, the elements ending at a node inside it (`_locate`'s half-open cells), in
element order.
"""
function _pl_vertex_bins!(C::PolylineCrossings, X, lines::Vector{SVector{2,Int32}},
                          L::PolylineLattice, b::PolylineBlock)
    d = _pl_dims(b)
    ncell = d[1] * d[2]
    resize!(C.vbin_start, ncell + 1)
    fill!(C.vbin_start, 0)
    resize!(C.tmp_edge, length(lines))
    for (e, c) in enumerate(lines)
        p = X[c[2]]
        l = _pl_cell_lin(d, _locate(L.xs, p[1]) - b.lo[1] + 1, _locate(L.ys, p[2]) - b.lo[2] + 1)
        C.tmp_edge[e] = l
        C.vbin_start[l + 1] += 1
    end
    C.vbin_start[1] = 1
    for l in 1:ncell
        C.vbin_start[l + 1] += C.vbin_start[l]
    end
    resize!(C.vbin_elem, length(lines))
    next = C.perm
    resize!(next, ncell)
    for l in 1:ncell
        next[l] = C.vbin_start[l]
    end
    for e in eachindex(lines)
        l = C.tmp_edge[e]
        C.vbin_elem[next[l]] = e
        next[l] += 1
    end
    return nothing
end

"""
    _pl_classify!(C, b)

The node states, from walking every line of the block through its crossings, and the cell states
and cut list from them. The x-line walks give the states; the y-line walks must agree node for
node, and every walk must alternate entering and leaving and end in fluid. A failure of either
flags the cells around it with `PL_FLAG_STATUS_CONFLICT` and counts in `C.nconflict` -- neither
should ever happen with the exact predicates, so it is a check rather than a mechanism.
"""
function _pl_classify!(C::PolylineCrossings, b::PolylineBlock)
    d = _pl_dims(b)
    nx, ny = d[1], d[2]
    nnode = (nx + 1) * (ny + 1)
    ncell = nx * ny
    resize!(C.node_solid, nnode)
    resize!(C.status, ncell); resize!(C.flags, ncell); resize!(C.slot, ncell)
    fill!(C.flags, 0x00)
    C.nconflict = 0
    conflict_cell!(il, jl) = (1 <= il <= nx && 1 <= jl <= ny) &&
        (C.flags[_pl_cell_lin(d, il, jl)] |= PL_FLAG_STATUS_CONFLICT)
    conflict_node!(il, jl) = for (u, v) in ((il - 1, jl - 1), (il, jl - 1), (il - 1, jl), (il, jl))
        conflict_cell!(u, v)
    end

    # x-lines, walked up: a +1 crossing enters the solid.
    for il in 1:(nx + 1)
        solid = false
        C.node_solid[_pl_node_lin(d, il, 1)] = false
        for jl in 1:ny
            m = _pl_xedge_lin(d, il, jl)
            for s in C.edge_start[m]:(C.edge_start[m + 1] - 1)
                enter = C.dir[s] > 0
                if enter == solid
                    C.nconflict += 1
                    conflict_cell!(il - 1, jl); conflict_cell!(il, jl)
                end
                solid = enter
            end
            C.node_solid[_pl_node_lin(d, il, jl + 1)] = solid
        end
        solid && (C.nconflict += 1; conflict_node!(il, ny + 1))
    end

    # y-lines, walked along +x: a +1 crossing leaves the solid. Checked against the x-lines' states.
    for jl in 1:(ny + 1)
        solid = false
        for il in 1:nx
            if solid != C.node_solid[_pl_node_lin(d, il, jl)]
                C.nconflict += 1
                conflict_node!(il, jl)
            end
            m = _pl_yedge_lin(d, il, jl)
            for s in C.edge_start[m]:(C.edge_start[m + 1] - 1)
                enter = C.dir[s] < 0
                if enter == solid
                    C.nconflict += 1
                    conflict_cell!(il, jl - 1); conflict_cell!(il, jl)
                end
                solid = enter
            end
        end
        if solid != C.node_solid[_pl_node_lin(d, nx + 1, jl)] || solid
            C.nconflict += 1
            conflict_node!(nx + 1, jl)
        end
    end

    empty!(C.cut_list)
    for jl in 1:ny, il in 1:nx
        l = _pl_cell_lin(d, il, jl)
        cut = C.vbin_start[l + 1] > C.vbin_start[l] ||
              _pl_ncross(C, _pl_xedge_lin(d, il, jl)) > 0 || _pl_ncross(C, _pl_xedge_lin(d, il + 1, jl)) > 0 ||
              _pl_ncross(C, _pl_yedge_lin(d, il, jl)) > 0 || _pl_ncross(C, _pl_yedge_lin(d, il, jl + 1)) > 0
        if cut
            push!(C.cut_list, CartesianIndex(il + b.lo[1] - 1, jl + b.lo[2] - 1))
            C.status[l] = PL_CUT
            C.slot[l] = length(C.cut_list)
        else
            # No crossing on any edge: the four corners agree, by the walks' own parity.
            s = C.node_solid[_pl_node_lin(d, il, jl)]
            corners = (C.node_solid[_pl_node_lin(d, il + 1, jl)], C.node_solid[_pl_node_lin(d, il, jl + 1)],
                       C.node_solid[_pl_node_lin(d, il + 1, jl + 1)])
            if any(!=(s), corners)
                C.nconflict += 1
                C.flags[l] |= PL_FLAG_STATUS_CONFLICT
            end
            C.status[l] = s ? PL_SOLID : PL_FLUID
            C.slot[l] = 0
        end
    end
    return nothing
end

# The four edges of block cell `(il, jl)` in perimeter order -- bottom, right, top, left, the
# counter-clockwise walk from the lower-left corner (`MS_EDGE_DIRECTION` gives their directions) --
# as block edge indices, and each edge's lower node.
@inline _pl_perimeter_edges(d, il, jl) =
    (_pl_yedge_lin(d, il, jl), _pl_xedge_lin(d, il + 1, jl), _pl_yedge_lin(d, il, jl + 1),
     _pl_xedge_lin(d, il, jl))
@inline _pl_perimeter_nodes(d, il, jl) =
    (_pl_node_lin(d, il, jl), _pl_node_lin(d, il + 1, jl), _pl_node_lin(d, il, jl + 1),
     _pl_node_lin(d, il, jl))

"""
    _pl_cut_lists!(C, X, lines, topo, L, b)

What the walk needs laid out before it runs, per cut slot: where its boundary segments, regions and
arcs go -- all counted exactly from the crossings and the vertex bins, so the kernel writes straight
into its slots -- and the islands it holds, each with the region it must be subtracted from found
here, exactly. Also the finalize list, and the overflow flag of a cell with more crossings than a
walk tracks.
"""
function _pl_cut_lists!(C::PolylineCrossings, X, lines, topo::PolylineTopology, L::PolylineLattice,
                        b::PolylineBlock)
    d = _pl_dims(b)
    ncut = length(C.cut_list)
    for v in (C.bseg_start, C.region_start, C.arc_start, C.island_start)
        resize!(v, ncut + 1)
        fill!(v, 0)
    end
    for (s, ci) in enumerate(C.cut_list)
        il, jl = ci[1] - b.lo[1] + 1, ci[2] - b.lo[2] + 1
        l = _pl_cell_lin(d, il, jl)
        E = _pl_perimeter_edges(d, il, jl)
        N = _pl_perimeter_nodes(d, il, jl)
        nc = nleave = narc = 0
        for q in 1:4
            n = _pl_ncross(C, E[q])
            nc += n
            for c in C.edge_start[E[q]]:(C.edge_start[E[q] + 1] - 1)
                _pl_enters(q, C.dir[c]) || (nleave += 1)
            end
            narc += (n + 1 + !C.node_solid[N[q]]) ÷ 2
        end
        nv = C.vbin_start[l + 1] - C.vbin_start[l]
        C.bseg_start[s + 1] = nleave + nv
        C.region_start[s + 1] = max(1, nc - nleave)
        C.arc_start[s + 1] = narc
        nc > PL_MAX_CROSS && (C.flags[l] |= PL_FLAG_OVERFLOW)
    end

    # Islands: loops that cross no line at all, so lie inside one cell.
    nl = nloops(topo)
    resize!(C.loop_ncross, nl)
    fill!(C.loop_ncross, 0)
    for e in C.elem
        C.loop_ncross[topo.elem_loop[e]] += 1
    end
    resize!(C.loop_island, nl)
    C.loop_island .= C.loop_ncross .== 0
    isl = C.isl
    empty!(isl)
    for m in 1:nl
        C.loop_island[m] || continue
        p = X[lines[topo.loop_first[m]][1]]
        l = _pl_cell_lin(d, _locate(L.xs, p[1]) - b.lo[1] + 1, _locate(L.ys, p[2]) - b.lo[2] + 1)
        push!(isl, (Int(C.slot[l]), m))
    end
    sort!(isl)
    empty!(C.island_loop); empty!(C.island_kind); empty!(C.island_target)
    for (s, m) in isl
        C.island_start[s + 1] += 1
        kind, target = _pl_island_target(C, X, lines, topo, L, b, C.cut_list[s], m)
        push!(C.island_loop, m)
        push!(C.island_kind, kind)
        push!(C.island_target, target)
    end

    for v in (C.bseg_start, C.region_start, C.arc_start, C.island_start)
        v[1] = 1
        for s in 1:ncut
            v[s + 1] += v[s]
        end
    end

    # The finalize list: the cut cells, and the neighbours a snapped sliver can hand a wall to.
    mark = C.mark
    resize!(mark, d[1] * d[2])
    fill!(mark, false)
    for ci in C.cut_list
        il, jl = ci[1] - b.lo[1] + 1, ci[2] - b.lo[2] + 1
        for (u, v) in ((il, jl), (il - 1, jl), (il + 1, jl), (il, jl - 1), (il, jl + 1))
            (1 <= u <= d[1] && 1 <= v <= d[2]) && (mark[_pl_cell_lin(d, u, v)] = true)
        end
    end
    empty!(C.fin_list)
    for jl in 1:d[2], il in 1:d[1]
        mark[_pl_cell_lin(d, il, jl)] && push!(C.fin_list, CartesianIndex(il + b.lo[1] - 1, jl + b.lo[2] - 1))
    end
    return nothing
end

"""
    _pl_island_target(C, X, lines, topo, L, b, ci, m) -> (kind, target)

Which fluid region of cell `ci` island loop `m` lies in, decided exactly: a ray from one of the
island's nodes along `+x`, against the pieces of the other loops in the cell, ordered by the exact
comparator. The first piece it meets bounds that region (`kind = 0`, `target` the element); if it
meets none it reaches the cell's right edge, in the fluid interval (`kind = 1`, `target` the
interval's number along the edge) whose arc that region owns.

The ray is taken at `y + δ` for a `δ` infinitesimal against the grid's own perturbation, which
settles an element or a crossing at the node's exact height the same way the side rule does.
"""
function _pl_island_target(C::PolylineCrossings, X, lines, topo::PolylineTopology,
                           L::PolylineLattice, b::PolylineBlock, ci::CartesianIndex{2}, m::Int)
    d = _pl_dims(b)
    il, jl = ci[1] - b.lo[1] + 1, ci[2] - b.lo[2] + 1
    v = X[lines[topo.loop_first[m]][1]]
    Xr = L.xs[ci[1] + 1]
    E = _pl_perimeter_edges(d, il, jl)
    l = _pl_cell_lin(d, il, jl)
    cand = C.cand
    empty!(cand)
    for q in 1:4, c in C.edge_start[E[q]]:(C.edge_start[E[q] + 1] - 1)
        push!(cand, C.elem[c])
    end
    for k in C.vbin_start[l]:(C.vbin_start[l + 1] - 1)
        e = C.vbin_elem[k]
        C.loop_island[topo.elem_loop[e]] || push!(cand, e)
    end
    unique!(sort!(cand))
    best = 0
    for e in cand
        a, bb = X[lines[e][1]], X[lines[e][2]]
        (a[2] > v[2]) != (bb[2] > v[2]) || continue
        sg = bb[2] > a[2] ? 1 : -1
        sg * _orient(a, bb, v) > 0 || continue                     # meets the ray right of v
        sg * _orient(a, bb, SVector(Xr, v[2])) > 0 && continue     # ... but beyond the right edge
        if best == 0 || _below_on_line(_line_frame(a, bb, 2)...,
                                       _line_frame(X[lines[best][1]], X[lines[best][2]], 2)...)
            best = e
        end
    end
    best == 0 || return Int8(0), Int32(best)
    # The right edge: count its crossings below the ray, then which fluid interval that leaves.
    nb = 0
    for c in C.edge_start[E[2]]:(C.edge_start[E[2] + 1] - 1)
        a, bb = X[lines[C.elem[c]][1]], X[lines[C.elem[c]][2]]
        sx = bb[1] > a[1] ? 1 : -1
        o = _orient(a, bb, SVector(Xr, v[2]))
        nb += o != 0 ? sx * o > 0 : sx * (bb[2] - a[2]) <= 0
    end
    fluid_lo = !C.node_solid[_pl_node_lin(d, il + 1, jl)]
    iseven(nb) == fluid_lo || throw(ArgumentError(
        "island loop $m reaches a solid stretch of its cell's right edge; the mesh should have " *
        "been rejected as nested"))
    return Int8(1), Int32((nb + (fluid_lo ? 2 : 1)) ÷ 2)
end
