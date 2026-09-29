# =====================================
# The walk: one cut cell's fluid regions, inside a kernel
#
# Everything combinatorial arrived from the host: each edge's crossings in order along it with
# their offsets and tags, the nodes' states, the vertex bins, and for each island the region it
# lies in. What is left is bookkeeping and arithmetic, and none of it allocates: the cell's
# crossings are read straight out of the edge CSR in perimeter order (`_pl_perim`), the crossings a
# walk has used are the bits of one `UInt64`, and every output has a slot the host counted.
#
# In one cell, in order:
#
#  1. **Arcs.** Every fluid interval of each of the four edges gets its slot, unowned (`region 0`).
#  2. **Pieces and the walk-free totals.** Every clipped element is enumerated exactly once by its
#     end -- a crossing where the polyline leaves the cell, or a node inside it -- and written as a
#     boundary segment. The cell's fluid area and first moment follow from Green's theorem over the
#     fluid arcs plus every piece reversed, one segment at a time, with no connectivity at all: the
#     record the cell is given however the walk fares.
#  3. **The walk.** From each unused crossing where a fluid arc starts, counter-clockwise along the
#     cell boundary to where the arc ends -- the polyline leaves the cell there -- then back along
#     the polyline, element by element, to where it entered, which starts the next arc; until the
#     loop closes. Each loop is one region, numbered in walk order from the lower-left corner; its
#     arcs and pieces are labelled with it. A failure -- a crossing used twice, an arc that does not
#     end where it should, a piece or arc left unowned -- lumps the cell into one region and marks
#     it invalid; the walk-free totals keep its record exact.
#  4. **Islands**, subtracted from the region the host found them in.
#  5. **Ranks.** A region below `drop_area` is a sliver and gets rank 0; the others are numbered
#     1, 2, ... in walk order, and every arc and piece is relabelled with its region's rank.
#
# Local coordinates throughout: offsets from the cell's lower lattice corner, so the products are
# `O(h^2)` wherever the cell is.
#
# The perimeter runs counter-clockwise from the lower-left corner over four edges `q`: bottom,
# right, top, left (`MS_EDGE_DIRECTION[q]` is each one's direction). Along each edge the crossings
# are stored in increasing offset, so the perimeter meets the bottom and right edges' crossings in
# stored order and the top and left edges' in reverse.

# The walk's view of one cut cell.
struct _PLCell{T}
    s::Int                   # cut slot
    l::Int                   # block-local linear index
    lo::SVector{2,T}         # lower lattice corner
    h::SVector{2,T}          # extent, as lattice differences
    start::NTuple{4,Int}     # each perimeter edge's first crossing
    n::NTuple{4,Int}         # and how many it has
    fluid::NTuple{4,Bool}    # each perimeter edge's lower node is fluid
    nc::Int
end

@inline function _pl_cell(m, g::CartesianGrid{2,T}, blo::CartesianIndex{2}, d::NTuple{2,Int},
                          s::Int) where {T}
    ci = @inbounds m.cut_list[s]
    il, jl = ci[1] - blo[1] + 1, ci[2] - blo[2] + 1
    E = _pl_perimeter_edges(d, il, jl)
    N = _pl_perimeter_nodes(d, il, jl)
    start = ntuple(q -> Int(@inbounds m.edge_start[E[q]]), Val(4))
    n = ntuple(q -> Int(@inbounds m.edge_start[E[q] + 1]) - start[q], Val(4))
    fluid = ntuple(q -> !(@inbounds m.node_solid[N[q]]), Val(4))
    lo = get_node(g, ci)
    h = get_node(g, ci + CartesianIndex(1, 1)) - lo
    return _PLCell{T}(s, _pl_cell_lin(d, il, jl), lo, h, start, n, fluid, n[1] + n[2] + n[3] + n[4])
end

# Whether the polyline enters the cell at a crossing on perimeter edge `q` running `dir` across its
# line: up through the bottom, left through the right edge, down through the top, right through the
# left. At such a crossing the perimeter passes from solid to fluid -- a fluid arc starts.
@inline _pl_enters(q::Integer, dir::Integer) = (q == 1 || q == 4) ? dir > 0 : dir < 0

# Perimeter position `p` (1:nc) -> perimeter edge `q`, index `t` along the edge, crossing index.
@inline function _pl_perim(C::_PLCell, p::Int)
    q = 1
    while q < 4 && p > C.n[q]
        p -= C.n[q]
        q += 1
    end
    t = q <= 2 ? p : C.n[q] - p + 1
    return q, t, C.start[q] + t - 1
end

@inline _pl_edge_len(C::_PLCell, q::Integer) = (q == 1 || q == 3) ? C.h[1] : C.h[2]

# The point at offset `s` along perimeter edge `q`, and the corner the perimeter reaches at `q`'s end.
@inline function _pl_edge_point(C::_PLCell{T}, q::Integer, s::T) where {T}
    return q == 1 ? SVector{2,T}(s, zero(T)) : q == 2 ? SVector{2,T}(C.h[1], s) :
           q == 3 ? SVector{2,T}(s, C.h[2]) : SVector{2,T}(zero(T), s)
end
@inline function _pl_corner(C::_PLCell{T}, q::Integer) where {T}
    return q == 1 ? SVector{2,T}(C.h[1], zero(T)) : q == 2 ? SVector{2,T}(C.h[1], C.h[2]) :
           q == 3 ? SVector{2,T}(zero(T), C.h[2]) : SVector{2,T}(zero(T), zero(T))
end

@inline function _pl_perim_point(m, C::_PLCell{T}, p::Int) where {T}
    q, _, c = _pl_perim(C, p)
    return _pl_edge_point(C, q, @inbounds m.cr_offset[c])
end

@inline _pl_local(m, C::_PLCell, v::Integer) = (@inbounds m.verts[v]) - C.lo

# One directed segment's share of Green's theorem: twice the signed area, and the first moment
# times six.
@inline function _pl_seg(a2::T, mom::SVector{2,T}, p::SVector{2,T}, q::SVector{2,T}) where {T}
    cr = p[1] * q[2] - p[2] * q[1]
    return a2 + cr, mom + (p + q) * cr
end

# Interval `iv` of an edge (0-based from its lower node) with `n` crossings from `start`: its ends.
@inline function _pl_interval(offset, start::Int, n::Int, iv::Int, len::T) where {T}
    s0 = iv == 0 ? zero(T) : T(@inbounds offset[start + iv - 1])
    s1 = iv == n ? len : T(@inbounds offset[start + iv])
    return s0, s1
end

# Fluid intervals of an edge with `n` crossings, from its lower node's state; and the fluid number
# `k` of fluid interval `iv` (whose parity must say fluid).
@inline _pl_nfluid(n::Int, fluid::Bool) = (n + 1 + fluid) ÷ 2
@inline _pl_fluid_k(iv::Int, fluid::Bool) = (iv + (fluid ? 2 : 1)) ÷ 2
@inline _pl_is_fluid(iv::Int, fluid::Bool) = iseven(iv) == fluid

# Where a cell's arcs on direction `dir` start among its arc slots: they are laid out by direction.
@inline function _pl_arc_off(C::_PLCell, dir::Int)
    off = 0
    for dd in 1:(dir - 1)
        q = MS_DIRECTION_EDGE[dd]
        off += _pl_nfluid(C.n[q], C.fluid[q])
    end
    return off
end
@inline _pl_arc_slot(m, C::_PLCell, q::Int, k::Int) =
    Int(@inbounds m.arc_start[C.s]) + _pl_arc_off(C, MS_EDGE_DIRECTION[q]) + k - 1

# 1. Every fluid interval of every edge, unowned.
function _pl_init_arcs!(m, C::_PLCell{T}) where {T}
    for dir in 1:4
        q = MS_DIRECTION_EDGE[dir]
        base = Int(@inbounds m.arc_start[C.s]) + _pl_arc_off(C, dir)
        for k in 1:_pl_nfluid(C.n[q], C.fluid[q])
            iv = 2(k - 1) + (C.fluid[q] ? 0 : 1)
            s0, s1 = _pl_interval(m.cr_offset, C.start[q], C.n[q], iv, _pl_edge_len(C, q))
            @inbounds m.arcs[base + k - 1] = PolylineArc{T}(Int16(0), Int8(dir), Int16(k), s0, s1,
                                                            Int16(0), false)
        end
    end
    return nothing
end

# The crossing where element `e` enters the cell, by perimeter position; 0 if it starts inside.
@inline function _pl_find_entry(m, C::_PLCell, e::Int)
    for p in 1:C.nc
        q, _, c = _pl_perim(C, p)
        (@inbounds(m.cr_elem[c]) == e && _pl_enters(q, @inbounds m.cr_dir[c])) && return p
    end
    return 0
end

# A piece's length is its own ends'; its normal its element's, from the element's two nodes -- the
# piece lies on the element, and a piece an ulp long (a crossing an ulp from a node) has ends too
# close together to give a direction.
@inline function _pl_write_piece!(m, C::_PLCell{T}, k::Int, e::Int, S::SVector{2,T},
                                  Ep::SVector{2,T}, has_vertex::Bool) where {T}
    f = Ep - S
    len = sqrt(f[1] * f[1] + f[2] * f[2])
    ln = @inbounds m.lines[e]
    v = (@inbounds m.verts[ln[2]]) - (@inbounds m.verts[ln[1]])
    lv = sqrt(v[1] * v[1] + v[2] * v[2])
    nrm = len > zero(T) ? SVector{2,T}(v[2], -v[1]) / lv : zero(SVector{2,T})
    @inbounds m.bsegs[k] = PolylineBoundarySeg{T}(Int16(0), m.elem_loop[e], Int32(e), len, nrm,
                                                  C.lo + T(0.5) * (S + Ep), has_vertex)
    return nothing
end

# 2. The pieces, as boundary segments, and the walk-free area sums.
function _pl_walk_free!(m, C::_PLCell{T}) where {T}
    a2 = zero(T)
    mom = zero(SVector{2,T})
    k = Int(@inbounds m.bseg_start[C.s])
    for p in 1:C.nc
        q, _, c = _pl_perim(C, p)
        _pl_enters(q, @inbounds m.cr_dir[c]) && continue
        e = Int(@inbounds m.cr_elem[c])
        Ep = _pl_edge_point(C, q, @inbounds m.cr_offset[c])
        pin = _pl_find_entry(m, C, e)
        S = pin > 0 ? _pl_perim_point(m, C, pin) : _pl_local(m, C, (@inbounds m.lines[e])[1])
        _pl_write_piece!(m, C, k, e, S, Ep, pin == 0)
        k += 1
        a2, mom = _pl_seg(a2, mom, Ep, S)
    end
    for v in Int(@inbounds m.vbin_start[C.l]):(Int(@inbounds m.vbin_start[C.l + 1]) - 1)
        e = Int(@inbounds m.vbin_elem[v])
        Ep = _pl_local(m, C, (@inbounds m.lines[e])[2])
        pin = _pl_find_entry(m, C, e)
        S = pin > 0 ? _pl_perim_point(m, C, pin) : _pl_local(m, C, (@inbounds m.lines[e])[1])
        _pl_write_piece!(m, C, k, e, S, Ep, true)
        k += 1
        a2, mom = _pl_seg(a2, mom, Ep, S)
    end
    if C.nc == 0
        if C.fluid[1]
            for q in 1:4
                a2, mom = _pl_seg(a2, mom, _pl_corner(C, q == 1 ? 4 : q - 1), _pl_corner(C, q))
            end
        end
    else
        for p in 1:C.nc
            q, _, c = _pl_perim(C, p)
            _pl_enters(q, @inbounds m.cr_dir[c]) || continue
            a2, mom, _ = _pl_arc_path(m, C, p, p % C.nc + 1, a2, mom, 0)
        end
    end
    return a2, mom
end

# The perimeter from crossing `p0` counter-clockwise to crossing `p1`, the next one: its share of
# Green's theorem, and -- for `r > 0` -- the fluid interval of each edge it runs along labelled
# with region `r`. `false` if an interval's parity says it is not fluid.
function _pl_arc_path(m, C::_PLCell{T}, p0::Int, p1::Int, a2::T, mom::SVector{2,T},
                      r::Int, rec=nothing) where {T}
    q0, t0, c0 = _pl_perim(C, p0)
    q1, _, c1 = _pl_perim(C, p1)
    P0 = _pl_edge_point(C, q0, @inbounds m.cr_offset[c0])
    P1 = _pl_edge_point(C, q1, @inbounds m.cr_offset[c1])
    ok = true
    # The interval the path starts along: after crossing t0 on an ascending edge, before it on a
    # descending one.
    iv0 = q0 <= 2 ? t0 : t0 - 1
    if q0 == q1 && p1 > p0
        a2, mom = _pl_seg(a2, mom, P0, P1)
        _pl_rec!(rec, P1)
        ok &= _pl_label_arc!(m, C, q0, iv0, r)
        return a2, mom, ok
    end
    K = _pl_corner(C, q0)
    a2, mom = _pl_seg(a2, mom, P0, K)
    _pl_rec!(rec, K)
    ok &= _pl_label_arc!(m, C, q0, iv0, r)
    q = q0 == 4 ? 1 : q0 + 1
    # Whole edges between carry no crossing, `p1` being the next one; each is one interval.
    for _ in 1:4
        q == q1 && break
        K2 = _pl_corner(C, q)
        a2, mom = _pl_seg(a2, mom, K, K2)
        _pl_rec!(rec, K2)
        ok &= _pl_label_arc!(m, C, q, 0, r)
        K = K2
        q = q == 4 ? 1 : q + 1
    end
    a2, mom = _pl_seg(a2, mom, K, P1)
    _pl_rec!(rec, P1)
    ok &= _pl_label_arc!(m, C, q1, q1 <= 2 ? 0 : C.n[q1], r)
    return a2, mom, ok
end

@inline function _pl_label_arc!(m, C::_PLCell, q::Int, iv::Int, r::Int)
    _pl_is_fluid(iv, C.fluid[q]) || return false
    r > 0 || return true
    slot = _pl_arc_slot(m, C, q, _pl_fluid_k(iv, C.fluid[q]))
    @inbounds m.arcs.region[slot] == 0 || return false
    @inbounds m.arcs.region[slot] = Int16(r)
    return true
end

# Label element `e`'s piece in this cell with region `r`: `false` if it has none or is already owned.
@inline function _pl_label_piece!(m, C::_PLCell, e::Int, r::Int)
    for k in Int(@inbounds m.bseg_start[C.s]):(Int(@inbounds m.bseg_start[C.s + 1]) - 1)
        if @inbounds(m.bsegs.elem[k]) == e
            @inbounds m.bsegs.region[k] == 0 || return false
            @inbounds m.bsegs.region[k] = Int16(r)
            return true
        end
    end
    return false
end

# The loops a walk traces, for drawing them (`region_loops`): in a kernel `rec` is `nothing` and
# all of this compiles away; on the host it is a `_PLRecorder`, which collects each loop's points
# in order -- regions counter-clockwise, islands (holes) as their own loops -- in local coordinates.
struct _PLRecorder{T}
    region::Vector{Int}      # raw region id, per loop
    hole::Vector{Bool}
    start::Vector{Int}       # first point of each loop
    points::Vector{SVector{2,T}}
end
_PLRecorder{T}() where {T} = _PLRecorder{T}(Int[], Bool[], Int[], SVector{2,T}[])

@inline _pl_rec_loop!(::Nothing, _, _) = nothing
@inline _pl_rec!(::Nothing, _) = nothing
function _pl_rec_loop!(R::_PLRecorder, r::Integer, hole::Bool)
    push!(R.region, r)
    push!(R.hole, hole)
    push!(R.start, length(R.points) + 1)
    return nothing
end
_pl_rec!(R::_PLRecorder, p) = (push!(R.points, p); nothing)

@inline _pl_bit(used::UInt64, p::Int) = (used >> (p - 1)) & one(UInt64) == one(UInt64)
@inline _pl_setbit(used::UInt64, p::Int) = used | (one(UInt64) << (p - 1))

# 3. The walk. Returns the regions walked and whether it succeeded; the regions' sums go to `racc`.
function _pl_walk!(m, C::_PLCell{T}, nv::Int, rec=nothing) where {T}
    rs = Int(@inbounds m.region_start[C.s])
    rmax = Int(@inbounds m.region_start[C.s + 1]) - rs
    if C.nc == 0
        # No crossings: an island's cell, all fluid, one region round the whole boundary.
        C.fluid[1] || return 0, false
        a2 = zero(T)
        mom = zero(SVector{2,T})
        ok = true
        _pl_rec_loop!(rec, 1, false)
        _pl_rec!(rec, _pl_corner(C, 4))
        for q in 1:4
            a2, mom = _pl_seg(a2, mom, _pl_corner(C, q == 1 ? 4 : q - 1), _pl_corner(C, q))
            _pl_rec!(rec, _pl_corner(C, q))
            ok &= _pl_label_arc!(m, C, q, 0, 1)
        end
        @inbounds m.racc[rs] = PolylineRegionAcc{T}(a2, mom, Int16(0))
        return 1, ok
    end
    C.nc > PL_MAX_CROSS && return 0, false
    used = zero(UInt64)
    nraw = 0
    for p0 in 1:C.nc
        q0, _, c0 = _pl_perim(C, p0)
        (_pl_enters(q0, @inbounds m.cr_dir[c0]) && !_pl_bit(used, p0)) || continue
        nraw += 1
        nraw > rmax && return nraw, false
        a2 = zero(T)
        mom = zero(SVector{2,T})
        _pl_rec_loop!(rec, nraw, false)
        _pl_rec!(rec, _pl_perim_point(m, C, p0))
        p = p0
        closed = false
        for _ in 1:C.nc       # each pass uses two crossings, so this bounds the loop
            used = _pl_setbit(used, p)
            p1 = p % C.nc + 1
            q1, _, c1 = _pl_perim(C, p1)
            (_pl_enters(q1, @inbounds m.cr_dir[c1]) || _pl_bit(used, p1)) && return nraw, false
            used = _pl_setbit(used, p1)
            a2, mom, ok = _pl_arc_path(m, C, p, p1, a2, mom, nraw, rec)
            ok || return nraw, false
            # Back along the polyline from where it leaves, to where it came in.
            cur = _pl_edge_point(C, q1, @inbounds m.cr_offset[c1])
            e = Int(@inbounds m.cr_elem[c1])
            pin = 0
            for _ in 0:nv
                _pl_label_piece!(m, C, e, nraw) || return nraw, false
                pin = _pl_find_entry(m, C, e)
                if pin > 0
                    a2, mom = _pl_seg(a2, mom, cur, _pl_perim_point(m, C, pin))
                    _pl_rec!(rec, _pl_perim_point(m, C, pin))
                    break
                end
                V = _pl_local(m, C, (@inbounds m.lines[e])[1])
                a2, mom = _pl_seg(a2, mom, cur, V)
                _pl_rec!(rec, V)
                cur = V
                e = Int(@inbounds m.elem_prev[e])
            end
            pin == 0 && return nraw, false
            if pin == p0
                closed = true
                break
            end
            _pl_bit(used, pin) && return nraw, false
            p = pin
        end
        closed || return nraw, false
        @inbounds m.racc[rs + nraw - 1] = PolylineRegionAcc{T}(a2, mom, Int16(0))
    end
    full = C.nc == 64 ? typemax(UInt64) : (one(UInt64) << C.nc) - one(UInt64)
    return nraw, used == full
end

# 4. Each island's pieces go to the region it lies in, and its area comes out of that region's.
function _pl_islands!(m, C::_PLCell{T}, rec=nothing) where {T}
    rs = Int(@inbounds m.region_start[C.s])
    for k in Int(@inbounds m.island_start[C.s]):(Int(@inbounds m.island_start[C.s + 1]) - 1)
        lp = Int(@inbounds m.island_loop[k])
        tg = Int(@inbounds m.island_target[k])
        r = 0
        if @inbounds(m.island_kind[k]) == 0
            for b in Int(@inbounds m.bseg_start[C.s]):(Int(@inbounds m.bseg_start[C.s + 1]) - 1)
                @inbounds m.bsegs.elem[b] == tg && (r = Int(m.bsegs.region[b]))
            end
        else
            q = MS_DIRECTION_EDGE[2]
            tg <= _pl_nfluid(C.n[q], C.fluid[q]) && (r = Int(@inbounds m.arcs.region[_pl_arc_slot(m, C, q, tg)]))
        end
        r > 0 || return false
        a2 = zero(T)
        mom = zero(SVector{2,T})
        _pl_rec_loop!(rec, r, true)
        e0 = Int(@inbounds m.loop_first[lp])
        e = e0
        while true
            ln = @inbounds m.lines[e]
            _pl_rec!(rec, _pl_local(m, C, ln[1]))
            # reversed: the fluid is outside an island
            a2, mom = _pl_seg(a2, mom, _pl_local(m, C, ln[2]), _pl_local(m, C, ln[1]))
            _pl_label_piece!(m, C, e, r) || return false
            e = Int(@inbounds m.elem_next[e])
            e == e0 && break
        end
        acc = @inbounds m.racc[rs + r - 1]
        @inbounds m.racc[rs + r - 1] = PolylineRegionAcc{T}(acc.area2 + a2, acc.mom + mom, Int16(0))
    end
    return true
end

# Every arc and piece of the cell owned by a walked region.
@inline function _pl_all_owned(m, C::_PLCell)
    a0 = Int(@inbounds m.arc_start[C.s])
    for k in a0:(Int(@inbounds m.arc_start[C.s + 1]) - 1)
        @inbounds m.arcs.region[k] == 0 && return false
    end
    for k in Int(@inbounds m.bseg_start[C.s]):(Int(@inbounds m.bseg_start[C.s + 1]) - 1)
        @inbounds m.bsegs.region[k] == 0 && return false
    end
    return true
end

# 5. Ranks: slivers get 0, the rest 1, 2, ... in walk order; arcs and pieces are relabelled.
function _pl_rank!(m, C::_PLCell{T}, nraw::Int, drop2::T) where {T}
    rs = Int(@inbounds m.region_start[C.s])
    nk = 0
    for r in 1:nraw
        acc = @inbounds m.racc[rs + r - 1]
        rank = 0
        if acc.area2 >= drop2
            nk += 1
            rank = nk
        end
        @inbounds m.racc[rs + r - 1] = PolylineRegionAcc{T}(acc.area2, acc.mom, Int16(rank))
    end
    for k in Int(@inbounds m.arc_start[C.s]):(Int(@inbounds m.arc_start[C.s + 1]) - 1)
        r = Int(@inbounds m.arcs.region[k])
        @inbounds m.arcs.region[k] = m.racc.rank[rs + r - 1]
    end
    for k in Int(@inbounds m.bseg_start[C.s]):(Int(@inbounds m.bseg_start[C.s + 1]) - 1)
        r = Int(@inbounds m.bsegs.region[k])
        @inbounds m.bsegs.region[k] = m.racc.rank[rs + r - 1]
    end
    return nk
end

# The fallback: the whole cell as one region, with the walk-free totals.
function _pl_lump!(m, C::_PLCell{T}, a2::T, mom::SVector{2,T}) where {T}
    rs = Int(@inbounds m.region_start[C.s])
    @inbounds m.racc[rs] = PolylineRegionAcc{T}(a2, mom, Int16(1))
    for k in Int(@inbounds m.arc_start[C.s]):(Int(@inbounds m.arc_start[C.s + 1]) - 1)
        @inbounds m.arcs.region[k] = Int16(1)
    end
    for k in Int(@inbounds m.bseg_start[C.s]):(Int(@inbounds m.bseg_start[C.s + 1]) - 1)
        @inbounds m.bsegs.region[k] = Int16(1)
    end
    return nothing
end

"""
    _pl_cut_cell!(m, g, blo, d, s, drop2)

The whole walk for cut slot `s`: arcs, pieces and walk-free totals, the walk, islands and ranks,
into `m`'s slot-indexed outputs. See this file's header.
"""
function _pl_cut_cell!(m, g::CartesianGrid{2,T}, blo::CartesianIndex{2}, d::NTuple{2,Int}, s::Int,
                       drop2::T, rec=nothing) where {T}
    C = _pl_cell(m, g, blo, d, s)
    _pl_init_arcs!(m, C)
    a2, mom = _pl_walk_free!(m, C)
    flags = @inbounds m.flags[C.l]
    nv = Int(@inbounds m.vbin_start[C.l + 1]) - Int(@inbounds m.vbin_start[C.l])
    ok = flags & (PL_FLAG_OVERFLOW | PL_FLAG_STATUS_CONFLICT) == 0
    nraw = 0
    if ok
        nraw, ok = _pl_walk!(m, C, nv, rec)
    end
    ok = ok && _pl_islands!(m, C, rec) && _pl_all_owned(m, C)
    if ok
        nregion = _pl_rank!(m, C, nraw, drop2)
    else
        _pl_lump!(m, C, a2, mom)
        nraw = nregion = 1
        flags & PL_FLAG_OVERFLOW == 0 && (flags |= PL_FLAG_WALK_FAIL)
    end
    @inbounds(m.island_start[s + 1] > m.island_start[s]) && (flags |= PL_FLAG_ISLAND)
    @inbounds m.sres[s] = PolylineSlotResult{T}(a2, mom, Int16(nraw), Int16(nregion), flags, ok)
    return nothing
end
