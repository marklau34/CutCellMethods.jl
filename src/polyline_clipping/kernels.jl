# =====================================
# The kernels: resetting cells and edges, and writing what the host classified
#
# One workitem per cell or edge. Everything combinatorial was decided on the host (`crossings.jl`)
# and arrives as block-sized arrays; the kernels only turn it into records, so the output is the
# same on every backend and for every thread count.

# The in-plane offset of each of a cell's four face centres from its lower lattice corner -- the
# centroid a fully open or fully covered face stores. Taken off lattice differences, the expression
# `_pl_edge_half` uses for the edge on its own, so the two cells sharing a face and the edge array
# all hold bitwise the same number.
@inline function _pl_face_centres_local(g::CartesianGrid{2,T}, ci::CartesianIndex{2}) where {T}
    lo = get_node(g, ci)
    half = T(0.5) * (get_node(g, ci + CartesianIndex(1, 1)) - lo)
    return SVector{4,SVector{1,T}}(SVector{1,T}(half[2]), SVector{1,T}(half[2]),
                                   SVector{1,T}(half[1]), SVector{1,T}(half[1]))
end

# Half the length of one edge, off the lattice: an x-edge `(i, j)` runs up x-line `i` from node
# `(i, j)`, a y-edge `(i, j)` along y-line `j` from node `(i, j)`.
@inline function _pl_edge_half(g::CartesianGrid{2,T}, ei::CartesianIndex{2}, axis::Int) where {T}
    lo = get_node(g, ei)
    if axis == 1
        return T(0.5) * (get_node(g, ei + CartesianIndex(0, 1))[2] - lo[2])
    else
        return T(0.5) * (get_node(g, ei + CartesianIndex(1, 0))[1] - lo[1])
    end
end

# The record of a cell the polylines do not cut: wholly fluid or wholly solid, every face alike.
@inline function _pl_uncut_cell(g::CartesianGrid{2,T}, ci::CartesianIndex{2}, kind::Int8) where {T}
    vf = kind == CELL_OUTSIDE ? one(T) : zero(T)
    centre = get_node(g, ci) + T(0.5) * g.d
    return CutCellData{2,T,4,1}(kind, false, vf, centre, SVector{4,T}(vf, vf, vf, vf),
                                _pl_face_centres_local(g, ci), centre)
end

@inline _pl_info(::Type{T}, status::UInt8, nregion::Integer, flags::UInt8, slot::Integer) where {T} =
    PolylineCellInfo{T}(status, Int16(nregion), flags, Int32(slot), Int32(0), zero(SVector{2,T}))

@kernel function _pl_reset_cells_kernel!(cells, info, g::CartesianGrid{2,T},
                                         lo::CartesianIndex{2}) where {T}
    I = @index(Global, Cartesian)
    ci = I + lo - oneunit(lo)
    @inbounds cells[ci] = _pl_uncut_cell(g, ci, CELL_OUTSIDE)
    @inbounds info[ci] = _pl_info(T, PL_FLUID, 1, 0x00, 0)
end

@kernel function _pl_reset_edges_kernel!(edges, g::CartesianGrid{2,T}, lo::CartesianIndex{2},
                                         axis::Int) where {T}
    I = @index(Global, Cartesian)
    ei = I + lo - oneunit(lo)
    @inbounds edges[ei] = PolylineEdge{T}(one(T), _pl_edge_half(g, ei, axis))
end

# Every block cell from the host's classification. A cut cell is marked here and measured by the
# walk (P3); until then it holds a fluid cell's numbers under `CELL_CUT`.
@kernel function _pl_classify_kernel!(cells, info, @Const(status), @Const(flags), @Const(slot),
                                      g::CartesianGrid{2,T}, lo::CartesianIndex{2},
                                      nx::Int) where {T}
    I = @index(Global, Cartesian)
    ci = I + lo - oneunit(lo)
    l = I[1] + nx * (I[2] - 1)
    @inbounds s = status[l]
    if s == PL_SOLID
        @inbounds cells[ci] = _pl_uncut_cell(g, ci, CELL_INSIDE)
        @inbounds info[ci] = _pl_info(T, s, 0, flags[l], 0)
    elseif s == PL_FLUID
        @inbounds cells[ci] = _pl_uncut_cell(g, ci, CELL_OUTSIDE)
        @inbounds info[ci] = _pl_info(T, s, 1, flags[l], 0)
    else
        m = _pl_uncut_cell(g, ci, CELL_OUTSIDE)
        @inbounds cells[ci] = CutCellData{2,T,4,1}(CELL_CUT, m.ambiguous, m.volume_fraction,
                                                   m.centroid, m.face_fraction,
                                                   m.face_centroid_local, m.interface_centroid)
        @inbounds info[ci] = _pl_info(T, s, 1, flags[l], slot[l])
    end
end

# Every block edge that carries no crossing: open or closed as its lower node. (Its other node
# agrees, since the walks alternate.) Edges with crossings are the cut pass's (P3).
@kernel function _pl_uncut_edges_kernel!(edges, @Const(node_solid), @Const(edge_start),
                                         g::CartesianGrid{2,T}, lo::CartesianIndex{2},
                                         nx::Int, base::Int, stride::Int, axis::Int) where {T}
    I = @index(Global, Cartesian)
    ei = I + lo - oneunit(lo)
    m = base + I[1] + stride * (I[2] - 1)
    @inbounds if edge_start[m + 1] == edge_start[m]
        solid = node_solid[I[1] + (nx + 1) * (I[2] - 1)]
        edges[ei] = PolylineEdge{T}(solid ? zero(T) : one(T), _pl_edge_half(g, ei, axis))
    end
end

# =====================================
# The cut: the walk, the edge pass and the finalize pass

@kernel function _pl_walk_kernel!(m, g::CartesianGrid{2,T}, blo::CartesianIndex{2}, d::NTuple{2,Int},
                                  ncut::Int, drop2::T) where {T}
    s = @index(Global, Linear)
    if s <= ncut
        _pl_cut_cell!(m, g, blo, d, s, drop2)
    end
end

# The cell on either side of a block edge, and the direction that edge is to each: an x-edge `(i, j)`
# is the right edge (2) of cell `(i-1, j)` and the left edge (1) of `(i, j)`; a y-edge the top (4)
# of `(i, j-1)` and the bottom (3) of `(i, j)`.
@inline _pl_edge_sides(ei::CartesianIndex{2}, axis::Int) =
    axis == 1 ? (ei - CartesianIndex(1, 0), 2, ei, 1) : (ei - CartesianIndex(0, 1), 4, ei, 3)

# One side of an edge: whether its cell is cut, and if so where that cell's arcs on this edge start.
@inline function _pl_side_arcs(m, info, g, blo, d, ci::CartesianIndex{2}, dir::Int)
    checkbounds(Bool, info, ci) || return false, 0
    @inbounds inf = info[ci]
    inf.status == PL_CUT || return false, 0
    C = _pl_cell(m, g, blo, d, Int(inf.slot))
    return true, Int(@inbounds m.arc_start[C.s]) + _pl_arc_off(C, dir)
end

"""
    _pl_cut_edge!(edges, info, m, g, blo, d, axis, I)

One block edge beside a cut cell, once:

- the region each of its fluid intervals belongs to on each side, recorded in both cells' arcs;
- the intervals a sliver owns on either side, closed on both;
- the edge's aperture and open-part centroid, summed over its intervals in order along it, so both
  cells copy one number.

An edge whose intervals of one kind all have zero length is exactly closed or exactly open: a
polyline touching a line leaves a zero-length solid interval, and that must not cost the edge an ulp
of its aperture.
"""
function _pl_cut_edge!(edges, info, m, g::CartesianGrid{2,T}, blo::CartesianIndex{2},
                       d::NTuple{2,Int}, axis::Int, I::CartesianIndex{2}) where {T}
    ei = I + blo - oneunit(blo)
    cA, dA, cB, dB = _pl_edge_sides(ei, axis)
    cutA, baseA = _pl_side_arcs(m, info, g, blo, d, cA, dA)
    cutB, baseB = _pl_side_arcs(m, info, g, blo, d, cB, dB)
    (cutA || cutB) || return nothing
    me = axis == 1 ? _pl_xedge_lin(d, I[1], I[2]) : _pl_yedge_lin(d, I[1], I[2])
    start = Int(@inbounds m.edge_start[me])
    n = Int(@inbounds m.edge_start[me + 1]) - start
    fluid = !(@inbounds m.node_solid[_pl_node_lin(d, I[1], I[2])])
    lo = get_node(g, ei)
    len = axis == 1 ? get_node(g, ei + CartesianIndex(0, 1))[2] - lo[2] :
                      get_node(g, ei + CartesianIndex(1, 0))[1] - lo[1]
    open = zero(T)
    omom = zero(T)
    any_open = any_solid = false
    for iv in 0:n
        s0, s1 = _pl_interval(m.cr_offset, start, n, iv, len)
        closed = true
        if _pl_is_fluid(iv, fluid)
            k = _pl_fluid_k(iv, fluid)
            rA = cutA ? Int(@inbounds m.arcs.region[baseA + k - 1]) : 1
            rB = cutB ? Int(@inbounds m.arcs.region[baseB + k - 1]) : 1
            closed = rA == 0 || rB == 0
            if cutA
                @inbounds m.arcs.nbr_region[baseA + k - 1] = Int16(rB)
                @inbounds m.arcs.closed[baseA + k - 1] = closed
            end
            if cutB
                @inbounds m.arcs.nbr_region[baseB + k - 1] = Int16(rA)
                @inbounds m.arcs.closed[baseB + k - 1] = closed
            end
        end
        if closed
            any_solid |= s1 > s0
        else
            open += s1 - s0
            omom += (s1 - s0) * (s0 + s1) / 2
            any_open |= s1 > s0
        end
    end
    frac = !any_open ? zero(T) : !any_solid ? one(T) : open / len
    off = (frac == zero(T) || frac == one(T)) ? _pl_edge_half(g, ei, axis) : omom / open
    @inbounds edges[ei] = PolylineEdge{T}(frac, off)
    return nothing
end

@kernel function _pl_edge_kernel!(edges, info, m, g::CartesianGrid{2,T}, blo::CartesianIndex{2},
                                  d::NTuple{2,Int}, axis::Int) where {T}
    I = @index(Global, Cartesian)
    _pl_cut_edge!(edges, info, m, g, blo, d, axis, I)
end

# A cell's four edges from the edge arrays, by direction.
@inline _pl_cell_edges(ax, ay, ci::CartesianIndex{2}) =
    @inbounds (ax[ci], ax[ci + CartesianIndex(1, 0)], ay[ci], ay[ci + CartesianIndex(0, 1)])

# Outward normal of face direction `dir`, and the local point at offset `s` along that face.
@inline _pl_nout(::Type{T}, dir::Integer) where {T} =
    dir == 1 ? SVector{2,T}(-1, 0) : dir == 2 ? SVector{2,T}(1, 0) :
    dir == 3 ? SVector{2,T}(0, -1) : SVector{2,T}(0, 1)
@inline _pl_dir_point(C::_PLCell, dir::Integer, s) = _pl_edge_point(C, MS_DIRECTION_EDGE[dir], s)

# Region `r`'s sums over the cell's arcs and pieces (`r = 0`: every kept region): open arc length
# and first moment per direction, and the wall's length, first moment and vector area `sum n L`
# with `n` into the fluid -- its pieces, and the arcs a neighbour's sliver closed onto it.
function _pl_region_sums(m, C::_PLCell{T}, r::Int) where {T}
    elen = zero(SVector{4,T})
    emom = zero(SVector{4,T})
    wlen = zero(T)
    wmom = zero(SVector{2,T})
    wvec = zero(SVector{2,T})
    for k in Int(@inbounds m.arc_start[C.s]):(Int(@inbounds m.arc_start[C.s + 1]) - 1)
        a = @inbounds m.arcs[k]
        (r == 0 ? a.region > 0 : a.region == r) || continue
        l = a.s1 - a.s0
        if a.closed
            wlen += l
            wmom += l * (C.lo + _pl_dir_point(C, a.dir, (a.s0 + a.s1) / 2))
            wvec -= l * _pl_nout(T, a.dir)
        else
            elen = setindex(elen, elen[a.dir] + l, Int(a.dir))
            emom = setindex(emom, emom[a.dir] + l * (a.s0 + a.s1) / 2, Int(a.dir))
        end
    end
    for k in Int(@inbounds m.bseg_start[C.s]):(Int(@inbounds m.bseg_start[C.s + 1]) - 1)
        b = @inbounds m.bsegs[k]
        (r == 0 ? b.region > 0 : b.region == r) || continue
        wlen += b.length
        wmom += b.length * b.midpoint
        wvec += b.length * b.normal
    end
    return elen, emom, wlen, wmom, wvec
end

# `sum_e n_e l_e - sum_s n_s L_s`, with the faces' outward normals and the wall's into the fluid.
@inline _pl_residual(::Type{T}, elen::SVector{4,T}, wvec::SVector{2,T}) where {T} =
    SVector{2,T}(elen[2] - elen[1], elen[4] - elen[3]) - wvec

"""
    _pl_finalize_cell!(cells, info, regions, rinfo, ax, ay, m, g, blo, d, tols, ci)

One cell of the finalize list, from its edges and, if cut, its walk:

- a cut cell gets its totals -- the walk-free area and centroid, the edges' apertures, the wall's
  centroid -- and, if split, one record per kept region. A cell all of whose fluid was slivers is
  exactly solid, and one whose wall has no length exactly fluid.
- an uncut fluid cell is left alone unless a neighbour's sliver closed one of its edges, which then
  becomes its wall.
"""
function _pl_finalize_cell!(cells, info, regions, rinfo, ax, ay, m, g::CartesianGrid{2,T},
                            blo::CartesianIndex{2}, d::NTuple{2,Int}, tols::PolylineTols{T},
                            ci::CartesianIndex{2}) where {T}
    old = @inbounds info[ci]
    ef = _pl_cell_edges(ax, ay, ci)
    lo = get_node(g, ci)
    h = get_node(g, ci + CartesianIndex(1, 1)) - lo
    centre = lo + T(0.5) * g.d
    len4 = SVector{4,T}(h[2], h[2], h[1], h[1])
    ff = SVector{4,T}(ntuple(k -> ef[k].fraction, Val(4)))
    fcl = SVector{4,SVector{1,T}}(ntuple(k -> SVector{1,T}(ef[k].offset), Val(4)))

    if old.status != PL_CUT
        v = old.status == PL_FLUID ? one(T) : zero(T)
        (ff[1] == v && ff[2] == v && ff[3] == v && ff[4] == v) && return nothing
        # A fluid cell whose edge a neighbour's sliver closed: that stretch of the edge is its wall.
        wlen = zero(T)
        wmom = zero(SVector{2,T})
        wvec = zero(SVector{2,T})
        elen = zero(SVector{4,T})
        for dir in 1:4
            L = len4[dir]
            ol = ff[dir] * L
            cl = L - ol
            elen = setindex(elen, ol, dir)
            cl > zero(T) || continue
            s = (L * L / 2 - ol * ef[dir].offset) / cl
            wlen += cl
            wmom += cl * (lo + _pl_edge_point(_pl_cell_frame(lo, h), MS_DIRECTION_EDGE[dir], s))
            wvec -= cl * _pl_nout(T, dir)
        end
        @inbounds cells[ci] = CutCellData{2,T,4,1}(CELL_CUT, false, one(T), centre, ff, fcl,
                                                   wlen > zero(T) ? wmom / wlen : centre)
        @inbounds info[ci] = PolylineCellInfo{T}(PL_CUT, Int16(1), old.flags | PL_FLAG_SNAPPED,
                                                 Int32(0), Int32(0), _pl_residual(T, elen, wvec))
        return nothing
    end

    s = Int(old.slot)
    res = @inbounds m.sres[s]
    C = _pl_cell(m, g, blo, d, s)
    flags = old.flags | res.flags
    nreg = Int(res.nregion)
    elen, _, wlen, wmom, wvec = _pl_region_sums(m, C, 0)
    for k in Int(@inbounds m.arc_start[s]):(Int(@inbounds m.arc_start[s + 1]) - 1)
        @inbounds m.arcs.closed[k] && (flags |= PL_FLAG_SNAPPED)
    end

    if res.ok && nreg == 0
        # Every fluid region was a sliver, closed onto its neighbours: exactly solid.
        @inbounds cells[ci] = CutCellData{2,T,4,1}(CELL_INSIDE, false, zero(T), centre, ff, fcl, centre)
        @inbounds info[ci] = PolylineCellInfo{T}(PL_SOLID, Int16(0), flags, Int32(s), Int32(0),
                                                 zero(SVector{2,T}))
        return nothing
    end
    if res.ok && nreg == 1 && wlen == zero(T) &&
       ff[1] == one(T) && ff[2] == one(T) && ff[3] == one(T) && ff[4] == one(T)
        # The polyline only touches the cell: exactly fluid.
        @inbounds cells[ci] = CutCellData{2,T,4,1}(CELL_OUTSIDE, false, one(T), centre, ff, fcl, centre)
        @inbounds info[ci] = PolylineCellInfo{T}(PL_FLUID, Int16(1), flags, Int32(s), Int32(0),
                                                 zero(SVector{2,T}))
        return nothing
    end

    area = h[1] * h[2]
    vf = clamp(res.area2 / (2 * area), zero(T), one(T))
    centroid = res.area2 > zero(T) ? lo + res.mom / (3 * res.area2) : centre
    island = flags & PL_FLAG_ISLAND != 0
    @inbounds cells[ci] = CutCellData{2,T,4,1}(CELL_CUT, nreg >= 2 || island || !res.ok, vf,
                                               centroid, ff, fcl,
                                               wlen > zero(T) ? wmom / wlen : centre)

    # Checks against the walk: its arcs make up the apertures, its areas the walk-free total.
    rs = Int(@inbounds m.region_start[s])
    if res.ok
        for dir in 1:4
            abs(elen[dir] - ff[dir] * len4[dir]) <= 16 * eps(T) * len4[dir] ||
                (flags |= PL_FLAG_APERTURE_MISMATCH)
        end
        a2 = zero(T)
        for r in 1:Int(res.nraw)
            a2 += @inbounds m.racc.area2[rs + r - 1]
        end
        abs(a2 - res.area2) <= 64 * eps(T) * area || (flags |= PL_FLAG_WALK_MISMATCH)
    end

    rslot = nreg >= 2 ? Int(@inbounds m.rslot[s]) : 0
    residual = _pl_residual(T, elen, wvec)
    if nreg >= 2
        worst = zero(SVector{2,T})
        for r in 1:Int(res.nraw)
            acc = @inbounds m.racc[rs + r - 1]
            acc.rank > 0 || continue
            relen, remom, rwlen, rwmom, rwvec = _pl_region_sums(m, C, Int(acc.rank))
            rres = _pl_residual(T, relen, rwvec)
            rflags = maximum(abs, rres) > tols.closure_tol ? PL_FLAG_CLOSURE : 0x00
            flags |= rflags
            maximum(abs, rres) > maximum(abs, worst) && (worst = rres)
            rc = acc.area2 > zero(T) ? lo + acc.mom / (3 * acc.area2) : centre
            rff = SVector{4,T}(ntuple(k -> relen[k] / len4[k], Val(4)))
            rfcl = SVector{4,SVector{1,T}}(ntuple(k -> SVector{1,T}(relen[k] > zero(T) ?
                                                   remom[k] / relen[k] : len4[k] / 2), Val(4)))
            j = rslot + Int(acc.rank) - 1
            @inbounds regions[j] = CutCellData{2,T,4,1}(CELL_CUT, false,
                                                        clamp(acc.area2 / (2 * area), zero(T), one(T)),
                                                        rc, rff, rfcl, rwlen > zero(T) ? rwmom / rwlen : rc)
            @inbounds rinfo[j] = PolylineRegionInfo{T}(ci, acc.rank, rwvec, rres, rflags)
        end
        residual = worst
    else
        maximum(abs, residual) > tols.closure_tol && (flags |= PL_FLAG_CLOSURE)
    end
    status = !res.ok ? PL_INVALID : nreg >= 2 ? PL_SPLIT : PL_CUT
    @inbounds info[ci] = PolylineCellInfo{T}(status, Int16(nreg), flags, Int32(s), Int32(rslot),
                                             residual)
    return nothing
end

# A frame-only `_PLCell`, for placing points on the edges of a cell that has no walk.
@inline _pl_cell_frame(lo::SVector{2,T}, h::SVector{2,T}) where {T} =
    _PLCell{T}(0, 0, lo, h, (0, 0, 0, 0), (0, 0, 0, 0), (true, true, true, true), 0)

@kernel function _pl_finalize_kernel!(cells, info, regions, rinfo, ax, ay, m, g::CartesianGrid{2,T},
                                      blo::CartesianIndex{2}, d::NTuple{2,Int},
                                      tols::PolylineTols{T}, @Const(fin_list), nfin::Int) where {T}
    k = @index(Global, Linear)
    if k <= nfin
        _pl_finalize_cell!(cells, info, regions, rinfo, ax, ay, m, g, blo, d, tols,
                           @inbounds fin_list[k])
    end
end
