# =====================================
# The cut passes: Pass A's patch moments per candidate cell, then each cut cell's fit groups,
# Boolean rule, polytope and moments.

# The block's cells in linear order, `x` fastest.
@inline function _block_ci(blo::CartesianIndex{3}, bdims::NTuple{3,Int}, l::Int)
    q = l - 1
    i = q % bdims[1]
    q ÷= bdims[1]
    j = q % bdims[2]
    k = q ÷ bdims[2]
    return blo + CartesianIndex(i, j, k)
end

@inline _block_lin(blo::CartesianIndex{3}, bdims::NTuple{3,Int}, ci::CartesianIndex{3}) =
    1 + (ci[1] - blo[1]) + bdims[1] * ((ci[2] - blo[2]) + bdims[2] * (ci[3] - blo[3]))

# =====================================
# Pass A: each candidate cell's patch moments
#
# Every triangle in the cell's bin is clipped to the cell, closed on every side, and its piece's
# area `A`, `A n` and first moment are added to its patch's entry. The patch-boundary edges of the
# same triangles that pass within the refit margin of the cell leave their labels. A cell is cut
# when the pieces' total area is above `drop_area`.

# Whether segment `a b` meets the box `lo..hi` (slab test).
@inline function _segment_meets_box(a::SVector{3,T}, b::SVector{3,T}, lo::SVector{3,T},
                                    hi::SVector{3,T}) where {T}
    t0 = zero(T)
    t1 = one(T)
    d = b - a
    @inbounds for ax in 1:3
        if d[ax] == zero(T)
            (a[ax] < lo[ax] || a[ax] > hi[ax]) && return false
        else
            ta = (lo[ax] - a[ax]) / d[ax]
            tb = (hi[ax] - a[ax]) / d[ax]
            t0 = max(t0, min(ta, tb))
            t1 = min(t1, max(ta, tb))
            t0 > t1 && return false
        end
    end
    return true
end

@kernel function _tri_passA_kernel!(pa, m, scr, g::CartesianGrid{3,T}, blo::CartesianIndex{3},
                                    bdims::NTuple{3,Int}, tols, ns::Int, ncand::Int) where {T}
    s = @index(Global, Linear)
    for c in _slot_items(s, ns, ncand)
        _passA_cell!(pa, m, scr, s, c, g, blo, bdims, tols)
    end
end

function _passA_cell!(pa, m, scr, s, c::Int, g::CartesianGrid{3,T}, blo::CartesianIndex{3},
                      bdims::NTuple{3,Int}, tols) where {T}
    l = Int(@inbounds m.cand[c])
    ci = _block_ci(blo, bdims, l)
    o = get_node(g, ci)
    U = get_node(g, ci + CartesianIndex(1, 1, 1)) - o
    zero3 = zero(SVector{3,T})
    lo_grown = -tols.margin
    hi_grown = U + tols.margin
    base = (c - 1) * TRI_K_MAX
    pbase = (c - 1) * TRI_MAX_PAIRS
    np = 0
    npair = 0
    flags = 0x00
    area = zero(T)
    @inbounds for mi in Int(m.bin_start[l]):(Int(m.bin_start[l + 1]) - 1)
        t = Int(m.bin_tri[mi])
        cc = m.tris[t]
        a = m.verts[cc[1]] - o
        b = m.verts[cc[2]] - o
        d = m.verts[cc[3]] - o
        p = m.tri_patch[t]
        buf, nn = pg_tri_box!(scr, s, a, b, d, zero3, U, tols.snap)
        if nn < 0
            flags |= FLAG_OVERFLOW
        else
            nt = m.tri_normal[t]
            A, M = pg_area_moment(scr, s, buf, nn, nt)
            if A > zero(T)
                k = 0
                for i in 1:np
                    if pa.id[base + i] == p
                        k = i
                        break
                    end
                end
                if k == 0
                    if np < TRI_K_MAX
                        np += 1
                        k = np
                        pa.id[base + k] = p
                        pa.A[base + k] = zero(T)
                        pa.N[base + k] = zero3
                        pa.M[base + k] = zero3
                    else
                        flags |= FLAG_OVERFLOW
                    end
                end
                if k > 0
                    pa.A[base + k] += A
                    pa.N[base + k] += A * nt
                    pa.M[base + k] += M
                end
                area += A
            end
        end
        sl = m.side_label[t]
        sp = m.side_patch[t]
        verts3 = (a, b, d)
        for e in 1:3
            lab = sl[e] & 0x07
            lab == 0x00 && continue
            _segment_meets_box(verts3[e], verts3[e % 3 + 1], lo_grown, hi_grown) || continue
            q = sp[e]
            lo = min(p, q)
            hi = max(p, q)
            k = 0
            for i in 1:npair
                if pa.pp[pbase + i] == lo && pa.pq[pbase + i] == hi
                    k = i
                    break
                end
            end
            if k == 0
                if npair < TRI_MAX_PAIRS
                    npair += 1
                    k = npair
                    pa.pp[pbase + k] = lo
                    pa.pq[pbase + k] = hi
                    pa.pm[pbase + k] = 0x00
                else
                    flags |= FLAG_OVERFLOW
                end
            end
            k > 0 && (pa.pm[pbase + k] |= lab)
        end
    end
    @inbounds pa.n[c] = Int8(np)
    @inbounds pa.npair[c] = Int8(npair)
    @inbounds pa.flag[c] = flags
    @inbounds pa.area[c] = area
    return nothing
end

