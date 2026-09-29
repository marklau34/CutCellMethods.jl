# =====================================
# Fit groups and their planes
#
# Pass A leaves, per candidate cell, a table of the patches whose triangles have area in it -- each
# with its clipped area `A`, `sum A n` and `sum A c` in the cell's local frame -- and the labels of
# the patch-boundary edges that pass near it. Groups are built from an *entry table* in the slot's
# scratch: one cell's own table, for its polytope, or the union of two neighbours' tables, for the
# face they share (`_merge_entries!`). Either way:
#
#  * patches meeting at a tangent edge (a CAD seam across one smooth surface) are merged, since two
#    nearly parallel planes from one surface would make the Boolean rule's choice arbitrary;
#  * each group gets one plane: its patch's global plane when it is a single planar patch -- what
#    makes a planar corner exact -- else the area-weighted fit `n = sum A n / |sum A n|` through the
#    area-weighted centroid;
#  * a group with less than `sliver_area` in the box is refit over the box grown by the margin,
#    and dropped if it is still a sliver -- unless it is the largest, or planar, since a global
#    plane is exact however little of it the box holds.
#
# How two patches meet is read from the edges near the box first -- the labels Pass A recorded for
# the one or two cells involved -- and from the mesh-wide label only when none of the pair's edges
# comes near: a pair can be convex along one stretch of its boundary and concave along another.
#
# A table built for a face is a pure function of its two cells taken low then high, so both cells
# build the same one and cut their shared face identically, to the bit.

# Candidate `c`'s local label for patches `(lo, hi)`, or 0.
@inline function _local_label(pa, c::Int, lo::Integer, hi::Integer)
    c == 0 && return 0x00
    base = (c - 1) * TRI_MAX_PAIRS
    @inbounds for i in 1:Int(pa.npair[c])
        if pa.pp[base + i] == lo && pa.pq[base + i] == hi
            return pa.pm[base + i] & 0x07
        end
    end
    return 0x00
end

# The label of patches `p` and `q`: what the edges near candidates `ca` and `cb` (0 for none) say,
# else the mesh-wide one. Only the convex/concave/tangent bits.
@inline function _pair_label(pa, m, ca::Int, cb::Int, p::Integer, q::Integer, np_total::Int)
    lo = min(p, q)
    hi = max(p, q)
    lab = _local_label(pa, ca, lo, hi) | _local_label(pa, cb, lo, hi)
    lab != 0x00 && return lab
    return @inbounds m.pair_mask[Int(lo) + np_total * (Int(hi) - 1)] & 0x07
end

"""
    _load_entries!(scr, s, pa, c) -> n

Candidate `c`'s patch table into slot `s`'s entry table.
"""
@inline function _load_entries!(scr, s, pa, c::Int)
    n = Int(@inbounds pa.n[c])
    base = (c - 1) * TRI_K_MAX
    @inbounds for i in 1:n
        scr.eid[i, s] = pa.id[base + i]
        scr.eA[i, s] = pa.A[base + i]
        scr.eN[i, s] = pa.N[base + i]
        scr.eM[i, s] = pa.M[base + i]
    end
    return n
end

"""
    _merge_entries!(scr, s, pa, cl, ch, shift) -> (n, overflow)

The union of candidates `cl` (low) and `ch` (high)'s patch tables, in `ch`'s frame: `cl`'s first
moments move by `A * shift`, `shift` being `cl`'s origin less `ch`'s. `cl`'s entries come first, in
their order, then `ch`'s new ones in theirs, so the table depends only on the ordered pair.
"""
function _merge_entries!(scr, s, pa, cl::Int, ch::Int, shift::SVector{3,T}) where {T}
    n = 0
    overflow = false
    bl = (cl - 1) * TRI_K_MAX
    @inbounds for i in 1:Int(pa.n[cl])
        n += 1
        A = pa.A[bl + i]
        scr.eid[n, s] = pa.id[bl + i]
        scr.eA[n, s] = A
        scr.eN[n, s] = pa.N[bl + i]
        scr.eM[n, s] = pa.M[bl + i] + A * shift
    end
    bh = (ch - 1) * TRI_K_MAX
    @inbounds for j in 1:Int(pa.n[ch])
        p = pa.id[bh + j]
        k = 0
        for i in 1:n
            if scr.eid[i, s] == p
                k = i
                break
            end
        end
        if k == 0
            if n < TRI_K_MAX
                n += 1
                k = n
                scr.eid[k, s] = p
                scr.eA[k, s] = zero(T)
                scr.eN[k, s] = zero(SVector{3,T})
                scr.eM[k, s] = zero(SVector{3,T})
            else
                overflow = true
                continue
            end
        end
        scr.eA[k, s] += pa.A[bh + j]
        scr.eN[k, s] += pa.N[bh + j]
        scr.eM[k, s] += pa.M[bh + j]
    end
    return n, overflow
end

@inline function _uf_find(scr, s, i::Int)
    @inbounds while Int(scr.gpar[i, s]) != i
        i = Int(scr.gpar[i, s])
    end
    return i
end

"""
    _build_groups!(scr, s, pa, m, ca, cb, n, np_total) -> ngroups

Merge slot `s`'s `n` entries across tangent pairs into fit groups: `gmem` (member entries as bits),
and the summed `garea`, `gnrm`, `gmom`. Groups are numbered in order of their first entry and summed
in entry order, so the result is deterministic.
"""
function _build_groups!(scr, s, pa, m, ca::Int, cb::Int, n::Int, np_total::Int)
    @inbounds for i in 1:n
        scr.gpar[i, s] = Int8(i)
    end
    @inbounds for i in 1:n, j in (i + 1):n
        if _pair_label(pa, m, ca, cb, scr.eid[i, s], scr.eid[j, s], np_total) == PAIR_TANGENT
            ri = _uf_find(scr, s, i)
            rj = _uf_find(scr, s, j)
            ri != rj && (scr.gpar[max(ri, rj), s] = Int8(min(ri, rj)))
        end
    end
    # Point every entry straight at its root. A union always keeps the smaller index as the root,
    # so an entry's root comes no later than the entry itself.
    @inbounds for i in 1:n
        scr.gpar[i, s] = Int8(_uf_find(scr, s, i))
    end
    T = eltype(scr.garea)
    ng = 0
    gid = ntuple(_ -> 0, Val(TRI_K_MAX))
    @inbounds for i in 1:n
        r = Int(scr.gpar[i, s])
        if gid[r] == 0
            ng += 1
            gid = Base.setindex(gid, ng, r)
            scr.gmem[ng, s] = 0x00
            scr.garea[ng, s] = zero(T)
            scr.gnrm[ng, s] = zero(SVector{3,T})
            scr.gmom[ng, s] = zero(SVector{3,T})
        end
        g = gid[r]
        scr.gmem[g, s] |= UInt8(1) << (i - 1)
        scr.garea[g, s] += scr.eA[i, s]
        scr.gnrm[g, s] += scr.eN[i, s]
        scr.gmom[g, s] += scr.eM[i, s]
    end
    return ng
end

# Group `g`'s member patch with the most area, and whether it is the only member.
@inline function _group_patch(scr, s, g::Int)
    mem = @inbounds scr.gmem[g, s]
    best = trailing_zeros(mem) + 1
    @inbounds for i in (best + 1):TRI_K_MAX
        (mem >> (i - 1)) & 0x01 != 0 && scr.eA[i, s] > scr.eA[best, s] && (best = i)
    end
    return @inbounds scr.eid[best, s], count_ones(mem) == 1
end

# Whether patch `p` is one of group `g`'s members.
@inline function _in_group(scr, s, g::Int, p::Integer)
    mem = @inbounds scr.gmem[g, s]
    @inbounds for i in 1:TRI_K_MAX
        (mem >> (i - 1)) & 0x01 != 0 && scr.eid[i, s] == p && return true
    end
    return false
end

# The bin of candidate `c`, as a range into `bin_tri`; empty for `c == 0`.
@inline function _cand_bin(m, c::Int)
    c == 0 && return 1:0
    l = Int(@inbounds m.cand[c])
    return Int(@inbounds m.bin_start[l]):(Int(@inbounds m.bin_start[l + 1]) - 1)
end

# Clip triangle `t`, moved into the frame whose origin is `o`, to `lo..hi` and add its piece to
# `(A, N, M)`.
@inline function _add_piece(scr, s, m, t::Int, o::SVector{3,T}, lo::SVector{3,T}, hi::SVector{3,T},
                            tol::T, A::T, N::SVector{3,T}, M::SVector{3,T}) where {T}
    cc = @inbounds m.tris[t]
    buf, nn = pg_tri_box!(scr, s, @inbounds(m.verts[cc[1]]) - o, @inbounds(m.verts[cc[2]]) - o,
                          @inbounds(m.verts[cc[3]]) - o, lo, hi, tol)
    nn < 0 && return A, N, M
    nt = @inbounds m.tri_normal[t]
    a, mom = pg_area_moment(scr, s, buf, nn, nt)
    a > zero(T) || return A, N, M
    return A + a, N + a * nt, M + mom
end

# Re-clip group `g`'s member patches' triangles over `lo..hi` in the frame at `o`, from the bins
# of candidates `ca` and `cb` -- merged by triangle id, both being sorted, so a triangle in both
# counts once. The margin the bins were built with covers any box within it of the two cells.
function _refit_group(scr, s, m, ca::Int, cb::Int, g::Int, o::SVector{3,T}, lo::SVector{3,T},
                      hi::SVector{3,T}, tol::T) where {T}
    A = zero(T)
    N = zero(SVector{3,T})
    M = zero(SVector{3,T})
    ra = _cand_bin(m, ca)
    rb = _cand_bin(m, cb)
    i = first(ra)
    j = first(rb)
    @inbounds while i <= last(ra) || j <= last(rb)
        ta = i <= last(ra) ? Int(m.bin_tri[i]) : typemax(Int)
        tb = j <= last(rb) ? Int(m.bin_tri[j]) : typemax(Int)
        t = min(ta, tb)
        ta == t && (i += 1)
        tb == t && (j += 1)
        _in_group(scr, s, g, m.tri_patch[t]) || continue
        A, N, M = _add_piece(scr, s, m, t, o, lo, hi, tol, A, N, M)
    end
    return A, N, M
end

"""
    _group_planes!(scr, s, m, ca, cb, ng, o, lo, hi, tols) -> (dropped, planar, flags)

A plane for each of the `ng` fit groups, into `gplane`, in the frame whose origin is `o` and in
which the groups were fitted over the box `lo..hi`. Returns the groups a sliver refit could not
save (`dropped`, as bits), those cut by a global plane (`planar`), and `FLAG_UNSUPPORTED` when a
group folds back on itself within the box.
"""
function _group_planes!(scr, s, m, ca::Int, cb::Int, ng::Int, o::SVector{3,T}, lo::SVector{3,T},
                        hi::SVector{3,T}, tols) where {T}
    dropped = 0x00
    planar = 0x00
    flags = 0x00
    @inbounds for g in 1:ng
        p, single = _group_patch(scr, s, g)
        if single && m.patch_planar[p]
            gp = m.patch_plane[p]
            n = SVector{3,T}(gp[1], gp[2], gp[3])
            scr.gplane[g, s] = SVector{4,T}(n[1], n[2], n[3], gp[4] - dot(n, o))
            planar |= UInt8(1) << (g - 1)
            continue
        end
        A = scr.garea[g, s]
        N = scr.gnrm[g, s]
        M = scr.gmom[g, s]
        if A < tols.sliver_area
            A2, N2, M2 = _refit_group(scr, s, m, ca, cb, g, o, lo - tols.margin, hi + tols.margin,
                                      tols.snap)
            if A2 > A
                A, N, M = A2, N2, M2
            end
            A < tols.sliver_area && (dropped |= UInt8(1) << (g - 1))
        end
        nN = sqrt(sum(abs2, N))
        nN < A / 2 && (flags |= FLAG_UNSUPPORTED)
        n = nN > zero(T) ? N / nN : SVector{3,T}(0, 0, 1)
        d = A > zero(T) ? dot(n, M / A) : zero(T)
        scr.gplane[g, s] = SVector{4,T}(n[1], n[2], n[3], d)
    end
    return dropped, planar, flags
end

# The largest group by clipped area.
@inline function _largest_group(scr, s, ng::Int)
    best = 1
    @inbounds for g in 2:ng
        scr.garea[g, s] > scr.garea[best, s] && (best = g)
    end
    return best
end

"""
    _fallback_plane(scr, s, ng) -> plane

One plane for a box no Boolean rule fits: the area-weighted fit over every group together, or the
largest group's own plane when the groups' normals cancel (a thin feature, where the combined fit
has no direction).
"""
function _fallback_plane(scr, s, ng::Int)
    T = eltype(scr.garea)
    A = zero(T)
    N = zero(SVector{3,T})
    M = zero(SVector{3,T})
    @inbounds for g in 1:ng
        A += scr.garea[g, s]
        N += scr.gnrm[g, s]
        M += scr.gmom[g, s]
    end
    nN = sqrt(sum(abs2, N))
    if nN < A / 2 || A <= zero(T)
        return @inbounds scr.gplane[_largest_group(scr, s, ng), s]
    end
    n = N / nN
    return SVector{4,T}(n[1], n[2], n[3], dot(n, M / A))
end

# Keep the groups a sliver refit could not save only when planar or the largest; compact the rest
# into slots `1..k`.
function _compact_groups!(scr, s, ng::Int, dropped::UInt8, planar::UInt8)
    largest = _largest_group(scr, s, ng)
    k = 0
    @inbounds for g in 1:ng
        drop = (dropped >> (g - 1)) & 0x01 != 0 && (planar >> (g - 1)) & 0x01 == 0 && g != largest
        drop && continue
        k += 1
        if k != g
            scr.gplane[k, s] = scr.gplane[g, s]
            scr.garea[k, s] = scr.garea[g, s]
            scr.gnrm[k, s] = scr.gnrm[g, s]
            scr.gmom[k, s] = scr.gmom[g, s]
            scr.gmem[k, s] = scr.gmem[g, s]
        end
    end
    return k
end

"""
    _fit!(scr, s, pa, m, ca, cb, n, o, lo, hi, tols, np_total) -> (k, rule, flags)

From slot `s`'s `n` entries to the planes a box is cut by: fit groups, their planes, the Boolean
rule, and -- when no rule fits -- one fallback plane. Leaves the `k` planes to cut with in
`gplane[1:k]` (and their groups in the other group arrays).
"""
function _fit!(scr, s, pa, m, ca::Int, cb::Int, n::Int, o::SVector{3,T}, lo::SVector{3,T},
               hi::SVector{3,T}, tols, np_total::Int, flags::UInt8) where {T}
    ng = _build_groups!(scr, s, pa, m, ca, cb, n, np_total)
    dropped, planar, fl = _group_planes!(scr, s, m, ca, cb, ng, o, lo, hi, tols)
    flags |= fl
    fallback = _fallback_plane(scr, s, ng)
    rule, fl = _chain_rule(scr, s, pa, m, ca, cb, ng, np_total)
    flags |= fl
    # A table that overflowed is missing patches, and no rule can be trusted on it.
    flags & FLAG_OVERFLOW != 0x00 && (flags |= FLAG_UNSUPPORTED)
    flags & FLAG_UNSUPPORTED != 0x00 && return _use_fallback!(scr, s, ng, fallback, flags)
    k = _compact_groups!(scr, s, ng, dropped, planar)
    # Too many planes for the mixed rule's arrangement: the fallback.
    (rule == RULE_MIXED && k > TRI_K_MIXED) && return _use_fallback!(scr, s, k, fallback, flags | FLAG_UNSUPPORTED)
    return k, (k == 1 ? RULE_SINGLE : rule), flags
end

# Cut by the one fallback plane, reported as the largest of the first `ng` groups' patch.
@inline function _use_fallback!(scr, s, ng::Int, fallback, flags::UInt8)
    largest = _largest_group(scr, s, ng)
    @inbounds scr.gmem[1, s] = scr.gmem[largest, s]
    @inbounds scr.garea[1, s] = scr.garea[largest, s]
    @inbounds scr.gmom[1, s] = scr.gmom[largest, s]
    @inbounds scr.gplane[1, s] = fallback
    return 1, RULE_FALLBACK, flags
end
