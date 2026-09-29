# =====================================
# The cut: each cut cell's faces, polytope, and closure
#
# One workitem per cut cell, in three steps:
#
#  1. Its six faces. A face whose neighbour is uncut takes that neighbour's status: any surface
#     crossing the face would have given the neighbour clipped area too, so an uncut neighbour means
#     the face is wholly fluid or wholly solid, and which is the neighbour's kind. A face between two
#     cut cells is cut by planes fitted over the *union* of the two cells, from their merged patch
#     tables -- the two cells taken low then high whichever of them is asking, so both compute the
#     same numbers and store bitwise the same fraction and centroid in their shared slots.
#  2. Its own polytope, from planes fitted over the cell alone: the fluid volume and centroid, and
#     each fit group's share of the clipped interface.
#  3. The closure. `interface_normal_area` forms the interface's vector area from the stored face
#     fractions, so the per-cell identity holds exactly whatever they are; what is left to do is to
#     split that vector over the groups. Each group keeps its clipped facet `S_g` and takes a share
#     of the mismatch `A_int - sum S_g` in proportion to `|S_g|` (the last group takes the
#     remainder, so the split sums to `A_int` to within one rounding). The mismatch is `O(h^3)` on a
#     curved patch -- the union-box and own-box fits differ at that order -- and zero on a planar one.

@kernel function _tri_cell_kernel!(cells, info, bf, pa, m, scr, g::CartesianGrid{3,T},
                                   blo::CartesianIndex{3}, bdims::NTuple{3,Int}, tols,
                                   np_total::Int, ns::Int, ncut::Int) where {T}
    s = @index(Global, Linear)
    for k in _slot_items(s, ns, ncut)
        _cut_cell!(cells, info, bf, pa, m, scr, s, Int(@inbounds m.cut_list[k]), g, blo, bdims,
                   tols, np_total)
    end
end

"""
    _clip_cell!(scr, s, U, k, rule, tol) -> (V, M, Aint, Mint, flags, ncomp)

Clip the box `[0, U]` by the first `k` planes in slot `s`'s `gplane` under `rule`: the fluid volume
and first moment, the interface's scalar area and first moment, the clip status, and -- for a
convex cell -- how many separate pieces the solid's interface forms. Each group's own share of the
interface goes to `gS` (vector area, out of the body), `gAs` and `gMs`.
"""
function _clip_cell!(scr, s, U::SVector{3,T}, k::Int, rule::UInt8, tol::T) where {T}
    r = U / 2
    V = zero(T)
    M = zero(SVector{3,T})
    st = 0x00
    ncomp = 0
    @inbounds for gi in 1:k
        scr.gS[gi, s] = zero(SVector{3,T})
        scr.gAs[gi, s] = zero(T)
        scr.gMs[gi, s] = zero(SVector{3,T})
    end
    if rule == RULE_CONVEX
        # The fluid, as disjoint convex pieces box ∩ S_1 ∩ ... ∩ S_(j-1) ∩ F_j.
        for j in 1:k
            poly_box!(scr, s, U)
            for i in 1:(j - 1)
                p = @inbounds scr.gplane[i, s]
                st |= poly_clip!(scr, s, _pn(p), _pd(p), 6 + i, tol, false)
            end
            p = @inbounds scr.gplane[j, s]
            st |= poly_clip!(scr, s, -_pn(p), -_pd(p), 6 + j, tol, true)
            v, mm = poly_volume_moment(scr, s, r)
            V += v
            M += mm
        end
        # The solid, whose patch faces are exactly the interface, wound out of the body.
        poly_box!(scr, s, U)
        for i in 1:k
            p = @inbounds scr.gplane[i, s]
            st |= poly_clip!(scr, s, _pn(p), _pd(p), 6 + i, tol, false)
        end
        @inbounds for gi in 1:k
            S, a, mm = poly_tag_moment(scr, s, 6 + gi)
            scr.gS[gi, s] = S
            scr.gAs[gi, s] = a
            scr.gMs[gi, s] = mm
        end
        ncomp = poly_patch_components(scr, s)
    else
        # One plane, or a concave crease: the fluid is convex, clipped directly. Its patch faces
        # are wound out of the fluid, into the body.
        poly_box!(scr, s, U)
        for i in 1:k
            p = @inbounds scr.gplane[i, s]
            st |= poly_clip!(scr, s, -_pn(p), -_pd(p), 6 + i, tol, true)
        end
        V, M = poly_volume_moment(scr, s, r)
        @inbounds for gi in 1:k
            S, a, mm = poly_tag_moment(scr, s, 6 + gi)
            scr.gS[gi, s] = -S
            scr.gAs[gi, s] = a
            scr.gMs[gi, s] = mm
        end
    end
    Aint = zero(T)
    Mint = zero(SVector{3,T})
    @inbounds for gi in 1:k
        Aint += scr.gAs[gi, s]
        Mint += scr.gMs[gi, s]
    end
    return V, M, Aint, Mint, st, ncomp
end

"""
    _clip_face(scr, s, ax, U, k, rule, tol) -> (area, first_moment, flags)

The open part of the Cartesian face normal to `ax` at `x[ax] = 0`, spanning `[0, U]` in-plane,
under the first `k` planes in `gplane` and `rule`. The fluid half-spaces are open, so a face lying
in a patch plane is wall; under the convex rule the fluid is the same disjoint union of pieces as
the cell's.
"""
function _clip_face(scr, s, ax::Int, U::SVector{3,T}, k::Int, rule::UInt8, tol::T) where {T}
    lo = zero(SVector{3,T})
    A = zero(T)
    M = zero(SVector{3,T})
    st = 0x00
    if rule == RULE_CONVEX
        for j in 1:k
            b, n, nrm = pg_rect!(scr, s, ax, zero(T), lo, U)
            for i in 1:(j - 1)
                p = @inbounds scr.gplane[i, s]
                b, n = pg_clip!(scr, s, b, n, _pn(p), _pd(p), tol, false)
            end
            p = @inbounds scr.gplane[j, s]
            b, n = pg_clip!(scr, s, b, n, -_pn(p), -_pd(p), tol, true)
            if n < 0
                st |= FLAG_OVERFLOW
            else
                a, mm = pg_area_moment(scr, s, b, n, nrm)
                A += a
                M += mm
            end
        end
    else
        b, n, nrm = pg_rect!(scr, s, ax, zero(T), lo, U)
        for i in 1:k
            p = @inbounds scr.gplane[i, s]
            b, n = pg_clip!(scr, s, b, n, -_pn(p), -_pd(p), tol, true)
        end
        if n < 0
            st |= FLAG_OVERFLOW
        else
            A, M = pg_area_moment(scr, s, b, n, nrm)
        end
    end
    return A, M, st
end

"""
    _cut_face(scr, s, pa, m, cells, c, ci, dir, g, blo, bdims, tols, np_total) -> (fraction, centroid, flags)

Face `dir` of cut cell `ci` (candidate `c`): its open fraction and the in-plane offsets of its open
part's centroid from the cell's lower corner. See this file's header.
"""
function _cut_face(scr, s, pa, m, cells, c::Int, ci::CartesianIndex{3}, dir::Int,
                   g::CartesianGrid{3,T}, blo::CartesianIndex{3}, bdims::NTuple{3,Int}, tols,
                   np_total::Int) where {T}
    ax = (dir + 1) >> 1
    up = iseven(dir)
    e = CartesianIndex(ntuple(a -> a == ax ? 1 : 0, Val(3)))
    cj = up ? ci + e : ci - e
    p, q = _inplane_axes(ax)
    cn = Int(@inbounds m.cut_slot[_block_lin(blo, bdims, cj)])
    if cn == 0
        frac = @inbounds(cells.kind[cj]) == CELL_OUTSIDE ? one(T) : zero(T)
        half = T(0.5) * (get_node(g, ci + CartesianIndex(1, 1, 1)) - get_node(g, ci))
        return frac, SVector{2,T}(half[p], half[q]), 0x00
    end
    cl, cil, ch, cih = up ? (c, ci, cn, cj) : (cn, cj, c, ci)
    ol = get_node(g, cil)
    oh = get_node(g, cih)
    Uh = get_node(g, cih + CartesianIndex(1, 1, 1)) - oh
    n, ovf = _merge_entries!(scr, s, pa, cl, ch, ol - oh)
    flags = (ovf ? FLAG_OVERFLOW : 0x00) | ((@inbounds(pa.flag[cl]) | @inbounds(pa.flag[ch])) & FLAG_OVERFLOW)
    lo = SVector{3,T}(ntuple(a -> a == ax ? ol[a] - oh[a] : zero(T), Val(3)))
    k, rule, flags = _fit!(scr, s, pa, m, cl, ch, n, oh, lo, Uh, tols, np_total, flags)
    if rule == RULE_MIXED
        A, Mf, st = _mixed_face(scr, s, m, cl, ch, ax, lo, Uh, k, oh, tols)
        if st != 0x00
            # An arrangement that could not be labelled: the face's fallback plane.
            n, _ = _merge_entries!(scr, s, pa, cl, ch, ol - oh)
            k, rule, flags = _fit!(scr, s, pa, m, cl, ch, n, oh, lo, Uh, tols, np_total,
                                   flags | FLAG_UNSUPPORTED)
            A, Mf, st = _clip_face(scr, s, ax, Uh, k, rule, tols.snap)
        end
    else
        A, Mf, st = _clip_face(scr, s, ax, Uh, k, rule, tols.snap)
    end
    flags |= st
    halfh = T(0.5) * Uh
    placeholder = SVector{2,T}(halfh[p], halfh[q])
    frac = clamp(A / (Uh[p] * Uh[q]), zero(T), one(T))
    centroid = A > zero(T) ? SVector{2,T}(Mf[p] / A, Mf[q] / A) : placeholder
    snap = tols.aperture_snap
    if snap > zero(T) && (frac <= snap || frac >= one(T) - snap)
        frac = frac <= snap ? zero(T) : one(T)
        centroid = placeholder
    end
    return frac, centroid, flags
end

# The six faces of cut cell `ci`, and any overflow among them.
function _cut_faces(scr, s, pa, m, cells, c::Int, ci::CartesianIndex{3}, g::CartesianGrid{3,T},
                    blo::CartesianIndex{3}, bdims::NTuple{3,Int}, tols, np_total::Int) where {T}
    ff = zero(SVector{6,T})
    fcl = zero(SVector{6,SVector{2,T}})
    flags = 0x00
    for dir in 1:6
        fr, cl, fl = _cut_face(scr, s, pa, m, cells, c, ci, dir, g, blo, bdims, tols, np_total)
        ff = Base.setindex(ff, fr, dir)
        fcl = Base.setindex(fcl, cl, dir)
        flags |= fl & FLAG_OVERFLOW
    end
    return ff, fcl, flags
end

# Split the interface vector `Aint` over the `k` groups' clipped facets, writing each group's
# boundary face; returns the correction magnitude and flags.
function _close_cell!(bf, scr, s, c::Int, k::Int, Aint::SVector{3,T}, o::SVector{3,T},
                      centre::SVector{3,T}, tols) where {T}
    base = (c - 1) * TRI_K_MAX
    Ssum = zero(SVector{3,T})
    Snorm = zero(T)
    Amesh = zero(T)
    @inbounds for gi in 1:k
        Ssum += scr.gS[gi, s]
        Snorm += sqrt(sum(abs2, scr.gS[gi, s]))
        Amesh += scr.garea[gi, s]
    end
    D = Aint - Ssum
    nD = sqrt(sum(abs2, D))
    flags = 0x00
    # With no clipped facet to share it by -- a patch plane lying in a face of the solid-side cell,
    # say -- the mismatch goes by the patches' clipped mesh area instead. Flagged only when there is
    # a mismatch to share.
    by_mesh = Snorm <= tols.drop_area
    (by_mesh && nD > tols.drop_area) && (flags |= FLAG_CLOSURE_FALLBACK)
    # A correction is large when it flips or outweighs a facet that matters: one of a tenth of a
    # cell face or more. On a grazing sliver any correction dwarfs the facet, and says nothing.
    big_facet = tols.h^2 / 10
    acc = zero(SVector{3,T})
    @inbounds for gi in 1:k
        S = scr.gS[gi, s]
        nS = sqrt(sum(abs2, S))
        w = by_mesh ? (Amesh > zero(T) ? scr.garea[gi, s] / Amesh : one(T) / k) : nS / Snorm
        Ag = gi == k ? Aint - acc : S + w * D
        acc += Ag
        corr = sqrt(sum(abs2, Ag - S))
        (nS >= big_facet && (dot(Ag, S) < zero(T) || corr > nS / 2)) && (flags |= FLAG_CORR_LARGE)
        area = sqrt(sum(abs2, Ag))
        nrm = area > zero(T) ? Ag / area : zero(SVector{3,T})
        a = scr.gAs[gi, s]
        cen = a > zero(T) ? o + scr.gMs[gi, s] / a :
              (scr.garea[gi, s] > zero(T) ? o + scr.gmom[gi, s] / scr.garea[gi, s] : centre)
        p, _ = _group_patch(scr, s, gi)
        bf[base + gi] = TriBoundaryFace{T}(p, area, nrm, cen)
    end
    @inbounds for gi in (k + 1):TRI_K_MAX
        bf[base + gi] = TriBoundaryFace{T}(Int32(0), zero(T), zero(SVector{3,T}), centre)
    end
    return nD, flags
end

function _cut_cell!(cells, info, bf, pa, m, scr, s, c::Int, g::CartesianGrid{3,T},
                    blo::CartesianIndex{3}, bdims::NTuple{3,Int}, tols, np_total::Int) where {T}
    ci = _block_ci(blo, bdims, Int(@inbounds m.cand[c]))
    o = get_node(g, ci)
    U = get_node(g, ci + CartesianIndex(1, 1, 1)) - o
    centre = o + T(0.5) * g.d

    # 1. The faces, each fitted over the union of its two cells.
    ff, fcl, flags = _cut_faces(scr, s, pa, m, cells, c, ci, g, blo, bdims, tols, np_total)

    # 2. The cell's own polytope, fitted over the cell alone.
    n = _load_entries!(scr, s, pa, c)
    k, rule, flags = _fit!(scr, s, pa, m, c, 0, n, o, zero(SVector{3,T}), U, tols, np_total,
                           flags | @inbounds(pa.flag[c]))
    V, M, Aint, Mint, st, ncomp = rule == RULE_MIXED ? _mixed_cell!(scr, s, m, c, U, k, o, tols) :
                                  _clip_cell!(scr, s, U, k, rule, tols.snap)
    if st != 0x00 && rule != RULE_FALLBACK
        # A clip that could not close its polytope: fall back to the one plane.
        n = _load_entries!(scr, s, pa, c)
        k, rule, flags = _fit!(scr, s, pa, m, c, 0, n, o, zero(SVector{3,T}), U, tols, np_total,
                               flags | st | FLAG_UNSUPPORTED)
        V, M, Aint, Mint, st, ncomp = _clip_cell!(scr, s, U, 1, RULE_SINGLE, tols.snap)
    end
    flags |= st
    ncomp >= 2 && (flags |= FLAG_SPLIT)
    k >= 2 && (flags |= FLAG_MULTI_PATCH)

    vfrac = clamp(V / (U[1] * U[2] * U[3]), zero(T), one(T))
    centroid = V > zero(T) ? o + M / V : centre
    icentroid = Aint > zero(T) ? o + Mint / Aint : centre
    # Ambiguous only where a fallback plane stood in for the surface. A split cell -- a crease
    # passing just outside it, its fluid in two pockets joined beyond the corner -- is exact, and
    # says so in `info.flags` (`FLAG_SPLIT`) rather than here.
    ambiguous = flags & FLAG_UNSUPPORTED != 0x00
    cell = CutCellData{3,T,6,2}(CELL_CUT, ambiguous, vfrac, centroid, ff, fcl, icentroid)

    # 3. The closure, split over the groups.
    corr, fl = _close_cell!(bf, scr, s, c, k, interface_normal_area(cell, g.d), o, centre, tols)
    flags |= fl
    @inbounds cells[ci] = cell
    @inbounds info[ci] = TriClipCellInfo{T}(Int8(k), rule, flags, corr)
    return nothing
end

# =====================================
# Uncut cells that disagree with an uncut neighbour: a face between them can be neither wholly
# fluid nor wholly solid, which only a self-intersecting or open surface produces.

@kernel function _tri_conflict_kernel!(info, cells, @Const(cut_slot), blo::CartesianIndex{3},
                                       bdims::NTuple{3,Int}, n::NTuple{3,Int})
    I = @index(Global, Cartesian)
    ci = I + blo - oneunit(blo)
    if @inbounds(cut_slot[_block_lin(blo, bdims, ci)]) == 0
        kind = @inbounds cells.kind[ci]
        conflict = false
        for dir in 1:6
            ax = (dir + 1) >> 1
            e = CartesianIndex(ntuple(a -> a == ax ? 1 : 0, Val(3)))
            cj = iseven(dir) ? ci + e : ci - e
            (1 <= cj[ax] <= n[ax]) || continue
            inblock = blo[ax] <= cj[ax] <= blo[ax] + bdims[ax] - 1
            cut = inblock && @inbounds(cut_slot[_block_lin(blo, bdims, cj)]) != 0
            (!cut && @inbounds(cells.kind[cj]) != kind) && (conflict = true)
        end
        if conflict
            old = @inbounds info[ci]
            @inbounds info[ci] = typeof(old)(old.npatch, old.rule, old.flags | FLAG_STATUS_CONFLICT,
                                             old.correction)
        end
    end
end
