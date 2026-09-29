# =====================================
# ConvexPoly: a convex polytope clipped by planes, on a slot's scratch
#
# The brief's face-loop representation: vertices, and faces as vertex-index loops wound
# counter-clockwise seen from outside, each tagged with the Cartesian face it lies in (`1..6`, by
# direction, `1 = -x`, `2 = +x`, ...) or with the fit group whose plane cut it (above `6`). The tags
# are what turn a clipped polytope's faces into apertures and interface facets.
#
# A clip keeps `n . x <= d`:
#
#  1. Signed distances, snapped to zero within `tol`. With nothing on the discarded side the
#     polytope is unchanged; with nothing on the kept side it is empty.
#  2. Kept vertices are copied to the other buffer, which compacts them, and one vertex is inserted
#     on each edge whose ends are *strictly* on opposite sides -- once, shared by the two faces on
#     that edge.
#  3. Every face loop is trimmed to its kept part. A loop left with fewer than three vertices is
#     dropped; never one merely small in area, which would leave an unmatched edge for the next cap.
#  4. The cap: the on-plane edges of the kept faces that no other kept face traverses the other
#     way, reversed and chained into one loop. This replaces the brief's angle sort around the
#     centroid, which is fragile when snapped on-plane vertices nearly coincide or are collinear. A
#     chain that does not close into exactly one loop is reported (`FLAG_CHAIN_FAIL`), not patched.
#
# Everything runs in a cell's local frame -- the box is `[0, U]` from the cell's lower lattice
# corner -- so roundoff is relative to the cell size rather than to the body's distance from the
# origin.

# The box's faces as corner loops, by direction. Corner `k` sits at bits `k - 1` = (x, y, z).
const BOX_FACES = ((1, 5, 7, 3), (2, 4, 8, 6), (1, 2, 6, 5), (3, 7, 8, 4), (1, 3, 4, 2), (5, 6, 8, 7))

@inline _pbuf(scr, s) = Int(@inbounds scr.hdr[1, s])
@inline poly_nv(scr, s) = Int(@inbounds scr.hdr[2, s])
@inline poly_nf(scr, s) = Int(@inbounds scr.hdr[3, s])
@inline poly_isempty(scr, s) = poly_nf(scr, s) == 0

"""
    poly_box!(scr, s, U)

Reset slot `s`'s polytope to the box `[0, U]`, its six faces tagged by direction.
"""
@inline poly_box!(scr, s, U::SVector{3,T}) where {T} = poly_box!(scr, s, zero(SVector{3,T}), U)

# The box `lo..hi`: the same faces and tags.
@inline function poly_box!(scr, s, lo::SVector{3,T}, hi::SVector{3,T}) where {T}
    @inbounds begin
        for k in 1:8
            bits = k - 1
            scr.x[k, 1, s] = SVector{3,T}(ifelse(bits & 1 != 0, hi[1], lo[1]),
                                          ifelse(bits & 2 != 0, hi[2], lo[2]),
                                          ifelse(bits & 4 != 0, hi[3], lo[3]))
        end
        for f in 1:6
            fl = BOX_FACES[f]
            for i in 1:4
                scr.loop[i, f, 1, s] = Int8(fl[i])
            end
            scr.nloop[f, 1, s] = Int8(4)
            scr.tag[f, 1, s] = Int16(f)
        end
        scr.hdr[1, s] = Int16(1)
        scr.hdr[2, s] = Int16(8)
        scr.hdr[3, s] = Int16(6)
    end
    return nothing
end

"""
    poly_clip!(scr, s, nrm, d, tag, tol, retag) -> flags::UInt8

Clip slot `s`'s polytope to `nrm . x <= d`, tagging the new cap face `tag`. `0x00` on success,
else `FLAG_OVERFLOW` or `FLAG_CHAIN_FAIL`, with the polytope then not to be trusted.

With `retag`, a face lying wholly in the plane of a clip that removes nothing is retagged `tag`:
where a patch plane coincides with a cell face, that face is the interface of the *fluid* piece
being clipped. Only a fluid piece may do this -- the same face of a solid polytope is the
neighbour's interface, not this cell's.
"""
function poly_clip!(scr, s, nrm::SVector{3,T}, d::T, tag::Integer, tol::T, retag::Bool) where {T}
    b = _pbuf(scr, s)
    nv = poly_nv(scr, s)
    nf = poly_nf(scr, s)
    nv == 0 && return 0x00
    nneg = 0
    npos = 0
    @inbounds for i in 1:nv
        σ = dot(nrm, scr.x[i, b, s]) - d
        σ = abs(σ) < tol ? zero(T) : σ
        scr.sigma[i, s] = σ
        nneg += σ < zero(T)
        npos += σ > zero(T)
    end
    if npos == 0
        retag && _retag_coplanar!(scr, s, b, nf, tag)
        return 0x00
    end
    if nneg == 0
        @inbounds scr.hdr[2, s] = Int16(0)
        @inbounds scr.hdr[3, s] = Int16(0)
        return 0x00
    end

    o = 3 - b
    m = 0
    @inbounds for i in 1:nv
        σ = scr.sigma[i, s]
        if σ <= zero(T)
            m += 1
            scr.x[m, o, s] = scr.x[i, b, s]
            scr.vmap[i, s] = Int8(m)
            scr.onp[m, s] = Int8(σ == zero(T))
        else
            scr.vmap[i, s] = Int8(0)
        end
    end

    ne = 0
    g = 0
    @inbounds for f in 1:nf
        g + 1 > POLY_MAXF && return FLAG_OVERFLOW
        L = Int(scr.nloop[f, b, s])
        cnt = 0
        for k in 1:L
            ia = Int(scr.loop[k, f, b, s])
            ib = Int(scr.loop[k == L ? 1 : k + 1, f, b, s])
            σa = scr.sigma[ia, s]
            σb = scr.sigma[ib, s]
            if σa <= zero(T)
                cnt >= POLY_MAXL && return FLAG_OVERFLOW
                cnt += 1
                scr.loop[cnt, g + 1, o, s] = scr.vmap[ia, s]
            end
            if (σa < zero(T) && σb > zero(T)) || (σa > zero(T) && σb < zero(T))
                lo = min(ia, ib)
                hi = max(ia, ib)
                idx = 0
                for e in 1:ne
                    if scr.enew[1, e, s] == lo && scr.enew[2, e, s] == hi
                        idx = Int(scr.enew[3, e, s])
                        break
                    end
                end
                if idx == 0
                    (m >= POLY_MAXV || ne >= POLY_MAXE) && return FLAG_OVERFLOW
                    m += 1
                    ne += 1
                    # Interpolated from the lower-indexed end whichever face reaches it first, so
                    # the point does not depend on the order the faces are visited in.
                    σl = scr.sigma[lo, s]
                    σh = scr.sigma[hi, s]
                    xl = scr.x[lo, b, s]
                    scr.x[m, o, s] = xl + (scr.x[hi, b, s] - xl) * (σl / (σl - σh))
                    scr.onp[m, s] = Int8(1)
                    scr.enew[1, ne, s] = Int8(lo)
                    scr.enew[2, ne, s] = Int8(hi)
                    scr.enew[3, ne, s] = Int8(m)
                    idx = m
                end
                cnt >= POLY_MAXL && return FLAG_OVERFLOW
                cnt += 1
                scr.loop[cnt, g + 1, o, s] = Int8(idx)
            end
        end
        if cnt >= 3
            g += 1
            scr.nloop[g, o, s] = Int8(cnt)
            scr.tag[g, o, s] = scr.tag[f, b, s]
        end
    end

    status, g = _poly_cap!(scr, s, o, g, tag)
    @inbounds scr.hdr[1, s] = Int16(o)
    @inbounds scr.hdr[2, s] = Int16(m)
    @inbounds scr.hdr[3, s] = Int16(g)
    return status
end

@inline function _retag_coplanar!(scr, s, b, nf, tag)
    @inbounds for f in 1:nf
        L = Int(scr.nloop[f, b, s])
        coplanar = true
        for k in 1:L
            if scr.sigma[Int(scr.loop[k, f, b, s]), s] != 0
                coplanar = false
                break
            end
        end
        coplanar && (scr.tag[f, b, s] = Int16(tag))
    end
    return nothing
end

# The cap face of a clip, appended to buffer `o` after its `g` trimmed faces. Returns the status and
# the new face count.
function _poly_cap!(scr, s, o::Int, g::Int, tag::Integer)
    nc = 0
    @inbounds for f in 1:g
        L = Int(scr.nloop[f, o, s])
        for k in 1:L
            u = Int(scr.loop[k, f, o, s])
            v = Int(scr.loop[k == L ? 1 : k + 1, f, o, s])
            if scr.onp[u, s] != 0 && scr.onp[v, s] != 0
                nc >= POLY_MAXC && return FLAG_OVERFLOW, g
                nc += 1
                scr.cap[1, nc, s] = Int8(u)
                scr.cap[2, nc, s] = Int8(v)
                scr.cap[3, nc, s] = Int8(0)
            end
        end
    end
    # An on-plane edge two kept faces traverse both ways is inside the kept polytope, not on the cap.
    @inbounds for i in 1:nc, j in (i + 1):nc
        if scr.cap[1, i, s] == scr.cap[2, j, s] && scr.cap[2, i, s] == scr.cap[1, j, s]
            scr.cap[3, i, s] = Int8(1)
            scr.cap[3, j, s] = Int8(1)
        end
    end
    first = 0
    @inbounds for i in 1:nc
        if scr.cap[3, i, s] == 0
            first = i
            break
        end
    end
    first == 0 && return FLAG_CHAIN_FAIL, g
    g >= POLY_MAXF && return FLAG_OVERFLOW, g
    face = g + 1
    cnt = 0
    cur = first
    # The cap traverses each of its edges the opposite way to the kept face beside it, so edge
    # `(u, v)` contributes `v -> u`, and the next edge is the one whose `v` is this one's `u`.
    @inbounds start = Int(scr.cap[2, first, s])
    @inbounds while true
        scr.cap[3, cur, s] = Int8(1)
        cnt >= POLY_MAXL && return FLAG_OVERFLOW, g
        cnt += 1
        scr.loop[cnt, face, o, s] = scr.cap[2, cur, s]
        u = Int(scr.cap[1, cur, s])
        u == start && break
        nxt = 0
        nfound = 0
        for i in 1:nc
            if scr.cap[3, i, s] == 0 && scr.cap[2, i, s] == u
                nfound += 1
                nxt = i
            end
        end
        nfound == 1 || return FLAG_CHAIN_FAIL, g
        cur = nxt
    end
    @inbounds for i in 1:nc
        scr.cap[3, i, s] == 0 && return FLAG_CHAIN_FAIL, g
    end
    cnt < 3 && return 0x00, g   # a degenerate cap has no area to carry
    @inbounds scr.nloop[face, o, s] = Int8(cnt)
    @inbounds scr.tag[face, o, s] = Int16(tag)
    return 0x00, face
end

"""
    poly_volume_moment(scr, s, r) -> (volume, first_moment)

Volume of slot `s`'s polytope and its first moment `volume * centroid`, summed over tetrahedra from
the reference point `r` -- the cell centre, so the terms stay of the cell's size.
"""
function poly_volume_moment(scr, s, r::SVector{3,T}) where {T}
    b = _pbuf(scr, s)
    nf = poly_nf(scr, s)
    V6 = zero(T)
    M = zero(SVector{3,T})
    @inbounds for f in 1:nf
        L = Int(scr.nloop[f, b, s])
        x0 = scr.x[Int(scr.loop[1, f, b, s]), b, s] - r
        for i in 2:(L - 1)
            xi = scr.x[Int(scr.loop[i, f, b, s]), b, s] - r
            xj = scr.x[Int(scr.loop[i + 1, f, b, s]), b, s] - r
            v = dot(x0, cross(xi, xj))
            V6 += v
            M += v * (x0 + xi + xj)
        end
    end
    V = V6 / 6
    return V, V * r + M / 24
end

"""
    poly_face_area_vector(scr, s, f) -> SVector{3}

Newell's vector area of face `f` of slot `s`'s polytope, by a fan from its first vertex: outward,
of the face's area.
"""
@inline function poly_face_area_vector(scr, s, f::Integer)
    b = _pbuf(scr, s)
    L = Int(@inbounds scr.nloop[f, b, s])
    @inbounds x0 = scr.x[Int(scr.loop[1, f, b, s]), b, s]
    A = zero(x0)
    @inbounds for i in 2:(L - 1)
        xi = scr.x[Int(scr.loop[i, f, b, s]), b, s]
        xj = scr.x[Int(scr.loop[i + 1, f, b, s]), b, s]
        A += cross(xi - x0, xj - x0)
    end
    return A / 2
end

"""
    poly_tag_moment(scr, s, tag) -> (vector_area, area, first_moment)

Summed over the faces of slot `s`'s polytope tagged `tag`: their outward vector area, their scalar
area, and its first moment `area * centroid`.
"""
function poly_tag_moment(scr, s, tag::Integer)
    b = _pbuf(scr, s)
    nf = poly_nf(scr, s)
    T = eltype(eltype(scr.x))
    S = zero(SVector{3,T})
    A = zero(T)
    M = zero(SVector{3,T})
    @inbounds for f in 1:nf
        scr.tag[f, b, s] == tag || continue
        Sf = poly_face_area_vector(scr, s, f)
        a2 = sqrt(sum(abs2, Sf))
        a2 > zero(T) || continue
        nhat = Sf / a2
        S += Sf
        L = Int(scr.nloop[f, b, s])
        x0 = scr.x[Int(scr.loop[1, f, b, s]), b, s]
        for i in 2:(L - 1)
            xi = scr.x[Int(scr.loop[i, f, b, s]), b, s]
            xj = scr.x[Int(scr.loop[i + 1, f, b, s]), b, s]
            a = dot(cross(xi - x0, xj - x0), nhat) / 2
            A += a
            M += a * (x0 + xi + xj) / 3
        end
    end
    return S, A, M
end

"""
    poly_area_vector_sum(scr, s) -> SVector{3}

`sum_f A_f` over every face: zero for a closed polytope, to roundoff.
"""
function poly_area_vector_sum(scr, s)
    T = eltype(eltype(scr.x))
    S = zero(SVector{3,T})
    for f in 1:poly_nf(scr, s)
        S += poly_face_area_vector(scr, s, f)
    end
    return S
end

# Whether faces `f` and `g` share an edge: a consistently wound polytope traverses it both ways.
@inline function _faces_share_edge(scr, s, b, f, g)
    Lf = Int(@inbounds scr.nloop[f, b, s])
    Lg = Int(@inbounds scr.nloop[g, b, s])
    @inbounds for k in 1:Lf
        u = scr.loop[k, f, b, s]
        v = scr.loop[k == Lf ? 1 : k + 1, f, b, s]
        for j in 1:Lg
            if scr.loop[j, g, b, s] == v && scr.loop[j == Lg ? 1 : j + 1, g, b, s] == u
                return true
            end
        end
    end
    return false
end

"""
    poly_patch_components(scr, s) -> Int

The number of edge-connected groups the patch-tagged faces (tag above 6) of slot `s`'s polytope
form. On the *solid* polytope of a convex cell, more than one means the solid crosses the cell and
the fluid is split -- a feature thinner than the cell.
"""
function poly_patch_components(scr, s)
    b = _pbuf(scr, s)
    nf = poly_nf(scr, s)
    remaining = UInt32(0)
    @inbounds for f in 1:nf
        scr.tag[f, b, s] > 6 && (remaining |= UInt32(1) << (f - 1))
    end
    ncomp = 0
    while remaining != 0
        ncomp += 1
        seed = trailing_zeros(remaining) + 1
        comp = UInt32(1) << (seed - 1)
        frontier = comp
        while frontier != 0
            f = trailing_zeros(frontier) + 1
            frontier &= ~(UInt32(1) << (f - 1))
            candidates = remaining & ~comp
            while candidates != 0
                g = trailing_zeros(candidates) + 1
                candidates &= ~(UInt32(1) << (g - 1))
                if _faces_share_edge(scr, s, b, f, g)
                    comp |= UInt32(1) << (g - 1)
                    frontier |= UInt32(1) << (g - 1)
                end
            end
        end
        remaining &= ~comp
    end
    return ncomp
end

# =====================================
# Planes and the Boolean rules on a box
#
# A plane is `SVector{4,T}(n..., d)`, `n` the unit normal out of the body: the solid half-space is
# `n . x <= d` and the fluid half-space `n . x >= d`.

@inline _pn(p::SVector{4,T}) where {T} = @inbounds SVector{3,T}(p[1], p[2], p[3])
@inline _pd(p::SVector{4}) = @inbounds p[4]

"""
    fluid_convex_moments(scr, s, U, planes, k, tol) -> (volume, first_moment, flags)

The fluid part of the box `[0, U]` when the solid is the convex intersection of the first `k`
planes' solid half-spaces -- a convex crease -- as the disjoint union over `j` of
`box ∩ S_1 ∩ ... ∩ S_(j-1) ∩ F_j`. Each piece is convex, so the volume and first moment accumulate
without the cancellation `V_box - V_solid` suffers when the fluid is a sliver.
"""
function fluid_convex_moments(scr, s, U::SVector{3,T}, planes, k::Integer, tol::T) where {T}
    r = U / 2
    V = zero(T)
    M = zero(SVector{3,T})
    status = 0x00
    for j in 1:k
        poly_box!(scr, s, U)
        for i in 1:(j - 1)
            p = @inbounds planes[i]
            status |= poly_clip!(scr, s, _pn(p), _pd(p), 7 + i - 1, tol, false)
        end
        p = @inbounds planes[j]
        status |= poly_clip!(scr, s, -_pn(p), -_pd(p), 7 + j - 1, tol, true)
        v, m = poly_volume_moment(scr, s, r)
        V += v
        M += m
    end
    return V, M, status
end
