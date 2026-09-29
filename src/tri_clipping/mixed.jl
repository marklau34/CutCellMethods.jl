# =====================================
# The mixed rule: convex and concave creases in one box
#
# Where the patches in a box are connected but meet both convexly and concavely -- a chine flat
# between a bottom it meets concavely and a side it meets convexly, a transom corner ringed by
# fillets -- neither intersection of half-spaces is the body. The `k` fitted planes cut the box
# into at most `2^k` convex pieces, one per sign pattern, and the body is a union of some of them.
# Which ones is decided from the mesh, not from a formula:
#
#  * a piece with a face on plane `g` that lies on patch `g` itself -- its centroid projects into
#    one of the patch's triangles -- is on that patch's fluid side if the piece is, and solid
#    otherwise: the outward normal says so;
#  * across a face that lies on plane `g` only where it extends beyond its patch, the two pieces
#    are the same, since no surface separates them;
#  * every piece must end up labelled, and every face must agree with both of its pieces.
#
# Anything contradictory -- a feature the planes cannot represent -- sends the box to the fallback,
# as before. On planar patches this is exact: an arrangement piece's face lies either wholly on a
# patch or wholly off it, since the pieces are cut at the creases.
#
# Leaf statuses: 0 unknown, 1 fluid, 2 solid, 3 empty.

# Whether `x` (in the frame whose origin is `o`) projects along plane `g`'s normal into one of
# group `g`'s triangles in the bins of candidates `ca` and `cb`.
function _on_patch(scr, s, m, ca::Int, cb::Int, g::Int, x::SVector{3,T}, o::SVector{3,T}) where {T}
    p = @inbounds scr.gplane[g, s]
    n = _pn(p)
    # An in-plane frame (u, v) for the projection.
    a0 = abs(n[1]) < T(0.9) ? SVector{3,T}(1, 0, 0) : SVector{3,T}(0, 1, 0)
    u = normalize(cross(n, a0))
    v = cross(n, u)
    xu = dot(u, x)
    xv = dot(v, x)
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
        cc = m.tris[t]
        A = m.verts[cc[1]] - o
        B = m.verts[cc[2]] - o
        C = m.verts[cc[3]] - o
        au, av = dot(u, A) - xu, dot(v, A) - xv
        bu, bv = dot(u, B) - xu, dot(v, B) - xv
        cu, cv = dot(u, C) - xu, dot(v, C) - xv
        w1 = bu * cv - bv * cu
        w2 = cu * av - cv * au
        w3 = au * bv - av * bu
        tot = w1 + w2 + w3
        tot == zero(T) && continue
        # Inside, inclusively, when all three weights share the triangle's orientation.
        tol = -T(1e-9) * abs(tot)
        if tot > zero(T)
            (w1 >= tol && w2 >= tol && w3 >= tol) && return true
        else
            (w1 <= -tol && w2 <= -tol && w3 <= -tol) && return true
        end
    end
    return false
end

@inline _leaf_solid(σ::Int, g::Int) = (σ >> (g - 1)) & 1 == 1

# The arrangement piece `σ` of the box `lo..hi`: the box clipped to each plane's solid side where
# bit `g` of `σ` is set, else its fluid side.
@inline function _leaf_poly!(scr, s, σ::Int, k::Int, lo::SVector{3,T}, hi::SVector{3,T}, tol::T) where {T}
    poly_box!(scr, s, lo, hi)
    st = 0x00
    for g in 1:k
        p = @inbounds scr.gplane[g, s]
        st |= _leaf_solid(σ, g) ? poly_clip!(scr, s, _pn(p), _pd(p), 6 + g, tol, false) :
              poly_clip!(scr, s, -_pn(p), -_pd(p), 6 + g, tol, false)
        poly_isempty(scr, s) && break
    end
    return st
end

"""
    _mixed_classify!(scr, s, m, ca, cb, k, o, lo, hi, tols) -> Bool

Label every piece of the arrangement of the first `k` planes in `gplane` over the box `lo..hi`
(frame origin `o`), into `lstat`/`lface`. `false` when the labels contradict each other.
"""
function _mixed_classify!(scr, s, m, ca::Int, cb::Int, k::Int, o::SVector{3,T}, lo::SVector{3,T},
                          hi::SVector{3,T}, tols) where {T}
    nleaf = 1 << k
    vmin = tols.drop_area * tols.h
    @inbounds for σ in 0:(nleaf - 1)
        for g in 1:k
            scr.lface[g, σ + 1, s] = 0x00
        end
        _leaf_poly!(scr, s, σ, k, lo, hi, tols.snap) != 0x00 && return false
        if poly_isempty(scr, s) || first(poly_volume_moment(scr, s, (lo + hi) / 2)) <= vmin
            scr.lstat[σ + 1, s] = Int8(3)
            continue
        end
        seed = Int8(0)
        for g in 1:k
            _, a, mm = poly_tag_moment(scr, s, 6 + g)
            a > tols.drop_area || continue
            f = 0x01
            if _on_patch(scr, s, m, ca, cb, g, mm / a, o)
                f |= 0x02
                want = _leaf_solid(σ, g) ? Int8(2) : Int8(1)
                seed == Int8(0) || seed == want || return false
                seed = want
            end
            scr.lface[g, σ + 1, s] = f
        end
        scr.lstat[σ + 1, s] = seed
    end
    # Carry labels across faces that are only a plane's extension.
    @inbounds for _ in 1:nleaf
        changed = false
        for σ in 0:(nleaf - 1)
            scr.lstat[σ + 1, s] == Int8(0) || continue
            for g in 1:k
                f = scr.lface[g, σ + 1, s]
                (f & 0x01 != 0x00 && f & 0x02 == 0x00) || continue
                st = scr.lstat[(σ ⊻ (1 << (g - 1))) + 1, s]
                if st == Int8(1) || st == Int8(2)
                    scr.lstat[σ + 1, s] = st
                    changed = true
                    break
                end
            end
        end
        changed || break
    end
    # Every piece labelled, and every face agreeing with both its pieces.
    @inbounds for σ in 0:(nleaf - 1)
        st = scr.lstat[σ + 1, s]
        st == Int8(3) && continue
        st == Int8(0) && return false
        for g in 1:k
            f = scr.lface[g, σ + 1, s]
            f & 0x01 == 0x00 && continue
            other = scr.lstat[(σ ⊻ (1 << (g - 1))) + 1, s]
            (other == Int8(1) || other == Int8(2)) || continue
            (f & 0x02 != 0x00) == (other != st) || return false
        end
    end
    return true
end

"""
    _mixed_cell!(scr, s, m, c, U, k, o, tols) -> (V, M, Aint, Mint, flags, ncomp)

`_clip_cell!` for the mixed rule: the fluid pieces' volume and moment, each group's interface --
the on-patch faces of fluid pieces -- in `gS`/`gAs`/`gMs`, and how many separate fluid regions the
pieces form. `FLAG_UNSUPPORTED` when the arrangement cannot be labelled.
"""
function _mixed_cell!(scr, s, m, c::Int, U::SVector{3,T}, k::Int, o::SVector{3,T}, tols) where {T}
    lo = zero(SVector{3,T})
    r = U / 2
    V = zero(T)
    M = zero(SVector{3,T})
    @inbounds for gi in 1:k
        scr.gS[gi, s] = zero(SVector{3,T})
        scr.gAs[gi, s] = zero(T)
        scr.gMs[gi, s] = zero(SVector{3,T})
    end
    _mixed_classify!(scr, s, m, c, 0, k, o, lo, U, tols) || return V, M, zero(T), M, FLAG_UNSUPPORTED, 0
    nleaf = 1 << k
    fluid = UInt32(0)
    @inbounds for σ in 0:(nleaf - 1)
        scr.lstat[σ + 1, s] == Int8(1) || continue
        fluid |= UInt32(1) << σ
        _leaf_poly!(scr, s, σ, k, lo, U, tols.snap)
        v, mm = poly_volume_moment(scr, s, r)
        V += v
        M += mm
        for g in 1:k
            scr.lface[g, σ + 1, s] & 0x02 == 0x00 && continue
            S, a, mg = poly_tag_moment(scr, s, 6 + g)
            # The fluid piece's face is wound out of it, into the body.
            scr.gS[g, s] -= S
            scr.gAs[g, s] += a
            scr.gMs[g, s] += mg
        end
    end
    # Fluid regions: fluid pieces joined across faces that are only a plane's extension.
    ncomp = 0
    remaining = fluid
    @inbounds while remaining != UInt32(0)
        ncomp += 1
        comp = UInt32(1) << trailing_zeros(remaining)
        frontier = comp
        while frontier != UInt32(0)
            σ = trailing_zeros(frontier)
            frontier &= ~(UInt32(1) << σ)
            for g in 1:k
                f = scr.lface[g, σ + 1, s]
                (f & 0x01 != 0x00 && f & 0x02 == 0x00) || continue
                τ = σ ⊻ (1 << (g - 1))
                bit = UInt32(1) << τ
                if remaining & bit != UInt32(0) && comp & bit == UInt32(0)
                    comp |= bit
                    frontier |= bit
                end
            end
        end
        remaining &= ~comp
    end
    Aint = zero(T)
    Mint = zero(SVector{3,T})
    @inbounds for gi in 1:k
        Aint += scr.gAs[gi, s]
        Mint += scr.gMs[gi, s]
    end
    return V, M, Aint, Mint, 0x00, ncomp
end

"""
    _mixed_face(scr, s, m, cl, ch, ax, lo, U, k, o, tols) -> (area, first_moment, flags)

`_clip_face` for the mixed rule: label the arrangement over the union box `lo..U` of the face's
two cells, then add up the face square's pieces that lie in fluid pieces. Fluid sides are open, so
a face lying in a patch plane is wall.
"""
function _mixed_face(scr, s, m, cl::Int, ch::Int, ax::Int, lo::SVector{3,T}, U::SVector{3,T},
                     k::Int, o::SVector{3,T}, tols) where {T}
    A = zero(T)
    M = zero(SVector{3,T})
    _mixed_classify!(scr, s, m, cl, ch, k, o, lo, U, tols) || return A, M, FLAG_UNSUPPORTED
    zero3 = zero(SVector{3,T})
    @inbounds for σ in 0:((1 << k) - 1)
        scr.lstat[σ + 1, s] == Int8(1) || continue
        b, n, nrm = pg_rect!(scr, s, ax, zero(T), zero3, U)
        for g in 1:k
            p = scr.gplane[g, s]
            b, n = _leaf_solid(σ, g) ? pg_clip!(scr, s, b, n, _pn(p), _pd(p), tols.snap, false) :
                   pg_clip!(scr, s, b, n, -_pn(p), -_pd(p), tols.snap, true)
        end
        n < 0 && return A, M, FLAG_OVERFLOW
        a, mm = pg_area_moment(scr, s, b, n, nrm)
        A += a
        M += mm
    end
    return A, M, 0x00
end
