# =====================================
# Planar polygons: Sutherland–Hodgman on a slot's `pg` buffers
#
# Two jobs use this. Pass A clips each triangle to a cell box, closed on every side, for the area
# and centroid of the piece a patch has in that cell. The face pass clips a Cartesian face square by
# a cell's planes for the face's open fraction and centroid. A polygon is a buffer index and a
# count: a clip reads buffer `b`, writes `3 - b`, and returns the new pair. A count of `-1` is an
# overflow, which no convex input within the documented limits can reach.
#
# Every clip keeps the side `n.x <= d`. A signed distance within `tol` of zero is snapped to zero,
# so a vertex on the plane is exactly on it for the sign tests that follow. Clipping is closed --
# a polygon lying in the plane is kept -- unless `open` is set, when it is dropped: the fluid side
# of a patch plane is open, so a face lying in the plane is wall, not flow.

# The two in-plane axes of the Cartesian face normal to `ax`, in increasing order -- the order
# `CutCellData.face_centroid_local` stores its offsets in.
@inline _inplane_axes(ax::Integer) = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)

@inline function pg_triangle!(scr, s, a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T}) where {T}
    @inbounds begin
        scr.pg[1, 1, s] = a
        scr.pg[2, 1, s] = b
        scr.pg[3, 1, s] = c
    end
    return 1, 3
end

"""
    pg_rect!(scr, s, ax, xa, lo, hi) -> (buffer, count, normal)

The Cartesian face normal to axis `ax` at `x[ax] = xa`, spanning `lo..hi` along the other two
axes, as a 4-vertex polygon. Wound counter-clockwise about `e_p × e_q` for its in-plane axes
`p < q`, which is the `normal` returned: `+x`, `-y` or `+z`.
"""
@inline function pg_rect!(scr, s, ax::Integer, xa::T, lo::SVector{3,T}, hi::SVector{3,T}) where {T}
    p, q = _inplane_axes(ax)
    @inline corner(u, v) = SVector{3,T}(ntuple(a -> a == ax ? xa : a == p ? u : v, Val(3)))
    @inbounds begin
        scr.pg[1, 1, s] = corner(lo[p], lo[q])
        scr.pg[2, 1, s] = corner(hi[p], lo[q])
        scr.pg[3, 1, s] = corner(hi[p], hi[q])
        scr.pg[4, 1, s] = corner(lo[p], hi[q])
    end
    nrm = SVector{3,T}(ntuple(a -> a == ax ? (ax == 2 ? -one(T) : one(T)) : zero(T), Val(3)))
    return 1, 4, nrm
end

# Classify every vertex against `n.x <= d`, snapping near-zero distances, and count each side.
@inline function _pg_classify!(scr, s, b, n, nrm::SVector{3,T}, d::T, tol::T) where {T}
    nneg = 0
    npos = 0
    @inbounds for i in 1:n
        σ = dot(nrm, scr.pg[i, b, s]) - d
        σ = abs(σ) < tol ? zero(T) : σ
        scr.pgsig[i, s] = σ
        nneg += σ < zero(T)
        npos += σ > zero(T)
    end
    return nneg, npos
end

"""
    pg_clip!(scr, s, b, n, nrm, d, tol, open) -> (buffer, count)

Clip polygon `(b, n)` of slot `s` to the half-space `nrm . x <= d`. See this file's header.
"""
@inline function pg_clip!(scr, s, b::Int, n::Int, nrm::SVector{3,T}, d::T, tol::T,
                          open::Bool) where {T}
    n <= 0 && return b, n
    nneg, npos = _pg_classify!(scr, s, b, n, nrm, d, tol)
    npos == 0 && return (open && nneg == 0) ? (b, 0) : (b, n)
    nneg == 0 && return b, 0
    o = 3 - b
    m = 0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        σi = scr.pgsig[i, s]
        σj = scr.pgsig[j, s]
        xi = scr.pg[i, b, s]
        if σi <= zero(T)
            m >= PG_MAXP && return o, -1
            m += 1
            scr.pg[m, o, s] = xi
        end
        if (σi < zero(T) && σj > zero(T)) || (σi > zero(T) && σj < zero(T))
            m >= PG_MAXP && return o, -1
            m += 1
            scr.pg[m, o, s] = xi + (scr.pg[j, b, s] - xi) * (σi / (σi - σj))
        end
    end
    return o, m
end

"""
    pg_clip_axis!(scr, s, b, n, ax, bound, upper, tol) -> (buffer, count)

Clip to `x[ax] <= bound` (`upper`) or `x[ax] >= bound`, closed. A new vertex gets `bound` exactly
in its `ax` coordinate, so a piece clipped to a cell face lies in that face bit for bit.
"""
@inline function pg_clip_axis!(scr, s, b::Int, n::Int, ax::Int, bound::T, upper::Bool,
                               tol::T) where {T}
    n <= 0 && return b, n
    nneg = 0
    npos = 0
    @inbounds for i in 1:n
        xa = scr.pg[i, b, s][ax]
        σ = upper ? xa - bound : bound - xa
        σ = abs(σ) < tol ? zero(T) : σ
        scr.pgsig[i, s] = σ
        nneg += σ < zero(T)
        npos += σ > zero(T)
    end
    npos == 0 && return b, n
    nneg == 0 && return b, 0
    o = 3 - b
    m = 0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        σi = scr.pgsig[i, s]
        σj = scr.pgsig[j, s]
        xi = scr.pg[i, b, s]
        if σi <= zero(T)
            m >= PG_MAXP && return o, -1
            m += 1
            scr.pg[m, o, s] = xi
        end
        if (σi < zero(T) && σj > zero(T)) || (σi > zero(T) && σj < zero(T))
            m >= PG_MAXP && return o, -1
            m += 1
            p = xi + (scr.pg[j, b, s] - xi) * (σi / (σi - σj))
            scr.pg[m, o, s] = Base.setindex(p, bound, ax)
        end
    end
    return o, m
end

"""
    pg_tri_box!(scr, s, a, b, c, lo, hi, tol) -> (buffer, count)

Triangle `abc` clipped to the closed box `lo..hi`. At most 9 vertices.
"""
@inline function pg_tri_box!(scr, s, a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T},
                             lo::SVector{3,T}, hi::SVector{3,T}, tol::T) where {T}
    buf, n = pg_triangle!(scr, s, a, b, c)
    for ax in 1:3
        buf, n = pg_clip_axis!(scr, s, buf, n, ax, @inbounds(lo[ax]), false, tol)
        buf, n = pg_clip_axis!(scr, s, buf, n, ax, @inbounds(hi[ax]), true, tol)
    end
    return buf, n
end

"""
    pg_area_moment(scr, s, b, n, nhat) -> (area, first_moment)

Signed area of polygon `(b, n)` measured along the unit normal `nhat`, and its first moment
`area * centroid`, by a fan from the first vertex. Clipping preserves winding, so a clipped piece of
a triangle measured along that triangle's normal is non-negative.
"""
@inline function pg_area_moment(scr, s, b::Int, n::Int, nhat::SVector{3,T}) where {T}
    A = zero(T)
    M = zero(SVector{3,T})
    n < 3 && return A, M
    @inbounds x1 = scr.pg[1, b, s]
    @inbounds for i in 2:n-1
        xi = scr.pg[i, b, s]
        xj = scr.pg[i + 1, b, s]
        a = dot(cross(xi - x1, xj - x1), nhat) / 2
        A += a
        M += a * (x1 + xi + xj) / 3
    end
    return A, M
end
