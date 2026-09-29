# =====================================
# The kernels: resetting cells, classifying them along grid rows, and (P4) cutting them
#
# Every kernel launches one workitem per scratch slot and walks its share of a work list with the
# slot count as stride (`_slot_items`), so the scratch a workitem uses is its own and nothing is
# allocated inside a kernel. Which slot handles an item never affects the item's result, so the
# output does not depend on the backend's scheduling or thread count.

# The in-plane offsets of each of a cell's six face centres from its lower lattice corner -- the
# placeholder centroid a fully open or fully covered face stores. Taken off lattice differences, so
# the two cells sharing a face store bitwise the same pair, as `face_centroid` needs.
@inline function _face_centres_local(g::CartesianGrid{3,T}, ci::CartesianIndex{3}) where {T}
    lo = get_node(g, ci)
    half = T(0.5) * (get_node(g, ci + CartesianIndex(1, 1, 1)) - lo)
    return SVector{6,SVector{2,T}}(ntuple(Val(6)) do dir
        p, q = _inplane_axes((dir + 1) >> 1)
        SVector{2,T}(half[p], half[q])
    end)
end

# The record of a cell the surface does not cut: wholly fluid or wholly solid, every face alike.
@inline function _uncut_cell(g::CartesianGrid{3,T}, ci::CartesianIndex{3}, kind::Int8) where {T}
    vf = kind == CELL_OUTSIDE ? one(T) : zero(T)
    centre = get_node(g, ci) + T(0.5) * g.d
    return CutCellData{3,T,6,2}(kind, false, vf, centre, SVector{6,T}(ntuple(_ -> vf, Val(6))),
                                _face_centres_local(g, ci), centre)
end

@kernel function _tri_reset_kernel!(cells, info, g::CartesianGrid{3,T}, lo::CartesianIndex{3}) where {T}
    I = @index(Global, Cartesian)
    ci = I + lo - oneunit(lo)
    @inbounds cells[ci] = _uncut_cell(g, ci, CELL_OUTSIDE)
    @inbounds info[ci] = TriClipCellInfo{T}(Int8(0), RULE_NONE, 0x00, zero(T))
end

# =====================================
# Classification: parity of ray crossings along each grid row
#
# A cell the surface does not cut is inside the body when the ray along +x from -inf to its centre
# crosses the surface an odd number of times. The crossings of a row are found once, then counted
# against every cell centre in it.
#
# The test of a row's line against a triangle is done in the (y, z) projection, and it has to
# count every crossing exactly once even where the line meets a mesh edge or vertex -- one wrong
# answer flips the whole rest of the row. It needs no exact arithmetic to do so:
#
#  * each edge's function is computed from its endpoints in a fixed order (lower vertex id first)
#    and negated to suit the triangle, so the two triangles on an edge get bitwise-opposite values,
#    and a point on the edge is inside exactly one of them;
#  * an exact zero is broken by a fixed symbolic perturbation of the line, (y + ε, z + ε²): the sign
#    of -Δz along the edge, else of Δy. One global perturbation, so it is consistent at a vertex too;
#  * a point is inside a triangle when all three edge signs agree, which a triangle whose projection
#    is degenerate can never satisfy.
#
# The mesh being watertight, the perturbed line crosses it an even number of times in total, and a
# centre is compared strictly (`x* < x_c`), so a cell whose centre lies on the surface is simply on
# one side; such a cell is cut anyway, and the cut pass overwrites it.

@inline _edge_raw(a::SVector{3,T}, b::SVector{3,T}, y::T, z::T) where {T} =
    (b[2] - a[2]) * (z - a[3]) - (b[3] - a[3]) * (y - a[2])

# Sign of the canonical edge a -> b at a point where its function is exactly zero.
@inline function _edge_tiebreak(a::SVector{3,T}, b::SVector{3,T}) where {T}
    b[3] != a[3] && return b[3] > a[3] ? -1 : 1
    return b[2] > a[2] ? 1 : (b[2] < a[2] ? -1 : 0)
end

# The edge function of p -> q at (y, z) and its sign, from the endpoint with the lower id.
@inline function _edge_fn(ip::Integer, p::SVector{3,T}, iq::Integer, q::SVector{3,T},
                          y::T, z::T) where {T}
    if ip < iq
        e = _edge_raw(p, q, y, z)
        return e, e > zero(T) ? 1 : e < zero(T) ? -1 : _edge_tiebreak(p, q)
    else
        e = -_edge_raw(q, p, y, z)
        return e, e > zero(T) ? 1 : e < zero(T) ? -1 : -_edge_tiebreak(q, p)
    end
end

"""
    _row_crossing(verts, c, y, z) -> (hit, x)

Whether the line `(y, z)` along `x` passes through triangle `c`, and where. `x` is interpolated
from the barycentric weights the edge functions already are, so it lies within the triangle's own
`x` range however edge-on the triangle is.
"""
@inline function _row_crossing(verts, c::SVector{3,Int32}, y::T, z::T) where {T}
    ia, ib, ic = c[1], c[2], c[3]
    A = @inbounds verts[ia]
    B = @inbounds verts[ib]
    C = @inbounds verts[ic]
    e1, s1 = _edge_fn(ia, A, ib, B, y, z)
    e2, s2 = _edge_fn(ib, B, ic, C, y, z)
    e3, s3 = _edge_fn(ic, C, ia, A, y, z)
    (s1 == s2 && s2 == s3 && s1 != 0) || return false, zero(T)
    den = e1 + e2 + e3
    den == zero(T) && return false, zero(T)
    return true, (e2 * A[1] + e3 * B[1] + e1 * C[1]) / den
end

@kernel function _tri_rows_kernel!(cells, info, scr, @Const(verts), @Const(tris),
                                   @Const(row_start), @Const(row_tri), g::CartesianGrid{3,T},
                                   blo::CartesianIndex{3}, bdims::NTuple{3,Int}, ns::Int,
                                   nrows::Int) where {T}
    s = @index(Global, Linear)
    for r in _slot_items(s, ns, nrows)
        _classify_row!(cells, info, scr, s, verts, tris, row_start, row_tri, g, blo, bdims, r)
    end
end

function _classify_row!(cells, info, scr, s, verts, tris, row_start, row_tri,
                        g::CartesianGrid{3,T}, blo::CartesianIndex{3}, bdims::NTuple{3,Int},
                        r::Int) where {T}
    j = blo[2] + (r - 1) % bdims[2]
    k = blo[3] + (r - 1) ÷ bdims[2]
    # The row's line, off the lattice the cell centres are placed on.
    c0 = get_node(g, CartesianIndex(blo[1], j, k)) + T(0.5) * g.d
    y = c0[2]
    z = c0[3]
    nc = 0
    overflow = false
    @inbounds for m in Int(row_start[r]):(Int(row_start[r + 1]) - 1)
        hit, x = _row_crossing(verts, tris[Int(row_tri[m])], y, z)
        if hit
            if nc < MAX_CROSS
                nc += 1
                scr.cross[nc, s] = x
            else
                overflow = true
            end
        end
    end
    flags = overflow ? FLAG_OVERFLOW : 0x00
    @inbounds for di in 0:(bdims[1] - 1)
        ci = CartesianIndex(blo[1] + di, j, k)
        xc = get_node(g, ci)[1] + T(0.5) * g.d[1]
        cnt = 0
        for m in 1:nc
            cnt += scr.cross[m, s] < xc
        end
        cells[ci] = _uncut_cell(g, ci, isodd(cnt) ? CELL_INSIDE : CELL_OUTSIDE)
        info[ci] = TriClipCellInfo{T}(Int8(0), RULE_NONE, flags, zero(T))
    end
    return nothing
end
