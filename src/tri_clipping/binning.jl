# =====================================
# Binning: which triangles each cell, and each grid row, has to look at
#
# On the host, in `Float64`, into grow-only buffers. Everything the kernels do per cell is bounded by
# its bin, so the bins must be *conservative* -- every triangle that reaches a cell grown by the
# refit margin `δ` is in it -- and *deterministic*: filled in triangle order, so a bin's order, and
# with it every sum a kernel forms over it, is the same whatever the thread count. The fill is
# serial for that reason; it is linear in the entries and small beside the passes it feeds.
#
# Bins cover only the *block*: the cells the body can touch, plus one cell of padding so a cut
# cell's six neighbours are always classified. Everything outside the block is untouched fluid.
#
# A triangle goes into a cell's bin when its bounding box grown by `δ` overlaps the cell, and its
# plane passes within the cell's half-diagonal grown by `δ` of the cell centre -- the cheap half of a
# separating-axis test, which is what keeps a large tilted triangle out of the cells its box covers
# but it does not. Rows are binned by the triangle's `(y, z)` box alone, which is exact for a ray.

"""
    TriClipBlock

The block of cells the body can touch, inclusive: `lo` to `hi`. Empty when `empty`.
"""
struct TriClipBlock
    lo::SVector{3,Int}
    hi::SVector{3,Int}
    empty::Bool
end
TriClipBlock() = TriClipBlock(SVector(1, 1, 1), SVector(0, 0, 0), true)

@inline _block_dims(b::TriClipBlock) = b.empty ? SVector(0, 0, 0) : b.hi - b.lo .+ 1
@inline _block_ncells(b::TriClipBlock) = prod(_block_dims(b))
@inline _block_nrows(b::TriClipBlock) = (d = _block_dims(b); d[2] * d[3])

"""
    _bin_block(X, tris, grid, margin) -> (block, tri_lo, tri_hi)

Each triangle's cell range -- its bounding box grown by `margin` -- and the block covering them all
plus one cell. Throws if the block does not fit inside the grid: the body, grown by the margin and
a cell, must lie inside the domain, so that every domain-boundary face is open.
"""
function _bin_block!(tri_lo::Vector{SVector{3,Int32}}, tri_hi::Vector{SVector{3,Int32}},
                     X::Vector{SVector{3,Float64}}, tris::Vector{SVector{3,Int32}},
                     grid::CartesianGrid{3}, margin::SVector{3})
    nt = length(tris)
    resize!(tri_lo, nt)
    resize!(tri_hi, nt)
    nt == 0 && return TriClipBlock()
    x0 = SVector{3,Float64}(grid.x0)
    d = SVector{3,Float64}(grid.d)
    δ = SVector{3,Float64}(margin)
    lo = SVector(typemax(Int), typemax(Int), typemax(Int))
    hi = SVector(typemin(Int), typemin(Int), typemin(Int))
    for t in 1:nt
        c = tris[t]
        bmin = min.(X[c[1]], X[c[2]], X[c[3]])
        bmax = max.(X[c[1]], X[c[2]], X[c[3]])
        ilo = floor.(Int, (bmin - δ - x0) ./ d) .+ 1
        ihi = floor.(Int, (bmax + δ - x0) ./ d) .+ 1
        tri_lo[t] = SVector{3,Int32}(ilo)
        tri_hi[t] = SVector{3,Int32}(ihi)
        lo = min.(lo, ilo)
        hi = max.(hi, ihi)
    end
    blo = lo .- 1
    bhi = hi .+ 1
    n = SVector{3,Int}(grid.n)
    if any(blo .< 1) || any(bhi .> n)
        bmin, bmax = x0 + (lo .- 1) .* d, x0 + hi .* d
        throw(ArgumentError(
            "the body reaches the edge of the grid: grown by the refit margin it spans cells " *
            "$(Tuple(lo)) to $(Tuple(hi)) (about $(Tuple(bmin)) to $(Tuple(bmax))), but a cut needs " *
            "at least one clear cell between the body and every side of a grid of $(Tuple(n)) cells"))
    end
    return TriClipBlock(blo, bhi, false)
end

# Whether triangle `t`'s plane passes within reach of the cell centred at `c`: `|n . (c - p)|`
# against the support of the cell's half-extent `r` along `n`, with a little slack so the test can
# only err towards keeping a triangle.
@inline function _plane_reaches(n::SVector{3,Float64}, p::SVector{3,Float64}, c::SVector{3,Float64},
                                r::SVector{3,Float64})
    reach = abs(n[1]) * r[1] + abs(n[2]) * r[2] + abs(n[3]) * r[3]
    return abs(dot(n, c - p)) <= reach * (1 + 1e-12) + 1e-12 * maximum(r)
end

"""
    _bin_cells!(start, list, X, tris, tri_lo, tri_hi, block, grid, margin)

CSR bins of triangles over the block's cells, linear in the block with `x` fastest: cell `k`'s
triangles are `list[start[k]:start[k+1]-1]`, in increasing triangle order.
"""
function _bin_cells!(start::Vector{Int32}, list::Vector{Int32}, X, tris, tri_lo, tri_hi,
                     block::TriClipBlock, grid::CartesianGrid{3}, margin::SVector{3})
    ncell = _block_ncells(block)
    resize!(start, ncell + 1)
    fill!(start, 0)
    block.empty && (resize!(list, 0); return nothing)
    x0 = SVector{3,Float64}(grid.x0)
    d = SVector{3,Float64}(grid.d)
    r = d / 2 + SVector{3,Float64}(margin)
    dims = _block_dims(block)
    lin(i, j, k) = 1 + (i - block.lo[1]) + dims[1] * ((j - block.lo[2]) + dims[2] * (k - block.lo[3]))
    # Two identical passes: count, then fill. The test is one function so they cannot disagree.
    for pass in 1:2
        for t in eachindex(tris)
            cc = tris[t]
            p1, p2, p3 = X[cc[1]], X[cc[2]], X[cc[3]]
            cr = cross(p2 - p1, p3 - p1)
            ncr = norm(cr)
            nrm = ncr > 0 ? cr / ncr : zero(SVector{3,Float64})
            lo, hi = tri_lo[t], tri_hi[t]
            for k in lo[3]:hi[3], j in lo[2]:hi[2], i in lo[1]:hi[1]
                c = x0 + (SVector(i, j, k) .- 0.5) .* d
                _plane_reaches(nrm, p1, c, r) || continue
                l = lin(i, j, k)
                if pass == 1
                    start[l + 1] += 1
                else
                    list[start[l]] = Int32(t)
                    start[l] += 1
                end
            end
        end
        if pass == 1
            # Counts to starts: start[l] is where cell l's run begins.
            start[1] = 1
            for l in 1:ncell
                start[l + 1] += start[l]
            end
            resize!(list, start[ncell + 1] - 1)
        else
            # The fill advanced each start to the next cell's; shift back.
            for l in ncell:-1:1
                start[l + 1] = start[l]
            end
            start[1] = 1
        end
    end
    return nothing
end

"""
    _bin_rows!(start, list, X, tris, block, grid)

CSR bins of triangles over the block's grid rows along `x`: row `(j, k)`, linear with `j` fastest,
lists every triangle whose `(y, z)` bounding box contains the row's line, in increasing triangle
order. Grown by a row either side, so a line on a box's edge is never missed; the kernel's own test
decides.
"""
function _bin_rows!(start::Vector{Int32}, list::Vector{Int32}, X, tris,
                    block::TriClipBlock, grid::CartesianGrid{3})
    nrows = _block_nrows(block)
    resize!(start, nrows + 1)
    fill!(start, 0)
    block.empty && (resize!(list, 0); return nothing)
    x0 = SVector{3,Float64}(grid.x0)
    d = SVector{3,Float64}(grid.d)
    dims = _block_dims(block)
    for pass in 1:2
        for t in eachindex(tris)
            cc = tris[t]
            bmin = min.(X[cc[1]], X[cc[2]], X[cc[3]])
            bmax = max.(X[cc[1]], X[cc[2]], X[cc[3]])
            # Row j's line is at y = x0 + (j - 1/2) d.
            jlo = max(block.lo[2], ceil(Int, (bmin[2] - x0[2]) / d[2] + 0.5) - 1)
            jhi = min(block.hi[2], floor(Int, (bmax[2] - x0[2]) / d[2] + 0.5) + 1)
            klo = max(block.lo[3], ceil(Int, (bmin[3] - x0[3]) / d[3] + 0.5) - 1)
            khi = min(block.hi[3], floor(Int, (bmax[3] - x0[3]) / d[3] + 0.5) + 1)
            for k in klo:khi, j in jlo:jhi
                l = 1 + (j - block.lo[2]) + dims[2] * (k - block.lo[3])
                if pass == 1
                    start[l + 1] += 1
                else
                    list[start[l]] = Int32(t)
                    start[l] += 1
                end
            end
        end
        if pass == 1
            start[1] = 1
            for l in 1:nrows
                start[l + 1] += start[l]
            end
            resize!(list, start[nrows + 1] - 1)
        else
            for l in nrows:-1:1
                start[l + 1] = start[l]
            end
            start[1] = 1
        end
    end
    return nothing
end
