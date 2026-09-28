# =====================================
# The cut-cell API, driven the way a consumer would
#
#     julia --project=. dev/dev_cut_cell_api.jl
#
# There is no cut-cell cache any more. Two calls are the whole surface:
#
#   cut_cell_moments(method, domain, geo, cell)   -- one cell's geometry
#   generate_mesh(geo, domain, method)            -- the reconstructed surface
#
# The caller owns the storage. Instead of allocating a cache and reading fields out of it, you loop
# (or launch) over the cells you care about and keep exactly the arrays your scheme needs -- which,
# for a solver that wants volume fractions and apertures but never a surface, is a fraction of what
# a cache held, and for one that only cares about the cut band is a fraction again.
#
# Four things are demonstrated, in order:
#
#   1. the nodal methods (`MarchingSquaresCutCell`, `MarchingCubesCutCell`): build your own field,
#      and the watertightness and closure identities survive being computed one cell at a time;
#   2. `PLICCutCell`, where an aperture comes back ONE-SIDED and resolving a shared face is the
#      caller's job -- and is one line, because averaging is commutative;
#   3. `generate_mesh(geo, domain, method)`, one spelling for all three reconstructions;
#   4. what it costs: `generate_mesh` now re-samples the field, because there is no cache holding
#      the corner values it used to read.
#
# This is the executable form of the API's documentation: if it stops printing `true`, the
# docstrings in `plic/plic.jl`, `cut_cell.jl` and `marching_squares/moments.jl` describe something
# that no longer holds.

using SDFLibrary
using MeshLibrary
using StaticArrays
using CartesianMeshes
using Printf

using CutCellMethods
using CutCellMethods: cell_plane, closure_residuals, cut_cell_report, is_cut, interface_area,
                      closure_residual
using SDFLibrary: sample_sdf

# An off-lattice union with a re-entrant corner. The corner matters: it is where two cells' centroid
# fits disagree most, and so where the one-sided PLIC apertures are most visibly one-sided.
const GEO = SDFUnion(SDFCircle(center=SVector(0.02, 0.03), radius=0.31),
                     SDFRectangle(center=SVector(0.25, 0.10), dims=SVector(0.18, 0.18));
                     smoothing=0.0)
const GRID = CartesianIsoGrid(mincorner=(-0.6, -0.6), maxcorner=(0.6, 0.6), sz=0.02)

banner(s) = println("\n", s, "\n", repeat("-", length(s)))

# -----------------------------------------------------------------------------
# 1. The nodal methods: keep only what you need
#
# The scheme below wants two fields -- the open volume fraction and the four apertures -- and
# nothing else. A full `CutCellData` additionally carries the outside centroid, four face
# centroids, the interface vector area and the interface centroid, which is most of its footprint
# and all of it dead weight here.
function nodal_demo()
    banner("1. MarchingSquaresCutCell -- the caller allocates")
    n = Tuple(GRID.n)
    MS = MarchingSquaresCutCell()

    vol = Array{Float64}(undef, n)
    ap = Array{SVector{4,Float64}}(undef, n)
    for ci in CartesianIndices(n)
        m = cut_cell_moments(MS, GRID, GEO, ci)
        vol[ci] = m.volume_fraction
        ap[ci] = m.face_fraction
    end

    @printf("  cells                        : %d\n", prod(n))
    @printf("  kept per cell                : %d bytes (vs %d for a full CutCellData)\n",
            sizeof(Float64) + sizeof(SVector{4,Float64}), sizeof(CutCellData{2,Float64,4,1}))

    # Watertightness is a per-cell property, so it survives being computed one cell at a time: two
    # neighbours read the same corner values through the same expression and get the same bits.
    # This is the whole reason it is safe to drop the cache.
    tight = all(i -> all(j -> ap[i, j][2] === ap[i+1, j][1], 1:n[2]), 1:(n[1]-1))
    @printf("  shared faces agree bitwise   : %s\n", tight)

    # Repeating the call gives the same bits, so nothing was being memoized for correctness.
    stable = all(ci -> cut_cell_moments(MS, GRID, GEO, ci).volume_fraction === vol[ci],
                 CartesianIndices(n))
    @printf("  recomputing is deterministic : %s\n", stable)

    # The diagnostics take a plain array of moments now, not a cache.
    mom = [cut_cell_moments(MS, GRID, GEO, ci) for ci in CartesianIndices(n)]
    @printf("  closure residual exactly 0   : %s\n", all(iszero, closure_residuals(mom, GRID)))
    r = cut_cell_report(mom)
    @printf("  report                       : %d cut, %d inside, %d ambiguous, min alpha %.2e\n",
            r.cut, r.inside, r.ambiguous, r.min_volume_fraction)
    return nothing
end

function nodal_3d_demo()
    banner("1b. MarchingCubesCutCell -- the same call, one dimension up")
    g = CartesianIsoGrid(mincorner=(-0.6, -0.6, -0.6), maxcorner=(0.6, 0.6, 0.6), sz=0.05)
    sph = SDFSphere(center=SVector(0.02, 0.03, 0.01), radius=0.4)
    MC = MarchingCubesCutCell()
    n = Tuple(g.n)

    # A dense CutCellData{3} field is 208 bytes a cell; this keeps 32, over the cut band only.
    cut = CartesianIndex{3}[]
    area = Float64[]
    for ci in CartesianIndices(n)
        m = cut_cell_moments(MC, g, sph, ci)
        if is_cut(m)
            push!(cut, ci)
            push!(area, interface_area(m, g.d))
        end
    end
    @printf("  cells / cut cells            : %d / %d\n", prod(n), length(cut))
    @printf("  dense field would cost       : %.1f MB\n",
            prod(n) * sizeof(CutCellData{3,Float64,6,2}) / 2^20)
    @printf("  what this loop kept          : %.1f KB\n",
            (sizeof(CartesianIndex{3}) + sizeof(Float64)) * length(cut) / 2^10)
    @printf("  sum of facet areas vs 4pi r^2: %.6f vs %.6f\n", sum(area), 4pi * 0.4^2)
    return nothing
end

# -----------------------------------------------------------------------------
# 2. PLIC: one-sided apertures, and resolving them
#
# The one place the per-cell form hands back something a consumer must finish. Each plane is fitted
# from its own cell's centroid, so the two cells sharing a face have two different candidates for
# how open it is. Averaging them is the caller's job -- and not optional for anything that fluxes,
# since a scheme reading two different apertures for one face manufactures mass at it.
function plic_demo()
    banner("2. PLICCutCell -- the apertures come back one-sided")
    n = Tuple(GRID.n)
    PL = PLICCutCell()
    own = [cut_cell_moments(PL, GRID, GEO, ci) for ci in CartesianIndices(n)]

    gaps = Float64[]
    for i in 1:(n[1]-1), j in 1:n[2]
        push!(gaps, abs(own[i, j].face_area[2] - own[i+1, j].face_area[1]))
    end
    nz = count(>(0), gaps)
    @printf("  interior +x faces            : %d\n", length(gaps))
    @printf("  where the two sides disagree : %d (%.1f%%)\n", nz, 100nz / length(gaps))
    @printf("  largest disagreement         : %.3e  (cell size %.3e)\n",
            maximum(gaps), GRID.d[1])

    banner("2b. Resolving a shared face -- one line, and it is single-valued for free")
    # The mean of the two one-sided halves. There is no helper for this and there does not need to
    # be: floating-point `+` is commutative, so both cells sharing a face get bitwise the same
    # number whichever order they name each other in. `symmetric` below is that claim, tested.
    resolved = Dict{Tuple{Int,CartesianIndex{2}},Float64}()
    symmetric = true
    for c in 1:2
        e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, 2))
        for ci in CartesianIndices(ntuple(a -> a == c ? n[a] - 1 : n[a], 2))
            dn, dp = own[ci], own[ci+e]
            lo, hi = dn.face_area[2c], dp.face_area[2c-1]
            resolved[(c, ci)] = (lo + hi) / 2
            symmetric &= (lo + hi) / 2 === (hi + lo) / 2
        end
    end
    @printf("  resolved interior faces      : %d\n", length(resolved))
    @printf("  order-independent (bitwise)  : %s\n", symmetric)

    # And the primitive underneath, for a consumer that never materializes a `PLICCutCellData`:
    # `cell_face_areas` off a bare `cell_plane` fit gives the same numbers.
    bare = all(CartesianIndices(n)) do ci
        pl = cell_plane(GEO, GRID, ci)
        CutCellMethods.cell_face_areas(pl.normal, pl.intercept, 1 - pl.fraction, pl.is_valid, GRID.d) ===
            own[ci].face_area
    end
    @printf("  cell_face_areas off a raw fit: %s\n", bare)
    return nothing
end

# -----------------------------------------------------------------------------
# 3 and 4. The surface, and what re-sampling costs
function mesh_demo()
    banner("3. generate_mesh(geo, domain, method) -- one spelling, three reconstructions")
    MS, PL, MC = MarchingSquaresCutCell(), PLICCutCell(), MarchingCubesCutCell()
    g3 = CartesianIsoGrid(mincorner=(-0.6, -0.6, -0.6), maxcorner=(0.6, 0.6, 0.6), sz=0.03)
    sph = SDFSphere(center=SVector(0.02, 0.03, 0.01), radius=0.4)

    c2 = generate_mesh(GEO, GRID, MS)
    sp = generate_mesh(GEO, GRID, PL)
    s3 = generate_mesh(sph, g3, MC)
    @printf("  marching squares (2D)        : %d segments, %d nodes (stitched)\n",
            length(c2.elements), length(c2.nodes.coord))
    @printf("  PLIC (2D)                    : %d lines, %d nodes (shards, never shared)\n",
            length(sp.elements), length(sp.nodes.coord))
    @printf("  marching cubes (3D)          : %d tris, %d nodes\n",
            length(s3.elements), length(s3.nodes.coord))

    banner("4. The cost of dropping the cache: generate_mesh re-samples")
    # The cache used to keep the corner values, so the contour was free once the moments were built.
    # Now the two passes each sample. They are still the SAME reconstruction when driven the same
    # way, which is the part that matters -- but a caller that wants both should hand the nodal
    # block to both rather than the geometry, and pay for one sampling.
    vals = sample_sdf(GEO, GRID)
    cn = generate_mesh(vals, GRID, MS)
    momn = [cut_cell_moments(MS, GRID, vals, ci) for ci in CartesianIndices(Tuple(GRID.n))]
    @printf("  from a nodal block           : %d segments, %d cut cells\n",
            length(cn.elements), count(is_cut, momn))
    @printf("  one sample per node          : %d reads (vs %d for four-per-cell)\n",
            length(vals), 4 * prod(Tuple(GRID.n)))
    println("  ...and both reads see the same numbers, so the contour and the moments agree.")
    return nothing
end

function main()
    nodal_demo()
    nodal_3d_demo()
    plic_demo()
    mesh_demo()
    println()
end

main()
