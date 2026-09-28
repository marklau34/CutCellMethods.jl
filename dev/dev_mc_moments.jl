# =====================================
# Validating the 3D cut-cell moments against the whole-grid march.
#
# `marching_cubes/moments.jl` reconstructs one cell at a time: it reads the Lewiner tables per cube
# and closes the cell's polyhedron out of six face clips plus the facets those tables produce. The
# obvious worry about that design is that it is a *second* implementation of a reconstruction the
# package already has -- `generate_mesh(geo, grid, MarchingCubesCutCell())`, which runs `MarchingCubes.march` over the
# whole grid -- and that the two could drift.
#
# This script is the answer to that worry. It takes the marched surface, sorts its triangles back
# into cells, and checks three things:
#
#   1. the per-cell dispatch produces the *same facets* the whole-grid march does, cell by cell
#      (tolerance, not bitwise -- see the note on vertex placement below),
#   2. those marched facets close the same apertures the moments report, so the polyhedron the
#      volume is integrated over is the one the mesh actually draws,
#   3. the moments' own diagnostics hold over the whole domain: closure exactly zero, volume
#      fractions in range, and the body volume converging at second order.
#
# `test/test_mc_moments.jl` pins (1) exhaustively over all 256 sign patterns on a single cube, which
# is the stronger statement about the *tables*. What this adds is the whole-grid setting: real
# fields, real cell indexing, shared vertices, and the `sample_sdf` node coordinates rather than
# `cell_nodes`'.
#
# ---------------------------------------------------------------------------
# Why this comparison is to a tolerance and not bitwise
#
# Two independent ulp-scale differences, both documented where they arise and neither a defect:
#
#  * `MarchingCubes.jl` anchors an edge crossing on the edge's low node; this package anchors on the
#    *outside* node, which is choice 1 of `marching_squares.jl`'s watertightness argument and what
#    lets the face clips and the facets share a crossing exactly. See
#    `marching_cubes/marching_cubes.jl`.
#  * the march builds node coordinates from `CartesianMeshes.grid_lines` while `cell_nodes` builds
#    them from `get_node`. Algebraically equal, not bitwise. `cell_values`' docstring says so.
#
# Both are `O(eps)` in a cell coordinate, so a mismatch that matters shows up orders of magnitude
# above the threshold used here rather than near it.
#
# Run with:  julia --project=. dev/dev_mc_moments.jl

using SDFLibrary
using CutCellMethods
using CutCellMethods: cell_nodes, cell_values, cell_interface, nudge_zeros,
                      is_cut, is_ambiguous, interface_normal_area, interface_area, interface_normal,
                      closure_residual, full_face_area, cell_indices
using CartesianMeshes
using StaticArrays
using LinearAlgebra
using Printf

grid3(N; lo = -1.5, span = 3.0) =
    CartesianGrid(SVector(lo, lo, lo), (N, N, N), SVector(span/N, span/N, span/N))

wiggle(x) = sin(2.7x[1]) * sin(3.1x[2]) * sin(2.3x[3]) - 0.12

"""Triangles of the whole-grid march, bucketed by the cell each one came from."""
function marched_by_cell(geo, grid)
    mesh = generate_mesh(geo, grid, MarchingCubesCutCell(); warn = false)
    owner = cell_indices(mesh, grid)
    x = mesh.nodes.coord
    buckets = Dict{Int,Vector{NTuple{3,SVector{3,Float64}}}}()
    for (t, tri) in enumerate(mesh.elements)
        c = tri.con
        push!(get!(buckets, owner[t], NTuple{3,SVector{3,Float64}}[]),
              (SVector{3,Float64}(x[c[1]]),
               SVector{3,Float64}(x[c[2]]),
               SVector{3,Float64}(x[c[3]])))
    end
    return buckets
end

"""A set-comparable, winding-sensitive key for a triangle."""
function tri_key(t; digits = 7)
    r(v) = round.(v; digits = digits)
    a, b, c = r(t[1]), r(t[2]), r(t[3])
    return sort([(a, b, c), (b, c, a), (c, a, b)], by = string)[1]
end

"""`sum_t A_t n_t` over a bucket of triangles, with `n_t` out of the body."""
function bucket_vector_area(tris)
    w = zero(SVector{3,Float64})
    for (a, b, c) in tris
        w += 0.5 * cross(b - a, c - a)
    end
    return w
end

# =====================================================================
"""(1) and (2): the per-cell dispatch against the whole-grid march, in the whole-grid setting."""
function compare_to_march(geo, N, tag)
    grid = grid3(N)
    h = SVector(grid.d...)
    buckets = marched_by_cell(geo, grid)
    lin = LinearIndices(grid.n)

    ncut = 0; nmatch = 0; ncount_bad = 0; ngeom_bad = 0
    worst_close = 0.0      # marched facets vs the moments' face-derived interface area
    worst_res = 0.0

    for idx in CartesianIndices(grid.n)
        nodes = cell_nodes(grid, idx)
        phi = cell_values(geo, grid, idx)
        m = cut_cell_moments(nodes, phi, h)
        worst_res = max(worst_res, maximum(abs.(closure_residual(m, h))))
        is_cut(m) || continue
        ncut += 1

        ref = get(buckets, lin[idx], NTuple{3,SVector{3,Float64}}[])
        n, a, b, c = cell_interface(nodes, nudge_zeros(phi))
        mine = [(a[t], b[t], c[t]) for t in 1:n]

        if length(ref) != length(mine)
            ncount_bad += 1
        elseif sort(tri_key.(ref), by = string) == sort(tri_key.(mine), by = string)
            nmatch += 1
        else
            ngeom_bad += 1
        end

        # (2) The marched facets must close the apertures the moments report -- this is the
        # statement that the mesh on screen bounds the volume in the flux balance.
        if !isempty(ref)
            worst_close = max(worst_close,
                              maximum(abs.(bucket_vector_area(ref) - interface_normal_area(m, h))))
        end
    end

    @printf("%-9s N=%3d  cut=%6d  facets match=%6d  count-mismatch=%3d  geom-mismatch=%3d\n",
            tag, N, ncut, nmatch, ncount_bad, ngeom_bad)
    @printf("%-9s        max |sum_t A_t n_t  -  A_int n_int| (marched facets vs apertures) = %.3e\n",
            "", worst_close)
    @printf("%-9s        max |closure residual| over every cell                            = %.3e\n",
            "", worst_res)
    return nothing
end

# =====================================================================
"""(3) Whole-domain diagnostics: convergence, range, and the ambiguity census."""
function convergence(geo, exact_volume, tag; Ns = (20, 40, 80))
    println()
    @printf("%-9s  %6s  %14s  %12s  %10s  %8s  %6s\n",
            tag, "N", "body volume", "rel err", "rate", "cut", "amb")
    prev = NaN
    for N in Ns
        grid = grid3(N)
        M = cut_cell_moments(geo, grid)
        vc = prod(SVector(grid.d...))
        outside = sum(m.volume_fraction for m in M) * vc
        body = 27.0 - outside
        rel = (body - exact_volume) / exact_volume
        rate = isnan(prev) ? NaN : log2(abs(prev / rel))
        nbad = count(m -> !(-1e-12 <= m.volume_fraction <= 1 + 1e-12), M)
        nbad == 0 || @warn "$nbad cells have a volume fraction outside [0,1]"
        @printf("%-9s  %6d  %14.8f  %+12.3e  %10s  %8d  %6d\n",
                "", N, body, rel, isnan(rate) ? "--" : @sprintf("%.2f", rate),
                count(is_cut, M), count(is_ambiguous, M))
        prev = rel
    end
    return nothing
end

# =====================================================================
"""The interface area the two routes report. They are *not* the same quantity, and this is the
clearest place to see it: `interface_area` is the norm of a vector sum and so measures a flat
facet, while the marched triangles measure the actual fan. In 2D a facet is a straight segment and
the two coincide; in 3D the fan is generally non-planar and the vector sum is strictly shorter.
Conservation wants the first, a surface force wants the second."""
function area_gap(geo, N, tag)
    grid = grid3(N)
    h = SVector(grid.d...)
    buckets = marched_by_cell(geo, grid)
    lin = LinearIndices(grid.n)
    lumped = 0.0; faceted = 0.0; worst_ratio = 1.0
    for idx in CartesianIndices(grid.n)
        m = cut_cell_moments(cell_nodes(grid, idx), cell_values(geo, grid, idx), h)
        is_cut(m) || continue
        tris = get(buckets, lin[idx], NTuple{3,SVector{3,Float64}}[])
        isempty(tris) && continue
        af = sum(0.5 * norm(cross(b - a, c - a)) for (a, b, c) in tris)
        al = interface_area(m, h)
        lumped += al; faceted += af
        af > 1e-14 && (worst_ratio = max(worst_ratio, af / max(al, 1e-300)))
    end
    @printf("%-9s N=%3d  sum interface_area = %.6f   sum facet area = %.6f   (%.3f%% low)  worst cell ratio = %.2fx\n",
            tag, N, lumped, faceted, 100 * (1 - lumped / faceted), worst_ratio)
    return nothing
end

# =====================================================================
function main()
    sph = SDFSphere(SVector(0.0, 0.0, 0.0), 1.0)
    slot = SDFSlottedSphere(SVector(0.0, 0.0, 0.0), 1.0, 0.3, 0.8)

    println("=== (1)+(2) per-cell dispatch vs the whole-grid march ===")
    compare_to_march(sph, 24, "sphere")
    compare_to_march(wiggle, 24, "wiggly")
    compare_to_march(slot, 32, "slotted")

    println()
    println("=== (3) whole-domain diagnostics ===")
    convergence(sph, 4pi/3, "sphere")

    println()
    println("=== interface_area is a flat-facet measure, by construction ===")
    area_gap(sph, 24, "sphere")
    area_gap(sph, 48, "sphere")
    area_gap(wiggle, 24, "wiggly")
    return nothing
end

main()
