# =====================================
# Tri clipping, phase 6: the reconstructed surface, its creases, and the VTK output.
#
# What is asserted:
#
#   * the reconstructed surface is the interface the cut used: wound out of the body, one element
#     set per patch, each patch's area its exact area on a planar body, and the whole of it the
#     interface the boundary faces report
#   * the corner line -- where two patches' facets meet in a cell -- lies on the mesh's feature
#     edges to roundoff on planar bodies, and converges at O(h^2) where a planar transom meets a
#     curved bottom
#   * `write_cache_vtk` writes the grid and the surface
#   * the GPPH hull's surface

using CutCellMethods: interface_mesh, interface_creases, write_cache_vtk, boundary_faces, CELL_CUT

isdefined(@__MODULE__, :tm_box) || include("tri_meshes.jl")

const TS = TriClippingCutCell()

ts_tri_area(X, c) = norm(cross(X[c[2]] - X[c[1]], X[c[3]] - X[c[1]])) / 2

# The distance from `p` to segment `a b`.
function ts_point_segment(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / max(dot(ab, ab), eps()), 0.0, 1.0)
    return norm(p - (a + t * ab))
end

# The mesh's feature edges: every edge between two patches.
function ts_feature_edges(mesh)
    topo = CutCellMethods.build_topology(mesh, TS, 1.0)
    X = CutCellMethods._mesh_coords(mesh)
    tris = CutCellMethods._mesh_tris(mesh)
    segs = Tuple{SVector{3,Float64},SVector{3,Float64}}[]
    for t in eachindex(tris), s in 1:3
        topo.side_label[t][s] == 0x00 && continue
        push!(segs, (X[tris[t][s]], X[tris[t][s % 3 + 1]]))
    end
    return segs
end

# The largest distance of the reconstructed creases -- their ends and midpoints -- from the mesh's
# feature edges.
function ts_corner_line_error(creases, features)
    worst = 0.0
    for (a, b) in creases, p in (a, b, (a + b) / 2)
        worst = max(worst, minimum(f -> ts_point_segment(p, f[1], f[2]), features))
    end
    return worst
end

@testset verbose = true "tri clipping: the surface" begin
    g = CartesianGrid(SVector(-1.0, -1.0, -1.0), (16, 16, 16), SVector(0.125, 0.125, 0.125))

    @testset "the surface is the interface that was cut" begin
        lo, hi = SVector(-0.33, -0.41, -0.27), SVector(0.47, 0.29, 0.52)
        box, _, areas = tm_box(lo, hi; n=3)
        cache = update_cache!(allocate_cache(g, TS), box, g)
        surf, cells = interface_mesh(cache, g)
        X = [p.coord for p in surf.nodes]
        tris = [e.con for e in surf.elements]
        @test length(cells) == length(tris) > 0
        # Every triangle wound out of the box, and inside the cell it came from.
        centre = (lo + hi) / 2
        @test all(c -> dot(cross(X[c[2]] - X[c[1]], X[c[3]] - X[c[1]]), (X[c[1]] + X[c[2]] + X[c[3]]) / 3 - centre) > 0, tris)
        @test all(k -> (x = (X[tris[k][1]] + X[tris[k][2]] + X[tris[k][3]]) / 3;
                        l = get_node(g, cells[k]); all(l .- 1e-12 .<= x .<= l .+ g.d .+ 1e-12)), eachindex(tris))
        # One set per patch, each its face's exact area.
        @test sort([s.name for s in surf.elemset]) == sort(collect(keys(areas)))
        for s in surf.elemset
            @test sum(k -> ts_tri_area(X, tris[k]), s.elems) ≈ areas[s.name] rtol = 1e-13
        end
        # The same interface the boundary faces report.
        bf_area = sum(ci -> sum(f -> f.area, boundary_faces(cache, ci); init=0.0), CartesianIndices(Tuple(g.n)))
        @test sum(c -> ts_tri_area(X, c), tris) ≈ bf_area rtol = 1e-13
        # `generate_mesh` cuts and reads it off in one call.
        @test length(generate_mesh(box, g, TS).elements) == length(tris)
    end

    @testset "concave and mixed creases draw too" begin
        for body in (tm_lblock(a=0.4, hz=0.9, t=SVector(-0.41, -0.39, -0.43))[1],
                     tm_chine_prism(b1=0.33, b2=0.36, nx=5, t=SVector(-0.7, 0.011, -0.31))[1])
            cache = update_cache!(allocate_cache(g, TS), body, g)
            surf, _ = interface_mesh(cache, g)
            X = [p.coord for p in surf.nodes]
            area = sum(c -> ts_tri_area(X, c.con), surf.elements)
            topo = cache.work.topo
            @test area ≈ sum(topo.patch_area) rtol = 1e-12
        end
    end

    @testset "the corner line" begin
        # Planar corners: every reconstructed crease lies on a feature edge of the mesh.
        for body in (tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5, t=SVector(-0.7, 0.013, -0.3))[1],
                     tm_chine_prism(b1=0.33, b2=0.36, nx=5, t=SVector(-0.7, 0.011, -0.31))[1])
            cache = update_cache!(allocate_cache(g, TS), body, g)
            creases = interface_creases(cache, g)
            @test length(creases) > 50
            @test ts_corner_line_error(creases, ts_feature_edges(body)) < 1e-12
        end
        # A planar transom meeting a curved bottom: the crease is a curve, reconstructed in each
        # cell as the transom's plane meeting the bottom's fit -- within O(h^2) of it.
        hull, _ = tm_curved_hull(t=SVector(-0.7, 0.013, -0.3), ny=64)
        features = ts_feature_edges(hull)
        errs = map((16, 32)) do n
            gg = CartesianGrid(SVector(-1.0, -1.0, -1.0), (n, n, n), SVector(2 / n, 2 / n, 2 / n))
            cache = update_cache!(allocate_cache(gg, TS), hull, gg)
            ts_corner_line_error(interface_creases(cache, gg), features)
        end
        @test errs[1] < 0.05 * 0.125
        @test errs[1] / errs[2] > 3
    end

    @testset "VTK" begin
        hull, _, _ = tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5, t=SVector(-0.7, 0.013, -0.3))
        cache = update_cache!(allocate_cache(g, TS), hull, g)
        mktempdir() do dir
            files = write_cache_vtk(joinpath(dir, "prism"), cache, g)
            @test length(files) == 2
            @test all(f -> isfile(f) && filesize(f) > 0, files)
        end
    end

    @testset "the GPPH hull's surface" begin
        hull = tm_gpph()
        if hull === nothing
            @info "gpph_clean.inp not generated (examples/geometry/run_generate_mesh.jl): skipped"
        else
            gh = CartesianGrid(SVector(-0.4, -1.5, -0.3), (172, 60, 40), SVector(0.05, 0.05, 0.05))
            cache = @test_logs match_mode = :any update_cache!(allocate_cache(gh, TS), hull, gh)
            surf, _ = interface_mesh(cache, gh)
            X = [p.coord for p in surf.nodes]
            @test length(surf.elemset) == 8
            # The reconstructed surface has the hull's area, to the fits' order.
            @test sum(c -> ts_tri_area(X, c.con), surf.elements) ≈ sum(cache.work.topo.patch_area) rtol = 1e-3
        end
    end
end
