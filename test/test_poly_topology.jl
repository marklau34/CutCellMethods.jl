# =====================================
# Polyline clipping, P1: loops and mesh validation
#
# A cache finds its mesh's loops and validates it on the first update, and keeps the loops as
# `cache.work.topo` and their areas as `cache.work.loop_area`; that is what is read here.

using Test
using StaticArrays
using MeshLibrary
using Random
using CutCellMethods: PolylineClippingCutCell, PolylineTopology, nloops

isdefined(@__MODULE__, :PV) || include("polyline_meshes.jl")

"""A grid of square cells round `mesh`, `n` across its longer side and a few cells clear of it."""
function pt_grid(mesh; n=32)
    X = mesh.nodes.coord
    lo = reduce((a, b) -> min.(a, b), X)
    hi = reduce((a, b) -> max.(a, b), X)
    h = maximum(hi - lo) / n
    return CartesianGrid(SVector{2,Float64}(lo .- 3h), Tuple(ceil.(Int, (hi - lo) ./ h) .+ 6), SVector(h, h))
end

"""A cache over `grid`, updated from `mesh`: the update that validates it."""
pt_cache(mesh; grid=pt_grid(mesh), method=PolylineClippingCutCell()) =
    poly_name_sets!(update_cache!(allocate_cache(grid, method), SDFMesh(mesh), grid), mesh)

"""The loops an update finds in `mesh`, and their areas."""
poly_check(mesh) = (c = pt_cache(mesh); (c.work.topo, c.work.loop_area))

@testset verbose = true "polyline clipping: topology" begin
    rng = MersenneTwister(7)

    @testset "method" begin
        m = PolylineClippingCutCell()
        @test m.drop_area == 1e-14 && m.closure_tol == 1e-13 && m.validate === :topology
        @test PolylineClippingCutCell(validate=:always).validate === :always
        @test_throws ArgumentError PolylineClippingCutCell(validate=:sometimes)
        @test_throws ArgumentError PolylineClippingCutCell(drop_area=-1)
        @test_throws ArgumentError PolylineClippingCutCell(closure_tol=0)
    end

    @testset "loops and areas" begin
        topo, area = poly_check(poly_mesh(square_pts((0.3, -0.2), 0.7)))
        @test topo isa PolylineTopology
        @test nloops(topo) == 1 && topo.loop_nelem == [4]
        @test area[1] ≈ 0.49 rtol = 1e-15
        topo, area = poly_check(poly_mesh(square_pts((0.3, -0.2), 0.7; θ=0.3)))
        @test area[1] ≈ 0.49 rtol = 1e-14
        topo, area = poly_check(poly_mesh(ngon_pts((0.1, 0.2), 0.8, 64; θ0=0.01)))
        @test area[1] ≈ ngon_area(0.8, 64) rtol = 1e-14
        foil = naca4_pts(; m=0.02, p=0.4, t=0.12)
        topo, area = poly_check(poly_mesh(foil))
        @test area[1] ≈ pts_area(foil) rtol = 1e-14
        @test area[1] ≈ 0.0822 rtol = 0.01          # a 12% section's area is about 0.685·t·c

        # Three bodies, named by their element sets.
        loops = three_element_pts()
        mesh = poly_mesh(loops; names=["slat", "main", "flap"])
        topo, area = poly_check(mesh)
        @test nloops(topo) == 3
        @test topo.loop_name == ["slat", "main", "flap"]
        @test area ≈ pts_area.(loops) rtol = 1e-14
        lines = [e.con for e in mesh.elements]
        @test all(e -> lines[topo.elem_next[e]][1] == lines[e][2], eachindex(lines))
        @test all(e -> topo.elem_next[topo.elem_prev[e]] == e, eachindex(lines))
        @test sum(topo.loop_nelem) == length(lines)

        # Element order does not matter: shuffled, the same loops come back.
        perm = randperm(rng, length(lines))
        topo2, area2 = poly_check(poly_mesh(loops; names=["slat", "main", "flap"], perm=perm))
        @test sort(area2) ≈ sort(area) rtol = 1e-14
        @test sort(topo2.loop_nelem) == sort(topo.loop_nelem)
        @test sort(topo2.loop_name) == sort(topo.loop_name)

        # No element sets: loops are still found, and named by number.
        bare = Mesh([Point(p) for p in square_pts((0, 0), 1.0)],
                    [Line(Int32(i), Int32(mod1(i + 1, 4))) for i in 1:4])
        topo, _ = poly_check(bare)
        @test topo.loop_name == ["loop 1"]

        # Islands and a thin plate are ordinary loops to the topology.
        topo, area = poly_check(poly_mesh([square_pts((0.0, 0.0), 1e-3), square_pts((0.01, 0.0), 1e-3),
                                           plate_pts((0.5, 0.5), 0.8, 1e-4; θ=0.2)]))
        @test nloops(topo) == 3
        @test area ≈ [1e-6, 1e-6, 0.8e-4] rtol = 1e-12
    end

    @testset "rejects $(name)" for (name, mesh, msg) in invalid_poly_meshes()
        err = try
            pt_cache(mesh)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        if name == "fold-back"
            # Folding back also puts the next node on the element folded over, so either check may
            # fire first.
            @test err isa ArgumentError && occursin(r"fold back|intersects itself", err.msg)
        else
            @test err isa ArgumentError && occursin(msg, err.msg)
        end
    end

    # (A mesh that is not 2D lines no longer reaches the cut: the cache takes an `SDFMesh{2}`, so
    # dispatch refuses it.)

    @testset "fingerprint" begin
        # What decides whether an update rebuilds the loops: the connectivity and the element
        # sets, not the coordinates.
        loops = three_element_pts()
        fingerprint(mesh) = pt_cache(mesh).work.topo.fingerprint
        fp = fingerprint(poly_mesh(loops; names=["slat", "main", "flap"]))
        # Rigid motion moves the nodes and keeps the connectivity: same fingerprint, same topology.
        R, t = _rot(0.37), PV(3.1, -2.4)
        moved = poly_mesh([[R * p + t for p in pts] for pts in loops]; names=["slat", "main", "flap"])
        @test fingerprint(moved) == fp
        topo, area = poly_check(moved)
        @test area ≈ pts_area.(loops) rtol = 1e-13
        # A change of connectivity or of set membership changes it. A renamed set does not: an
        # `SDFMesh` keeps no names, so the cut never sees one.
        @test fingerprint(poly_mesh(loops; names=["slat", "main", "flap2"])) == fp
        @test fingerprint(poly_mesh(loops[1:2]; names=["slat", "main"])) != fp
        perm = randperm(rng, sum(length, loops))
        @test fingerprint(poly_mesh(loops; names=["slat", "main", "flap"], perm=perm)) != fp
    end
end
