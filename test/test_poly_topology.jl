# =====================================
# Polyline clipping, P1: loops and mesh validation

using Test
using StaticArrays
using MeshLibrary
using Random
using CutCellMethods: PolylineClippingCutCell, PolylineTopology, build_polyline_topology,
                      validate_polyline_geometry, nloops, _pl_fingerprint, _pl_coords, _pl_lines

isdefined(@__MODULE__, :PV) || include("polyline_meshes.jl")

# Topology plus the geometric checks, as an update runs them.
function poly_check(mesh)
    topo = build_polyline_topology(mesh)
    return topo, validate_polyline_geometry(topo, _pl_coords(mesh), _pl_lines(mesh))
end

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
        lines = _pl_lines(mesh)
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
            poly_check(mesh)
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

    @testset "rejects non-line meshes" begin
        @test_throws ArgumentError build_polyline_topology(Mesh(Point{3,Float64}, Tri{Int32}))
        tri2 = Mesh([Point(PV(0, 0)), Point(PV(1, 0)), Point(PV(0, 1))], [Tri(Int32(1), Int32(2), Int32(3))])
        @test_throws ArgumentError build_polyline_topology(tri2)
    end

    @testset "fingerprint" begin
        loops = three_element_pts()
        mesh = poly_mesh(loops; names=["slat", "main", "flap"])
        fp = _pl_fingerprint(mesh)
        # Rigid motion moves the nodes and keeps the connectivity: same fingerprint, same topology.
        R, t = _rot(0.37), PV(3.1, -2.4)
        moved = poly_mesh([[R * p + t for p in pts] for pts in loops]; names=["slat", "main", "flap"])
        @test _pl_fingerprint(moved) == fp
        topo, area = poly_check(moved)
        @test area ≈ pts_area.(loops) rtol = 1e-13
        # A change of connectivity or of the sets changes it.
        @test _pl_fingerprint(poly_mesh(loops; names=["slat", "main", "flap2"])) != fp
        @test _pl_fingerprint(poly_mesh(loops[1:2]; names=["slat", "main"])) != fp
        perm = randperm(rng, sum(length, loops))
        @test _pl_fingerprint(poly_mesh(loops; names=["slat", "main", "flap"], perm=perm)) != fp
    end
end
