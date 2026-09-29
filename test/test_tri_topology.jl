# =====================================
# Tri clipping, phase 2: validating a mesh and reading its patches.
#
# What is asserted:
#
#   * every malformed input is refused with an ArgumentError naming the problem: a collapsed
#     triangle, an open edge, a non-manifold edge, a triangle wound the wrong way, a surface wound
#     inwards, element sets that overlap or leave triangles out, a mesh that is not of triangles
#   * patches are element sets split into connected components
#   * how patches meet: every box pair convex, the L-block's inner corner concave, a hemisphere
#     seam tangent under a wide enough tangent angle
#   * planarity, and the forced-planar names
#   * the mesh's own totals against the analytic ones
#   * the GPPH hull (gpph_clean.inp) loads and validates

using CutCellMethods: build_topology, TriTopology, npatches, PAIR_CONVEX, PAIR_CONCAVE,
                      PAIR_TANGENT, _patch_planes, _mesh_coords, _mesh_tris

include("tri_meshes.jl")

const TT_H = 0.05
const TT_TC = TriClippingCutCell()

tt_patch(topo, name) = findfirst(==(name), topo.patch_name)
tt_pair(topo, a, b) = topo.pair_mask[tt_patch(topo, a), tt_patch(topo, b)]

@testset verbose = true "tri clipping: topology" begin
    box, Vbox, box_areas = tm_box(SVector(0.1, -0.2, 0.3), SVector(1.4, 0.9, 1.0); n=3)

    @testset "a valid box" begin
        topo = build_topology(box, TT_TC, TT_H)
        @test topo isa TriTopology
        @test npatches(topo) == 6
        @test topo.ntri == 6 * 2 * 9
        @test topo.volume ≈ Vbox rtol = 1e-14
        for (name, A) in box_areas
            @test topo.patch_area[tt_patch(topo, name)] ≈ A rtol = 1e-14
        end
        @test all(topo.patch_planar)
        @test all(<(1e-14), topo.patch_planarity)
        # Faces meeting along an edge are convex; opposite faces never meet.
        for a in ("-x", "+x"), b in ("-y", "+y", "-z", "+z")
            @test tt_pair(topo, a, b) == PAIR_CONVEX
        end
        @test tt_pair(topo, "-x", "+x") == 0x00
        # Inside a patch a side carries no label; across a patch boundary it names the neighbour.
        for t in 1:topo.ntri, s in 1:3
            u = topo.side_nbr[t][s]
            if topo.tri_patch[u] == topo.tri_patch[t]
                @test topo.side_label[t][s] == 0x00 && topo.side_patch[t][s] == 0
            else
                @test topo.side_label[t][s] == PAIR_CONVEX && topo.side_patch[t][s] == topo.tri_patch[u]
            end
        end
        # The global planes: each face's outward normal and offset.
        X = _mesh_coords(box)
        planes = _patch_planes(topo, X, _mesh_tris(box))
        p = planes[tt_patch(topo, "+y")]
        @test SVector(p[1], p[2], p[3]) ≈ SVector(0.0, 1.0, 0.0)
        @test p[4] ≈ 0.9
        p = planes[tt_patch(topo, "-z")]
        @test SVector(p[1], p[2], p[3]) ≈ SVector(0.0, 0.0, -1.0) && p[4] ≈ -0.3
    end

    @testset "a rotated box" begin
        R = SMatrix{3,3,Float64}(cosd(30), sind(30), 0, -sind(30), cosd(30), 0, 0, 0, 1) *
            SMatrix{3,3,Float64}(1, 0, 0, 0, cosd(17), sind(17), 0, -sind(17), cosd(17))
        rbox, V, _ = tm_box(SVector(-0.5, -0.4, -0.3), SVector(0.5, 0.4, 0.3); n=2, R=R,
                            t=SVector(0.2, 0.1, -0.05))
        topo = build_topology(rbox, TT_TC, TT_H)
        @test topo.volume ≈ V rtol = 1e-14
        @test all(topo.patch_planar)
        @test tt_pair(topo, "+x", "+z") == PAIR_CONVEX
    end

    @testset "the prism hull" begin
        hull, V, _ = tm_prism(L=2.0, beam=1.0, deadrise=20.0, depth=0.8, nx=5)
        topo = build_topology(hull, TT_TC, TT_H)
        @test npatches(topo) == 7
        @test topo.volume ≈ V rtol = 1e-14
        @test all(topo.patch_planar)
        # A convex hull: every crease convex, including the keel and both chines.
        @test tt_pair(topo, "bottom+", "bottom-") == PAIR_CONVEX
        @test tt_pair(topo, "bottom+", "side+") == PAIR_CONVEX
        @test tt_pair(topo, "transom", "bottom-") == PAIR_CONVEX
        @test tt_pair(topo, "transom", "bow") == 0x00
    end

    @testset "the L-block's concave corner" begin
        lb, V = tm_lblock(a=0.5, hz=0.7)
        topo = build_topology(lb, TT_TC, TT_H)
        @test topo.volume ≈ V rtol = 1e-14
        @test tt_pair(topo, "y=a", "x=a") == PAIR_CONCAVE
        @test tt_pair(topo, "y=0", "x=2a") == PAIR_CONVEX
        @test tt_pair(topo, "z=hz", "x=a") == PAIR_CONVEX
        @test tt_pair(topo, "z=0", "y=a") == PAIR_CONVEX
    end

    @testset "a smooth seam" begin
        sph, V, A = tm_icosphere(r=0.9, level=3, split=true)
        # At the default 1°, the faceting of a coarse sphere reads as convex creases...
        topo = build_topology(sph, TT_TC, TT_H)
        @test npatches(topo) == 2
        @test tt_pair(topo, "north", "south") == PAIR_CONVEX
        @test !any(topo.patch_planar)
        # ...and above the facet angle as one surface split in two.
        topo15 = build_topology(sph, TriClippingCutCell(tangent_angle=15), TT_H)
        @test tt_pair(topo15, "north", "south") == PAIR_TANGENT
        # The faceted sphere is inside the true one, a little short of its volume and area.
        @test 0.97 * V < topo.volume < V
        @test 0.97 * A < sum(topo.patch_area) < A
        # One patch covering the whole sphere has no plane to be near.
        one = build_topology(tm_icosphere(r=0.9, level=2)[1], TT_TC, TT_H)
        @test npatches(one) == 1 && isinf(one.patch_planarity[1])
    end

    @testset "sets split into connected patches" begin
        # Both chine-to-deck sides in one set: two patches, one name.
        hull, _, _ = tm_prism(nx=3)
        sets = [MeshElementSet(s.name == "side-" ? "side+" : s.name, s.elems) for s in hull.elemset]
        merged = Dict{String,Vector{Int}}()
        for s in sets
            append!(get!(merged, s.name, Int[]), s.elems)
        end
        hull2 = tm_retri(hull, tm_tris(hull); sets=[MeshElementSet(k, v) for (k, v) in merged])
        topo = build_topology(hull2, TT_TC, TT_H)
        @test npatches(topo) == 7
        @test count(==("side+"), topo.patch_name) == 2
        @test length(unique(topo.patch_set)) == 6
    end

    @testset "forced planar" begin
        sph, _, _ = tm_icosphere(level=2, split=true)
        topo = build_topology(sph, TriClippingCutCell(planar_patches=["north"]), TT_H)
        @test topo.patch_planar[tt_patch(topo, "north")]
        @test !topo.patch_planar[tt_patch(topo, "south")]
        @test_logs (:warn, r"not an element set") match_mode = :any build_topology(
            sph, TriClippingCutCell(planar_patches=["nope"]), TT_H)
    end

    @testset "malformed input is refused" begin
        tris = tm_tris(box)
        err(mesh) = try
            build_topology(mesh, TT_TC, TT_H)
            ""
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
        # An open surface: one triangle removed (and dropped from its set).
        sets = [MeshElementSet(s.name, [i > 1 ? i - 1 : i for i in s.elems if i != 1]) for s in box.elemset]
        @test occursin("not closed", err(tm_retri(box, tris[2:end]; sets=sets)))
        # One triangle wound the wrong way.
        flipped = copy(tris)
        flipped[5] = SVector(tris[5][1], tris[5][3], tris[5][2])
        @test occursin("not consistently wound", err(tm_retri(box, flipped)))
        # Every triangle wound the wrong way.
        inward = [SVector(c[1], c[3], c[2]) for c in tris]
        @test occursin("wound inwards", err(tm_retri(box, inward)))
        # A third triangle on an existing edge, in the "-x" set.
        extra = vcat(tris, [SVector(tris[1][1], tris[1][2], Int32(length(box.nodes)))])
        sets = [MeshElementSet(s.name, s.name == "-x" ? vcat(s.elems, length(extra)) : s.elems) for s in box.elemset]
        msg = err(tm_retri(box, extra; sets=sets))
        @test occursin("not manifold", msg) || occursin("not closed", msg)
        # A collapsed triangle.
        collapsed = copy(tris)
        collapsed[3] = SVector(tris[3][1], tris[3][1], tris[3][2])
        @test occursin("repeats a node", err(tm_retri(box, collapsed)))
        # Element sets that overlap, and that leave triangles out.
        s1 = box.elemset
        overlap = [MeshElementSet(s1[1].name, vcat(s1[1].elems, s1[2].elems[1])); s1[2:end]]
        @test occursin("two element sets", err(tm_retri(box, tris; sets=overlap)))
        @test occursin("in no element set", err(tm_retri(box, tris; sets=s1[2:end])))
        @test occursin("outside", err(tm_retri(box, tris; sets=[s1; MeshElementSet("bad", [10_000])])))
        # Not a triangle mesh.
        quads = Mesh([Point(SVector(0.0, 0.0)), Point(SVector(1.0, 0.0)), Point(SVector(1.0, 1.0))],
                     [Tri(SVector{3,Int32}(1, 2, 3))])
        @test_throws ArgumentError build_topology(quads, TT_TC, TT_H)
    end

    @testset "no element sets: one patch, with a warning" begin
        bare = Mesh(collect(box.nodes), collect(box.elements))
        # Warned twice: no sets, and the box's 90° edges now fall inside the one patch.
        topo = @test_logs (:warn, r"no element sets") (:warn, r"bends by 90") build_topology(bare, TT_TC, TT_H)
        @test npatches(topo) == 1 && topo.patch_name == ["body"]
        @test !topo.patch_planar[1]
    end

    @testset "the GPPH hull" begin
        hull = tm_gpph()
        if hull === nothing
            @info "gpph_clean.inp not generated (examples/geometry/run_generate_mesh.jl): skipped"
        else
            # (Its patch-check warnings, if any, are captured rather than required.)
            topo = @test_logs match_mode = :any build_topology(hull, TT_TC, 0.02)
            @test npatches(topo) == 8
            # The transom and the deck are the planar CAD surfaces.
            @test sort(topo.patch_name[topo.patch_planar]) == ["surface_4", "surface_7"]
            # The bottom meets its chine flat concavely, on both sides.
            @test topo.pair_mask[tt_patch(topo, "surface_2"), tt_patch(topo, "surface_5")] & PAIR_CONCAVE != 0
            @test topo.pair_mask[tt_patch(topo, "surface_3"), tt_patch(topo, "surface_6")] & PAIR_CONCAVE != 0
            # The mesh's volume is the hull's to within its faceting.
            @test topo.volume ≈ 16.21 rtol = 2e-3
        end
    end
end
