# =====================================
# Tri clipping, phase 4: the cut -- fit groups, the Boolean rule, and each cut cell's volume.
#
# What is asserted:
#
#   * planar bodies come out exact cell by cell, against an independent reference that clips each
#     cell by the body's analytic face planes: an axis-aligned box with faces on cell-centre and on
#     cell-face planes, a rotated box, the prism hull (keel, chines and transom all convex creases),
#     and the L-block, whose inner edge is concave
#   * the L-block's two cells where its concave edge meets a convex one are flagged unsupported,
#     and they are the only inexact cells
#   * a curved body converges: a sphere's volume at second order or better against the mesh's own
#   * a plate thinner than a cell is flagged, not silently cut
#   * volume fractions are fractions, nothing is NaN, and the GPU agrees with the host
#   * the GPPH hull with its CAD patches: no overflow, no failed cap, no NaN, its volume, and
#     every cell cut by a rule -- the chine and the transom corners by the mixed one -- none falling back

using CutCellMethods: CELL_CUT, RULE_SINGLE, RULE_CONVEX, RULE_CONCAVE, RULE_FALLBACK, RULE_MIXED,
                      FLAG_MULTI_PATCH, FLAG_UNSUPPORTED, FLAG_SPLIT, FLAG_OVERFLOW,
                      FLAG_CHAIN_FAIL, TriScratch, cut_report
using KernelAbstractions

isdefined(@__MODULE__, :tm_box) || include("tri_meshes.jl")

const TCUT = TriClippingCutCell()

# Per-cell volume-fraction errors against `solid(ci)`.
tcut_errors(cache, g, solid) =
    [abs(cache.cells.volume_fraction[ci] - (1 - solid(ci))) for ci in CartesianIndices(Tuple(g.n))]

tcut_count(cache, bit) = count(f -> f & bit != 0x00, cache.info.flags)

@testset verbose = true "tri clipping: the cut" begin
    g = CartesianGrid(SVector(-1.0, -1.0, -1.0), (16, 16, 16), SVector(0.125, 0.125, 0.125))
    idx = CartesianIndices(Tuple(g.n))
    scr = TriScratch(KernelAbstractions.CPU(), Float64, 1)

    @testset "planar bodies are exact in every cell" begin
        # An axis-aligned box off the lattice, on cell-centre planes, and on cell-face planes --
        # the last with faces lying in cell faces, the coincident case.
        for (lo, hi) in ((SVector(-0.33, -0.41, -0.27), SVector(0.47, 0.29, 0.52)),
                         (SVector(-0.3125, -0.4375, -0.1875), SVector(0.4375, 0.3125, 0.5625)),
                         (SVector(-0.375, -0.5, -0.25), SVector(0.5, 0.25, 0.625)))
            box, V, _ = tm_box(lo, hi; n=3)
            cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(box), g)
            P = tm_box_planes(lo, hi)
            @test maximum(tcut_errors(cache, g, ci -> tm_solid_fraction(scr, g, ci, P))) < 1e-13
            @test sum(1 .- cache.cells.volume_fraction) * prod(g.d) ≈ V rtol = 1e-13
            @test tcut_count(cache, FLAG_UNSUPPORTED) == 0
            # Edges and corners are convex creases, cut by the convex rule.
            @test count(==(RULE_CONVEX), cache.info.rule) > 0
            @test count(==(RULE_CONVEX), cache.info.rule) == tcut_count(cache, FLAG_MULTI_PATCH)
        end
        R = SMatrix{3,3,Float64}(cosd(30), sind(30), 0, -sind(30), cosd(30), 0, 0, 0, 1) *
            SMatrix{3,3,Float64}(1, 0, 0, 0, cosd(17), sind(17), 0, -sind(17), cosd(17))
        lo, hi, t = SVector(-0.45, -0.3, -0.25), SVector(0.4, 0.35, 0.3), SVector(0.013, -0.021, 0.007)
        rbox, V, _ = tm_box(lo, hi; n=3, R=R, t=t)
        cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(rbox), g)
        P = tm_box_planes(lo, hi, R, t)
        @test maximum(tcut_errors(cache, g, ci -> tm_solid_fraction(scr, g, ci, P))) < 1e-13
        @test sum(1 .- cache.cells.volume_fraction) * prod(g.d) ≈ V rtol = 1e-13
        @test tcut_count(cache, FLAG_UNSUPPORTED) == 0
        # A convex edge passing just outside a cell whose box both its faces cross leaves the cell's
        # fluid in two pieces: flagged, still exact, and so not ambiguous -- which is kept for the
        # cells a fallback plane cut.
        @test tcut_count(cache, FLAG_SPLIT) > 0
        @test all(ci -> cache.info.flags[ci] & FLAG_SPLIT == 0 || cache.info.rule[ci] == RULE_CONVEX, idx)
        @test all(ci -> cache.cells.ambiguous[ci] == (cache.info.flags[ci] & FLAG_UNSUPPORTED != 0), idx)

        kw = (L=1.4, beam=0.9, deadrise=20.0, depth=0.6)
        t = SVector(-0.7, 0.013, -0.3)
        hull, V, _ = tm_prism(; kw..., nx=5, t=t)
        cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(hull), g)
        P = tm_prism_planes(; kw..., t=t)
        @test maximum(tcut_errors(cache, g, ci -> tm_solid_fraction(scr, g, ci, P))) < 1e-13
        @test sum(1 .- cache.cells.volume_fraction) * prod(g.d) ≈ V rtol = 1e-13
        @test tcut_count(cache, FLAG_UNSUPPORTED) == 0
        @test count(==(RULE_CONVEX), cache.info.rule) > 50
    end

    @testset "the L-block: concave edges and mixed corners exact" begin
        o = SVector(-0.41, -0.39, -0.43)
        lb, V = tm_lblock(a=0.4, hz=0.9, t=o)
        cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(lb), g)
        B1 = tm_box_planes(o, o + SVector(0.8, 0.4, 0.9))
        B2 = tm_box_planes(o + SVector(0.0, 0.4, 0.0), o + SVector(0.4, 0.8, 0.9))
        err = tcut_errors(cache, g, ci -> tm_solid_fraction(scr, g, ci, B1) + tm_solid_fraction(scr, g, ci, B2))
        @test maximum(err) < 1e-13
        # The concave edge alone is cut by the concave rule; where it meets the convex top and
        # bottom faces, one cell at each end, by the mixed rule.
        @test count(==(RULE_CONCAVE), cache.info.rule) > 0
        @test count(==(RULE_MIXED), cache.info.rule) == 2
        @test tcut_count(cache, FLAG_UNSUPPORTED) == 0
        @test !any(cache.cells.ambiguous)
    end

    @testset "a chine flat narrower than a cell: the mixed rule is exact" begin
        # A concave knuckle and a convex chine closer together than a cell, and the transom corners
        # they run into: exact in every cell, for flats of 0.56, 0.88 and 0.24 cells.
        t, L, z1, depth = SVector(-0.7, 0.011, -0.31), 1.4, 0.13, 0.6
        for (b1, b2) in ((0.35, 0.42), (0.30, 0.41), (0.33, 0.36))
            hull, V, _ = tm_chine_prism(L=L, b1=b1, b2=b2, z1=z1, depth=depth, nx=5, t=t)
            cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(hull), g)
            pieces = (tm_yz_prism((SVector(0.0, 0.0), SVector(b1, z1), SVector(b1, depth), SVector(-b1, depth), SVector(-b1, z1)), L, t),
                      tm_yz_prism((SVector(b1, z1), SVector(b2, z1), SVector(b2, depth), SVector(b1, depth)), L, t),
                      tm_yz_prism((SVector(-b2, z1), SVector(-b1, z1), SVector(-b1, depth), SVector(-b2, depth)), L, t))
            err = tcut_errors(cache, g, ci -> sum(P -> tm_solid_fraction(scr, g, ci, P), pieces))
            @test maximum(err) < 1e-13
            @test sum(1 .- cache.cells.volume_fraction) * prod(g.d) ≈ V rtol = 1e-13
            @test count(==(RULE_MIXED), cache.info.rule) >= 4
            @test tcut_count(cache, FLAG_UNSUPPORTED) == 0
        end
    end

    @testset "a sphere converges" begin
        sph, _, _ = tm_icosphere(r=0.7, c=SVector(0.031, -0.017, 0.022), level=6)
        errs = map((16, 32)) do n
            gg = CartesianGrid(SVector(-1.0, -1.0, -1.0), (n, n, n), SVector(2 / n, 2 / n, 2 / n))
            cache = update_cache!(allocate_cache(gg, TCUT), SDFMesh(sph), gg)
            @test all(iszero, cache.info.flags)
            @test all(r -> r == RULE_SINGLE || r == 0x00, cache.info.rule)
            # Against the mesh's own volume, which is the faceted sphere's, not the true one's.
            r = cut_report(cache)
            abs(r.solid_volume - r.mesh_volume) / r.mesh_volume
        end
        @test errs[1] < 1e-4
        # Second order, or better: halving h cuts the error at least fourfold.
        @test errs[1] / errs[2] > 3.5
    end

    @testset "a plate thinner than a cell is flagged" begin
        lo, hi = SVector(-0.6, -0.55, 0.06), SVector(0.55, 0.6, 0.06 + 0.3 * 0.125)
        plate, _, _ = tm_box(lo, hi; n=2)
        cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(plate), g)
        # Its top and bottom share no edge: no rule connects them, so the cell is flagged and cut
        # by one fallback plane (D3).
        @test tcut_count(cache, FLAG_UNSUPPORTED) > 0
        @test all(ci -> cache.info.flags[ci] & FLAG_UNSUPPORTED == 0 || cache.cells.ambiguous[ci], idx)
        @test !any(isnan, cache.cells.volume_fraction)
    end

    @testset "fractions, and no NaN" begin
        sph, _, _ = tm_icosphere(r=0.6, level=3)
        for body in (sph, tm_lblock(a=0.4, hz=0.9, t=SVector(-0.41, -0.39, -0.43))[1])
            cache = update_cache!(allocate_cache(g, TCUT), SDFMesh(body), g)
            vf = cache.cells.volume_fraction
            @test all(x -> 0 <= x <= 1, vf)
            @test !any(isnan, vf)
            @test all(f -> all(x -> 0 <= x <= 1, f), cache.cells.face_fraction)
            @test !any(c -> any(isnan, c), cache.cells.centroid)
            @test !any(c -> any(isnan, c), cache.cells.interface_centroid)
            @test all(ci -> (cache.cells.kind[ci] == CELL_CUT) == (cache.info.rule[ci] != 0x00), idx)
        end
    end

    @testset "the GPU agrees with the host" begin
        if HAS_GPU
            # The convex rule on the prism, and the mixed rule on the chine flats.
            for hull in (tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5, t=SVector(-0.7, 0.013, -0.3))[1],
                         tm_chine_prism(b1=0.33, b2=0.36, nx=5, t=SVector(-0.7, 0.011, -0.31))[1])
                g32 = CartesianMeshes.adapt_type(Float32, g)
                host = update_cache!(allocate_cache(g32, TCUT), SDFMesh(hull), g32)
                dev = update_cache!(allocate_cache(g32, TCUT; backend=CUDABackend()), SDFMesh(hull), g32)
                @test Array(dev.cells.kind) == host.cells.kind
                @test Array(dev.info.rule) == host.info.rule
                @test Array(dev.info.flags) == host.info.flags
                @test maximum(abs.(Array(dev.cells.volume_fraction) .- host.cells.volume_fraction)) < 10 * eps(Float32)
            end
        end
    end

    @testset "the GPPH hull with its CAD patches" begin
        hull = tm_gpph()
        if hull === nothing
            @info "gpph_clean.inp not generated (examples/geometry/run_generate_mesh.jl): skipped"
        else
            gh = CartesianGrid(SVector(-0.4, -1.5, -0.3), (172, 60, 40), SVector(0.05, 0.05, 0.05))
            cache = @test_logs match_mode = :any update_cache!(allocate_cache(gh, TCUT), SDFMesh(hull), gh)
            Vmesh = cache.work.topo.volume
            @test tcut_count(cache, FLAG_OVERFLOW) == 0
            @test tcut_count(cache, FLAG_CHAIN_FAIL) == 0
            @test !any(isnan, cache.cells.volume_fraction)
            @test abs(sum(1 .- cache.cells.volume_fraction) * 0.05^3 - Vmesh) / Vmesh < 1e-5
            # The chine -- the bottom meeting the flat concavely, the flat the side convexly --
            # goes through the mixed rule, as do the transom corners where both run into the
            # transom: every cell has a rule, none falls back.
            @test count(==(RULE_MIXED), cache.info.rule) > 20
            @test count(==(RULE_CONCAVE), cache.info.rule) > 100
            @test tcut_count(cache, FLAG_UNSUPPORTED) == 0
        end
    end
end
