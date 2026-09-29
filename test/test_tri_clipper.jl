# =====================================
# Tri clipping, phase 1: the two clippers on a slot's scratch.
#
# What is asserted:
#
#   * the convex-polytope clipper against the brief's cases: an axis-aligned cut exact, a corner
#     cut through three edge midpoints 1/48 of the cube, a two-plane wedge against its analytic
#     volume, and planes through a vertex, along an edge and coincident with a face
#   * 10^4 random single planes against CartesianMeshes' analytic plane-cube volume, on the unit
#     cube and on a stretched box, with every face's vector area summing to zero
#   * 10^5 random eight-plane clips: never an overflow or a failed cap, always closed, volume never
#     growing, and a sample against Monte Carlo
#   * the convex-crease fluid decomposition against box minus solid
#   * the planar polygon clipper: triangle against box, face square against planes, closed vs open
#   * zero allocations and inferred return types, and the same numbers from inside a
#     KernelAbstractions kernel on the CPU and, where there is one, the GPU

using Random
using KernelAbstractions
using CutCellMethods: TriScratch, poly_box!, poly_clip!, poly_volume_moment, poly_tag_moment,
                      poly_area_vector_sum, poly_face_area_vector, poly_nf, poly_nv,
                      poly_patch_components, fluid_convex_moments, pg_tri_box!, pg_rect!,
                      pg_clip!, pg_area_moment, FLAG_OVERFLOW, FLAG_CHAIN_FAIL

const TC_TOL = 1e-12

tc_box_volume(scr, U) = first(poly_volume_moment(scr, 1, U / 2))

# The volume of `n . x <= d` in the box [0, U], by CartesianMeshes' analytic unit-cube formula:
# with x = U .* y the half-space is (n .* U) . y <= d in the unit cube.
tc_exact_volume(n, d, U) = CartesianMeshes.get_volume_fraction(n .* U, d) * prod(U)

# A random unit normal.
tc_randn3(rng) = normalize(SVector(randn(rng), randn(rng), randn(rng)))

# Monte Carlo volume of the box [0, U] kept by every plane `(n, d)` in `planes` (n . x <= d).
function tc_mc_volume(rng, U, planes, N)
    hit = 0
    for _ in 1:N
        x = SVector(rand(rng), rand(rng), rand(rng)) .* U
        hit += all(p -> dot(p[1], x) <= p[2], planes)
    end
    return hit / N * prod(U)
end

@kernel function tc_clip_kernel!(vol, scr, @Const(normals), @Const(offsets), U, tol)
    s = @index(Global, Linear)
    poly_box!(scr, s, U)
    poly_clip!(scr, s, normals[s], offsets[s], 7, tol, false)
    v, _ = poly_volume_moment(scr, s, U / 2)
    @inbounds vol[s] = v
end

@testset verbose = true "tri clipping: clippers" begin
    scr = TriScratch(KernelAbstractions.CPU(), Float64, 1)
    U1 = SVector(1.0, 1.0, 1.0)
    Us = SVector(0.7, 1.3, 0.45)

    @testset "the box" begin
        for U in (U1, Us)
            poly_box!(scr, 1, U)
            V, M = poly_volume_moment(scr, 1, U / 2)
            @test V ≈ prod(U) atol = 1e-15
            @test M / V ≈ U / 2 atol = 1e-15
            @test poly_area_vector_sum(scr, 1) == zero(SVector{3,Float64})
            @test poly_nf(scr, 1) == 6 && poly_nv(scr, 1) == 8
            # Each face's vector area is its outward normal times its area, tagged by direction.
            for dir in 1:6
                S, A, _ = poly_tag_moment(scr, 1, dir)
                ax = CartesianMeshes.direction_axis(dir)
                full = prod(U) / U[ax]
                @test S ≈ CartesianMeshes.direction_sign(dir) * full * SVector(ntuple(a -> a == ax ? 1.0 : 0.0, 3))
                @test A ≈ full
            end
        end
    end

    @testset "axis-aligned cuts are exact" begin
        for U in (U1, Us), ax in 1:3, t in (0.1, 0.3, 0.5, 0.77)
            e = SVector(ntuple(a -> a == ax ? 1.0 : 0.0, 3))
            poly_box!(scr, 1, U)
            @test poly_clip!(scr, 1, e, t * U[ax], 7, TC_TOL, false) == 0x00
            @test tc_box_volume(scr, U) ≈ t * prod(U) rtol = 4eps()
            poly_box!(scr, 1, U)
            @test poly_clip!(scr, 1, -e, -t * U[ax], 7, TC_TOL, false) == 0x00
            @test tc_box_volume(scr, U) ≈ (1 - t) * prod(U) rtol = 4eps()
            # The cap is the cut face: area of the box's cross-section, normal along +e.
            S, A, M = poly_tag_moment(scr, 1, 7)
            @test A ≈ prod(U) / U[ax]
            @test S ≈ -A * e
            @test (M / A)[ax] ≈ t * U[ax]
        end
    end

    @testset "corner cut through three edge midpoints is 1/48" begin
        n = normalize(SVector(1.0, 1.0, 1.0))
        poly_box!(scr, 1, U1)
        @test poly_clip!(scr, 1, n, 0.5 / sqrt(3), 7, TC_TOL, false) == 0x00
        @test tc_box_volume(scr, U1) ≈ 1 / 48 rtol = 1e-14
        @test poly_nf(scr, 1) == 4
        _, A, _ = poly_tag_moment(scr, 1, 7)
        @test A ≈ sqrt(3) / 8
    end

    @testset "a two-plane wedge" begin
        # Solid under z = 0.5 - 0.3|x - 0.5|, extruded in y: volume 0.5 - 0.3/4.
        p1 = SVector(-0.3, 0.0, 1.0)
        p2 = SVector(0.3, 0.0, 1.0)
        n1, d1 = p1 / norm(p1), 0.35 / norm(p1)
        n2, d2 = p2 / norm(p2), 0.65 / norm(p2)
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, n1, d1, 7, TC_TOL, false)
        poly_clip!(scr, 1, n2, d2, 8, TC_TOL, false)
        Vs, Ms = poly_volume_moment(scr, 1, U1 / 2)
        @test Vs ≈ 0.425 rtol = 1e-14
        @test poly_area_vector_sum(scr, 1) ≈ zero(SVector{3,Float64}) atol = 1e-15
        # The two patch faces meet along the crease: one component.
        @test poly_patch_components(scr, 1) == 1
        # The fluid, as disjoint convex pieces, is the box less the solid.
        planes = (SVector(n1..., d1), SVector(n2..., d2))
        Vf, Mf, st = fluid_convex_moments(scr, 1, U1, planes, 2, TC_TOL)
        @test st == 0x00
        @test Vf ≈ 1 - 0.425 rtol = 1e-14
        @test Mf + Ms ≈ U1 / 2 atol = 1e-15
    end

    @testset "degenerate planes" begin
        n = normalize(SVector(1.0, 1.0, 1.0))
        # Through three vertices: the corner tetrahedron, 1/6.
        poly_box!(scr, 1, U1)
        @test poly_clip!(scr, 1, n, 1 / sqrt(3), 7, TC_TOL, false) == 0x00
        @test tc_box_volume(scr, U1) ≈ 1 / 6 rtol = 1e-14
        @test poly_nv(scr, 1) == 4 && poly_nf(scr, 1) == 4
        # ...and its complement, which keeps five of the corners on the plane side.
        poly_box!(scr, 1, U1)
        @test poly_clip!(scr, 1, -n, -1 / sqrt(3), 7, TC_TOL, false) == 0x00
        @test tc_box_volume(scr, U1) ≈ 5 / 6 rtol = 1e-14
        @test poly_area_vector_sum(scr, 1) ≈ zero(SVector{3,Float64}) atol = 1e-15
        # Along a face diagonal edge-to-edge: half the cube, both ways.
        m = normalize(SVector(1.0, 1.0, 0.0))
        for sgn in (1, -1)
            poly_box!(scr, 1, U1)
            @test poly_clip!(scr, 1, sgn * m, sgn / sqrt(2), 7, TC_TOL, false) == 0x00
            @test tc_box_volume(scr, U1) ≈ 0.5 rtol = 1e-14
            @test poly_nf(scr, 1) == 5
        end
        # Containing one box edge: removes nothing, or everything.
        poly_box!(scr, 1, U1)
        @test poly_clip!(scr, 1, m, 2 / sqrt(2), 7, TC_TOL, false) == 0x00
        @test tc_box_volume(scr, U1) ≈ 1
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, -m, -2 / sqrt(2), 7, TC_TOL, false)
        @test poly_nf(scr, 1) == 0
        # Coincident with a face: nothing kept on the far side, nothing lost on the near side.
        ex = SVector(1.0, 0.0, 0.0)
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, ex, 0.0, 7, TC_TOL, false)
        @test poly_nf(scr, 1) == 0
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, -ex, 0.0, 7, TC_TOL, false)
        @test tc_box_volume(scr, U1) ≈ 1
        # With `retag` (a fluid piece), the coincident face becomes the interface...
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, -ex, 0.0, 9, TC_TOL, true)
        S, A, _ = poly_tag_moment(scr, 1, 9)
        @test A ≈ 1 && S ≈ SVector(-1.0, 0.0, 0.0)
        @test first(poly_tag_moment(scr, 1, 1)) == zero(SVector{3,Float64})
        # ...and without it the Cartesian face keeps its tag.
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, -ex, 0.0, 9, TC_TOL, false)
        @test poly_tag_moment(scr, 1, 1)[2] ≈ 1
        # Within the snap tolerance counts as on the plane.
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, -ex, 1e-14, 9, TC_TOL, true)
        @test tc_box_volume(scr, U1) ≈ 1 && poly_tag_moment(scr, 1, 9)[2] ≈ 1
        # The same plane twice, and a plane through vertices a previous clip inserted.
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, ex, 0.5, 7, TC_TOL, false)
        @test poly_clip!(scr, 1, ex, 0.5, 8, TC_TOL, false) == 0x00
        @test tc_box_volume(scr, U1) ≈ 0.5
        @test poly_clip!(scr, 1, normalize(SVector(1.0, 0.0, 1.0)), 0.5 / sqrt(2), 8, TC_TOL, false) == 0x00
        @test tc_box_volume(scr, U1) ≈ 0.125 rtol = 1e-14
        @test poly_area_vector_sum(scr, 1) ≈ zero(SVector{3,Float64}) atol = 1e-15
    end

    @testset "10^4 random planes against the analytic volume" begin
        rng = Xoshiro(20260928)
        worst = 0.0
        worst_closure = 0.0
        bad = 0
        for U in (U1, Us), _ in 1:5000
            n = tc_randn3(rng)
            # An offset between the box's lowest and highest corner along n: always a real cut.
            lo = sum(min.(n .* U, 0.0))
            hi = sum(max.(n .* U, 0.0))
            d = lo + rand(rng) * (hi - lo)
            poly_box!(scr, 1, U)
            bad += poly_clip!(scr, 1, n, d, 7, TC_TOL, false) != 0x00
            V = tc_box_volume(scr, U)
            worst = max(worst, abs(V - tc_exact_volume(n, d, U)) / prod(U))
            worst_closure = max(worst_closure, maximum(abs, poly_area_vector_sum(scr, 1)))
        end
        @test bad == 0
        @test worst < 1e-12
        @test worst_closure < 1e-14
    end

    @testset "10^5 random eight-plane clips" begin
        rng = Xoshiro(7)
        bad = 0
        grew = 0
        worst_closure = 0.0
        mc_worst = 0.0
        for trial in 1:100_000
            U = trial % 2 == 0 ? U1 : Us
            poly_box!(scr, 1, U)
            Vprev = prod(U)
            planes = NTuple{2,Any}[]
            for k in 1:8
                n = tc_randn3(rng)
                # Through a random point of the box, so every plane is a candidate cut.
                d = dot(n, SVector(rand(rng), rand(rng), rand(rng)) .* U)
                push!(planes, (n, d))
                bad += poly_clip!(scr, 1, n, d, 6 + k, TC_TOL, false) != 0x00
                V = tc_box_volume(scr, U)
                grew += V > Vprev + 1e-15
                Vprev = V
            end
            worst_closure = max(worst_closure, maximum(abs, poly_area_vector_sum(scr, 1)))
            if trial <= 200
                est = tc_mc_volume(rng, U, planes, 20_000)
                # Five binomial standard deviations.
                p = Vprev / prod(U)
                mc_worst = max(mc_worst, abs(est - Vprev) / prod(U) / (5 * sqrt(max(p * (1 - p), 1e-4) / 20_000)))
            end
        end
        @test bad == 0
        @test grew == 0
        @test worst_closure < 1e-14
        @test mc_worst < 1
    end

    @testset "fluid decomposition equals box minus solid" begin
        rng = Xoshiro(11)
        worst_v = 0.0
        worst_m = 0.0
        for trial in 1:2000, k in 2:4
            U = trial % 2 == 0 ? U1 : Us
            c = SVector(rand(rng), rand(rng), rand(rng)) .* U
            planes = ntuple(_ -> (n = tc_randn3(rng); SVector(n..., dot(n, c) + 0.2 * randn(rng))), k)
            poly_box!(scr, 1, U)
            for (i, p) in enumerate(planes)
                poly_clip!(scr, 1, SVector(p[1], p[2], p[3]), p[4], 6 + i, TC_TOL, false)
            end
            Vs, Ms = poly_volume_moment(scr, 1, U / 2)
            Vf, Mf, st = fluid_convex_moments(scr, 1, U, planes, k, TC_TOL)
            @test st == 0x00
            worst_v = max(worst_v, abs(Vf + Vs - prod(U)) / prod(U))
            worst_m = max(worst_m, maximum(abs, Mf + Ms - prod(U) * U / 2) / prod(U))
        end
        @test worst_v < 1e-14
        @test worst_m < 1e-14
    end

    @testset "a split solid" begin
        # A slab crossing the cell: two patch faces with no edge between them.
        poly_box!(scr, 1, U1)
        poly_clip!(scr, 1, SVector(0.0, 0.0, 1.0), 0.6, 7, TC_TOL, false)
        poly_clip!(scr, 1, SVector(0.0, 0.0, -1.0), -0.4, 8, TC_TOL, false)
        @test poly_patch_components(scr, 1) == 2
        @test tc_box_volume(scr, U1) ≈ 0.2
    end

    @testset "planar polygons" begin
        lo = zero(SVector{3,Float64})
        # A triangle covering the whole z = 0.5 cross-section clips to that square.
        b, m = pg_tri_box!(scr, 1, SVector(-1.0, -1.0, 0.5), SVector(3.0, -1.0, 0.5),
                           SVector(-1.0, 3.0, 0.5), lo, U1, TC_TOL)
        A, M = pg_area_moment(scr, 1, b, m, SVector(0.0, 0.0, 1.0))
        @test m == 4 && A ≈ 1 && M / A ≈ SVector(0.5, 0.5, 0.5)
        # Every new vertex sits exactly on the box face it was clipped to.
        @test all(i -> all(x -> x in (0.0, 1.0), scr.pg[i, b, 1][1:2]), 1:m)
        # Inside: untouched. Lying in a box face: kept (the box is closed). Outside: gone.
        b, m = pg_tri_box!(scr, 1, SVector(0.1, 0.1, 0.2), SVector(0.8, 0.2, 0.3),
                           SVector(0.3, 0.9, 0.7), lo, U1, TC_TOL)
        @test m == 3
        b, m = pg_tri_box!(scr, 1, SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0),
                           SVector(0.0, 1.0, 0.0), lo, U1, TC_TOL)
        @test pg_area_moment(scr, 1, b, m, SVector(0.0, 0.0, 1.0))[1] ≈ 0.5
        b, m = pg_tri_box!(scr, 1, SVector(0.0, 0.0, 1.5), SVector(1.0, 0.0, 1.5),
                           SVector(0.0, 1.0, 1.5), lo, U1, TC_TOL)
        @test m == 0
        # A triangle crossing a corner, against a fine Monte Carlo of its plane.
        rng = Xoshiro(3)
        a, bb, c = SVector(-0.4, 0.3, 0.2), SVector(0.9, -0.5, 0.6), SVector(0.5, 1.4, 1.3)
        b, m = pg_tri_box!(scr, 1, a, bb, c, lo, U1, TC_TOL)
        nt = normalize(cross(bb - a, c - a))
        A, _ = pg_area_moment(scr, 1, b, m, nt)
        N = 400_000
        inside = count(1:N) do _
            u, v = rand(rng), rand(rng)
            u + v > 1 && ((u, v) = (1 - u, 1 - v))
            x = a + u * (bb - a) + v * (c - a)
            all(0 .<= x .<= 1)
        end
        Atri = norm(cross(bb - a, c - a)) / 2
        @test A ≈ inside / N * Atri rtol = 0.01

        # A face square against a plane: half of it by the diagonal.
        b, m, nrm = pg_rect!(scr, 1, 3, 0.0, lo, U1)
        b, m = pg_clip!(scr, 1, b, m, normalize(SVector(1.0, 1.0, 0.0)), 1 / sqrt(2), TC_TOL, true)
        @test pg_area_moment(scr, 1, b, m, nrm)[1] ≈ 0.5
        for ax in 1:3
            b, m, nrm = pg_rect!(scr, 1, ax, 0.25, lo, Us)
            @test pg_area_moment(scr, 1, b, m, nrm)[1] ≈ prod(Us) / Us[ax]
            # A plane containing the face: closed keeps it, open (the fluid side) drops it.
            e = SVector(ntuple(a -> a == ax ? 1.0 : 0.0, 3))
            @test pg_clip!(scr, 1, b, m, e, 0.25, TC_TOL, false)[2] == 4
            @test pg_clip!(scr, 1, b, m, e, 0.25, TC_TOL, true)[2] == 0
        end
    end

    @testset "allocation-free and inferred" begin
        n = normalize(SVector(0.3, -0.5, 0.8))
        clip_once(scr, U, n) = (poly_box!(scr, 1, U); poly_clip!(scr, 1, n, 0.1, 7, TC_TOL, false))
        clip_once(scr, U1, n)
        @test (@allocated clip_once(scr, U1, n)) == 0
        @test (@inferred poly_clip!(scr, 1, n, 0.2, 8, TC_TOL, false)) isa UInt8
        @test (@allocated poly_volume_moment(scr, 1, U1 / 2)) == 0
        @test (@inferred poly_volume_moment(scr, 1, U1 / 2)) isa Tuple{Float64,SVector{3,Float64}}
        @test (@allocated poly_tag_moment(scr, 1, 7)) == 0
        @test (@allocated poly_patch_components(scr, 1)) == 0
        planes = (SVector(n..., 0.3), SVector(-n[2], n[1], n[3], 0.2))
        fluid_convex_moments(scr, 1, U1, planes, 2, TC_TOL)
        @test (@allocated fluid_convex_moments(scr, 1, U1, planes, 2, TC_TOL)) == 0
        tri_once(scr) = pg_tri_box!(scr, 1, SVector(-0.4, 0.3, 0.2), SVector(0.9, -0.5, 0.6),
                                    SVector(0.5, 1.4, 1.3), zero(SVector{3,Float64}), U1, TC_TOL)
        tri_once(scr)
        @test (@allocated tri_once(scr)) == 0
        @test (@inferred tri_once(scr)) isa Tuple{Int,Int}
    end

    @testset "inside a kernel" begin
        rng = Xoshiro(5)
        ns = 512
        normals = [tc_randn3(rng) for _ in 1:ns]
        offsets = [dot(normals[i], SVector(rand(rng), rand(rng), rand(rng)) .* Us) for i in 1:ns]
        host = map(1:ns) do i
            poly_box!(scr, 1, Us)
            poly_clip!(scr, 1, normals[i], offsets[i], 7, TC_TOL, false)
            tc_box_volume(scr, Us)
        end
        cpu = KernelAbstractions.CPU()
        kscr = TriScratch(cpu, Float64, ns)
        vol = zeros(ns)
        tc_clip_kernel!(cpu, 64)(vol, kscr, normals, offsets, Us, TC_TOL; ndrange=ns)
        KernelAbstractions.synchronize(cpu)
        @test vol == host
        if HAS_GPU
            gpu = CUDABackend()
            gscr = TriScratch(gpu, Float64, ns)
            gvol = CUDA.zeros(Float64, ns)
            tc_clip_kernel!(gpu, 64)(gvol, gscr, CuArray(normals), CuArray(offsets), Us, TC_TOL; ndrange=ns)
            KernelAbstractions.synchronize(gpu)
            @test Array(gvol) ≈ host rtol = 1e-12
        end
    end
end
