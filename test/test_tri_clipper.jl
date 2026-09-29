# =====================================
# Tri clipping, phase 1: the clippers, through the cache.
#
# The convex-polytope clipper cuts every cut cell, so it is asserted here on bodies whose cells it
# cuts against planes known exactly, read off the cache as a consumer reads it:
#
#   * single planes: a box's faces at random orientations, every cell only one face crosses
#     against CartesianMeshes' analytic plane-cube volume, on a cubic grid and on a stretched one
#   * degenerate planes: faces through lattice nodes and through edge midpoints, where the cut runs
#     through cell vertices and along cell edges, and a corner tetrahedron is exactly 1/48
#   * many planes in one cell: pyramids of 3 to 8 sides, whose apex cell the convex rule cuts with
#     every side at once, against a clip by the body's own planes and against Monte Carlo
#   * the update is inferred, and a device cache cuts the eight-sided apex as the host does
#   * the clipper itself, called directly, as the one exception to going through the cache: every
#     clipped polytope stays closed -- its faces' area vectors sum to zero -- and a clip allocates
#     nothing, neither of which a cell's outputs can show
#
# Near a many-sided apex a cell can hold two sides that meet only at the apex, not along an edge;
# no Boolean rule covers that pair, so the cell is cut by the fallback plane and flagged. Those
# cells are checked for the flag, not compared.

using Random
using KernelAbstractions
using CutCellMethods: TriScratch, boundary_faces, CELL_CUT, RULE_CONVEX,
                      FLAG_OVERFLOW, FLAG_CHAIN_FAIL, FLAG_UNSUPPORTED,
                      poly_box!, poly_clip!, poly_nf, poly_isempty, poly_face_area_vector,
                      poly_volume_moment

isdefined(@__MODULE__, :tm_box) || include("tri_meshes.jl")

const TC = TriClippingCutCell()
const TC_BOX_FACES = ("-x", "+x", "-y", "+y", "-z", "+z")

# The fluid fraction of cell `ci` of `g` against the plane `n . x = d`, solid below, by
# CartesianMeshes' analytic unit-cube formula: with x = o + U .* y the solid is
# (n .* U) . y <= d - n . o in the unit cube.
function tc_exact_fluid(g, ci, (n, d))
    o = get_node(g, ci)
    U = get_node(g, ci + CartesianIndex(1, 1, 1)) - o
    return 1 - CartesianMeshes.get_volume_fraction(n .* U, d - dot(n, o))
end

# Every cut cell a single box face alone crosses, one of `faces`, against that face's exact plane:
# the cells' fluid fractions and the worst error.
function tc_single_face(cache, g, planes; faces=TC_BOX_FACES)
    topo = cache.work.topo
    fluid = Float64[]
    worst = 0.0
    for ci in CartesianIndices(Tuple(g.n))
        (cache.cells.kind[ci] == CELL_CUT && cache.info.npatch[ci] == 1) || continue
        name = topo.patch_name[only(boundary_faces(cache, ci)).patch]
        name in faces || continue
        exact = tc_exact_fluid(g, ci, planes[findfirst(==(name), TC_BOX_FACES)])
        push!(fluid, exact)
        worst = max(worst, abs(cache.cells.volume_fraction[ci] - exact))
    end
    return fluid, worst
end

tc_count(cache, bits) = count(f -> f & bits != 0x00, cache.info.flags)

# The sum of slot 1's face area vectors: zero for a closed polytope.
function tc_area_sum(scr)
    S = zero(SVector{3,Float64})
    for f in 1:poly_nf(scr, 1)
        S += poly_face_area_vector(scr, 1, f)
    end
    return S
end

# The box `[0, U]` clipped by `planes` in turn, as a cell's cut runs: the clips' flags OR'd, the
# volume, and the area-vector sum.
function tc_clip_all(scr, U, planes, tol)
    poly_box!(scr, 1, U)
    st = 0x00
    for (i, (n, d)) in enumerate(planes)
        st |= poly_clip!(scr, 1, n, d, 6 + i, tol, false)
        poly_isempty(scr, 1) && break
    end
    return st, first(poly_volume_moment(scr, 1, U / 2)), tc_area_sum(scr)
end

# A random plane through the box `[0, U]`, or -- a third of the time each -- a degenerate one: an
# axis or diagonal normal through a box corner or an edge midpoint, so the cut runs exactly through
# vertices, along edges or over a face. Oriented to keep the box's centre, and never through it --
# two opposite planes through the centre would leave a slab of no thickness -- so a stack of them
# leaves something to measure.
function tc_plane(rng, U)
    while true
        n, d = _tc_plane(rng, U)
        c = dot(n, U / 2)
        c != d && return c < d ? (n, d) : (-n, -d)
    end
end

function _tc_plane(rng, U)
    r = rand(rng)
    if r < 1 / 3
        n = normalize(SVector(randn(rng), randn(rng), randn(rng)))
        p = SVector(rand(rng), rand(rng), rand(rng)) .* U
    else
        n = SVector{3,Float64}(rand(rng, (-1, 0, 1), 3))
        n == zero(n) && (n = SVector(1.0, 0.0, 0.0))
        p = SVector{3,Float64}(rand(rng, (0, 1), 3)) .* U
        r < 2 / 3 || (p = p .* SVector{3,Float64}(rand(rng, (0.5, 1), 3)))
    end
    return n, dot(n, p)
end

@testset verbose = true "tri clipping: clippers" begin
    # Dyadic cell size and origin, so lattice nodes and edge midpoints are exact numbers.
    g = CartesianGrid(SVector(-1.0, -1.0, -1.0), (16, 16, 16), SVector(0.125, 0.125, 0.125))
    # ...and a stretched one, whose cells are 0.7 x 1.3 x 0.45 of a cubic cell.
    gs = CartesianGrid(SVector(-1.05, -0.975, -1.0125), (24, 12, 36), SVector(0.0875, 0.1625, 0.05625))

    @testset "single planes against the analytic volume: $tag" for (tag, grid, trials) in
            (("cubic cells", g, 60), ("stretched cells", gs, 40))
        # A box turned to a random orientation and moved a random fraction of a cell, 2 x 2 quads a
        # face: every cell crossed by one face alone is that face's plane clipping the cell.
        rng = Xoshiro(20260928)
        half = SVector(0.42, 0.37, 0.33)
        centre = SVector{3,Float64}(grid.x0) + SVector{3,Float64}(grid.n .* grid.d) / 2
        cache = allocate_cache(grid, TC)
        ncells = 0
        worst = 0.0
        hard = 0
        for _ in 1:trials
            R = tm_frame(normalize(SVector(randn(rng), randn(rng), randn(rng)))) *
                SMatrix{3,3,Float64}(cos(1.0), sin(1.0), 0, -sin(1.0), cos(1.0), 0, 0, 0, 1)
            t = centre + 0.3 * (SVector(rand(rng), rand(rng), rand(rng)) .- 0.5) .* grid.d
            box, _, _ = tm_box(-half, half; n=2, R=R, t=t)
            update_cache!(cache, box, grid)
            fluid, w = tc_single_face(cache, grid, tm_box_planes(-half, half, R, t))
            ncells += length(fluid)
            worst = max(worst, w)
            hard += tc_count(cache, FLAG_OVERFLOW | FLAG_CHAIN_FAIL | FLAG_UNSUPPORTED)
        end
        @test ncells > 10^4
        @test worst < 1e-12
        @test hard == 0
    end

    @testset "degenerate planes: through lattice nodes and edge midpoints" begin
        # A box whose +x face lies on `n . x = s`. The grid's nodes sit at -1 + i/8, so with
        # n = (1,1,1)/sqrt(3) and s = 0 the face passes through lattice nodes, three to a cell --
        # a cut through cell vertices, leaving a corner tetrahedron of exactly 1/6 -- and half a
        # step on, through three edge midpoints, leaving one of exactly 1/48. With n = (1,1,0)/sqrt(2)
        # it runs along cell edges, halving the cells it passes diagonally through.
        s3, s2 = sqrt(3.0), sqrt(2.0)
        half = SVector(0.35, 0.3, 0.3)
        for (nrm, s, want) in ((SVector(1.0, 1.0, 1.0) / s3, 0.0, 5 / 6),
                               (SVector(1.0, 1.0, 1.0) / s3, 0.0625 / s3, 47 / 48),
                               (SVector(1.0, 1.0, 0.0) / s2, 0.0, 1 / 2),
                               (SVector(1.0, 1.0, 0.0) / s2, 0.0625 / s2, nothing))
            R = tm_frame(nrm)
            lo, hi = SVector(s - 2half[1], -half[2], -half[3]), SVector(s, half[2], half[3])
            box, _, _ = tm_box(lo, hi; n=2, R=R)
            cache = update_cache!(allocate_cache(g, TC), box, g)
            fluid, worst = tc_single_face(cache, g, tm_box_planes(lo, hi, R); faces=("+x",))
            @test length(fluid) >= 8
            @test worst < 1e-14
            want === nothing || @test count(f -> isapprox(f, want; atol=1e-14), fluid) > 0
            @test tc_count(cache, FLAG_OVERFLOW | FLAG_CHAIN_FAIL | FLAG_UNSUPPORTED) == 0
        end
    end

    @testset "many planes in one cell: pyramids of 3 to 8 sides" begin
        rng = Xoshiro(7)
        scr = TriScratch(KernelAbstractions.CPU(), Float64, 1)
        cache = allocate_cache(g, TC)
        apex_ok = 0
        sides_seen = Set{Int}()
        worst = 0.0
        mc_worst = 0.0
        ncut = nunsupported = 0
        flagged = true
        hard = 0
        for trial in 1:300
            k = rand(rng, 3:8)
            apex = 0.3 * (SVector(rand(rng), rand(rng), rand(rng)) .- 0.5)
            axis = normalize(SVector(randn(rng), randn(rng), randn(rng)))
            mesh, planes, _ = tm_pyramid(; apex, axis, k, height=0.4 + 0.1rand(rng),
                                         radius=0.25 + 0.1rand(rng), θ0=2π * rand(rng))
            update_cache!(cache, mesh, g)
            hard += tc_count(cache, FLAG_OVERFLOW | FLAG_CHAIN_FAIL)
            # The apex cell: every side at once, by the convex rule.
            ca = CartesianIndex(Tuple(floor.(Int, (apex - g.x0) ./ g.d) .+ 1))
            if cache.info.npatch[ca] == k && cache.info.rule[ca] == RULE_CONVEX
                apex_ok += 1
                push!(sides_seen, k)
            end
            # Every cut cell a rule covers, against the body's own planes; the rest flagged.
            for ci in CartesianIndices(Tuple(g.n))
                cache.cells.kind[ci] == CELL_CUT || continue
                ncut += 1
                unsupported = cache.info.flags[ci] & FLAG_UNSUPPORTED != 0
                flagged &= cache.cells.ambiguous[ci] == unsupported
                if unsupported
                    nunsupported += 1
                else
                    worst = max(worst, abs(cache.cells.volume_fraction[ci] -
                                           (1 - tm_solid_fraction(scr, g, ci, planes))))
                end
            end
            # ...and the apex cell against Monte Carlo, which shares no code with the clipper: five
            # binomial standard deviations.
            if trial <= 50
                o, N = get_node(g, ca), 20_000
                hit = count(1:N) do _
                    x = o + SVector(rand(rng), rand(rng), rand(rng)) .* g.d
                    all(p -> dot(p[1], x) <= p[2], planes)
                end
                solid = 1 - cache.cells.volume_fraction[ca]
                mc_worst = max(mc_worst, abs(hit / N - solid) / (5 * sqrt(max(solid * (1 - solid), 1e-4) / N)))
            end
        end
        @test apex_ok == 300
        @test sides_seen == Set(3:8)
        @test worst < 1e-13
        @test mc_worst < 1
        @test hard == 0
        @test flagged
        @test nunsupported < 1e-3 * ncut
    end

    @testset "the clipper keeps its polytopes closed, and allocates nothing" begin
        rng = Xoshiro(48)
        scr = TriScratch(KernelAbstractions.CPU(), Float64, 1)
        U = SVector(0.0875, 0.1625, 0.05625)   # the stretched grid's cell
        h = minimum(U)
        tol = 1e-10 * h
        worst = 0.0
        hard = 0x00
        bad_volume = false
        for _ in 1:10^4
            planes = [tc_plane(rng, U) for _ in 1:rand(rng, 1:8)]
            st, V, S = tc_clip_all(scr, U, planes, tol)
            hard |= st
            worst = max(worst, maximum(abs, S))
            bad_volume |= !(0 < V <= prod(U) * (1 + 1e-14))
        end
        @test hard == 0x00
        @test worst <= 1e-14 * h^2
        @test !bad_volume
        planes = [tc_plane(rng, U) for _ in 1:8]
        tc_clip_all(scr, U, planes, tol)
        @test (@allocated tc_clip_all(scr, U, planes, tol)) == 0
        @inferred poly_clip!(scr, 1, planes[1][1], planes[1][2], 7, tol, false)
        @inferred poly_volume_moment(scr, 1, U / 2)
    end

    @testset "the update is inferred" begin
        box, _, _ = tm_box(SVector(-0.4, -0.3, -0.35), SVector(0.35, 0.4, 0.3); n=2)
        cache = allocate_cache(g, TC)
        @test (@inferred update_cache!(cache, box, g)) === cache
    end

    if HAS_GPU
        @testset "the eight-sided apex on the device" begin
            mesh, _, _ = tm_pyramid(; apex=SVector(0.013, -0.021, 0.037), axis=normalize(SVector(0.3, -0.5, 0.8)),
                                    k=8, height=0.45, radius=0.3, θ0=0.2)
            g32 = CartesianMeshes.adapt_type(Float32, g)
            host = update_cache!(allocate_cache(g32, TC), mesh, g32)
            dev = update_cache!(allocate_cache(g32, TC; backend=CUDABackend()), mesh, g32)
            @test maximum(host.info.npatch) == 8
            @test Array(dev.cells.kind) == host.cells.kind
            @test Array(dev.info.rule) == host.info.rule && Array(dev.info.npatch) == host.info.npatch
            @test maximum(abs.(Array(dev.cells.volume_fraction) .- host.cells.volume_fraction)) < 10 * eps(Float32)
        end
    end
end
