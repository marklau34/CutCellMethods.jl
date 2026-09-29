# =====================================
# Cut-cell moments over the nodal marching-cubes reconstruction.
#
# The properties under test, in order:
#
#   * the corner order, pinned against `MarchingCubes.jl` itself
#   * the moments against the surface `MarchingCubes.march` draws through the same corner values:
#     every sign pattern, random magnitudes, random fields, and smooth bodies
#   * the construction against a plane, where every moment has a closed form
#   * volume against an independent oracle (`CartesianMeshes.get_volume_fraction`)
#   * bitwise watertightness between neighbours, for fractions AND centroids
#   * convergence on a sphere, and recovery of a body's first moment
#   * ambiguous configurations, and that they are flagged
#
# Everything is read off a marching-cubes cache. A cell with prescribed corner values is a cache over
# a grid of that one cell, updated from a field that returns those values at its nodes
# (`MCNodalField`); a random field is the same over a whole grid.
#
# Three of these are asserted as *bitwise* equalities rather than tolerances: watertightness, the
# closure residual, and a cell's moments depending on its own corner values alone. They hold exactly
# or the reasoning in `marching_cubes/marching_cubes.jl` and `marching_cubes/moments.jl` is wrong
# somewhere, and a tolerance would hide precisely the failure worth catching.
#
# ---------------------------------------------------------------------------
# On what is and is not circular here
#
# `closure_residual` is zero **by construction**: `interface_normal_area` is *defined* as the signed
# sum of the open face areas, so a test that it vanishes is a regression guard on the arithmetic and
# nothing more. It cannot detect a wrong face pairing, a wrong triangulation, or a polyhedron with a
# hole in it.
#
# The test that can is the comparison with the march. `generate_mesh(cache, grid)` is
# `MarchingCubes.march` over the cache's own corner values: that package's implementation of the
# Lewiner tables, placing its own vertices. The moments agree with that surface -- its vector area
# with the one the six face clips imply, and, closed against the cache's open faces, its volume and
# centroids with the ones the cache stores -- only if the facets the construction integrates close
# the apertures it reports, with the same tiling. The march anchors each crossing on the edge's low
# node where the moments anchor on the outside one, so the two agree to an ulp, not bitwise.

using CutCellMethods: CutCellData, MarchingCubesCutCell,
                      cell_nodes, cell_indices, MC_NODE_BITS, MS_NODE_BITS,
                      is_cut, is_inside, is_outside, is_ambiguous,
                      interface_area, interface_normal, interface_normal_area, face_vector_area,
                      closure_residual, face_centroid, face_area_of, full_face_area
using SDFLibrary: sample_sdf

using CartesianMeshes: get_elem_size, direction_axis, direction_sign
import MarchingCubes

# A tiny deterministic PRNG, rather than a dependency on `Random`.
#
# Not for statistical quality -- these tests want *reproducible coverage* of the Lewiner case space,
# and a xorshift gives that. Declaring `Random` as a test dependency instead would re-resolve the
# test environment, which is more disruption than a six-line generator is worth.
const _RNG_STATE = Ref{UInt64}(0x2545f4914f6cdd1d)
_seed!(s::Integer) = (_RNG_STATE[] = UInt64(s) | 0x1; nothing)
function _rand()
    x = _RNG_STATE[]
    x ⊻= x << 13; x ⊻= x >> 7; x ⊻= x << 17
    _RNG_STATE[] = x
    return Float64(x >> 11) * (1.0 / 9007199254740992.0)   # 53 bits, in [0,1)
end
"""A standard normal by Box-Muller, off `_rand`."""
_randn() = sqrt(-2 * log(1 - _rand())) * cospi(2 * _rand())

const H1 = SVector(1.0, 1.0, 1.0)
"""The grid of one unit cube at the origin: its corners are `MC_NODE_BITS` exactly."""
const MC_CUBE = CartesianGrid(SVector(0.0, 0.0, 0.0), (1, 1, 1), H1)
const MC_CI = CartesianIndex(1, 1, 1)

"""A field that is `vals` at the nodes of the grid with origin `x0` and cell size `d`: what a cache
samples, so it stores `vals` as its `phi` -- any corner values at all, through `update_cache!`."""
struct MCNodalField{A}
    vals::A
    x0::SVector{3,Float64}
    d::SVector{3,Float64}
end
(f::MCNodalField)(x) = @inbounds f.vals[round(Int, (x[1] - f.x0[1]) / f.d[1]) + 1,
                                        round(Int, (x[2] - f.x0[2]) / f.d[2]) + 1,
                                        round(Int, (x[3] - f.x0[3]) / f.d[3]) + 1]
MCNodalField(vals, g::CartesianGrid{3}) = MCNodalField(vals, SVector{3,Float64}(g.x0), SVector{3,Float64}(g.d))

"""The plane `n . x = d`, as a level set."""
struct MCPlane
    n::SVector{3,Float64}
    d::Float64
end
(p::MCPlane)(x) = dot(p.n, x) - p.d

"""`cache`, over the one cell of `g`, updated to the corner values `phi` in `MC_NODE_BITS` order."""
function mc_cube!(cache, g::CartesianGrid{3}, phi::SVector{8})
    vals = Array{Float64,3}(undef, 2, 2, 2)
    for k in 1:8
        b = MC_NODE_BITS[k]
        vals[1 + b[1], 1 + b[2], 1 + b[3]] = phi[k]
    end
    return update_cache!(cache, MCNodalField(vals, g), g)
end

"""A marching-cubes cache over `g`, updated from `geo`: every cell's moments in `cells`, and the
nodal field they were built from in `phi`."""
mc_cache(geo, g) = update_cache!(allocate_cache(g, MarchingCubesCutCell()), geo, g)

"""One cell's corner values out of a nodal block, in `MC_NODE_BITS` order."""
mc_corners(vals, ci) = SVector{8,Float64}(ntuple(k -> vals[(Tuple(ci) .+ MC_NODE_BITS[k])...], Val(8)))

"""What the triangles `generate_mesh` marches through a one-cell cache imply for that cell's
moments: the interface's vector area (out of the body) and centroid, and -- closing the surface
against the cache's own open faces by the divergence theorem -- the outside volume fraction and
centroid. Accumulated in the cell's own coordinates, as the construction is."""
function mc_marched(cache, g::CartesianGrid{3})
    m = only(cache.cells)
    o, h = SVector{3,Float64}(g.x0), SVector{3,Float64}(g.d)
    surf = generate_mesh(cache, g; warn=false)
    x = surf.nodes.coord
    S = zero(SVector{3,Float64})
    A = 0.0
    Ac = zero(SVector{3,Float64})
    vol3 = 0.0
    cm2 = zero(SVector{3,Float64})
    # The open faces: `x . n` and `x_a^2` are constant on a face, and zero on the three at the
    # cell's own origin.
    for ax in 1:3
        a_open = face_area_of(m, 2ax, h)
        vol3 += h[ax] * a_open
        cm2 += SVector{3,Float64}(ntuple(a -> a == ax ? h[ax]^2 * a_open : 0.0, Val(3)))
    end
    # The marched triangles, whose normal points out of the body and so into the outside region.
    for t in surf.elements
        a, b, c = x[t.con[1]] - o, x[t.con[2]] - o, x[t.con[3]] - o
        w = 0.5 * cross(b - a, c - a)
        xc = (a + b + c) / 3
        S += w
        len = norm(w)
        A += len
        Ac += len * xc
        vol3 -= dot(w, xc)
        cm2 -= w .* ((a .* a + b .* b + c .* c + a .* b + b .* c + c .* a) / 6)
    end
    V = vol3 / 3
    return (vector_area=S, interface_centroid=o + Ac / A, volume_fraction=V / prod(h),
            centroid=o + cm2 / (2V))
end

"""The worst disagreement between one-cell `cache`'s moments and the surface marched through it."""
function mc_march_errors(cache, g::CartesianGrid{3})
    m = only(cache.cells)
    r = mc_marched(cache, g)
    return (area=maximum(abs, r.vector_area - interface_normal_area(m, SVector{3,Float64}(g.d))),
            volume=abs(r.volume_fraction - m.volume_fraction),
            centroid=norm(r.centroid - m.centroid),
            interface_centroid=norm(r.interface_centroid - m.interface_centroid))
end

"""Every cut cell of `cache` over `g` against the triangles `generate_mesh` marches through it, as
`(ncut, remarched, worst)`: the worst disagreement in the interface's vector area. A triangle lying
flush in a face two cells share cannot be told which of them it came from (`cell_indices` floors its
centroid), so a cell whose bucket disagrees is marched again on a grid of that one cell, where there
is no neighbour to confuse it with, and checked in full."""
function mc_march_disagreement(cache, g::CartesianGrid{3})
    surf = generate_mesh(cache, g; warn=false)
    owner = cell_indices(surf, g)
    x = surf.nodes.coord
    S = fill(zero(SVector{3,Float64}), Tuple(g.n))
    for (t, e) in enumerate(surf.elements)
        a, b, c = x[e.con[1]], x[e.con[2]], x[e.con[3]]
        S[owner[t]] += 0.5 * cross(b - a, c - a)
    end
    h = SVector{3,Float64}(g.d)
    one = allocate_cache(CartesianGrid(g.x0, (1, 1, 1), g.d), MarchingCubesCutCell())
    ncut = remarched = 0
    worst = 0.0
    for ci in CartesianIndices(Tuple(g.n))
        m = cache.cells[ci]
        is_cut(m) || continue
        ncut += 1
        err = maximum(abs, S[ci] - interface_normal_area(m, h))
        if err > 1e-13
            remarched += 1
            g1 = CartesianGrid(get_node(g, ci), (1, 1, 1), g.d)
            e1 = mc_march_errors(mc_cube!(one, g1, mc_corners(cache.phi, ci)), g1)
            err = max(e1.area, e1.volume)
        end
        worst = max(worst, err)
    end
    return ncut, remarched, worst
end

"""The Lewiner case of a cube with corner values `phi` in `MC_NODE_BITS` order, as
`MarchingCubes.lut_entry` indexes its tables: bit `p` set when corner `p + 1` is outside."""
mc_case(phi) = Int(MarchingCubes.cases[1 + sum(p -> Int(phi[p + 1] >= 0) << p, 0:7)][1])

grid3(N; lo = -1.5, span = 3.0) =
    CartesianGrid(SVector(lo, lo, lo), (N, N, N), SVector(span/N, span/N, span/N))

"""A wiggly analytic level set. Not a distance field, deliberately -- the construction reads corner
values only, and this one produces saddle faces and ambiguous cubes that a smooth convex body
never does."""
wiggle(x) = sin(2.7x[1]) * sin(3.1x[2]) * sin(2.3x[3]) - 0.12

const SPH = SDFSphere(SVector(0.0, 0.0, 0.0), 1.0)

@testset "marching cubes cut-cell moments" begin

    # =====================================================================
    @testset "the corner order, pinned against MarchingCubes.jl" begin
        # `MC_NODE_BITS` has to be the order `lut_entry` classifies corners in, or every ambiguous
        # cube selects the wrong tiling -- and it is the order the corner values below are handed
        # over in. Re-derived here from that function's own index arithmetic rather than copied
        # from the constant under test.
        for p in 0:7
            @test MC_NODE_BITS[p+1] == ((p ⊻ (p >> 1)) & 1, (p >> 1) & 1, (p >> 2) & 1)
        end
        # ...and the bottom four are the 2D convention, unpermuted.
        @test MC_NODE_BITS[1:4] == ntuple(k -> (MS_NODE_BITS[k]..., 0), 4)
    end

    # =====================================================================
    @testset "every sign pattern agrees with MarchingCubes.march" begin
        # All 256 sign patterns, four random magnitude draws each: the magnitudes are what select
        # among the sub-configurations of an ambiguous case, so signs alone would leave most of the
        # Lewiner tables untouched. Each cube is a cache of its own, marched on its own.
        _seed!(20260919)
        cache = allocate_cache(MC_CUBE, MarchingCubesCutCell())
        worst = (area=0.0, volume=0.0, centroid=0.0, interface_centroid=0.0)
        ncut = 0
        cases_seen = Set{Int}()
        for mask in 0:255, _ in 1:4
            phi = SVector{8,Float64}(ntuple(
                k -> ((mask >> (k-1)) & 1 == 1 ? 1 : -1) * (0.05 + _rand()), Val(8)))
            mc_cube!(cache, MC_CUBE, phi)
            is_cut(only(cache.cells)) || continue
            ncut += 1
            push!(cases_seen, mc_case(phi))
            e = mc_march_errors(cache, MC_CUBE)
            worst = map(max, worst, e)
        end
        @test ncut == 254 * 4
        @test length(cases_seen) == 14
        @test worst.area < 1e-14
        @test worst.volume < 1e-14
        @test worst.centroid < 1e-12
        @test worst.interface_centroid < 1e-12
    end

    # =====================================================================
    @testset "a plane, where every moment is known in closed form" begin
        # phi = x - 0.3: the outside region is the slab x >= 0.3.
        m = only(mc_cache(x -> x[1] - 0.3, MC_CUBE).cells)
        @test is_cut(m)
        @test m.volume_fraction ≈ 0.7 atol = 1e-15
        @test m.centroid ≈ SVector(0.65, 0.5, 0.5) atol = 1e-15
        @test interface_area(m, H1) ≈ 1.0 atol = 1e-15
        @test interface_normal(m, H1) ≈ SVector(1.0, 0.0, 0.0) atol = 1e-15
        @test m.interface_centroid ≈ SVector(0.3, 0.5, 0.5) atol = 1e-15
        # -x face wholly inside the body, +x face wholly open, the four side faces cut at 0.7.
        @test m.face_fraction ≈ SVector(0.0, 1.0, 0.7, 0.7, 0.7, 0.7) atol = 1e-15
        # A fully open face's centroid is a real quadrature point, not a placeholder.
        @test face_centroid(m, 2, cell_nodes(MC_CUBE, MC_CI)) ≈ SVector(1.0, 0.5, 0.5) atol = 1e-15
    end

    # =====================================================================
    @testset "volume against an independent oracle, over random planes" begin
        # `CartesianMeshes.get_volume_fraction` clips a unit cell against a plane analytically --
        # a different algorithm entirely, and exact. The nodal reconstruction is *also* exact for a
        # plane (linear interpolation hits the crossing dead on), so these must agree to roundoff
        # rather than merely converge.
        _seed!(4242)
        cache = allocate_cache(MC_CUBE, MarchingCubesCutCell())
        worst_vol = 0.0
        worst_nrm = 0.0
        worst_clo = 0.0
        ntested = 0
        for _ in 1:1500
            nv = normalize(SVector{3,Float64}(_randn(), _randn(), _randn()))
            d = 0.05 + 0.9 * _rand()
            m = only(update_cache!(cache, MCPlane(nv, d), MC_CUBE).cells)
            is_cut(m) || continue
            ntested += 1
            worst_vol = max(worst_vol,
                            abs(m.volume_fraction - (1 - CartesianMeshes.get_volume_fraction(nv, d))))
            # The non-circular normal check: the closure-derived normal against the plane's own.
            worst_nrm = max(worst_nrm, norm(interface_normal(m, H1) - nv))
            worst_clo = max(worst_clo, maximum(abs.(closure_residual(m, H1))))
        end
        @test ntested > 900        # a sample-size floor, not a property of the construction
        @test worst_vol < 1e-14
        @test worst_nrm < 1e-12
        @test worst_clo == 0.0          # bitwise, at every cut
    end

    # =====================================================================
    @testset "the polyhedron closes on smooth bodies -- the non-circular test" begin
        # `interface_normal_area` comes from the six face clips; the marched triangles through each
        # cell come from `MarchingCubes.jl`'s own tables. They agree only if the facets the moments
        # integrate really do close the apertures. See this file's header.
        for (tag, geo, N) in (("sphere", SPH, 24),
                              ("wiggly", wiggle, 24),
                              ("slotted", SDFSlottedSphere(SVector(0.0,0.0,0.0), 1.0, 0.3, 0.8), 32))
            g = grid3(N)
            h = SVector(g.d...)
            cache = mc_cache(geo, g)
            ncut, _, worst = mc_march_disagreement(cache, g)
            worst_c = maximum(m -> maximum(abs, closure_residual(m, h)), cache.cells)
            nbadfrac = count(m -> !(-1e-12 <= m.volume_fraction <= 1 + 1e-12), cache.cells)
            @testset "$tag" begin
                @test ncut > 100
                @test worst < 1e-14
                @test worst_c == 0.0
                @test nbadfrac == 0
            end
        end
    end

    # =====================================================================
    @testset "watertightness -- neighbours agree bitwise" begin
        # The claim everything else rests on: the two cells touching a face compute the same
        # aperture AND the same centroid for it, bit for bit, with no communication. Asserted with
        # `===` rather than `≈` deliberately -- see this file's header.
        for (tag, geo) in ("sphere" => SPH, "wiggly" => wiggle)
            g = grid3(24)
            M = mc_cache(geo, g).cells
            badf = 0
            badc = 0
            nchecked = 0
            for ax in 1:3
                e = CartesianIndex(ntuple(a -> a == ax ? 1 : 0, 3))
                rng = ntuple(a -> a == ax ? (1:g.n[a]-1) : (1:g.n[a]), 3)
                for idx in CartesianIndices(rng)
                    A, B = M[idx], M[idx + e]
                    nchecked += 1
                    A.face_fraction[2ax] === B.face_fraction[2ax-1] || (badf += 1)
                    face_centroid(A, 2ax, cell_nodes(g, idx)) ===
                        face_centroid(B, 2ax-1, cell_nodes(g, idx + e)) || (badc += 1)
                end
            end
            @testset "$tag" begin
                @test nchecked > 10_000
                @test badf == 0
                @test badc == 0
            end
        end
    end

    # =====================================================================
    @testset "a sphere: volume converges, and the first moment is recovered" begin
        errs = Float64[]
        for N in (20, 40, 80)
            g = grid3(N)
            M = mc_cache(SPH, g).cells
            outside = sum(m.volume_fraction for m in M) * prod(SVector(g.d...))
            push!(errs, (27.0 - outside - 4pi/3) / (4pi/3))
        end
        # Second order, and systematically *under* -- every facet is a chord of the true sphere, so
        # the reconstruction is inscribed in a convex body. This leans the opposite way to
        # `calc_volume`, whose plane fit contains it.
        @test all(e -> e < 0, errs)
        @test abs(errs[2]) < abs(errs[1]) / 3.5
        @test abs(errs[3]) < abs(errs[2]) / 3.5

        # The centroids, globally: a symmetric grid has zero first moment, so the body's first
        # moment is minus the outside region's, and its centroid must come back as the sphere's.
        # This is the one check that exercises the `x^2` term of every cut cell at once.
        c0 = SVector(0.13, -0.21, 0.07)
        for (N, tol) in ((30, 1e-4), (60, 1e-6))
            g = grid3(N)
            M = mc_cache(SDFSphere(c0, 1.0), g).cells
            vc = prod(SVector(g.d...))
            outv = 0.0
            outmom = zero(SVector{3,Float64})
            for m in M
                v = m.volume_fraction * vc
                outv += v
                outmom += v * m.centroid
            end
            @test norm(-outmom / (27.0 - outv) - c0) < tol
        end
    end

    # =====================================================================
    @testset "uncut cells are exactly uncut" begin
        allout = only(mc_cache(x -> 1.0, MC_CUBE).cells)
        allin = only(mc_cache(x -> -1.0, MC_CUBE).cells)
        @test is_outside(allout) && !is_cut(allout) && !is_ambiguous(allout)
        @test is_inside(allin) && !is_cut(allin) && !is_ambiguous(allin)
        # Exactly, not to roundoff: an `O(eps)` aperture error in every uncut cell is a free-stream
        # error over most of the domain.
        @test allout.volume_fraction === 1.0
        @test allin.volume_fraction === 0.0
        @test all(allout.face_fraction .=== 1.0)
        @test all(allin.face_fraction .=== 0.0)
        @test all(interface_normal_area(allout, H1) .=== 0.0)
        @test interface_normal(allout, H1) === zero(SVector{3,Float64})
        @test closure_residual(allout, H1) === zero(SVector{3,Float64})
        @test closure_residual(allin, H1) === zero(SVector{3,Float64})
        # A fully open face of an uncut cell carries full weight, so its centroid is real.
        @test face_centroid(allout, 2, cell_nodes(MC_CUBE, MC_CI)) ≈ SVector(1.0, 0.5, 0.5) atol = 1e-15
        @test face_area_of(allout, 2, H1) === 1.0
    end

    # =====================================================================
    @testset "ambiguous configurations are resolved and flagged" begin
        cache = allocate_cache(MC_CUBE, MarchingCubesCutCell())
        cube(phi) = only(mc_cube!(cache, MC_CUBE, SVector{8,Float64}(phi)).cells)
        # An alternating z = 0 face: the saddle, and the only face pairing that is a choice.
        saddle = cube((1, -1, 1, -1, -1, -1, -1, -1))
        @test is_cut(saddle) && is_ambiguous(saddle)
        # The body-diagonal alternation, the classic ambiguous cube.
        diag = cube((1, -1, 1, -1, -1, 1, -1, 1))
        @test is_cut(diag) && is_ambiguous(diag)
        # Both still close, which is the point: the pairing is a guess, the conservation is not.
        for m in (saddle, diag)
            @test closure_residual(m, H1) == zero(SVector{3,Float64})
            @test 0 < m.volume_fraction < 1
        end
        # A plane cuts no face ambiguously, so nothing there is flagged.
        @test !is_ambiguous(only(mc_cache(MCPlane(normalize(SVector(1.0, 2.0, 3.0)), 0.5), MC_CUBE).cells))
    end

    # =====================================================================
    @testset "a random field: every case, closed, single-valued, and the march's" begin
        # The strongest form of the non-circular test, and the one that earned its place: random
        # corner values reach Lewiner cases that no smooth body produces at a sane resolution.
        #
        # It caught a real defect. An earlier face clip read its chord pairing off the facets, which
        # is true but not sufficient -- in case 10.1.2 four of the eight facets lie entirely inside
        # a cube face plane, so all three edges of each registered as chords of that face and
        # overwrote each other. About 0.015% of random cut cubes, and where it bit, the volume was
        # wrong by a whole corner triangle. Smooth-field tests never saw it.
        _seed!(31337)
        n = 35
        g = CartesianGrid(SVector(0.0, 0.0, 0.0), (n, n, n), H1)
        vals = [_randn() for _ in 1:(n + 1), _ in 1:(n + 1), _ in 1:(n + 1)]
        cache = update_cache!(allocate_cache(g, MarchingCubesCutCell()), MCNodalField(vals, g), g)
        @test cache.phi == vals
        idx = CartesianIndices(Tuple(g.n))
        M = cache.cells

        # Aggregated rather than one assertion per cell: forty thousand passing `@test`s bury the
        # rest of the suite, and a count is what a failure here actually wants to report.
        @test count(is_cut, M) > 40_000
        @test count(m -> closure_residual(m, H1) != zero(SVector{3,Float64}), M) == 0
        @test count(m -> !(-1e-12 <= m.volume_fraction <= 1 + 1e-12), M) == 0
        # Every non-empty Lewiner case, so the ambiguous families are genuinely exercised.
        @test length(Set(mc_case(mc_corners(vals, ci)) for ci in idx if is_cut(M[ci]))) == 14

        # Every face two cells share is single-valued, saddle faces included. The saddle's pairing
        # is resolved by a decider each cell runs on the face's four corner values alone, under a
        # different face code from each side; a disagreement would show here as a mismatch.
        nsaddle = badf = badc = 0
        for ax in 1:3
            e = CartesianIndex(ntuple(a -> a == ax ? 1 : 0, 3))
            for ci in CartesianIndices(ntuple(a -> a == ax ? (1:n - 1) : (1:n), 3))
                A, B = M[ci], M[ci + e]
                A.face_fraction[2ax] === B.face_fraction[2ax - 1] || (badf += 1)
                A.face_centroid_local[2ax] === B.face_centroid_local[2ax - 1] || (badc += 1)
                # the shared face's four corners, round the face
                p, q = Tuple(a for a in 1:3 if a != ax)
                s = [vals[(Tuple(ci + e) .+ ntuple(a -> a == p ? u : a == q ? v : 0, 3))...] >= 0
                     for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1))]
                (s[1] == s[3] && s[2] == s[4] && s[1] != s[2]) && (nsaddle += 1)
            end
        end
        @test nsaddle > 1000
        @test badf == 0
        @test badc == 0

        # Every cut cell against the march, and the cells whose facets lie flush in a shared face
        # -- 10.1.2 among them -- re-marched on their own.
        ncut, remarched, worst = mc_march_disagreement(cache, g)
        @test ncut == count(is_cut, M)
        @test remarched > 0
        @test worst < 1e-14

        # A cell's moments are its own eight corner values', and nothing else: the same values on a
        # cube of their own give bitwise the same fractions. (Centroids are global, so they move.)
        one = allocate_cache(MC_CUBE, MarchingCubesCutCell())
        @test all(ci for ci in idx[1:37:end] if is_cut(M[ci])) do ci
            m1 = only(mc_cube!(one, MC_CUBE, mc_corners(vals, ci)).cells)
            m1.volume_fraction === M[ci].volume_fraction && m1.face_fraction === M[ci].face_fraction &&
                m1.face_centroid_local === M[ci].face_centroid_local && m1.ambiguous === M[ci].ambiguous
        end

        # The exact cube that exposed the defect: case 10.1.2, with two facets flush against each
        # of the two ambiguous x-faces. Pinned so the failure cannot come back silently.
        flush_cube = SVector{8,Float64}(-0.6727807769355654, -0.23277645053557436,
                                        1.0026954720119468, 0.22568989049603774,
                                        1.8172940934743478, 0.02419684567652038,
                                        -0.7057160958113278, -2.13709186873607)
        @test mc_case(flush_cube) == 11
        mc_cube!(one, MC_CUBE, flush_cube)
        e = mc_march_errors(one, MC_CUBE)
        @test e.area < 1e-14 && e.volume < 1e-14
        @test closure_residual(only(one.cells), H1) == zero(SVector{3,Float64})
    end

    # =====================================================================
    @testset "stretched cells, and the dimensional readers" begin
        # Marching cubes converts no distance into unit-cell coordinates, so unlike `calc_volume`
        # it carries no isotropy requirement -- but everything above this point used cubic cells,
        # which is exactly where an axis mix-up hides. `full_face_area`'s `D`-generic method is the
        # piece under test: a single `grid.d[1]^(D-1)` would be wrong here and right everywhere else.
        h = SVector(2.0, 0.5, 1.5)
        box = CartesianGrid(SVector(0.0, 0.0, 0.0), (1, 1, 1), h)
        cache = allocate_cache(box, MarchingCubesCutCell())
        # phi = x - 0.5 over a 2.0 x 0.5 x 1.5 box: the outside region is the slab x >= 0.5.
        m = only(update_cache!(cache, x -> x[1] - 0.5, box).cells)
        @test m.volume_fraction ≈ 0.75 atol = 1e-15
        @test m.centroid ≈ SVector(1.25, 0.25, 0.75) atol = 1e-15
        @test m.face_fraction ≈ SVector(0.0, 1.0, 0.75, 0.75, 0.75, 0.75) atol = 1e-15

        # The full measure of each face is the product over the *other* two axes.
        @test full_face_area(h, 1) == full_face_area(h, 2) == 0.5 * 1.5
        @test full_face_area(h, 3) == full_face_area(h, 4) == 2.0 * 1.5
        @test full_face_area(h, 5) == full_face_area(h, 6) == 2.0 * 0.5

        @test face_area_of(m, 2, h) ≈ 0.75 atol = 1e-15          # 1.0 * (0.5 * 1.5)
        @test face_area_of(m, 4, h) ≈ 0.75 * 3.0 atol = 1e-15
        @test face_vector_area(m, 2, h) ≈ SVector(0.75, 0.0, 0.0) atol = 1e-15
        @test face_vector_area(m, 3, h) ≈ SVector(0.0, -2.25, 0.0) atol = 1e-15
        # The cut plane really is 0.5 x 1.5, and the closure still lands on zero exactly.
        @test interface_area(m, h) ≈ 0.75 atol = 1e-15
        @test interface_normal(m, h) ≈ SVector(1.0, 0.0, 0.0) atol = 1e-15
        @test closure_residual(m, h) == zero(SVector{3,Float64})

        # Tilted planes on the stretched box, against the same analytic oracle. Substituting
        # x = h .* xi maps the box onto the unit cube and the half-space onto another half-space,
        # so the volume *fraction* is the oracle's with the normal scaled by `h`.
        _seed!(3141)
        worst_vol = 0.0
        worst_clo = 0.0
        worst_close = 0.0
        ntested = 0
        for _ in 1:800
            nv = normalize(SVector{3,Float64}(_randn(), _randn(), _randn()))
            d = dot(nv, h) * (0.1 + 0.8 * _rand())
            mm = only(update_cache!(cache, MCPlane(nv, d), box).cells)
            is_cut(mm) || continue
            ntested += 1
            scaled = nv .* h
            ns = norm(scaled)
            worst_vol = max(worst_vol,
                            abs(mm.volume_fraction -
                                (1 - CartesianMeshes.get_volume_fraction(scaled / ns, d / ns))))
            worst_clo = max(worst_clo, maximum(abs.(closure_residual(mm, h))))
            worst_close = max(worst_close, mc_march_errors(cache, box).area)
        end
        @test ntested > 400
        @test worst_vol < 1e-14
        @test worst_clo == 0.0
        @test worst_close < 1e-14
    end

    # =====================================================================
    @testset "element type" begin
        # Float32 all the way through, with no widening.
        c32 = CartesianGrid(SVector(0f0, 0f0, 0f0), (1, 1, 1), SVector(1f0, 1f0, 1f0))
        m32 = only(mc_cache(x -> x[1] - 0.3f0, c32).cells)
        @test m32 isa CutCellData{3,Float32,6}
        @test eltype(m32) === Float32
        @test m32.volume_fraction ≈ 0.7f0 atol = 1f-6

        g = CartesianMeshes.adapt_type(Float32, grid3(24))
        cache = mc_cache(SDFLibrary.adapt_type(Float32, SPH), g)
        @test eltype(cache.cells) === CutCellData{3,Float32,6,2}
        @test count(is_cut, cache.cells) > 1000
        @test all(m -> closure_residual(m, g.d) == zero(SVector{3,Float32}), cache.cells)
    end

    # =====================================================================
    if HAS_GPU
        @testset "GPU: the cache's kernel compiles and matches the host" begin
            # Regression guard for a real bug (fixed 2026-09-21): `cut_cell_faces` closed over `c2`,
            # a variable reassigned across an if/else branch inside the `for dir in 1:6` loop.
            # Julia cannot prove a captured, branch-reassigned variable's type stable across
            # iterations and boxes it (`Core.Box`), which turns every read inside the closure into a
            # dynamic dispatch -- a hard GPU-compilation failure ("unsupported dynamic function
            # invocation"), not merely a slow path. The Lewiner table dispatch compiled and ran on
            # the GPU throughout, so a test that only exercised that half would have missed this
            # entirely. A device cache runs the whole per-cell construction, which is what HYBIS's
            # `update_cut_cells!` reaches.
            g32 = CartesianMeshes.adapt_type(Float32, grid3(16))
            sph32 = SDFLibrary.adapt_type(Float32, SPH)
            host = mc_cache(sph32, g32)
            dev = update_cache!(allocate_cache(g32, MarchingCubesCutCell(); backend=CUDABackend()),
                                sph32, g32)
            dcells = Adapt.adapt(Array, dev.cells)
            @test dcells.kind == host.cells.kind
            # A few ULPs of GPU/CPU float non-associativity is expected (FMA fusion, evaluation
            # order); anything beyond that would be a real discrepancy, not noise.
            @test maximum(abs.(dcells.volume_fraction .- host.cells.volume_fraction)) < 100 * eps(Float32)
            @test all(m -> closure_residual(m, g32.d) == zero(SVector{3,Float32}), dcells)
        end
    end
end
