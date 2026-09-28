# =====================================
# Cut-cell moments over the nodal marching-cubes reconstruction.
#
# The properties under test, in order:
#
#   * the corner, edge and face conventions, pinned against `MarchingCubes.jl` itself
#   * the per-cell Lewiner dispatch against that package's own whole-grid `march`
#   * the construction against a plane, where every moment has a closed form
#   * volume against an independent oracle (`CartesianMeshes.get_volume_fraction`)
#   * that the cell's polyhedron actually closes -- the non-circular test
#   * bitwise watertightness between neighbours, for fractions AND centroids
#   * convergence on a sphere, and recovery of a body's first moment
#   * ambiguous configurations, and that they are flagged
#
# Three of these are asserted as *bitwise* equalities rather than tolerances: watertightness, the
# closure residual, and threaded-equals-serial. They hold exactly or the reasoning in
# `marching_cubes/marching_cubes.jl` and `marching_cubes/moments.jl` is wrong somewhere, and a
# tolerance would hide precisely the failure worth catching.
#
# ---------------------------------------------------------------------------
# On what is and is not circular here
#
# `closure_residual` is zero **by construction**: `interface_normal_area` is *defined* as the signed
# sum of the open face areas, so a test that it vanishes is a regression guard on the arithmetic and
# nothing more. It cannot detect a wrong face pairing, a wrong triangulation, or a polyhedron with a
# hole in it.
#
# The test that can is "the polyhedron closes" below: it sums the interface triangles' own vector
# areas and compares against that face-derived quantity. Those two are computed from different
# things -- one from six 2D clips, the other from the Lewiner tables -- and they agree only if the
# facets really do close the apertures. That is the assertion the volume rests on.

using CutCellMethods: cut_cell_moments, cut_cell_faces, CutCellData,
                      cell_nodes, cell_values, cell_interface, cell_case, cell_triangles,
                      edge_crossings, nudge_zeros, face_outside_connected,
                      MC_NODE_BITS, MC_EDGE_NODES, MC_FACE_NODES,
                      MC_FACE_TEST_CODE,
                      is_cut, is_inside, is_outside, is_ambiguous,
                      interface_area, interface_normal, interface_normal_area, face_vector_area,
                      closure_residual, volume_rule, face_rule, face_centroid, interface_rule,
                      measure,
                      face_area_of, full_face_area, MS_NODE_BITS,
                      MarchingCubesCutCell
using SDFLibrary: sample_sdf

using CartesianMeshes: get_elem_size, direction_axis, direction_sign
using KernelAbstractions
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
const UNIT_NODES = SVector{8,SVector{3,Float64}}(
    ntuple(k -> SVector{3,Float64}(MC_NODE_BITS[k]...), Val(8)))

"""The eight corner values of a plane `n . x = d`, on the unit cube."""
plane_phi(n, d) = SVector{8,Float64}(ntuple(k -> dot(n, UNIT_NODES[k]) - d, Val(8)))

"""Sum of the interface triangles' vector areas, `sum_t A_t n_t`, with `n_t` out of the body.

Computed from the facets themselves, so comparing it against `interface_normal_area` -- which comes
from the six face clips -- is the test that the cell's polyhedron closes."""
function facet_vector_area(nodes, phi)
    n, a, b, c = cell_interface(nodes, phi)
    w = zero(SVector{3,Float64})
    for t in 1:n
        w += 0.5 * cross(b[t] - a[t], c[t] - a[t])
    end
    return w
end

"""One cube marched by `MarchingCubes.jl` itself, as a list of vertex triples in unit-cube
coordinates -- the reference the per-cell dispatch is pinned against."""
function march_one_cube(phi::SVector{8,Float64})
    vol = Array{Float64,3}(undef, 2, 2, 2)
    for k in 1:8
        b = MC_NODE_BITS[k]
        vol[1+b[1], 1+b[2], 1+b[3]] = phi[k]
    end
    mc = MarchingCubes.MC(vol; x = [0.0, 1.0], y = [0.0, 1.0], z = [0.0, 1.0])
    fill!(mc.vert_indices, 0)
    MarchingCubes.march(mc, 0.0)
    return [(SVector{3,Float64}(mc.vertices[t[1]]), SVector{3,Float64}(mc.vertices[t[2]]),
             SVector{3,Float64}(mc.vertices[t[3]])) for t in mc.triangles]
end

"""A rotation-invariant but winding-*sensitive* key for a triangle, so two triangulations compare
as sets without either losing the orientation that makes the normals point outwards."""
function tri_key(t; digits = 9)
    r(v) = round.(v; digits = digits)
    a, b, c = r(t[1]), r(t[2]), r(t[3])
    return sort([(a, b, c), (b, c, a), (c, a, b)], by = string)[1]
end

grid3(N; lo = -1.5, span = 3.0) =
    CartesianGrid(SVector(lo, lo, lo), (N, N, N), SVector(span/N, span/N, span/N))

"""A wiggly analytic level set. Not a distance field, deliberately -- the construction reads corner
values only, and this one produces saddle faces and ambiguous cubes that a smooth convex body
never does."""
wiggle(x) = sin(2.7x[1]) * sin(3.1x[2]) * sin(2.3x[3]) - 0.12

const SPH = SDFSphere(SVector(0.0, 0.0, 0.0), 1.0)

# One thread per cell, over the same tagged per-cell entry point a real consumer (HYBIS's
# `update_cut_cells!`) calls -- not a re-implementation, so a GPU-only failure in
# `cut_cell_moments`/`cut_cell_faces` shows up here rather than only downstream. Kept minimal
# (two scalars) rather than returning the whole `CutCellData`, since that struct is not itself
# under test here -- `kind`/`volume_fraction` are enough to catch a wrong or non-compiling
# dispatch.
@kernel function _mc_gpu_regression_kernel!(vf, kind, vals, grid, method)
    ci = @index(Global, Cartesian)
    @inbounds begin
        m = cut_cell_moments(method, grid, vals, ci)
        vf[ci] = m.volume_fraction
        kind[ci] = m.kind
    end
end

@testset "marching cubes cut-cell moments" begin

    # =====================================================================
    @testset "the conventions, pinned against MarchingCubes.jl" begin
        # `MC_NODE_BITS` has to be the order `lut_entry` classifies corners in, or every ambiguous
        # cube selects the wrong tiling. Re-derived here from that function's own index arithmetic
        # rather than copied from the constant under test.
        for p in 0:7
            @test MC_NODE_BITS[p+1] == ((p ⊻ (p >> 1)) & 1, (p >> 1) & 1, (p >> 2) & 1)
        end
        # ...and the bottom four are the 2D convention, unpermuted.
        @test MC_NODE_BITS[1:4] == ntuple(k -> (MS_NODE_BITS[k]..., 0), 4)

        # Every edge joins two corners differing in exactly one bit.
        for (a, b) in MC_EDGE_NODES
            @test count(MC_NODE_BITS[a] .!= MC_NODE_BITS[b]) == 1
        end

        # Choice 3: a face's four corners share the normal axis' bit, and their two in-plane bits
        # are `MS_NODE_BITS` in order. Asserted as a property, not against a transcribed table.
        for dir in 1:6
            ax = direction_axis(dir)
            b_ax = direction_sign(dir) > 0 ? 1 : 0
            p, q = Tuple(a for a in 1:3 if a != ax)
            for k in 1:4
                bits = MC_NODE_BITS[MC_FACE_NODES[dir][k]]
                @test bits[ax] == b_ax
                @test (bits[p], bits[q]) == MS_NODE_BITS[k]
            end
        end

    end

    # =====================================================================
    @testset "the per-cell dispatch reproduces MarchingCubes.march" begin
        # All 256 sign patterns, four random magnitude draws each: the magnitudes are what select
        # among the sub-configurations of an ambiguous case, so signs alone would leave most of the
        # Lewiner tables untouched.
        _seed!(20260919)
        nbad_count = 0
        nbad_geom = 0
        for mask in 0:255, _ in 1:4
            phi = SVector{8,Float64}(ntuple(
                k -> ((mask >> (k-1)) & 1 == 1 ? 1 : -1) * (0.05 + _rand()), Val(8)))
            ref = march_one_cube(phi)
            n, a, b, c = cell_interface(UNIT_NODES, phi)
            mine = [(a[t], b[t], c[t]) for t in 1:n]
            length(ref) == length(mine) || (nbad_count += 1; continue)
            sort(tri_key.(ref), by = string) == sort(tri_key.(mine), by = string) ||
                (nbad_geom += 1)
        end
        @test nbad_count == 0
        @test nbad_geom == 0

        # The case index itself, against `lut_entry`'s own bit packing.
        for mask in 0:255
            phi = SVector{8,Float64}(ntuple(k -> ((mask >> (k-1)) & 1 == 1 ? 1.0 : -1.0), Val(8)))
            @test cell_case(phi) == mask + 1
        end
    end

    # =====================================================================
    @testset "a plane, where every moment is known in closed form" begin
        # phi = x - 0.3: the outside region is the slab x >= 0.3.
        m = cut_cell_moments(UNIT_NODES, plane_phi(SVector(1.0, 0.0, 0.0), 0.3), H1)
        @test is_cut(m)
        @test m.volume_fraction ≈ 0.7 atol = 1e-15
        @test m.centroid ≈ SVector(0.65, 0.5, 0.5) atol = 1e-15
        @test interface_area(m, H1) ≈ 1.0 atol = 1e-15
        @test interface_normal(m, H1) ≈ SVector(1.0, 0.0, 0.0) atol = 1e-15
        @test m.interface_centroid ≈ SVector(0.3, 0.5, 0.5) atol = 1e-15
        # -x face wholly inside the body, +x face wholly open, the four side faces cut at 0.7.
        @test m.face_fraction ≈ SVector(0.0, 1.0, 0.7, 0.7, 0.7, 0.7) atol = 1e-15
        # A fully open face's centroid is a real quadrature point, not a placeholder.
        @test face_centroid(m, 2, UNIT_NODES) ≈ SVector(1.0, 0.5, 0.5) atol = 1e-15
        # The quadrature-rule layer, on the same cell.
        @test measure(volume_rule(m, H1)) ≈ 0.7 atol = 1e-15
        @test measure(face_rule(m, 2, UNIT_NODES, H1)) ≈ 1.0 atol = 1e-15
        @test measure(interface_rule(m, H1)) ≈ 1.0 atol = 1e-15
    end

    # =====================================================================
    @testset "volume against an independent oracle, over random planes" begin
        # `CartesianMeshes.get_volume_fraction` clips a unit cell against a plane analytically --
        # a different algorithm entirely, and exact. The nodal reconstruction is *also* exact for a
        # plane (linear interpolation hits the crossing dead on), so these must agree to roundoff
        # rather than merely converge.
        _seed!(4242)
        worst_vol = 0.0
        worst_nrm = 0.0
        worst_clo = 0.0
        ntested = 0
        for _ in 1:1500
            nv = normalize(SVector{3,Float64}(_randn(), _randn(), _randn()))
            d = 0.05 + 0.9 * _rand()
            phi = plane_phi(nv, d)
            (all(phi .> 0) || all(phi .< 0)) && continue
            ntested += 1
            m = cut_cell_moments(UNIT_NODES, phi, H1)
            worst_vol = max(worst_vol,
                            abs(m.volume_fraction -
                                (1 - CartesianMeshes.get_volume_fraction(nv, d))))
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
    @testset "the polyhedron closes -- the non-circular test" begin
        # `interface_normal_area` comes from the six face clips; `facet_vector_area` comes from the
        # Lewiner facets. They agree only if the facets really do close the apertures, which is what
        # the divergence-theorem volume above them assumes. See this file's header.
        for (tag, geo, N) in (("sphere", SPH, 24),
                              ("wiggly", wiggle, 24),
                              ("slotted", SDFSlottedSphere(SVector(0.0,0.0,0.0), 1.0, 0.3, 0.8), 32))
            g = grid3(N)
            h = SVector(g.d...)
            worst_w = 0.0
            worst_c = 0.0
            nbadfrac = 0
            ncut = 0
            for idx in CartesianIndices(g.n)
                nodes = cell_nodes(g, idx)
                phi = cell_values(geo, g, idx)
                m = cut_cell_moments(nodes, phi, h)
                worst_c = max(worst_c, maximum(abs.(closure_residual(m, h))))
                if is_cut(m)
                    ncut += 1
                    w = facet_vector_area(nodes, nudge_zeros(phi))
                    worst_w = max(worst_w, maximum(abs.(w - interface_normal_area(m, h))))
                    (-1e-12 <= m.volume_fraction <= 1 + 1e-12) || (nbadfrac += 1)
                end
            end
            @testset "$tag" begin
                @test ncut > 100
                @test worst_w < 1e-14
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
            M = cut_cell_moments(geo, g)
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
                    face_centroid(A, 2ax, g, idx) === face_centroid(B, 2ax-1, g, idx + e) ||
                        (badc += 1)
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
    @testset "corner coordinates survive a level jump" begin
        # Choice 2, in 3D: `h/2` is exact, so a child's corner on a shared face is bitwise its
        # parent's. Without this the apertures above would agree only to roundoff across a level
        # jump, and the closure would stop being exact there.
        base = CartesianGrid(mincorner = (-1.5, -1.5, -1.5), maxcorner = (1.5, 1.5, 1.5),
                             n = (4, 4, 4))
        mesh = AdaptiveMesh(base; max_level = 4, initial_level = 1)
        MCM = MarchingCubesCutCell()
        refine!(mesh, [is_cut(cut_cell_moments(MCM, mesh, SPH, i)) for i in eachleaf(mesh)])
        refine!(mesh, [is_cut(cut_cell_moments(MCM, mesh, SPH, i)) for i in eachleaf(mesh)])
        # Every leaf's corners must be reproducible from the lattice at its own level.
        worst = 0.0
        for i in eachleaf(mesh)
            c = leaf(mesh, i)
            h = get_elem_size(mesh, c)
            nodes = cell_nodes(mesh, c)
            x0 = mesh.base.x0
            for k in 1:8
                b = MC_NODE_BITS[k]
                want = SVector{3,Float64}(ntuple(a -> x0[a] + (c.coord[a] + b[a]) * h[a], 3))
                worst = max(worst, maximum(abs.(nodes[k] - want)))
            end
        end
        @test worst == 0.0

        # ...and the moments over that refined mesh still close exactly, at every level.
        V = cut_cell_moments(SPH, mesh)
        @test length(V) == nleaves(mesh)
        worst_c = 0.0
        for i in eachleaf(mesh)
            h = SVector(get_elem_size(mesh, leaf(mesh, i))...)
            worst_c = max(worst_c, maximum(abs.(closure_residual(V[i], h))))
        end
        @test worst_c == 0.0
        @test count(is_cut, V) > 0
    end

    # =====================================================================
    @testset "a sphere: volume converges, and the first moment is recovered" begin
        errs = Float64[]
        for N in (20, 40, 80)
            g = grid3(N)
            M = cut_cell_moments(SPH, g)
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
            M = cut_cell_moments(SDFSphere(c0, 1.0), g)
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
        allout = cut_cell_moments(UNIT_NODES, SVector{8,Float64}(ntuple(_ -> 1.0, Val(8))), H1)
        allin = cut_cell_moments(UNIT_NODES, SVector{8,Float64}(ntuple(_ -> -1.0, Val(8))), H1)
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
        @test face_centroid(allout, 2, UNIT_NODES) ≈ SVector(1.0, 0.5, 0.5) atol = 1e-15
        @test measure(face_rule(allout, 2, UNIT_NODES, H1)) ≈ 1.0 atol = 1e-15
    end

    # =====================================================================
    @testset "ambiguous configurations are resolved and flagged" begin
        # An alternating z = 0 face: the saddle, and the only face pairing that is a choice.
        saddle = cut_cell_moments(UNIT_NODES,
                                  SVector{8,Float64}(1, -1, 1, -1, -1, -1, -1, -1), H1)
        @test is_cut(saddle) && is_ambiguous(saddle)
        # The body-diagonal alternation, the classic ambiguous cube.
        diag = cut_cell_moments(UNIT_NODES,
                                SVector{8,Float64}(1, -1, 1, -1, -1, 1, -1, 1), H1)
        @test is_cut(diag) && is_ambiguous(diag)
        # Both still close, which is the point: the pairing is a guess, the conservation is not.
        for m in (saddle, diag)
            @test closure_residual(m, H1) == zero(SVector{3,Float64})
            @test 0 < m.volume_fraction < 1
        end
        # A plane cuts no face ambiguously, so nothing there is flagged.
        @test !is_ambiguous(cut_cell_moments(UNIT_NODES,
                                             plane_phi(normalize(SVector(1.0, 2.0, 3.0)), 0.5), H1))
    end

    # =====================================================================
    @testset "cut_cell_faces is complete on its own" begin
        # The half that needs no case table: it must agree exactly with what the full construction
        # reports, since the full one is built on it.
        _seed!(99)
        for _ in 1:200
            phi = nudge_zeros(SVector{8,Float64}(ntuple(_ -> _randn(), Val(8))))
            kind, amb, ff, fc = cut_cell_faces(UNIT_NODES, phi, H1)
            m = cut_cell_moments(UNIT_NODES, phi, H1)
            @test kind === m.kind
            @test ff === m.face_fraction
            @test fc === m.face_centroid_local
        end
    end

    # =====================================================================
    @testset "the polyhedron closes for arbitrary corner values" begin
        # The strongest form of the non-circular test, and the one that earned its place: random
        # corner values reach Lewiner cases that no smooth body produces at a sane resolution.
        #
        # It caught a real defect. An earlier face clip read its chord pairing off the facets, which
        # is true but not sufficient -- in case 10.1.2 four of the eight facets lie entirely inside
        # a cube face plane, so all three edges of each registered as chords of that face and
        # overwrote each other. About 0.015% of random cut cubes, and where it bit, the volume was
        # wrong by a whole corner triangle. Smooth-field tests never saw it.
        _seed!(31337)
        worst = 0.0
        ncut = 0
        nbadclosure = 0
        nbadfrac = 0
        cases_seen = Set{Int}()
        for _ in 1:40_000
            phi = nudge_zeros(SVector{8,Float64}(ntuple(_ -> _randn(), Val(8))))
            m = cut_cell_moments(UNIT_NODES, phi, H1)
            is_cut(m) || continue
            ncut += 1
            push!(cases_seen, Int(MarchingCubes.cases[cell_case(phi)][1]))
            worst = max(worst, maximum(abs.(facet_vector_area(UNIT_NODES, phi) -
                                            interface_normal_area(m, H1))))
            closure_residual(m, H1) == zero(SVector{3,Float64}) || (nbadclosure += 1)
            -1e-12 <= m.volume_fraction <= 1 + 1e-12 || (nbadfrac += 1)
        end
        # Aggregated rather than one assertion per draw: forty thousand passing `@test`s bury the
        # rest of the suite, and a count is what a failure here actually wants to report.
        @test nbadclosure == 0
        @test nbadfrac == 0
        @test ncut > 30_000
        # Every non-empty Lewiner case, so the ambiguous families are genuinely exercised.
        @test length(cases_seen) == 14
        @test worst < 1e-14

        # The exact cube that exposed the defect: case 10.1.2, with two facets flush against each
        # of the two ambiguous x-faces. Pinned so the failure cannot come back silently.
        flush_cube = SVector{8,Float64}(-0.6727807769355654, -0.23277645053557436,
                                        1.0026954720119468, 0.22568989049603774,
                                        1.8172940934743478, 0.02419684567652038,
                                        -0.7057160958113278, -2.13709186873607)
        mf = cut_cell_moments(UNIT_NODES, flush_cube, H1)
        @test Int(MarchingCubes.cases[cell_case(nudge_zeros(flush_cube))][1]) == 11
        @test maximum(abs.(facet_vector_area(UNIT_NODES, nudge_zeros(flush_cube)) -
                           interface_normal_area(mf, H1))) < 1e-14
        @test closure_residual(mf, H1) == zero(SVector{3,Float64})
    end

    # =====================================================================
    @testset "the saddle decider is neighbour-consistent" begin
        # `moments.jl` resolves an ambiguous face with `face_outside_connected`, which reads only
        # that face's four corner values. The two cells touching the face spell it with different
        # `MarchingCubes` face codes, and those order the quadruple oppositely -- so the decider's
        # `A*C - B*D` comes out negated *and* `A` lands on the other diagonal. On a saddle those two
        # sign flips cancel. This asserts that cancellation rather than trusting the argument.
        _seed!(777)
        nchecked = 0
        ndisagree = 0
        for _ in 1:20_000
            # One cell's values, and its +x neighbour's: they share the face, so the neighbour's
            # corners 1,4,8,5 are this cell's 2,3,7,6.
            phi = nudge_zeros(SVector{8,Float64}(ntuple(_ -> _randn(), Val(8))))
            nb = nudge_zeros(SVector{8,Float64}(phi[2], _randn(), _randn(), phi[3],
                                                phi[6], _randn(), _randn(), phi[7]))
            fphi = SVector{4,Float64}(ntuple(k -> phi[MC_FACE_NODES[2][k]], 4))
            # only a saddle face has anything to decide
            (((fphi[1] >= 0) == (fphi[3] >= 0)) && ((fphi[2] >= 0) == (fphi[4] >= 0)) &&
             ((fphi[1] >= 0) != (fphi[2] >= 0))) || continue
            nchecked += 1
            face_outside_connected(phi, 2) == face_outside_connected(nb, 1) || (ndisagree += 1)
        end
        @test nchecked > 1000
        @test ndisagree == 0
    end

    # =====================================================================
    @testset "stretched cells, and the dimensional readers" begin
        # Marching cubes converts no distance into unit-cell coordinates, so unlike `calc_volume`
        # it carries no isotropy requirement -- but everything above this point used cubic cells,
        # which is exactly where an axis mix-up hides. `full_face_area`'s `D`-generic method is the
        # piece under test: a single `grid.d[1]^(D-1)` would be wrong here and right everywhere else.
        h = SVector(2.0, 0.5, 1.5)
        nodes = SVector{8,SVector{3,Float64}}(
            ntuple(k -> SVector{3,Float64}(MC_NODE_BITS[k]...) .* h, Val(8)))
        # phi = x - 0.5 over a 2.0 x 0.5 x 1.5 box: the outside region is the slab x >= 0.5.
        m = cut_cell_moments(nodes, SVector{8,Float64}(ntuple(k -> nodes[k][1] - 0.5, Val(8))), h)
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
            phi = SVector{8,Float64}(ntuple(k -> dot(nv, nodes[k]) - d, Val(8)))
            (all(phi .> 0) || all(phi .< 0)) && continue
            ntested += 1
            mm = cut_cell_moments(nodes, phi, h)
            scaled = nv .* h
            ns = norm(scaled)
            worst_vol = max(worst_vol,
                            abs(mm.volume_fraction -
                                (1 - CartesianMeshes.get_volume_fraction(scaled / ns, d / ns))))
            worst_clo = max(worst_clo, maximum(abs.(closure_residual(mm, h))))
            worst_close = max(worst_close,
                              maximum(abs.(facet_vector_area(nodes, nudge_zeros(phi)) -
                                           interface_normal_area(mm, h))))
        end
        @test ntested > 400
        @test worst_vol < 1e-14
        @test worst_clo == 0.0
        @test worst_close < 1e-14
    end

    # =====================================================================
    @testset "the interior vertex is the mean of the crossings" begin
        # Cases 6.1.2, 7.3, 10.2, 12.2, 13.3 and 13.4 place a vertex on no edge; `edge_crossings`
        # supplies it in slot `MC_INTERIOR_CODE` so a tiling code is a plain index. It has to be
        # what `MarchingCubes.add_c_vertex` computes, which is the mean over the edges that have a
        # crossing at all.
        _seed!(2024)
        for _ in 1:200
            phi = nudge_zeros(SVector{8,Float64}(ntuple(_ -> _randn(), Val(8))))
            crossed, pts = edge_crossings(UNIT_NODES, phi)
            any(crossed) || continue
            want = sum(pts[e] for e in 1:12 if crossed[e]) / count(crossed)
            @test pts[13] ≈ want atol = 1e-15
        end
    end

    # =====================================================================
    @testset "element type and threading" begin
        # Float32 all the way through, with no widening.
        n32 = SVector{8,SVector{3,Float32}}(ntuple(k -> SVector{3,Float32}(MC_NODE_BITS[k]...),
                                                   Val(8)))
        p32 = SVector{8,Float32}(ntuple(k -> Float32(dot(SVector(1f0, 0f0, 0f0), n32[k]) - 0.3f0),
                                        Val(8)))
        m32 = cut_cell_moments(n32, p32, SVector{3,Float32}(1, 1, 1))
        @test m32 isa CutCellData{3,Float32,6}
        @test eltype(m32) === Float32
        @test m32.volume_fraction ≈ 0.7f0 atol = 1f-6

        # The threaded bulk pass must be bitwise the serial one -- each cell writes only its own
        # slot and reads only its own corners, so there is nothing for a race to perturb.
        g = grid3(24)
        vals = sample_sdf(SPH, g)
        h = SVector(g.d...)
        threaded = cut_cell_moments(vals, g)
        serial = [cut_cell_moments(cell_nodes(g, idx), cell_values(vals, idx, Float64), h)
                  for idx in CartesianIndices(g.n)]
        @test all(threaded[i] === serial[i] for i in eachindex(serial))
    end

    # =====================================================================
    @testset "the method-tagged per-cell entry point" begin
        # The 3D half of `cut_cell_moments(method, domain, geo, cell)`. Asserted bitwise against the
        # untagged forms, because both bottom out in the same `(nodes, phi, cellsize)` primitive --
        # `===` is what says "the same construction" rather than "an agreeing one".
        MCM = MarchingCubesCutCell()
        g = grid3(12)
        CI = CartesianIndices(g.n)

        @testset "grid, nodal array" begin
            # `_moments_over_grid` is routed through this method, so the whole-grid array and the
            # per-cell call have to be the same numbers to the last bit.
            vals = sample_sdf(SPH, g)
            bulk = cut_cell_moments(vals, g)
            @test all(ci -> cut_cell_moments(MCM, g, vals, ci) === bulk[ci], CI)
        end

        @testset "grid, geometry" begin
            # Corner-sampled per cell rather than read from a node array; the two routes agree to
            # roundoff, not bitwise (see `cell_values`), so this is checked against its own route.
            h = SVector(g.d...)
            @test all(CI) do ci
                cut_cell_moments(MCM, g, SPH, ci) ===
                    cut_cell_moments(cell_nodes(g, ci), cell_values(SPH, g, ci), h)
            end
        end

        @testset "the closure is still exact through this route" begin
            @test all(CI) do ci
                closure_residual(cut_cell_moments(MCM, g, SPH, ci), SVector(g.d...)) ==
                    zero(SVector{3,Float64})
            end
        end

        @testset "the returned type is the shared one" begin
            @test cut_cell_moments(MCM, g, SPH, first(CI)) isa CutCellData{3,Float64,6}
        end
    end

    # =====================================================================
    if HAS_GPU
        @testset "GPU: the per-cell dispatch compiles and matches the host" begin
            # Regression guard for a real bug (fixed 2026-09-21): `cut_cell_faces` closed over `c2`,
            # a variable reassigned across an if/else branch inside the `for dir in 1:6` loop.
            # Julia cannot prove a captured, branch-reassigned variable's type stable across
            # iterations and boxes it (`Core.Box`), which turns every read inside the closure into a
            # dynamic dispatch -- a hard GPU-compilation failure ("unsupported dynamic function
            # invocation"), not merely a slow path. `cell_triangles` (the Lewiner table dispatch)
            # compiled and ran on GPU throughout, bitwise matching the host over all 256 corner sign
            # configurations -- so a test that only exercised that half would have missed this
            # entirely. This one goes through the same tagged entry point
            # (`cut_cell_moments(MarchingCubesCutCell(), grid, vals, ci)`) HYBIS's
            # `update_cut_cells!` actually calls.
            MCM = MarchingCubesCutCell()
            g32 = CartesianMeshes.adapt_type(Float32, grid3(16))
            vals = Float32.(sample_sdf(SPH, g32))
            CI = CartesianIndices(g32.n)

            host = [cut_cell_moments(MCM, g32, vals, ci) for ci in CI]
            vf_host = Float32[m.volume_fraction for m in host]
            kind_host = Int8[m.kind for m in host]

            dvals = adapt(CuArray, vals)
            dvf = CUDA.zeros(Float32, g32.n)
            dkind = CUDA.zeros(Int8, g32.n)
            backend = KernelAbstractions.get_backend(dvals)
            _mc_gpu_regression_kernel!(backend, 64)(dvf, dkind, dvals, g32, MCM; ndrange=g32.n)
            KernelAbstractions.synchronize(backend)

            @test Array(dkind) == kind_host
            # A few ULPs of GPU/CPU float non-associativity is expected (FMA fusion, evaluation
            # order); anything beyond that would be a real discrepancy, not noise.
            @test maximum(abs.(Array(dvf) .- vf_host)) < 100 * eps(Float32)
        end
    end
end
