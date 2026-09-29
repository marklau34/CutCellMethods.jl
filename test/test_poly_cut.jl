# =====================================
# Polyline clipping, P3: the cut -- regions, apertures, walls, islands, slivers
#
# The brief's validation table, less AMR and threads. The method is exact for polygons, so areas are
# checked at roundoff against the polygon itself -- per cell against exact rational clipping, and in
# total against the domain less the body -- rather than for a convergence rate. The invariants that
# hold by construction (shared faces, closure of `CutCellData`) are checked bitwise, and the ones
# that only hold if the walk is right (arcs summing to apertures, regions to the walk-free total,
# closure over the actual segments, neighbours agreeing on who owns an arc) on every cell.

using Test
using StaticArrays
using CartesianMeshes
using MeshLibrary
using Random
using CutCellMethods: PolylineClippingCutCellCache, PL_FLUID, PL_SOLID, PL_CUT, PL_SPLIT, PL_INVALID,
                      PL_FLAG_SNAPPED, PL_FLAG_ISLAND, CELL_INSIDE, CELL_OUTSIDE, CELL_CUT,
                      closure_residual, interface_normal_area, region, regions, arcs,
                      boundary_segments

isdefined(@__MODULE__, :PV) || include("polyline_meshes.jl")
isdefined(@__MODULE__, :shared_face_mismatches) || function shared_face_mismatches(ff::AbstractArray{<:Any,D}) where {D}
    bad = 0
    for c in 1:D
        e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, D))
        for ci in CartesianIndices(ntuple(a -> a == c ? size(ff, a) - 1 : size(ff, a), D))
            ff[ci][2c] === ff[ci + e][2c - 1] || (bad += 1)
        end
    end
    return bad
end

const PX_G = CartesianGrid(SVector(-1.0, -1.0), (64, 64), SVector(1 / 32, 1 / 32))
const PX_M = PolylineClippingCutCell()
const PX_ALLOWED = PL_FLAG_SNAPPED | PL_FLAG_ISLAND     # the flags that are information, not trouble

cut(loops; g=PX_G, method=PX_M) = update_cache!(allocate_cache(g, method), poly_mesh(loops), g)

# ---- references ----------------------------------------------------------------------------------

const RQ = Rational{BigInt}

# The cell box clipped to the half-plane left of the directed line a -> b, exactly (Sutherland-
# Hodgman in rationals); the polygon as a vector of points.
function clip_left(poly::Vector{SVector{2,RQ}}, a, b)
    A, B = SVector{2,RQ}(a), SVector{2,RQ}(b)
    side(p) = (B[1] - A[1]) * (p[2] - A[2]) - (B[2] - A[2]) * (p[1] - A[1])
    out = SVector{2,RQ}[]
    for i in eachindex(poly)
        P, Q = poly[i], poly[mod1(i + 1, length(poly))]
        sP, sQ = side(P), side(Q)
        sP >= 0 && push!(out, P)
        if (sP > 0 && sQ < 0) || (sP < 0 && sQ > 0)
            push!(out, P + (Q - P) * (sP / (sP - sQ)))
        end
    end
    return out
end
box_poly(lo, hi) = SVector{2,RQ}[SVector{2,RQ}(lo[1], lo[2]), SVector{2,RQ}(hi[1], lo[2]),
                                 SVector{2,RQ}(hi[1], hi[2]), SVector{2,RQ}(lo[1], hi[2])]
function poly_area(poly)
    a = RQ(0)
    for i in eachindex(poly)
        p, q = poly[i], poly[mod1(i + 1, length(poly))]
        a += p[1] * q[2] - p[2] * q[1]
    end
    return a / 2
end

# Every invariant a finished cache must satisfy, whatever the body. Returns the number of failures
# of each kind, so a test reports what broke.
function cut_invariants(cache, g)
    idx = CartesianIndices(Tuple(g.n))
    tol = cache.tols.closure_tol
    bad = Dict{Symbol,Int}()
    note(k) = (bad[k] = get(bad, k, 0) + 1)
    shared_face_mismatches(cache.cells.face_fraction) == 0 || note(:shared_fraction)
    shared_face_mismatches(cache.cells.face_centroid_local) == 0 || note(:shared_centroid)
    for ci in idx
        m = cache.cells[ci]
        inf = cache.info[ci]
        closure_residual(m, g.d) == zero(SVector{2,Float64}) || note(:closure_by_construction)
        inf.status == PL_INVALID && note(:invalid)
        inf.flags & ~PX_ALLOWED == 0 || note(:flags)
        maximum(abs, inf.residual) <= tol || note(:residual)
        all(f -> 0 <= f <= 1, m.face_fraction) || note(:fraction_range)
        0 <= m.volume_fraction <= 1 || note(:volume_range)
        # Region records exist exactly for split cells, and add up to the cell.
        if inf.nregion >= 2
            inf.status == PL_SPLIT || note(:split_status)
            rs = regions(cache, ci)
            length(rs) == inf.nregion || note(:region_count)
            isapprox(sum(r -> r.volume_fraction, rs), m.volume_fraction; atol=1e-13) || note(:region_sum)
            all(r -> isapprox(sum(rr -> rr.face_fraction[r], rs), m.face_fraction[r]; atol=1e-13), 1:4) ||
                note(:region_faces)
        else
            isempty(regions(cache, ci)) || note(:stray_regions)
        end
        # Neighbours agree on who owns each arc.
        for a in arcs(cache, ci)
            nb = ci + (a.dir == 1 ? CartesianIndex(-1, 0) : a.dir == 2 ? CartesianIndex(1, 0) :
                       a.dir == 3 ? CartesianIndex(0, -1) : CartesianIndex(0, 1))
            opp = a.dir == 1 ? 2 : a.dir == 2 ? 1 : a.dir == 3 ? 4 : 3
            other = [b for b in arcs(cache, nb) if b.dir == opp && b.k == a.k]
            if isempty(other)
                a.nbr_region == 1 || note(:nbr_uncut)
            else
                b = only(other)
                (b.region == a.nbr_region && b.nbr_region == a.region && b.closed == a.closed &&
                 b.s0 === a.s0 && b.s1 === a.s1) || note(:nbr_mismatch)
            end
        end
    end
    for (k, ri) in enumerate(cache.rinfo)
        maximum(abs, ri.residual) <= tol || note(:region_residual)
        cache.info[ri.cell].rslot + ri.region - 1 == k || note(:rinfo_index)
    end
    return bad
end

total_fluid(cache, g) = sum(cache.cells.volume_fraction) * prod(g.d)
domain_area(g) = prod(g.n .* g.d)

# ---- tests ----------------------------------------------------------------------------------------

@testset verbose = true "polyline clipping: the cut" begin

    @testset "polygon areas are exact: $name" for (name, loops) in [
            "rotated square" => [square_pts((0.013, -0.021), 0.9; θ=0.3)],
            "64-gon" => [ngon_pts((0.01, 0.02), 0.7, 64; θ0=0.05)],
            "airfoil" => [naca4_pts(; m=0.02, p=0.4, t=0.12, n=100, chord=1.4, le=(-0.7, 0.01), α=0.1)],
            "three elements" => [[p + PV(-0.55, 0.0) for p in pts] for pts in three_element_pts()],
            "thin plate" => [plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21)]]
        cache = cut(loops)
        want = domain_area(PX_G) - sum(pts_area, loops)
        @test total_fluid(cache, PX_G) ≈ want rtol = 1e-13
        @test isempty(cut_invariants(cache, PX_G))
        # And cell by cell, against the polygon clipped exactly, for the convex bodies.
        if length(loops) == 1 && name in ("rotated square", "64-gon")
            pts = loops[1]
            worst = 0.0
            for ci in CartesianIndices(Tuple(PX_G.n))
                cache.info.status[ci] in (PL_CUT, PL_SPLIT) || continue
                lo, hi = get_node(PX_G, ci), get_node(PX_G, ci + CartesianIndex(1, 1))
                solid = box_poly(lo, hi)
                for i in eachindex(pts)
                    solid = clip_left(solid, pts[i], pts[mod1(i + 1, length(pts))])
                end
                fluid = Float64(poly_area(box_poly(lo, hi)) - poly_area(solid))
                worst = max(worst, abs(cache.cells.volume_fraction[ci] * prod(PX_G.d) - fluid))
            end
            @test worst <= 1e-13 * prod(PX_G.d)
        end
    end

    @testset "a square on the grid's nodes" begin
        # Vertices on nodes, edges along lines: every degeneracy at once. Areas exact; the top and
        # right faces, which the perturbation puts just inside the solid cells, are snapped so their
        # walls land in the fluid cells, as the bottom and left faces' do by themselves.
        lo, hi = (-0.5, -0.25), (0.25, 0.5)
        cache = cut([rect_pts(lo, hi)])
        @test isempty(cut_invariants(cache, PX_G))
        @test total_fluid(cache, PX_G) == domain_area(PX_G) - 0.75^2
        h = 1 / 32
        I(x, y) = CartesianIndex(round(Int, (x + 1) / h) + 1, round(Int, (y + 1) / h) + 1)  # cell at lower-left (x, y)
        i0, j0 = I(lo...).I
        i1, j1 = I(hi...).I            # first cell past the square
        for ci in CartesianIndices(Tuple(PX_G.n))
            i, j = ci.I
            inside = i0 <= i < i1 && j0 <= j < j1
            m = cache.cells[ci]
            if inside
                @test m.volume_fraction == 0 && m.kind == CELL_INSIDE
            else
                @test m.volume_fraction == 1
            end
        end
        # The four sides' walls, each on the fluid side with the body's outward normal.
        side_ok = true
        for i in (i0 + 1):(i1 - 2)
            side_ok &= interface_normal_area(cache.cells[CartesianIndex(i, j1)], PX_G.d) == SVector(0, h)
            side_ok &= interface_normal_area(cache.cells[CartesianIndex(i, j0 - 1)], PX_G.d) == SVector(0, -h)
        end
        for j in (j0 + 1):(j1 - 2)
            side_ok &= interface_normal_area(cache.cells[CartesianIndex(i1, j)], PX_G.d) == SVector(h, 0)
            side_ok &= interface_normal_area(cache.cells[CartesianIndex(i0 - 1, j)], PX_G.d) == SVector(-h, 0)
        end
        @test side_ok
        # The snapped ones say so.
        @test cache.info.flags[CartesianIndex(i0 + 3, j1)] & PL_FLAG_SNAPPED != 0
        @test cache.info.flags[CartesianIndex(i0 + 3, j0 - 1)] & PL_FLAG_SNAPPED == 0
    end

    @testset "a thin plate splits every cell it spans in two, exactly" begin
        pts = plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21)
        cache = cut([pts])
        @test isempty(cut_invariants(cache, PX_G))
        X = pts
        vcell = Set(CartesianIndex(CutCellMethods._locate(CutCellMethods.PolylineLattice(PX_G).xs, p[1]),
                                   CutCellMethods._locate(CutCellMethods.PolylineLattice(PX_G).ys, p[2])) for p in X)
        # Spanned: both long sides (elements 1 and 3) cross the cell and no corner of the plate is in
        # it. (A cell only one side clips a corner of is cut but has one region.)
        crosses_both(ci) = issubset((1, 3), Set(s.elem for s in boundary_segments(cache, ci)))
        cutc = [ci for ci in CartesianIndices(Tuple(PX_G.n)) if cache.info.status[ci] in (PL_CUT, PL_SPLIT)]
        spanned = [ci for ci in cutc if !(ci in vcell) && crosses_both(ci)]
        @test length(spanned) > 40
        @test all(ci -> cache.info.nregion[ci] == 2, spanned)
        @test all(ci -> cache.info.nregion[ci] == 1, [ci for ci in cutc if !crosses_both(ci)])
        # Each region against the cell clipped by the plate's side it lies on, in rationals.
        nrm = SVector(-sin(0.21), cos(0.21))
        worst = 0.0
        for ci in spanned
            lo, hi = get_node(PX_G, ci), get_node(PX_G, ci + CartesianIndex(1, 1))
            below = Float64(poly_area(clip_left(box_poly(lo, hi), X[2], X[1])))   # right of the lower side
            above = Float64(poly_area(clip_left(box_poly(lo, hi), X[4], X[3])))   # left of the upper side reversed
            for r in 1:2
                rg = region(cache, ci, r)
                want = nrm' * (rg.centroid - SVector(0.0, 0.1)) < 0 ? below : above
                worst = max(worst, abs(rg.volume_fraction * prod(PX_G.d) - want))
            end
        end
        @test worst <= 1e-13 * prod(PX_G.d)
    end

    @testset "two plates in one cell: three regions" begin
        cache = cut([plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21), plate_pts((0.0, 0.112), 1.3, 1e-3; θ=0.21)])
        @test isempty(cut_invariants(cache, PX_G))
        @test count(==(3), cache.info.nregion) > 10
        @test total_fluid(cache, PX_G) ≈ domain_area(PX_G) - 2 * 1.3e-3 rtol = 1e-13
    end

    @testset "a sharp trailing edge: split upstream, one region at the tip" begin
        foil = naca4_pts(; m=0.0, t=0.12, n=100, chord=1.4, le=(-0.7, 0.013), α=0.0)
        cache = cut([foil])
        @test isempty(cut_invariants(cache, PX_G))
        L = CutCellMethods.PolylineLattice(PX_G)
        te = foil[1]
        tip = CartesianIndex(CutCellMethods._locate(L.xs, te[1]), CutCellMethods._locate(L.ys, te[2]))
        @test cache.info.status[tip] == PL_CUT && cache.info.nregion[tip] == 1
        @test !cache.cells.ambiguous[tip]
        # The tip cell's region wraps round the spike: open on all four sides.
        @test all(>(0), cache.cells.face_fraction[tip])
        split = [ci for ci in CartesianIndices(Tuple(PX_G.n)) if cache.info.status[ci] == PL_SPLIT]
        @test !isempty(split)
        @test all(ci -> ci[1] < tip[1] && tip[1] - ci[1] <= 3 && abs(ci[2] - tip[2]) <= 1, split)
        @test total_fluid(cache, PX_G) ≈ domain_area(PX_G) - pts_area(foil) rtol = 1e-13
    end

    @testset "islands" begin
        h = 1 / 32
        # Two islands in one otherwise fluid cell.
        a1, a2 = 0.004, 0.003
        loops = [square_pts((-0.4016, 0.3016), a1), square_pts((-0.3934, 0.3094), a2)]
        cache = cut(loops)
        @test isempty(cut_invariants(cache, PX_G))
        ci = findfirst(==(PL_CUT), cache.info.status)
        @test count(==(PL_CUT), cache.info.status) == 1
        m = cache.cells[ci]
        @test cache.info.flags[ci] & PL_FLAG_ISLAND != 0
        @test m.volume_fraction ≈ 1 - (a1^2 + a2^2) / h^2 rtol = 1e-14
        @test m.face_fraction == SVector(1.0, 1.0, 1.0, 1.0)
        @test interface_normal_area(m, PX_G.d) == SVector(0.0, 0.0)
        @test m.ambiguous
        segs = boundary_segments(cache, ci)
        @test length(segs) == 8 && all(s -> s.region == 1, segs)
        @test sum(s -> s.length, segs) ≈ 4 * (a1 + a2) rtol = 1e-14

        # An island beside a body's wall in the same cell, and one inside a split cell: each comes
        # out of the region it lies in, and nothing else changes.
        body = square_pts((0.0, 0.0), 0.9)                 # right face at x = 0.45, inside cell column 47
        plate = plate_pts((0.0, 0.6), 1.2, 1e-3)            # splits row 52, y in [0.59375, 0.625]
        c2 = cut([body, plate])
        wall_cell = CartesianIndex(47, 33)                  # x in [0.4375, 0.46875], y in [0, 0.03125]
        split_cell = CartesianIndex(40, 52)
        @test c2.info.status[wall_cell] == PL_CUT
        @test c2.info.status[split_cell] == PL_SPLIT
        lo = get_node(PX_G, split_cell)
        isle_wall = square_pts((0.46, 0.01), 0.004)         # between the face and the cell's right edge
        isle_split = square_pts((lo[1] + 0.4h, (0.6005 + lo[2] + h) / 2), 0.002)   # above the plate
        c3 = cut([body, plate, isle_wall, isle_split])
        @test isempty(cut_invariants(c3, PX_G))
        @test c3.info.nregion[wall_cell] == 1 && c3.info.flags[wall_cell] & PL_FLAG_ISLAND != 0
        @test c3.cells.volume_fraction[wall_cell] * h^2 ≈
              c2.cells.volume_fraction[wall_cell] * h^2 - 0.004^2 rtol = 1e-12
        @test c3.cells.face_fraction[wall_cell] == c2.cells.face_fraction[wall_cell]
        @test c3.info.nregion[split_cell] == 2 && c3.info.flags[split_cell] & PL_FLAG_ISLAND != 0
        ra = [region(c3, split_cell, r) for r in 1:2]
        rb = [region(c2, split_cell, r) for r in 1:2]
        up = argmax([r.centroid[2] for r in rb])
        @test ra[up].volume_fraction * h^2 ≈ rb[up].volume_fraction * h^2 - 0.002^2 rtol = 1e-12
        @test ra[3 - up].volume_fraction == rb[3 - up].volume_fraction
        @test length(boundary_segments(c3, split_cell)) == length(boundary_segments(c2, split_cell)) + 4
        @test count(s -> s.region == up, boundary_segments(c3, split_cell)) ==
              count(s -> s.region == up, boundary_segments(c2, split_cell)) + 4
        @test total_fluid(c3, PX_G) ≈ domain_area(PX_G) - 0.81 - 1.2e-3 - 0.004^2 - 0.002^2 rtol = 1e-13
    end

    @testset "the readers" begin
        cache = cut([plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21)])
        sp = findfirst(==(PL_SPLIT), cache.info.status)
        @test region(cache, sp, 1) === cache.regions[cache.info.rslot[sp]]
        @test length(regions(cache, sp)) == 2
        @test_throws BoundsError region(cache, sp, 3)
        fl = findfirst(==(PL_FLUID), cache.info.status)
        @test region(cache, fl, 1) === cache.cells[fl]
        @test isempty(arcs(cache, fl)) && isempty(boundary_segments(cache, fl))
        @test all(s -> s.region in (1, 2), boundary_segments(cache, sp))
        @test sum(s -> s.length, cache.bsegs) ≈ 2 * (1.3 + 1e-3) rtol = 1e-12
    end

    @testset "offset and rotation sweep" begin
        # The brief's gate: >= 10^4 random sub-cell shifts and rotations, no invalid cell, the
        # invariants everywhere, the area at roundoff.
        rng = MersenneTwister(2026)
        g0 = CartesianGrid(SVector(-1.0, -1.0), (32, 32), SVector(1 / 16, 1 / 16))
        shapes = [square_pts((0.0, 0.0), 0.9), ngon_pts((0.0, 0.0), 0.6, 64),
                  naca4_pts(; m=0.02, p=0.4, t=0.12, n=60, chord=1.2, le=(-0.6, 0.0)),
                  plate_pts((0.0, 0.0), 1.2, 2e-3)]
        areas = pts_area.(shapes)
        caches = [allocate_cache(g0, PX_M) for _ in shapes]
        nbad = ninvalid = 0
        worst = 0.0
        n = 0
        for trial in 1:2500, (k, pts) in enumerate(shapes)
            R = _rot(2π * rand(rng))
            mesh = poly_mesh([R * p for p in pts])
            g = CartesianGrid(g0.x0 + g0.d .* SVector(rand(rng), rand(rng)), Tuple(g0.n), g0.d)
            c = update_cache!(caches[k], mesh, g)
            n += 1
            ninvalid += count(==(PL_INVALID), c.info.status)
            worst = max(worst, abs(total_fluid(c, g) - (domain_area(g) - areas[k])) / domain_area(g))
            # the full invariant check on a sample, it is the slow part
            if trial % 25 == 0
                nbad += !isempty(cut_invariants(c, g))
            else
                nbad += any(f -> f & ~PX_ALLOWED != 0, c.info.flags)
                nbad += shared_face_mismatches(c.cells.face_fraction) != 0
            end
        end
        @test n >= 10^4
        @test ninvalid == 0
        @test nbad == 0
        @test worst <= 1e-13
    end

    @testset "stretched cells, and the staggered grids a solver shifts: $gname" for (gname, g0) in let
            p = CartesianGrid(SVector(-1.0, -1.0), (32, 48), SVector(1 / 16, 1 / 24))
            ["dx != dy" => p,
             # a MAC solver's U and V families: the pressure grid shifted by half a cell, one wider
             "u family" => CartesianGrid(p.x0 - SVector(p.d[1] / 2, 0), Tuple(p.n) .+ (1, 0), p.d),
             "v family" => CartesianGrid(p.x0 - SVector(0, p.d[2] / 2), Tuple(p.n) .+ (0, 1), p.d),
             "4:1 cells" => CartesianGrid(SVector(-1.0, -1.0), (64, 16), SVector(1 / 32, 1 / 8))]
        end
        rng = MersenneTwister(5)
        shapes = [square_pts((0.0, 0.0), 0.9), ngon_pts((0.0, 0.0), 0.6, 64),
                  naca4_pts(; m=0.02, p=0.4, t=0.12, n=60, chord=1.2, le=(-0.6, 0.0)),
                  plate_pts((0.0, 0.0), 1.2, 2e-3)]
        c = allocate_cache(g0, PX_M)
        ninvalid = nbad = 0
        worst = 0.0
        for trial in 1:100, pts in shapes
            R = _rot(2π * rand(rng))                 # one rotation per trial, not per point
            mesh = poly_mesh([R * q for q in pts])
            g = CartesianGrid(g0.x0 + g0.d .* SVector(rand(rng), rand(rng)), Tuple(g0.n), g0.d)
            update_cache!(c, mesh, g)
            ninvalid += count(==(PL_INVALID), c.info.status)
            worst = max(worst, abs(total_fluid(c, g) - (domain_area(g) - pts_area(pts))) / domain_area(g))
            trial % 10 == 0 && (nbad += !isempty(cut_invariants(c, g)))
        end
        @test ninvalid == 0
        @test nbad == 0
        @test worst <= 1e-13
    end

    @testset "element type" begin
        g32 = CartesianGrid(SVector(-1.0f0, -1.0f0), (64, 64), SVector(1.0f0 / 32, 1.0f0 / 32))
        loops = [naca4_pts(; m=0.02, p=0.4, t=0.12, n=100, chord=1.4, le=(-0.7, 0.01), α=0.1)]
        c32 = cut(loops; g=g32)
        c64 = cut(loops)
        @test eltype(c32.cells) === CutCellData{2,Float32,4,1}
        @test c32.info.status == c64.info.status
        @test count(==(PL_INVALID), c32.info.status) == 0
        @test shared_face_mismatches(c32.cells.face_fraction) == 0
        @test maximum(abs.(c32.cells.volume_fraction .- c64.cells.volume_fraction)) < 1e-5
    end
end
