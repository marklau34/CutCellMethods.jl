# =====================================
# The caches: `allocate_cache` / `update_cache!` for each method.
#
# What is asserted, per method:
#
#   * a fresh cache reads as no body -- every cell outside, all fluid, every face open
#   * the stored cells are BITWISE the per-cell entry point's, so a cache is the same construction
#     rather than one that nearly agrees
#   * a shared face is single-valued (for PLIC, after the face pass resolves it)
#   * the grid may move between updates but not change size; PLIC refuses an anisotropic grid
#   * the element type follows the grid, and the device path agrees with the host

using CutCellMethods: MarchingSquaresCutCellCache, MarchingCubesCutCellCache, PLICCutCellCache,
                      CELL_OUTSIDE, CELL_CUT, closure_residual, face_fraction_of

const CACHE_G2 = CartesianGrid(SVector(-1.3, -1.25), (48, 44), SVector(0.055, 0.055))
const CACHE_G3 = CartesianGrid(SVector(-1.5, -1.5, -1.5), (24, 24, 24), SVector(0.125, 0.125, 0.125))
const CACHE_CIRCLE = SDFCircle(center=SVector(0.013, -0.021), radius=0.83)
const CACHE_SPHERE = SDFSphere(SVector(0.01, 0.02, -0.03), 1.0)

"""How many interior faces of a field of per-cell `SVector{2D}` face values disagree between the two
cells that share them -- cell `ci`'s slot `2c` against cell `ci + e_c`'s slot `2c-1`."""
function shared_face_mismatches(ff::AbstractArray{<:Any,D}) where {D}
    bad = 0
    for c in 1:D
        e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, D))
        for ci in CartesianIndices(ntuple(a -> a == c ? size(ff, a) - 1 : size(ff, a), D))
            ff[ci][2c] === ff[ci + e][2c - 1] || (bad += 1)
        end
    end
    return bad
end

@testset verbose = true "caches" begin

    @testset "marching squares" begin
        MS = MarchingSquaresCutCell()
        g = CACHE_G2
        idx = CartesianIndices(Tuple(g.n))
        cache = allocate_cache(g, MS)
        @test cache isa MarchingSquaresCutCellCache
        @test size(cache.cells) == Tuple(g.n)
        @test size(cache.phi) == Tuple(g.n) .+ 1

        # Seeded as no body, not as whatever memory the allocation returned.
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)
        @test all(f -> all(isone, f), cache.cells.face_fraction)

        @test update_cache!(cache, CACHE_CIRCLE, g) === cache
        @test count(==(CELL_CUT), cache.cells.kind) > 100

        # Bitwise the per-corner route: `phi` is sampled at `get_node`, the expression `cell_nodes`
        # places a corner with, so every cell is built from the same numbers. And bitwise the nodal
        # route on `phi` itself, which is the construction the kernel runs.
        @test all(ci -> cache.cells[ci] === cut_cell_moments(MS, g, CACHE_CIRCLE, ci), idx)
        @test all(ci -> cache.cells[ci] === cut_cell_moments(MS, g, cache.phi, ci), idx)

        @test shared_face_mismatches(cache.cells.face_fraction) == 0
        @test all(ci -> closure_residual(cache.cells[ci], g.d) == zero(SVector{2,Float64}), idx)

        # The contour of the very same reconstruction, straight off the stored samples.
        @test length(generate_mesh(cache.phi, g, MS).elements) == count(==(CELL_CUT), cache.cells.kind)

        # A moving body is a moving grid: same `n`, new origin.
        moved = CartesianGrid(g.x0 .+ SVector(0.013, -0.007), Tuple(g.n), g.d)
        update_cache!(cache, CACHE_CIRCLE, moved)
        @test all(ci -> cache.cells[ci] === cut_cell_moments(MS, moved, CACHE_CIRCLE, ci), idx)
        @test_throws DimensionMismatch update_cache!(
            cache, CACHE_CIRCLE, CartesianGrid(SVector(0.0, 0.0), (10, 10), SVector(0.1, 0.1)))

        # A callable level set is a field too.
        update_cache!(cache, x -> x[2] - 0.3, g)
        @test count(==(CELL_CUT), cache.cells.kind) == g.n[1]

        c32 = allocate_cache(CartesianMeshes.adapt_type(Float32, g), MS)
        @test eltype(c32.cells) === CutCellData{2,Float32,4,1}
        @test eltype(c32.phi) === Float32
    end

    @testset "marching cubes" begin
        MC = MarchingCubesCutCell()
        g = CACHE_G3
        idx = CartesianIndices(Tuple(g.n))
        cache = allocate_cache(g, MC)
        @test cache isa MarchingCubesCutCellCache
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)

        update_cache!(cache, CACHE_SPHERE, g)
        @test count(==(CELL_CUT), cache.cells.kind) > 1000
        @test all(ci -> cache.cells[ci] === cut_cell_moments(MC, g, CACHE_SPHERE, ci), idx)
        @test shared_face_mismatches(cache.cells.face_fraction) == 0
        @test all(ci -> closure_residual(cache.cells[ci], g.d) == zero(SVector{3,Float64}), idx)
        @test !isempty(generate_mesh(cache.phi, g, MC; warn=false).elements)
    end

    @testset "PLIC" begin
        PL = PLICCutCell()
        g = CACHE_G2
        idx = CartesianIndices(Tuple(g.n))
        cache = allocate_cache(g, PL)
        @test cache isa PLICCutCellCache
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(f -> all(isone, f), cache.face_fraction)

        update_cache!(cache, CACHE_CIRCLE, g)
        # The fit pass is the per-cell entry point, stored whole -- one-sided `face_area` and all.
        @test all(ci -> cache.cells[ci] === cut_cell_moments(PL, g, CACHE_CIRCLE, ci), idx)

        # The face pass: every interior face single-valued, and equal to the mean of its two
        # one-sided fractions; a boundary face keeps its one cell's.
        ff = cache.face_fraction
        @test shared_face_mismatches(ff) == 0
        @test all(f -> all(x -> 0 <= x <= 1, f), ff)
        frac(ci, k) = face_fraction_of(cache.cells[ci], k, g.d)
        @test all(CartesianIndices((1:g.n[1]-1, 1:g.n[2]))) do ci
            ff[ci][2] == (frac(ci, 2) + frac(ci + CartesianIndex(1, 0), 1)) / 2
        end
        @test all(j -> ff[1, j][1] == frac(CartesianIndex(1, j), 1), 1:g.n[2])
        @test all(j -> ff[g.n[1], j][2] == frac(CartesianIndex(g.n[1], j), 2), 1:g.n[2])

        # The one-sided slots genuinely disagree somewhere, or the face pass would be a no-op.
        @test any(CartesianIndices((1:g.n[1]-1, 1:g.n[2]))) do ci
            cache.cells[ci].face_area[2] != cache.cells[ci + CartesianIndex(1, 0)].face_area[1]
        end

        aniso = CartesianGrid(SVector(-1.5, -1.5), (30, 15), SVector(0.1, 0.2))
        @test_throws ArgumentError allocate_cache(aniso, PL)

        # Dimension-generic: the 3D cache is the same two passes.
        c3 = allocate_cache(CACHE_G3, PL)
        update_cache!(c3, CACHE_SPHERE, CACHE_G3)
        @test eltype(c3.face_fraction) === SVector{6,Float64}
        @test shared_face_mismatches(c3.face_fraction) == 0
    end

    @testset "the uniform readers" begin
        # Every cache answers the same three questions, whatever its layout, and hands back its own
        # storage -- so a write through a reader is a write to the cache.
        ms = update_cache!(allocate_cache(CACHE_G2, MarchingSquaresCutCell()), CACHE_CIRCLE, CACHE_G2)
        mc = update_cache!(allocate_cache(CACHE_G3, MarchingCubesCutCell()), CACHE_SPHERE, CACHE_G3)
        pl = update_cache!(allocate_cache(CACHE_G2, PLICCutCell()), CACHE_CIRCLE, CACHE_G2)
        for c in (ms, mc)
            @test CutCellMethods.face_fractions(c) === c.cells.face_fraction
            @test CutCellMethods.volume_fractions(c) === c.cells.volume_fraction
            @test CutCellMethods.kinds(c) === c.cells.kind
        end
        @test CutCellMethods.face_fractions(pl) === pl.face_fraction      # resolved, not face_area
        @test CutCellMethods.volume_fractions(pl) === pl.cells.volume_fraction
        @test CutCellMethods.kinds(pl) === pl.cells.kind
        # The polyline clipper takes a line mesh rather than a field: a 64-gon of the same circle.
        circ = [CACHE_CIRCLE.center + CACHE_CIRCLE.radius * SVector(cos(2π * k / 64), sin(2π * k / 64))
                for k in 0:63]
        poly = Mesh([Point(p) for p in circ], [Line(Int32(k), Int32(mod1(k + 1, 64))) for k in 1:64])
        pc = update_cache!(allocate_cache(CACHE_G2, PolylineClippingCutCell()), poly, CACHE_G2)
        @test CutCellMethods.face_fractions(pc) === pc.cells.face_fraction
        @test CutCellMethods.volume_fractions(pc) === pc.cells.volume_fraction
        @test CutCellMethods.kinds(pc) === pc.cells.kind
        @test count(==(CELL_CUT), pc.cells.kind) > 100
        @test shared_face_mismatches(pc.cells.face_fraction) == 0

        ci = findfirst(==(CELL_CUT), ms.cells.kind)
        CutCellMethods.face_fractions(ms)[ci] = zero(SVector{4,Float64})
        CutCellMethods.kinds(ms)[ci] = CutCellMethods.CELL_INSIDE
        @test ms.cells[ci].face_fraction == zero(SVector{4,Float64})
        @test ms.cells[ci].kind == CutCellMethods.CELL_INSIDE
    end

    if HAS_GPU
        @testset "GPU agrees with the host" begin
            for (method, g, geo) in ((MarchingSquaresCutCell(), CACHE_G2, CACHE_CIRCLE),
                                     (MarchingCubesCutCell(), CACHE_G3, CACHE_SPHERE),
                                     (PLICCutCell(), CACHE_G2, CACHE_CIRCLE))
                g32 = CartesianMeshes.adapt_type(Float32, g)
                geo32 = SDFLibrary.adapt_type(Float32, geo)
                host = update_cache!(allocate_cache(g32, method), geo32, g32)
                dev = update_cache!(allocate_cache(g32, method; backend=CUDABackend()), geo32, g32)
                @test Array(dev.cells.kind) == host.cells.kind
                @test maximum(abs.(Array(dev.cells.volume_fraction) .- host.cells.volume_fraction)) <
                      100 * eps(Float32)
            end
        end
    end
end
