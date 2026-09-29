# =====================================
# Tri clipping, phase 5: shared faces, the closure, and the interface split by patch.
#
# What is asserted:
#
#   * every face two cells share holds bitwise the same fraction and centroid in both
#   * the closure identity is exact in every cell, and each cell's boundary faces sum to its
#     interface vector area
#   * on planar bodies the closure has nothing to correct, and the interface split by patch is
#     each face's exact area
#   * on a curved body the correction is O(h^3)
#   * 1000 random sub-cell shifts and small rotations of the prism hull: exact, closed, single-valued
#   * a second update allocates a bounded amount; the output does not depend on the thread count
#   * the GPU agrees with the host, and the GPPH hull closes, symmetric at its transom corners

using CutCellMethods: CELL_CUT, FLAG_UNSUPPORTED, FLAG_STATUS_CONFLICT, FLAG_CLOSURE_FALLBACK,
                      FLAG_CORR_LARGE, closure_residual, interface_normal_area, boundary_faces,
                      cut_report, TriScratch, poly_box!, poly_clip!, poly_volume_moment

isdefined(@__MODULE__, :tm_box) || include("tri_meshes.jl")
isdefined(@__MODULE__, :tcut_box_planes) || include("test_tri_cut.jl")
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

const TF = TriClippingCutCell()

# Everything P5 promises of one updated cache, as one NamedTuple of worst cases.
function tf_audit(cache, g)
    idx = CartesianIndices(Tuple(g.n))
    h2 = minimum(g.d)^2
    split = 0.0
    for ci in idx
        cache.cells.kind[ci] == CELL_CUT || continue
        s = sum(f -> f.area * f.normal, boundary_faces(cache, ci); init=zero(SVector{3,eltype(g.d)}))
        split = max(split, maximum(abs, s - interface_normal_area(cache.cells[ci], g.d)) / h2)
    end
    return (fraction_mismatch=shared_face_mismatches(cache.cells.face_fraction),
            centroid_mismatch=shared_face_mismatches(cache.cells.face_centroid_local),
            closure=maximum(ci -> maximum(abs, closure_residual(cache.cells[ci], g.d)), idx),
            split=split,
            correction=maximum(cache.info.correction) / h2,
            nan=count(isnan, cache.cells.volume_fraction),
            conflict=count(f -> f & FLAG_STATUS_CONFLICT != 0, cache.info.flags))
end

tf_clean(a; correction=Inf) = a.fraction_mismatch == 0 && a.centroid_mismatch == 0 &&
                              a.closure == 0 && a.split < 1e-14 && a.nan == 0 && a.conflict == 0 &&
                              a.correction < correction

@testset verbose = true "tri clipping: faces and closure" begin
    g = CartesianGrid(SVector(-1.0, -1.0, -1.0), (16, 16, 16), SVector(0.125, 0.125, 0.125))
    idx = CartesianIndices(Tuple(g.n))
    scr = TriScratch(KernelAbstractions.CPU(), Float64, 1)

    @testset "planar bodies: single-valued, closed, nothing to correct" begin
        for (lo, hi) in ((SVector(-0.33, -0.41, -0.27), SVector(0.47, 0.29, 0.52)),
                         (SVector(-0.3125, -0.4375, -0.1875), SVector(0.4375, 0.3125, 0.5625)),
                         (SVector(-0.375, -0.5, -0.25), SVector(0.5, 0.25, 0.625)))
            box, _, areas = tm_box(lo, hi; n=3)
            cache = update_cache!(allocate_cache(g, TF), box, g)
            @test tf_clean(tf_audit(cache, g); correction=1e-13)
            @test count(f -> f & (FLAG_CLOSURE_FALLBACK | FLAG_CORR_LARGE) != 0, cache.info.flags) == 0
            # The interface split by patch is each face's exact area.
            topo = cache.work.topo
            tot = Dict{String,Float64}()
            for ci in idx, f in boundary_faces(cache, ci)
                tot[topo.patch_name[f.patch]] = get(tot, topo.patch_name[f.patch], 0.0) + f.area
            end
            @test all(k -> isapprox(tot[k], areas[k]; rtol=1e-13), keys(areas))
        end
        R = SMatrix{3,3,Float64}(cosd(30), sind(30), 0, -sind(30), cosd(30), 0, 0, 0, 1) *
            SMatrix{3,3,Float64}(1, 0, 0, 0, cosd(17), sind(17), 0, -sind(17), cosd(17))
        rbox, _, _ = tm_box(SVector(-0.45, -0.3, -0.25), SVector(0.4, 0.35, 0.3); n=3, R=R,
                            t=SVector(0.013, -0.021, 0.007))
        @test tf_clean(tf_audit(update_cache!(allocate_cache(g, TF), rbox, g), g); correction=1e-13)
        hull, _, _ = tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5, t=SVector(-0.7, 0.013, -0.3))
        @test tf_clean(tf_audit(update_cache!(allocate_cache(g, TF), hull, g), g); correction=1e-13)
        # Concave and mixed creases too: the L-block, and chine flats narrower than a cell -- the
        # mixed rule's faces are cut by the same labelled arrangement, over the two cells' union.
        lb, _ = tm_lblock(a=0.4, hz=0.9, t=SVector(-0.41, -0.39, -0.43))
        @test tf_clean(tf_audit(update_cache!(allocate_cache(g, TF), lb, g), g); correction=1e-13)
        for (b1, b2) in ((0.35, 0.42), (0.33, 0.36))
            ch, _, _ = tm_chine_prism(b1=b1, b2=b2, nx=5, t=SVector(-0.7, 0.011, -0.31))
            @test tf_clean(tf_audit(update_cache!(allocate_cache(g, TF), ch, g), g); correction=1e-13)
        end
        # A fallback cell's face neighbours absorb its error through the shared face, and nothing
        # further: the thin plate, whose top and bottom share no edge.
        plate, _, _ = tm_box(SVector(-0.6, -0.55, 0.06), SVector(0.55, 0.6, 0.06 + 0.3 * 0.125); n=2)
        cache = update_cache!(allocate_cache(g, TF), plate, g)
        @test tf_clean(tf_audit(cache, g))
        uns = findall(f -> f & FLAG_UNSUPPORTED != 0, cache.info.flags)
        near(ci) = any(u -> sum(abs, Tuple(ci - u)) <= 1, uns)
        @test all(ci -> near(ci) || cache.info.correction[ci] < 1e-13 * 0.125^2, idx)
    end

    @testset "a curved body: the correction is O(h^3)" begin
        sph, _, _ = tm_icosphere(r=0.7, c=SVector(0.031, -0.017, 0.022), level=6)
        p99 = map((16, 32)) do n
            gg = CartesianGrid(SVector(-1.0, -1.0, -1.0), (n, n, n), SVector(2 / n, 2 / n, 2 / n))
            cache = update_cache!(allocate_cache(gg, TF), sph, gg)
            @test tf_clean(tf_audit(cache, gg))
            corr = sort([cache.info.correction[ci] for ci in CartesianIndices(Tuple(gg.n)) if cache.cells.kind[ci] == CELL_CUT])
            corr[ceil(Int, 0.99 * length(corr))]
        end
        # Halving h divides it by about eight.
        @test 6 < p99[1] / p99[2] < 11
    end

    @testset "1000 random shifts and small rotations of the prism hull" begin
        rng = Xoshiro(20260929)
        kw = (L=1.4, beam=0.9, deadrise=20.0, depth=0.6)
        t0 = SVector(-0.7, 0.013, -0.3)
        hull, _, _ = tm_prism(; kw..., nx=5, t=t0)
        planes0 = tcut_prism_planes(; kw..., t=t0)
        X0 = copy(hull.nodes.coord)
        centre = SVector(0.0, 0.013, 0.0)
        cache = allocate_cache(g, TF)
        worst_vf = 0.0
        bad = 0
        for trial in 1:1000
            θ = deg2rad.(10 .* (rand(rng, 3) .- 0.5))
            R = SMatrix{3,3,Float64}(1, 0, 0, 0, cos(θ[1]), sin(θ[1]), 0, -sin(θ[1]), cos(θ[1])) *
                SMatrix{3,3,Float64}(cos(θ[2]), 0, -sin(θ[2]), 0, 1, 0, sin(θ[2]), 0, cos(θ[2])) *
                SMatrix{3,3,Float64}(cos(θ[3]), sin(θ[3]), 0, -sin(θ[3]), cos(θ[3]), 0, 0, 0, 1)
            s = (SVector(rand(rng), rand(rng), rand(rng)) .- 0.5) .* g.d
            hull.nodes.coord .= Ref(R) .* (X0 .- Ref(centre)) .+ Ref(centre + s)
            update_cache!(cache, hull, g)
            planes = [(R * n, d - dot(n, centre) + dot(R * n, centre + s)) for (n, d) in planes0]
            for ci in idx
                cache.cells.kind[ci] == CELL_CUT || continue
                worst_vf = max(worst_vf, abs(cache.cells.volume_fraction[ci] - (1 - tcut_solid_fraction(scr, g, ci, planes))))
            end
            a = tf_audit(cache, g)
            bad += !(tf_clean(a; correction=1e-12) && count(f -> f & FLAG_UNSUPPORTED != 0, cache.info.flags) == 0)
        end
        @test worst_vf < 1e-12
        @test bad == 0
    end

    @testset "a second update allocates a bounded amount" begin
        sph, _, _ = tm_icosphere(r=0.6, level=4)
        cache = allocate_cache(g, TF)
        g2 = CartesianGrid(g.x0 .+ 0.37 .* g.d, Tuple(g.n), g.d)
        update_cache!(cache, sph, g)
        update_cache!(cache, sph, g2)
        update_cache!(cache, sph, g)
        bytes = @allocated update_cache!(cache, sph, g2)
        # Kernel launches and a few host scalars; nothing that scales with the cells or the mesh.
        @test bytes < 256 * 1024
    end

    @testset "the output does not depend on the thread count" begin
        # Three subprocesses, about half a minute; CUTCELL_SKIP_THREAD_TEST=1 skips them.
        if get(ENV, "CUTCELL_SKIP_THREAD_TEST", "0") != "1"
            script = joinpath(@__DIR__, "tri_thread_invariance.jl")
            proj = dirname(Base.active_project())
            digests = map((1, 4, 16)) do t
                out = read(`$(Base.julia_cmd()) --project=$proj -t $t $script`, String)
                m = match(r"DIGEST (\d+) THREADS (\d+)", out)
                m === nothing ? out : (m[1], parse(Int, m[2]))
            end
            @test all(d -> d isa Tuple, digests)
            @test [d[2] for d in digests] == [1, 4, 16]
            @test allequal(first.(digests))
        else
            @info "thread invariance skipped (CUTCELL_SKIP_THREAD_TEST=1)"
        end
    end

    @testset "the GPU agrees with the host" begin
        if HAS_GPU
            hull, _, _ = tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5, t=SVector(-0.7, 0.013, -0.3))
            g32 = CartesianMeshes.adapt_type(Float32, g)
            host = update_cache!(allocate_cache(g32, TF), hull, g32)
            dev = update_cache!(allocate_cache(g32, TF; backend=CUDABackend()), hull, g32)
            dcells = Adapt.adapt(Array, dev.cells)
            @test dcells.kind == host.cells.kind
            @test Array(dev.info.rule) == host.info.rule && Array(dev.info.flags) == host.info.flags
            @test shared_face_mismatches(dcells.face_fraction) == 0
            @test shared_face_mismatches(dcells.face_centroid_local) == 0
            @test all(ci -> closure_residual(dcells[ci], g32.d) == zero(SVector{3,Float32}), idx)
            @test maximum(x -> maximum(abs, x), dcells.face_fraction .- host.cells.face_fraction) < 10 * eps(Float32)
        end
    end

    @testset "the GPPH hull closes" begin
        hull = tm_gpph()
        if hull === nothing
            @info "gpph_clean.inp not generated (examples/geometry/run_generate_mesh.jl): skipped"
        else
            gh = CartesianGrid(SVector(-0.4, -1.5, -0.3), (172, 60, 40), SVector(0.05, 0.05, 0.05))
            cache = @test_logs match_mode = :any update_cache!(allocate_cache(gh, TF), hull, gh)
            @test tf_clean(tf_audit(cache, gh))
            r = cut_report(cache)
            @test r.overflow == 0 && r.chain_fail == 0 && r.status_conflict == 0
            # No cell falls back. (A cell the surface only grazes can still share its tiny closure
            # mismatch by mesh area, `FLAG_CLOSURE_FALLBACK`; that is what the flag is for.)
            @test r.unsupported == 0
            @test abs(r.solid_volume - r.mesh_volume) / r.mesh_volume < 1e-5
            # The transom corners, now fillet-free: symmetric port and starboard.
            idx = CartesianIndices(Tuple(gh.n))
            mirror(ci) = CartesianIndex(ci[1], gh.n[2] + 1 - ci[2], ci[3])
            corner = [ci for ci in idx if cache.cells.kind[ci] == CELL_CUT &&
                      (x = get_node(gh, ci) + gh.d / 2; x[1] < 0.05 && abs(x[2]) > 1.0 && 0.25 < x[3] < 0.45)]
            @test !isempty(corner)
            # (gmsh triangulates the two sides differently, so to the fits' agreement, not bitwise.)
            @test all(ci -> isapprox(cache.cells.volume_fraction[ci], cache.cells.volume_fraction[mirror(ci)]; atol=1e-6), corner)
        end
    end
end
