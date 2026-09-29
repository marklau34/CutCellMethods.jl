# Timing TriClippingCutCell's update, stage by stage, on the GPPH hull.
#
#   julia --project=<env with CutCellMethods, CUDA> -t auto dev/bench_tri_clipping.jl
#
# The stages are `update_cache!`'s own, in its order, each timed to a synchronize: the host's
# geometry refresh and binning, then the kernels. The first update (topology, compilation) is not
# timed.

using CutCellMethods, CartesianMeshes, MeshLibrary, StaticArrays, KernelAbstractions, Printf, Logging
const CC = CutCellMethods

const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

function staged_update!(cache, mesh, grid)
    T = eltype(cache)
    g = convert(CartesianGrid{3,T}, grid)
    w = cache.work
    backend = get_backend(cache)
    sync() = KernelAbstractions.synchronize(backend)
    t = Pair{String,Float64}[]
    push!(t, "fingerprint (host)" => @elapsed CC._topology_fingerprint(mesh))
    push!(t, "geometry (host)" => @elapsed CC._refresh_geometry!(w, mesh))
    block = CC.TriClipBlock()
    push!(t, "binning (host)" => @elapsed begin
        block = CC._bin_block!(w.tri_lo, w.tri_hi, w.X, w.tris, g, cache.tols.margin)
        CC._bin_cells!(w.bin_start, w.bin_tri, w.X, w.tris, w.tri_lo, w.tri_hi, block, g, cache.tols.margin)
        CC._bin_rows!(w.row_start, w.row_tri, w.X, w.tris, block, g)
        for (dst, src) in ((w.dev.bin_start, w.bin_start), (w.dev.bin_tri, w.bin_tri),
                           (w.dev.row_start, w.row_start), (w.dev.row_tri, w.row_tri))
            CC._upload!(dst, src)
        end
    end)
    push!(t, "reset" => @elapsed (CC._reset_block!(cache, g, w.block); sync()))
    push!(t, "pass A + cut list" => @elapsed (CC._pass_a!(cache, g, block); sync()))
    push!(t, "rows (parity)" => @elapsed (CC._classify_rows!(cache, g, block); sync()))
    push!(t, "conflicts" => @elapsed (CC._flag_conflicts!(cache, g, block); sync()))
    push!(t, "cut cells" => @elapsed (CC._cut_cells!(cache, g, block); sync()))
    w.block = block
    return t
end

function bench(label, grid, mesh; backend=KernelAbstractions.CPU())
    cache = allocate_cache(grid, TriClippingCutCell(); backend)
    with_logger(NullLogger()) do
        update_cache!(cache, mesh, grid)
        update_cache!(cache, mesh, grid)
    end
    total = minimum(@elapsed(update_cache!(cache, mesh, grid)) for _ in 1:3)
    stages = staged_update!(cache, mesh, grid)
    r = CC.cut_report(cache)
    @printf("\n%s: %s cells (%.1f M), %d cut\n", label, Tuple(grid.n), prod(grid.n) / 1e6, r.cut)
    @printf("  update_cache!  %8.1f ms\n", 1e3 * total)
    for (name, s) in stages
        @printf("    %-18s %8.1f ms\n", name, 1e3 * s)
    end
    return cache
end

inp = joinpath(@__DIR__, "..", "examples", "geometry", "gpph_clean.inp")
hull = load_mesh(inp; elem_types=[:Tri3], element_sets=["surface_$i" for i in 1:8])
println("GPPH hull: ", length(hull.elements), " triangles; ", Threads.nthreads(), " threads")
lo, hi = SVector(-0.4, -1.5, -0.3), SVector(8.2, 1.5, 1.7)
for h in (0.05, 0.025, 0.0175)
    grid = CartesianGrid(lo, Tuple(ceil.(Int, (hi - lo) ./ h)), SVector(h, h, h))
    bench("CPU Float64, h = $h", grid, hull)
    if HAS_CUDA
        g32 = CartesianMeshes.adapt_type(Float32, grid)
        bench("CUDA Float32, h = $h", g32, hull; backend=CUDABackend())
    end
end
