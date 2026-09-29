# =====================================
# PolylineClippingCutCell: an airfoil rebuilt every step, timed
#
# NACA 0012 (chord 1) and a three-element section, their elements at most h/4 long, in a
# [-0.5, 1.5] x [-0.5, 0.5] domain at h = 1/N. Each "step" moves the grid by a fraction of a cell --
# the rigid-body rebuild a moving-body solver does every time step -- and updates the cache. Reports
# the update time (median over the steps) split into the host's share and the kernels', on the CPU
# at the current thread count and on a CUDA device if there is one, with the cut's own checks.
#
#     julia --project=<env with CutCellMethods> -t 16 examples/benchmark_polyline.jl [N...]
#
# The N = 256 cut is also written to `examples/outputs/` for ParaView.

using CutCellMethods, CartesianMeshes, MeshLibrary, StaticArrays, KernelAbstractions
using CutCellMethods: PL_INVALID, PL_SPLIT, write_cache_vtk
const CM = CutCellMethods
median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
include(joinpath(@__DIR__, "..", "test", "polyline_meshes.jl"))

const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Cosine spacing puts its longest elements mid-chord, at about pi/(2n) of the chord.
stations(h) = ceil(Int, 2π / h)
naca0012(h) = [naca4_pts(; t=0.12, n=stations(h), chord=1.0, le=(0.0, 0.0))]
function three_element(h)
    n = stations(h)
    return [naca4_pts(; m=0.02, p=0.4, t=0.10, n=ceil(Int, 0.18n), chord=0.18, le=(-0.20, -0.06), α=0.45),
            naca4_pts(; m=0.02, p=0.4, t=0.12, n=n, chord=1.0, le=(0.0, 0.0), α=0.0),
            naca4_pts(; m=0.02, p=0.4, t=0.10, n=ceil(Int, 0.3n), chord=0.30, le=(1.03, -0.05), α=0.35)]
end

grid_for(N, T) = CartesianGrid(SVector{2,T}(-0.5, -0.5), (2N, N), SVector{2,T}(1 / N, 1 / N))

# The host's share of an update, done again on the cache's own buffers: everything before the
# first kernel launch (with the topology cached, as it is from the second update on).
function host_part!(cache, mesh, g)
    w = cache.work
    C = w.cross
    CM._pl_refresh_geometry!(w, mesh)
    CM._pl_check_loops(w.topo, w.X, w.lines)
    CM._pl_lattice!(w.lattice, g)
    b = CM._pl_block(w.X, w.lines, w.lattice)
    CM._pl_crossings!(C, w.X, w.lines, w.lattice, b)
    CM._pl_vertex_bins!(C, w.X, w.lines, w.lattice, b)
    CM._pl_classify!(C, b)
    CM._pl_cut_lists!(C, w.X, w.lines, w.topo, w.lattice, b)
    CM._pl_upload!(w, w.topo)
    return nothing
end

# `moves = :body`: the body moves through a fixed grid (its nodes rotated a little and shifted by a
# fraction of a cell each step), so an update resets only the cells the body covered. `:grid`: the
# grid's origin moves instead, and every cell is rewritten, since a cell's stored centroids are
# global coordinates of the old grid.
function run_case(name, loops, N; backend=CPU(), T=Float64, steps=20, moves=:body)
    g0 = grid_for(N, T)
    h = Float64(g0.d[1])
    cache = allocate_cache(g0, PolylineClippingCutCell(); backend)
    if moves === :body
        meshes = map(1:steps) do _
            R, t = _rot(1e-3 * randn()), h * PV(rand(), rand())    # one rigid motion per step
            poly_mesh([[R * p + t for p in pts] for pts in loops])
        end
        grids = fill(g0, steps)
    else
        meshes = fill(poly_mesh(loops), steps)
        grids = [CartesianGrid(g0.x0 + g0.d .* SVector{2,T}(rand(), rand()), Tuple(g0.n), g0.d)
                 for _ in 1:steps]
    end
    update_cache!(cache, meshes[1], grids[1]); update_cache!(cache, meshes[2], grids[2])  # warm up
    t_total = [(@elapsed update_cache!(cache, meshes[k], grids[k])) for k in 1:steps]
    t_host = [(@elapsed host_part!(cache, meshes[k], grids[k])) for k in 1:steps]
    update_cache!(cache, meshes[end], grids[end])
    mesh = meshes[end]
    st = Array(cache.info.status)
    vf = Array(cache.cells.volume_fraction)
    res = Array(cache.info.residual)
    g = grids[end]
    fluid = sum(Float64, vf) * prod(Float64.(g.d))
    want = prod(Float64.(g.n .* g.d)) - sum(pts_area, loops)
    return (case=name, N=N, cells=prod(g0.n), elements=length(mesh.elements),
            cut=cache.work.stats.ncut, split=count(==(PL_SPLIT), st), invalid=count(==(PL_INVALID), st),
            area_err=abs(fluid - want) / want, residual=maximum(r -> maximum(abs, r), res) / Float64(g.d[1]),
            total_ms=1000median(t_total), host_ms=1000median(t_host), cache=cache, grid=g)
end

function report(r, backend)
    println(rpad(r.case, 14), lpad(r.N, 5), lpad(r.cells, 10), lpad(r.elements, 7), lpad(r.cut, 7),
            lpad(r.split, 6), lpad(r.invalid, 5), "   ", rpad(string(round(r.area_err; sigdigits=2)), 8),
            rpad(string(round(r.residual; sigdigits=2)), 8), lpad(round(r.total_ms; digits=2), 9),
            lpad(round(r.host_ms; digits=2), 9), lpad(round(r.total_ms - r.host_ms; digits=2), 9), "  ", backend)
end

function main(Ns)
    println("threads: ", Threads.nthreads(), HAS_CUDA ? "   device: $(CUDA.name(CUDA.device()))" : "")
    println(rpad("case", 14), lpad("N", 5), lpad("cells", 10), lpad("elems", 7), lpad("cut", 7),
            lpad("split", 6), lpad("inv", 5), "   ", rpad("area", 8), rpad("resid/h", 8),
            lpad("total ms", 9), lpad("host ms", 9), lpad("kern ms", 9), "  backend")
    for N in Ns, (name, body) in (("NACA 0012", naca0012), ("three-element", three_element))
        h = 1 / N
        loops = body(h)
        r = run_case(name, loops, N)
        report(r, "CPU x$(Threads.nthreads()), body moves")
        report(run_case(name, loops, N; moves=:grid), "CPU x$(Threads.nthreads()), grid moves")
        if N == 256
            mkpath(joinpath(@__DIR__, "outputs"))
            tag = name == "NACA 0012" ? "naca0012" : "three_element"
            write_cache_vtk(joinpath(@__DIR__, "outputs", "polyline_$tag"), r.cache, r.grid)
        end
        if HAS_CUDA
            for T in (Float32, Float64)
                report(run_case(name, loops, N; backend=CUDABackend(), T), "CUDA $T, body moves")
            end
        end
        GC.gc()
    end
end

main(isempty(ARGS) ? (256, 512, 1024) : parse.(Int, ARGS))
