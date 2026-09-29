# Run as a subprocess by `test_poly_output.jl` at several thread counts: prints a digest of every
# output of several updates, which must not depend on the thread count.

using CutCellMethods, CartesianMeshes, StaticArrays, MeshLibrary, StructArrays
using SDFLibrary: SDFMesh

include(joinpath(@__DIR__, "polyline_meshes.jl"))

# Every element hashed, in order: `hash` of a whole array samples it, so it is not a digest.
function fold(h::UInt, col)
    for x in col
        h = hash(x, h)
    end
    return h
end

function digest()
    h = hash(:poly_thread_invariance)
    g = CartesianGrid(SVector(-1.0, -1.0), (96, 96), SVector(1 / 48, 1 / 48))
    bodies = ([naca4_pts(; m=0.02, p=0.4, t=0.12, n=150, chord=1.4, le=(-0.7, 0.013), α=0.1)],
              [[p + PV(-0.55, 0.0) for p in pts] for pts in three_element_pts()],
              [plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21), square_pts((0.3, -0.5), 0.004)],
              [rect_pts((-0.5, -0.25), (0.25, 0.5))])
    cache = allocate_cache(g, PolylineClippingCutCell())
    for body in bodies
        # a fresh update and one on a moved grid, through the same cache
        for gg in (g, CartesianGrid(g.x0 .+ 0.29 .* g.d, Tuple(g.n), g.d))
            c = update_cache!(cache, SDFMesh(poly_mesh(body)), gg)
            for sa in (c.cells, c.info, c.edges.ax, c.edges.ay, c.regions, c.rinfo, c.arcs, c.bsegs)
                for col in StructArrays.components(sa)
                    h = fold(h, col)
                end
            end
        end
    end
    return h
end

println("DIGEST ", digest(), " THREADS ", Threads.nthreads())
