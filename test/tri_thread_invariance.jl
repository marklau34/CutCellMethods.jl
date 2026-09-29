# Run as a subprocess by `test_tri_faces.jl` at several thread counts: prints a digest of every
# output of three updates, which must not depend on the thread count.

using CutCellMethods, CartesianMeshes, StaticArrays, LinearAlgebra, MeshLibrary

include(joinpath(@__DIR__, "tri_meshes.jl"))

# Every element hashed, in order: `hash` of a whole array samples it, so it is not a digest.
function fold(h::UInt, col)
    for x in col
        h = hash(x, h)
    end
    return h
end

function digest()
    h = hash(:tri_thread_invariance)
    g = CartesianGrid(SVector(-1.0, -1.0, -1.0), (24, 24, 24), SVector(1 / 12, 1 / 12, 1 / 12))
    bodies = (tm_icosphere(r=0.7, c=SVector(0.031, -0.017, 0.022), level=4)[1],
              tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5, t=SVector(-0.7, 0.013, -0.3))[1],
              tm_lblock(a=0.4, hz=0.9, t=SVector(-0.41, -0.39, -0.43))[1])
    for body in bodies
        c = update_cache!(allocate_cache(g, TriClippingCutCell()), body, g)
        for col in (c.cells.kind, c.cells.ambiguous, c.cells.volume_fraction, c.cells.centroid,
                    c.cells.face_fraction, c.cells.face_centroid_local, c.cells.interface_centroid,
                    c.info.npatch, c.info.rule, c.info.flags, c.info.correction)
            h = fold(h, col)
        end
        for ci in CartesianIndices(Tuple(g.n))
            c.cells.kind[ci] == CutCellMethods.CELL_CUT || continue
            h = fold(h, CutCellMethods.boundary_faces(c, ci))
        end
    end
    return h
end

println("DIGEST ", digest(), " THREADS ", Threads.nthreads())
