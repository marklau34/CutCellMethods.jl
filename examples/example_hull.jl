using MeshLibrary
using CartesianMeshes
using CutCellMethods
using SDFLibrary
using StaticArrays
import GLMakie

# Gmsh msh file
element_sets = ["surface_$i" for i in 1:8]
mesh = load_mesh(joinpath(@__DIR__, "geometry", "gpph_clean.inp"), elem_types=[:Tri3,], element_sets=element_sets)
# viz(mesh, showsegments = true)

# Bottom right hull
# surface_mesh = Mesh(mesh.nodes, mesh.elements[mesh.elemset[2].elems])
# viz(surface_mesh, showsegments = true)

# ==================================================================================
# Cut cells

# A uniform grid around the hull's bounding box, padded by a few cells on every side: the cut needs
# the body clear of the grid's edge.
h = 0.05 # cell size, ft
pad = 4h
lo = reduce((a, b) -> min.(a, b), mesh.nodes.coord) .- pad
hi = reduce((a, b) -> max.(a, b), mesh.nodes.coord) .+ pad
grid = CartesianGrid(lo, Tuple(ceil.(Int, (hi - lo) ./ h)), SVector(h, h, h))

# One cache for the grid, updated from the hull. Each element set is a patch the cut fits one
# plane to per cell.
cache = allocate_cache(grid, TriClippingCutCell())
@time update_cache!(cache, mesh, grid)
display(CutCellMethods.cut_report(cache))

# VTK: the cells (volume fraction, kind, rule, flags, face fractions, ...) and the reconstructed
# interface (`generate_mesh(cache, grid)`), each triangle tagged with its patch.
outdir = joinpath(@__DIR__, "outputs")
mkpath(outdir)
files = CutCellMethods.write_cache_vtk(joinpath(outdir, "hull"), cache, grid)
println("wrote ", join(files, ", "))

# ==================================================================================
# PLIC and marching cubes on the same grid, for comparison

# Neither reads the triangles or their patches: both cut the hull's signed distance, which
# SDFLibrary's `SDFMesh` evaluates from the mesh. PLIC fits a plane in each cell to the distance at
# its centre, so its surface is disconnected shards; marching cubes contours the distance sampled at
# the grid nodes, so it cuts the corners off the chine, transom and keel.
geo = SDFMesh(mesh)

plic = allocate_cache(grid, PLICCutCell())
@time update_cache!(plic, geo, grid)
mc = allocate_cache(grid, MarchingCubesCutCell())
@time update_cache!(mc, geo, grid)

# The hull's volume by each method, against the mesh's own.
V = CutCellMethods.cut_report(cache).mesh_volume
for (name, c) in (("tri clipping", cache), ("PLIC", plic), ("marching cubes", mc))
    Vc = sum(1 .- CutCellMethods.volume_fractions(c)) * prod(grid.d)
    println(rpad(name, 16), round(Vc; digits=6), " ft³, relative error ", round((Vc - V) / V; sigdigits=3))
end

# VTK: each method's surface, next to hull_surface.vtu, and one cell file holding the three volume
# fractions and their differences from tri clipping.
MeshLibrary.write_vtk(generate_mesh(plic, grid), joinpath(outdir, "hull_plic_surface"))
MeshLibrary.write_vtk(generate_mesh(mc, grid), joinpath(outdir, "hull_mc_surface"))
vf_tri, vf_plic, vf_mc = CutCellMethods.volume_fractions.((cache, plic, mc))
CartesianMeshes.write_vtk(joinpath(outdir, "hull_compare"), grid;
                          cell_data=Dict("volume_fraction_tri" => vf_tri,
                                         "volume_fraction_plic" => vf_plic,
                                         "volume_fraction_mc" => vf_mc,
                                         "plic_minus_tri" => vf_plic .- vf_tri,
                                         "mc_minus_tri" => vf_mc .- vf_tri))
