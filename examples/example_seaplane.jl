using MeshLibrary
using CartesianMeshes
using CutCellMethods
using SDFLibrary
using StaticArrays
import GLMakie

# Gmsh msh file
element_sets = ["surface_$i" for i in 1:39]
mesh = load_mesh(joinpath(@__DIR__, "geometry", "AMRA_Subscale_v2_fuselage.inp"), elem_types=[:Tri3,], element_sets=element_sets)

# ==================================================================================
# Cut cells

# A uniform grid around the fuselage's bounding box, padded by a few cells on every side: the cut
# needs the body clear of the grid's edge. The fuselage is 0.83 ft across, so 33 cells span it.
h = 0.025 # cell size, ft
pad = 4h
lo = reduce((a, b) -> min.(a, b), mesh.nodes.coord) .- pad
hi = reduce((a, b) -> max.(a, b), mesh.nodes.coord) .+ pad
grid = CartesianGrid(lo, Tuple(ceil.(Int, (hi - lo) ./ h)), SVector(h, h, h))

# One cache for the grid, updated from the fuselage. Each element set is a patch the cut fits one
# plane to per cell.
cache = allocate_cache(grid, TriClippingCutCell())
@time update_cache!(cache, mesh, grid)
display(CutCellMethods.cut_report(cache))

# VTK: the cells (volume fraction, kind, rule, flags, face fractions, ...) and the reconstructed
outdir = joinpath(@__DIR__, "outputs")
mkpath(outdir)
clipped_surface = generate_mesh(cache, grid)
files = CutCellMethods.write_cache_vtk(joinpath(outdir, "seaplane"), cache, grid)
println("wrote ", join(files, ", "))

# ==================================================================================
# Marching cubes surface on the same grid, for comparison

# Marching cubes contours the fuselage's signed distance, which SDFLibrary's `SDFMesh` evaluates
# from the mesh, sampled at the grid nodes by the cache's update. Its surface, marched from those
# samples, is watertight, where the tri clipping one above is per-cell facets, but it cuts the
# corners off every crease.
geo = SDFMesh(mesh)
mc = allocate_cache(grid, MarchingCubesCutCell())
@time update_cache!(mc, geo, grid)
@time mc_surface = generate_mesh(mc, grid)
MeshLibrary.write_vtk(mc_surface, joinpath(outdir, "seaplane_mc_surface"))

# ==================================================================================
# PLIC surface on the same grid, for comparison

# PLIC fits a plane in each cell to the same signed distance at the cell's centre, and its surface is
# those fits, clipped. Like tri clipping, its surface is per-cell facets, but it has one plane per
# cell, so it cannot hold a crease inside a cell.
plic = allocate_cache(grid, PLICCutCell())
@time update_cache!(plic, geo, grid)
@time plic_surface = generate_mesh(plic, grid)
MeshLibrary.write_vtk(plic_surface, joinpath(outdir, "seaplane_plic_surface"))