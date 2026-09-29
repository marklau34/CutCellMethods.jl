using MeshLibrary
using SDFLibrary
using CutCellMethods
using CartesianMeshes
using PAGE
using Plots

outdir = joinpath(@__DIR__, "outputs")

# 2D Mesh
airfoil_2412 = AirfoilNACA4(0.02, 0.4, 0.12, :naca2412)
mesh = Mesh(PAGE.generate_mesh(airfoil_2412, CosineFull(), 100))
mesh.elements[end] = Line(mesh.elements[end][1], mesh.elements[1][1]) # seal trailing edge
deleteat!(mesh.nodes, length(mesh.nodes))

# add element sets
top = MeshElementSet("top", collect(1:99))
bottom = MeshElementSet("bottom", collect(100:198))
mesh = Mesh(mesh.nodes, mesh.elements, element_sets=[top,bottom])

p = plot(Mesh(mesh.nodes, mesh.elements[mesh.elemset[1].elems]), color=:blue)
plot!(Mesh(mesh.nodes, mesh.elements[mesh.elemset[2].elems]), color=:red)
display(p)

# Converte mesh into SDF
geo = SDFMesh(mesh, cache=LineGrid)
write_vtk(geo, joinpath(outdir, "airfoil_surface"); N=100)
write_vtk(mesh, joinpath(outdir, "airfoil_mesh"))

# Marching cubes extraction
ms = MarchingSquaresCutCell()
grid = bounding_grid(geo; N=100)
# `generate_mesh` is qualified because PAGE exports one too (used above for the airfoil itself).
write_vtk(MeshLibrary.generate_mesh(geo, grid, ms), joinpath(outdir, "airfoil_contour"))

# ==================================================================================
# Polyline clipping: exact cut cells straight from the line mesh, no SDF

# A uniform grid of square cells around the airfoil, padded so the body stays clear of the grid's
# edge -- the cut needs at least one clear cell between the body and every side.
h = 0.005 # cell size, chords
pad = 4h
lo = reduce((a, b) -> min.(a, b), mesh.nodes.coord) .- pad
hi = reduce((a, b) -> max.(a, b), mesh.nodes.coord) .+ pad
pgrid = CartesianGrid(lo, Tuple(ceil.(Int, (hi - lo) ./ h)), SVector(h, h))

# One cache for the grid, updated from the mesh: closed loops of `Line`s wound counter-clockwise,
# so the solid is on the left of every element. A moving airfoil is the same call with moved nodes.
pc = PolylineClippingCutCell()
cache = allocate_cache(pgrid, pc)
update_cache!(cache, mesh, pgrid)
println(CutCellMethods.cut_cell_report(cache.cells))

# The walls the cut used, as a line mesh wound like the airfoil, so its normals point into the
# fluid. (`CutCellMethods.interface_mesh(cache, pgrid)` reads the same walls off the cache without
# cutting again.)
walls = MeshLibrary.generate_mesh(mesh, pgrid, pc)
write_vtk(walls, joinpath(outdir, "airfoil_walls"); write_normals=true)

# Everything the cache holds, for ParaView: cells, walls, each fluid region's loop, edge apertures.
CutCellMethods.write_cache_vtk(joinpath(outdir, "airfoil_polyline"), cache, pgrid)