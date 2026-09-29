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