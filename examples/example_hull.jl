using MeshLibrary
import GLMakie

# Gmsh msh file
element_sets = ["surface_$i" for i in 1:15]
mesh = load_mesh(joinpath(@__DIR__, "geometry", "GPPH8_Official.inp"), elem_types=[:Tri3,], element_sets=element_sets)
viz(mesh, showsegments = true)

# Bottom right hull
# surface_mesh = Mesh(mesh.nodes, mesh.elements[mesh.elemset[2].elems])
# viz(surface_mesh, showsegments = true)