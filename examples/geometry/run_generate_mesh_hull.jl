using PAGE
import Gmsh: gmsh
using GmshUtilities
using MeshLibrary

stp_name = joinpath(@__DIR__, "gpph_clean.step")
save_name = joinpath(@__DIR__,  "gpph_clean")
show_gmsh = true

sizefac = 0.5

# Initialize setup
gmsh_config = GmshConfig()
set_mesh_algorithm(:delaunay)
if !Bool(gmsh.is_initialized())
    gmsh.initialize()
end
# Units to ft
gmsh.option.setString("Geometry.OCCTargetUnit", "FT")

# read in the geometry
gmsh.merge(abspath(stp_name))

# Visibility settings
gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
gmsh.option.setNumber("Geometry.Surfaces", 1)
gmsh.option.setNumber("General.NumThreads", 8)

# Set mesh options
# https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-options
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.option.setNumber("Mesh.MeshSizeFactor", sizefac)
gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 0)
gmsh.option.setNumber("Mesh.Smoothing", 10)

# Physical groups
for i in 1:8
    add_params!(gmsh_config, GmshPhysicalGroup(name="surface_$i", ndims=2, tags=[i,]))
end

# ===============================================
# Write all params to gmsh
write_config(gmsh_config)

# Sync the CAD engine
gmsh.model.geo.synchronize()
gmsh.model.occ.synchronize()

# =======================

# Generate mesh
gmsh.model.mesh.generate(1)
gmsh.model.mesh.generate(2)

# Recombine into quads
# gmsh.model.mesh.recombine()

# Visualize
if show_gmsh && !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.write(save_name * ".inp")

gmsh.finalize()