using LinearAlgebra
using StaticArrays
import Gmsh: gmsh

stp_name = joinpath(@__DIR__, "GPPH8_Official.step")
save_name = joinpath(@__DIR__,  "GPPH8_Official")
show_gmsh = true

sizefac = 1.0

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
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.option.setNumber("Mesh.MeshSizeFactor", sizefac)
gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 0)
gmsh.option.setNumber("Mesh.Smoothing", 10)

# ==================================================================================

# Sync the CAD engine
gmsh.model.geo.synchronize()
gmsh.model.occ.synchronize()

# ==================================================================================
# Physical groups

# Generate mesh
gmsh.model.mesh.generate(1)
gmsh.model.mesh.generate(2)

# Recombine into quads
# gmsh.model.mesh.recombine()

# Visualize
if show_gmsh && !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.write(save_name*".stl")

gmsh.finalize()