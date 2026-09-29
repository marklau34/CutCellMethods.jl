module CutCellMethods

# =====================================
# Cut-cell methods
export AbstractCutCellMethod, MarchingSquaresCutCell, MarchingCubesCutCell, PLICCutCell,
       TriClippingCutCell, PolylineClippingCutCell
export CutCellData
export allocate_cache, update_cache!

using CartesianMeshes
using MeshLibrary
using StaticArrays
using LinearAlgebra
using Reexport
using Adapt
using KernelAbstractions
using Polyester
using MarchingCubes
using StructArrays: StructArray

# Not exported by CartesianMeshes -- the direction algebra (`1 = -x`, `2 = +x`, `3 = -y`, ...) is
# internal there, but the moment layer needs it to orient an open face's normal and to pick out the
# axis a Cartesian face is normal to.
using CartesianMeshes: direction_axis, direction_sign

# The bodies being cut are SDFLibrary.jl's: the nodal methods sample corners through its
# `sdf_value` (which also takes any callable `x -> phi`), and the PLIC fit reads `get_sdf` at a
# cell centroid. The clipping methods cut an `SDFMesh`, whose element sets it carries as a
# per-element label that survives a move to the GPU. Named explicitly rather than `using
# SDFLibrary` wholesale, so its exports do not land in this namespace.
using SDFLibrary: SDFLibrary, AbstractSDFGeometry, EmptyGeo, get_sdf, sdf_value, SDFMesh

# Exact orientation predicates for the polyline clipper's combinatorics, which run on the host.
# Qualified at every use (`ExactPredicates.orient`), so nothing of it lands in this namespace.
using ExactPredicates: ExactPredicates

# The surfaces are methods of `MeshLibrary.generate_mesh`, so it is re-exported: `using
# CutCellMethods` alone is enough to draw one. SDFLibrary.jl re-exports the same function, so
# loading both is no clash.
@reexport using MeshLibrary: generate_mesh

"""
    AbstractCutCellMethod

The supertype of every cut-cell reconstruction. Each method is a stateless tag that selects a
construction through dispatch: [`allocate_cache`](@ref)`(domain, method)` for storage,
[`update_cache!`](@ref) to fill it, and `generate_mesh(cache, domain)` for the matching surface --
`domain` a grid, or an `AdaptiveMesh` for [`MarchingSquaresCutCell`](@ref).
"""
abstract type AbstractCutCellMethod end
abstract type AbstractCutCellCache  end

const DEFAULT_WORKGROUP = 256
# The cell count below which a host loop runs serially: a small grid is not worth the `@batch` launch.
const THREAD_FLOOR = 4096
const CELL_INSIDE  = Int8(0)
const CELL_OUTSIDE = Int8(1)
const CELL_CUT     = Int8(2)

include("cut_cell.jl")
include("cache.jl")
include("plic/include.jl")
include("marching_squares/include.jl")
include("marching_cubes/include.jl")
include("tri_clipping/include.jl")
include("polyline_clipping/include.jl")
include("volume.jl")

end # module CutCellMethods
