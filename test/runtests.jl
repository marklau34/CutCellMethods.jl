using Test
using LinearAlgebra
using StaticArrays
using Adapt
using MeshLibrary
using CartesianMeshes
using SDFLibrary
using CutCellMethods
using CUDA

const HAS_GPU = CUDA.functional()
HAS_GPU || @warn "no CUDA device: the device halves of the moment tests are skipped"

@testset verbose = true "CutCellMethods.jl" begin
    # Cheap, and it guards a class of mistake the rest of the suite only catches by luck.
    # `cell_values` and `update_cache!` mix typed methods with ones taking an untyped field, and
    # two overloads differing only in an untyped argument are an ambiguity raised at the CALL -- on
    # whichever path happens to reach it first -- rather than at the definition.
    @testset "no method ambiguities" begin
        @test isempty(Test.detect_ambiguities(CutCellMethods))
    end

    include("test_moments.jl")
    include("test_mc_moments.jl")
    include("test_plic.jl")
    include("test_bounded.jl")
    include("test_cache.jl")

    # Tri clipping, by phase: each gate passes before the next phase builds on it.
    include("test_tri_clipper.jl")    # P1: the clippers
    include("test_tri_topology.jl")   # P2: mesh validation and patches
    include("test_tri_cache.jl")      # P3: the cache, binning and classification
    include("test_tri_cut.jl")        # P4: fit groups, the Boolean rule, volumes
    include("test_tri_faces.jl")      # P5: shared faces, closure, the interface by patch
    include("test_tri_surface.jl")    # P6: the reconstructed surface, the corner line, VTK

    # Polyline clipping, by phase.
    include("test_poly_topology.jl")  # P1: loops and mesh validation
    include("test_poly_crossings.jl") # P1: the exact predicates
    include("test_poly_cache.jl")     # P2: crossings, node and cell states, the cut list
    include("test_poly_cut.jl")       # P3: regions, apertures, walls, islands, slivers, the sweep
    include("test_poly_output.jl")    # P4: allocation, threads, GPU, walls, loops, VTK
end
