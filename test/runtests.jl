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
    # `cut_cell_moments` carries three method-tagged families alongside the untagged ones, and two
    # overloads differing only in an untyped argument are an ambiguity raised at the CALL -- on
    # whichever path happens to reach it first -- rather than at the definition.
    @testset "no method ambiguities" begin
        @test isempty(Test.detect_ambiguities(CutCellMethods))
    end

    include("test_moments.jl")
    include("test_mc_moments.jl")
    include("test_plic.jl")
    include("test_bounded.jl")
    include("test_cache.jl")
end
