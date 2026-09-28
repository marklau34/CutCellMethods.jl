# =====================================
# SDFLibrary.jl's `SDFBounded`, as a cut-cell reconstruction sees it.
#
# The constraint substitutes `fill_distance` for the child outside a band around its bounding box.
# `pad_distance` wider than a cell diagonal has to leave the cut cells -- and so the measured
# volume and the reconstructed contour -- bit-for-bit alone. SDFLibrary.jl's own suite covers the
# constraint itself.

using CutCellMethods: calc_volume

@testset "SDFBounded: the substituted value stays out of the cut cells" begin
    circle = SDFCircle(center=SVector(0.0, 0.0), radius=1.0)

    # Cell size 0.1, diagonal 0.14, pad 0.5: no cell the surface crosses can reach the jump, so
    # the measured area has to be the unconstrained one.
    grid = CartesianGrid(Float64; mincorner=(-2.0, -2.0), maxcorner=(2.0, 2.0), n=(40, 40))
    bounded = SDFBounded(circle; pad_distance=0.5)
    @test calc_volume(bounded, grid) ≈ calc_volume(circle, grid) rtol = 1e-12
    @test calc_volume(bounded, grid) ≈ π rtol = 1e-3

    # Same for the nodal reconstruction: the contour is the circle's, and the fill band never
    # produces a facet of its own.
    mesh = generate_mesh(bounded, grid, MarchingSquaresCutCell())
    @test length(mesh.nodes) == length(generate_mesh(circle, grid, MarchingSquaresCutCell()).nodes)
end
