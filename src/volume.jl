"""
    calc_volume(geo::AbstractSDFGeometry, grid::CartesianGrid{D,T}) -> T

The `D`-dimensional measure of the region where `get_sdf(geo, ...)` is negative, over `grid`'s own
bounds and in `geo`'s (i.e. `grid`'s) local frame -- a volume on a `CartesianGrid{3}`, the enclosed
area on a `CartesianGrid{2}`.

Only the part of `geo` inside `grid` is counted, so a grid that does not contain the body returns
the clipped measure; SDFLibrary.jl's `bounding_grid` sizes one that does.

Every cell contributes `cell_plane(geo, grid, ci).fraction * prod(grid.d)`: one `get_sdf` at the
cell centroid, its `(distance, normal)` taken as a plane through that cell and clipped analytically
([`cell_plane`](@ref)). Cells the surface misses fall out of the same expression, with no separate
branch. [`PLICCutCell`](@ref) fits the very same planes, so a volume measured here and a surface
drawn there are one reconstruction rather than two that nearly agree.

`grid` must be isotropic (square cells in 2D, cubic in 3D), which is checked up front: that single
`dx` is what converts a distance into unit-cell coordinates.

Exact for any cell the surface crosses as a plane, approximate where it curves or where a cell
holds an edge or corner of the body; either way the error lives only in the `O(h^(D-1))` band of
cut cells. A plane reconstruction contains a convex surface, so a convex body comes out
systematically slightly over-measured, but the measure varies smoothly as the body moves against
the grid, where counting whole cells would step by one cell at a time.

For an SDFLibrary.jl `SDFBounded` geometry, centroids beyond its pad band evaluate to
`fill_distance`, which is required to be positive, so those cells read as empty.
"""
function calc_volume(geo::AbstractSDFGeometry, grid::CartesianGrid{D,T}) where {D,T}
    if !CartesianMeshes.is_isotropic(grid)
        throw(ArgumentError("calc_volume needs an isotropic grid (equal cell size on every axis), got grid.d = $(Tuple(grid.d))"))
    end
    vol = zero(T)
    for ci in CartesianIndices(grid)
        # `.fraction` is the solid fraction, already guarded against a degenerate fit -- see
        # [`cell_plane`](@ref), which is where that decision is made.
        vol += cell_plane(geo, grid, ci).fraction
    end

    return vol * prod(grid.d)
end