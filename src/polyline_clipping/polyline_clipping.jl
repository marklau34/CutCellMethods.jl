# =====================================
# Polyline clipping: exact 2D cut cells from closed polylines
#
# The body is one or more closed polylines -- a `Mesh{2}` of `Line` elements, each loop wound
# counter-clockwise, so the solid is on the left of every element and MeshLibrary's element normal
# (to the right) points into the fluid. Every cut quantity is built from the crossings of those
# elements with the grid lines, and a crossing depends only on the element and the line's value, so
# the two cells sharing an edge read the same numbers and agree on it bitwise with no consistency
# pass. In each cut cell the fluid is recovered by walking the cell boundary counter-clockwise and
# turning back along the polyline wherever the boundary runs into the solid; each closed walk is one
# fluid *region*. The result is exact for the polygon: sharp corners, trailing edges thinner than a
# cell, and cells a thin body splits in two are all reproduced rather than smoothed.
#
# The construction is the brief's (`Brief 2D cut cells by exact boundary walking.md`), without its
# pose (the caller passes the mesh in grid coordinates) and with its closure formula's sign
# corrected: with boundary normals into the fluid, a region's closure is
# `sum_e n_e l_e - sum_s n_s L_s = 0`.
#
# Every combinatorial decision -- which lines an element crosses, which edge a crossing lands on,
# the order of crossings along an edge, which cell a vertex is in -- is made on the host with exact
# orientation predicates against one symbolic perturbation of the grid (`predicates.jl`), so the
# decisions are facts about one arrangement and cannot contradict each other. The kernels only do
# arithmetic on what the host decided.

"""
    PolylineClippingCutCell(; drop_area=1e-14, closure_tol=1e-13, validate=:topology)

Exact cut cells on a 2D grid from closed polylines: a `Mesh{2}` of `Line` elements whose loops are
wound counter-clockwise (solid on the left of each element). Exact for the polygon, including sharp
corners and trailing edges thinner than a cell, and a cell a thin body splits reports each fluid
region on its own.

- `drop_area` -- in units of `h^2`: a fluid region smaller than this is a sliver. It is dropped, and
  its open edge closed on both sides so the wall moves to the fluid neighbour.
- `closure_tol` -- in units of `h`: a region whose closure over its actual boundary segments is
  worse than this is flagged.
- `validate` -- `:topology` checks the mesh (winding, intersections, nesting) when its connectivity
  changes; `:always` rechecks the geometric conditions on every update, for bodies that move
  relative to each other (a deflecting flap).

`h = minimum(grid.d)`. The method is used through its cache; see [`allocate_cache`](@ref).
"""
struct PolylineClippingCutCell <: AbstractCutCellMethod
    drop_area::Float64
    closure_tol::Float64
    validate::Symbol
end

function PolylineClippingCutCell(; drop_area::Real=1e-14, closure_tol::Real=1e-13,
                                 validate::Symbol=:topology)
    drop_area >= 0 || throw(ArgumentError("drop_area must be non-negative, got $drop_area"))
    closure_tol > 0 || throw(ArgumentError("closure_tol must be positive, got $closure_tol"))
    validate in (:topology, :always) || throw(ArgumentError(
        "validate must be :topology or :always, got :$validate"))
    return PolylineClippingCutCell(Float64(drop_area), Float64(closure_tol), validate)
end

"""
    PolylineTols{T}

The method's tolerances made dimensional for one grid and in its element type -- the isbits form
the kernels take. A tolerance below a few ulps of `T` does nothing, so each is floored there.
"""
struct PolylineTols{T}
    h::T            # smallest cell size
    drop_area::T    # a region below this area is a sliver
    closure_tol::T  # a region closure residual above this is flagged
end

function PolylineTols(m::PolylineClippingCutCell, grid::CartesianGrid{2,T}) where {T}
    h = minimum(grid.d)
    e = eps(T)
    return PolylineTols{T}(T(h), T(max(m.drop_area, 8e^2)) * h^2, T(max(m.closure_tol, 8e)) * h)
end
