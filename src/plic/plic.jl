"""
    PLICCutCell()

The cut-cell reconstruction a VOF solver's PLIC step uses: one plane per cell, fitted from a single
`get_sdf` at that cell's centroid ([`cell_plane`](@ref)). A stateless tag, one of the
`AbstractCutCellMethod`s:

    method = PLICCutCell()

    cache = update_cache!(allocate_cache(grid, method), geo, grid)
    cache.cells                                        # every cell's CutCellData, apertures resolved
    cache.normals, cache.intercepts                    # every cell's plane
    surface = generate_mesh(cache, grid)               # the interface, as a mesh

**The cache is the only way to PLIC data** -- there is no per-cell `cut_cell_moments` form.
[`PLICCutCellCache`](@ref) holds every cell's plane and, in the same [`CutCellData`](@ref) every
other cache holds, its fluid fraction and resolved per-face apertures; [`generate_mesh`](@ref) clips
the stored planes into a surface. It is also exactly what [`calc_volume`](@ref) integrates, so a
volume measured there and a surface drawn here are one reconstruction.

**The grid must be isotropic**, which the cache checks. Each cell's own plane gives one-sided
apertures, which the cache resolves; see `apertures.jl`'s header.

Each plane is fitted from its own cell alone, so two cells sharing a face place their endpoints at
different points on it: the surface is a field of disconnected shards rather than a stitched
contour -- the wrong thing for a cut-cell flux balance. For that, reconstruct nodally with
[`MarchingSquaresCutCell`](@ref) or [`MarchingCubesCutCell`](@ref), where a shared face is
interpolated from shared corner values and is watertight by construction.
"""
struct PLICCutCell <: AbstractCutCellMethod
end

"""
    cell_plane(geo, grid::CartesianGrid{D,T}, idx) -> (; normal, intercept, fraction, is_valid)

The PLIC plane one cell of `grid` reconstructs from `geo`, from one `get_sdf` at the cell's
centroid, already guarded against a degenerate fit so every field is finite and directly storable:

- `normal` -- the surface normal at the centroid, or zero where the fit is degenerate.
- `intercept` -- the matching `a` of the plane `n̂ ⋅ ξ = a` in unit-cell coordinates `ξ ∈ [0,1]^D`,
  or zero where the fit is degenerate. `φ(ξ) ≈ dist + dx * n̂ ⋅ (ξ - 1/2)` puts the inside half-space
  at `n̂ ⋅ ξ <= sum(n̂)/2 - dist/dx`, the form `CartesianMeshes.get_volume_fraction` and
  [`generate_mesh`](@ref) want. Cells the surface misses need no branch: `a` falls outside `[0,1]`
  and the fraction clamps to empty or full.
- `fraction` -- the solid fraction the plane cuts off, the integrand of [`calc_volume`](@ref) and
  the sense a VOF caller expects; the cache's `cells.volume_fraction` stores its complement, the
  fluid fraction.
- `is_valid` -- `false` where the fit is degenerate.

# The degenerate fit

Where the distance field is not differentiable the gradient cannot be normalized, in either of the
two forms `get_sdf` produces:

- a `NaN` normal, from normalizing a zero gradient -- a centroid landing exactly on a body's medial
  axis (dead centre of an `SDFCircle`, say, which a symmetric grid with an odd cell count does hit).
- a zero normal, where a geometry resolves the same degeneracy componentwise (`SDFRectangle` on its
  centreline, where `sign(0)` is `0`). `get_volume_fraction` then divides by
  `sum(abs, normal) == 0`, usually landing correctly via its clamp branch but yielding `NaN` when
  the centroid also sits exactly on the surface.

Either way the point is locally as far from the surface as it can be, so it is never in a cell the
surface actually cuts, and its centroid's sign is the whole answer. `fraction` is answered from that
sign, `is_valid` reports which case it was, and `normal`/`intercept` are zeroed so a caller can store
them without a `NaN` reaching an array later reduced over or written to VTK. The zeroing is not a
second encoding of the flag -- branch on `is_valid`.

Assumes `grid` is isotropic, since that single `dx` converts the distance into unit-cell
coordinates; the caller checks once ([`calc_volume`](@ref) and the PLIC cache both do). Unexported
-- [`PLICCutCellCache`](@ref)`.normals`/`.intercepts` are the supported way to a cell's plane.
"""
@inline function cell_plane(geo::AbstractSDFGeometry, grid::CartesianGrid{D,T},
                            idx::CartesianIndex{D}) where {D,T}
    d, n = get_sdf(geo, CartesianMeshes.get_elem_centroid(grid, idx))
    dist = T(d)
    fitted = SVector{D,T}(n)
    a = T(0.5) * sum(fitted) - dist / grid.d[1]

    # Note the guarded arm reads `dist <= 0` rather than `dist > 0` with results swapped: they
    # differ when `dist` is `NaN` (both comparisons false), and this branch keeps such a cell
    # reading as empty.
    if isnan(a) || iszero(fitted)
        return (normal = zero(SVector{D,T}), intercept = zero(T),
                fraction = dist <= zero(T) ? one(T) : zero(T), is_valid = false)
    end
    return (normal = fitted, intercept = a,
            fraction = CartesianMeshes.get_volume_fraction(fitted, a), is_valid = true)
end
