"""
    PLICCutCell()

The cut-cell reconstruction a VOF solver's PLIC step uses: one plane per cell, fitted from a single
`get_sdf` at that cell's centroid ([`cell_plane`](@ref)). A stateless tag, one of the
`AbstractCutCellMethod`s:

    method = PLICCutCell()

    surface = generate_mesh(geo, grid, method)         # the interface, as a mesh
    d = cut_cell_moments(method, grid, geo, ci)        # ...or one cell's fit, fraction and apertures

One fit read two ways: [`cut_cell_moments`](@ref) gives a cell's plane, its fluid fraction and its
per-face apertures, and [`generate_mesh`](@ref) clips the same fit into a surface. It is
also exactly what [`calc_volume`](@ref) integrates, so a volume measured there and a surface
drawn here are one reconstruction rather than two that nearly agree.

**The grid must be isotropic** -- [`generate_mesh`](@ref) checks it, the per-cell form
assumes it -- and **the apertures come back one-sided**; see [`PLICCutCellData`](@ref)`.face_area`.

Each plane is fitted from its own cell alone, so two cells sharing a face place their endpoints at
different points on it: the surface is a field of disconnected shards rather than a stitched
contour. That is an honest picture of what a VOF solver believes, and the wrong thing to hand a
cut-cell flux balance -- for that, reconstruct nodally with [`MarchingSquaresCutCell`](@ref) or
[`MarchingCubesCutCell`](@ref), where a shared face is interpolated from shared corner values and
is watertight by construction.
"""
struct PLICCutCell <: AbstractCutCellMethod
end

"""
    PLICCutCellData{D,T,NF}

Everything a [`PLICCutCell`](@ref) reconstruction knows about **one** cell, with `NF == 2D` faces.
A consumer wanting a field over the grid allocates an array of these and fills it from
[`cut_cell_moments`](@ref).

- `normal` -- the fitted unit normal.
- `intercept` -- the matching intercept `a`, in unit-cell coordinates ([`cell_plane`](@ref)).
- `volume_fraction` -- the **fluid** fraction, the part of the cell outside the body: `1` where the
  body misses the cell entirely, `0` where the cell is wholly inside it, strictly between in a cut
  cell. Same sense as [`CutCellData`](@ref)`.volume_fraction`, and the *complement* of
  [`cell_plane`](@ref)`.fraction` and `CartesianMeshes.get_volume_fraction`, which measure the solid
  half-space -- so re-deriving the fraction from `normal`/`intercept` gives the complement of what
  is stored, not a copy of it.
- `is_valid` -- `false` where the centroid fit is degenerate ([`cell_plane`](@ref), which is where
  that is decided). `volume_fraction` is still right in such a cell, and the stored plane is zeroed
  so the data stays finite. **This flag is the authority**, not `iszero(normal)`.
- `face_area` -- the open measure of each of this cell's own faces, indexed by direction in
  `CartesianMeshes`' numbering (`1 = -x`, `2 = +x`, `3 = -y`, ...), in raw physical units: a length
  in 2D, an area in 3D. Read it through [`face_area_of`](@ref)/[`face_fraction_of`](@ref).

  **It is ONE-SIDED**, which is a property of the reconstruction rather than an oversight: it is
  this cell's own plane evaluated on its own faces, with no neighbour consulted, so the two cells
  sharing an interior face report different numbers wherever their centroid fits disagree.
  Combining them is the caller's, and it is the mean of the two -- see `apertures.jl`'s header.
  Boundary faces have no second candidate and are already final.
- `kind` -- [`CELL_INSIDE`](@ref), `CELL_OUTSIDE` or `CELL_CUT`: `volume_fraction`'s three-way
  reading, `INSIDE` at `0`, `OUTSIDE` at `1`, `CUT` strictly between. Same `Int8` encoding and
  polarity as [`CutCellData`](@ref)`.kind`, and the same set [`generate_mesh`](@ref)
  draws at `tol = 0`, so a consumer branches on one integer compare instead of two float ones.

Isbits and fixed-size, so it passes into a kernel by value and lives in a plain device array.
"""
struct PLICCutCellData{D,T,NF}
    normal::SVector{D,T}
    intercept::T
    volume_fraction::T
    is_valid::Bool
    face_area::SVector{NF,T}
    kind::Int8
end

@inline Base.ndims(::PLICCutCellData{D}) where {D} = D
@inline Base.eltype(::PLICCutCellData{D,T}) where {D,T} = T
@inline Base.ndims(::Type{<:PLICCutCellData{D}}) where {D} = D
@inline Base.eltype(::Type{<:PLICCutCellData{D,T}}) where {D,T} = T

"""
    plic_fit(grid::CartesianGrid{D,T}, geo, ci) -> (; normal, intercept, frac, is_valid, kind)

One cell's plane fit in this package's storage conventions but **without** the apertures -- the
shared core of [`cut_cell_moments`](@ref)`(::PLICCutCell, ...)` and `_cut_planes` (`surface.jl`),
which screens a grid for the cells its PLIC surface is drawn in.

[`cell_plane`](@ref) at the cell's centroid, with `kind` classified from the fraction while it is
still in a register. `frac` is the **fluid** fraction, the complement of `cell_plane`'s own solid
`fraction`; this is the only place in the package the two conventions meet. A degenerate fit is
reported in `is_valid` and comes back with a zeroed plane -- both decided by `cell_plane`, not here.

It returns the pieces rather than a [`PLICCutCellData`](@ref) because its two callers want
different amounts of work: meshing needs no apertures at all, and each is a Sutherland-Hodgman clip
in 3D. A half-filled struct with a placeholder `face_area` would make that field mean "not computed
yet" in one path and a real measure in another.
"""
@inline function plic_fit(grid::CartesianGrid{D,T}, geo, ci::CartesianIndex{D}) where {D,T}
    plane = cell_plane(geo, grid, ci)
    # `plane.fraction` measures the **solid** -- it is `calc_volume`'s integrand and the sense every
    # VOF caller expects, so it keeps that meaning -- while what is stored is the **fluid**,
    # matching `CutCellData.volume_fraction`.
    frac = one(T) - plane.fraction
    # Classified from the **stored** fraction rather than from the raw solid one, so that
    # `kind == CELL_CUT` and `generate_mesh`'s `0 < vol_frac < 1` stay exactly the same set.
    kind = iszero(frac) ? CELL_INSIDE : isone(frac) ? CELL_OUTSIDE : CELL_CUT
    return (normal=plane.normal, intercept=plane.intercept, frac=frac,
            is_valid=plane.is_valid, kind=kind)
end

"""
    cut_cell_moments(method::PLICCutCell, grid::CartesianGrid{D}, geo, ci) -> PLICCutCellData

Everything a [`PLICCutCell`](@ref) reconstruction knows about the single cell `ci` of `grid`: the
plane fitted at its centroid, the fluid fraction that plane cuts off, the open measure of each of
its `2D` faces, and its classification. One `get_sdf`, nothing allocated. `T` comes from
`grid`.

**This is the entry point** -- the [`PLICCutCell`](@ref) method of the name every cut-cell
reconstruction answers to. A consumer builds whatever field it wants by looping or launching over
the cells and keeping exactly the arrays it needs.

Despite the shared name it returns a [`PLICCutCellData`](@ref), not a [`CutCellData`](@ref): a
centroid plane fit has no outside centroid and no interface centroid to report. The two share the
`volume_fraction`/`face_area`/`kind` conventions and the whole accessor layer, so one consumer
reads either -- but a caller that needs moments proper wants a nodal method.

**`face_area` comes back one-sided**: this cell's own plane on its own faces, no neighbour read.
Resolving the two candidates on an interior face is the caller's, and it is their mean; see
[`PLICCutCellData`](@ref)`.face_area`.

**Assumes `grid` is isotropic** and does not check: this is per-cell code that has to stay
launchable, and a `throw` reached from device code is not something GPU codegen handles gracefully.
`generate_mesh(geo, grid, PLICCutCell())` checks, and is the only place that does.
"""
@inline function cut_cell_moments(::PLICCutCell, grid::CartesianGrid{D,T}, geo,
                                  ci::CartesianIndex{D}) where {D,T}
    f = plic_fit(grid, geo, ci)
    faces = cell_face_areas(f.normal, f.intercept, f.frac, f.is_valid, grid.d)
    return PLICCutCellData{D,T,2D}(f.normal, f.intercept, f.frac, f.is_valid, faces, f.kind)
end



"""
    cell_plane(geo, grid::CartesianGrid{D,T}, idx) -> (; normal, intercept, fraction, is_valid)

The PLIC plane one cell of `grid` reconstructs from `geo`, from one `get_sdf` at the cell's
centroid, **already guarded against a degenerate fit** so that every field is finite and directly
storable:

- `normal` -- the surface normal at the centroid, or zero where the fit is degenerate.
- `intercept` -- the matching `a` of the plane `n̂ ⋅ ξ = a` in unit-cell coordinates `ξ ∈ [0,1]^D`,
  or zero where the fit is degenerate. `φ(ξ) ≈ dist + dx * n̂ ⋅ (ξ - 1/2)` puts the inside half-space
  at `n̂ ⋅ ξ <= sum(n̂)/2 - dist/dx`, which is the form `CartesianMeshes.get_volume_fraction` and
  [`generate_mesh`](@ref) want. Cells the surface misses need no branch: `a` falls outside
  `[0,1]` and the fraction clamps to empty or full.
- `fraction` -- the **solid** fraction the plane cuts off. This is the integrand of [`calc_volume`](@ref)
  and the sense a VOF caller expects; [`PLICCutCellData`](@ref)`.volume_fraction`
  stores its complement, the fluid fraction.
- `is_valid` -- `false` where the fit is degenerate.

# The degenerate fit, decided here and nowhere else

Where the distance field is not differentiable the gradient cannot be normalized, in either of the
two forms `get_sdf` produces:

- a **`NaN`** normal, from normalizing a zero gradient -- a centroid landing exactly on a body's
  medial axis (dead centre of an `SDFCircle`, say, which a symmetric grid with an odd cell
  count does hit). One `NaN` fraction would poison a whole sum.
- a **zero** normal, where a geometry resolves the same degeneracy componentwise (`SDFRectangle` on
  its centreline, where `sign(0)` is `0`). `get_volume_fraction` then divides by
  `sum(abs, normal) == 0`, which usually still lands correctly via its clamp branch but yields
  `NaN` when the centroid also sits exactly on the surface.

Either way the point is locally as far from the surface as it can be, so it is never in a cell the
surface actually cuts, and **its centroid's sign is the whole answer**. `fraction` is answered from
that sign, `is_valid` reports which case it was, and `normal`/`intercept` are zeroed so a caller can
store them without a `NaN` reaching an array that is later reduced over or written to VTK. The
zeroing is not a second encoding of the flag -- branch on `is_valid`.

**Assumes `grid` is isotropic**, since that single `dx` converts the distance into unit-cell
coordinates; the caller checks once ([`calc_volume`](@ref) and `generate_mesh` both do).
Unexported -- [`cut_cell_moments`](@ref) is the supported way to a cell's fit, and this is the
primitive underneath it.
"""
@inline function cell_plane(geo::AbstractSDFGeometry, grid::CartesianGrid{D,T},
                            idx::CartesianIndex{D}) where {D,T}
    d, n = get_sdf(geo, CartesianMeshes.get_elem_centroid(grid, idx))
    dist = T(d)
    fitted = SVector{D,T}(n)
    a = T(0.5) * sum(fitted) - dist / grid.d[1]

    # One test, one place. Note the guarded arm reads `dist <= 0` rather than the friendlier
    # `dist > 0` with the results swapped: the two differ when `dist` itself is `NaN`, where both
    # comparisons are false, and this is the branch that keeps such a cell reading as *empty*.
    if isnan(a) || iszero(fitted)
        return (normal = zero(SVector{D,T}), intercept = zero(T),
                fraction = dist <= zero(T) ? one(T) : zero(T), is_valid = false)
    end
    return (normal = fitted, intercept = a,
            fraction = CartesianMeshes.get_volume_fraction(fitted, a), is_valid = true)
end
