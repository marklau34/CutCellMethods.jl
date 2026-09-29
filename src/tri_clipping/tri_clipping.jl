# =====================================
# Tri clipping: cut cells from a segmented triangle surface
#
# The body is a watertight, outward-oriented triangle mesh whose triangles are grouped into smooth
# *patches* by the caller (`mesh.elemset`, one `MeshElementSet` per patch). In every control volume
# the surface crosses, each patch present is replaced by one plane -- its area-weighted fit over the
# patch's triangles clipped to that CV, or the patch's own global plane when it is planar -- and the
# CV is clipped against those planes by a small convex-polyhedron clipper. Where two or more patches
# meet in a CV the solid is the intersection of their half-spaces (a convex crease, like a transom
# or a chine) or the fluid is (a concave one), so a planar corner comes out exact rather than
# chamfered the way a single level set or a single plane per cell would leave it.
#
# With `n_int` out of the body, as everywhere in this package, the interface vector area is
# `+sum_k A_k n_k`, which is what `interface_normal_area` forms from the stored face fractions.
#
# Unlike the other methods this one exists only as a cache: a cell's apertures are resolved from
# its neighbours' fits, and an uncut cell is classified by a ray cast along its grid row, so there
# is no per-cell entry point. See `cache.jl` for the passes.

"""
    TriClippingCutCell(; margin=0.25, sliver_area=1e-6, snap_tol=1e-10, drop_tol=1e-14,
                         planar_tol=1e-6, tangent_angle=1.0, planar_patches=String[],
                         aperture_snap=0.0)

Cut cells from a watertight triangle mesh whose triangles the caller has grouped into smooth
patches, one `MeshElementSet` per patch in `mesh.elemset`, handed to the cache as `SDFMesh(mesh)`:
the cache takes the `SDFMesh` and nothing else, and reads the patches from its per-element set
labels. Planar corners between patches -- a transom meeting the bottom, a chine -- are reproduced
exactly; curved patches to second order.

An `SDFMesh` keeps set membership but not the set names, so the cut sees each set named by its index
in the original `mesh.elemset`, `"1"`, `"2"`, ...: `planar_patches` names them that way.

Every tolerance is relative to the grid's smallest cell size `h = minimum(grid.d)`:

- `margin` -- the sliver-refit margin `δ`, in units of `h`: a patch with less than `sliver_area`
  of area in a cell is refit over the cell grown by `δ` on every side. Triangles are binned with the
  same margin.
- `sliver_area` -- in units of `h^2`: below this a patch's piece of a cell is a sliver.
- `snap_tol` -- in units of `h`: a vertex this close to a cutting plane is taken to be on it.
- `drop_tol` -- in units of `h^2`: a clipped area below this is zero. A cell whose clipped triangle
  area is not above it is uncut.
- `planar_tol` -- in units of `h`: a patch whose vertices all lie within this of its best-fit plane
  is cut by that one global plane everywhere, which is what makes its corners exact.
- `tangent_angle` -- degrees: two patches meeting at a smaller dihedral than this are one smooth
  surface split in two (a CAD seam), and are fitted as one.
- `planar_patches` -- names of element sets to treat as planar whatever their measured planarity.
- `aperture_snap` -- a face fraction within this of `0` or `1` is snapped there before the
  interface is formed, so the snap cannot break the closure. `0` disables it.

The method is used through its cache; see [`allocate_cache`](@ref).
"""
struct TriClippingCutCell <: AbstractCutCellMethod
    margin::Float64
    sliver_area::Float64
    snap_tol::Float64
    drop_tol::Float64
    planar_tol::Float64
    tangent_angle::Float64
    planar_patches::Vector{String}
    aperture_snap::Float64
end

function TriClippingCutCell(; margin::Real=0.25, sliver_area::Real=1e-6, snap_tol::Real=1e-10,
                            drop_tol::Real=1e-14, planar_tol::Real=1e-6, tangent_angle::Real=1.0,
                            planar_patches=String[], aperture_snap::Real=0.0)
    margin >= 0 || throw(ArgumentError("margin must be non-negative, got $margin"))
    sliver_area >= 0 || throw(ArgumentError("sliver_area must be non-negative, got $sliver_area"))
    snap_tol >= 0 || throw(ArgumentError("snap_tol must be non-negative, got $snap_tol"))
    drop_tol >= 0 || throw(ArgumentError("drop_tol must be non-negative, got $drop_tol"))
    planar_tol >= 0 || throw(ArgumentError("planar_tol must be non-negative, got $planar_tol"))
    0 <= tangent_angle < 90 || throw(ArgumentError("tangent_angle must be in [0, 90) degrees, got $tangent_angle"))
    0 <= aperture_snap < 0.5 || throw(ArgumentError("aperture_snap must be in [0, 0.5), got $aperture_snap"))
    return TriClippingCutCell(Float64(margin), Float64(sliver_area), Float64(snap_tol),
                              Float64(drop_tol), Float64(planar_tol), Float64(tangent_angle),
                              String[string(p) for p in planar_patches], Float64(aperture_snap))
end

"""
    TriClipTols{T}

The method's tolerances made dimensional for one grid and in its element type -- the isbits form
the kernels take. Built once by [`allocate_cache`](@ref); a grid may move between updates but its
cell size may not, so these stay valid.

A relative tolerance below a few ulps of `T` cannot do anything, so each is floored there: in
`Float32` a `1e-10` snap is no snap at all, and the clipper's sign logic has to carry the
degenerate cases on its own.
"""
struct TriClipTols{T}
    h::T                # smallest cell size
    margin::SVector{3,T} # the refit and binning margin, per axis
    snap::T             # a plane distance below this is zero
    drop_area::T        # an area below this is zero
    sliver_area::T      # a patch piece below this is a sliver
    planar_dist::T      # planarity tolerance
    cos_tangent::T      # cos(tangent_angle): a pair dihedral cosine above this is tangent
    aperture_snap::T
end

function TriClipTols(m::TriClippingCutCell, grid::CartesianGrid{3,T}) where {T}
    h = minimum(grid.d)
    e = eps(T)
    return TriClipTols{T}(T(h), T(m.margin) .* SVector{3,T}(grid.d),
                          T(max(m.snap_tol, 8e)) * h,
                          T(max(m.drop_tol, 8e^2)) * h^2,
                          T(m.sliver_area) * h^2,
                          T(max(m.planar_tol, 8e)) * h,
                          T(cosd(m.tangent_angle)),
                          T(m.aperture_snap))
end
