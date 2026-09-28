# =====================================
# Cut-cell moments, from the nodal marching-squares reconstruction
#
# The reconstruction itself lives next door in `marching_squares.jl`: sample `phi` at a cell's four
# corners, interpolate each edge's zero crossing from its two corner values, and report the open
# part of every edge (`cell_boundary_walk`) plus the segments that bridge the gaps between them
# (`cell_gap_segments`). This file turns one cell's walk into a `CutCellData`, built on the
# *same* walk the contour `generate_mesh` draws comes from -- so the aperture a flux passes through
# and the surface a plot shows are one reconstruction rather than two that might disagree.
#
# Why nodal, restated because the moments are where it pays off: a per-cell fit gives cell A and its
# neighbour B two different planes through the face they share, anchored `O(h)` apart, so they
# compute different apertures for it and mass is manufactured at the face -- and for a consumer
# whose surface forces come out of the flux balance rather than a separate surface integration, that
# error lands directly on the force it was measuring. Nodally, both cells read the same corner values
# with the same floating-point expression, so their apertures agree bitwise with no communication and
# no consistency pass, which is what makes it safe to recompute one cell's geometry on demand at any
# level. `test/test_moments.jl` checks that as bitwise assertions rather than tolerances, because it
# either holds exactly or the reasoning is wrong somewhere.
#
# The 3D construction is `marching_cubes/moments.jl`, as methods on these same names producing this
# same `CutCellData` -- `{3,T,6}` instead of `{2,T,4}`, read through the same `volume_rule` /
# `face_rule` / `interface_rule` and satisfying the same exactly-zero `closure_residual`. Two thirds
# of it is this file's construction reused rather than reimplemented: each of the six cell faces is a
# 2D square, so `cell_boundary_walk` gives its aperture and open-face centroid on corner values the
# neighbour reads too, and the interface's vector area follows from the closure identity with no
# interface polygon at all. What genuinely needs marching cubes is the outside volume, the outside
# centroid and the interface centroid.
#
# One thing does *not* carry over: `cell_gap_segments`' unconditional "outside connected through the
# middle" resolution of an ambiguous cell. In 3D the interface facets come from the Lewiner tables,
# and a face clip that paired its crossings differently would leave the cell's polyhedron open, so
# the 3D face clip takes its pairing from the same decider those tables use. See that file's header.

# The centre of the full Cartesian face in `direction`, from the two lattice corners of the edge on
# it. It is the reported centroid of every face the walk does not split: a fully open face, where it
# carries full quadrature weight, and a closed one, where its weight is zero and it is only a finite
# placeholder.
#
# The open case is why it comes from the corners and not from `centre + h/2`. An outside cell and a
# cut neighbour must report the same point for the face they share, or the flux through it is
# evaluated at two different places: `centre + h/2` is not bitwise the neighbour's `centre - h/2`,
# while this is one sum of the same two lattice points for both cells (addition commutes), and is
# bitwise the midpoint the cut path takes for a fully open edge. The 3D face clip does the same.
@inline function _full_face_centre(nodes::SVector{4,SVector{2,T}}, direction::Integer) where {T}
    k = MS_DIRECTION_EDGE[direction]
    return @inbounds T(0.5) * (nodes[k] + nodes[_ms_next(k)])
end

# A face centroid `x` as it is stored: its one in-plane coordinate, as an offset from the cell's
# lower lattice corner (the normal coordinate is the face plane, and is not kept). Two cells sharing
# a face compute the same `x` and have the same in-plane corner coordinate, so they store the same
# offset -- and `face_centroid` adds the same number back for both.
@inline function _face_offset(x::SVector{2,T}, origin::SVector{2,T}, direction::Integer) where {T}
    p = 3 - direction_axis(direction)
    return @inbounds SVector{1,T}(x[p] - origin[p])
end

"""
    cut_cell_moments(nodes, phi, cellsize) -> CutCellData

The primitive: every moment of one cut cell, from the corner coordinates `nodes`, the corner values
`phi`, and the cell size.

`nodes` and `phi` are in [`MS_NODE_BITS`](@ref) order -- counter-clockwise from the lower corner --
which is what [`cell_nodes`](@ref) and [`cell_values`](@ref) return. That the signature takes *node
values* rather than a geometry object is the watertightness contract made explicit: the construction
cannot depend on anything a neighbour does not also see.

See [`cut_cell_moments`](@ref)`(method, domain, geo, cell)` for the entry point that samples `phi`
itself, which is the form a consumer builds its own field with.
"""
function cut_cell_moments(nodes::SVector{4,SVector{2,T}}, phi::SVector{4,T},
                          cellsize::SVector{2,T}) where {T}
    origin = @inbounds nodes[1]
    centre = origin + T(0.5) * cellsize
    vol_cell = prod(cellsize)
    zero2 = zero(SVector{2,T})

    noutside = 0
    for k in 1:4
        noutside += ifelse(@inbounds(phi[k]) >= zero(T), 1, 0)
    end

    # --- uncut cells, handled up front ------------------------------------
    #
    # Not merely a fast path: the general route computes the volume from a shoelace over the four
    # corners, which agrees with `prod(cellsize)` only to roundoff, and an uncut cell whose volume
    # fraction is 0.9999... rather than exactly 1 injects an `O(eps)` free-stream error into every
    # cell in the domain.
    if noutside == 4 || noutside == 0
        outside = noutside == 4
        return CutCellData{2,T,4,1}(
            outside ? CELL_OUTSIDE : CELL_INSIDE, false, outside ? one(T) : zero(T), centre,
            outside ? ones(SVector{4,T}) : zeros(SVector{4,T}),
            SVector{4,SVector{1,T}}(ntuple(Val(4)) do dir
                _face_offset(_full_face_centre(nodes, dir), origin, dir)
            end),
            centre)
    end

    # --- the boundary walk -------------------------------------------------
    #
    # `marching_squares.jl`'s, not a copy of it: the contour `generate_mesh` stitches comes out of
    # this same function on these same corner values, so the two cannot drift apart.
    fracs, present, seg_a, seg_b = cell_boundary_walk(nodes, phi)

    # --- open Cartesian faces ---------------------------------------------
    #
    # Only the fraction is stored. The open areas, and the interface's vector area the closure
    # identity makes of them, are formed from it on read (`face_area_of`, `interface_normal_area`).
    face_fraction = SVector{4,T}(ntuple(dir -> @inbounds(fracs[MS_DIRECTION_EDGE[dir]]), Val(4)))
    face_centroid_local = SVector{4,SVector{1,T}}(ntuple(Val(4)) do dir
        k = MS_DIRECTION_EDGE[dir]
        x = @inbounds present[k] ? T(0.5) * (seg_a[k] + seg_b[k]) : _full_face_centre(nodes, dir)
        _face_offset(x, origin, dir)
    end)

    # --- the outside polygon -----------------------------------------------
    #
    # Its edges are the open parts of the cell's own faces, plus the interface segments that bridge
    # the gaps between them. Shoelace accumulated in cell-local coordinates: the products are then
    # `O(h^2)` rather than `O(|x|^2)`, and a cell far from the origin keeps its precision.
    area2 = zero(T)
    cmom = zero2
    for k in 1:4
        if @inbounds present[k]
            la = @inbounds(seg_a[k]) - origin
            lb = @inbounds(seg_b[k]) - origin
            cr = la[1] * lb[2] - la[2] * lb[1]
            area2 += cr
            cmom += (la + lb) * cr
        end
    end

    n_interface, iface_a, iface_b = cell_gap_segments(present, seg_a, seg_b)

    interface_length = zero(T)
    interface_cmom = zero2
    for s in 1:n_interface
        a = @inbounds iface_a[s]
        b = @inbounds iface_b[s]
        la = a - origin
        lb = b - origin
        cr = la[1] * lb[2] - la[2] * lb[1]
        area2 += cr
        cmom += (la + lb) * cr

        v = b - a
        len = sqrt(v[1] * v[1] + v[2] * v[2])
        # Length and first moment only. `SVector(v[2], -v[1])` would be the outward normal times
        # the length and is the right answer, but taking the normal from the closure identity
        # instead (`interface_normal_area`, on read) is what makes the per-cell closure exact rather
        # than merely accurate. `test/test_moments.jl` recomputes this expression independently and
        # compares.
        interface_length += len
        interface_cmom += (T(0.5) * len) * (a + b)
    end

    area = T(0.5) * area2
    centroid = area2 != zero(T) ? origin + cmom / (T(3) * area2) : centre

    # --- the interface facet ----------------------------------------------
    #
    # Only its centroid is stored. Its vector area is `sum_k A_k n_k` by the closure identity, which
    # `interface_normal_area` forms from `face_fraction` on read -- see `cut_cell.jl`'s header.
    interface_centroid = interface_length > zero(T) ? interface_cmom / interface_length : centre

    return CutCellData{2,T,4,1}(
        CELL_CUT, n_interface > 1, area / vol_cell, centroid,
        face_fraction, face_centroid_local, interface_centroid)
end

"""
    cut_cell_moments(geo, lo, hi) -> CutCellData

Sample `geo` at the corners of the box `[lo, hi]` and build its moments.

The standalone entry point: no mesh, no tree, just a box. Convenient for a unit test or a one-off
query, and the form to drive with an analytic level set when checking against a known answer.

Prefer [`cut_cell_moments`](@ref)`(method, domain, geo, cell)` when cells adjoin each other: this
version derives the corners by arithmetic on `lo` and `hi`, which is correct but gives up the
bitwise agreement between neighbours that the mesh and grid forms get from the integer lattice.
"""
function cut_cell_moments(geo, lo::SVector{2,T}, hi::SVector{2,T}) where {T}
    cellsize = hi - lo
    nodes = SVector{4,SVector{2,T}}(ntuple(Val(4)) do k
        b = MS_NODE_BITS[k]
        lo + SVector{2,T}(T(b[1]) * cellsize[1], T(b[2]) * cellsize[2])
    end)
    phi = SVector{4,T}(ntuple(k -> T(sdf_value(geo, @inbounds nodes[k])), Val(4)))
    return cut_cell_moments(nodes, phi, cellsize)
end

# =====================================
# The method-tagged per-cell entry point
#
# `cut_cell_moments(method, domain, geo, cell)` is the one per-cell spelling on a domain, and every
# `AbstractCutCellMethod` answers to it: `MarchingSquaresCutCell` and `MarchingCubesCutCell` here and
# in `marching_cubes/`, `PLICCutCell` in `plic/plic.jl`. A consumer says which reconstruction it
# wants rather than having it implied by the dimensionality of the domain it passed, and gets back
# one cell's worth of data to store however it likes -- nothing allocated.
#
# Arity four is what keeps these clear of the untagged forms above -- the `(nodes, phi, cellsize)`
# primitive and the `(geo, lo, hi)` box -- since Julia partitions dispatch by arity first. The
# fourth argument is typed in every method so the grid and tree methods stay disjoint.
#
# A nodal block and a geometry take the same method: [`cell_values`](@ref) is what tells them
# apart, with one read per node for the block and four `sdf_value`s per cell for the geometry. The
# two agree to roundoff rather than bitwise -- `cell_values`' docstring says why -- and each is
# watertight within itself, which is what matters. `generate_mesh` reaches the corners through that
# same `cell_values` call, so the contour and the moments stay one reconstruction.

"""
    cut_cell_moments(method::MarchingSquaresCutCell, grid::CartesianGrid{2}, geo, ci)
    cut_cell_moments(method::MarchingSquaresCutCell, grid::CartesianGrid{2}, vals, ci)
    cut_cell_moments(method::MarchingSquaresCutCell, mesh::AdaptiveMesh{2}, geo, cell)

The moments of the single cell `ci` (or leaf `cell`) of the domain, reconstructed nodally -- the
[`MarchingSquaresCutCell`](@ref) method of the per-cell entry point every
[`AbstractCutCellMethod`](@ref) answers to. Returns a [`CutCellData`](@ref)`{2,T,4}`, with `T`
taken from the domain.

**This is the form to build a consumer's own storage on.** Nothing is cached and nothing is
allocated: loop or launch over the cells yourself and keep exactly the fields you need. A caller
that wants the contour as well should hand the same nodal block to [`generate_mesh`](@ref)
and to this, so that both read one sampling.

`geo` is anything SDFLibrary.jl's `sdf_value` accepts, or, on a grid, a 2D array of nodal values already
sampled on it (`nodegrid_size(grid)` of them), which is the cheaper way in when the field already
exists -- one read per node rather than four per cell.

Corner coordinates come from [`cell_nodes`](@ref), off the integer lattice, so a face shared by two
cells is built from the same floating-point numbers for both and their apertures agree **bitwise**.
That holds per cell with no communication, which is what makes it safe to call this on one cell in
isolation, at any level, in any order.

See [`cut_cell_moments`](@ref)`(::MarchingCubesCutCell, ...)` for the 3D nodal method and
[`cut_cell_moments`](@ref)`(::PLICCutCell, ...)` for the centroid-fit one.
"""
@inline cut_cell_moments(::MarchingSquaresCutCell, grid::CartesianGrid{2}, geo,
                         ci::CartesianIndex{2}) =
    cut_cell_moments(cell_nodes(grid, ci), cell_values(geo, grid, ci), grid.d)

@inline cut_cell_moments(::MarchingSquaresCutCell, mesh::AdaptiveMesh{2}, geo, c::TreeCell{2}) =
    cut_cell_moments(cell_nodes(mesh, c), cell_values(geo, mesh, c), get_elem_size(mesh, c))

@inline cut_cell_moments(m::MarchingSquaresCutCell, mesh::AdaptiveMesh{2}, geo, i::Integer) =
    cut_cell_moments(m, mesh, geo, leaf(mesh, i))

"""
    cut_cell_moments(geo, grid::CartesianGrid{2})
    cut_cell_moments(geo, grid::CartesianGrid{3})

Moments for every cell of a uniform `grid` against `geo`, reconstructed nodally -- marching squares
in 2D, marching cubes in 3D -- without building a contour mesh.

`geo` is anything SDFLibrary.jl's `sdf_value` accepts, or an array of nodal values already sampled on `grid`
(`nodegrid_size(grid)` of them).

**This allocates one [`CutCellData`](@ref) per cell** -- 112 bytes in 2D and 208 in 3D at
`Float64`, about 1.7 GB over a `200^3` grid -- while the moments are only non-trivial in the
`O(h^(D-1))` band of cut cells. In 3D at any real resolution a consumer wants
[`cut_cell_moments`](@ref)`(method, grid, geo, ci)` recomputed per cell instead; this form is for a
convergence study or a reference answer.
"""
function cut_cell_moments(geo, grid::CartesianGrid{D}) where {D}
    vals = geo isa AbstractArray ? geo : sample_sdf(geo, grid)
    T = float(eltype(vals))
    return _moments_over_grid(vals, convert(CartesianGrid{D,T}, grid))
end

# The nodal reconstruction a dimension implies. Methods rather than a branch, so the tag is a
# compile-time constant and the loop below stays type-stable; the 3D one is in
# `marching_cubes/moments.jl`, beside the construction it names.
@inline _nodal_method(::CartesianGrid{2}) = MarchingSquaresCutCell()

@noinline _nodal_method(::CartesianGrid{D}) where {D} = throw(ArgumentError(
    "the nodal cut-cell construction is 2D (marching squares) or 3D (marching cubes), but `grid` " *
    "is $(D)D"))

function _moments_over_grid(vals::AbstractArray{<:Real,D}, grid::CartesianGrid{D,T}) where {D,T}
    _check_nodal_size(vals, grid)
    method = _nodal_method(grid)
    out = Array{CutCellData{D,T,2D,D-1},D}(undef, grid.n)
    ci = CartesianIndices(grid.n)
    # Through the tagged per-cell form rather than spelling the primitive out again, so this loop
    # and a consumer's own loop are the same construction, bitwise. Threaded above a size floor:
    # each cell is a pure function of its own corner values and writes only its own slot, so there
    # is nothing to synchronise, but a small grid is not worth the launch.
    if length(ci) > THREAD_FLOOR
        @batch for i in eachindex(ci)
            @inbounds idx = ci[i]
            @inbounds out[idx] = cut_cell_moments(method, grid, vals, idx)
        end
    else
        for idx in ci
            @inbounds out[idx] = cut_cell_moments(method, grid, vals, idx)
        end
    end
    return out
end

# One value per grid node is what every nodal construction reads, whether it is filling moments,
# stitching a contour or marching a surface. A geometry or callable has no size to check.
@inline _check_nodal_size(_, ::CartesianGrid) = nothing
function _check_nodal_size(vals::AbstractArray{<:Real,D}, grid::CartesianGrid{D}) where {D}
    sz = CartesianMeshes.nodegrid_size(grid)
    size(vals) == sz || throw(DimensionMismatch(
        "`vals` is $(size(vals)), but `grid` has $sz nodes ($(grid.n) elements per axis); the " *
        "nodal construction reads one value per grid node"))
    return nothing
end

# =====================================
# Whole-domain convenience
#
# `cut_cell_moments(method, domain, geo, cell)` is the whole API, and a consumer that wants a field
# over the domain allocates it and fills it itself -- which is what the one-shot below does for the
# common case, and what a consumer with its own storage or its own kernel does directly.

# One thread per leaf, over the same per-cell entry point a consumer calls, so a cell filled into an
# array here and the same cell asked for directly are one construction rather than two that can
# drift.
@kernel function ms_moments_kernel!(mom, mesh, geo)
    i = @index(Global)
    if i <= nleaves(mesh)
        @inbounds mom[i] = cut_cell_moments(MarchingSquaresCutCell(), mesh, geo, leaf(mesh, i))
    end
end

"""
    cut_cell_moments(geo, mesh::AdaptiveMesh{2})

Moments for every leaf of `mesh`, as an array on the mesh's own backend.

The whole-domain one-shot: allocates, fills and hands back the bare array. Use it when a field over
the whole domain is what you want; use [`cut_cell_moments`](@ref)`(method, domain, geo, cell)` per
cell when you want to choose your own storage, which is cheaper whenever only some of the fields are
wanted or only the cut band matters.

One kernel over the flat leaf array, so the same call serves CPU and GPU. Note what the result may
*not* be asked for: bitwise agreement between the two backends. A field may reach `@fastmath`
primitives (every SDFLibrary.jl geometry does), and a device kernel contracts `a*b + c` into an FMA
where the host does not. What survives exactly is cancellation *within* one backend -- which is
what the per-cell closure and any free-stream property built on it actually rest on.
"""
function cut_cell_moments(geo, mesh::AdaptiveMesh{2,T}) where {T}
    # CartesianMeshes gives `AdaptiveMesh` a `get_backend` method: the backend of its leaf keys.
    dev = KernelAbstractions.get_backend(mesh)
    n = nleaves(mesh)
    out = KernelAbstractions.allocate(dev, CutCellData{2,T,4,1}, n)
    ms_moments_kernel!(dev, DEFAULT_WORKGROUP)(out, mesh, geo; ndrange=n)
    # `out` goes straight back to the caller to be read, so the launch has to have finished. A CPU
    # backend blocks inside the launch anyway; a device one does not.
    KernelAbstractions.synchronize(dev)
    return out
end
