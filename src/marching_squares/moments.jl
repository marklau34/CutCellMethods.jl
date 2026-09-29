# =====================================
# Cut-cell moments, from the nodal marching-squares reconstruction
#
# The reconstruction itself lives next door in `marching_squares.jl`: sample `phi` at a cell's four
# corners, interpolate each edge's zero crossing, and report the open part of every edge
# (`cell_boundary_walk`) plus the segments bridging the gaps between them (`cell_gap_segments`). This
# file turns one cell's walk into a `CutCellData`, built on the *same* walk the contour
# `generate_mesh` draws comes from -- so the aperture a flux passes through and the surface a plot
# shows are one reconstruction rather than two that might disagree.
#
# Why nodal matters here: a per-cell fit gives cell A and neighbour B two different planes through
# the face they share, anchored `O(h)` apart, so they compute different apertures for it and mass is
# manufactured at the face. Nodally, both cells read the same corner values with the same expression,
# so their apertures agree bitwise with no communication or consistency pass, which is what makes it
# safe to recompute one cell's geometry on demand at any level. `test/test_moments.jl` checks that as
# bitwise assertions rather than tolerances.
#
# The 3D construction is `marching_cubes/moments.jl`, as methods on these same names producing the
# same `CutCellData` -- `{3,T,6}` instead of `{2,T,4}` -- through the same accessors and the same
# exactly-zero `closure_residual`. Most of it reuses this file's construction: each of the six cell
# faces is a 2D square, so `cell_boundary_walk` gives its aperture and centroid on corner values the
# neighbour reads too, and the interface's vector area follows from the closure identity with no
# interface polygon at all. What genuinely needs marching cubes is the outside volume, the outside
# centroid and the interface centroid.
#
# One thing does *not* carry over: `cell_gap_segments`' unconditional "outside connected through the
# middle" resolution of an ambiguous cell. In 3D the interface facets come from the Lewiner tables, so
# the 3D face clip takes its pairing from the same decider those tables use instead. See that file's
# header.

# The centre of the full Cartesian face in `direction`, from the two lattice corners of the edge on
# it. Reported as the centroid of every face the walk doesn't split: a fully open face (full
# quadrature weight) and a closed one (zero weight, a finite placeholder).
#
# Comes from the corners rather than `centre + h/2` because an outside cell and a cut neighbour must
# report the same point for a shared face, or the flux through it is evaluated at two different
# places: `centre + h/2` is not bitwise the neighbour's `centre - h/2`, while this sum of the same two
# lattice points is, for both cells. The 3D face clip does the same.
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

What both marching-squares caches' kernels, on a grid and on a tree, build every cell with.
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
        # Length and first moment only. `SVector(v[2], -v[1])` would give the outward normal times
        # length and is correct, but taking the normal from the closure identity instead
        # (`interface_normal_area`, on read) makes the per-cell closure exact rather than merely
        # accurate. `test/test_moments.jl` recomputes this independently and compares.
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
