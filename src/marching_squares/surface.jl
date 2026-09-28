# =====================================
# The reconstructed surface, for looking at
#
# `CutCellData` keeps the interface as a one-point rule -- centroid, area, normal -- because that
# is what a flux integrates against. `generate_mesh` returns the *facets* instead, which is what
# shows the body the discretization actually has, as opposed to the body you thought you specified.
# An under-resolved feature, a blend width that has inflated a fillet, a sharp corner rounded off by
# the mesh rather than by the geometry: all obvious in a picture of this contour, invisible in a
# volume fraction. Worth doing at least once whenever a case behaves oddly.
#
# This pass samples the field itself, through exactly the `cell_values`/`cell_nodes` pair
# `cut_cell_moments` uses, and walks each cut cell with the same `cell_interface` -- so driving both
# with the same `geo` (or the same nodal array) over the same domain builds them from bit-identical
# corner values, and the interface a flux integrates over and the interface a plot shows cannot
# drift apart. A caller that already has the nodal block -- `sample_sdf(geo, grid)`, or a level set
# carried by someone else's solver -- should pass that instead of the geometry: one read per node
# rather than four per cell, and the same numbers `cut_cell_moments(method, grid, vals, ci)` sees.

# Raised by the parts of this file that are structurally 2D: they walk four corners and stitch
# `Line` elements.
@noinline _ms_2d_only() = throw(ArgumentError(
    "marching squares is a 2D reconstruction -- it walks four corners and stitches a contour of " *
    "`Line` elements. For a 3D domain use `MarchingCubesCutCell`, which is the same nodal " *
    "shared-vertex construction one dimension up."))

# =====================================
# Stitching a whole domain's segments into one mesh
#
# The two helpers below turn per-cell segments into a shared-vertex `Mesh{2}`. Their only consumer
# is `generate_mesh` below; the exactness argument they rest on is `marching_squares.jl`'s header.

# One cell's segments, appended to the mesh under construction. The `Line` is `(a, b)`, the
# orientation `cell_interface` already returns, so that `MeshLibrary.get_normal` points out of the
# body -- there is deliberately no reversal here.
@inline function _push_segments!(nodes, elements, cells, ids, n::Integer, a, b, cell::Integer)
    for s in 1:n
        i1 = _vertex_id!(ids, nodes, @inbounds a[s])
        i2 = _vertex_id!(ids, nodes, @inbounds b[s])
        push!(elements, Line(i1, i2))
        push!(cells, Int(cell))
    end
    return nothing
end

# The index of the vertex at `x`, adding it if this is the first cell to reach it.
#
# Keyed on the coordinate itself, with no tolerance: two cells sharing an edge produce the crossing
# on it as the identical floating-point pair, so exact equality is the correct test, and a tolerance
# would risk welding genuinely distinct vertices across a thin feature. `-0.0` and `0.0` are
# distinct `Dict` keys and both can arise for a crossing landing on a corner, so the key is
# normalised by adding zero.
function _vertex_id!(ids::Dict{SVector{2,T},Int}, nodes::Vector{Point{2,T}},
                     x::SVector{2,T}) where {T}
    key = x .+ zero(T)
    return get!(ids, key) do
        push!(nodes, Point(key))
        length(nodes)
    end
end

"""
    generate_mesh(geo, grid::CartesianGrid{2}, method::MarchingSquaresCutCell; warn_open=true)
    generate_mesh(geo, mesh::AdaptiveMesh{2}, method::MarchingSquaresCutCell; warn_open=true)

The reconstructed `phi == 0` contour of `geo` over the domain, as a `MeshLibrary.Mesh{2}` of `Line`
elements with shared vertices.

`geo` is anything SDFLibrary.jl's `sdf_value` accepts -- an SDFLibrary.jl geometry or a callable
`x -> phi` -- or, on a `CartesianGrid`, a 2D array of nodal values already sampled on it
(`CartesianMeshes.nodegrid_size(grid)` of them), which is the cheaper way in when the field already
exists and the only way in for a level set that is no geometry at all. `method` is last rather than
first, to match `MeshLibrary.generate_mesh` elsewhere.

# Nodal (shared-vertex) reconstruction

Each contour vertex sits on the cell edge between two corners of opposite sign, placed by linear
interpolation of the two values. Two cells sharing an edge read the same two numbers with the same
expression, so they cannot disagree about where the contour crosses it: the vertex is shared and
the polyline is stitched by **exact** coordinate equality, with no merge tolerance anywhere. See
`marching_squares.jl`'s header for the two choices that buy that, and why a per-cell tangent-line
fit ([`PLICCutCell`](@ref), and [`calc_volume`](@ref)) cannot.

Linear interpolation along an edge is *exact* wherever the interface is straight -- a true signed
distance field is linear along any line -- and second-order in the cell size where it curves.

`Line` elements are wound so that `MeshLibrary.get_normal` points **out of the body**, matching
what [`MarchingCubesCutCell`](@ref) produces in 3D. That is the orientation [`cell_interface`](@ref)
already returns, so nothing is reversed anywhere between the walk and the mesh.

# What the reconstruction cannot do

- **Sharp features are chamfered.** A crease -- the reentrant corner where two bodies meet in an
  unsmoothed SDFLibrary.jl `SDFUnion` -- is rounded off within one cell, because vertices can only sit on
  cell edges. Refining narrows the chamfer but never removes it. Either resolve it, or give the
  union a `smoothing` comparable to the cell size so the geometry itself is what is round.
- **Saddle cells are ambiguous.** Two opposite corners outside and the other two inside is a case
  marching squares cannot resolve; such a cell contributes two segments whose pairing is a guess.
  See [`cell_gap_segments`](@ref), and [`is_ambiguous`](@ref) to count them.
- **A convex body comes out slightly small**, since every segment is a chord of the true contour;
  the error falls as `O(h^2)`. This leans the opposite way to [`calc_volume`](@ref), whose
  line reconstruction contains a convex body and so over-measures it.
- **On an `AdaptiveMesh`, a cut cell at a level jump leaves a gap.** A coarse cell interpolates
  across its whole face while its two fine neighbours interpolate across half-faces each, so the
  crossings genuinely differ and the contour is open there. Refine a band around the body so every
  cut cell sits at the finest level and the case does not arise.

The contour is clipped to the domain's own bounds. If the body reaches the edge of a
`CartesianGrid` the contour is left **open** there rather than capped, and a warning says so; the
`pad` in SDFLibrary.jl's `bounding_grid` normally keeps that from happening. Pass `warn_open=false` to
silence that and the empty-contour warning.

Runs on the host and returns host data -- it builds `Vector`-backed mesh arrays.
"""
function MeshLibrary.generate_mesh(geo, grid::CartesianGrid{D,TG},
                                   method::MarchingSquaresCutCell;
                                   warn_open::Bool=true) where {D,TG}
    D == 2 || _ms_2d_only()
    T = _ms_mesh_eltype(geo, TG)
    g = convert(CartesianGrid{2,T}, grid)
    _check_nodal_size(geo, g)

    idx = CartesianIndices(g.n)
    cut, clip = _ms_screen(geo, g, idx)

    nodes = Point{2,T}[]
    elements = Line{Int64}[]
    cells = Int[]
    ids = Dict{SVector{2,T},Int}()

    # Serial, and it has to be: `_push_segments!` welds vertices through a shared `Dict` and
    # appends to shared `Vector`s, so threading it would race and make the numbering depend on
    # which thread got there first. It walks the cut band only, which is `O(perimeter)`.
    for k in eachindex(cut)
        @inbounds cut[k] || continue
        ci = @inbounds idx[k]
        n, a, b = cell_interface(cell_nodes(g, ci), cell_values(geo, g, ci))
        _push_segments!(nodes, elements, cells, ids, n, a, b, k)
    end

    if warn_open
        any(clip) && @warn "the contour reaches the boundary of `grid`, so it is left open where it is clipped. Enlarge the grid (a larger `pad`) to close it."
        isempty(cells) && @warn "no `phi == 0` contour was found anywhere in `grid` -- returning an empty mesh. The grid may miss the body entirely, or lie wholly inside it."
    end
    return Mesh(nodes, elements)
end

# Which cells the contour passes through, and which of those leave it open at the edge of the grid.
#
# **This is the threaded half, and the only part that can be.** Each thread writes one `Bool` per
# cell into two preallocated masks -- disjoint slots, no shared accumulator, no order dependence --
# so `@batch` is safe here in a way it is not over the stitch that follows. Two bytes a cell is the
# whole extra footprint; storing the walk itself instead would be ~72.
#
# The cut band is then re-sampled by the serial pass: `4` extra `get_sdf` per *cut* cell against
# `4/nthreads` saved on every cell, so it pays from a few thousand cells up, which is what
# `THREAD_FLOOR` tests.
#
# A cell is "cut" exactly when its corners do not all share a sign, the same condition as
# `cell_interface` returning a nonzero segment count, so skipping the rest changes no output.
function _ms_screen(geo, g::CartesianGrid{2,T}, idx) where {T}
    cut = Vector{Bool}(undef, length(idx))
    clip = Vector{Bool}(undef, length(idx))
    if length(idx) > THREAD_FLOOR
        @batch for k in eachindex(cut)
            @inbounds cut[k], clip[k] = _ms_screen_cell(geo, g, idx[k])
        end
    else
        for k in eachindex(cut)
            @inbounds cut[k], clip[k] = _ms_screen_cell(geo, g, idx[k])
        end
    end
    return cut, clip
end

@inline function _ms_screen_cell(geo, g::CartesianGrid{2,T}, ci::CartesianIndex{2}) where {T}
    phi = cell_values(geo, g, ci)
    @inbounds allsame = (phi[1] >= zero(T)) == (phi[2] >= zero(T)) == (phi[3] >= zero(T)) ==
                        (phi[4] >= zero(T))
    return !allsame, _cell_clips_boundary(phi, ci, g.n)
end

function MeshLibrary.generate_mesh(geo, mesh::AdaptiveMesh{D,T},
                                   method::MarchingSquaresCutCell;
                                   warn_open::Bool=true) where {D,T}
    D == 2 || _ms_2d_only()
    geo isa AbstractArray && _no_nodal_array_on_tree()

    nodes = Point{2,T}[]
    elements = Line{Int64}[]
    cells = Int[]
    ids = Dict{SVector{2,T},Int}()

    for i in eachleaf(mesh)
        c = leaf(mesh, i)
        n, a, b = cell_interface(cell_nodes(mesh, c), cell_values(geo, mesh, c))
        _push_segments!(nodes, elements, cells, ids, n, a, b, i)
    end

    warn_open && isempty(cells) && @warn "no `phi == 0` contour was found on any leaf of `mesh` -- returning an empty mesh. The mesh may miss the body entirely, or lie wholly inside it."
    return Mesh(nodes, elements)
end

@noinline _no_nodal_array_on_tree() = throw(ArgumentError(
    "a nodal array can only be reconstructed on a `CartesianGrid{2}`; an `AdaptiveMesh` has no " *
    "single nodal array to index, so pass the geometry (or a callable `x -> phi`) instead"))

# The element type the mesh comes out in. A nodal block decides it -- a `Float32` field should not be
# widened by a `Float64` grid, the same choice `extract_surface_plic` makes -- and a geometry or
# callable leaves it to the grid, which is what samples the corners.
@inline _ms_mesh_eltype(vals::AbstractArray{<:Real,2}, ::Type{TG}) where {TG} = float(eltype(vals))
@inline _ms_mesh_eltype(geo, ::Type{TG}) where {TG} = TG

# Does this cell leave the contour open at the edge of the grid? A domain-boundary edge whose two
# corner values differ in sign is one the interface crosses. Asked on the corners, which is the same
# test as `0 < face_fraction < 1` on that boundary face (see `_open_fraction`).
@inline function _cell_clips_boundary(phi::SVector{4,T}, idx::CartesianIndex{2}, n) where {T}
    @inbounds begin
        # Corners in `MS_NODE_BITS` order: 1 = (0,0), 2 = (1,0), 3 = (1,1), 4 = (0,1).
        idx[1] == 1 && _straddles(phi[1], phi[4]) && return true   # -x face
        idx[1] == n[1] && _straddles(phi[2], phi[3]) && return true # +x face
        idx[2] == 1 && _straddles(phi[1], phi[2]) && return true   # -y face
        idx[2] == n[2] && _straddles(phi[4], phi[3]) && return true # +y face
    end
    return false
end

# Matches `_open_fraction`'s `>= 0` convention exactly, so "partly open" and "straddles" are the
# same set of faces rather than two nearly-equal ones.
@inline _straddles(a::T, b::T) where {T} = (a >= zero(T)) != (b >= zero(T))
