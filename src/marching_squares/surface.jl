"""
    generate_mesh(cache::MarchingSquaresCutCellCache, grid::CartesianGrid{2}; warn=true)
    generate_mesh(cache::MarchingSquaresCutCellCache, mesh::AdaptiveMesh{2}; warn=true)

The reconstructed `phi == 0` contour of `cache`'s last update, as a `MeshLibrary.Mesh{2}` of `Line`
elements with shared vertices, marched straight from the field the cache stored in `phi` -- the very
numbers its cells were built from, with no second sampling of the body. The domain is the one that
update ran on. On a device cache, `phi` is copied to the host first: the march is host code.

The two domains take the cache in two layouts:

- on a grid, `phi` is the nodal block, one value per grid node (`nodegrid_size(grid)` of them), and
  `cells` one record per cell, as [`allocate_cache`](@ref) builds it. The cache keeps only its `n`,
  which is checked; a grid moved since would place the contour wrongly.
- on an `AdaptiveMesh`, both are **linear lists over the leaves**, in leaf order: `phi[i]` is leaf
  `i`'s four corner values as an `SVector{4}` in [`MS_NODE_BITS`](@ref) order, and `cells[i]` its
  moments, as `allocate_cache(mesh, MarchingSquaresCutCell())` builds it. Both lengths are checked
  against `nleaves(mesh)`.

# Nodal (shared-vertex) reconstruction

Each contour vertex sits on the cell edge between two corners of opposite sign, placed by linear
interpolation of the two values. Two cells sharing an edge read the same two numbers with the same
expression, so they can't disagree about the crossing: the vertex is shared and the polyline is
stitched by **exact** coordinate equality, with no merge tolerance anywhere. See
`marching_squares.jl`'s header for the two choices that buy that, and why a per-cell tangent-line fit
([`PLICCutCell`](@ref), and [`calc_volume`](@ref)) cannot.

Linear interpolation along an edge is exact wherever the interface is straight, second-order where
it curves.

`Line` elements are wound so `MeshLibrary.get_normal` points **out of the body**, matching what
[`MarchingCubesCutCell`](@ref) produces in 3D -- the orientation [`cell_interface`](@ref) already
returns, so nothing is reversed between the walk and the mesh.

# What the reconstruction cannot do

- **Sharp features are chamfered.** A crease -- the reentrant corner where two bodies meet in an
  unsmoothed SDFLibrary.jl `SDFUnion` -- is rounded off within one cell, since vertices can only sit
  on cell edges. Refining narrows the chamfer but never removes it. Either resolve it, or give the
  union a `smoothing` comparable to the cell size so the geometry itself is round.
- **Saddle cells are ambiguous.** Two opposite corners outside and the other two inside is a case
  marching squares cannot resolve; such a cell contributes two segments whose pairing is a guess.
  See [`cell_gap_segments`](@ref), and [`is_ambiguous`](@ref) to count them.
- **A convex body comes out slightly small**, since every segment is a chord of the true contour;
  the error falls as `O(h^2)`. This leans the opposite way to [`calc_volume`](@ref), whose line
  reconstruction contains a convex body and so over-measures it.
- **On an `AdaptiveMesh`, a cut cell at a level jump leaves a gap.** A coarse cell interpolates
  across its whole face while its two fine neighbours interpolate across half-faces each, so the
  crossings genuinely differ and the contour is open there. Refine a band around the body so every
  cut cell sits at the finest level.

The contour is clipped to the domain's own bounds: where it reaches the edge it is left **open**
rather than capped. Choosing a domain that suits the problem is the caller's call. Pass
`warn=false` to silence the empty-contour warning.

Runs on the host and returns host data -- it builds `Vector`-backed mesh arrays.
"""
function MeshLibrary.generate_mesh(cache::MarchingSquaresCutCellCache{<:AbstractMatrix{<:Real}},
                                   grid::CartesianGrid{2}; warn::Bool=true)
    vals = Adapt.adapt(Array, cache.phi)
    T = eltype(vals)
    g = convert(CartesianGrid{2,T}, grid)
    _check_nodal_size(vals, g)

    idx = CartesianIndices(g.n)
    cut = _ms_screen(vals, g, idx)

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
        n, a, b = cell_interface(cell_nodes(g, ci), cell_values(vals, g, ci))
        _push_segments!(nodes, elements, cells, ids, n, a, b, k)
    end

    warn && isempty(cells) && @warn "no `phi == 0` contour was found anywhere in `grid` -- returning an empty mesh. The grid may miss the body entirely, or lie wholly inside it."
    return Mesh(nodes, elements)
end

# On a tree the cache is laid out on the leaves: `phi[i]` is leaf `i`'s four corner values and
# `cells[i]` its moments, both in leaf order. There is no shared nodal block to march -- a hanging
# node belongs to a coarse leaf's edge, not its corners -- but the corners come off `cell_nodes`,
# which is bitwise exact across levels, so two leaves sampled at a shared corner hold one number.
function MeshLibrary.generate_mesh(cache::MarchingSquaresCutCellCache{<:AbstractVector{<:SVector{4}}},
                                   mesh::AdaptiveMesh{2,T}; warn::Bool=true) where {T}
    _check_leaf_size(cache, mesh)
    phi = Adapt.adapt(Array, cache.phi)

    nodes = Point{2,T}[]
    elements = Line{Int64}[]
    cells = Int[]
    ids = Dict{SVector{2,T},Int}()

    for i in eachleaf(mesh)
        n, a, b = cell_interface(cell_nodes(mesh, i), SVector{4,T}(@inbounds phi[i]))
        _push_segments!(nodes, elements, cells, ids, n, a, b, i)
    end

    warn && isempty(cells) && @warn "no `phi == 0` contour was found on any leaf of `mesh` -- returning an empty mesh. The mesh may miss the body entirely, or lie wholly inside it."
    return Mesh(nodes, elements)
end

# Which cells the contour passes through.
#
# **The threaded half, and the only part that can be.** Each thread writes one `Bool` per cell into
# a preallocated mask -- disjoint slots, no shared accumulator, no order dependence -- so `@batch` is
# safe here in a way it is not over the stitch that follows. One byte a cell is the whole extra
# footprint; storing the walk itself instead would be ~72.
#
# The cut band is read again by the serial pass, so the screen pays off from a few thousand cells up,
# which is what `THREAD_FLOOR` tests.
#
# A cell is "cut" exactly when its corners don't all share a sign, the same condition as
# `cell_interface` returning a nonzero segment count, so skipping the rest changes no output.
function _ms_screen(vals, g::CartesianGrid{2,T}, idx) where {T}
    cut = Vector{Bool}(undef, length(idx))
    if length(idx) > THREAD_FLOOR
        @batch for k in eachindex(cut)
            @inbounds cut[k] = _ms_screen_cell(vals, g, idx[k])
        end
    else
        for k in eachindex(cut)
            @inbounds cut[k] = _ms_screen_cell(vals, g, idx[k])
        end
    end
    return cut
end

@inline function _ms_screen_cell(vals, g::CartesianGrid{2,T}, ci::CartesianIndex{2}) where {T}
    phi = cell_values(vals, g, ci)
    @inbounds allsame = (phi[1] >= zero(T)) == (phi[2] >= zero(T)) == (phi[3] >= zero(T)) ==
                        (phi[4] >= zero(T))
    return !allsame
end

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