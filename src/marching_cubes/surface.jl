# =====================================
# The reconstructed surface, and the two quantities the mesh cannot hold
#
# The whole-grid march over `MarchingCubesCutCell`'s nodal samples, by `MarchingCubes.jl`'s own
# `march`. It shares nothing with the per-cell Lewiner dispatch the moments are built on but the
# corner values -- that dispatch anchors each crossing on the outside node, where `march` anchors on
# the low one, so the two place a vertex an ulp apart. See `marching_cubes.jl`'s header.
#
# The contour, clipped to the grid's bounds, is left open where the body reaches its edge; that is
# what `clipped_by_boundary` asks of the nodal block.
#
# `generate_mesh` marches and repacks; `vertex_normals` marches and rescales; `cell_indices` needs
# neither, because the cell a triangle came from is recoverable from the triangle itself.
#
# What that costs, stated plainly: a caller that wants both the surface AND the corrected normals
# marches twice. `MarchingCubes.jl`'s working set is the expensive part of that (see `_mc_march`),
# so a consumer doing this every step for a moving body should keep its own `MC` and call
# `MarchingCubes.march` itself rather than going through these.

"""
    _mc_march(geo, grid::CartesianGrid{3}; warn=true) -> MarchingCubes.MC

Sample `geo` over `grid` and march it, returning `MarchingCubes.jl`'s own `MC` object -- the shared
core of [`generate_mesh`](@ref) and [`vertex_normals`](@ref).

`geo` is anything SDFLibrary.jl's `sample_sdf!` accepts -- an SDFLibrary.jl geometry or a callable
`x -> phi` -- or an already-sampled nodal block sized `nodegrid_size(grid)`, which is the cheaper
way in when the field exists and the only way in for a level set that is no geometry.

**This is the expensive call.** At `N = 200` the `MC` object reserves roughly half a gigabyte:
`vert_indices` alone is `3 * (N+1)^3` indices (~195 MB), plus the vertex, normal and triangle
buffers it `sizehint!`s. It is freed when the mesh is returned.

Two degenerate cases warn and still return, because both are ordinary consequences of the grid
rather than bad input: a body reaching the boundary of `grid` leaves the mesh open where it is
clipped, and a grid that finds no sign change yields an empty surface.

Serial host code apart from the sampling pass, which is threaded: `MarchingCubes.jl` stores its
volume as a dense `Array` and its march is serial, so there is no device path.
"""
function _mc_march(geo, grid::CartesianGrid{3,T}; warn::Bool=true) where {T}
    F = float(T)
    vals = Array{F,3}(undef, CartesianMeshes.nodegrid_size(grid))
    _mc_fill!(vals, geo, grid)

    warn && clipped_by_boundary(vals) &&
        @warn "the surface reaches the boundary of `grid`, so the reconstructed mesh is open where it is clipped. Enlarge the grid (a larger `pad`) to close it."

    gF = convert(CartesianGrid{3,F}, grid)
    lines = CartesianMeshes.grid_lines(gF)
    # The axis vectors have to be dense `Vector{F}`: `MarchingCubes.denormalize` takes their
    # `extrema` to map index space back to world coordinates.
    mc = MarchingCubes.MC(vals; x=collect(F, lines[1]), y=collect(F, lines[2]),
                          z=collect(F, lines[3]))
    # A freshly built `MC` has `vert_indices` zeroed, which is what marching into it requires.
    MarchingCubes.march(mc, zero(F))

    warn && isempty(mc.triangles) &&
        @warn "no `sdf == 0` surface was found anywhere in `grid` -- returning an empty mesh. The grid may miss the body entirely, or lie wholly inside it."

    return mc
end

"""
    clipped_by_boundary(vals::AbstractArray{T,D}) -> Bool

Whether any node on the outer boundary of the sampled block is inside the body (`< 0`), i.e.
whether the surface runs off the edge of the grid rather than closing inside it. Scans the `2D`
boundary slices only.
"""
function clipped_by_boundary(vals::AbstractArray{T,D}) where {T,D}
    for a in 1:D, side in (firstindex(vals, a), lastindex(vals, a))
        slice = ntuple(b -> b == a ? (side:side) : axes(vals, b), D)
        any(<(zero(T)), view(vals, slice...)) && return true
    end
    return false
end

# Sampling, or a copy. The block is **converted** into `F` rather than aliased, so an integer or
# `Float32` field still marches with the isolevel and the edge interpolation in the grid's own
# precision.
@inline _mc_fill!(vals, geo, grid::CartesianGrid{3}) = sample_sdf!(vals, geo, grid)

function _mc_fill!(vals, src::AbstractArray{<:Real,3}, grid::CartesianGrid{3})
    _check_nodal_size(src, grid)
    copyto!(vals, src)
    return vals
end

"""
    generate_mesh(geo, grid::CartesianGrid{3}, method::MarchingCubesCutCell; warn=true) -> Mesh{3}

The reconstructed `sdf == 0` surface of `geo` over `grid`, as a `MeshLibrary.Mesh{3}` of `Tri`
elements with shared vertices, watertight and wound so `get_normal` points out of the body.

    grid = bounding_grid(geo; N=200)              # SDFLibrary.jl

    method = MarchingCubesCutCell()

    surface = generate_mesh(geo, grid, method)    # the isosurface, as a mesh
    cells = cell_indices(surface, grid)           # ...or which cell each triangle came from
    normals = vertex_normals(geo, grid, method)   # ...or the field gradient at each vertex

The [`MarchingCubesCutCell`](@ref) member of the `generate_mesh(geo, domain, method)` family that
[`MarchingSquaresCutCell`](@ref) and [`PLICCutCell`](@ref) also answer to. `geo` is anything
SDFLibrary.jl's `sample_sdf!` accepts, or a 3D array of nodal values already sampled on `grid`. Note what
those three calls cost each other: `cell_indices` is a pure function of the finished mesh, but
`vertex_normals` **marches a second time**, because the gradient it corrects is the march's own and
cannot be recovered from the triangles.

This produces the **surface** alone. The per-cell moments a cut-cell scheme needs are
[`cut_cell_moments`](@ref)'s 3D methods, which rebuild one cell from its own eight corner values.

`grid` must be 3D, and its coordinate frame is taken to be the geometry's own local frame. Axes may
differ from each other -- unlike [`calc_volume`](@ref) there is no isotropy requirement,
since nothing here converts a distance into unit-cell coordinates -- though isotropic cells give the
best-shaped triangles.

This is the general-purpose counterpart to the analytic `generate_mesh(geo; ...)` methods
SDFLibrary.jl's primitives carry: it needs nothing from `geo` but its distance, so it works on
bodies with no closed-form parameterisation at all -- an `SDFUnion` above all, whose own
`generate_mesh(geo)` only concatenates the children's meshes, keeping the internal surfaces the
union removes and missing the fillets `smoothing` adds. Where a geometry has an analytic
`generate_mesh`, prefer it: it is exact, cheaper, and better-shaped.

# What marching cubes gives, and what it costs

The march is `MarchingCubes.jl`'s Lewiner et al. (2003) variant, which resolves the ambiguous cube
configurations consistently, so the surface is watertight and manifold -- each interior mesh edge
borders exactly two triangles, which is what SDFLibrary.jl's `SDFMesh` even-odd sign test and
pseudonormal caches need from a mesh handed back to them. Sign conventions line up with no
flipping: the field is negative inside, and feeding that straight in makes marching cubes wind
every triangle so `MeshLibrary.get_normal` points **outwards**.

Two limitations are inherent to reconstructing on a fixed grid:

- **Sharp features are chamfered.** A crease -- the reentrant corner where two bodies meet in an
  unsmoothed `SDFUnion` -- is rounded off within one cell, because vertices can only sit on grid
  edges. Refining narrows the chamfer but never removes it; either resolve the feature with `N`, or
  give the union a `smoothing` comparable to the cell size so what is round is the geometry.
- **A closed surface comes out slightly small.** Every triangle is a chord of the true surface, so
  a convex body is under-measured -- the unit sphere reconstructs to an area of `12.5266` against
  `4π = 12.5664`, `-0.32%` at `h = 0.1`, shrinking as `O(h^2)`. This leans the opposite way to
  [`calc_volume`](@ref), whose plane reconstruction *contains* a convex body.

The surface is clipped to `grid`'s own bounds. If the body reaches the edge of the grid the mesh is
left **open** there rather than capped, and a warning says so; the `pad` in SDFLibrary.jl's
`bounding_grid` normally keeps that from happening. For an `SDFBounded` geometry, nodes beyond its
pad band evaluate to `fill_distance`, which is required to be positive, so they read as outside --
which is what they are.

A repack rather than a conversion: `MarchingCubes.jl` hands its surface back as a position per
vertex and a triple of indices per triangle, which is already `Mesh`'s own nodes/elements split. The
indices are 1-based and dense, so they carry over as `Tri` connectivity untouched, with no
renumbering and no merge tolerance -- the vertices are shared already, each sitting on a grid edge
interpolated from values every cell around that edge reads.

The per-vertex normals are left behind here, because `Mesh` has nowhere to put one; they are
[`vertex_normals`](@ref), in this same node order, at the cost of a second march.

Pass `warn=false` to silence the clipped-surface and empty-surface warnings.
"""
function MeshLibrary.generate_mesh(geo, grid::CartesianGrid{3},
                                   method::MarchingCubesCutCell; warn::Bool=true)
    mc = _mc_march(geo, grid; warn)
    nodes = [Point(v) for v in mc.vertices]
    elements = Vector{Tri{Int64}}(undef, length(mc.triangles))
    for i in eachindex(mc.triangles)
        @inbounds t = mc.triangles[i]
        @inbounds elements[i] = Tri(t[1], t[2], t[3])
    end
    return Mesh(nodes, elements)
end

# Strictly less specific than the method above, so a valid pair never reaches it -- its whole job is
# to say which argument is wrong.
function MeshLibrary.generate_mesh(geo, grid::CartesianGrid{D},
                                   method::MarchingCubesCutCell; warn::Bool=true) where {D}
    throw(ArgumentError("marching cubes is a 3D algorithm, but `grid` is $(D)D; for a 2D grid use " *
                        "`MarchingSquaresCutCell`, which is the same nodal shared-vertex " *
                        "construction one dimension down"))
end

"""
    cell_indices(mesh::Mesh{3}, grid::CartesianGrid{3}) -> Vector{Int}

Which cell of `grid` each triangle of `mesh` came from, as a linear index into
`CartesianIndices(grid.n)`, ordered to match `mesh.elements`. `mesh` must be one
[`generate_mesh`](@ref) built with [`MarchingCubesCutCell`](@ref) over this same `grid`.

**It needs nothing but the mesh.** `MarchingCubes.jl` knows the cell and discards it, and recovering
it afterwards is exact rather than approximate: every vertex of a triangle lies on an edge or in the
interior of one single cube, so the centroid is strictly inside that cube and flooring it into the
grid names that cube and no other. That holds for the extra interior vertex too -- it is not on a
grid edge, but it is inside the cube. The `clamp` is for the boundary only, where a centroid landing
exactly on the far face of the last cell would index one past the end.
"""
function cell_indices(mesh::Mesh{3}, grid::CartesianGrid{3})
    g = convert(CartesianGrid{3,Float64}, grid)
    x0, d, n = g.x0, g.d, g.n
    lin = LinearIndices(n)
    x = mesh.nodes.coord
    cells = Vector{Int}(undef, length(mesh.elements))
    for (t, tri) in enumerate(mesh.elements)
        con = tri.con
        @inbounds c = (x[con[1]] + x[con[2]] + x[con[3]]) / 3
        idx = ntuple(a -> clamp(floor(Int, (c[a] - x0[a]) / d[a]) + 1, 1, n[a]), 3)
        @inbounds cells[t] = lin[idx...]
    end
    return cells
end

"""
    vertex_normals(geo, grid::CartesianGrid{3}, method::MarchingCubesCutCell; warn=false) -> Vector{SVector{3}}

One unit normal per vertex of the reconstructed surface, in `grid`'s own coordinates, ordered to
match the nodes [`generate_mesh`](@ref) returns for the same `(geo, grid)` -- so `normals[i]`
belongs to `mesh.nodes[i]`.

**This marches again.** The normals are the march's own interpolated gradient and cannot be
recovered from the finished mesh, so a caller that wants both the surface and these pays two
marches; see this file's header for when to keep your own `MC` instead. `warn` defaults to `false`
because the caller has usually just built the mesh and been warned once already.

For a true signed distance field this is the surface normal; for a field that is not eikonal it is
still the field's own gradient, which is what the march interpolated along. It is a per-vertex
quantity a `Mesh` has nowhere to put -- `MeshLibrary`'s own normals are per-face, from each `Tri`'s
winding.

`MarchingCubes.jl`'s normals cannot be used as they stand: it central-differences the field over
*index* offsets and never divides by the cell size, so what it normalizes is the gradient in index
space. On a cubic cell that is parallel to the true gradient and the error cancels in the
normalization; on anything else it is skewed towards the short axis -- measured on a sphere over a
60x20x20 grid, up to 30 degrees off the true radial. Rescaling each component by `1/d` recovers the
physical gradient exactly, and renormalizing absorbs the normalization already applied.

A vertex where the field is flat gets a zero normal rather than a `NaN`.
"""
function vertex_normals(geo, grid::CartesianGrid{3,T},
                        method::MarchingCubesCutCell; warn::Bool=false) where {T}
    F = float(T)
    mc = _mc_march(geo, grid; warn)
    g = convert(CartesianGrid{3,F}, grid)
    inv_d = one(F) ./ g.d
    normals = Vector{SVector{3,F}}(undef, length(mc.normals))
    for i in eachindex(mc.normals)
        @inbounds g_i = mc.normals[i] .* inv_d
        mag = norm(g_i)
        @inbounds normals[i] = mag > eps(F) ? g_i ./ mag : zero(SVector{3,F})
    end
    return normals
end
