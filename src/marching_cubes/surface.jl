# =====================================
# The reconstructed surface
#
# The whole-grid march over `MarchingCubesCutCell`'s nodal samples, by `MarchingCubes.jl`'s own
# `march`. It shares nothing with the per-cell Lewiner dispatch the moments are built on but the
# corner values -- that dispatch anchors each crossing on the outside node, where `march` anchors on
# the low one, so the two place a vertex an ulp apart. See `marching_cubes.jl`'s header.
#
# The contour, clipped to the grid's bounds, is left open where the body reaches its edge.
#
# `generate_mesh` marches the cache's own `phi` and repacks; `cell_indices` needs none of that,
# because the cell a triangle came from is recoverable from the triangle itself.

"""
    generate_mesh(cache::MarchingCubesCutCellCache, grid::CartesianGrid{3}; warn=true) -> Mesh{3}

The reconstructed `sdf == 0` surface of `cache`'s last update, as a `MeshLibrary.Mesh{3}` of `Tri`
elements with shared vertices, watertight and wound so `get_normal` points out of the body. Marched
straight from the nodal field the cache stored, `cache.phi` -- the very numbers its cells were built
from -- so the body is not sampled again.

    method = MarchingCubesCutCell()
    grid = bounding_grid(geo; N=200)                                # SDFLibrary.jl
    cache = update_cache!(allocate_cache(grid, method), geo, grid)

    surface = generate_mesh(cache, grid)          # the isosurface, as a mesh
    cells = cell_indices(surface, grid)           # ...or which cell each triangle came from

`grid` is the grid of that update. The cache keeps only its `n`, which is checked; a grid moved
since would place the surface wrongly. On a device cache, `phi` is copied to the host first: the
march is host code, and it is the expensive part: at `N = 200` the `MarchingCubes.MC` object
reserves roughly half a gigabyte -- `vert_indices` alone is `3 * (N+1)^3` indices (~195 MB), plus
the vertex, normal and triangle buffers it `sizehint!`s -- and it is serial, with no device path.

`cell_indices` is a pure function of the finished mesh and costs nothing extra.

This produces the **surface** alone; the per-cell moments are the cache's `cells`. Axes may differ
from each other -- there is no isotropy requirement, since nothing here converts a distance into
unit-cell coordinates -- though isotropic cells give the best-shaped triangles.

This is the general-purpose counterpart to the analytic `generate_mesh(geo; ...)` methods
SDFLibrary.jl's primitives carry: it needs nothing from the body but its distance, so it works on
bodies with no closed-form parameterisation -- an `SDFUnion` above all, whose own `generate_mesh(geo)`
only concatenates the children's meshes, keeping internal surfaces the union removes and missing the
fillets `smoothing` adds. Where a geometry has an analytic `generate_mesh`, prefer it: exact,
cheaper, better-shaped.

# What marching cubes gives, and what it costs

The march is `MarchingCubes.jl`'s Lewiner et al. (2003) variant, which resolves ambiguous cube
configurations consistently, so the surface is watertight and manifold -- each interior mesh edge
borders exactly two triangles, which is what SDFLibrary.jl's `SDFMesh` even-odd sign test and
pseudonormal caches need from a mesh handed back to them. Sign conventions line up with no flipping:
the field is negative inside, and feeding that straight in makes marching cubes wind every triangle
so `MeshLibrary.get_normal` points **outwards**.

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
left **open** there rather than capped; choosing a grid that suits the problem is the caller's call.
For an `SDFBounded` geometry, nodes beyond its pad band evaluate to `fill_distance`, required to be
positive, so they read as outside -- which is what they are.

A repack rather than a conversion: `MarchingCubes.jl` hands its surface back as a position per
vertex and a triple of indices per triangle, already `Mesh`'s own nodes/elements split. The indices
are 1-based and dense, so they carry over as `Tri` connectivity untouched, with no renumbering and
no merge tolerance -- the vertices are shared already, each sitting on a grid edge interpolated from
values every cell around that edge reads.

A grid that finds no sign change warns and still returns an empty surface, since that is an ordinary
consequence of the grid rather than bad input. Pass `warn=false` to silence the warning.
"""
function MeshLibrary.generate_mesh(cache::MarchingCubesCutCellCache, grid::CartesianGrid{3,T};
                                   warn::Bool=true) where {T}
    F = float(T)
    phi = Adapt.adapt(Array, cache.phi)
    vals = Array{F,3}(undef, CartesianMeshes.nodegrid_size(grid))
    # `copyto!` **converts** into `F` rather than aliasing, so an integer or `Float32` cache still
    # marches with the isolevel and the edge interpolation in the grid's own precision.
    _check_nodal_size(phi, grid)
    copyto!(vals, phi)

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

    # A repack rather than a conversion: see the docstring above.
    nodes = [Point(v) for v in mc.vertices]
    elements = Vector{Tri{Int64}}(undef, length(mc.triangles))
    for i in eachindex(mc.triangles)
        @inbounds t = mc.triangles[i]
        @inbounds elements[i] = Tri(t[1], t[2], t[3])
    end
    return Mesh(nodes, elements)
end

"""
    cell_indices(mesh::Mesh{3}, grid::CartesianGrid{3}) -> Vector{Int}

Which cell of `grid` each triangle of `mesh` came from, as a linear index into
`CartesianIndices(grid.n)`, ordered to match `mesh.elements`. `mesh` must be a marching-cubes
surface, [`generate_mesh`](@ref)`(cache, grid)`, over this same `grid`.

**Needs nothing but the mesh.** `MarchingCubes.jl` knows the cell and discards it, and recovering it
afterwards is exact: every vertex of a triangle lies on an edge or in the interior of one single
cube, so the centroid is strictly inside that cube and flooring it into the grid names that cube and
no other. That holds for the extra interior vertex too -- it is not on a grid edge, but it is inside
the cube. The `clamp` is for the boundary only, where a centroid landing exactly on the far face of
the last cell would index one past the end.
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

