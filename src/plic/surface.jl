"""
    generate_mesh(cache::PLICCutCellCache, grid::CartesianGrid{D}; min_tri_area_fraction=1e-4) -> Mesh{D}

The [`PLICCutCell`](@ref) surface of `cache`'s last update, as a `MeshLibrary.Mesh{D}` of `Line`
(2D) or `Tri` (3D) elements: every cell's stored plane, clipped to its cell by
[`extract_surface_plic`](@ref). Read off the cache, not fitted again, so the surface is the very fit
the cache's fractions came from.

Every cell is handed over, cut or not, and the clip is what leaves the uncut ones out: a plane that
does not cross its cell clips to fewer than three points in 3D (drawing nothing) or to a zero-length
segment in 2D (dropped by `min_tri_area_fraction`). This costs memory proportional to the whole grid
rather than the cut band, so it is meant for a picture, not a solver's inner loop.

`min_tri_area_fraction` drops an element smaller than that fraction of a cell's own: a triangle's
area against a cell face in 3D, a segment's length against a cell edge in 2D. `0` keeps everything.

`grid` is the grid of that update. The cache keeps only its `n`, which is checked; a grid moved
since would place the surface wrongly. Its isotropy is checked when the cache is allocated/updated.

Each plane is fitted from one cell alone, so the surface comes out as a field of disconnected
shards rather than a stitched contour. For a watertight surface, reconstruct nodally with
[`MarchingSquaresCutCell`](@ref) or [`MarchingCubesCutCell`](@ref).

An empty result warns and returns an empty mesh rather than throwing: a grid that misses the body,
or lies wholly inside it, describes the grid rather than bad input. Runs on the host and returns
host data; on a device cache the planes are copied to the host first.
"""
function MeshLibrary.generate_mesh(cache::PLICCutCellCache, grid::CartesianGrid; min_tri_area_fraction::Real=1e-4)
    _check_cache_size(cache.cells, grid)
    normals = vec(Adapt.adapt(Array, cache.normals))
    intercepts = vec(Adapt.adapt(Array, cache.intercepts))
    mesh = extract_surface_plic(normals, intercepts, vec(CartesianIndices(Tuple(grid.n))), grid;
                                min_tri_area_fraction)
    isempty(mesh.elements) && @warn "no cut cell was found anywhere in `grid` -- returning an empty mesh. The grid may miss the body entirely, or lie wholly inside it."
    return mesh
end

# The one isotropy gate for the PLIC reconstruction: the cache's `allocate_cache` and
# `update_cache!` can afford to throw, where the per-cell fit has to stay launchable from device
# code.
@noinline function _require_isotropic(grid::CartesianGrid)
    CartesianMeshes.is_isotropic(grid) || throw(ArgumentError(
        "PLICCutCell needs an isotropic grid (equal cell size on every axis), got grid.d = $(Tuple(grid.d))"))
    return nothing
end

"""
    extract_surface_plic(normals, intercepts, cell_indices, grid::CartesianGrid{2}; min_tri_area_fraction=1e-4) -> Mesh{2}

The PLIC interface over `cell_indices` as a `MeshLibrary.Mesh{2}` of `Line` elements: one segment
per cell, clipped from that cell's plane by `CartesianMeshes.cell_plane_clip` and mapped into
`grid`'s frame. The half of [`generate_mesh`](@ref) that fits nothing itself -- a VOF solver
holding planes of its own passes them straight in.

- `cell_indices` -- the cells to draw, as `CartesianIndex{2}`es into `grid`. Which cells those are
  is the caller's business: `generate_mesh` passes every cell, a VOF solver its interfacial cells.
  A plane that does not cross its cell (misses it, or touches one corner) comes back from
  `cell_plane_clip` as a zero-length segment rather than a stray one.
- `normals`/`intercepts` -- that plane, one entry per entry of `cell_indices` and not per grid cell,
  in that cell's own unit-cell frame, the frame `cell_plane_clip` works in.
- `min_tri_area_fraction` -- drops a segment shorter than that fraction of a cell edge
  (`minimum(grid.d)`), the 2D counterpart of the 3D method's triangle area against a cell face. At
  the default that removes the zero-length segments above and roundoff slivers; `0` keeps one
  segment per entry.

Two nodes per element, never shared between elements: each plane was fitted from one cell with no
reference to its neighbours, so the surface is a field of disconnected shards with deliberately no
merge pass. The element type comes from `normals`/`intercepts`, with `grid` converted to match, so
a `Float32` field is not widened by a `Float64` grid.

The clip runs as a `KernelAbstractions` kernel over `cell_indices`, on whatever backend they live
on, but the `Mesh` it fills is host-resident, so the inputs must be host arrays.
"""
function extract_surface_plic(normals, intercepts, cell_indices::AbstractVector{CartesianIndex{2}},
                              grid::CartesianGrid{2}; min_tri_area_fraction::Real=1e-4)
    T = float(promote_type(eltype(eltype(normals)), eltype(intercepts)))
    g = convert(CartesianGrid{2,T}, grid)

    # An empty `cell_indices` is an ordinary state -- a field that is still everywhere pure has no
    # interface to draw -- but an empty `ndrange` is not a launch, so it returns an empty mesh
    # rather than reaching the kernel.
    n_interface = length(cell_indices)
    ends = Vector{Point{2,T}}(undef, 2 * n_interface)
    if n_interface > 0
        backend = get_backend(normals)
        plic_clip_kernel!(backend, 64)(ends, cell_indices, normals, intercepts, g, g.d[1], g.d[2];
                                       ndrange=n_interface)
    end

    min_length = min_tri_area_fraction * T(minimum(g.d))
    kept = [k for k in 1:n_interface if norm(ends[2k].coord - ends[2k-1].coord) >= min_length]
    nodes = [ends[2k - 1 + s] for k in kept for s in 0:1]
    elements = [Line(2j - 1, 2j) for j in eachindex(kept)]

    return Mesh(nodes, elements)
end

"""
One thread per entry of `cell_indices`. `nodes[2k-1]`/`nodes[2k]` are cell `k`'s two endpoints, so
each thread writes only its own two slots and no atomics are needed. `normals`/`intercepts` are
indexed by `k`; `cell_indices[k]` is needed only for the cell-local to world mapping.
"""
@kernel function plic_clip_kernel!(nodes::AbstractVector{<:Point{2}},
                                   cell_indices::AbstractVector{CartesianIndex{2}},
                                   normals::AbstractVector{<:SVector{2}},
                                   intercepts::AbstractVector{<:Real},
                                   grid::CartesianGrid{2}, dx::Real, dy::Real)
    k = @index(Global, Linear)
    @inbounds begin
        ci = cell_indices[k]
        p1, p2 = CartesianMeshes.cell_plane_clip(normals[k], intercepts[k], dx, dy)
        nodes[2k-1] = Point(CartesianMeshes.local_to_global(grid, p1, ci))
        nodes[2k] = Point(CartesianMeshes.local_to_global(grid, p2, ci))
    end
end

# =======================================================================================

"""
    extract_surface_plic(normals, intercepts, cell_indices, grid::CartesianGrid{3}; min_tri_area_fraction=1e-4) -> Mesh{3}

The 3D counterpart of the 2D method above -- same contract, same arguments, same caveats -- as a
`MeshLibrary.Mesh{3}` of `Tri` elements. A plane cuts a cell in a polygon of 3 to 6 vertices rather
than a segment, so each cell contributes 1 to 4 triangles: clip
([`plic_clip_poly_kernel!`](@ref)), order each polygon's vertices
([`plic_sort_poly_kernel!`](@ref)), then fan-triangulate, compacting past the padding slots in the
same pass.

`min_tri_area_fraction` drops a fan triangle whose area falls below that fraction of a cell face
(`minimum(grid.d)^2`): a clipped polygon routinely has two vertices a rounding error apart where
the plane passes near a cell corner, and the zero-area triangle between them is a degenerate
element rather than a piece of surface. Its vertices are still kept, since they belong to the
polygon's other triangles.
"""
function extract_surface_plic(normals, intercepts, cell_indices::AbstractVector{CartesianIndex{3}},
                              grid::CartesianGrid{3}; min_tri_area_fraction::Real=1e-4)
    T = float(promote_type(eltype(eltype(normals)), eltype(intercepts)))
    g = convert(CartesianGrid{3,T}, grid)

    # Empty `cell_indices`: as in 2D, an ordinary state but not a launch.
    n_interface = length(cell_indices)
    points = Matrix{Point{3,T}}(undef, 6, n_interface)
    npoints = fill(0, n_interface)
    if n_interface > 0
        backend = get_backend(normals)
        plic_clip_poly_kernel!(backend, 64)(points, npoints, cell_indices, normals, intercepts, g,
                                            g.d[1], g.d[2], g.d[3]; ndrange=n_interface)
        plic_sort_poly_kernel!(backend, 64)(points, npoints, normals; ndrange=n_interface)
    end

    n_verts = 0
    n_tris = 0
    for k in 1:n_interface
        n = npoints[k]
        n < 3 && continue # degenerate: plane only grazes an edge/corner of the cell
        n_verts += n
        n_tris += n - 2
    end

    nodes = Vector{Point{3,T}}(undef, n_verts)
    elements = Vector{Tri{Int}}(undef, n_tris)

    min_area = min_tri_area_fraction * T(minimum(g.d))^2
    min_cross_sq = (2 * min_area)^2

    base_v = 0
    base_t = 0
    for k in 1:n_interface
        n = npoints[k]
        n < 3 && continue # degenerate: plane only grazes an edge/corner of the cell

        for m in 1:n
            nodes[base_v + m] = points[m, k]
        end
        p1 = nodes[base_v + 1].coord
        for v in 2:n-1
            p2 = nodes[base_v + v].coord
            p3 = nodes[base_v + v + 1].coord
            cr = cross(p2 - p1, p3 - p1)
            if sum(abs2, cr) >= min_cross_sq
                base_t += 1
                elements[base_t] = Tri(base_v + 1, base_v + v, base_v + v + 1)
            end
        end
        base_v += n
    end
    resize!(elements, base_t)

    return Mesh(nodes, elements)
end


"""
One thread per entry of `cell_indices`. Clips only -- ordering is a separate pass
([`plic_sort_poly_kernel!`](@ref)), so `cell_plane_clip` stays reusable for callers that do not
need an ordered result. `points` is `6 x n_interface`: column `k` holds cell `k`'s clipped polygon
in world coordinates, unsorted and padded to 6 vertices, and `npoints[k]` says how many of those
rows are real. Each thread writes only its own column, so no atomics are needed.
"""
@kernel function plic_clip_poly_kernel!(points::AbstractMatrix{<:Point{3}},
                                        npoints::AbstractVector{<:Integer},
                                        cell_indices::AbstractVector{CartesianIndex{3}},
                                        normals::AbstractVector{<:SVector{3}},
                                        intercepts::AbstractVector{<:Real},
                                        grid::CartesianGrid{3}, dx::Real, dy::Real, dz::Real)
    k = @index(Global, Linear)
    @inbounds begin
        ci = cell_indices[k]
        poly, n = CartesianMeshes.cell_plane_clip(normals[k], intercepts[k], dx, dy, dz)
        npoints[k] = n
        for m in 1:6
            points[m, k] = Point(CartesianMeshes.local_to_global(grid, poly[m], ci))
        end
    end
end

"""
One thread per cell: orders that cell's real vertices -- `points[1:npoints[k], k]`, already clipped
and in world coordinates -- into a consistent winding around their centroid, using an in-plane 2D
basis derived from the cell's own normal. Cells with `npoints[k] < 3` (the plane only grazes an
edge or corner) and the padding slots beyond `npoints[k]` are left untouched.

Insertion sort over at most 6 elements with the angles precomputed into a fixed-size buffer, rather
than `sort!` with a per-comparison closure: KernelAbstractions' CPU backend boxes those, which is
allocation-heavy inside a kernel.
"""
@kernel function plic_sort_poly_kernel!(points::AbstractMatrix{<:Point{3}},
                                        npoints::AbstractVector{<:Integer},
                                        normals::AbstractVector{<:SVector{3,T}}) where {T}
    k = @index(Global, Linear)
    @inbounds begin
        n = npoints[k]
        if n >= 3
            normal = normals[k]

            centroid = zero(SVector{3,T})
            for m in 1:n
                centroid += points[m, k].coord
            end
            centroid = centroid / n

            ref = abs(normal[3]) < T(0.9) ? SVector{3,T}(0, 0, 1) : SVector{3,T}(1, 0, 0)
            u = normalize(cross(ref, normal))
            v = cross(normal, u)

            angles = MVector{6,T}(undef)
            for m in 1:n
                d = points[m, k].coord - centroid
                angles[m] = atan(dot(d, v), dot(d, u))
            end

            # insertion sort points[1:n, k] (and angles[1:n] alongside) by angle
            for i in 2:n
                key = points[i, k]
                key_angle = angles[i]
                j = i - 1
                while j >= 1 && angles[j] > key_angle
                    points[j+1, k] = points[j, k]
                    angles[j+1] = angles[j]
                    j -= 1
                end
                points[j+1, k] = key
                angles[j+1] = key_angle
            end
        end
    end
end

