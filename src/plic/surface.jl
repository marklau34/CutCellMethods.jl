"""
    generate_mesh(geo, grid::CartesianGrid{2}, method::PLICCutCell; tol=0) -> Mesh{2}
    generate_mesh(geo, grid::CartesianGrid{3}, method::PLICCutCell; tol=0, min_tri_area_fraction=1e-4) -> Mesh{3}

The [`PLICCutCell`](@ref) surface of `geo` over `grid`, as a `MeshLibrary.Mesh{D}` of `Line` (2D) or
`Tri` (3D) elements.

Two steps: fit each cell's plane and keep the cells it actually cuts, then hand those to
[`extract_surface_plic`](@ref), which clips them. The fit is one `get_sdf` per cell; the clip
is paid only on the cut band. `method` is last rather than first, to match
`MeshLibrary.generate_mesh` elsewhere.

`tol` is the cut-cell test's noise floor: a cell is drawn where `tol < volume_fraction < 1 - tol`,
which at the default `0` is exactly the geometric condition that the plane crosses the cell's
interior -- a plane grazing a corner, or a degenerate normal, gives exactly `0` or `1` and drops
out. Raise it to drop the slivers a plane passing just inside a corner still produces.
`min_tri_area_fraction` is 3D-only and passes straight through to `extract_surface_plic`.

**`grid` must be isotropic** (square cells in 2D, cubic in 3D), and this is the only place that is
checked -- once per call rather than per cell; [`cut_cell_moments`](@ref) assumes it, because it
has to stay launchable.

Each plane is fitted from one cell alone, so **the surface comes out as a field of disconnected
shards** rather than a stitched contour. For a watertight surface, reconstruct nodally with
[`MarchingSquaresCutCell`](@ref) or [`MarchingCubesCutCell`](@ref).

An empty result warns and returns an empty mesh rather than throwing: a grid that misses the body,
or lies wholly inside it, is a description of the grid rather than bad input. Runs on the host and
returns host data.
"""
function MeshLibrary.generate_mesh(geo, grid::CartesianGrid{2}, method::PLICCutCell; tol::Real=0)
    idx, normals, intercepts = _cut_planes(geo, grid, tol)
    return extract_surface_plic(normals, intercepts, idx, grid)
end

function MeshLibrary.generate_mesh(geo, grid::CartesianGrid{3}, method::PLICCutCell; tol::Real=0,
                                   min_tri_area_fraction::Real=1e-4)
    idx, normals, intercepts = _cut_planes(geo, grid, tol)
    return extract_surface_plic(normals, intercepts, idx, grid; min_tri_area_fraction)
end

"""
The cut cells and their planes, in the `(cell_indices, normals, intercepts)` shape
[`extract_surface_plic`](@ref) takes: one [`plic_fit`](@ref) per cell, kept where
`tol < volume_fraction < 1 - tol`.

`volume_fraction` is the **fluid** fraction, so this is the complement of `cell_plane`'s own
`fraction`; the test is symmetric in `tol` either way, but the fit is taken through `plic_fit` --
the same fit `cut_cell_moments(PLICCutCell(), ...)` stores -- so that the number tested here is
the one a consumer sees.

One threaded screening pass over the whole grid, then a serial gather that re-fits only the kept
cells, so nothing `O(volume)` is stored for the cells that are dropped. Host-only, like
[`generate_mesh`](@ref) itself -- it builds `Vector`s.
"""
function _cut_planes(geo, grid::CartesianGrid{D,T}, tol::Real) where {D,T}
    _require_isotropic(grid)
    lo = T(tol)
    hi = one(T) - lo
    all_ci = CartesianIndices(Tuple(grid.n))

    # The threaded half: one `Bool` per cell into a preallocated mask. Disjoint slots, no shared
    # accumulator, no order dependence -- unlike the gather below, where three `push!`es into
    # shared `Vector`s would race and scramble the element order the mesh is numbered by.
    #
    # A mask rather than the fits themselves: `normal` plus `intercept` is 32 bytes a cell in 3D
    # against one byte here, at the price of a second `get_sdf` on the cut band alone.
    keep = Vector{Bool}(undef, length(all_ci))
    if length(all_ci) > THREAD_FLOOR
        @batch for k in eachindex(keep)
            @inbounds keep[k] = lo < plic_fit(grid, geo, all_ci[k]).frac < hi
        end
    else
        for k in eachindex(keep)
            @inbounds keep[k] = lo < plic_fit(grid, geo, all_ci[k]).frac < hi
        end
    end

    n_cut = count(keep)
    idx = Vector{CartesianIndex{D}}(undef, n_cut)
    normals = Vector{SVector{D,T}}(undef, n_cut)
    intercepts = Vector{T}(undef, n_cut)
    j = 0
    for k in eachindex(keep)
        @inbounds keep[k] || continue
        ci = @inbounds all_ci[k]
        f = plic_fit(grid, geo, ci)
        j += 1
        @inbounds idx[j] = ci
        @inbounds normals[j] = f.normal
        @inbounds intercepts[j] = f.intercept
    end

    iszero(n_cut) && @warn "no cut cell was found anywhere in `grid` -- returning an empty mesh. The grid may miss the body entirely, or lie wholly inside it."
    return idx, normals, intercepts
end

# The one isotropy gate for the PLIC reconstruction: `generate_mesh` is the only entry point that
# can afford to throw, since `cut_cell_moments` has to stay launchable from device code.
@noinline function _require_isotropic(grid::CartesianGrid)
    CartesianMeshes.is_isotropic(grid) || throw(ArgumentError(
        "PLICCutCell needs an isotropic grid (equal cell size on every axis), got grid.d = $(Tuple(grid.d))"))
    return nothing
end


"""
    extract_surface_plic(normals, intercepts, cell_indices, grid::CartesianGrid{2}) -> Mesh{2}

The PLIC interface over `cell_indices` as a `MeshLibrary.Mesh{2}` of `Line` elements: one segment
per cell, clipped from that cell's plane by `CartesianMeshes.cell_plane_clip` and mapped into
`grid`'s frame. The half of [`generate_mesh`](@ref) that fits nothing itself -- a VOF solver
holding planes of its own passes them straight in.

- `cell_indices` -- the cells to draw, as `CartesianIndex{2}`es into `grid`, already compacted.
  Which cells those are is the caller's business: `generate_mesh` passes the cut cells its `tol`
  picked out, a VOF solver its interfacial cells. Nothing here re-tests them, and a plane that
  misses its cell entirely produces a degenerate segment rather than being dropped.
- `normals`/`intercepts` -- that plane, one entry per entry of `cell_indices` and **not** per grid
  cell, in that cell's own unit-cell frame, which is the frame `cell_plane_clip` works in.

Two nodes per element, never shared between elements: each plane was fitted from one cell with no
reference to its neighbours, so the surface is a field of disconnected shards and there is
deliberately no merge pass. The element type comes from `normals`/`intercepts`, with `grid`
converted to match, so a `Float32` field is not widened by a `Float64` grid.

The clip runs as a `KernelAbstractions` kernel over `cell_indices`, on whatever backend they live
on, but the `Mesh` it fills is host-resident, so the inputs must be host arrays.
"""
function extract_surface_plic(normals, intercepts, cell_indices::AbstractVector{CartesianIndex{2}},
                              grid::CartesianGrid{2})
    T = float(promote_type(eltype(eltype(normals)), eltype(intercepts)))
    g = convert(CartesianGrid{2,T}, grid)

    # An empty `cell_indices` is an ordinary state -- a field that is still everywhere pure has no
    # interface to draw -- but an empty `ndrange` is not a launch, so it returns an empty mesh
    # rather than reaching the kernel.
    n_interface = length(cell_indices)
    nodes = Vector{Point{2,T}}(undef, 2 * n_interface)
    if n_interface > 0
        backend = get_backend(normals)
        plic_clip_kernel!(backend, 64)(nodes, cell_indices, normals, intercepts, g, g.d[1], g.d[2];
                                       ndrange=n_interface)
    end

    elements = [Line(2k - 1, 2k) for k in 1:n_interface]

    return Mesh(nodes, elements)
end

"""
One thread per entry of `cell_indices` -- the already-compacted list, not one thread per grid cell.
`nodes[2k-1]`/`nodes[2k]` are cell `k`'s two endpoints, exactly the pair `Line(2k-1, 2k)` joins, so
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

