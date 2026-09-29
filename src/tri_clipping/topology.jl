# =====================================
# The mesh's topology: validation, patches, and how patches meet
#
# Built on the host from `mesh.elements` and `mesh.elemset` in `Float64`, and cached by the
# `TriClippingCutCell` cache between updates: none of it changes when the body moves rigidly, so a
# moving hull pays for it once. Global-plane *coefficients* are the exception -- they move with the
# body -- and are refit from the current vertices on every update (`_patch_planes`).
#
# Validation is strict because everything downstream assumes it. The ray-cast classification of
# uncut cells needs a closed surface; the Boolean rules need every edge between two patches to
# have exactly one neighbour across it; the half-space senses need the triangles wound outwards.

# A triangle whose normal cannot be trusted for a convexity sign: its area is tiny against its
# longest edge. Such a triangle defers to its neighbour when an edge's sense is decided.
const TOPO_SLIVER_QUALITY = 1e-8
# An edge label with neither triangle's normal trustworthy: recorded, but carrying no sense.
const PAIR_UNKNOWN = 0x08
# A patch whose internal dihedral exceeds this is warned about: a crease inside one patch is
# fitted as a smooth surface and so rounded off.
const TOPO_WARN_INTERNAL_DIHEDRAL = 20.0

"""
    TriTopology

What [`TriClippingCutCell`](@ref)'s cache knows about a mesh's connectivity, built once per mesh by
[`build_topology`](@ref):

- `tri_patch` -- each triangle's patch. A patch is one edge-connected component of one element
  set, so a set holding both sides of a hull is two patches with the set's name.
- `side_nbr` / `side_patch` / `side_label` -- per triangle side `s` (the edge from vertex `s` to
  vertex `s % 3 + 1`), the triangle across it, that triangle's patch, and how the two patches meet
  there: `0` inside one patch, else `PAIR_CONVEX`, `PAIR_CONCAVE` or `PAIR_TANGENT`, or
  `PAIR_UNKNOWN` where neither triangle's normal can be trusted.
- `patch_name`, `patch_set`, `patch_ntri`, `patch_area` -- per patch.
- `patch_planarity` -- the largest distance of a patch's vertices from its best-fit plane, and
  `patch_planar` whether the patch is cut by one global plane: within `planar_tol` of flat, or
  named in `planar_patches`.
- `pair_mask` -- every label seen on the edges between two patches, OR'd: the fallback for a cell
  none of whose own edges says how its patches meet.
- `volume` -- the mesh's own divergence-theorem volume, the reference the cache's totals are
  checked against.
"""
struct TriTopology
    fingerprint::UInt
    nvert::Int
    ntri::Int
    tri_patch::Vector{Int32}
    side_nbr::Vector{SVector{3,Int32}}
    side_patch::Vector{SVector{3,Int32}}
    side_label::Vector{SVector{3,UInt8}}
    patch_name::Vector{String}
    patch_set::Vector{Int32}
    patch_ntri::Vector{Int32}
    patch_area::Vector{Float64}
    patch_planarity::Vector{Float64}
    patch_planar::Vector{Bool}
    pair_mask::Matrix{UInt8}
    volume::Float64
end

npatches(topo::TriTopology) = length(topo.patch_name)

# What identifies a mesh's topology between updates: a hash of its connectivity and its sets'
# contents. By content rather than identity, since a moving body is naturally a new `Mesh` wrapped
# around the same connectivity each step; linear in the mesh and cheap beside the cut itself. The
# coordinates are deliberately not part of it -- they change every step.
function _topology_fingerprint(mesh::Mesh)
    h = hash(length(mesh.elements), hash(:tri_topology))
    for e in mesh.elements
        h = hash(e.con, h)
    end
    for set in mesh.elemset
        h = hash(set.name, h)
        for i in set.elems
            h = hash(i, h)
        end
    end
    return h
end

function _mesh_coords(mesh::Mesh{3})
    return SVector{3,Float64}[SVector{3,Float64}(c) for c in mesh.nodes.coord]
end

function _mesh_tris(mesh::Mesh{3})
    eltype(mesh.elements) <: Tri || throw(ArgumentError(
        "TriClippingCutCell needs a mesh of `Tri` elements, got $(eltype(mesh.elements))"))
    return SVector{3,Int32}[SVector{3,Int32}(e.con) for e in mesh.elements]
end
_mesh_tris(mesh::Mesh) = throw(ArgumentError(
    "TriClippingCutCell needs a 3D triangle mesh, got a $(ndims_of(mesh))D one"))
ndims_of(::Mesh{D}) where {D} = D

"""
    build_topology(mesh, method, h) -> TriTopology

Validate `mesh` and build its [`TriTopology`](@ref). Throws an `ArgumentError` naming the first
problem found: a degenerate triangle, an open or non-manifold edge, inconsistent winding, an
inward-wound (negative-volume) surface, or element sets that do not partition the triangles.

`h` is the grid's smallest cell size, which `method.planar_tol` is relative to.
"""
function build_topology(mesh::Mesh, method::TriClippingCutCell, h::Real)
    tris = _mesh_tris(mesh)
    X = _mesh_coords(mesh)
    nv = length(X)
    nt = length(tris)

    for (t, c) in enumerate(tris)
        all(i -> 1 <= c[i] <= nv, 1:3) || throw(ArgumentError(
            "triangle $t references a node outside 1:$nv"))
        (c[1] == c[2] || c[2] == c[3] || c[3] == c[1]) && throw(ArgumentError(
            "triangle $t repeats a node ($c): remove collapsed triangles before cutting"))
    end

    side_nbr = _match_edges(tris, nt)
    volume = _mesh_volume(X, tris)
    volume > 0 || throw(ArgumentError(
        "the mesh encloses a negative volume ($volume): it is wound inwards. Its triangles must be " *
        "wound so their normals point out of the body (MeshLibrary's `flip_normals!` reverses them)"))

    tri_patch, patch_set, patch_name = _partition_patches(mesh, side_nbr, nt)
    np = length(patch_name)

    tn, ta, tq = _tri_geometry(X, tris)
    side_patch = Vector{SVector{3,Int32}}(undef, nt)
    side_label = Vector{SVector{3,UInt8}}(undef, nt)
    pair_mask = zeros(UInt8, np, np)
    internal = zeros(np)
    cos_tangent = cosd(method.tangent_angle)
    for t in 1:nt
        p = tri_patch[t]
        sp = MVector{3,Int32}(0, 0, 0)
        sl = MVector{3,UInt8}(0, 0, 0)
        for s in 1:3
            u = Int(side_nbr[t][s])
            q = tri_patch[u]
            cosθ = clamp(dot(tn[t], tn[u]), -1.0, 1.0)
            if q == p
                internal[p] = max(internal[p], acosd(cosθ))
                continue
            end
            lab = _edge_label(X, tris, tn, tq, t, u, s, cosθ, cos_tangent)
            sp[s] = q
            sl[s] = lab
            pair_mask[p, q] |= lab
            pair_mask[q, p] |= lab
        end
        side_patch[t] = SVector(sp)
        side_label[t] = SVector(sl)
    end

    patch_ntri = zeros(Int32, np)
    patch_area = zeros(np)
    for t in 1:nt
        patch_ntri[tri_patch[t]] += 1
        patch_area[tri_patch[t]] += ta[t]
    end
    planarity = _patch_planarity(X, tris, tn, ta, tri_patch, np)
    planar_names = Set(method.planar_patches)
    patch_planar = [planarity[p] <= method.planar_tol * h || patch_name[p] in planar_names
                    for p in 1:np]
    for name in planar_names
        name in patch_name || @warn "planar_patches names \"$name\", which is not an element set of the mesh"
    end

    _warn_patches(patch_name, patch_ntri, internal, tq, tri_patch)

    return TriTopology(_topology_fingerprint(mesh), nv, nt, tri_patch, side_nbr, side_patch,
                       side_label, patch_name, patch_set, patch_ntri, patch_area, planarity,
                       patch_planar, pair_mask, volume)
end

# Every edge must be shared by exactly two triangles traversing it in opposite directions: that is
# closed, manifold and consistently wound at once. Returns each triangle side's neighbour.
function _match_edges(tris::Vector{SVector{3,Int32}}, nt::Int)
    keys = Vector{UInt64}(undef, 3nt)
    for t in 1:nt, s in 1:3
        a = UInt64(tris[t][s])
        b = UInt64(tris[t][s % 3 + 1])
        keys[3(t - 1) + s] = (min(a, b) << 32) | max(a, b)
    end
    perm = sortperm(keys)
    side_nbr = [MVector{3,Int32}(0, 0, 0) for _ in 1:nt]
    i = 1
    while i <= 3nt
        j = i
        while j < 3nt && keys[perm[j + 1]] == keys[perm[i]]
            j += 1
        end
        k = j - i + 1
        e1 = perm[i]
        t1, s1 = divrem(e1 - 1, 3) .+ (1, 1)
        a, b = tris[t1][s1], tris[t1][s1 % 3 + 1]
        k == 1 && throw(ArgumentError(
            "the mesh is not closed: edge ($a, $b) of triangle $t1 has no neighbour. A cut needs a " *
            "watertight surface -- duplicate nodes left unmerged also open it"))
        k > 2 && throw(ArgumentError(
            "the mesh is not manifold: edge ($a, $b) is shared by $k triangles"))
        e2 = perm[j]
        t2, s2 = divrem(e2 - 1, 3) .+ (1, 1)
        # Opposite traversal: the second triangle runs the edge b -> a.
        tris[t2][s2] == b || throw(ArgumentError(
            "the mesh is not consistently wound: triangles $t1 and $t2 traverse edge ($a, $b) in " *
            "the same direction, so one of them faces the wrong way"))
        side_nbr[t1][s1] = t2
        side_nbr[t2][s2] = t1
        i = j + 1
    end
    return SVector{3,Int32}.(side_nbr)
end

function _mesh_volume(X, tris)
    V = 0.0
    for c in tris
        V += dot(X[c[1]], cross(X[c[2]], X[c[3]]))
    end
    return V / 6
end

# Unit normals, areas and a quality -- twice the area over the longest edge squared, 0.87 for an
# equilateral triangle and 0 for a degenerate one.
function _tri_geometry(X, tris)
    nt = length(tris)
    tn = Vector{SVector{3,Float64}}(undef, nt)
    ta = Vector{Float64}(undef, nt)
    tq = Vector{Float64}(undef, nt)
    for (t, c) in enumerate(tris)
        p1, p2, p3 = X[c[1]], X[c[2]], X[c[3]]
        cr = cross(p2 - p1, p3 - p1)
        a2 = norm(cr)
        l2 = max(sum(abs2, p2 - p1), sum(abs2, p3 - p2), sum(abs2, p1 - p3))
        tn[t] = a2 > 0 ? cr / a2 : zero(SVector{3,Float64})
        ta[t] = a2 / 2
        tq[t] = l2 > 0 ? a2 / l2 : 0.0
    end
    return tn, ta, tq
end

# How the patches of triangles `t` and `u` meet across side `s` of `t`. The sense is read off the
# better-shaped triangle: the far vertex of the other one lies below its plane at a convex edge.
function _edge_label(X, tris, tn, tq, t, u, s, cosθ, cos_tangent)
    cosθ > cos_tangent && return PAIR_TANGENT
    max(tq[t], tq[u]) < TOPO_SLIVER_QUALITY && return PAIR_UNKNOWN
    c = tris[t]
    a = c[s]
    b = c[s % 3 + 1]
    far_t = c[(s + 1) % 3 + 1]
    cu = tris[u]
    far_u = cu[findfirst(v -> v != a && v != b, cu)]
    σ = tq[t] >= tq[u] ? dot(tn[t], X[far_u] - X[a]) : dot(tn[u], X[far_t] - X[a])
    return σ < 0 ? PAIR_CONVEX : PAIR_CONCAVE
end

# Element sets -> patches: each triangle in exactly one set, and each set split into its
# edge-connected components.
function _partition_patches(mesh, side_nbr, nt)
    sets = mesh.elemset
    owner = zeros(Int32, nt)
    set_names = String[]
    if isempty(sets)
        @warn "the mesh has no element sets, so it is cut as a single smooth patch and any crease " *
              "is rounded off; give each smooth patch its own `MeshElementSet` in `mesh.elemset`"
        owner .= 1
        push!(set_names, "body")
    else
        for (k, set) in enumerate(sets)
            push!(set_names, set.name)
            for e in set.elems
                1 <= e <= nt || throw(ArgumentError(
                    "element set \"$(set.name)\" lists element $e, outside 1:$nt"))
                owner[e] == 0 || throw(ArgumentError(
                    "element $e is in two element sets, \"$(sets[owner[e]].name)\" and " *
                    "\"$(set.name)\": the sets must partition the triangles"))
                owner[e] = k
            end
        end
        missing_tri = findfirst(iszero, owner)
        missing_tri === nothing || throw(ArgumentError(
            "$(count(iszero, owner)) triangles are in no element set (the first is $missing_tri): " *
            "the sets must partition the triangles"))
    end

    tri_patch = zeros(Int32, nt)
    patch_set = Int32[]
    patch_name = String[]
    stack = Int[]
    for t0 in 1:nt
        tri_patch[t0] == 0 || continue
        push!(patch_set, owner[t0])
        push!(patch_name, set_names[owner[t0]])
        p = Int32(length(patch_set))
        tri_patch[t0] = p
        push!(stack, t0)
        while !isempty(stack)
            t = pop!(stack)
            for s in 1:3
                u = side_nbr[t][s]
                if tri_patch[u] == 0 && owner[u] == owner[t]
                    tri_patch[u] = p
                    push!(stack, u)
                end
            end
        end
    end
    return tri_patch, patch_set, patch_name
end

# The largest distance of each patch's vertices from its area-weighted best-fit plane. A patch
# that folds back on itself (a sphere as one patch) has no plane to be near: Inf.
function _patch_planarity(X, tris, tn, ta, tri_patch, np)
    N = zeros(SVector{3,Float64}, np)
    C = zeros(SVector{3,Float64}, np)
    A = zeros(np)
    for (t, c) in enumerate(tris)
        p = tri_patch[t]
        N[p] += ta[t] * tn[t]
        C[p] += ta[t] * (X[c[1]] + X[c[2]] + X[c[3]]) / 3
        A[p] += ta[t]
    end
    planarity = zeros(np)
    for p in 1:np
        nN = norm(N[p])
        if nN < 0.5 * A[p] || A[p] == 0
            planarity[p] = Inf
        end
    end
    for (t, c) in enumerate(tris)
        p = tri_patch[t]
        isinf(planarity[p]) && continue
        nhat = N[p] / norm(N[p])
        centre = C[p] / A[p]
        for i in 1:3
            planarity[p] = max(planarity[p], abs(dot(nhat, X[c[i]] - centre)))
        end
    end
    return planarity
end

function _warn_patches(patch_name, patch_ntri, internal, tq, tri_patch)
    msgs = String[]
    for p in eachindex(patch_name)
        if internal[p] > TOPO_WARN_INTERNAL_DIHEDRAL
            push!(msgs, "patch \"$(patch_name[p])\" (#$p) bends by $(round(internal[p]; digits=1))° " *
                        "across one of its own edges: a crease inside a patch is rounded off")
        end
        if patch_ntri[p] == 1
            push!(msgs, "patch \"$(patch_name[p])\" (#$p) is a single triangle")
        end
    end
    sliver_only = trues(length(patch_name))
    for (t, p) in enumerate(tri_patch)
        tq[t] >= TOPO_SLIVER_QUALITY && (sliver_only[p] = false)
    end
    for p in eachindex(patch_name)
        sliver_only[p] && push!(msgs, "patch \"$(patch_name[p])\" (#$p) is made only of sliver triangles")
    end
    isempty(msgs) && return nothing
    shown = first(msgs, 10)
    more = length(msgs) > 10 ? "\n  ... and $(length(msgs) - 10) more" : ""
    @warn "TriClippingCutCell: patch checks\n  " * join(shown, "\n  ") * more
    return nothing
end

"""
    _patch_planes(topo, X) -> Vector{SVector{4,Float64}}

The current global plane `(n..., d)` of every planar patch -- `n` its unit normal out of the body,
`n . x = d` on it -- from the vertices `X` as they are now, and zeros for the others.
"""
function _patch_planes(topo::TriTopology, X::AbstractVector{SVector{3,Float64}},
                       tris::AbstractVector{SVector{3,Int32}})
    np = npatches(topo)
    N = zeros(SVector{3,Float64}, np)
    C = zeros(SVector{3,Float64}, np)
    A = zeros(np)
    for (t, c) in enumerate(tris)
        p = topo.tri_patch[t]
        topo.patch_planar[p] || continue
        p1, p2, p3 = X[c[1]], X[c[2]], X[c[3]]
        cr = cross(p2 - p1, p3 - p1)
        N[p] += cr / 2
        a = norm(cr) / 2
        C[p] += a * (p1 + p2 + p3) / 3
        A[p] += a
    end
    planes = zeros(SVector{4,Float64}, np)
    for p in 1:np
        (topo.patch_planar[p] && A[p] > 0) || continue
        nhat = N[p] / norm(N[p])
        planes[p] = SVector(nhat[1], nhat[2], nhat[3], dot(nhat, C[p] / A[p]))
    end
    return planes
end
