# =====================================
# Test bodies for tri clipping.
#
# Each generator returns a watertight, outward-wound `Mesh{3}` of `Tri` elements with one element
# set per patch -- the input `TriClippingCutCell` expects -- together with what is known about it
# analytically. Winding is fixed per patch against its known outward normal rather than assumed from
# the construction order, so a generator cannot silently hand back an inward face.

using MeshLibrary: Mesh, Point, Tri, MeshElementSet

"""A `Mesh{3}` from vertices, triangles and named triangle lists."""
function tm_mesh(X::Vector{SVector{3,Float64}}, tris::Vector{SVector{3,Int32}},
                 sets::Vector{Pair{String,Vector{Int}}})
    return Mesh([Point(x) for x in X], [Tri(c) for c in tris];
                element_sets=[MeshElementSet(name, elems) for (name, elems) in sets])
end

# Reverse triangle `t` if its normal points against `outward`.
function tm_orient(X, c::SVector{3,Int32}, outward::SVector{3,Float64})
    n = cross(X[c[2]] - X[c[1]], X[c[3]] - X[c[1]])
    return dot(n, outward) >= 0 ? c : SVector(c[1], c[3], c[2])
end

# A vertex table keyed on anything hashable, so shared points are one node.
struct TmVerts{K}
    X::Vector{SVector{3,Float64}}
    id::Dict{K,Int32}
end
TmVerts{K}() where {K} = TmVerts{K}(SVector{3,Float64}[], Dict{K,Int32}())
function tm_vert!(V::TmVerts, key, x::SVector{3,Float64})
    get!(V.id, key) do
        push!(V.X, x)
        Int32(length(V.X))
    end
end

"""
    tm_box(lo, hi; n=1, R=I, t=0) -> (mesh, volume, face_areas)

The box `lo..hi`, each face a grid of `n × n` quads split in two, one element set per face named
by direction (`"-x"`, `"+x"`, ...), then rotated by `R` about the origin and translated by `t`.
"""
function tm_box(lo::SVector{3,Float64}, hi::SVector{3,Float64}; n::Int=1,
                R=SMatrix{3,3,Float64}(I), t=zero(SVector{3,Float64}))
    V = TmVerts{NTuple{3,Int}}()
    tris = SVector{3,Int32}[]
    sets = Pair{String,Vector{Int}}[]
    names = ("-x", "+x", "-y", "+y", "-z", "+z")
    place(ijk) = R * (lo + (hi - lo) .* SVector{3,Float64}(ijk) / n) + t
    for dir in 1:6
        ax = (dir + 1) >> 1
        side = iseven(dir) ? n : 0
        p, q = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
        outward = R * SVector{3,Float64}(ntuple(a -> a == ax ? (iseven(dir) ? 1.0 : -1.0) : 0.0, 3))
        mine = Int[]
        for u in 0:(n - 1), v in 0:(n - 1)
            corner(du, dv) = begin
                ijk = MVector(0, 0, 0)
                ijk[ax] = side
                ijk[p] = u + du
                ijk[q] = v + dv
                key = Tuple(ijk)
                tm_vert!(V, key, place(key))
            end
            c1, c2, c3, c4 = corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)
            for c in (SVector(c1, c2, c3), SVector(c1, c3, c4))
                push!(tris, tm_orient(V.X, c, outward))
                push!(mine, length(tris))
            end
        end
        push!(sets, names[dir] => mine)
    end
    ext = hi - lo
    areas = Dict(names[d] => prod(ext) / ext[(d + 1) >> 1] for d in 1:6)
    return tm_mesh(V.X, tris, sets), prod(ext), areas
end

"""
    tm_prism(; L, beam, deadrise, depth, nx=4, x0=0, t=0) -> (mesh, volume, section_area)

A prismatic planing hull: a pentagonal section -- keel at `y = z = 0`, chines at
`y = ±beam/2, z = beam/2 * tan(deadrise)`, deck at `z = depth` -- from a planar transom at `x0` to
a flat bow at `x0 + L`. Seven planar patches: `transom`, `bow`, `bottom+`, `bottom-`, `side+`,
`side-`, `deck`.
"""
function tm_prism(; L::Float64=2.0, beam::Float64=1.0, deadrise::Float64=20.0, depth::Float64=0.8,
                  nx::Int=4, x0::Float64=0.0, t=zero(SVector{3,Float64}))
    b = beam / 2
    zc = b * tand(deadrise)
    # Section corners (y, z), counter-clockwise seen from +x.
    sec = (SVector(0.0, 0.0), SVector(b, zc), SVector(b, depth), SVector(-b, depth), SVector(-b, zc))
    panel_names = ("bottom+", "side+", "deck", "side-", "bottom-")
    V = TmVerts{NTuple{2,Int}}()
    tris = SVector{3,Int32}[]
    sets = Pair{String,Vector{Int}}[]
    xs = range(x0, x0 + L; length=nx + 1)
    vid(i, k) = tm_vert!(V, (i, k), SVector(xs[i], sec[k][1], sec[k][2]) + t)
    for k in 1:5
        k2 = k % 5 + 1
        e = sec[k2] - sec[k]
        outward = SVector(0.0, e[2], -e[1])   # the section is CCW about +x
        mine = Int[]
        for i in 1:nx
            a, bb, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k2), vid(i, k2)
            for tri in (SVector(a, bb, c), SVector(a, c, d))
                push!(tris, tm_orient(V.X, tri, outward))
                push!(mine, length(tris))
            end
        end
        push!(sets, panel_names[k] => mine)
    end
    for (i, name, outward) in ((1, "transom", SVector(-1.0, 0.0, 0.0)), (nx + 1, "bow", SVector(1.0, 0.0, 0.0)))
        mine = Int[]
        for k in 2:4
            push!(tris, tm_orient(V.X, SVector(vid(i, 1), vid(i, k), vid(i, k + 1)), outward))
            push!(mine, length(tris))
        end
        push!(sets, name => mine)
    end
    area = 0.0
    for k in 1:5
        p, q = sec[k], sec[k % 5 + 1]
        area += (p[1] * q[2] - q[1] * p[2]) / 2
    end
    return tm_mesh(V.X, tris, sets), area * L, area
end

"""
    tm_chine_prism(; L, b1, b2, z1, depth, nx=4, t=0) -> (mesh, volume, pieces)

A prismatic hull with chine flats: a V bottom from the keel (`y = z = 0`) to knuckles at
`y = ±b1, z = z1`, horizontal flats out to the chines at `y = ±b2`, vertical sides up to the deck
at `z = depth`, from a planar transom at `x = 0` to a flat bow at `x = L`. Each knuckle is a
concave crease and each chine a convex one, so a cell holding both meets the mixed rule. Patches:
`transom`, `bow`, `bottom±`, `flat±`, `side±`, `deck`. `pieces` are the three convex prisms whose
union it is -- the V-bottomed middle and the two outboard boxes -- as `(lo, hi)` sections for the
tests' exact reference.
"""
function tm_chine_prism(; L::Float64=1.4, b1::Float64=0.35, b2::Float64=0.42, z1::Float64=0.13,
                        depth::Float64=0.6, nx::Int=4, t=zero(SVector{3,Float64}))
    # Section (y, z), counter-clockwise seen from +x, with a deck midpoint to fan the caps from.
    sec = (SVector(0.0, 0.0), SVector(b1, z1), SVector(b2, z1), SVector(b2, depth),
           SVector(0.0, depth), SVector(-b2, depth), SVector(-b2, z1), SVector(-b1, z1))
    panel = ("bottom+", "flat+", "side+", "deck", "deck", "side-", "flat-", "bottom-")
    V = TmVerts{NTuple{2,Int}}()
    tris = SVector{3,Int32}[]
    byname = Dict{String,Vector{Int}}()
    xs = range(0.0, L; length=nx + 1)
    vid(i, k) = tm_vert!(V, (i, k), SVector(xs[i], sec[k][1], sec[k][2]) + t)
    ns = length(sec)
    for k in 1:ns
        k2 = k % ns + 1
        e = sec[k2] - sec[k]
        outward = SVector(0.0, e[2], -e[1])
        mine = get!(byname, panel[k], Int[])
        for i in 1:nx
            a, bb, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k2), vid(i, k2)
            for tri in (SVector(a, bb, c), SVector(a, c, d))
                push!(tris, tm_orient(V.X, tri, outward))
                push!(mine, length(tris))
            end
        end
    end
    for (i, name, outward) in ((1, "transom", SVector(-1.0, 0.0, 0.0)), (nx + 1, "bow", SVector(1.0, 0.0, 0.0)))
        mine = get!(byname, name, Int[])
        for k in (6, 7, 8, 1, 2, 3)   # the fan from the deck midpoint (corner 5)
            k2 = k % ns + 1
            push!(tris, tm_orient(V.X, SVector(vid(i, 5), vid(i, k), vid(i, k2)), outward))
            push!(mine, length(tris))
        end
    end
    sets = [name => byname[name] for name in sort(collect(keys(byname)))]
    middle = b1 * depth * 2 - b1 * z1          # V bottom to the deck, between the knuckles
    outboard = 2 * (b2 - b1) * (depth - z1)
    pieces = (mid=(b1=b1, z1=z1, depth=depth), box=((b1, b2), (z1, depth)))
    return tm_mesh(V.X, tris, sets), (middle + outboard) * L, pieces
end

"""
    tm_curved_hull(; L, b, a, depth, ny=48, nx=6, t=0) -> (mesh, section_area)

A hull with a curved bottom: the section is the parabola `z = a y^2` for `|y| <= b`, vertical sides
up to the deck at `z = depth`, from a planar transom at `x = 0` to a flat bow at `x = L`. The
bottom is one smooth patch faceted `ny` times across, so its crease with the transom is a curve --
the case the corner-line check measures at `O(h^2)`. Patches: `bottom`, `side+`, `side-`, `deck`,
`transom`, `bow`.
"""
function tm_curved_hull(; L::Float64=1.4, b::Float64=0.45, a::Float64=0.8, depth::Float64=0.6,
                        ny::Int=48, nx::Int=6, t=zero(SVector{3,Float64}))
    iseven(ny) || throw(ArgumentError("ny must be even, so the keel is a vertex"))
    # Section (y, z), counter-clockwise seen from +x: the bottom left to right, up the right side,
    # the deck right to left through its midpoint, down the left side.
    bottom = [SVector(y, a * y^2) for y in range(-b, b; length=ny + 1)]
    sec = vcat(bottom, [SVector(b, depth), SVector(0.0, depth), SVector(-b, depth)])
    ns = length(sec)
    mid = ns - 1                                  # the deck midpoint the caps fan from
    edge_name(k) = k <= ny ? "bottom" : k == ny + 1 ? "side+" : k <= ny + 3 ? "deck" : "side-"
    V = TmVerts{NTuple{2,Int}}()
    tris = SVector{3,Int32}[]
    byname = Dict{String,Vector{Int}}()
    xs = range(0.0, L; length=nx + 1)
    vid(i, k) = tm_vert!(V, (i, k), SVector(xs[i], sec[k][1], sec[k][2]) + t)
    for k in 1:ns
        k2 = k % ns + 1
        e = sec[k2] - sec[k]
        outward = SVector(0.0, e[2], -e[1])
        mine = get!(byname, edge_name(k), Int[])
        for i in 1:nx
            p, q, r, s = vid(i, k), vid(i + 1, k), vid(i + 1, k2), vid(i, k2)
            for tri in (SVector(p, q, r), SVector(p, r, s))
                push!(tris, tm_orient(V.X, tri, outward))
                push!(mine, length(tris))
            end
        end
    end
    for (i, name, outward) in ((1, "transom", SVector(-1.0, 0.0, 0.0)), (nx + 1, "bow", SVector(1.0, 0.0, 0.0)))
        mine = get!(byname, name, Int[])
        for k in 1:ns
            k2 = k % ns + 1
            (k == mid || k2 == mid) && continue    # the two deck edges at the fan's own vertex
            push!(tris, tm_orient(V.X, SVector(vid(i, mid), vid(i, k), vid(i, k2)), outward))
            push!(mine, length(tris))
        end
    end
    sets = [name => byname[name] for name in sort(collect(keys(byname)))]
    return tm_mesh(V.X, tris, sets), 2b * depth - 2a * b^3 / 3
end

"""
    tm_icosphere(; r=1, c=0, level=3, split=false) -> (mesh, volume, area)

A sphere of radius `r` about `c`: an icosahedron subdivided `level` times with its new vertices
pushed onto the sphere. One patch, or with `split` two -- `north` and `south` by triangle centroid
-- whose seam is a smooth join, the tangent case. `volume` and `area` are the exact sphere's.
"""
function tm_icosphere(; r::Float64=1.0, c=zero(SVector{3,Float64}), level::Int=3, split::Bool=false)
    φ = (1 + sqrt(5)) / 2
    P = [SVector(-1.0, φ, 0), SVector(1.0, φ, 0), SVector(-1.0, -φ, 0), SVector(1.0, -φ, 0),
         SVector(0.0, -1, φ), SVector(0.0, 1, φ), SVector(0.0, -1, -φ), SVector(0.0, 1, -φ),
         SVector(φ, 0, -1), SVector(φ, 0, 1), SVector(-φ, 0, -1), SVector(-φ, 0, 1)]
    P = [normalize(p) for p in P]
    F = [(1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12), (2, 6, 10), (6, 12, 5),
         (12, 11, 3), (11, 8, 7), (8, 2, 9), (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9),
         (4, 9, 10), (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2)]
    for _ in 1:level
        mid = Dict{Tuple{Int,Int},Int}()
        m(a, b) = get!(mid, minmax(a, b)) do
            push!(P, normalize(P[a] + P[b]))
            length(P)
        end
        F = [f for (a, b, cc) in F for f in (let ab = m(a, b), bc = m(b, cc), ca = m(cc, a)
                                                 ((a, ab, ca), (b, bc, ab), (cc, ca, bc), (ab, bc, ca))
                                             end)]
    end
    X = [r * p + c for p in P]
    tris = [tm_orient(X, SVector{3,Int32}(f...), (X[f[1]] + X[f[2]] + X[f[3]]) / 3 - c) for f in F]
    if split
        north = [i for (i, f) in enumerate(tris) if sum(X[f[k]][3] for k in 1:3) / 3 >= c[3]]
        south = setdiff(1:length(tris), north)
        sets = ["north" => north, "south" => south]
    else
        sets = ["sphere" => collect(1:length(tris))]
    end
    return tm_mesh(X, tris, sets), 4π * r^3 / 3, 4π * r^2
end

"""
    tm_lblock(; a=1, hz=1, t=0) -> (mesh, volume)

An L-shaped prism: the section `[0,2a]×[0,a] ∪ [0,a]×[0,2a]` extruded over `z ∈ [0, hz]`, one
patch per planar face. The face pair meeting at the inner corner `(a, a)` is the concave crease.
"""
function tm_lblock(; a::Float64=1.0, hz::Float64=1.0, t=zero(SVector{3,Float64}))
    sec = (SVector(0.0, 0.0), SVector(2a, 0.0), SVector(2a, a), SVector(a, a), SVector(a, 2a), SVector(0.0, 2a))
    names = ("y=0", "x=2a", "y=a", "x=a", "y=2a", "x=0")
    V = TmVerts{NTuple{2,Int}}()
    vid(k, top) = tm_vert!(V, (k, top), SVector(sec[k][1], sec[k][2], top * hz) + t)
    tris = SVector{3,Int32}[]
    sets = Pair{String,Vector{Int}}[]
    for k in 1:6
        k2 = k % 6 + 1
        e = sec[k2] - sec[k]
        outward = SVector(e[2], -e[1], 0.0)   # the section is CCW about +z
        mine = Int[]
        for tri in (SVector(vid(k, 0), vid(k2, 0), vid(k2, 1)), SVector(vid(k, 0), vid(k2, 1), vid(k, 1)))
            push!(tris, tm_orient(V.X, tri, outward))
            push!(mine, length(tris))
        end
        push!(sets, names[k] => mine)
    end
    for (top, name, outward) in ((0, "z=0", SVector(0.0, 0.0, -1.0)), (1, "z=hz", SVector(0.0, 0.0, 1.0)))
        mine = Int[]
        for (i, j, k) in ((1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6))
            push!(tris, tm_orient(V.X, SVector(vid(i, top), vid(j, top), vid(k, top)), outward))
            push!(mine, length(tris))
        end
        push!(sets, name => mine)
    end
    return tm_mesh(V.X, tris, sets), 3a^2 * hz
end

"""
    tm_frame(n) -> SMatrix{3,3}

A rotation taking `+x` to the unit vector `n`: its columns are `n` and two unit vectors completing
a right-handed frame.
"""
function tm_frame(n::SVector{3,Float64})
    u = normalize(cross(n, abs(n[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)))
    return SMatrix{3,3,Float64}(hcat(n, u, cross(n, u)))
end

"""
    tm_pyramid(; apex, axis, k, height, radius, θ0=0) -> (mesh, planes, volume)

A right pyramid over a regular `k`-gon: its apex at `apex`, its base `height` back along `axis` and
of circumradius `radius`, first corner at angle `θ0`. `k` planar side patches `side1`...`sidek` meet
at the apex -- `k` convex creases in the one cell holding it -- and a `base` patch closes it. Each
side is two triangles, split at its base edge's midpoint. `planes` are the `k + 1` outward
`(n, d)`, solid where `n . x <= d`, sides first.
"""
function tm_pyramid(; apex::SVector{3,Float64}, axis::SVector{3,Float64}, k::Int, height::Float64,
                    radius::Float64, θ0::Float64=0.0)
    R = tm_frame(normalize(axis))
    ax, u, v = R[:, 1], R[:, 2], R[:, 3]
    c = apex - height * ax
    B = [c + radius * (cos(θ0 + 2π * i / k) * u + sin(θ0 + 2π * i / k) * v) for i in 0:(k - 1)]
    # Nodes: the apex, the base centre, then each base corner followed by its edge's midpoint.
    X = SVector{3,Float64}[apex, c]
    for i in 1:k
        push!(X, B[i], (B[i] + B[i % k + 1]) / 2)
    end
    corner(i) = Int32(1 + 2i)
    edge_mid(i) = Int32(2 + 2i)
    centroid = (apex + 3c) / 4
    tris = SVector{3,Int32}[]
    sets = Pair{String,Vector{Int}}[]
    planes = Tuple{SVector{3,Float64},Float64}[]
    for i in 1:k
        j = i % k + 1
        mine = Int[]
        for tri in (SVector{3,Int32}(1, corner(i), edge_mid(i)), SVector{3,Int32}(1, edge_mid(i), corner(j)))
            push!(tris, tm_orient(X, tri, (apex + B[i] + B[j]) / 3 - centroid))
            push!(mine, length(tris))
        end
        push!(sets, "side$i" => mine)
        n = normalize(cross(B[i] - apex, B[j] - apex))
        dot(n, centroid - apex) > 0 && (n = -n)
        push!(planes, (n, dot(n, apex)))
    end
    base = Int[]
    for i in 1:k
        for tri in (SVector{3,Int32}(2, corner(i), edge_mid(i)), SVector{3,Int32}(2, edge_mid(i), corner(i % k + 1)))
            push!(tris, tm_orient(X, tri, -ax))
            push!(base, length(tris))
        end
    end
    push!(sets, "base" => base)
    push!(planes, (-ax, dot(-ax, c)))
    return tm_mesh(X, tris, sets), planes, k / 2 * radius^2 * sin(2π / k) * height / 3
end

# =====================================
# Exact references
#
# A convex body as its outward planes `(n, d)`, solid where `n . x <= d`, and the solid fraction of
# a cell cut by them -- clipped with the body's own analytic planes, not the method's fitted ones.

"""The solid fraction of cell `ci` of `g` inside every `(n, d)` of `planes`, on the clipper's
scratch `scr`: the cell box clipped by each plane in turn."""
function tm_solid_fraction(scr, g, ci, planes)
    o = get_node(g, ci)
    U = get_node(g, ci + CartesianIndex(1, 1, 1)) - o
    CutCellMethods.poly_box!(scr, 1, U)
    for (n, d) in planes
        CutCellMethods.poly_clip!(scr, 1, n, d - dot(n, o), 7, 1e-12, false)
    end
    return first(CutCellMethods.poly_volume_moment(scr, 1, U / 2)) / prod(U)
end

"""The box `lo..hi` of `tm_box`, rotated by `R` and moved by `t`, as its six outward planes in its
element-set order (`-x`, `+x`, `-y`, ...)."""
function tm_box_planes(lo, hi, R=SMatrix{3,3,Float64}(I), t=zero(SVector{3,Float64}))
    P = Tuple{SVector{3,Float64},Float64}[]
    for ax in 1:3, sgn in (-1, 1)
        n = R * SVector(ntuple(a -> a == ax ? Float64(sgn) : 0.0, 3))
        push!(P, (n, dot(n, R * (sgn > 0 ? hi : lo) + t)))
    end
    return P
end

"""The prism hull of `tm_prism` as its seven outward planes."""
function tm_prism_planes(; L, beam, deadrise, depth, t)
    b = beam / 2
    zc = b * tand(deadrise)
    sec = (SVector(0.0, 0.0), SVector(b, zc), SVector(b, depth), SVector(-b, depth), SVector(-b, zc))
    return tm_yz_prism(sec, L, t)
end

"""A convex prism along x over `[t[1], t[1] + L]` whose section is `sec`, (y, z) corners
counter-clockwise seen from +x, offset by `t`, as its outward planes."""
function tm_yz_prism(sec, L, t)
    P = Tuple{SVector{3,Float64},Float64}[]
    for k in eachindex(sec)
        p, q = sec[k], sec[k % length(sec) + 1]
        e = q - p
        n = normalize(SVector(0.0, e[2], -e[1]))
        push!(P, (n, dot(n, SVector(0.0, p[1], p[2]) + t)))
    end
    push!(P, (SVector(-1.0, 0.0, 0.0), -t[1]))
    push!(P, (SVector(1.0, 0.0, 0.0), t[1] + L))
    return P
end

"""
    tm_gpph() -> Mesh or nothing

The GPPH planing hull, cleaned of its transom fillets, as `examples/geometry/run_generate_mesh.jl`
writes it from `gpph_clean.step`: an Abaqus `.inp` with one element set per CAD surface
(`surface_1` ... `surface_8`: port and starboard bottom, chine flat and side, the transom and the
deck), read as triangles. `nothing` if the file has not been generated, so the tests that use it
skip rather than fail. AbaqusReader calls gmsh's CPS3 triangles `:Tri3`.
"""
function tm_gpph()
    inp = joinpath(@__DIR__, "..", "examples", "geometry", "gpph_clean.inp")
    isfile(inp) || return nothing
    return MeshLibrary.load_mesh(inp; elem_types=[:Tri3], element_sets=["surface_$i" for i in 1:8])
end

# The same mesh with its triangles replaced, keeping the nodes and (by default) the element sets.
function tm_retri(mesh, tris::Vector{SVector{3,Int32}}; sets=mesh.elemset)
    return Mesh(collect(mesh.nodes), [Tri(c) for c in tris]; element_sets=sets)
end
tm_tris(mesh) = SVector{3,Int32}[SVector{3,Int32}(e.con) for e in mesh.elements]
tm_coords(mesh) = SVector{3,Float64}[SVector{3,Float64}(c) for c in mesh.nodes.coord]
