# =====================================
# Closed-polyline test bodies for `PolylineClippingCutCell`
#
# Each generator returns the points of one loop, counter-clockwise (solid on the left), and
# `poly_mesh` turns loops into the `Mesh{2}` of `Line` elements the method takes, with one element
# set per loop. The invalid meshes at the bottom each break exactly one of the checks the topology
# makes.

using MeshLibrary
using StaticArrays

const PV = SVector{2,Float64}

@inline _rot(θ) = SMatrix{2,2,Float64}(cos(θ), sin(θ), -sin(θ), cos(θ))

"""
    poly_mesh(loops; names=nothing, perm=nothing) -> Mesh{2}

One `Line` per consecutive pair of points of each loop, closing it; loop `k`'s elements form the
element set `names[k]` (default `"loop k"`). `perm` shuffles the element order -- the topology must
not depend on it.
"""
function poly_mesh(loops::AbstractVector; names=nothing, perm=nothing)
    nodes = Point{2,Float64}[]
    elems = Line{Int32}[]
    sets = Vector{Int}[]
    for pts in loops
        base = length(nodes)
        n = length(pts)
        for p in pts
            push!(nodes, Point(PV(p)))
        end
        first_e = length(elems) + 1
        for i in 1:n
            push!(elems, Line(Int32(base + i), Int32(base + mod1(i + 1, n))))
        end
        push!(sets, collect(first_e:length(elems)))
    end
    if perm !== nothing
        inv = invperm(perm)
        elems = elems[perm]
        sets = [sort(inv[s]) for s in sets]
    end
    mesh = Mesh(nodes, elems)
    for (k, s) in enumerate(sets)
        push!(mesh.elemset, MeshElementSet(names === nothing ? "loop $k" : names[k], s))
    end
    return mesh
end
poly_mesh(pts::AbstractVector{<:SVector{2}}; kwargs...) = poly_mesh([pts]; kwargs...)

"""A square of side `a` centred at `c`, rotated by `θ`."""
square_pts(c, a; θ=0.0) =
    [PV(c) + _rot(θ) * PV(s) for s in ((-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2))]

"""An axis-aligned rectangle from `lo` to `hi`, corners exactly as given."""
rect_pts(lo, hi) = [PV(lo[1], lo[2]), PV(hi[1], lo[2]), PV(hi[1], hi[2]), PV(lo[1], hi[2])]

"""A regular `n`-gon of circumradius `r` about `c`, first vertex at angle `θ0`."""
ngon_pts(c, r, n; θ0=0.0) = [PV(c) + r * PV(cos(θ0 + 2π * k / n), sin(θ0 + 2π * k / n)) for k in 0:(n - 1)]

ngon_area(r, n) = n / 2 * r^2 * sin(2π / n)

"""
    naca4_pts(; m=0.0, p=0.0, t=0.12, n=80, chord=1.0, le=(0, 0), α=0.0)

A NACA 4-digit section with a **sharp** trailing edge (the closed-TE thickness coefficient), `n`
cosine-spaced stations per surface. Counter-clockwise: from the trailing edge along the upper surface
to the leading edge, and back along the lower. Rotated nose-up by `α` about the leading edge, which is
placed at `le`.
"""
function naca4_pts(; m=0.0, p=0.0, t=0.12, n=80, chord=1.0, le=(0.0, 0.0), α=0.0)
    yt(x) = 5t * (0.2969sqrt(x) - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    yc(x) = m == 0 ? 0.0 : x < p ? m / p^2 * (2p * x - x^2) : m / (1 - p)^2 * ((1 - 2p) + 2p * x - x^2)
    dyc(x) = m == 0 ? 0.0 : x < p ? 2m / p^2 * (p - x) : 2m / (1 - p)^2 * (p - x)
    xs = [(1 - cos(π * k / n)) / 2 for k in 0:n]     # 0 (LE) .. 1 (TE)
    surf(x, s) = (θ = atan(dyc(x)); PV(x - s * yt(x) * sin(θ), yc(x) + s * yt(x) * cos(θ)))
    upper = [surf(x, 1) for x in reverse(xs)]         # TE -> LE
    lower = [surf(x, -1) for x in xs[2:(end - 1)]]    # LE -> TE, both ends shared
    R = _rot(-α)
    return [PV(le) + chord * (R * q) for q in vcat(upper, lower)]
end

"""A thin plate: a rectangle `len` by `thick` centred at `c`, rotated by `θ`."""
plate_pts(c, len, thick; θ=0.0) =
    [PV(c) + _rot(θ) * PV(s) for s in ((-len / 2, -thick / 2), (len / 2, -thick / 2),
                                       (len / 2, thick / 2), (-len / 2, thick / 2))]

"""A slat, main element and flap, from NACA sections, well separated."""
function three_element_pts()
    slat = naca4_pts(; m=0.02, p=0.4, t=0.10, n=30, chord=0.18, le=(-0.20, -0.06), α=0.45)
    main = naca4_pts(; m=0.02, p=0.4, t=0.12, n=80, chord=1.0, le=(0.0, 0.0), α=0.0)
    flap = naca4_pts(; m=0.02, p=0.4, t=0.10, n=40, chord=0.30, le=(1.03, -0.05), α=0.35)
    return [slat, main, flap]
end

"""Shoelace area of a point loop, for checking the topology's."""
function pts_area(pts)
    a = 0.0
    for i in eachindex(pts)
        p, q = pts[i], pts[mod1(i + 1, length(pts))]
        a += p[1] * q[2] - p[2] * q[1]
    end
    return a / 2
end

# -------------------------------------------------------------------------------------------------
# Invalid meshes: each breaks one check, with a fragment of the message it must raise.

function _raw_mesh(pts, cons)
    nodes = [Point(PV(p)) for p in pts]
    elems = [Line(Int32(c[1]), Int32(c[2])) for c in cons]
    return Mesh(nodes, elems)
end

const _SQ = [PV(0, 0), PV(1, 0), PV(1, 1), PV(0, 1)]

function invalid_poly_meshes()
    return [
        ("open", _raw_mesh(_SQ, [(1, 2), (2, 3), (3, 4)]), "open"),
        ("branching", _raw_mesh(vcat(_SQ, [PV(2, 2)]), [(1, 2), (2, 3), (3, 4), (4, 1), (3, 5)]),
         "branches"),
        ("reversed element", _raw_mesh(_SQ, [(1, 2), (3, 2), (3, 4), (4, 1)]), "wrong way"),
        ("collapsed element", _raw_mesh(_SQ, [(1, 2), (2, 2), (2, 3), (3, 4), (4, 1)]), "collapsed"),
        ("clockwise", poly_mesh(reverse(_SQ)), "clockwise"),
        ("zero length", poly_mesh([PV(0, 0), PV(1, 0), PV(1, 0), PV(1, 1), PV(0, 1)]), "zero length"),
        ("unmerged duplicate node", _raw_mesh(vcat(_SQ, [PV(0, 0)]), [(1, 2), (2, 3), (3, 4), (4, 5)]),
         "duplicate node"),
        ("bow tie", poly_mesh([PV(0, 0), PV(1, 0), PV(0, 1), PV(1, 1)]), "encloses no area"),
        ("self-intersecting", poly_mesh([PV(0, 0), PV(2, 0), PV(2, 2), PV(1, -1), PV(0, 2)]),
         "intersects itself"),
        ("self-touching", poly_mesh([PV(0, 0), PV(2, 0), PV(2, 2), PV(1, 0), PV(0, 2)]),
         "intersects itself"),
        ("fold-back", poly_mesh([PV(0, 0), PV(2, 0), PV(1, 0), PV(1, 1), PV(0, 1)]), "fold back"),
        ("touching loops", poly_mesh([_SQ, [p + PV(1, 1) for p in _SQ]]), "touch or overlap"),
        ("overlapping loops", poly_mesh([_SQ, [p + PV(0.5, 0.5) for p in _SQ]]), "touch or overlap"),
        ("nested loops", poly_mesh([rect_pts((0, 0), (3, 3)), rect_pts((1, 1), (2, 2))]), "inside"),
    ]
end
