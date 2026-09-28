# =====================================
# Cut-cell moments in 3D, from the nodal marching-cubes reconstruction
#
# The 3D methods of `cut_cell_moments`, producing the same `CutCellData` the 2D construction next
# door does -- `{3,T,6}` instead of `{2,T,4}`, read through the same `volume_rule` / `face_rule` /
# `interface_rule` and satisfying the same exactly-zero `closure_residual`. Read `cut_cell.jl`'s
# header first for the sign convention and the opposite senses of the face and interface normals,
# and `marching_squares/moments.jl`'s for why the construction is nodal; both are unchanged here.
# What follows is only what is different in 3D.
#
# ---------------------------------------------------------------------------
# The two halves, and why only one of them needs marching cubes
#
# The boundary of a cell's outside region is six open Cartesian face polygons plus the interface
# facets that close it. The six faces come essentially free: each is a 2D square carrying four of
# the cell's eight corner values, so `cell_boundary_walk` -- the *same* function the 2D construction
# walks a whole cell with -- gives its open fraction and open centroid on corner values the
# neighbour reads too. The interface's vector area then follows from the closure identity
#
#     A_int n_int  =  sum_k A_k n_k
#
# with no interface polygon at all, because the divergence theorem applied to a constant field over
# a closed region does not care how the closing surface is shaped. That identity is exact for *any*
# closing surface with the right boundary, which is why this half is trustworthy before any case
# table is consulted.
#
# What genuinely needs marching cubes is the outside **volume**, the outside **centroid** and the
# **interface centroid**: every divergence-theorem route to a volume integrates a moment of the
# interface itself, and that needs to know which crossings the facets actually connect.
#
# ---------------------------------------------------------------------------
# The face pairing, which is the subtle part
#
# A cube face whose four corner values **alternate in sign** is ambiguous: its four edge crossings
# can be joined into chords two different ways, giving genuinely different open regions. The 2D
# layer resolves this by naive gap-bridging and says so.
#
# **That guess cannot be used here.** The interface facets are placed by the Lewiner tables and
# their boundary edges lie on the cube's faces, so if a face clip paired its crossings one way while
# the facets paired them the other, the polyhedron this file integrates over would not close and the
# volume would be wrong by a whole corner triangle -- not by a rounding.
#
# So the pairing comes from `face_outside_connected`: the asymptotic decider, asked about that face,
# which is the *same* decision the Lewiner tables make when they resolve it. It reads only that
# face's four corner values, so the two cells sharing a face reach it independently and agree.
#
# Reading the pairing **off the facets** instead -- a triangle edge with both endpoints on one face
# being a chord of it -- is true but not sufficient: in Lewiner case 10.1.2 four of the eight facets
# lie *entirely within* a cube face plane, so all three of their edges register as chords and
# overwrite each other. It happens in about 0.015% of randomly-valued cut cubes, and where it does
# the volume comes out wrong by a whole corner triangle. `test/test_mc_moments.jl` drives random
# corner values precisely to keep that class of case in range, because no smooth body produces one
# at a sane resolution.
#
# ---------------------------------------------------------------------------
# Everything is accumulated in cell-local coordinates, and that is load-bearing here
#
# The 2D construction subtracts the cell origin before its shoelace for precision; in 3D the same
# shift buys **exactness**. A face clip works in the face's own 2D frame and the interface triangles
# work in 3D, and for the two to close, a crossing has to be the same number in both. Computing the
# 2D frame as `(node[p] - origin[p], node[q] - origin[q])` and the 3D crossings on `nodes .- origin`
# makes `_open_segment` and `_edge_crossing` apply the identical affine expression componentwise, so
# the face chord and the triangle edge that share a crossing agree bit for bit. Subtracting the
# origin *after* interpolating would leave them an ulp apart -- `(a-o) + s*((b-o)-(a-o))` is not
# `(a + s*(b-a)) - o` -- and the polyhedron would be very slightly open along every cut edge.
#
# Centroids have `origin` added back at the end; the interface's vector area is a difference of
# areas and never needs it.
#
# ---------------------------------------------------------------------------
# Two things that are weaker in 3D than in 2D, stated plainly
#
# - **`interface_area` measures a flat facet, so in 3D it is a lower bound on the facet area.** In
#   2D the facet is a straight segment, so `norm(sum_k A_k n_k)` *is* its length. In 3D the facets
#   in one cell are a triangle fan that is generally **not planar**, and the vector sum of a
#   non-planar patch is shorter than its area. The gap is nonzero in essentially every cut cell but
#   **small**: measured over a sphere and over a saddle-heavy trigonometric field at `h ~ 1/24`, the
#   lumped area runs 0.001%-0.02% below the summed facet area domain-wide, and at most about 1% in
#   any single cell (`dev/dev_mc_moments.jl` reports both).
#
#   So it is a correctness distinction rather than an accuracy problem. Use this one wherever the
#   area is differenced against the apertures -- the closure identity and every flux balance --
#   because there it is exact rather than merely accurate. Use the triangles (`cell_interface`)
#   where the area itself is the answer, such as integrating a surface stress.
# - **`ambiguous` has two causes in 3D.** A saddle *face* (pairing resolved by the tables and shared
#   with the neighbour, so conservation is untouched) and an ambiguous cube *interior* (topology
#   resolved by `_test_interior`, cell-local). One `Bool` cannot tell them apart and deliberately
#   does not try: both mean the same thing to a consumer, which is that the reconstructed normal in
#   this cell is a resolution rather than a measurement.

# Where the chord leaving face-edge slot `k` lands: the next slot, cyclically, whose open part
# *begins* mid-edge. Scanned forward when the outside is connected through the face and backward
# when it is not.
#
# The two scans differ only on a saddle. Anywhere else exactly one slot has an open part beginning
# on it, so both directions find the same one and `forward` is irrelevant -- which is why the
# decider is consulted only where it means something.
@inline function _chord_target(fphi::SVector{4,T}, k::Int, forward::Bool) where {T}
    j = k
    for _ in 1:4
        j = forward ? _ms_next(j) : _ms_prev(j)
        @inbounds (fphi[j] < zero(T) && fphi[_ms_next(j)] >= zero(T)) && return j
    end
    return 0
end

# The open polygon of one cube face, in that face's own 2D frame.
#
# `fnodes` / `fphi` are the face's four corners and values in `MS_NODE_BITS` order -- choice 3 of the
# watertightness argument, see `marching_cubes.jl`. `connect_outside` resolves a saddle and is
# ignored on every other face; it comes from [`face_outside_connected`](@ref).
#
# Returns `(area2, cmom)`: twice the open area and the shoelace's first moment. The shoelace is
# accumulated as a sum over **directed edges**, which is order-free -- so the chords need no sorting
# into a walk, only a consistent orientation. An open part of a face edge runs in traversal order; a
# chord runs from where one open part ended (`seg_b`) to where the next begins (`seg_a`), which is
# what continues the counter-clockwise traversal of the outside polygon.
@inline function _face_polygon(fnodes::SVector{4,SVector{2,T}}, fphi::SVector{4,T},
                               connect_outside::Bool) where {T}
    _, present, seg_a, seg_b = cell_boundary_walk(fnodes, fphi)
    area2 = zero(T)
    cmom = zero(SVector{2,T})
    for k in 1:4
        if @inbounds present[k]
            la = @inbounds seg_a[k]
            lb = @inbounds seg_b[k]
            cr = la[1] * lb[2] - la[2] * lb[1]
            area2 += cr
            cmom += (la + lb) * cr
        end
    end
    for k in 1:4
        # A slot emits a chord only where its open part *ends* mid-edge (outside -> inside in
        # traversal order); the matching slot is where the next open part begins. That visits each
        # chord exactly once, with the orientation the shoelace needs.
        @inbounds (fphi[k] >= zero(T) && fphi[_ms_next(k)] < zero(T)) || continue
        j = _chord_target(fphi, k, connect_outside)
        j == 0 && continue
        la = @inbounds seg_b[k]
        lb = @inbounds seg_a[j]
        cr = la[1] * lb[2] - la[2] * lb[1]
        area2 += cr
        cmom += (la + lb) * cr
    end
    return area2, cmom
end

"""
One face's clip, independent of the other five (see `cut_cell_faces`'s docstring) -- pulled out to a
plain function, rather than a loop body closing over a per-iteration `MVector` slot, for a reason
that is not stylistic. `MVector{6,SVector{3,T}}` is 72 bytes, mutated once per `dir` and converted to
an `SVector` only at the end; on CPU LLVM reliably stack-promotes that pattern, but under GPUCompiler
it does not always, and a heap-escaping "stack" array pulls from the device's tiny dynamic-allocation
pool. Verified 2026-09-21 on the `square_stern_morabito` production case (a real hull, `256x64x64`):
`@device_code_llvm` on `nodal_moments_kernel!` showed exactly one allocation site, sized exactly 72
bytes, inside a compiled specialization of `StaticArrays._zeros` -- `zero(MVector{6,SVector{3,T}})`
above, to the byte. It never showed up on a smaller sphere test (`32^3`, verified bitwise GPU/CPU
agreement in `test_mc_moments.jl`) because escaping-or-not is a compile-time question independent of
which cell is being processed; what changed was thread *count*, not geometry -- enough concurrent
threads hitting the same allocation exhausts the device's small malloc pool
("Out of dynamic GPU memory (trying to allocate 72 bytes)"), a fully-parallel kernel launch a small
test never puts enough concurrent pressure on. Returning a plain tuple from a pure per-`dir` function
and assembling the three `SVector` results with `ntuple` (the same idiom `fnodes`/`fphi` below already
use) needs no mutable array at all, so there is nothing for the compiler to get wrong either way.
"""
@inline function _cut_cell_face(dir::Int, local_nodes::SVector{8,SVector{3,T}}, phi::SVector{8,T},
                                cellsize::SVector{3,T}) where {T}
    p, q = MC_FACE_AXES[dir]
    idx = MC_FACE_NODES[dir]
    # The face's own 2D frame, taken off the *cell-local* corners so that a crossing here is
    # bitwise the one `edge_crossings` puts on the same cube edge -- see this file's header.
    # The in-plane origin is shared with the neighbour across this face (it differs from this
    # cell's only along `ax`), so these four numbers are the neighbour's four numbers.
    fnodes = SVector{4,SVector{2,T}}(ntuple(Val(4)) do k
        n = @inbounds local_nodes[idx[k]]
        SVector{2,T}(n[p], n[q])
    end)
    fphi = SVector{4,T}(ntuple(k -> @inbounds(phi[idx[k]]), Val(4)))

    nout = 0
    for k in 1:4
        nout += ifelse(@inbounds(fphi[k]) >= zero(T), 1, 0)
    end
    # Alternating corners around the face: the saddle, and the only pairing that is a choice.
    ambiguous_dir = @inbounds (nout == 2) && ((fphi[1] >= zero(T)) == (fphi[3] >= zero(T)))

    if nout == 4 || nout == 0
        frac = nout == 4 ? one(T) : zero(T)
        # The full face's centre from its own four corners, not from `centre + h/2`: the corner
        # average is one expression on four shared lattice nodes and so is bitwise the
        # neighbour's, where the arithmetic form is not. It carries real weight on a fully open
        # face, so it cannot be approximate.
        c2 = T(0.25) * (fnodes[1] + fnodes[2] + fnodes[3] + fnodes[4])
    else
        # The decider is only consulted where it means something -- a saddle. `_face_polygon`
        # ignores it otherwise, but asking unconditionally keeps the branch out of the loop.
        area2, cmom = _face_polygon(fnodes, fphi, face_outside_connected(phi, dir))
        frac = T(0.5) * area2 / full_face_area(cellsize, dir)
        c2 = area2 != zero(T) ? cmom / (T(3) * area2) :
             T(0.25) * (fnodes[1] + fnodes[2] + fnodes[3] + fnodes[4])
    end

    # `c2` is already the stored form: the centroid's in-plane coordinates `(p, q)`, measured from
    # the cell's lower corner, since `fnodes` are the cell-local corners. `face_centroid` adds the
    # corner back and takes the normal coordinate off the lattice -- the 3D instance of choice 2,
    # see `marching_cubes.jl`'s corner-convention notes.
    #
    # Returned as it is, never captured by a closure: `c2` is assigned by the `if`/`else` above,
    # and a closure capturing a conditionally-assigned variable is boxed (`Core.Box`), which is a
    # GPU-compilation failure (`InvalidIRError`). Building the global point here needed a `let` to
    # dodge exactly that.
    return frac, c2, ambiguous_dir
end

"""
    cut_cell_faces(nodes, phi, cellsize) -> (kind, ambiguous, face_fraction, face_centroid_local)

The half of a 3D cell's moments that the six face clips determine on their own:

- `kind` -- [`CELL_INSIDE`](@ref), `CELL_OUTSIDE` or `CELL_CUT`, from the eight corner signs.
- `ambiguous` -- some face's corner values alternate in sign, so a pairing had to be resolved.
- `face_fraction`, `face_centroid_local` -- per direction, the open fraction of that Cartesian
  face and the centroid of the open part in that face's in-plane coordinates, as
  [`CutCellData`](@ref) stores them, in `CartesianMeshes`' direction numbering.

The interface's vector area `A_int n_int` follows from `face_fraction` by the closure identity, so
it is not returned; [`interface_normal_area`](@ref) forms it on read.

`nodes` and `phi` are the eight corners in [`MC_NODE_BITS`](@ref) order, and `phi` is expected to
have been through [`nudge_zeros`](@ref) already.

**Complete and exact on its own**, which is why it is a function rather than a stage: it needs no
interface polygon, no case table and no volume -- an ambiguous face is resolved by
[`face_outside_connected`](@ref), which reads that face's own four corner values -- and every
quantity in it is final. The three moments it does not produce, the outside volume, the outside
centroid and the interface centroid, are what [`cut_cell_moments`](@ref) adds on top. Returning them
separately rather than as a half-filled [`CutCellData`](@ref) keeps that struct's promises
honest: it never exists with a sentinel in it.

Each of the six faces is independent (see [`_cut_cell_face`](@ref)) -- `ambiguous` is the only value
that reduces across them, and `|` is commutative, so computing all six in whatever order costs
nothing bitwise.
"""
function cut_cell_faces(nodes::SVector{8,SVector{3,T}}, phi::SVector{8,T},
                        cellsize::SVector{3,T}) where {T}
    origin = @inbounds nodes[1]
    local_nodes = SVector{8,SVector{3,T}}(ntuple(k -> @inbounds(nodes[k]) - origin, Val(8)))

    noutside = 0
    for k in 1:8
        noutside += ifelse(@inbounds(phi[k]) >= zero(T), 1, 0)
    end
    kind = noutside == 8 ? CELL_OUTSIDE : noutside == 0 ? CELL_INSIDE : CELL_CUT

    # **No early return for an uncut cell**, unlike the 2D construction, and the difference is not
    # cosmetic. Every face of an uncut cell is wholly open or wholly closed, which the per-face
    # branch below already reports as an exact `1` or `0` -- so the reason 2D short-circuits (a
    # shoelace would give `0.9999...` where the answer is `1`) is already handled one level down,
    # and routing uncut cells through the same branch costs only the corner averages.
    #
    # What it buys is the thing a short-circuit quietly breaks. A *fully open* face carries full
    # quadrature weight, not zero, so its centroid is a real number in a real integral rather than a
    # placeholder -- and an outside cell and its cut neighbour must report the same one, bitwise, or
    # the flux through that shared face is evaluated at two different points. `centre + h/2` is not
    # bitwise the neighbour's `centre - h/2`; the average of the face's own four lattice corners is.
    # The 2D construction takes its full-face centres off the lattice corners for the same reason.
    per_dir = ntuple(dir -> _cut_cell_face(dir, local_nodes, phi, cellsize), Val(6))
    face_fraction = SVector{6,T}(ntuple(dir -> (@inbounds per_dir[dir][1]), Val(6)))
    face_centroid_local = SVector{6,SVector{2,T}}(ntuple(dir -> (@inbounds per_dir[dir][2]),
                                                         Val(6)))
    # A plain loop, not `any(f, 1:6)`: `Base.any` over a generic iterator is not the same compiled
    # shape as the `for` loops elsewhere in this function and was not GPU-compilable here
    # ("InvalidIRError", unrelated to the `MVector` fix above -- caught by the GPU regression test
    # in `test/test_mc_moments.jl`, which is exactly what it is for).
    ambiguous = false
    for dir in 1:6
        ambiguous |= @inbounds per_dir[dir][3]
    end

    return kind, ambiguous, face_fraction, face_centroid_local
end

"""
    cut_cell_moments(nodes, phi, cellsize) -> CutCellData{3,T,6,2}

The primitive in 3D: every moment of one cut cell, from the eight corner coordinates `nodes`, the
eight corner values `phi`, and the cell size.

`nodes` and `phi` are in [`MC_NODE_BITS`](@ref) order, which is what [`cell_nodes`](@ref) and
[`cell_values`](@ref) return. As in 2D, taking *node values* rather than a geometry object is the
watertightness contract made explicit: the construction cannot depend on anything a neighbour does
not also see.

The face apertures come from [`cut_cell_faces`](@ref), and the interface's vector area follows
from them on read; the outside volume, the outside centroid and the interface centroid come from
integrating the closed polyhedron by the divergence theorem, with the interface facets supplied by
[`cell_triangles`](@ref).
"""
function cut_cell_moments(nodes::SVector{8,SVector{3,T}}, phi_raw::SVector{8,T},
                          cellsize::SVector{3,T}) where {T}
    # Once, here, and then used for the case index, the six face clips and every crossing alike --
    # see `nudge_zeros`. Applying it to some of those and not others would tear the reconstruction.
    phi = nudge_zeros(phi_raw)
    origin = @inbounds nodes[1]
    centre = origin + T(0.5) * cellsize
    vol_cell = prod(cellsize)

    n_tri, codes = cell_triangles(phi)
    kind, amb_face, face_fraction, face_centroid_local = cut_cell_faces(nodes, phi, cellsize)

    if kind != CELL_CUT
        return CutCellData{3,T,6,2}(kind, false, kind == CELL_OUTSIDE ? one(T) : zero(T), centre,
                                    face_fraction, face_centroid_local, centre)
    end

    local_nodes = SVector{8,SVector{3,T}}(ntuple(k -> @inbounds(nodes[k]) - origin, Val(8)))
    _, pts = edge_crossings(local_nodes, phi)

    # --- the closed polyhedron, by the divergence theorem ------------------
    #
    #   V       = (1/3) * oint x . n dA
    #   c_a     = (1/2V) * oint x_a^2 n_a dA
    #
    # over the boundary of the outside region: the six open Cartesian faces, where `x . n` and
    # `x_a^2` are constant and the integral is one multiply, plus the interface facets. The facets'
    # outward normal *for the outside region* is minus the winding normal `cell_triangles` returns,
    # which points out of the body -- hence the sign on `w` below, and it is the same opposition of
    # senses the closure identity encodes.
    vol3 = zero(T)
    cm2 = zero(SVector{3,T})
    for dir in 1:6
        ax = direction_axis(dir)
        s = T(direction_sign(dir))
        x_ax = @inbounds local_nodes[MC_FACE_NODES[dir][1]][ax]
        a_open = @inbounds face_fraction[dir] * full_face_area(cellsize, dir)
        vol3 += s * x_ax * a_open
        cm2 += SVector{3,T}(ntuple(a -> a == ax ? s * x_ax * x_ax * a_open : zero(T), Val(3)))
    end

    iface_area = zero(T)
    iface_cmom = zero(SVector{3,T})
    for t in 1:n_tri
        c = @inbounds codes[t]
        a = @inbounds pts[c[1]]
        b = @inbounds pts[c[2]]
        d = @inbounds pts[c[3]]
        # Twice the vector area, halved once: `w` is `A_tri * n`, with `n` out of the body.
        w = T(0.5) * cross(b - a, d - a)
        xc = (a + b + d) / T(3)
        vol3 -= dot(w, xc)
        # The mean of `x_a^2` over a triangle, exactly: (sum of squares + sum of products) / 6.
        m2 = (a .* a + b .* b + d .* d + a .* b + b .* d + d .* a) / T(6)
        cm2 -= w .* m2
        len = sqrt(sum(abs2, w))
        iface_area += len
        iface_cmom += len * xc
    end

    volume = vol3 / T(3)
    centroid = volume > zero(T) ? origin + cm2 / (T(2) * volume) : centre
    interface_centroid = iface_area > zero(T) ? origin + iface_cmom / iface_area : centre

    # `ambiguous` covers both causes: a saddle face, and a cube interior the Lewiner dispatch had to
    # resolve with `_test_interior`. See this file's header for why one flag serves both.
    lut_case = Int(@inbounds MarchingCubes.cases[cell_case(phi)][1])
    ambiguous = amb_face || lut_case == 5 || lut_case == 7 || lut_case == 8 ||
                lut_case == 11 || lut_case == 13 || lut_case == 14

    return CutCellData{3,T,6,2}(CELL_CUT, ambiguous, volume / vol_cell, centroid,
                                face_fraction, face_centroid_local, interface_centroid)
end

"""
    cut_cell_moments(geo, lo::SVector{3}, hi::SVector{3}) -> CutCellData{3,T,6,2}

Sample `geo` at the eight corners of the box `[lo, hi]` and build its moments.

The 3D method of the standalone entry point, carrying the 2D one's caveat: corners are derived by
arithmetic on `lo` and `hi`, which is correct but gives up the bitwise agreement between adjoining
cells. Prefer [`cut_cell_moments`](@ref)`(method, domain, geo, cell)`, which builds corners off the
integer lattice.
"""
function cut_cell_moments(geo, lo::SVector{3,T}, hi::SVector{3,T}) where {T}
    cellsize = hi - lo
    nodes = SVector{8,SVector{3,T}}(ntuple(Val(8)) do k
        b = MC_NODE_BITS[k]
        lo + SVector{3,T}(T(b[1]) * cellsize[1], T(b[2]) * cellsize[2], T(b[3]) * cellsize[3])
    end)
    phi = SVector{8,T}(ntuple(k -> T(sdf_value(geo, @inbounds nodes[k])), Val(8)))
    return cut_cell_moments(nodes, phi, cellsize)
end

"""
    cut_cell_moments(method::MarchingCubesCutCell, grid::CartesianGrid{3}, geo, ci)
    cut_cell_moments(method::MarchingCubesCutCell, grid::CartesianGrid{3}, vals, ci)
    cut_cell_moments(method::MarchingCubesCutCell, mesh::AdaptiveMesh{3}, geo, cell)

The moments of the single cell `ci` (or leaf `cell`) of a 3D domain -- the
[`MarchingCubesCutCell`](@ref) method of the per-cell entry point every
[`AbstractCutCellMethod`](@ref) answers to, and the exact 3D counterpart of
[`cut_cell_moments`](@ref)`(::MarchingSquaresCutCell, ...)`. Returns a
[`CutCellData`](@ref)`{3,T,6}`, with `T` taken from the domain.

`geo` is anything SDFLibrary.jl's `sdf_value` accepts, or, on a grid, a 3D array of nodal values already
sampled on it.

Corners come from [`cell_nodes`](@ref) and so from the integer lattice, which makes a corner shared
by two cells -- at the same level or across a level jump -- the same floating-point number for both,
and their apertures bitwise equal.

**Per-cell matters more in 3D than in 2D.** A whole-grid `CutCellData{3,T,6,2}` field is 208 bytes
a cell -- about 1.7 GB over `200^3` -- while the moments are only non-trivial in the `O(h^2)` band
of cut cells. Recomputing one cell on demand is a pure function of its eight corner values, with no
neighbour to consult, so at any real resolution this is the form to build on and the dense array is
for a convergence study.

Note this is the moment layer, not the surface: `generate_mesh(geo, grid, method)`
runs `MarchingCubes.jl`'s own march and shares nothing with this construction but the corner values.
"""
@inline function cut_cell_moments(::MarchingCubesCutCell, grid::CartesianGrid{3,T}, geo,
                                  ci::CartesianIndex{3}) where {T}
    return cut_cell_moments(cell_nodes(grid, ci), cell_values(geo, grid, ci), grid.d)
end

@inline function cut_cell_moments(::MarchingCubesCutCell, mesh::AdaptiveMesh{3}, geo,
                                  c::TreeCell{3})
    return cut_cell_moments(cell_nodes(mesh, c), cell_values(geo, mesh, c), get_elem_size(mesh, c))
end

@inline cut_cell_moments(m::MarchingCubesCutCell, mesh::AdaptiveMesh{3}, geo, i::Integer) =
    cut_cell_moments(m, mesh, geo, leaf(mesh, i))

# The 3D half of the dimension-to-method map `_moments_over_grid` dispatches through; see
# `marching_squares/moments.jl`, which carries the 2D one and the whole-grid entry point itself.
@inline _nodal_method(::CartesianGrid{3}) = MarchingCubesCutCell()

"""
    cut_cell_moments(geo, mesh::AdaptiveMesh{3})

Moments for every leaf of a 3D `mesh` against `geo`, as a `Vector` indexed by leaf.

The 3D counterpart of [`cut_cell_moments(geo, mesh::AdaptiveMesh{2})`](@ref), and the one place the
two genuinely differ: that one launches a `KernelAbstractions` kernel and so runs on whichever
backend the mesh lives on, while this one is a **threaded host loop**.

**Not because the Lewiner dispatch is device-incompatible** -- verified 2026-09-21 by compiling and
launching `cell_triangles` inside a bare `KernelAbstractions` kernel on `CUDABackend` and diffing
against this same host path over all 256 corner sign configurations: bitwise identical.
`MarchingCubes.jl`'s tables (`cases`, `tiling1`..`tiling14`, `test3`..`test13`) are `const` literal
nested `NTuple`s -- already `isbits`, already fixed-size -- and constant-fold into a kernel like any
other `const` global; nothing about "reading them as nested tuples" makes them host-only. The host
loop here is simply because nobody has written a kernel walking `AdaptiveMesh{3}`'s leaves yet, an
orthogonal problem (tree traversal on device) from the Lewiner dispatch itself, which was always
fixed-size and allocation-free per cell. The march behind [`generate_mesh`](@ref) is a
separate case: *that* has no device path because it accumulates a whole-domain triangle list of a
priori unknown size, not because of the table lookups.

That said, `cell_triangles` being GPU-clean did not mean the whole `cut_cell_moments` construction
was: [`cut_cell_faces`](@ref) had a real bug of its own (fixed 2026-09-21, see its source) -- a
closure capturing a branch-reassigned variable, boxed by the compiler, which is a genuine
GPU-compilation failure and not merely a style nit. Confirmed fixed end to end against
`HYBIS.jl`'s `CartesianGrid{3}` kernel path (a sphere, `32^3`), GPU vs CPU(Float32) agreeing to 18
ULPs. This host loop was never blocked by it (a host loop tolerates a boxed closure fine), but a
`CartesianGrid{3}`-based device kernel over `MarchingCubesCutCell` would have been, until now.

Far cheaper than the `CartesianGrid` form by the ratio of leaves to cells, which on a mesh refined
around a surface is most of the reason to be on a tree at all.
"""
function cut_cell_moments(geo, mesh::AdaptiveMesh{3,T}) where {T}
    n = nleaves(mesh)
    out = Vector{CutCellData{3,T,6,2}}(undef, n)
    method = MarchingCubesCutCell()
    if n > THREAD_FLOOR
        @batch for i in 1:n
            @inbounds out[i] = cut_cell_moments(method, mesh, geo, leaf(mesh, i))
        end
    else
        for i in 1:n
            @inbounds out[i] = cut_cell_moments(method, mesh, geo, leaf(mesh, i))
        end
    end
    return out
end
