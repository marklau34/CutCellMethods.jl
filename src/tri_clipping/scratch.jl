# =====================================
# Per-slot scratch: the working storage of the clippers, in preallocated arrays
#
# A clip rewrites a polytope's vertex and face arrays several times per cell, so it needs mutable
# storage. It does not get a kernel-local `MVector`: under GPUCompiler such a buffer is not reliably
# stack-promoted, and one that escapes pulls from the device's small malloc pool -- the 72-byte
# failure `marching_cubes/moments.jl` documents. Instead every workitem owns one *slot* of the
# arrays below, allocated once on the cache's backend, so nothing is allocated inside a kernel on
# any backend. A kernel launches one workitem per slot and walks its share of the work list with a
# stride of `nslots` (`_slot_items`).
#
# The polytope and the planar polygon are each double-buffered: a clip reads one buffer and writes
# the other, which is also what compacts the polytope's vertices after every clip.

const POLY_MAXV = 32   # polytope vertices
const POLY_MAXF = 16   # polytope faces
const POLY_MAXL = 16   # vertices in one face loop
const POLY_MAXE = 32   # polytope edges one plane may cross
const POLY_MAXC = 32   # on-plane edges collected for one cap
const PG_MAXP   = 16   # planar polygon vertices: triangle ∩ box <= 9, face square ∩ 8 planes <= 12
const MAX_CROSS = 64   # ray crossings along one grid row

"""
    TriScratch

The clippers' working storage: slot `s` of every array belongs to one workitem. See this file's
header for why it is not kernel-local.
"""
struct TriScratch{AX,AL,AN,AG,AH,AS,AM,AE,AO,AC,AP,AQ,AR,GA,GV,GP,GI,GB,EI,LS,LF}
    x::AX       # SVector{3,T} (POLY_MAXV, 2, nslots): polytope vertices
    loop::AL    # Int8 (POLY_MAXL, POLY_MAXF, 2, nslots): face loops, counter-clockwise seen from outside
    nloop::AN   # Int8 (POLY_MAXF, 2, nslots): loop lengths
    tag::AG     # Int16 (POLY_MAXF, 2, nslots): 1-6 a Cartesian face by direction, above 6 a patch
    hdr::AH     # Int16 (3, nslots): current buffer, vertex count, face count
    sigma::AS   # T (POLY_MAXV, nslots): signed distances to the plane being clipped
    vmap::AM    # Int8 (POLY_MAXV, nslots): old vertex -> new
    enew::AE    # Int8 (3, POLY_MAXE, nslots): a crossing edge (lo, hi) -> its new vertex
    onp::AO     # Int8 (POLY_MAXV, nslots): new vertex lies on the plane
    cap::AC     # Int8 (3, POLY_MAXC, nslots): on-plane edges (u, v) and a used mark
    pg::AP      # SVector{3,T} (PG_MAXP, 2, nslots): planar polygon
    pgsig::AQ   # T (PG_MAXP, nslots)
    cross::AR   # T (MAX_CROSS, nslots): ray crossings of one row
    # One cell's fit groups (`planefit.jl`), at most `TRI_K_MAX`:
    garea::GA   # T (TRI_K_MAX, nslots): area
    gnrm::GV    # SVector{3,T} (TRI_K_MAX, nslots): sum of A n
    gmom::GV    # SVector{3,T} (TRI_K_MAX, nslots): sum of A c, cell-local
    gplane::GP  # SVector{4,T} (TRI_K_MAX, nslots): the group's plane (n..., d), cell-local
    gmem::GI    # UInt8 (TRI_K_MAX, nslots): member patch entries, as bits
    gpar::GB    # Int8 (TRI_K_MAX, nslots): union-find parent of each patch entry
    # Each group's share of the clipped interface (`passes.jl`): its vector area out of the body,
    # its scalar area, and that area's first moment.
    gS::GV      # SVector{3,T} (TRI_K_MAX, nslots)
    gAs::GA     # T (TRI_K_MAX, nslots)
    gMs::GV     # SVector{3,T} (TRI_K_MAX, nslots)
    # The patch-entry table the groups are built from: one cell's (from Pass A) or a face's (the
    # union of its two cells'), at most `TRI_K_MAX` patches.
    eid::EI     # Int32 (TRI_K_MAX, nslots)
    eA::GA      # T (TRI_K_MAX, nslots)
    eN::GV      # SVector{3,T} (TRI_K_MAX, nslots)
    eM::GV      # SVector{3,T} (TRI_K_MAX, nslots)
    # The mixed rule's arrangement (`mixed.jl`): each piece's status, and per piece and plane
    # whether its face on that plane exists and lies on the plane's patch.
    lstat::LS   # Int8 (TRI_NLEAF, nslots)
    lface::LF   # UInt8 (TRI_K_MIXED, TRI_NLEAF, nslots)
end

Adapt.@adapt_structure TriScratch

function TriScratch(backend, ::Type{T}, nslots::Integer) where {T}
    a(S, dims...) = KernelAbstractions.allocate(backend, S, dims...)
    return TriScratch(a(SVector{3,T}, POLY_MAXV, 2, nslots),
                      a(Int8, POLY_MAXL, POLY_MAXF, 2, nslots),
                      a(Int8, POLY_MAXF, 2, nslots),
                      a(Int16, POLY_MAXF, 2, nslots),
                      a(Int16, 3, nslots),
                      a(T, POLY_MAXV, nslots),
                      a(Int8, POLY_MAXV, nslots),
                      a(Int8, 3, POLY_MAXE, nslots),
                      a(Int8, POLY_MAXV, nslots),
                      a(Int8, 3, POLY_MAXC, nslots),
                      a(SVector{3,T}, PG_MAXP, 2, nslots),
                      a(T, PG_MAXP, nslots),
                      a(T, MAX_CROSS, nslots),
                      a(T, TRI_K_MAX, nslots),
                      a(SVector{3,T}, TRI_K_MAX, nslots),
                      a(SVector{3,T}, TRI_K_MAX, nslots),
                      a(SVector{4,T}, TRI_K_MAX, nslots),
                      a(UInt8, TRI_K_MAX, nslots),
                      a(Int8, TRI_K_MAX, nslots),
                      a(SVector{3,T}, TRI_K_MAX, nslots),
                      a(T, TRI_K_MAX, nslots),
                      a(SVector{3,T}, TRI_K_MAX, nslots),
                      a(Int32, TRI_K_MAX, nslots),
                      a(T, TRI_K_MAX, nslots),
                      a(SVector{3,T}, TRI_K_MAX, nslots),
                      a(SVector{3,T}, TRI_K_MAX, nslots),
                      a(Int8, TRI_NLEAF, nslots),
                      a(UInt8, TRI_K_MIXED, TRI_NLEAF, nslots))
end

nslots(scr::TriScratch) = size(scr.hdr, 2)

# Enough slots to fill the device: a few workgroups per CPU thread, a few waves of a GPU.
_default_nslots(::KernelAbstractions.CPU) = 64 * Threads.nthreads()
_default_nslots(_) = 16384

# The work items slot `s` of `ns` walks: `s, s + ns, s + 2ns, ...` up to `n`.
@inline _slot_items(s::Integer, ns::Integer, n::Integer) = s:ns:n
