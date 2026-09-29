# =====================================
# The polyline-clipping cache: every cell's exact cut geometry over a whole grid, kept between updates
#
# An update runs, in order:
#
#  0. the topology -- loops and validation -- only when the mesh's connectivity changed (host,
#     cached); winding every time, and intersections and nesting with the topology or, under
#     `validate = :always`, every time;
#  1. the block, the crossings, the vertex bins, the node and cell states, and the cut cells' output
#     slots and islands (host, `crossings.jl`);
#  2. a reset of the cells and edges the body covered last time, and the classification kernels:
#     every block cell's status and every uncrossed block edge;
#  3. the walk kernel, one workitem per cut cell (`walk.jl`);
#  4. the pack (host): which cut cells split, and where their region records go;
#  5. the edge kernel: every edge beside a cut cell, once -- region matching, slivers closed, the
#     aperture both cells then copy;
#  6. the finalize kernel: every cut cell's record, its regions', and any neighbour a sliver handed
#     a wall to.
#
# Only the block -- the cells the body can reach, plus one -- is written after allocation, so the
# rest of the grid keeps the no-body seed.

"""
    PolylineClippingCutCellCache

What [`allocate_cache`](@ref)`(grid, PolylineClippingCutCell(); backend)` builds and
[`update_cache!`](@ref)`(cache, mesh, grid)` refreshes:

- `cells` -- every cell's [`CutCellData`](@ref)`{2,T,4,1}`, a `StructArray` sized `grid.n`, as in
  every other cache. In a cut cell these are the cell's totals over all its fluid regions.
- `info` -- every cell's [`PolylineCellInfo`](@ref): its status (fluid, solid, cut, split,
  invalid), region count, diagnostic flags and closure residual.
- `edges` -- `(ax, ay)`: every grid edge's [`PolylineEdge`](@ref) aperture, x-edges sized
  `(n[1]+1, n[2])` and y-edges `(n[1], n[2]+1)`. Both cells sharing an edge copy it.
- `regions`, `rinfo` -- one [`CutCellData`](@ref) and one [`PolylineRegionInfo`](@ref) per fluid
  region of each **split** cell, in cell order; a cell's are `info.rslot` on. A cell with one
  region is its own record.
- `arcs` -- every fluid interval of every cut cell's edges, with the region owning it on each side
  ([`PolylineArc`](@ref)).
- `bsegs` -- every element's piece in every cut cell ([`PolylineBoundarySeg`](@ref)).

Read one cell's through [`region`](@ref), [`regions`](@ref), [`arcs`](@ref) and
[`boundary_segments`](@ref).

`mesh` is a `Mesh{2}` of `Line` elements in the grid's frame, each loop counter-clockwise. A moving
body is the same mesh with moved nodes, or the same body under a moved grid; either way the
topology is not rebuilt while the connectivity does not change.
"""
struct PolylineClippingCutCellCache{T,C<:AbstractArray{<:CutCellData},I,E,R,Q,A,B,W} <: AbstractCutCellCache
    method::PolylineClippingCutCell
    tols::PolylineTols{T}
    grid::CartesianGrid{2,T}
    cells::C
    info::I
    edges::E
    regions::R
    rinfo::Q
    arcs::A
    bsegs::B
    work::W
end

KernelAbstractions.get_backend(cache::PolylineClippingCutCellCache) =
    get_backend(cache.cells.volume_fraction)
Base.eltype(::PolylineClippingCutCellCache{T}) where {T} = T
@inline face_fractions(cache::PolylineClippingCutCellCache) = cache.cells.face_fraction

"""
    PolylineClippingCutCellView

What adapting a [`PolylineClippingCutCellCache`](@ref) gives: its per-cell results, `cells` and
`info`, moved by the adaptor, and none of what an update runs on (the topology, the host's crossing
record, the walk's buffers). Answers the uniform readers (`face_fractions`, `volume_fractions`,
`kinds`) as the cache does.

Lets a cache pass through a kernel launch's adaptor, directly or nested in a consumer's own struct:
the kernel gets the view, which is isbits on the device where the cache is not.
`Adapt.adapt(Array, cache)` gives a host copy of the results. A view cannot be updated; to run on
another backend, allocate the cache there.
"""
struct PolylineClippingCutCellView{C,I} <: AbstractCutCellCache
    cells::C
    info::I
end

Adapt.@adapt_structure PolylineClippingCutCellView
Adapt.adapt_structure(to, cache::PolylineClippingCutCellCache) =
    PolylineClippingCutCellView(Adapt.adapt(to, cache.cells), Adapt.adapt(to, cache.info))
KernelAbstractions.get_backend(view::PolylineClippingCutCellView) =
    get_backend(view.cells.volume_fraction)
@inline face_fractions(view::PolylineClippingCutCellView) = view.cells.face_fraction

# What an update keeps between calls: the topology, the block, the host's crossing record, and
# grow-only backend copies of what the kernels read.
mutable struct PolylineWork{T,D<:NamedTuple}
    topo::Union{Nothing,PolylineTopology}
    loop_area::Vector{Float64}
    block::PolylineBlock
    x0::SVector{2,T}                 # the grid origin the cells were last written for
    X::Vector{SVector{2,Float64}}
    lines::Vector{SVector{2,Int32}}
    lattice::PolylineLattice
    cross::PolylineCrossings
    stageV::Vector{SVector{2,T}}
    stageT::Vector{T}
    stats::NamedTuple
    dev::D
end

function _pl_dev_buffers(backend, ::Type{T}) where {T}
    a(E) = KernelAbstractions.allocate(backend, E, 0)
    return (status=a(UInt8), flags=a(UInt8), slot=a(Int32), node_solid=a(Bool), edge_start=a(Int32),
            verts=a(SVector{2,T}), lines=a(SVector{2,Int32}), elem_prev=a(Int32), elem_next=a(Int32),
            elem_loop=a(Int32), loop_first=a(Int32), cr_elem=a(Int32), cr_offset=a(T), cr_dir=a(Int8),
            vbin_start=a(Int32), vbin_elem=a(Int32), cut_list=a(CartesianIndex{2}),
            bseg_start=a(Int32), region_start=a(Int32), arc_start=a(Int32), island_start=a(Int32),
            island_loop=a(Int32), island_kind=a(Int8), island_target=a(Int32),
            fin_list=a(CartesianIndex{2}), rslot=a(Int32),
            racc=_allocate_cells(backend, PolylineRegionAcc{T}, (0,)),
            sres=_allocate_cells(backend, PolylineSlotResult{T}, (0,)))
end

const _PL_NO_STATS = (ncut=0, nsplit=0, ninvalid=0, noverflow=0, nisland=0)

function allocate_cache(grid::CartesianGrid{2,T}, method::PolylineClippingCutCell;
                        backend=KernelAbstractions.CPU()) where {T}
    n = Tuple(grid.n)
    edges = (ax=_allocate_cells(backend, PolylineEdge{T}, (n[1] + 1, n[2])),
             ay=_allocate_cells(backend, PolylineEdge{T}, (n[1], n[2] + 1)))
    dev = _pl_dev_buffers(backend, T)
    work = PolylineWork{T,typeof(dev)}(
        nothing, Float64[], PolylineBlock(), grid.x0, SVector{2,Float64}[], SVector{2,Int32}[],
        PolylineLattice(grid), PolylineCrossings(), SVector{2,T}[], T[], _PL_NO_STATS, dev)
    cache = PolylineClippingCutCellCache(
        method, PolylineTols(method, grid), grid,
        _allocate_cells(backend, CutCellData{2,T,4,1}, n),
        _allocate_cells(backend, PolylineCellInfo{T}, n), edges,
        _allocate_cells(backend, CutCellData{2,T,4,1}, (0,)),
        _allocate_cells(backend, PolylineRegionInfo{T}, (0,)),
        _allocate_cells(backend, PolylineArc{T}, (0,)),
        _allocate_cells(backend, PolylineBoundarySeg{T}, (0,)), work)
    # Seeded as no body, over the whole grid.
    _pl_reset!(cache, grid, _pl_full_block(grid))
    KernelAbstractions.synchronize(backend)
    return cache
end

allocate_cache(grid::CartesianGrid, ::PolylineClippingCutCell; kwargs...) = throw(ArgumentError(
    "PolylineClippingCutCell cuts 2D grids only, got a $(length(grid.n))D grid"))

_pl_full_block(g::CartesianGrid{2}) = PolylineBlock(SVector(1, 1), SVector{2,Int}(g.n), false)

function update_cache!(cache::PolylineClippingCutCellCache{T}, mesh::Mesh,
                       grid::CartesianGrid{2}) where {T}
    _check_cache_size(cache.cells, grid)
    g = convert(CartesianGrid{2,T}, grid)
    g.d == cache.grid.d || throw(ArgumentError(
        "the grid's cell size $(Tuple(g.d)) is not the $(Tuple(cache.grid.d)) the cache was " *
        "allocated for: only a grid's origin may move between updates"))
    w = cache.work
    backend = get_backend(cache)
    # What has to be reset: the cells the body covered last time -- or all of them if the grid moved,
    # since an untouched cell's centroids are global coordinates of the old grid. Recorded only once
    # the reset has run, so an update that throws part-way leaves this right.
    prev = g.x0 == w.x0 ? w.block : _pl_full_block(g)

    if length(mesh.elements) == 0
        _pl_reset!(cache, g, prev)
        _pl_resize_outputs!(cache, 0, 0, 0)
        w.block = PolylineBlock()
        w.x0 = g.x0
        w.stats = _PL_NO_STATS
        KernelAbstractions.synchronize(backend)
        return cache
    end

    # The topology is kept only once the mesh has passed validation, so a mesh that fails is
    # checked again next time rather than trusted.
    fp = _pl_fingerprint(mesh)
    fresh = w.topo === nothing || w.topo.fingerprint != fp
    topo = fresh ? build_polyline_topology(mesh) : w.topo
    _pl_refresh_geometry!(w, mesh)
    # Winding and zero length are coordinate facts as cheap as reading the coordinates, so they are
    # checked every time; intersections and nesting when the connectivity changes, or on request.
    area = _pl_check_loops(topo, w.X, w.lines)
    if fresh || cache.method.validate === :always
        _pl_check_intersections(topo, w.X, w.lines)
        _pl_check_nesting(topo, w.X, w.lines)
    end
    w.topo = topo
    w.loop_area = area

    _pl_lattice!(w.lattice, g)
    block = _pl_block(w.X, w.lines, w.lattice)
    C = w.cross
    _pl_crossings!(C, w.X, w.lines, w.lattice, block)
    _pl_vertex_bins!(C, w.X, w.lines, w.lattice, block)
    _pl_classify!(C, block)
    _pl_cut_lists!(C, w.X, w.lines, topo, w.lattice, block)
    _pl_upload!(w, topo)
    ncut = length(C.cut_list)
    _pl_resize_outputs!(cache, C.arc_start[end] - 1, C.bseg_start[end] - 1, 0)
    resize!(w.dev.racc, C.region_start[end] - 1)
    resize!(w.dev.sres, ncut)

    _pl_reset!(cache, g, prev)
    w.x0 = g.x0
    _pl_write_block!(cache, g, block)
    blo = CartesianIndex(Tuple(block.lo))
    d = Tuple(_pl_dims(block))
    m = _pl_kernel_args(cache)
    drop2 = 2 * cache.tols.drop_area
    ncut > 0 && _pl_walk_kernel!(backend, 64)(m, g, blo, d, ncut, drop2; ndrange=ncut)
    KernelAbstractions.synchronize(backend)

    # Which cut cells split, and where their region records go.
    nregion = _pl_download!(C.nregion, w.dev.sres.nregion)
    rslot = C.rslot
    resize!(rslot, ncut)
    nrec = 0
    for s in 1:ncut
        rslot[s] = 0
        if nregion[s] >= 2
            rslot[s] = nrec + 1
            nrec += nregion[s]
        end
    end
    _upload!(w.dev.rslot, rslot)
    _pl_resize_outputs!(cache, length(cache.arcs), length(cache.bsegs), nrec)
    m = _pl_kernel_args(cache)

    if ncut > 0
        _pl_edge_kernel!(backend, 64)(cache.edges.ax, cache.info, m, g, blo, d, 1;
                                      ndrange=(d[1] + 1, d[2]))
        _pl_edge_kernel!(backend, 64)(cache.edges.ay, cache.info, m, g, blo, d, 2;
                                      ndrange=(d[1], d[2] + 1))
        nfin = length(C.fin_list)
        _pl_finalize_kernel!(backend, 64)(cache.cells, cache.info, cache.regions, cache.rinfo,
                                          cache.edges.ax, cache.edges.ay, m, g, blo, d, cache.tols,
                                          w.dev.fin_list, nfin; ndrange=nfin)
    end
    KernelAbstractions.synchronize(backend)
    w.block = block
    _pl_gather_stats!(w, ncut, nregion)
    return cache
end

"""
    reset_topology!(cache::PolylineClippingCutCellCache) -> cache

Forget the cached topology, so the next update rebuilds and revalidates the loops -- after
deforming a body in a way that could make it touch itself, say, with `validate = :topology`.
"""
function reset_topology!(cache::PolylineClippingCutCellCache)
    cache.work.topo = nothing
    return cache
end

function _pl_refresh_geometry!(w::PolylineWork, mesh::Mesh)
    nv = length(mesh.nodes)
    resize!(w.X, nv)
    for (i, c) in enumerate(mesh.nodes.coord)
        w.X[i] = SVector{2,Float64}(c)
    end
    resize!(w.lines, length(mesh.elements))
    for (e, el) in enumerate(mesh.elements)
        w.lines[e] = SVector{2,Int32}(el.con)
    end
    return nothing
end

# A backend vector back into a grow-only host one.
function _pl_download!(dst::Vector, src::AbstractVector)
    resize!(dst, length(src))
    isempty(src) || copyto!(dst, src)
    return dst
end

# Everything the kernels read, onto the backend.
function _pl_upload!(w::PolylineWork{T}, topo::PolylineTopology) where {T}
    C = w.cross
    dv = w.dev
    _upload!(dv.status, C.status)
    _upload!(dv.flags, C.flags)
    _upload!(dv.slot, C.slot)
    _upload!(dv.node_solid, C.node_solid)
    _upload!(dv.edge_start, C.edge_start)
    resize!(w.stageV, length(w.X))
    for i in eachindex(w.X)
        w.stageV[i] = SVector{2,T}(w.X[i])
    end
    _upload!(dv.verts, w.stageV)
    _upload!(dv.lines, w.lines)
    _upload!(dv.elem_prev, topo.elem_prev)
    _upload!(dv.elem_next, topo.elem_next)
    _upload!(dv.elem_loop, topo.elem_loop)
    _upload!(dv.loop_first, topo.loop_first)
    _upload!(dv.cr_elem, C.elem)
    resize!(w.stageT, length(C.offset))
    for i in eachindex(C.offset)
        w.stageT[i] = T(C.offset[i])
    end
    _upload!(dv.cr_offset, w.stageT)
    _upload!(dv.cr_dir, C.dir)
    _upload!(dv.vbin_start, C.vbin_start)
    _upload!(dv.vbin_elem, C.vbin_elem)
    _upload!(dv.cut_list, C.cut_list)
    _upload!(dv.bseg_start, C.bseg_start)
    _upload!(dv.region_start, C.region_start)
    _upload!(dv.arc_start, C.arc_start)
    _upload!(dv.island_start, C.island_start)
    _upload!(dv.island_loop, C.island_loop)
    _upload!(dv.island_kind, C.island_kind)
    _upload!(dv.island_target, C.island_target)
    _upload!(dv.fin_list, C.fin_list)
    return nothing
end

function _pl_resize_outputs!(cache::PolylineClippingCutCellCache, narc::Integer, nbseg::Integer,
                             nregion::Integer)
    resize!(cache.arcs, narc)
    resize!(cache.bsegs, nbseg)
    resize!(cache.regions, nregion)
    resize!(cache.rinfo, nregion)
    return nothing
end

# The kernels' view of the cache: the uploads, the walk's working sums, and the outputs.
function _pl_kernel_args(cache::PolylineClippingCutCellCache)
    dv = cache.work.dev
    return (verts=dv.verts, lines=dv.lines, elem_prev=dv.elem_prev, elem_next=dv.elem_next,
            elem_loop=dv.elem_loop, loop_first=dv.loop_first, cr_elem=dv.cr_elem,
            cr_offset=dv.cr_offset, cr_dir=dv.cr_dir, edge_start=dv.edge_start,
            vbin_start=dv.vbin_start, vbin_elem=dv.vbin_elem, node_solid=dv.node_solid,
            flags=dv.flags, cut_list=dv.cut_list, bseg_start=dv.bseg_start,
            region_start=dv.region_start, arc_start=dv.arc_start, island_start=dv.island_start,
            island_loop=dv.island_loop, island_kind=dv.island_kind, island_target=dv.island_target,
            rslot=dv.rslot, racc=dv.racc, sres=dv.sres, arcs=cache.arcs, bsegs=cache.bsegs)
end

function _pl_reset!(cache::PolylineClippingCutCellCache, g::CartesianGrid{2}, b::PolylineBlock)
    b.empty && return nothing
    backend = get_backend(cache)
    d = Tuple(_pl_dims(b))
    lo = CartesianIndex(Tuple(b.lo))
    _pl_reset_cells_kernel!(backend, 64)(cache.cells, cache.info, g, lo; ndrange=d)
    _pl_reset_edges_kernel!(backend, 64)(cache.edges.ax, g, lo, 1; ndrange=(d[1] + 1, d[2]))
    _pl_reset_edges_kernel!(backend, 64)(cache.edges.ay, g, lo, 2; ndrange=(d[1], d[2] + 1))
    return nothing
end

function _pl_write_block!(cache::PolylineClippingCutCellCache, g::CartesianGrid{2}, b::PolylineBlock)
    b.empty && return nothing
    backend = get_backend(cache)
    dv = cache.work.dev
    d = Tuple(_pl_dims(b))
    lo = CartesianIndex(Tuple(b.lo))
    _pl_classify_kernel!(backend, 64)(cache.cells, cache.info, dv.status, dv.flags, dv.slot, g, lo,
                                      d[1]; ndrange=d)
    _pl_uncut_edges_kernel!(backend, 64)(cache.edges.ax, dv.node_solid, dv.edge_start, g, lo, d[1],
                                         0, d[1] + 1, 1; ndrange=(d[1] + 1, d[2]))
    _pl_uncut_edges_kernel!(backend, 64)(cache.edges.ay, dv.node_solid, dv.edge_start, g, lo, d[1],
                                         (d[1] + 1) * d[2], d[1], 2; ndrange=(d[1], d[2] + 1))
    return nothing
end

# Counts of what this update produced, and a warning for any cell the walk could not do.
function _pl_gather_stats!(w::PolylineWork, ncut::Int, nregion::Vector{Int16})
    C = w.cross
    fl = _pl_download!(C.sflags, w.dev.sres.flags)
    ok = _pl_download!(C.sok, w.dev.sres.ok)
    ninvalid = count(!, ok)
    noverflow = count(f -> f & PL_FLAG_OVERFLOW != 0, fl)
    w.stats = (ncut=ncut, nsplit=count(>=(2), nregion), ninvalid=ninvalid, noverflow=noverflow,
               nisland=count(C.loop_island))
    if ninvalid > 0
        @warn "PolylineClippingCutCell: $ninvalid cut cells could not be walked and hold their " *
              "walk-free totals as one region ($noverflow over the $PL_MAX_CROSS-crossing limit); " *
              "see `cache.info.status .== PL_INVALID`"
    end
    return nothing
end
