# =====================================
# The tri-clipping cache: every cell's cut geometry over a whole grid, kept between updates
#
# An update runs, in order:
#
#  0. the topology -- validation, patches, how they meet -- only when the mesh's connectivity or
#     element sets changed (host, cached);
#  1. the geometry: vertices, triangle normals and planar patches' global planes, as they are now;
#  2. the bins: which triangles each cell of the body's block, and each grid row, looks at (host);
#  3. a reset of the cells the body covered last time;
#  4. the row kernel: every cell of the block classified by ray parity as if uncut;
#  5. (P4, P5) the cut: per-cell patch moments, then per-cut-cell faces, polytope and closure.
#
# Only the block -- the cells the body can reach, plus one -- is ever written after allocation, so
# the rest of the grid keeps the no-body seed.

"""
    TriClippingCutCellCache

What [`allocate_cache`](@ref)`(grid, TriClippingCutCell(); backend)` builds and
[`update_cache!`](@ref)`(cache, mesh, grid)` refreshes:

- `cells` -- every cell's [`CutCellData`](@ref)`{3,T,6,2}`, a `StructArray` sized `grid.n`, as in
  every other cache: `cells.volume_fraction`, `cells.face_fraction`, `cells.kind`, ... as plain
  arrays, `cells[ci]` as one cell's struct.
- `info` -- every cell's [`TriClipCellInfo`](@ref): how many patches cut it, the Boolean rule, the
  diagnostic flags, the closure correction.

`mesh` is a watertight, outward-wound `Mesh{3}` of `Tri` elements in the grid's frame, with one
`MeshElementSet` per smooth patch in `mesh.elemset`. A moving body is the same mesh with moved
nodes, or the same body under a moved grid; either way the topology is not rebuilt as long as the
connectivity and the sets do not change.
"""
struct TriClippingCutCellCache{T,C<:AbstractArray{<:CutCellData},I,S,W} <: AbstractCutCellCache
    method::TriClippingCutCell
    tols::TriClipTols{T}
    grid::CartesianGrid{3,T}
    cells::C
    info::I
    scratch::S
    work::W
end

KernelAbstractions.get_backend(cache::TriClippingCutCellCache) = get_backend(cache.cells.volume_fraction)
Base.eltype(::TriClippingCutCellCache{T}) where {T} = T
@inline face_fractions(cache::TriClippingCutCellCache) = cache.cells.face_fraction

"""
    TriClippingCutCellView

What adapting a [`TriClippingCutCellCache`](@ref) gives: its results, `cells` and `info`, moved by
the adaptor, and none of what an update runs on (the topology, the binning buffers, the scratch).
It answers the uniform readers (`face_fractions`, `volume_fractions`, `kinds`) as the cache does.

The adaptor a GPU kernel launch applies to its arguments reaches a cache through here, so a cache
can be passed to a kernel directly or inside a consumer's own struct: the kernel gets the view,
which is isbits on the device, where the cache is not. `Adapt.adapt(Array, cache)` is a host copy of
the results. A view cannot be updated; to run on another backend, allocate the cache there.
"""
struct TriClippingCutCellView{C,I} <: AbstractCutCellCache
    cells::C
    info::I
end

Adapt.@adapt_structure TriClippingCutCellView
Adapt.adapt_structure(to, cache::TriClippingCutCellCache) =
    TriClippingCutCellView(Adapt.adapt(to, cache.cells), Adapt.adapt(to, cache.info))
KernelAbstractions.get_backend(view::TriClippingCutCellView) = get_backend(view.cells.volume_fraction)
@inline face_fractions(view::TriClippingCutCellView) = view.cells.face_fraction

# What an update keeps between calls: the topology, the block, and grow-only buffers -- host ones for
# the binning, backend ones (`dev`) for the kernels.
mutable struct TriClipWork{T,D<:NamedTuple}
    topo::Union{Nothing,TriTopology}
    block::TriClipBlock
    dev::D
    X::Vector{SVector{3,Float64}}
    tris::Vector{SVector{3,Int32}}
    tri_lo::Vector{SVector{3,Int32}}
    tri_hi::Vector{SVector{3,Int32}}
    bin_start::Vector{Int32}
    bin_tri::Vector{Int32}
    row_start::Vector{Int32}
    row_tri::Vector{Int32}
    cand::Vector{Int32}
    cut_list::Vector{Int32}
    cut_slot::Vector{Int32}
    stage3::Vector{SVector{3,T}}
    stage4::Vector{SVector{4,T}}
    x0::SVector{3,T}   # the grid origin the cells were last written for
end

function _tri_dev_buffers(backend, ::Type{T}) where {T}
    a(E) = KernelAbstractions.allocate(backend, E, 0)
    return (verts=a(SVector{3,T}), tris=a(SVector{3,Int32}), tri_patch=a(Int32),
            tri_normal=a(SVector{3,T}), side_patch=a(SVector{3,Int32}),
            side_label=a(SVector{3,UInt8}), patch_plane=a(SVector{4,T}), patch_planar=a(Bool),
            pair_mask=a(UInt8), bin_start=a(Int32), bin_tri=a(Int32), row_start=a(Int32),
            row_tri=a(Int32),
            # candidate cells (a non-empty bin), and those Pass A found cut
            cand=a(Int32), cut_list=a(Int32), cut_slot=a(Int32),
            # Pass A, per candidate: patch count, pair count, flags, clipped area ...
            pa_n=a(Int8), pa_npair=a(Int8), pa_flag=a(UInt8), pa_area=a(T),
            # ... TRI_K_MAX patch entries: id, A, sum A n, sum A c ...
            pa_id=a(Int32), pa_A=a(T), pa_N=a(SVector{3,T}), pa_M=a(SVector{3,T}),
            # ... and TRI_MAX_PAIRS local pair labels.
            pa_pp=a(Int32), pa_pq=a(Int32), pa_pm=a(UInt8),
            # each cut cell's boundary faces, TRI_K_MAX slots per candidate
            bfaces=_allocate_cells(backend, TriBoundaryFace{T}, (0,)))
end

# Pass A's outputs, and the geometry and lists every cut kernel reads, as the two NamedTuples the
# kernels take.
_pa_view(d) = (n=d.pa_n, npair=d.pa_npair, flag=d.pa_flag, area=d.pa_area, id=d.pa_id, A=d.pa_A,
               N=d.pa_N, M=d.pa_M, pp=d.pa_pp, pq=d.pa_pq, pm=d.pa_pm)
_mesh_view(d) = (verts=d.verts, tris=d.tris, tri_patch=d.tri_patch, tri_normal=d.tri_normal,
                 side_patch=d.side_patch, side_label=d.side_label, patch_plane=d.patch_plane,
                 patch_planar=d.patch_planar, pair_mask=d.pair_mask, bin_start=d.bin_start,
                 bin_tri=d.bin_tri, cand=d.cand, cut_list=d.cut_list, cut_slot=d.cut_slot)

# Resize every Pass A buffer for `ncand` candidates; grow-only in practice, since `resize!` keeps
# the capacity.
function _size_passA!(d, ncand::Int)
    for v in (d.pa_n, d.pa_npair, d.pa_flag, d.pa_area)
        resize!(v, ncand)
    end
    for v in (d.pa_id, d.pa_A, d.pa_N, d.pa_M)
        resize!(v, TRI_K_MAX * ncand)
    end
    for v in (d.pa_pp, d.pa_pq, d.pa_pm)
        resize!(v, TRI_MAX_PAIRS * ncand)
    end
    resize!(d.bfaces, TRI_K_MAX * ncand)
    return nothing
end

_on_host(x::Array) = x
_on_host(x) = Array(x)

function TriClipWork(backend, ::Type{T}, x0::SVector{3,T}) where {T}
    return TriClipWork(nothing, TriClipBlock(), _tri_dev_buffers(backend, T),
                       SVector{3,Float64}[], SVector{3,Int32}[], SVector{3,Int32}[],
                       SVector{3,Int32}[], Int32[], Int32[], Int32[], Int32[],
                       Int32[], Int32[], Int32[], SVector{3,T}[], SVector{4,T}[], x0)
end

# Copy a host vector into a grow-only backend vector.
function _upload!(dst::AbstractVector, src::AbstractVector)
    length(dst) == length(src) || resize!(dst, length(src))
    isempty(src) || copyto!(dst, src)
    return dst
end

function allocate_cache(grid::CartesianGrid{3,T}, method::TriClippingCutCell;
                        backend=KernelAbstractions.CPU(),
                        nslots::Integer=_default_nslots(backend)) where {T}
    cache = TriClippingCutCellCache(method, TriClipTols(method, grid), grid,
                                    _allocate_cells(backend, CutCellData{3,T,6,2}, Tuple(grid.n)),
                                    _allocate_cells(backend, TriClipCellInfo{T}, Tuple(grid.n)),
                                    TriScratch(backend, T, nslots), TriClipWork(backend, T, grid.x0))
    # Seeded as no body. An update against an empty mesh touches only what a previous update did,
    # so the first seed covers the whole grid.
    full = TriClipBlock(SVector(1, 1, 1), SVector{3,Int}(grid.n), false)
    _reset_block!(cache, grid, full)
    KernelAbstractions.synchronize(backend)
    return cache
end

allocate_cache(grid::CartesianGrid, ::TriClippingCutCell; kwargs...) =
    throw(ArgumentError("TriClippingCutCell cuts 3D grids only, got a $(length(grid.n))D grid"))

function update_cache!(cache::TriClippingCutCellCache{T}, mesh::Mesh,
                       grid::CartesianGrid{3}) where {T}
    _check_cache_size(cache.cells, grid)
    g = convert(CartesianGrid{3,T}, grid)
    g.d == cache.grid.d || throw(ArgumentError(
        "the grid's cell size $(Tuple(g.d)) is not the $(Tuple(cache.grid.d)) the cache was " *
        "allocated for: only a grid's origin may move between updates"))
    w = cache.work
    backend = get_backend(cache)
    # What has to be reset: the cells the body covered last time -- or, when the grid moved, every
    # cell, since an untouched cell's stored centroids are global coordinates of the old grid.
    # Recorded only once the reset has run, so an update that throws part-way leaves this right.
    prev = g.x0 == w.x0 ? w.block : TriClipBlock(SVector(1, 1, 1), SVector{3,Int}(g.n), false)

    if length(mesh.elements) == 0
        _reset_block!(cache, g, prev)
        w.block = TriClipBlock()
        w.x0 = g.x0
        KernelAbstractions.synchronize(backend)
        return cache
    end

    if w.topo === nothing || w.topo.fingerprint != _topology_fingerprint(mesh)
        w.topo = build_topology(mesh, cache.method, cache.tols.h)
        _upload_topology!(w, w.topo)
    end
    _refresh_geometry!(w, mesh)

    block = _bin_block!(w.tri_lo, w.tri_hi, w.X, w.tris, g, cache.tols.margin)
    _bin_cells!(w.bin_start, w.bin_tri, w.X, w.tris, w.tri_lo, w.tri_hi, block, g, cache.tols.margin)
    _bin_rows!(w.row_start, w.row_tri, w.X, w.tris, block, g)
    _upload!(w.dev.bin_start, w.bin_start)
    _upload!(w.dev.bin_tri, w.bin_tri)
    _upload!(w.dev.row_start, w.row_start)
    _upload!(w.dev.row_tri, w.row_tri)

    _reset_block!(cache, g, prev)
    w.x0 = g.x0
    _pass_a!(cache, g, block)
    _classify_rows!(cache, g, block)
    _flag_conflicts!(cache, g, block)
    _cut_cells!(cache, g, block)
    KernelAbstractions.synchronize(backend)
    w.block = block
    return cache
end

# Pass A over the candidates -- the block's cells with a non-empty bin -- then, on the host, the cut
# list and each block cell's slot in it.
function _pass_a!(cache::TriClippingCutCellCache{T}, g::CartesianGrid{3}, b::TriClipBlock) where {T}
    w = cache.work
    d = w.dev
    ncell = _block_ncells(b)
    resize!(w.cand, 0)
    for l in 1:ncell
        w.bin_start[l + 1] > w.bin_start[l] && push!(w.cand, Int32(l))
    end
    ncand = length(w.cand)
    _upload!(d.cand, w.cand)
    _size_passA!(d, ncand)
    resize!(w.cut_slot, ncell)
    fill!(w.cut_slot, 0)
    resize!(w.cut_list, 0)
    if ncand > 0
        backend = get_backend(cache)
        ns = min(nslots(cache.scratch), ncand)
        _tri_passA_kernel!(backend, 64)(_pa_view(d), _mesh_view(d), cache.scratch, g,
                                        CartesianIndex(Tuple(b.lo)), Tuple(_block_dims(b)),
                                        cache.tols, ns, ncand; ndrange=ns)
        KernelAbstractions.synchronize(backend)
        area = _on_host(d.pa_area)
        for c in 1:ncand
            if area[c] > cache.tols.drop_area
                push!(w.cut_list, Int32(c))
                w.cut_slot[w.cand[c]] = Int32(c)
            end
        end
    end
    _upload!(d.cut_list, w.cut_list)
    _upload!(d.cut_slot, w.cut_slot)
    return nothing
end

function _cut_cells!(cache::TriClippingCutCellCache, g::CartesianGrid{3}, b::TriClipBlock)
    w = cache.work
    ncut = length(w.cut_list)
    ncut == 0 && return nothing
    backend = get_backend(cache)
    ns = min(nslots(cache.scratch), ncut)
    d = w.dev
    _tri_cell_kernel!(backend, 64)(cache.cells, cache.info, d.bfaces, _pa_view(d), _mesh_view(d),
                                   cache.scratch, g, CartesianIndex(Tuple(b.lo)),
                                   Tuple(_block_dims(b)), cache.tols, npatches(w.topo), ns, ncut;
                                   ndrange=ns)
    return nothing
end

# Mark uncut cells an uncut neighbour disagrees with. After the row kernel, which wrote their
# kinds, and before the cut kernel, which writes only cut cells.
function _flag_conflicts!(cache::TriClippingCutCellCache, g::CartesianGrid{3}, b::TriClipBlock)
    b.empty && return nothing
    backend = get_backend(cache)
    _tri_conflict_kernel!(backend, 64)(cache.info, cache.cells, cache.work.dev.cut_slot,
                                       CartesianIndex(Tuple(b.lo)), Tuple(_block_dims(b)),
                                       Tuple(g.n); ndrange=Tuple(_block_dims(b)))
    return nothing
end

"""
    boundary_faces(cache, ci) -> AbstractVector{TriBoundaryFace}

Cut cell `ci`'s interface split by fit group: one [`TriBoundaryFace`](@ref) per group -- its patch,
area, unit normal out of the body and centroid -- whose `area * normal` sum to the cell's
`interface_normal_area` to within a rounding. Empty for an uncut cell. A view into the cache,
valid until the next update; on the host only.
"""
function boundary_faces(cache::TriClippingCutCellCache, ci::CartesianIndex{3})
    w = cache.work
    bf = w.dev.bfaces
    b = w.block
    (b.empty || any(Tuple(ci) .< b.lo) || any(Tuple(ci) .> b.hi)) && return view(bf, 1:0)
    c = Int(w.cut_slot[_block_lin(CartesianIndex(Tuple(b.lo)), Tuple(_block_dims(b)), ci)])
    c == 0 && return view(bf, 1:0)
    k = Int(cache.info.npatch[ci])
    return view(bf, ((c - 1) * TRI_K_MAX + 1):((c - 1) * TRI_K_MAX + k))
end

# By linear index too -- what `findfirst(==(CELL_CUT), cache.cells.kind)` returns, since Base's
# byte-search fast path for `Int8` arrays gives a linear index rather than a Cartesian one.
boundary_faces(cache::TriClippingCutCellCache, i::Integer) =
    boundary_faces(cache, CartesianIndices(size(cache.cells))[i])

"""
    cut_report(cache) -> NamedTuple

What the last update produced: cell counts by kind and by Boolean rule, how many cells carry each
diagnostic flag, the largest closure correction, and the body's volume against the mesh's own.
Copies to the host on a device backend.
"""
function cut_report(cache::TriClippingCutCellCache{T}) where {T}
    kind = _on_host(cache.cells.kind)
    vf = _on_host(cache.cells.volume_fraction)
    rule = _on_host(cache.info.rule)
    flags = _on_host(cache.info.flags)
    corr = _on_host(cache.info.correction)
    nflag(bit) = count(f -> f & bit != 0x00, flags)
    topo = cache.work.topo
    solid = sum(x -> one(T) - x, vf) * prod(cache.grid.d)
    return (cut=count(==(CELL_CUT), kind), inside=count(==(CELL_INSIDE), kind),
            outside=count(==(CELL_OUTSIDE), kind),
            single=count(==(RULE_SINGLE), rule), convex=count(==(RULE_CONVEX), rule),
            concave=count(==(RULE_CONCAVE), rule), mixed=count(==(RULE_MIXED), rule),
            fallback=count(==(RULE_FALLBACK), rule),
            multi_patch=nflag(FLAG_MULTI_PATCH), unsupported=nflag(FLAG_UNSUPPORTED),
            split=nflag(FLAG_SPLIT), closure_fallback=nflag(FLAG_CLOSURE_FALLBACK),
            corr_large=nflag(FLAG_CORR_LARGE), overflow=nflag(FLAG_OVERFLOW),
            chain_fail=nflag(FLAG_CHAIN_FAIL), status_conflict=nflag(FLAG_STATUS_CONFLICT),
            max_correction=isempty(corr) ? zero(T) : maximum(corr),
            solid_volume=solid, mesh_volume=topo === nothing ? zero(T) : T(topo.volume))
end

"""
    reset_topology!(cache) -> cache

Forget the cached topology, so the next update validates and segments the mesh again. Needed only
if a mesh's connectivity or element sets were changed in place in a way that leaves their contents'
hash unchanged -- which is not expected to happen -- or after non-rigid motion that changes how
patches meet.
"""
function reset_topology!(cache::TriClippingCutCellCache)
    cache.work.topo = nothing
    return cache
end

function _upload_topology!(w::TriClipWork, topo::TriTopology)
    d = w.dev
    _upload!(d.tri_patch, topo.tri_patch)
    _upload!(d.side_patch, topo.side_patch)
    _upload!(d.side_label, topo.side_label)
    _upload!(d.patch_planar, topo.patch_planar)
    _upload!(d.pair_mask, vec(topo.pair_mask))
    return nothing
end

# The body as it is now: vertices, triangles, unit normals and the planar patches' global planes,
# computed in `Float64` and handed to the kernels in the cache's type.
function _refresh_geometry!(w::TriClipWork{T}, mesh::Mesh) where {T}
    nv = length(mesh.nodes)
    nt = length(mesh.elements)
    resize!(w.X, nv)
    for (i, p) in enumerate(mesh.nodes)
        w.X[i] = SVector{3,Float64}(p.coord)
    end
    resize!(w.tris, nt)
    for (t, e) in enumerate(mesh.elements)
        w.tris[t] = SVector{3,Int32}(e.con)
    end
    resize!(w.stage3, nv)
    for i in 1:nv
        w.stage3[i] = SVector{3,T}(w.X[i])
    end
    _upload!(w.dev.verts, w.stage3)
    _upload!(w.dev.tris, w.tris)
    resize!(w.stage3, nt)
    for t in 1:nt
        c = w.tris[t]
        p1, p2, p3 = w.X[c[1]], w.X[c[2]], w.X[c[3]]
        cr = cross(p2 - p1, p3 - p1)
        a2 = norm(cr)
        w.stage3[t] = SVector{3,T}(a2 > 0 ? cr / a2 : zero(cr))
    end
    _upload!(w.dev.tri_normal, w.stage3)
    planes = _patch_planes(w.topo, w.X, w.tris)
    resize!(w.stage4, length(planes))
    for p in eachindex(planes)
        w.stage4[p] = SVector{4,T}(planes[p])
    end
    _upload!(w.dev.patch_plane, w.stage4)
    return nothing
end

function _reset_block!(cache::TriClippingCutCellCache, g::CartesianGrid{3}, b::TriClipBlock)
    b.empty && return nothing
    backend = get_backend(cache)
    _tri_reset_kernel!(backend, 64)(cache.cells, cache.info, g, CartesianIndex(Tuple(b.lo));
                                    ndrange=Tuple(_block_dims(b)))
    return nothing
end

function _classify_rows!(cache::TriClippingCutCellCache, g::CartesianGrid{3}, b::TriClipBlock)
    b.empty && return nothing
    backend = get_backend(cache)
    nrows = _block_nrows(b)
    ns = min(nslots(cache.scratch), nrows)
    d = cache.work.dev
    _tri_rows_kernel!(backend, 64)(cache.cells, cache.info, cache.scratch, d.verts, d.tris,
                                   d.row_start, d.row_tri, g, CartesianIndex(Tuple(b.lo)),
                                   Tuple(_block_dims(b)), ns, nrows; ndrange=ns)
    return nothing
end
