# =====================================
# The reconstructed surface: what the cut looks like
#
# Each cut cell's interface is re-derived, on the host, from exactly what its update used -- the
# cell's Pass A table, its fit groups and planes, and its Boolean rule -- and emitted as the
# polygons its polytope has on the patch planes, fanned into triangles. So the surface drawn is the
# surface cut, cell by cell: per-cell facets, not stitched across cells (neighbouring cells fit
# their own planes, which meet only to the fits' order on a curved patch).
#
# The same walk gives the *creases*: the edges where two different patches' facets meet inside a
# cell. On a planar corner they are the true crease; on a curved one, within `O(h^2)` of it
# (`interface_creases`).

# A polygon face `f` of slot 1's polytope, fanned into triangles in global coordinates, appended to
# `X`/`tris`; `flip` reverses the winding. Returns how many triangles were added.
function _emit_face!(X, tris, scr, f::Int, o::SVector{3,T}, flip::Bool) where {T}
    b = _pbuf(scr, 1)
    L = Int(scr.nloop[f, b, 1])
    L < 3 && return 0
    base = length(X)
    for i in 1:L
        push!(X, SVector{3,Float64}(o + scr.x[Int(scr.loop[i, f, b, 1]), b, 1]))
    end
    nt = 0
    for i in 2:(L - 1)
        a, c, d = Int32(base + 1), Int32(base + i), Int32(base + i + 1)
        push!(tris, flip ? SVector(a, d, c) : SVector(a, c, d))
        nt += 1
    end
    return nt
end

# The edges shared by two patch-tagged faces of slot 1's polytope with different tags, appended
# to `segs` in global coordinates.
function _emit_creases!(segs, scr, o::SVector{3,T}, istagged) where {T}
    b = _pbuf(scr, 1)
    nf = poly_nf(scr, 1)
    for f in 1:nf, g in (f + 1):nf
        tf = scr.tag[f, b, 1]
        tg = scr.tag[g, b, 1]
        (istagged(f) && istagged(g) && tf != tg) || continue
        Lf = Int(scr.nloop[f, b, 1])
        Lg = Int(scr.nloop[g, b, 1])
        for k in 1:Lf
            u = scr.loop[k, f, b, 1]
            v = scr.loop[k == Lf ? 1 : k + 1, f, b, 1]
            for j in 1:Lg
                if scr.loop[j, g, b, 1] == v && scr.loop[j == Lg ? 1 : j + 1, g, b, 1] == u
                    push!(segs, (SVector{3,Float64}(o + scr.x[Int(u), b, 1]),
                                 SVector{3,Float64}(o + scr.x[Int(v), b, 1])))
                end
            end
        end
    end
    return segs
end

# The host's copies of what the cut kernels read.
function _host_views(cache::TriClippingCutCellCache)
    d = cache.work.dev
    pa = Adapt.adapt(Array, _pa_view(d))
    m = Adapt.adapt(Array, _mesh_view(d))
    return pa, m
end

"""
    _interface_geometry(cache, grid) -> (X, tris, tri_patch, tri_cell, creases)

Every cut cell's interface as triangles -- vertices `X` in global coordinates, `tris`, the patch
and the cell each came from -- wound out of the body, and the creases between patches' facets.
"""
function _interface_geometry(cache::TriClippingCutCellCache{T}, grid::CartesianGrid{3}) where {T}
    w = cache.work
    X = SVector{3,Float64}[]
    tris = SVector{3,Int32}[]
    tri_patch = Int32[]
    tri_cell = CartesianIndex{3}[]
    creases = Tuple{SVector{3,Float64},SVector{3,Float64}}[]
    (w.block.empty || w.topo === nothing) && return X, tris, tri_patch, tri_cell, creases
    g = convert(CartesianGrid{3,T}, grid)
    g.x0 == w.x0 || throw(ArgumentError("`grid` is not the grid the cache was last updated with"))
    pa, m = _host_views(cache)
    scr = TriScratch(KernelAbstractions.CPU(), T, 1)
    tols = cache.tols
    np_total = npatches(w.topo)
    blo = CartesianIndex(Tuple(w.block.lo))
    bdims = Tuple(_block_dims(w.block))
    for c in w.cut_list
        c = Int(c)
        ci = _block_ci(blo, bdims, Int(w.cand[c]))
        o = get_node(g, ci)
        U = get_node(g, ci + CartesianIndex(1, 1, 1)) - o
        n = _load_entries!(scr, 1, pa, c)
        k, rule, _ = _fit!(scr, 1, pa, m, c, 0, n, o, zero(SVector{3,T}), U, tols, np_total, pa.flag[c])
        ok = _emit_cell!(X, tris, tri_patch, tri_cell, creases, scr, m, c, ci, o, U, k, rule, tols)
        if !ok
            # What the cut kernel did too: the fallback plane.
            n = _load_entries!(scr, 1, pa, c)
            k, rule, _ = _fit!(scr, 1, pa, m, c, 0, n, o, zero(SVector{3,T}), U, tols, np_total,
                               pa.flag[c] | FLAG_UNSUPPORTED)
            _emit_cell!(X, tris, tri_patch, tri_cell, creases, scr, m, c, ci, o, U, 1, RULE_SINGLE, tols)
        end
    end
    return X, tris, tri_patch, tri_cell, creases
end

# One cell's interface and creases under `rule`; `false` if its clip failed (the kernel's cue for
# the fallback).
function _emit_cell!(X, tris, tri_patch, tri_cell, creases, scr, m, c::Int, ci, o::SVector{3,T},
                     U::SVector{3,T}, k::Int, rule::UInt8, tols) where {T}
    tol = tols.snap
    patch_of(tag) = first(_group_patch(scr, 1, Int(tag) - 6))
    function emit_tagged!(flip, keep)
        b = _pbuf(scr, 1)
        for f in 1:poly_nf(scr, 1)
            tag = scr.tag[f, b, 1]
            (tag > 6 && keep(f)) || continue
            nt = _emit_face!(X, tris, scr, f, o, flip)
            for _ in 1:nt
                push!(tri_patch, patch_of(tag))
                push!(tri_cell, ci)
            end
        end
    end
    if rule == RULE_MIXED
        _mixed_classify!(scr, 1, m, c, 0, k, o, zero(SVector{3,T}), U, tols) || return false
        for σ in 0:((1 << k) - 1)
            st = scr.lstat[σ + 1, 1]
            (st == Int8(1) || st == Int8(2)) || continue
            _leaf_poly!(scr, 1, σ, k, zero(SVector{3,T}), U, tol)
            b = _pbuf(scr, 1)
            onpatch(f) = (tag = Int(scr.tag[f, b, 1]); tag > 6 && scr.lface[tag - 6, σ + 1, 1] & 0x02 != 0x00)
            # The interface once, from its fluid side (wound into the body, so flipped); the
            # creases from both sides, since a convex one lies inside a solid piece.
            st == Int8(1) && emit_tagged!(true, onpatch)
            _emit_creases!(creases, scr, o, onpatch)
        end
        return true
    end
    poly_box!(scr, 1, U)
    st = 0x00
    if rule == RULE_CONVEX
        for i in 1:k
            p = scr.gplane[i, 1]
            st |= poly_clip!(scr, 1, _pn(p), _pd(p), 6 + i, tol, false)
        end
        st == 0x00 || return false
        emit_tagged!(false, f -> true)
    else
        for i in 1:k
            p = scr.gplane[i, 1]
            st |= poly_clip!(scr, 1, -_pn(p), -_pd(p), 6 + i, tol, true)
        end
        st == 0x00 || return false
        emit_tagged!(true, f -> true)
    end
    b = _pbuf(scr, 1)
    _emit_creases!(creases, scr, o, f -> scr.tag[f, b, 1] > 6)
    return true
end

# One element set per patch, named after its element set (with `#k` for a set split into several
# connected patches).
function _patch_sets(topo::TriTopology, tri_patch::Vector{Int32})
    sets = MeshElementSet{Int,Vector{Int}}[]
    seen = Dict{String,Int}()
    for p in 1:npatches(topo)
        elems = findall(==(Int32(p)), tri_patch)
        isempty(elems) && continue
        name = topo.patch_name[p]
        k = get(seen, name, 0) + 1
        seen[name] = k
        push!(sets, MeshElementSet(k == 1 ? name : "$name#$k", elems))
    end
    return sets
end

"""
    interface_mesh(cache, grid) -> (surface::Mesh{3}, cells::Vector{CartesianIndex{3}})

The interface `cache`'s last update cut, as a triangle mesh wound out of the body, with one
`MeshElementSet` per patch, and the cell each triangle came from. Per-cell facets, not stitched;
on the host. `grid` is the grid of that update.
"""
function interface_mesh(cache::TriClippingCutCellCache, grid::CartesianGrid{3})
    X, tris, tri_patch, tri_cell, _ = _interface_geometry(cache, grid)
    topo = cache.work.topo
    sets = topo === nothing ? MeshElementSet{Int,Vector{Int}}[] : _patch_sets(topo, tri_patch)
    surface = Mesh([Point(x) for x in X], [Tri(t) for t in tris]; element_sets=sets)
    return surface, tri_cell
end

"""
    interface_creases(cache, grid) -> Vector{Tuple{SVector{3},SVector{3}}}

The segments where two patches' facets meet inside a cut cell, in global coordinates: the
reconstructed creases. On a planar corner they lie on the true crease to roundoff; on a curved
patch within `O(h^2)` of it.
"""
interface_creases(cache::TriClippingCutCellCache, grid::CartesianGrid{3}) =
    last(_interface_geometry(cache, grid))

"""
    generate_mesh(cache, grid) -> Mesh{3}

The interface `cache`'s last update cut, as a `Mesh{3}` of `Tri` wound out of the body, with one
`MeshElementSet` per patch: each cut cell's facets, one patch plane at a time, not stitched across
cells. Read off the cache, not cut again; on the host. `grid` is the grid of that update.
[`interface_mesh`](@ref) also gives the cell each triangle came from.
"""
MeshLibrary.generate_mesh(cache::TriClippingCutCellCache, grid::CartesianGrid{3}) =
    first(interface_mesh(cache, grid))

"""
    write_cache_vtk(prefix, cache, grid; surface=true) -> Vector{String}

Write what `cache`'s last update produced for ParaView: `prefix_cells.vti`, the grid with every
cell's volume fraction, kind, Boolean rule, flags, patch count, closure correction and six face
fractions; and, with `surface`, `prefix_surface.vtu`, the reconstructed interface
([`generate_mesh`](@ref)`(cache, grid)`) with each triangle's patch. Returns the files written.
"""
function write_cache_vtk(prefix::AbstractString, cache::TriClippingCutCellCache, grid::CartesianGrid{3};
                         surface::Bool=true)
    cells = Adapt.adapt(Array, cache.cells)
    info = Adapt.adapt(Array, cache.info)
    data = Dict{String,Any}("volume_fraction" => cells.volume_fraction,
                            "kind" => Int32.(cells.kind),
                            "ambiguous" => Int32.(cells.ambiguous),
                            "rule" => Int32.(info.rule),
                            "flags" => Int32.(info.flags),
                            "npatch" => Int32.(info.npatch),
                            "correction" => info.correction)
    for (dir, name) in enumerate(("-x", "+x", "-y", "+y", "-z", "+z"))
        data["face_fraction_$name"] = map(f -> f[dir], cells.face_fraction)
    end
    files = String[]
    CartesianMeshes.write_vtk(prefix * "_cells", grid; cell_data=data)
    # WriteVTK picks the file type from the grid lines: image data for a uniform grid's ranges.
    push!(files, first(filter(isfile, prefix * "_cells" .* (".vti", ".vtr"))))
    if surface
        surf = MeshLibrary.generate_mesh(cache, grid)
        # VTK has no element sets, so each triangle's set becomes a per-triangle field.
        patch = zeros(Int32, length(surf.elements))
        for (k, set) in enumerate(surf.elemset)
            patch[set.elems] .= k
        end
        MeshLibrary.write_vtk(surf, prefix * "_surface"; cell_data=Dict("patch" => patch))
        push!(files, prefix * "_surface.vtu")
    end
    return files
end
