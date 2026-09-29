# =====================================
# The PLIC cut-cell reconstruction: the per-cell plane fit, the per-face apertures it implies, and
# the surface clipped out of it -- one fit, all read through the cache a consumer holds (a
# `CutCellData` per cell, apertures resolved, and each cell's plane).
#
# The properties under test, in order:
#
#   * the fit is one `get_sdf` at each centroid, stored in this layer's conventions (fluid
#     fraction, `kind`)
#   * every aperture slot is written, boundary slots included, and none is a NaN
#   * a shared face is single-valued even at a sharp corner, where the two cells' planes disagree
#   * fractions against `calc_volume`, and the mesh clipped out of the same fit
#   * degenerate fits are flagged in `is_valid`, and never leave a `NaN` behind
#   * the extremes: a grid wholly inside the body, and one wholly outside it
#   * the grid must be isotropic, and the cache is where that is caught

using CutCellMethods: PLICCutCell, CutCellData, calc_volume, closure_residual,
                      face_area_of, is_cell_open, generate_mesh,
                      CELL_INSIDE, CELL_OUTSIDE, CELL_CUT
using CartesianMeshes: get_elem_centroid

const PLIC_METHOD = PLICCutCell()

# The 2D fixture: an off-lattice circle, so the cuts land at generic sub-cell offsets.
plic_grid_2d() = CartesianIsoGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), sz=0.1)
plic_geo_2d() = SDFCircle(center=SVector(0.0, 0.0), radius=1.0)

"""The PLIC cache over `grid`, updated from `geo`: what a consumer reads every field from."""
plic_cache(geo, grid) = update_cache!(allocate_cache(grid, PLIC_METHOD), geo, grid)

"""How many interior faces of a field of per-cell face fractions disagree between their two cells."""
function plic_face_mismatches(ff::AbstractArray{<:Any,D}) where {D}
    bad = 0
    for c in 1:D
        e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, D))
        for ci in CartesianIndices(ntuple(a -> a == c ? size(ff, a) - 1 : size(ff, a), D))
            ff[ci][2c] === ff[ci + e][2c - 1] || (bad += 1)
        end
    end
    return bad
end

@testset "PLIC cut cells" begin

    @testset "the fit is one `get_sdf` per centroid, in this layer's conventions" begin
        grid = plic_grid_2d()
        geo = plic_geo_2d()
        cache = plic_cache(geo, grid)
        f = cache.cells

        @test size(f) == size(cache.normals) == size(cache.intercepts) == size(cache.is_valid) ==
              Tuple(grid.n)
        @test eltype(f) === CutCellData{2,Float64,4,1}

        # The cache stores the normal `get_sdf` returned at the cell's centroid, and the plane it
        # implies through the cell: `n . xi = sum(n)/2 - dist/dx` in unit-cell coordinates, whose
        # clip is the SOLID fraction -- `volume_fraction` is the FLUID half, its complement.
        @test all(CartesianIndices(Tuple(grid.n))) do ci
            d, nrm = get_sdf(geo, get_elem_centroid(grid, ci))
            n = SVector{2,Float64}(nrm)
            a = sum(n) / 2 - d / grid.d[1]
            !cache.is_valid[ci] || (cache.normals[ci] === n && cache.intercepts[ci] ≈ a &&
                isapprox(f.volume_fraction[ci], 1 - CartesianMeshes.get_volume_fraction(n, a); atol=1e-14))
        end

        # `kind` is `volume_fraction`'s three-way reading, in one Int8 per cell. Its polarity
        # matches `CutCellData`: INSIDE means inside the BODY, which under a fluid fraction is
        # `volume_fraction == 0` -- the swap the flip implies.
        @test all(d -> d.kind == (iszero(d.volume_fraction) ? CELL_INSIDE :
                                  isone(d.volume_fraction) ? CELL_OUTSIDE : CELL_CUT), f)
        @test count(d -> d.kind == CELL_CUT, f) == count(d -> 0 < d.volume_fraction < 1, f)
        @test count(d -> d.kind == CELL_INSIDE, f) + count(d -> d.kind == CELL_OUTSIDE, f) +
              count(d -> d.kind == CELL_CUT, f) == length(f)

        # The fluid fraction is `calc_volume`'s integrand complemented, cell for cell.
        domain = prod(grid.n) * prod(grid.d)
        @test domain - sum(d -> d.volume_fraction, f) * prod(grid.d) ≈
              calc_volume(geo, grid) rtol = 1e-12
        @test all(d -> 0 <= d.volume_fraction <= 1, f)
    end

    @testset "every aperture slot is written, and none is a NaN" begin
        grid = plic_grid_2d()
        # All `2D` slots of every cell, boundary slots included -- there is no seed left unwritten
        # and no slot that only a neighbour would have filled.
        cache = plic_cache(plic_geo_2d(), grid)
        @test all(r -> all(x -> isfinite(x) && 0 <= x <= 1, r), cache.cells.face_fraction)
        # ...and the same in 3D, where a slot is an area rather than a length.
        g3 = CartesianIsoGrid(mincorner=(-1.5, -1.5, -1.5), maxcorner=(1.5, 1.5, 1.5), sz=0.3)
        c3 = plic_cache(SDFSphere(center=SVector(0.0, 0.0, 0.0), radius=1.0), g3)
        @test eltype(c3.cells) === CutCellData{3,Float64,6,2}
        @test eltype(c3.normals) === SVector{3,Float64}
        @test all(r -> all(x -> isfinite(x) && 0 <= x <= 1, r), c3.cells.face_fraction)
    end

    @testset "a shared face is single-valued at a sharp corner" begin
        # A union with a re-entrant corner, deliberately: that is where the two cells flanking a face
        # fit genuinely different planes, so each has its own candidate for how open the face is --
        # and the cache must still hand both cells one number, or a flux through the face
        # manufactures mass there.
        geo = SDFUnion(SDFCircle(center=SVector(0.02, 0.03), radius=0.31),
                       SDFRectangle(center=SVector(0.25, 0.10), dims=SVector(0.18, 0.18));
                       smoothing=0.0)
        grid = CartesianIsoGrid(mincorner=(-0.6, -0.6), maxcorner=(0.6, 0.6), sz=0.05)
        cache = plic_cache(geo, grid)
        ff = cache.cells.face_fraction
        @test count(==(CELL_CUT), cache.cells.kind) > 40
        @test plic_face_mismatches(ff) == 0
        @test all(r -> all(x -> 0 <= x <= 1, r), ff)
        # The interface is formed from those fractions, so every cell still closes exactly.
        @test all(m -> closure_residual(m, grid.d) == zero(SVector{2,Float64}), cache.cells)
    end

    @testset "the mesh clipped out of the same fit" begin
        grid = plic_grid_2d()
        geo = plic_geo_2d()
        cache = plic_cache(geo, grid)
        f = cache.cells

        # `generate_mesh` hands over every cell, and draws exactly the `kind == CELL_CUT` ones: an
        # uncut cell's plane clips to a zero-length segment, which the length floor drops.
        mesh = generate_mesh(cache, grid)
        @test length(mesh.elements) == count(d -> d.kind == CELL_CUT, f)
        @test length(mesh.elements) == 84
        # Two nodes per cut cell, never shared: the shards are deliberately disconnected.
        @test length(mesh.nodes) == 2 * length(mesh.elements)
        # The cache's own planes, clipped in the grid's cell order: the same segments a VOF solver
        # gets handing `extract_surface_plic` the cut cells' planes itself.
        cut = [ci for ci in CartesianIndices(Tuple(grid.n)) if f[ci].kind == CELL_CUT]
        ref = CutCellMethods.extract_surface_plic(cache.normals[cut], cache.intercepts[cut], cut, grid)
        @test mesh.nodes.coord == ref.nodes.coord

        # With nothing dropped, one segment per cell of the grid; a higher floor only ever shrinks it.
        @test length(generate_mesh(cache, grid; min_tri_area_fraction=0).elements) == prod(grid.n)
        @test length(generate_mesh(cache, grid; min_tri_area_fraction=0.1).elements) <= length(mesh.elements)
        # The cache knows its `n`, so the wrong grid is caught.
        @test_throws DimensionMismatch generate_mesh(cache, CartesianGrid(grid.x0, Tuple(grid.n) .+ 1, grid.d))

        grid3 = CartesianIsoGrid(mincorner=(-1.5, -1.5, -1.5), maxcorner=(1.5, 1.5, 1.5), sz=0.15)
        geo3 = SDFSphere(center=SVector(0.0, 0.0, 0.0), radius=1.0)
        cache3 = plic_cache(geo3, grid3)
        domain3 = prod(grid3.n) * prod(grid3.d)
        @test domain3 - sum(cache3.cells.volume_fraction) * prod(grid3.d) ≈
              calc_volume(geo3, grid3) rtol = 1e-12

        mesh3 = generate_mesh(cache3, grid3)
        @test length(mesh3.nodes) == 3384
        @test length(mesh3.elements) == 1688
    end

    @testset "a plane that does not cross its cell draws a point, not a stray segment" begin
        # A line through cell corners, so some cells are touched at exactly one corner -- where
        # `cell_plane_clip` alone would draw a false diagonal -- handed to `extract_surface_plic`
        # for every cell, cut or not, as a caller choosing its own cells may.
        grid = plic_grid_2d()
        nrm = normalize(SVector(1.0, 1.0))
        cache = plic_cache(SDFPlane(point=SVector(0.0, 0.0), normal=nrm), grid)
        all_ci = vec(collect(CartesianIndices(Tuple(grid.n))))
        # Nothing dropped, so every cell's segment is there to look at.
        m = CutCellMethods.extract_surface_plic(vec(cache.normals), vec(cache.intercepts), all_ci, grid;
                                                min_tri_area_fraction=0)
        @test length(m.elements) == length(all_ci)
        seg(k) = (m.nodes.coord[2k-1], m.nodes.coord[2k])
        cut = vec(cache.cells.kind) .== CELL_CUT
        @test count(cut) > 0
        # Every uncut cell draws a point: a zero-length segment, or -- where the line passes a hair
        # inside a corner, so the clip finds a real sliver the fraction rounded away -- one no longer
        # than roundoff. Never the false diagonal a one-corner touch used to draw.
        len(k) = norm(seg(k)[2] - seg(k)[1])
        @test all(k -> cut[k] || len(k) < 1e-12 * grid.d[1], eachindex(all_ci))
        @test count(k -> !cut[k] && iszero(len(k)), eachindex(all_ci)) > 0
        # ...and at the default floor those points are dropped: `generate_mesh` draws exactly the cut
        # cells' segments, both ends on the line.
        ref = generate_mesh(cache, grid)
        @test length(ref.elements) == count(cut)
        @test [seg(k) for k in findall(cut)] == [(ref.nodes.coord[2j-1], ref.nodes.coord[2j]) for j in 1:count(cut)]
        @test all(p -> abs(dot(nrm, p)) < 1e-12, ref.nodes.coord)
    end

    @testset "degenerate fits are flagged, not propagated" begin
        # A 3x3 grid over [-1.5, 1.5] puts a cell centroid exactly on the circle's centre -- its
        # medial axis, where the gradient is undefined and `cell_plane` returns a NaN normal.
        grid = CartesianGrid(SVector(-1.5, -1.5), (3, 3), SVector(1.0, 1.0))
        geo = plic_geo_2d()
        # The fixture is real: the distance field's own normal there is a NaN.
        @test any(isnan, last(get_sdf(geo, get_elem_centroid(grid, CartesianIndex(2, 2)))))

        # The fit is flagged degenerate, and only there: every neighbour, off the medial axis, is a
        # perfectly ordinary valid fit.
        cache = plic_cache(geo, grid)
        f = cache.cells
        @test !cache.is_valid[2, 2]
        @test count(!, cache.is_valid) == 1

        # The flag is the signal; the stored plane is zeroed only so the data stays finite.
        @test all(isfinite, cache.intercepts)
        @test all(n -> all(isfinite, n), cache.normals)
        @test iszero(cache.normals[2, 2])
        @test iszero(cache.intercepts[2, 2])

        # `volume_fraction` is still right there -- the centroid's sign is the whole answer -- and
        # it is what the face pass falls back on, so no NaN reaches the faces.
        @test f[2, 2].volume_fraction == 0        # centroid is inside the body, so no fluid here
        @test all(d -> all(isfinite, d.face_fraction), f)
        @test all(dir -> face_area_of(f[2, 2], dir, grid.d) < grid.d[1], 1:4)
    end

    @testset "wholly inside, wholly outside" begin
        # Inside the body: every cell CELL_INSIDE and every face dry.
        grid = CartesianIsoGrid(mincorner=(-0.2, -0.2), maxcorner=(0.2, 0.2), sz=0.1)
        inside = plic_cache(plic_geo_2d(), grid).cells
        @test all(d -> iszero(d.volume_fraction), inside)   # no fluid anywhere
        @test all(d -> all(iszero, d.face_fraction), inside)
        @test all(d -> d.kind == CELL_INSIDE, inside)
        @test !any(is_cell_open, inside)

        # Clear of the body: every cell CELL_OUTSIDE and every face fully open.
        far = CartesianIsoGrid(mincorner=(10.0, 10.0), maxcorner=(11.0, 11.0), sz=0.25)
        outside = plic_cache(plic_geo_2d(), far).cells
        @test all(d -> isone(d.volume_fraction), outside)   # all fluid
        @test all(d -> all(isone, d.face_fraction), outside)
        @test all(d -> d.kind == CELL_OUTSIDE, outside)
        @test all(is_cell_open, outside)

        # ...and in between, a fraction is always a fraction.
        grid2 = plic_grid_2d()
        cut = plic_cache(plic_geo_2d(), grid2).cells
        @test all(d -> all(x -> 0 <= x <= 1, d.face_fraction), cut)
    end

    @testset "the grid must be isotropic, and the cache is where that is caught" begin
        aniso = CartesianGrid(SVector(-1.5, -1.5), (30, 15), SVector(0.1, 0.2))
        @test_throws ArgumentError allocate_cache(aniso, PLIC_METHOD)
        iso = CartesianGrid(SVector(-1.5, -1.5), (30, 15), SVector(0.1, 0.1))
        @test_throws ArgumentError update_cache!(allocate_cache(iso, PLIC_METHOD), plic_geo_2d(), aniso)
    end

    @testset "Float32" begin
        # Built through the positional constructor, not `CartesianIsoGrid`, which promotes its
        # keyword corners to Float64 and would quietly make this a second Float64 test.
        g32 = CartesianGrid(SVector(-0.6f0, -0.6f0), (24, 24), SVector(0.05f0, 0.05f0))
        geo32 = SDFCircle(center=SVector(0.02f0, 0.03f0), radius=0.31f0)
        c32 = plic_cache(geo32, g32)
        f32 = c32.cells
        @test eltype(f32) === CutCellData{2,Float32,4,1}
        @test eltype(c32.normals) === SVector{2,Float32}
        @test eltype(c32.intercepts) === Float32
        @test all(d -> all(isfinite, d.face_fraction), f32)
        @test all(d -> 0 <= d.volume_fraction <= 1, f32)
    end
end
