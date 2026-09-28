# =====================================
# The PLIC cut-cell reconstruction: the per-cell plane fit, the per-face apertures it implies, and
# the surface clipped out of it -- one fit, read two ways, with nothing cached between them.
#
# The properties under test, in order:
#
#   * the fit is `cell_plane`'s, stored in this layer's conventions (fluid fraction, `kind`)
#   * every aperture slot is written, boundary slots included, and none is a NaN
#   * an interior face's two slots are ONE-SIDED -- they genuinely differ -- and averaging them is
#     order-independent, which is what makes a shared face single-valued
#   * fractions against `calc_volume`, and the mesh clipped out of the same fit
#   * degenerate fits are flagged in `is_valid`, and never leave a `NaN` behind
#   * the extremes: a grid wholly inside the body, and one wholly outside it
#   * the grid must be isotropic, and `generate_mesh` is where that is caught
#
# The averaging equality is asserted bitwise rather than to a tolerance for the same reason
# `test_moments.jl` does it: floating-point addition is commutative, so it holds exactly or the
# claim in `plic/apertures.jl`'s header is wrong.

using CutCellMethods: PLICCutCell, PLICCutCellData, cut_cell_moments, cell_plane, calc_volume,
                      face_area_of, face_fraction_of, is_face_open, is_cell_open, generate_mesh,
                      CELL_INSIDE, CELL_OUTSIDE, CELL_CUT

const PLIC_METHOD = PLICCutCell()

# The 2D fixture: an off-lattice circle, so the cuts land at generic sub-cell offsets.
plic_grid_2d() = CartesianIsoGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), sz=0.1)
plic_geo_2d() = SDFCircle(center=SVector(0.0, 0.0), radius=1.0)

"""The whole grid's worth of per-cell data, which is what a consumer builds for itself now that
there is no cache. An `Array{PLICCutCellData}` shaped like the grid."""
plic_field(geo, grid) =
    [cut_cell_moments(PLIC_METHOD, grid, geo, ci) for ci in CartesianIndices(Tuple(grid.n))]

@testset "PLIC cut cells" begin

    @testset "the fit is `cell_plane`'s, in this layer's conventions" begin
        grid = plic_grid_2d()
        geo = plic_geo_2d()
        f = plic_field(geo, grid)

        @test size(f) == Tuple(grid.n)
        @test eltype(f) === PLICCutCellData{2,Float64,4}

        # Bitwise: this path stores what `cell_plane` returned, taking exactly one complement.
        # `volume_fraction` is the FLUID half, `plane.fraction` the solid one.
        @test all(CartesianIndices(Tuple(grid.n))) do ci
            p = cell_plane(geo, grid, ci)
            d = f[ci]
            d.normal === p.normal && d.intercept === p.intercept &&
                d.volume_fraction === one(Float64) - p.fraction && d.is_valid === p.is_valid
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
        f = plic_field(plic_geo_2d(), grid)
        # All `2D` slots of every cell, boundary slots included -- there is no seed left unwritten
        # and no slot that only a neighbour would have filled.
        @test all(d -> all(isfinite, d.face_area), f)
        @test all(d -> all(a -> 0 <= a <= grid.d[1], d.face_area), f)
        # ...and the same in 3D, where a slot is an area rather than a length.
        g3 = CartesianIsoGrid(mincorner=(-1.5, -1.5, -1.5), maxcorner=(1.5, 1.5, 1.5), sz=0.3)
        f3 = plic_field(SDFSphere(center=SVector(0.0, 0.0, 0.0), radius=1.0), g3)
        @test eltype(f3) === PLICCutCellData{3,Float64,6}
        @test all(d -> all(isfinite, d.face_area), f3)
        @test all(d -> all(a -> 0 <= a <= g3.d[1]^2, d.face_area), f3)
    end

    @testset "an interior face's two slots are one-sided" begin
        # A union with a re-entrant corner, deliberately: the whole point of the one-sided apertures
        # is that two cells disagree about a shared face where their centroid fits disagree, and on
        # a smooth convex body they barely do. `ndisagree > 0` is what would catch a neighbour
        # average creeping back into this path.
        geo = SDFUnion(SDFCircle(center=SVector(0.02, 0.03), radius=0.31),
                       SDFRectangle(center=SVector(0.25, 0.10), dims=SVector(0.18, 0.18));
                       smoothing=0.0)
        grid = CartesianIsoGrid(mincorner=(-0.6, -0.6), maxcorner=(0.6, 0.6), sz=0.05)
        f = plic_field(geo, grid)
        n = Tuple(grid.n)

        ndisagree = 0
        for c in 1:2
            e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, 2))
            for ci in CartesianIndices(ntuple(a -> a == c ? n[a] - 1 : n[a], 2))
                lo, hi = f[ci].face_area[2c], f[ci+e].face_area[2c-1]
                lo === hi || (ndisagree += 1)
                # Averaging is order-independent BITWISE, which is what makes the resolved face
                # single-valued without any accumulation-order contract. `+` is commutative;
                # this is the assertion that nothing else in the expression breaks that.
                @test (lo + hi) / 2 === (hi + lo) / 2
            end
        end
        @test ndisagree > 0
    end

    @testset "the mesh clipped out of the same fit" begin
        grid = plic_grid_2d()
        geo = plic_geo_2d()
        f = plic_field(geo, grid)

        # `kind == CELL_CUT` is exactly the set `generate_mesh` draws at `tol = 0`.
        mesh = generate_mesh(geo, grid, PLIC_METHOD)
        @test length(mesh.elements) == count(d -> d.kind == CELL_CUT, f)
        @test length(mesh.elements) == 84
        # Two nodes per cut cell, never shared: the shards are deliberately disconnected.
        @test length(mesh.nodes) == 2 * length(mesh.elements)

        # `tol` drops slivers, so it can only ever shrink the set.
        @test length(generate_mesh(geo, grid, PLIC_METHOD; tol=1e-3).elements) <=
              length(mesh.elements)

        grid3 = CartesianIsoGrid(mincorner=(-1.5, -1.5, -1.5), maxcorner=(1.5, 1.5, 1.5), sz=0.15)
        geo3 = SDFSphere(center=SVector(0.0, 0.0, 0.0), radius=1.0)
        f3 = plic_field(geo3, grid3)
        domain3 = prod(grid3.n) * prod(grid3.d)
        @test domain3 - sum(d -> d.volume_fraction, f3) * prod(grid3.d) ≈
              calc_volume(geo3, grid3) rtol = 1e-12

        mesh3 = generate_mesh(geo3, grid3, PLIC_METHOD; tol=1e-6)
        @test length(mesh3.nodes) == 3384
        @test length(mesh3.elements) == 1688
    end

    @testset "degenerate fits are flagged, not propagated" begin
        # A 3x3 grid over [-1.5, 1.5] puts a cell centroid exactly on the circle's centre -- its
        # medial axis, where the gradient is undefined and `cell_plane` returns a NaN normal.
        grid = CartesianGrid(SVector(-1.5, -1.5), (3, 3), SVector(1.0, 1.0))
        geo = plic_geo_2d()
        # The fixture is real, and `cell_plane` guards it at the source: the fit is flagged
        # degenerate and every field it hands back is finite, so no NaN ever reaches a caller.
        degenerate = cell_plane(geo, grid, CartesianIndex(2, 2))
        @test !degenerate.is_valid
        @test isfinite(degenerate.intercept)
        @test all(isfinite, degenerate.normal)
        @test isfinite(degenerate.fraction)
        # ...and a neighbouring cell, off the medial axis, is a perfectly ordinary valid fit.
        @test cell_plane(geo, grid, CartesianIndex(1, 1)).is_valid

        f = plic_field(geo, grid)
        @test !f[2, 2].is_valid
        @test count(d -> !d.is_valid, f) == 1

        # The flag is the signal; the stored plane is zeroed only so the data stays finite.
        @test all(d -> isfinite(d.intercept), f)
        @test all(d -> all(isfinite, d.normal), f)
        @test iszero(f[2, 2].normal)
        @test iszero(f[2, 2].intercept)

        # `volume_fraction` is still right there -- the centroid's sign is the whole answer -- and
        # it is what `cell_face_areas` falls back on, so no NaN reaches the faces.
        @test f[2, 2].volume_fraction == 0        # centroid is inside the body, so no fluid here
        @test all(d -> all(isfinite, d.face_area), f)
        @test all(dir -> face_area_of(f[2, 2], dir, grid.d) < grid.d[1], 1:4)
    end

    @testset "wholly inside, wholly outside" begin
        # Inside the body: every cell CELL_INSIDE and every face dry.
        grid = CartesianIsoGrid(mincorner=(-0.2, -0.2), maxcorner=(0.2, 0.2), sz=0.1)
        inside = plic_field(plic_geo_2d(), grid)
        @test all(d -> iszero(d.volume_fraction), inside)   # no fluid anywhere
        @test all(d -> all(iszero, d.face_area), inside)
        @test all(d -> d.kind == CELL_INSIDE, inside)
        @test !any(is_cell_open, inside)
        @test all(iszero, (face_fraction_of(d, dir, grid.d) for d in inside, dir in 1:4))

        # Clear of the body: every cell CELL_OUTSIDE and every face fully open.
        far = CartesianIsoGrid(mincorner=(10.0, 10.0), maxcorner=(11.0, 11.0), sz=0.25)
        outside = plic_field(plic_geo_2d(), far)
        @test all(d -> isone(d.volume_fraction), outside)   # all fluid
        @test all(d -> all(==(far.d[1]), d.face_area), outside)
        @test all(d -> d.kind == CELL_OUTSIDE, outside)
        @test all(is_cell_open, outside)
        @test all(==(1), (face_fraction_of(d, dir, far.d) for d in outside, dir in 1:4))
        @test all(d -> all(dir -> is_face_open(d, dir), 1:4), outside)

        # ...and in between, a fraction is always a fraction.
        grid2 = plic_grid_2d()
        cut = plic_field(plic_geo_2d(), grid2)
        @test all(d -> all(dir -> 0 <= face_fraction_of(d, dir, grid2.d) <= 1, 1:4), cut)
    end

    @testset "the grid must be isotropic, and generate_mesh is where that is caught" begin
        aniso = CartesianGrid(SVector(-1.5, -1.5), (30, 15), SVector(0.1, 0.2))
        @test_throws ArgumentError generate_mesh(plic_geo_2d(), aniso, PLIC_METHOD)
        # The per-cell form deliberately does NOT check -- it has to stay launchable -- so it
        # answers rather than throwing. This pins that difference so neither side drifts.
        @test cut_cell_moments(PLIC_METHOD, aniso, plic_geo_2d(), CartesianIndex(1, 1)) isa
              PLICCutCellData{2,Float64,4}
    end

    @testset "Float32" begin
        # Built through the positional constructor, not `CartesianIsoGrid`, which promotes its
        # keyword corners to Float64 and would quietly make this a second Float64 test.
        g32 = CartesianGrid(SVector(-0.6f0, -0.6f0), (24, 24), SVector(0.05f0, 0.05f0))
        geo32 = SDFCircle(center=SVector(0.02f0, 0.03f0), radius=0.31f0)
        f32 = plic_field(geo32, g32)
        @test eltype(f32) === PLICCutCellData{2,Float32,4}
        @test all(d -> all(isfinite, d.face_area), f32)
        @test all(d -> 0 <= d.volume_fraction <= 1, f32)
    end
end
