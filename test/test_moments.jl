# =====================================
# Cut-cell moments over the nodal marching-squares reconstruction.
#
# The properties under test, in order:
#
#   * the per-cell construction against moments computable by hand
#   * fractions AND centroids against the analytic circle, with convergence rates
#   * bitwise watertightness between neighbours -- the claim everything else rests on
#   * the reconstructed contour, and that it agrees with the moments built alongside it
#   * per-cell geometric conservation, at every level, on either backend
#   * refinement consistency: children are recomputed, never inherited
#   * kernel discipline, and CPU/GPU parity
#
# Everything is read off a marching-squares cache -- on a grid, on a tree, or on a grid of one cell
# for the hand-checkable cases -- which is how the construction is used.
#
# Two of these are asserted as *bitwise* equalities rather than tolerances -- watertightness and the
# closure residual. They hold exactly or the reasoning in `marching_squares.jl` and `moments.jl` is
# wrong somewhere, and a tolerance would hide precisely the failure worth catching.

using CutCellMethods: CutCellData, MarchingSquaresCutCell, generate_mesh,
                      is_cut, is_inside, is_outside, is_ambiguous,
                      interface_area, interface_normal, interface_normal_area, face_vector_area,
                      closure_residual, cut_cell_report,
                      face_centroid, face_area_of, is_cell_open,
                      cell_nodes, MS_NODE_BITS

using CartesianMeshes: get_elem_volume, get_elem_centroid, get_elem_size

const GEO_CENTRE = SVector(0.13, -0.07)     # off-centre and off-lattice, so the
const GEO_RADIUS = 0.9                      # cuts land at generic sub-cell offsets
const GEO_CIRCLE = SDFCircle(GEO_CENTRE, GEO_RADIUS)

const MS = MarchingSquaresCutCell()

"""Every cell's moments over `domain`, which is what most of the assertions below want: a
marching-squares cache's `cells`, over a tree's leaves in leaf order or over a grid's cells."""
cut_field(geo, domain) = update_cache!(allocate_cache(domain, MS), geo, domain).cells

"""One isolated box's moments: a marching-squares cache over a grid of that one cell, updated from
`geo` -- the hand-checkable cells below."""
function box_moments(geo, lo::SVector{2,T}, hi::SVector{2,T}) where {T}
    g = CartesianGrid(lo, (1, 1), hi - lo)
    return only(cut_field(geo, g))
end

"""Whether every cell of `cells`, laid out over `mesh`'s leaves, closes exactly."""
closes_exactly(cells, mesh::AdaptiveMesh) =
    all(i -> closure_residual(cells[i], get_elem_size(mesh, leaf(mesh, i))) == zero(SVector{2,eltype(eltype(cells))}),
        eachindex(cells))

"""The contour `generate_mesh` built, as `(p, q)` endpoint pairs in element order -- what the
assertions below are phrased in. The mesh stitches shared vertices, so this reads the segments
back out of the connectivity rather than storing them twice."""
function contour_segments(contour)
    x = contour.nodes.coord
    return [(x[e.con[1]], x[e.con[2]]) for e in contour.elements]
end

"""A uniform mesh of `n x n` cells over `[-2, 2]^2`."""
function uniform_geometry_mesh(n::Integer)
    base = CartesianGrid(mincorner=(-2.0, -2.0), maxcorner=(2.0, 2.0), n=(n, n))
    return AdaptiveMesh(base; max_level=1, initial_level=0)
end

"""A mesh carrying genuine level jumps, with a uniformly refined band around the circle so that
every cut cell sits at the finest level and none touches a non-conforming face.

The band is built here rather than imported because refining around a surface is a *consumer's*
policy, not this package's: what is under test below is that the moments are right on a mesh with
level jumps in it, and this is the cheapest way to get one."""
function banded_geometry_mesh(; level=4, width=2)
    base = CartesianGrid(mincorner=(-2.0, -2.0), maxcorner=(2.0, 2.0), n=(8, 8))
    mesh = AdaptiveMesh(base; max_level=5, initial_level=1)
    for _ in 1:level
        flags = falses(nleaves(mesh))
        any_flagged = false
        for i in eachleaf(mesh)
            leaf_level(mesh, i) < level || continue
            c = leaf(mesh, i)
            h = get_elem_size(mesh, c)
            diag = sqrt(sum(h .* h))
            d = abs(first(get_sdf(GEO_CIRCLE, get_elem_centroid(mesh, c))))
            if d <= (width + 0.5) * diag
                flags[i] = true
                any_flagged = true
            end
        end
        any_flagged || break
        refine!(mesh, flags)
    end
    return mesh
end

# A marching-squares cache over a tree's leaves, updated from `geo`: each leaf's four corner values
# and its moments, in leaf order, as `generate_mesh(cache, mesh)` takes it.
ms_leaf_cache(geo, mesh) = update_cache!(allocate_cache(mesh, MarchingSquaresCutCell()), geo, mesh)

# Convergence rates from errors on successively halved h.
#
# Asserted as an **overall** rate across the whole sweep, with a looser floor on each individual
# pair. That is not slack: every statistic here is a max or a sum over the cut cells, and which
# cells those are shifts with the resolution, so consecutive pairs fluctuate by a few tenths either
# side of the true rate while the trend over three halvings does not.
pair_rates(e) = [log2(e[j] / e[j+1]) for j in 1:(length(e)-1)]
overall_rate(e) = log2(e[begin] / e[end]) / (length(e) - 1)

@testset verbose = true "cut-cell moments" begin

    # =====================================================================
    @testset "the nodal construction" begin
        # A body occupying y < 0.3 through the unit cell. Every moment is computable by hand, which
        # is the point of choosing it.
        UNIT = SVector(1.0, 1.0)
        UNIT_NODES = SVector{4,SVector{2,Float64}}(SVector(0.0, 0.0), SVector(1.0, 0.0),
                                                   SVector(1.0, 1.0), SVector(0.0, 1.0))
        m = box_moments(x -> x[2] - 0.3, SVector(0.0, 0.0), UNIT)
        @test is_cut(m)
        @test !is_outside(m) && !is_inside(m) && !is_ambiguous(m)
        @test m.volume_fraction ≈ 0.7
        @test m.centroid ≈ [0.5, 0.65]
        @test m.face_fraction ≈ [0.7, 0.7, 0.0, 1.0]          # -x +x -y +y
        @test [face_area_of(m, k, UNIT) for k in 1:4] ≈ [0.7, 0.7, 0.0, 1.0]
        @test face_centroid(m, 1, UNIT_NODES) ≈ [0.0, 0.65]
        @test face_centroid(m, 4, UNIT_NODES) ≈ [0.5, 1.0]
        # Stored as the one in-plane offset from the lower corner: y for an x-face, x for a y-face.
        @test m.face_centroid_local[1] ≈ [0.65] && m.face_centroid_local[4] ≈ [0.5]
        @test interface_area(m, UNIT) ≈ 1.0
        @test interface_normal(m, UNIT) ≈ [0.0, 1.0]          # out of the body
        @test m.interface_centroid ≈ [0.5, 0.3]

        # A diagonal cut, so the interface normal is not axis-aligned.
        m2 = box_moments(x -> x[1] + x[2] - 0.5, SVector(0.0, 0.0), UNIT)
        @test m2.volume_fraction ≈ 1 - 0.125
        @test interface_area(m2, UNIT) ≈ sqrt(0.5)
        @test interface_normal(m2, UNIT) ≈ SVector(1, 1) / sqrt(2)
        @test m2.interface_centroid ≈ [0.25, 0.25]

        # Anisotropic cells are supported: nothing here divides by a single `dx` the way
        # `calc_volume` has to. This is also the test that `face_area_of` uses each face's OWN full
        # measure -- the x-faces scale by 4.0 and the y-faces by 1.0, so a single shared divisor
        # would fail here and pass on the unit cell above.
        ANISO = SVector(1.0, 4.0)
        m3 = box_moments(x -> x[1] - 0.25, SVector(0.0, 0.0), ANISO)
        @test m3.volume_fraction ≈ 0.75
        @test [face_area_of(m3, k, ANISO) for k in 1:4] ≈ [0.0, 4.0, 0.75, 0.75]
        @test m3.face_fraction ≈ [0.0, 1.0, 0.75, 0.75]
        @test interface_area(m3, ANISO) ≈ 4.0

        @testset "uncut cells are exact, not merely close" begin
            SMALL = SVector(0.5, 0.25)
            mo = box_moments(x -> 1.0, SVector(1.5, -3.0), SVector(2.0, -2.75))
            @test is_outside(mo)
            @test mo.volume_fraction === 1.0                  # exactly, not 0.999...
            @test mo.face_fraction === ones(SVector{4,Float64})
            @test interface_area(mo, SMALL) == 0.0
            @test interface_normal(mo, SMALL) == zero(SVector{2,Float64})    # not NaN
            @test mo.centroid ≈ [1.75, -2.875]
            # A fully open face reads back as exactly its full measure, which is what the stored
            # area used to be. `face_area_of` is `1.0 * full_face_area`, so this is exact.
            @test face_area_of(mo, 1, SMALL) === 0.25 && face_area_of(mo, 3, SMALL) === 0.5

            mi = box_moments(x -> -1.0, SVector(1.5, -3.0), SVector(2.0, -2.75))
            @test is_inside(mi)
            @test mi.volume_fraction === 0.0
            @test all(iszero, mi.face_fraction)
            @test !is_cell_open(mi)
            @test interface_area(mi, SMALL) == 0.0
        end

        @testset "sign convention: the measured region is phi >= 0, outside the body" begin
            # A circle of radius 0.3 centred in the unit cell. The measured region is the part
            # *outside* it, so the volume fraction exceeds 1/2.
            g = SDFCircle(SVector(0.5, 0.5), 0.3)
            mc = box_moments(g, SVector(0.0, 0.0), SVector(1.0, 1.0))
            # Every corner is outside a radius-0.3 circle, so nodally this cell reads as uncut: the
            # body is smaller than the cell. That is the resolution limit of any nodal scheme --
            # assert it explicitly so the limitation is recorded rather than discovered.
            @test is_outside(mc)

            # Resolve it and the volume fraction is the complement of the disc.
            g2 = SDFCircle(SVector(0.0, 0.0), 0.6)
            m4 = box_moments(g2, SVector(0.0, 0.0), SVector(1.0, 1.0))
            @test is_cut(m4)
            @test m4.volume_fraction > 0.5
            @test interface_normal(m4, UNIT)[1] > 0 && interface_normal(m4, UNIT)[2] > 0   # out of the body
        end

        @testset "the complement comes from negating the field" begin
            # The documented route to inside-moments, and the reason no separate API exists for
            # them: the construction is a pure function of the corner values.
            out = box_moments(x -> x[2] - 0.3, SVector(0.0, 0.0), SVector(1.0, 1.0))
            inn = box_moments(x -> -(x[2] - 0.3), SVector(0.0, 0.0), SVector(1.0, 1.0))
            @test out.volume_fraction + inn.volume_fraction ≈ 1.0
            @test inn.volume_fraction ≈ 0.3
            # The two interfaces are the same facet with opposite normals.
            @test interface_area(inn, UNIT) ≈ interface_area(out, UNIT)
            @test interface_normal(inn, UNIT) ≈ -interface_normal(out, UNIT)
        end

        @testset "degenerate corners do not produce NaN" begin
            # phi exactly zero on a whole edge.
            md = box_moments(x -> x[2], SVector(0.0, 0.0), SVector(1.0, 1.0))
            @test all(isfinite, md.face_fraction)
            @test isfinite(md.volume_fraction)
            @test all(isfinite, md.centroid)
            @test all(isfinite, interface_normal(md, UNIT))
            @test md.volume_fraction ≈ 1.0                     # phi >= 0 is the outside
        end

        @testset "multi-valued cuts are flagged, not silently resolved" begin
            # The saddle: the outside reaches two opposite corners. Two interface segments, and
            # marching squares cannot tell which way they connect.
            ma = box_moments(x -> (x[1] - 0.5) * (x[2] - 0.5),
                                  SVector(0.0, 0.0), SVector(1.0, 1.0))
            @test is_ambiguous(ma)
            @test is_cut(ma)
            # Conservation still holds on an ambiguous cell -- the interface term is the vector sum
            # of both segments -- which is why the failure mode is a wrong surface flux rather than
            # a leak.
            @test closure_residual(ma, SVector(1.0, 1.0)) == zero(SVector{2,Float64})
            # Exactness is not an artifact of a unit cell: the open-face areas are re-formed from
            # fractions, so an anisotropic cell is the case where a mis-scaled side would show.
            ma2 = box_moments(x -> (x[1] - 0.5) * (x[2] - 2.0),
                                   SVector(0.0, 0.0), SVector(1.0, 4.0))
            @test closure_residual(ma2, SVector(1.0, 4.0)) == zero(SVector{2,Float64})
            # An ordinary cut is not flagged.
            @test !is_ambiguous(box_moments(x -> x[2] - 0.3,
                                                 SVector(0.0, 0.0), SVector(1.0, 1.0)))
        end
    end

    # =====================================================================
    @testset "fractions and centroids vs the analytic circle" begin
        # Global moments of the circle, each with a closed-form value:
        #   enclosed area          pi r^2
        #   outside first moment   (domain moment) - pi r^2 * c
        #   interface length       2 pi r
        #   interface first moment 2 pi r * c
        # The centroids are here on purpose -- they are the part that is easy to omit, and their
        # absence shows up much later as a loss of order near the interface rather than as an
        # obvious failure here.
        function circle_errors(n)
            mesh = uniform_geometry_mesh(n)
            cg = cut_field(GEO_CIRCLE, mesh)
            area_inside = 0.0
            outside_moment = zero(SVector{2,Float64})
            domain_moment = zero(SVector{2,Float64})
            interface_length = 0.0
            interface_moment = zero(SVector{2,Float64})
            for i in eachleaf(mesh)
                m = cg[i]
                cell = leaf(mesh, i)
                V = get_elem_volume(mesh, cell)
                dcell = get_elem_size(mesh, cell)
                area_inside += V - m.volume_fraction * V
                outside_moment += (m.volume_fraction * V) * m.centroid
                domain_moment += V * get_elem_centroid(mesh, cell)
                A_int = interface_area(m, dcell)
                interface_length += A_int
                interface_moment += A_int * m.interface_centroid
            end
            A = pi * GEO_RADIUS^2
            L = 2pi * GEO_RADIUS
            return (abs(area_inside - A) / A,
                    norm(outside_moment - (domain_moment - A * GEO_CENTRE)) / (A * norm(GEO_CENTRE)),
                    abs(interface_length - L) / L,
                    norm(interface_moment - L * GEO_CENTRE) / (L * norm(GEO_CENTRE)))
        end

        resolutions = (64, 128, 256, 512)
        errs = [circle_errors(n) for n in resolutions]
        names = ("enclosed area", "outside first moment", "interface length",
                 "interface first moment")
        for k in 1:4
            e = [errs[j][k] for j in eachindex(resolutions)]
            @testset "$(names[k])" begin
                @test e[end] < 1e-4
                # Linear reconstruction of a smooth surface: second order.
                @test overall_rate(e) > 1.8
                @test overall_rate(e) < 2.2
                @test all(>(1.4), pair_rates(e))
            end
        end

        @testset "open-face apertures vs the exact chord" begin
            # Along a vertical grid line at distance d from the centre, the covered length is the
            # chord 2*sqrt(r^2 - d^2). Lines within 0.8r only: near tangency the chord itself is
            # O(sqrt(h)) and the max-over-lines statistic stops measuring the aperture.
            function chord_error(n)
                mesh = uniform_geometry_mesh(n)
                cg = cut_field(GEO_CIRCLE, mesh)
                h = 4.0 / n
                covered = zeros(n)
                for i in eachleaf(mesh)
                    covered[leaf(mesh, i).coord[1] + 1] +=
                        face_area_of(cg[i], 1, get_elem_size(mesh, leaf(mesh, i)))
                end
                worst = 0.0
                for ix in 0:(n-1)
                    dx = (-2.0 + ix * h) - GEO_CENTRE[1]
                    abs(dx) < 0.8 * GEO_RADIUS || continue
                    chord = 2 * sqrt(GEO_RADIUS^2 - dx^2)
                    worst = max(worst, abs((4.0 - covered[ix+1]) - chord))
                end
                return worst
            end
            e = [chord_error(n) for n in (64, 128, 256, 512)]
            @test e[end] < 1e-4
            @test overall_rate(e) > 1.8
            @test all(>(1.4), pair_rates(e))
        end
    end

    # =====================================================================
    @testset "watertightness -- neighbours agree bitwise" begin
        # The claim the whole construction rests on: two cells sharing a face compute the same
        # aperture, not merely close ones. `===`, not `isapprox` -- it holds exactly or the
        # reasoning in `marching_squares.jl` is wrong.
        mesh = banded_geometry_mesh()
        faces = FaceList(mesh)
        cg = cut_field(GEO_CIRCLE, mesh)
        mom = cg
        @test n_interfaces(faces) > 1000

        area_mismatches = 0
        centroid_mismatches = 0
        for f in faces.interfaces
            # On the FRACTION, which is what the construction actually produces and stores. The two
            # cells of an interface are at the same level, so their full face measures are equal
            # and the area agrees bitwise iff the fraction does -- but the fraction is the claim
            # `marching_squares.jl` makes, and it holds with no cell size in it at all.
            ax = Int(f.axis)
            mom[Int(f.left)].face_fraction[2ax] === mom[Int(f.right)].face_fraction[2ax-1] ||
                (area_mismatches += 1)
            l, r = Int(f.left), Int(f.right)
            face_centroid(mom[l], 2ax, mesh, l) === face_centroid(mom[r], 2ax-1, mesh, r) ||
                (centroid_mismatches += 1)
        end
        @test area_mismatches == 0
        @test centroid_mismatches == 0

        @testset "face centroids agree on an off-lattice grid, uncut neighbours included" begin
            # The banded mesh above is dyadic, where `centre +- h/2` happens to round exactly, so it
            # cannot catch a face centre built by arithmetic on the cell centre. This grid can: only
            # a centre taken from the face's own lattice corners agrees bitwise with the
            # neighbour's. It matters most on a fully open face of an outside cell, whose centroid
            # carries full quadrature weight.
            grid = CartesianGrid(mincorner=(-1.3, 0.7), maxcorner=(2.1, 3.9), n=(17, 23))
            field(x) = sin(3x[1]) * cos(2x[2]) + 0.1
            m = cut_field(field, grid)
            mismatches = 0
            outside_by_cut = 0
            for ci in CartesianIndices(m), ax in 1:2
                cj = ci + CartesianIndex(ntuple(a -> a == ax ? 1 : 0, 2))
                checkbounds(Bool, m, cj) || continue
                face_centroid(m[ci], 2ax, cell_nodes(grid, ci)) ===
                    face_centroid(m[cj], 2ax-1, cell_nodes(grid, cj)) || (mismatches += 1)
                ((is_outside(m[ci]) && is_cut(m[cj])) || (is_cut(m[ci]) && is_outside(m[cj]))) &&
                    (outside_by_cut += 1)
            end
            @test outside_by_cut > 50       # the case the test exists for is actually exercised
            @test mismatches == 0
        end

        @testset "a corner shared across a level jump is one sample" begin
            # Leaf corners come off the integer lattice, so a coarse leaf and a fine one meeting at a
            # corner sample `phi` at the *same float* and the cache stores one number for it, where
            # building corners as `lo + h` would not: `2c * (h/2)` reproducing `c * h` is what makes
            # it hold. Keyed on the corner's integer coordinates at the finest level, so grouping the
            # corners involves no floating point at all.
            cache = ms_leaf_cache(GEO_CIRCLE, mesh)
            L = maximum(i -> leaf_level(mesh, i), eachleaf(mesh))
            seen = Dict{NTuple{2,Int},Tuple{Float64,Set{Int}}}()
            mismatches = 0
            for i in eachleaf(mesh)
                c = leaf(mesh, i)
                s = 2^(L - c.level)
                for k in 1:4
                    key = ntuple(a -> (c.coord[a] + MS_NODE_BITS[k][a]) * s, 2)
                    v = cache.phi[i][k]
                    if haskey(seen, key)
                        first(seen[key]) === v || (mismatches += 1)
                        push!(last(seen[key]), c.level)
                    else
                        seen[key] = (v, Set(c.level))
                    end
                end
            end
            @test mismatches == 0
            # ...and the level jump is genuinely exercised.
            @test count(p -> length(last(p)) > 1, values(seen)) > 100
        end
    end

    # =====================================================================
    @testset "the reconstructed contour" begin
        # The zero contour as segments. `CutCellData` keeps only the interface's centroid, so these
        # come from `generate_mesh(cache, domain)` -- the contour of the cache's own corner values,
        # sharing the boundary walk with the moments rather than reconstructing a second time.
        @testset "per cell" begin
            # One unit cell, so the contour is that cell's segments.
            unit = CartesianGrid(SVector(0.0, 0.0), (1, 1), SVector(1.0, 1.0))
            contour1(f) = generate_mesh(update_cache!(allocate_cache(unit, MS), f, unit), unit; warn=false)
            # The half-plane again: the body is y < 0.3.
            (p, q), = contour_segments(contour1(x -> x[2] - 0.3))
            @test sort([p[2], q[2]]) ≈ [0.3, 0.3]
            @test sort([p[1], q[1]]) ≈ [0.0, 1.0]
            # Wound so the right-hand normal points out of the body: +y.
            v = q - p
            @test SVector(v[2], -v[1]) ≈ [0.0, 1.0]

            # Uncut cells contribute nothing.
            @test isempty(contour1(x -> 1.0).elements)
            @test isempty(contour1(x -> -1.0).elements)
            # The saddle contributes two, which is what `ambiguous` reports.
            @test length(contour1(x -> 4 * (x[1] - 0.5) * (x[2] - 0.5)).elements) == 2
        end

        mesh = banded_geometry_mesh()
        cg = cut_field(GEO_CIRCLE, mesh)
        contour = generate_mesh(ms_leaf_cache(GEO_CIRCLE, mesh), mesh)
        segments = contour_segments(contour)

        # One segment per cut cell, since nothing here is ambiguous.
        @test length(segments) == count(is_cut, cg)
        @test length(segments) > 100

        @testset "it agrees with the moments it was computed alongside" begin
            # Same walk, so the segment midpoints and the stored interface centroids must coincide,
            # and the lengths must sum to the interface areas. This is what makes the contour
            # a picture of the moments rather than a second opinion about them.
            cutidx = [i for i in eachleaf(mesh) if is_cut(cg[i])]
            @test all(zip(cutidx, segments)) do (i, (p, q))
                0.5 * (p + q) ≈ cg[i].interface_centroid
            end
            @test sum(norm(q - p) for (p, q) in segments) ≈
                  sum(interface_area(cg[i], get_elem_size(mesh, leaf(mesh, i))) for i in cutidx) rtol = 1e-12
        end

        @testset "the contour is closed, and closed BITWISE" begin
            # Every endpoint sits on a cell face and is therefore shared with the neighbouring
            # cell's segment. If the two cells agree to the last bit, each distinct endpoint is
            # reached by exactly two segments and the contour is one closed loop with no merge
            # tolerance anywhere. `Dict` lookup is exact equality, so this cannot pass by rounding.
            counts = Dict{SVector{2,Float64},Int}()
            for (p, q) in segments
                counts[p] = get(counts, p, 0) + 1
                counts[q] = get(counts, q, 0) + 1
            end
            @test all(==(2), values(counts))
            # A closed loop of N segments has exactly N distinct vertices.
            @test length(counts) == length(segments)
        end

        @testset "it lies on the surface, and has the right length" begin
            # Every vertex is an edge crossing found by linear interpolation, so phi there is
            # O(h^2) rather than zero.
            h = 4.0 / (8 * 2^4)
            @test maximum(max(abs(first(get_sdf(GEO_CIRCLE, p))),
                              abs(first(get_sdf(GEO_CIRCLE, q)))) for (p, q) in segments) < 0.01h
            total = sum(norm(q - p) for (p, q) in segments)
            @test total ≈ 2pi * GEO_RADIUS rtol = 1e-3
        end
    end

    # =====================================================================
    @testset "per-cell geometric conservation" begin
        for (name, mesh) in ("uniform" => uniform_geometry_mesh(64),
                             "refined" => banded_geometry_mesh())
            cg = cut_field(GEO_CIRCLE, mesh)
            @testset "$name" begin
                # Exactly zero. The interface's vector area is defined as minus the sum of the open
                # faces', so this holds to the last bit at every level -- see `cut_cell.jl`'s header
                # for why that is the honest construction rather than a circular one.
                @test closes_exactly(cg, mesh)
                @test count(is_cut, cg) > 50
            end
        end

        @testset "the interface normal agrees with the facet's own -- the non-circular check" begin
            # Recompute the interface segment from the cache's own corner values with an independent
            # implementation and compare against the closure-derived normal and area. This is what
            # stops the closure test being a tautology.
            grid = CartesianGrid(mincorner=(-2.0, -2.0), maxcorner=(2.0, 2.0), n=(128, 128))
            cache = update_cache!(allocate_cache(grid, MS), GEO_CIRCLE, grid)
            h = grid.d
            worst_angle = 0.0
            worst_area = 0.0
            checked = 0
            worst_sense = Inf
            for ci in CartesianIndices(Tuple(grid.n))
                m = cache.cells[ci]
                (is_cut(m) && !is_ambiguous(m)) || continue
                corners = [ci + CartesianIndex(b) for b in MS_NODE_BITS]
                nodes = [get_node(grid, c) for c in corners]
                phi = [cache.phi[c] for c in corners]
                pts = SVector{2,Float64}[]
                for k in 1:4
                    j = k == 4 ? 1 : k + 1
                    if (phi[k] >= 0) != (phi[j] >= 0)
                        t = phi[k] / (phi[k] - phi[j])
                        push!(pts, nodes[k] + t * (nodes[j] - nodes[k]))
                    end
                end
                length(pts) == 2 || continue
                v = pts[2] - pts[1]
                nA = SVector(v[2], -v[1])
                len = norm(nA)
                len > 1e-13 || continue
                worst_angle = max(worst_angle, abs(1 - abs(dot(nA / len, interface_normal(m, h)))))
                worst_area = max(worst_area, abs(len - interface_area(m, h)) / len)
                # ...and it points OUT of the body. Independent of the walk's own pair ordering,
                # which is why this is a separate check rather than dropping the `abs` above: the
                # body is a disc, so outward-from-body at the facet is away from its centre.
                worst_sense = min(worst_sense,
                                  dot(interface_normal(m, h), m.interface_centroid - GEO_CENTRE))
                checked += 1
            end
            @test checked > 100
            @test worst_angle < 1e-12
            @test worst_area < 1e-11
            @test worst_sense > 0            # every facet normal points out of the body
        end

        @testset "the closure is the sum of the face vector areas" begin
            # The identity `interface_normal_area == sum_k face_vector_area` stated directly: the
            # per-axis subtraction and the signed sum over faces have to agree bitwise.
            h = SVector(1.0, 1.0)
            m = box_moments(x -> x[1] + 2x[2] - 0.9, SVector(0.0, 0.0), h)
            @test is_cut(m)
            @test sum(face_vector_area(m, d, h) for d in 1:4) == interface_normal_area(m, h)
        end
    end

    # =====================================================================
    @testset "refinement consistency" begin
        # Refining a cut cell and summing its children's outside volumes must reproduce the
        # parent's, to the order of the reconstruction. It cannot be exact: parent and children
        # interpolate `phi` between different nodes, which is precisely why moments are recomputed
        # rather than inherited.
        function parent_child_gap(n)
            coarse = uniform_geometry_mesh(n)
            fine = uniform_geometry_mesh(2n)
            vp = sum(m.volume_fraction * get_elem_volume(coarse, leaf(coarse, i))
                     for (i, m) in enumerate(cut_field(GEO_CIRCLE, coarse)))
            vc = sum(m.volume_fraction * get_elem_volume(fine, leaf(fine, i))
                     for (i, m) in enumerate(cut_field(GEO_CIRCLE, fine)))
            return abs(vp - vc)
        end
        gaps = [parent_child_gap(n) for n in (32, 64, 128, 256)]
        @test overall_rate(gaps) > 1.8
        @test all(>(1.4), pair_rates(gaps))
        @test gaps[end] < 5e-3

        @testset "children are recomputed, never inherited" begin
            # The structural half. Refine one cut cell and check its children carry genuinely
            # different geometry -- a child fully inside the body must have volume exactly zero,
            # which no subdivision of a parent fraction would produce.
            mesh = uniform_geometry_mesh(32)
            cg = cut_field(GEO_CIRCLE, mesh)
            i = findfirst(m -> is_cut(m) && m.volume_fraction < 0.35, cg)
            @test i !== nothing
            parent_frac = cg[i].volume_fraction
            parent_vol = cg[i].volume_fraction * get_elem_volume(mesh, leaf(mesh, i))
            pcell = leaf(mesh, i)

            flags = falses(nleaves(mesh))
            flags[i] = true
            refine!(mesh, flags)
            # Recomputed, not inherited -- a tree cache is sized for the leaves as they are, so the
            # refined tree gets a fresh one and there is nothing that *could* be inherited, which is
            # the point the next assertions make concrete.
            cg = cut_field(GEO_CIRCLE, mesh)

            kids = [j for j in eachleaf(mesh)
                    if leaf_level(mesh, j) > 0 && parent(leaf(mesh, j)) == pcell]
            @test length(kids) == 4
            child_fracs = [cg[j].volume_fraction for j in kids]
            @test !all(≈(parent_frac), child_fracs)          # not a copied fraction
            @test any(iszero, child_fracs)                    # at least one fully inside
            @test sum(cg[j].volume_fraction * get_elem_volume(mesh, leaf(mesh, j)) for j in kids) ≈ parent_vol rtol = 0.2
            # and every child still closes
            @test all(j -> closure_residual(cg[j], get_elem_size(mesh, leaf(mesh, j))) == zero(SVector{2,Float64}), kids)
        end

        @testset "cut_cell_report surfaces the smallest cell" begin
            mesh = banded_geometry_mesh()
            cg = cut_field(GEO_CIRCLE, mesh)
            r = cut_cell_report(cg)
            @test r.n == nleaves(mesh)
            @test r.cut + r.inside + r.outside == r.n
            @test r.ambiguous == 0
            # A cut cell far below the Cartesian volume is what caps an explicit consumer's step.
            @test 0 < r.min_volume_fraction < 0.05
        end
    end

    # =====================================================================
    # The routes into the construction, as distinct from the construction itself: the grid cache
    # and the tree cache. Both read lattice corners into the same per-cell construction, so where
    # the two domains coincide they agree BITWISE.
    @testset "the routes into the construction" begin
        @testset "a grid cache and a tree cache on the same cells are one construction" begin
            # A tree left at its base level has the grid's cells as its leaves. `===`, because an
            # `isapprox` here would hide exactly the drift that sharing the construction prevents
            # -- and on an off-lattice grid, where a corner built as `lo + h` would not round the
            # same way the two domains place it.
            for (base, geo) in ((CartesianGrid(mincorner=(-2.0, -2.0), maxcorner=(2.0, 2.0), n=(64, 64)), GEO_CIRCLE),
                                (CartesianGrid(SVector(-1.3, 0.7), (17, 23), SVector(0.2, 0.139)),
                                 x -> sin(3x[1]) * cos(2x[2]) + 0.1))
                tree = AdaptiveMesh(base; max_level=1, initial_level=0)
                on_tree = cut_field(geo, tree)
                on_grid = cut_field(geo, base)
                @test length(on_tree) == prod(base.n)
                @test count(is_cut, on_tree) > 50
                @test all(i -> on_tree[i] === on_grid[CartesianIndex(leaf(tree, i).coord .+ 1)],
                          eachleaf(tree))
                @test closes_exactly(on_tree, tree)
            end
        end

        @testset "a cache's contour is asked for on its own domain" begin
            grid = CartesianIsoGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), sz=0.1)
            @test_throws DimensionMismatch generate_mesh(allocate_cache(grid, MS),
                                                         CartesianGrid(grid.x0, Tuple(grid.n) .+ 1, grid.d))
            # ...and on a tree the cache is one entry per leaf, checked against the tree it is asked
            # on; a grid's nodal block has no meaning there, since a tree has no single array to index.
            tree = banded_geometry_mesh()
            @test_throws DimensionMismatch generate_mesh(ms_leaf_cache(GEO_CIRCLE, tree),
                                                         banded_geometry_mesh(level=2))
            @test_throws MethodError generate_mesh(update_cache!(allocate_cache(grid, MS), GEO_CIRCLE, grid), tree)
        end

        @testset "a field that is no geometry at all" begin
            # Any callable `x -> phi` is sampled the same way, as a level set carried by someone
            # else's solver would be.
            grid = CartesianIsoGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), sz=0.1)
            field = cut_field(x -> x[2] - 0.25, grid)
            @test all(m -> 0 <= m.volume_fraction <= 1, field)
            @test count(is_cut, field) == grid.n[1]
        end

        @testset "3D says so rather than returning a slice" begin
            base3 = CartesianGrid(mincorner=(-2.0, -2.0, -2.0), maxcorner=(2.0, 2.0, 2.0), n=(4, 4, 4))
            m3 = AdaptiveMesh(base3; max_level=1, initial_level=0)
            # A marching-squares cache, and so its contour, is 2D by type, on a grid or a tree.
            @test_throws MethodError allocate_cache(m3, MS)
            @test_throws MethodError generate_mesh(ms_leaf_cache(GEO_CIRCLE, banded_geometry_mesh(level=1)), m3)
        end

        @testset "generate_mesh and the moments see the same corners" begin
            # The contour a plot draws and the apertures a flux integrates come from one
            # reconstruction: both are read off one cache's nodal field, so the contour has exactly
            # one segment per cut cell.
            grid = CartesianIsoGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), sz=0.1)
            cache = update_cache!(allocate_cache(grid, MS), GEO_CIRCLE, grid)
            contour = generate_mesh(cache, grid)
            field = cache.cells
            # One segment per cut cell (no ambiguous cells on a circle at this resolution).
            @test count(is_cut, field) == length(contour.elements)
            @test all(m -> !m.ambiguous, field)
            # Every contour vertex lies on the circle, to the interpolation's own order.
            @test all(x -> abs(first(get_sdf(GEO_CIRCLE, x))) < 2e-3, contour.nodes.coord)
        end
    end

    # =====================================================================

    @testset "the shared cut-cell accessor layer" begin
        # `face_area_of`/`is_cell_open` read one `CutCellData`, whichever method or cache it came
        # from. What is asserted here is that they report the cell's own stored moments.
        grid = CartesianIsoGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), sz=0.1)
        geo = SDFCircle(center=SVector(0.0, 0.0), radius=1.0)
        idx = CartesianIndices(Tuple(grid.n))
        ms = cut_field(geo, grid)

        @test all(idx) do ci
            all(1:4) do dir
                full = grid.d[3 - CartesianMeshes.direction_axis(dir)]
                face_area_of(ms[ci], dir, grid.d) === ms[ci].face_fraction[dir] * full
            end
        end

        # A fraction is a fraction, and a raw measure is a length.
        @test all(0 .<= (ms[ci].face_fraction[d] for ci in idx, d in 1:4) .<= 1)
        @test all(0 .<= (face_area_of(ms[ci], d, grid.d) for ci in idx, d in 1:4) .<= grid.d[1])

        # `is_cell_open` is exactly "some face is open", and is deliberately NOT `kind`: an
        # entirely-inside cell has no open face, but the converse does not follow.
        @test all(ci -> is_cell_open(ms[ci]) == !all(iszero, ms[ci].face_fraction), idx)
        @test !any(ci -> is_cell_open(ms[ci]), filter(ci -> is_inside(ms[ci]), collect(idx)))

        # The broadcastable form a bulk consumer slices with.
        @test face_area_of.(view(ms, 4:9, 4:9), 2, Ref(grid.d)) ==
              [face_area_of(ms[ci], 2, grid.d) for ci in CartesianIndices((4:9, 4:9))]

        # The fraction is STORED, normalized per face, so it is right on a stretched grid --
        # where PLIC's single-`dx` divisor would not be. This grid is deliberately anisotropic.
        aniso = CartesianGrid(SVector(-1.5, -1.5), (30, 15), SVector(0.1, 0.2))
        msa = cut_field(geo, aniso)
        @test all(CartesianIndices(Tuple(aniso.n))) do ci
            # a u-face's full measure is the spacing along y, and vice versa
            all(1:4) do dir
                full = dir <= 2 ? aniso.d[2] : aniso.d[1]
                isapprox(msa[ci].face_fraction[dir], face_area_of(msa[ci], dir, aniso.d) / full;
                         atol=1e-14)
            end
        end
    end

    @testset "kernel discipline" begin
        # A cell is isbits, so the cache's `cells` live in plain device arrays and pass into a
        # consumer's kernel by value; and an update allocates only launch overhead, nothing per
        # leaf. It is what the GPU path below needs.
        @test isbitstype(CutCellData{2,Float64,4,1})
        mesh = uniform_geometry_mesh(256)
        cache = ms_leaf_cache(GEO_CIRCLE, mesh)
        @test isbits(cache.cells[1])
        update_cache!(cache, GEO_CIRCLE, mesh)                    # warm up
        # 65536 leaves: a byte per leaf would be 64 KiB, and one `CutCellData` each a megabyte.
        @test (@allocated update_cache!(cache, GEO_CIRCLE, mesh)) < 64 * 1024

        # Float32 all the way through, so precision stays an explicit choice rather than something
        # `adapt` does to one operand.
        mesh32 = CartesianMeshes.adapt_type(Float32, uniform_geometry_mesh(32))
        geo32 = SDFLibrary.adapt_type(Float32, GEO_CIRCLE)
        c32 = cut_field(geo32, mesh32)
        @test eltype(c32) === CutCellData{2,Float32,4,1}
        @test count(is_cut, c32) > 50
        @test closes_exactly(c32, mesh32)
    end

    # =====================================================================
    if HAS_GPU
        @testset "GPU parity" begin
            mesh = banded_geometry_mesh()
            host = cut_field(GEO_CIRCLE, mesh)

            # The tree adapted to the device, and a cache allocated on the tree's backend.
            dmesh = adapt(CuArray, mesh)
            dcache = update_cache!(allocate_cache(dmesh, MS), adapt(CuArray, GEO_CIRCLE), dmesh)
            @test dcache.cells.volume_fraction isa CuArray
            dev = Adapt.adapt(Array, dcache.cells)

            # Classification is a sign test on `phi`, so it must agree exactly even though the
            # values behind it need not.
            @test all(i -> host[i].kind == dev[i].kind, eachindex(host))
            @test all(i -> host[i].ambiguous == dev[i].ambiguous, eachindex(host))

            # Values agree to backend arithmetic, NOT bitwise. Two independent mechanisms are in
            # play: this package's primitives are `@fastmath`, and a device kernel contracts
            # `a*b + c` into an FMA where the host does not. Asking for bitwise agreement here
            # would be asking for something that does not exist.
            @test maximum(i -> abs(host[i].volume_fraction - dev[i].volume_fraction), eachindex(host)) < 1e-13
            @test maximum(i -> maximum(abs, host[i].face_fraction - dev[i].face_fraction),
                          eachindex(host)) < 1e-13
            @test maximum(i -> norm(host[i].centroid - dev[i].centroid), eachindex(host)) < 1e-14

            # What *does* survive exactly is cancellation within one backend -- so the closure
            # still demands exact zeros of the device's own fractions.
            @test closes_exactly(dev, mesh)
        end
    end
end
