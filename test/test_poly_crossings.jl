# =====================================
# Polyline clipping, P1-P2: the predicates, and (P2) the crossings built on them
#
# The predicates are the exact limit of one symbolic perturbation of the grid -- x-lines at
# `X + ε`, y-lines at `Y + ε²`. Their reference here is that perturbation made concrete: the same
# questions asked in `BigFloat` with a numerical `ε` far below any gap the test geometry has, so a
# wrong tie-break rule, and not only a wrong sign, shows up as a mismatch.

using Test
using StaticArrays
using CartesianMeshes
using Random
using CutCellMethods: PolylineLattice, _locate, _crossed_lines, _cross_interval, _cross_offset,
                      _xcross_above, _ycross_right, _line_frame, _below_on_line, _segments_touch

isdefined(@__MODULE__, :PV) || include("polyline_meshes.jl")

# The perturbation made numerical. Coordinates are O(1) doubles, so a nonzero exact gap is far above
# 2^-400; ε = 2^-600 keeps ε·slope far below that, and ε² below ε·slope for any nonzero slope.
const PC_PREC = 3000
const PC_EPS = setprecision(() -> big(2.0)^-600, PC_PREC)

# Whether a -> b meets the perturbed x-line `X + ε` above the perturbed node `(X + ε, Y + ε²)`.
function ref_xcross_above(a, b, X, Y)
    setprecision(PC_PREC) do
        ε = PC_EPS
        ax, ay, bx, by = big.((a[1], a[2], b[1], b[2]))
        y = ay + (big(X) + ε - ax) * (by - ay) / (bx - ax)
        y > big(Y) + ε^2
    end
end

# Whether a -> b meets the perturbed y-line `Y + ε²` right of the perturbed node.
function ref_ycross_right(a, b, X, Y)
    setprecision(PC_PREC) do
        ε = PC_EPS
        ax, ay, bx, by = big.((a[1], a[2], b[1], b[2]))
        x = ax + (big(Y) + ε^2 - ay) * (bx - ax) / (by - ay)
        x > big(X) + ε
    end
end

# Whether segment 1 crosses the perturbed line (x-line `line + ε` for axis 1, y-line `line + ε²` for
# axis 2) at a smaller coordinate along it than segment 2.
function ref_below(a1, b1, a2, b2, axis, line)
    setprecision(PC_PREC) do
        ε = axis == 1 ? PC_EPS : PC_EPS^2
        p = 3 - axis
        at(a, b) = big(a[p]) + (big(line) + ε - big(a[axis])) * (big(b[p]) - big(a[p])) /
                                (big(b[axis]) - big(a[axis]))
        at(a1, b1) < at(a2, b2)
    end
end

# Walk a segment through the lattice by its crossings alone: from the cell holding `a`, each step
# must find exactly one unused crossing on the current cell's boundary, and the walk must end in the
# cell holding `b`. This is the consistency the cell walk relies on -- the x- and y-line decisions
# agreeing on which cells the element passes through.
function lattice_path_ok(a, b, L::PolylineLattice)
    cr = Tuple{Int,Int,Int}[]   # (axis, line, interval)
    for axis in 1:2, k in _crossed_lines(axis == 1 ? L.xs : L.ys, a[axis], b[axis])
        push!(cr, (axis, k, _cross_interval(a, b, axis, k, L)))
    end
    i, j = _locate(L.xs, a[1]), _locate(L.ys, a[2])
    used = falses(length(cr))
    for _ in eachindex(cr)
        hits = findall(m -> !used[m] && begin
                           axis, k, iv = cr[m]
                           axis == 1 ? (iv == j && (k == i || k == i + 1)) :
                                       (iv == i && (k == j || k == j + 1))
                       end, eachindex(cr))
        length(hits) == 1 || return false
        m = only(hits)
        used[m] = true
        axis, k, _ = cr[m]
        if axis == 1
            i = k == i + 1 ? i + 1 : i - 1
        else
            j = k == j + 1 ? j + 1 : j - 1
        end
    end
    return (i, j) == (_locate(L.xs, b[1]), _locate(L.ys, b[2]))
end

@testset verbose = true "polyline clipping: predicates" begin
    rng = MersenneTwister(20260928)

    @testset "locate: a vertex on a line is below it" begin
        # x0 = d = 0.1: the lattice values are not the decimal multiples, and dividing by d puts
        # most on-line vertices in the upper cell. `_locate` must not.
        for g in (CartesianGrid(SVector(0.1, 0.1), (10, 10), SVector(0.1, 0.1)),
                  CartesianGrid(SVector(-1.3, 0.7), (37, 23), SVector(0.055, 0.0731)),
                  CartesianGrid(SVector(0.1f0, 0.1f0), (10, 10), SVector(0.1f0, 0.1f0)))
            L = PolylineLattice(g)
            for lines in (L.xs, L.ys)
                n = length(lines) - 1
                @test all(i -> _locate(lines, lines[i]) == i - 1, 1:(n + 1))
                @test all(i -> _locate(lines, nextfloat(lines[i])) == i, 1:n)
                @test all(i -> _locate(lines, prevfloat(lines[i])) == i - 1, 1:(n + 1))
                @test _locate(lines, prevfloat(lines[1])) == 0
                @test _locate(lines, nextfloat(lines[end])) == n + 1
            end
        end
        # Float32 lattice: the Float64 line values are the Float32 nodes, exactly.
        g32 = CartesianGrid(SVector(0.1f0, 0.1f0), (10, 10), SVector(0.1f0, 0.1f0))
        @test PolylineLattice(g32).xs[4] == Float64(get_node(g32, CartesianIndex(4, 1))[1])
    end

    @testset "crossed lines follow the side rule" begin
        L = PolylineLattice(CartesianGrid(SVector(0.0, 0.0), (8, 8), SVector(0.25, 0.25)))
        # endpoints exactly on lines: the lower endpoint's line is crossed, the upper's is not
        @test _crossed_lines(L.xs, 0.5, 1.0) == 3:4          # lines at 0.5, 0.75 (not 1.0)
        @test _crossed_lines(L.xs, 1.0, 0.5) == 3:4
        @test isempty(_crossed_lines(L.xs, 0.5, 0.5))        # along a line: never crosses it
        @test isempty(_crossed_lines(L.xs, 0.6, 0.7))        # no line between
        for _ in 1:2000
            u, w = 2rand(rng), 2rand(rng)
            rand(rng) < 0.3 && (u = L.xs[rand(rng, 1:9)])
            rand(rng) < 0.3 && (w = L.xs[rand(rng, 1:9)])
            @test collect(_crossed_lines(L.xs, u, w)) == [k for k in 1:9 if (u > L.xs[k]) != (w > L.xs[k])]
        end
    end

    @testset "ownership matches the perturbed arrangement" begin
        # Dyadic coordinates, so segments through a node are exactly through it and axis-aligned
        # ones exactly aligned: the cases where the tie-break alone decides.
        dy(k) = k / 16
        L = PolylineLattice(CartesianGrid(SVector(0.0, 0.0), (8, 8), SVector(0.25, 0.25)))
        nbad_x = nbad_y = ncases = 0
        for _ in 1:4000
            X, Y = L.xs[rand(rng, 3:7)], L.ys[rand(rng, 3:7)]
            kind = rand(rng, 1:4)
            if kind == 1        # exactly through the node, any direction
                p, q = dy(rand(rng, -6:6)), dy(rand(rng, -6:6))
                (p, q) == (0, 0) && continue
                s, t = rand(rng, 1:3), rand(rng, 1:3)
                a, b = PV(X - s * p, Y - s * q), PV(X + t * p, Y + t * q)
            elseif kind == 2    # starting exactly at the node
                a = PV(X, Y)
                b = a + PV(dy(rand(rng, -6:6)), dy(rand(rng, -6:6)))
            elseif kind == 3    # along the node's lines
                a = PV(X + dy(rand(rng, -6:6)), Y)
                b = PV(X + dy(rand(rng, -6:6)), Y)
                if rand(rng, Bool)      # the same, along the node's x-line
                    a, b = PV(a[2] - Y + X, a[1] - X + Y), PV(b[2] - Y + X, b[1] - X + Y)
                end
            else                # generic, near the node
                a = PV(X, Y) + 0.3 * PV(randn(rng), randn(rng))
                b = PV(X, Y) + 0.3 * PV(randn(rng), randn(rng))
            end
            a == b && continue
            ncases += 1
            if (a[1] > X) != (b[1] > X)
                nbad_x += _xcross_above(a, b, X, Y) != ref_xcross_above(a, b, X, Y)
            end
            if (a[2] > Y) != (b[2] > Y)
                nbad_y += _ycross_right(a, b, X, Y) != ref_ycross_right(a, b, X, Y)
            end
        end
        @test ncases > 3000
        @test nbad_x == 0
        @test nbad_y == 0
    end

    @testset "x- and y-line ownership agree along every element" begin
        L = PolylineLattice(CartesianGrid(SVector(-0.3, 0.2), (12, 12), SVector(0.1, 0.1)))
        nodes = [PV(L.xs[i], L.ys[j]) for i in 4:9 for j in 4:9]
        nbad = 0
        n = 0
        # Every direction through a node, exactly and a few ulps either side, and from the node.
        for N in nodes[1:6], θ in range(0, 2π; length=49)[1:48], k in (-3, 0, 3), mode in 1:2
            d = PV(cos(θ), sin(θ))
            c = PV(N[1] + k * eps(N[1]), N[2] - k * eps(N[2]))
            a, b = mode == 1 ? (c - 0.17d, c + 0.23d) : (N, N + 0.23d)
            n += 1
            nbad += !lattice_path_ok(a, b, L)
        end
        # Axis-aligned elements lying on lines, and random elements anywhere.
        for N in nodes[1:6], (u, w) in ((PV(-0.21, 0), PV(0.19, 0)), (PV(0, -0.21), PV(0, 0.19)))
            n += 2
            nbad += !lattice_path_ok(N + u, N + w, L) + !lattice_path_ok(N + w, N + u, L)
        end
        for _ in 1:3000
            a = PV(L.xs[3], L.ys[3]) + 0.6 * PV(rand(rng), rand(rng))
            b = a + 0.1 * PV(randn(rng), randn(rng))
            # the walk is only defined inside the lattice
            all(p -> 1 <= _locate(L.xs, p[1]) <= 12 && 1 <= _locate(L.ys, p[2]) <= 12, (a, b)) || continue
            n += 1
            nbad += !lattice_path_ok(a, b, L)
        end
        @test n > 3000
        @test nbad == 0
    end

    @testset "an on-line vertex gives back its own coordinate" begin
        L = PolylineLattice(CartesianGrid(SVector(0.1, 0.1), (10, 10), SVector(0.1, 0.1)))
        for _ in 1:2000
            k = rand(rng, 3:9)
            X = L.xs[k]
            v = PV(X, L.ys[3] + 0.5 * rand(rng))
            u = v + PV(0.05 + 0.1 * rand(rng), 0.2 * randn(rng))
            j = _cross_interval(u, v, 1, k, L)
            lo, hi = L.ys[j], L.ys[j + 1]
            @test _cross_offset(u, v, 1, X, lo, hi) == clamp(v[2] - lo, 0.0, hi - lo)
        end
    end

    @testset "crossing order: a thin trailing edge" begin
        # Two elements share a tip just past an x-line; their crossings are within ulps of each
        # other. Interpolated coordinates flip this order; the orientation comparator must not.
        nbad = 0
        n = 0
        for _ in 1:20000
            X = 0.5 + rand(rng)
            δ = X * 10.0^(-16 + 4rand(rng))
            v = PV(nextfloat(X + δ), 0.2 * randn(rng))
            v[1] > X || continue
            α = deg2rad(0.25 + 4.75rand(rng))           # half-angle
            φ = deg2rad(20 * randn(rng))                  # chord direction
            up = v + (0.01 + 0.2rand(rng)) * PV(-cos(φ - α), -sin(φ - α))
            lo = v + (0.01 + 0.2rand(rng)) * PV(-cos(φ + α), -sin(φ + α))
            (up[1] <= X && lo[1] <= X) || continue
            # upper surface runs into the tip, lower one out of it: up -> v -> lo
            f1 = _line_frame(up, v, 1)
            f2 = _line_frame(v, lo, 1)
            n += 1
            nbad += _below_on_line(f1..., f2...) != ref_below(up, v, v, lo, 1, X)
            nbad += _below_on_line(f2..., f1...) != ref_below(v, lo, up, v, 1, X)
        end
        @test n > 15000
        @test nbad == 0
    end

    @testset "crossing order: random non-touching elements, both line families" begin
        nbad = 0
        n = 0
        for _ in 1:20000
            axis = rand(rng, 1:2)
            line = rand(rng)
            p = 3 - axis
            mk() = begin
                u1, u2 = line - rand(rng)^3, line + max(rand(rng)^3, eps(line))
                rand(rng) < 0.1 && (u1 = line)        # an endpoint on the line: below it
                v1 = rand(rng); v2 = v1 + 0.05 * randn(rng)
                a = axis == 1 ? PV(u1, v1) : PV(v1, u1)
                b = axis == 1 ? PV(u2, v2) : PV(v2, u2)
                rand(rng, Bool) ? (a, b) : (b, a)
            end
            a1, b1 = mk()
            a2, b2 = mk()
            # close, nearly parallel pairs as well
            if rand(rng) < 0.3
                off = (axis == 1 ? PV(0, 1) : PV(1, 0)) * 1e-9 * randn(rng)
                a2, b2 = a1 + off, b1 + 1.0001off
            end
            crosses(a, b) = (a[axis] > line) != (b[axis] > line)
            (crosses(a1, b1) && crosses(a2, b2)) || continue
            _segments_touch(a1, b1, a2, b2) && continue
            n += 1
            nbad += _below_on_line(_line_frame(a1, b1, axis)..., _line_frame(a2, b2, axis)...) !=
                    ref_below(a1, b1, a2, b2, axis, line)
        end
        @test n > 15000
        @test nbad == 0
    end
end
