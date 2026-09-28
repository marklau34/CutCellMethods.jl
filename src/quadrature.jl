# =====================================
# Quadrature rules over cut regions
#
# A volume fraction plus a centroid *is* a one-point quadrature rule over the outside part of a
# cell; a face-area fraction plus an open-face centroid *is* a one-point rule over the open part of
# that face. Naming them that way is what lets one interface serve a first-order consumer and a
# higher-order one alike, because raising the order changes `N` and nothing else at the call site.
#
# The rules are fixed-size (`N` in the type), isbits and allocation-free, so they pass into a kernel
# by value. Everything this package constructs is `N = 1`.

"""
    QuadratureRule{N,D,T}

An `N`-point rule over a `D`-dimensional region: points and weights, the weights summing to the
region's measure, so that `integrate(rule, one)` is the measure itself -- the outside volume for a
volume rule, the open area for a face rule.

A one-point rule is the centroid rule: the point is the region's centroid, the weight its measure.
It is exact for any linear integrand, which is what a second-order reconstruction over these cells
needs.
"""
struct QuadratureRule{N,D,T}
    points::SVector{N,SVector{D,T}}
    weights::SVector{N,T}
end

"""
    QuadratureRule(point, weight)

The one-point rule at `point` with weight `weight`.
"""
@inline QuadratureRule(point::SVector{D,T}, weight::T) where {D,T} =
    QuadratureRule{1,D,T}(SVector{1,SVector{D,T}}(point), SVector{1,T}(weight))

"""
    SurfaceQuadratureRule{N,D,T}

A [`QuadratureRule`](@ref) carrying an outward unit normal at every point. Separate from
`QuadratureRule` because a surface integrand needs the normal and a volume integrand has none, and
because the normal genuinely varies across a higher-order rule over a curved facet.

For the interface rule the normal points **out of the body** -- the field's own sense, and the
opposite of the open-face normals, which point out of the outside region. The closure identity is
therefore

    sum_k A_k n_k  -  A_int n_int  =  0

so a consumer integrating a flux over the outside region negates the interface normal at the point
of use. See `cut_cell.jl`'s header.
"""
struct SurfaceQuadratureRule{N,D,T}
    points::SVector{N,SVector{D,T}}
    weights::SVector{N,T}
    normals::SVector{N,SVector{D,T}}
end

@inline SurfaceQuadratureRule(point::SVector{D,T}, weight::T, normal::SVector{D,T}) where {D,T} =
    SurfaceQuadratureRule{1,D,T}(SVector{1,SVector{D,T}}(point),
                                 SVector{1,T}(weight),
                                 SVector{1,SVector{D,T}}(normal))

const AnyQuadratureRule{N,D,T} = Union{QuadratureRule{N,D,T},SurfaceQuadratureRule{N,D,T}}

@inline Base.length(::AnyQuadratureRule{N}) where {N} = N
@inline Base.ndims(::AnyQuadratureRule{N,D}) where {N,D} = D
@inline Base.eltype(::AnyQuadratureRule{N,D,T}) where {N,D,T} = T

"""
    measure(rule)

The measure of the region the rule covers: the sum of its weights. Outside volume for a volume
rule, open area for a face rule, interface area for an interface rule.
"""
@inline measure(r::AnyQuadratureRule) = sum(r.weights)

"""
    integrate(rule, f)

`sum(w_i * f(x_i))` over the rule's points. For a [`SurfaceQuadratureRule`](@ref), `f` is called as
`f(x_i, n_i)` so a surface integrand can see the normal it needs.

A method on `CartesianMeshes.integrate` rather than a new function of the same name: that one
integrates a field over a mesh, this one a function over a region, and two separately exported
`integrate`s would collide at any call site with both packages in scope.
"""
@inline function CartesianMeshes.integrate(r::QuadratureRule{N}, f) where {N}
    return sum(ntuple(i -> r.weights[i] * f(r.points[i]), Val(N)))
end

@inline function CartesianMeshes.integrate(r::SurfaceQuadratureRule{N}, f) where {N}
    return sum(ntuple(i -> r.weights[i] * f(r.points[i], r.normals[i]), Val(N)))
end

"""
    quadrature_points(rule)
    quadrature_weights(rule)
    quadrature_normals(rule)

The rule's raw arrays, for a caller that wants to drive the loop itself rather than hand over an
integrand.
"""
@inline quadrature_points(r::AnyQuadratureRule) = r.points
@inline quadrature_weights(r::AnyQuadratureRule) = r.weights
@inline quadrature_normals(r::SurfaceQuadratureRule) = r.normals

function Base.show(io::IO, r::QuadratureRule{N,D,T}) where {N,D,T}
    print(io, "QuadratureRule{", N, ",", D, ",", T, "}(measure = ", measure(r), ")")
end

function Base.show(io::IO, r::SurfaceQuadratureRule{N,D,T}) where {N,D,T}
    print(io, "SurfaceQuadratureRule{", N, ",", D, ",", T, "}(measure = ", measure(r), ")")
end
