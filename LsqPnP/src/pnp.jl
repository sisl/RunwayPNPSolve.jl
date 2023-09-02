using Rotations
using ComponentArrays
using CoordinateTransformations, Geodesy, GeodesyXYZExt
using LinearAlgebra: dot, norm, I, normalize
using Optim
using ReTest
using Tau
using Roots
using LeastSquaresOptim
using LsqFit
using Unitful: Length
using StaticArrays: StaticVector, MVector, SVector, MArray, FieldVector
using Unitful: ustrip
import LsqFit.DiffResults: DiffResult, MutableDiffResult
import LsqFit.ForwardDiff: Dual
import StatsBase: mean
import Base: Fix1

### Solution using Levenberg-Marquardt algo from LsqFit.jl
# I've tried before to use LeastSquaresOptim but got bad results.
# LsqFit expects vector valued in- and outputs, so we have to flatten/unflatten the vector of points.
# Also, we overload the Optim.minimizer/converged functions for easier back-and-forth between Optim.jl and LsqFit.jl.
flatten_points(pts::Vector{<:StaticVector}) = stack(pts, dims=1)[:]
flatten_points(pts::Vector{<:FieldVector{2, T}}) where {T} = begin
    data = stack(pts, dims=1)
    ComponentVector((; x=data[:, 1],
                       y=data[:, 2]))
end
flatten_points(pts::Vector{<:FieldVector{3, T}}) where {T} = begin
    data = stack(pts, dims=1)
    ComponentVector((; x=data[:, 1],
                       y=data[:, 2],
                       z=data[:, 3]))
end
unflatten_points(P::Type{<:StaticVector}, pts::AbstractVector{<:Number}) = P.(eachrow(reshape(pts, :, length(P))))

# We need to overload DiffResults to support mutable static arrays, see https://github.com/JuliaDiff/DiffResults.jl/issues/25
DiffResult(value::MArray, derivs::Tuple{Vararg{MArray}}) = MutableDiffResult(value, derivs)
DiffResult(value::Union{Number, AbstractArray}, derivs::Tuple{Vararg{MVector}}) = MutableDiffResult(value, derivs)
function pnp(world_pts::Vector{XYZ{Meters}},
             pixel_locations::Vector{ImgProj{Pixels}},
             cam_rotation::Rotation{3, Float64};
             initial_guess::XYZ{Meters} = XYZ(-100.0m, 0m, 30m),
             components=[:x, :y],
             )
    # early exit if no points given
    (length(world_pts) == 0) && return PNP3Sol((pos=initial_guess, ))

    # strip units for Optim.jl package. See https://github.com/JuliaNLSolvers/Optim.jl/issues/695.
    world_pts = map(p->ustrip.(m, p), world_pts) |> collect
    N_p = length(world_pts)
    pixel_locations = map(p->ustrip.(pxl, p), pixel_locations) |> collect
    initial_guess = ustrip.(m, initial_guess)


    function model(world_pts_flat, pos::StaticVector{3, <:Real})
        proj = make_projection_fn(AffineMap(cam_rotation, XYZ(pos*1m)))
        unflatten_to_xyz(pts) = unflatten_points(XYZ, pts*1m)
        f = flatten_points ∘ Fix1(broadcast, proj) ∘ unflatten_to_xyz

        res = f(world_pts_flat) .|> p′->ustrip.(pxl, p′)
        res[components] |> collect
    end

    fit = curve_fit(model,
                    flatten_points(world_pts),
                    flatten_points(pixel_locations)[components] |> collect,
                    MVector(initial_guess);
                    autodiff=:forward, store_trace=true)
    return PNP3Sol((pos=XYZ(fit.param)*1m,))
end


"Hough transform."
function compute_rho_theta(p1::StaticVector{2, T}, p2::StaticVector{2, T}, p3::StaticVector{2, T}) where T
    p4(λ) = p1 + λ*(p2-p1)
    λ = dot((p2-p1), (p3-p1)) / norm(p2-p1)^2
    @assert isapprox(dot(p2-p1, p4(λ)-p3)/oneunit(T)^2, 0.; atol=1e-4) "$(dot(p2-p1, p4(λ)-p3))"
    @debug λ, p4(λ)
    ρ = norm(p4(λ) - p3)

    vec1 = Point2d(1, 0)
    vec2 = normalize(p4(λ) - p3)
    y = vec1 - vec2
    x = vec1 + vec2
    θ = 2*atan(norm(y), norm(x)) * -sign(vec2[2])
    return ρ, θ
end
@testset "compute_rho_theta" begin
    ρ, θ = compute_rho_theta(Point2d(-2, 0), Point2d(0, -2), Point2d(0, 0))
    @test all((ρ, θ) .≈ (√(2), 3/8*τ))

    ρ, θ = compute_rho_theta(Point2d(0, 2), Point2d(-2, 2), Point2d(0, 0))
    @test all((ρ, θ) .≈ (2, -2/8*τ))

    ρ, θ = compute_rho_theta(Point2d(0, -2), Point2d(-1, -3), Point2d(0, 0))
    @test all((ρ, θ) .≈ (sqrt(2), 1/8*τ))

    ρ, θ = compute_rho_theta(Point2d(0, 2), Point2d(-1, 3), Point2d(0, 0))
    @test all((ρ, θ) .≈ (sqrt(2), -1/8*τ))
end

function hough_transform(projected_points)  # front left, front right, back left, back right
    ppts = projected_points
    ρ_θ_lhs = compute_rho_theta(ppts[1], ppts[3], (ppts[1]+ppts[2])/2)
    ρ_θ_rhs = compute_rho_theta(ppts[2], ppts[4], (ppts[1]+ppts[2])/2)
    ρ = (; lhs=ρ_θ_lhs[1], rhs=ρ_θ_rhs[1])
    θ = (; lhs=ρ_θ_lhs[2]*1rad, rhs=ρ_θ_rhs[2]*1rad)
    ρ, θ
end

PNP3Sol = @NamedTuple begin
  pos::XYZ{Meters}
end
Optim.minimizer(sol::PNP3Sol) = sol.pos
Optim.converged(sol::PNP3Sol) = true
