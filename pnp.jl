using Rotations
using CoordinateTransformations, Geodesy
using LinearAlgebra: dot, norm, I
using Optim
using ReTest
using Tau
using Roots
using LeastSquaresOptim
using Unitful: Length
using StaticArrays: StaticVector, MVector
include("typedefs.jl")

function build_pnp_objective(
             world_pts, pixel_locations, cam_rotation;
             rhos=nothing,
             thetas=nothing,
             feature_mask=[1;1;1],
    )

    f(C_t::StaticVector{3, Float64})::Float64 = let
        projected_points = project_points(AffineMap(cam_rotation, C_t), world_pts)
        ρ, θ = (!isnothing(rhos) || !isnothing(thetas) ? hough_transform(projected_points) : (nothing, nothing))
        @assert size(projected_points) == size(pixel_locations)
        return sum(norm.(projected_points.-pixel_locations))
    end
    return f
end

Optim.minimizer(lsr::LeastSquaresResult) = lsr.minimizer
Optim.converged(lsr::LeastSquaresResult) = LeastSquaresOptim.converged(lsr)
function pnp(world_pts::Vector{ENU{Meters}},
             pixel_locations::Vector{Point{2, Pixels}},
             cam_rotation::Union{LinearMap{<:Rotation{3, Float64}}, Rotation{3, Float64}};
             initial_guess::ENU{Meters} = ENU(-100.0m, 0m, 30m),
             method=NelderMead()
             )
    # strip units for Optim.jl package. See https://github.com/JuliaNLSolvers/Optim.jl/issues/695.
    world_pts = world_pts .|> ustrip
    pixel_locations = pixel_locations .|> ustrip
    initial_guess = initial_guess |> ustrip

    f = build_pnp_objective(world_pts,
                            pixel_locations,
                            cam_rotation)
    f_ = TwiceDifferentiable(f, MVector(1., 1, 1))

    initial_guess = typeof(initial_guess)(initial_guess[1], initial_guess[2], max(initial_guess[3], 1.0))
    presolve = Optim.optimize(f_,
                   MVector(initial_guess),
                   method,
                   # NewtonTrustRegion(),
                   Optim.Options(f_tol=1e-7),
                   autodiff=:forward,
                   )
    return presolve
    # sol = LeastSquaresOptim.optimize(f,
    #                Optim.minimizer(presolve),
    #                LevenbergMarquardt();
    #                lower=[-Inf, -Inf, 0],
    #                autodiff=:forward,
    #                g_tol=1e-7,
    #                iterations=1_000,
    #                )
    # @assert f(Optim.minimizer(sol)) < 1e4 (sol, Optim.minimizer(sol))
    # (!isnothing(opt_traces) && push!(opt_traces, sol))
    # @debug sol
    return sol
end

"Hough transform."
function compute_rho_theta(p1, p2, p3)
    p4(λ) = p1 + λ*(p2-p1)
    λ = dot((p2-p1), (p3-p1)) / norm(p2-p1)^2
    @assert isapprox(dot(p2-p1, p4(λ)-p3)/norm(p2), 0.; atol=1e-4) "$(dot(p2-p1, p4(λ)-p3))"
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
    θ = (; lhs=ρ_θ_lhs[2], rhs=ρ_θ_rhs[2])
    ρ, θ
end
