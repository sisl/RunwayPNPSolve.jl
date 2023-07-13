using Rotations
using CoordinateTransformations
using LinearAlgebra: dot, norm
using Optim
using ReTest
using Tau
using Roots
using LeastSquaresOptim
include("typedefs.jl")

function build_pnp_objective(
             world_pts, pixel_locations;
             rhos=nothing,
             thetas=nothing,
             feature_mask=[1;1;1],
             gt_rot=Rotations.IdentityMap(),
    )

    f(C_t) = let
        R_t_true = RotY{Float32}(π/2)
        projected_points = project_points(AffineMap(R_t_true, C_t), world_pts)
        ρ, θ = hough_transform(projected_points)
        @assert size(projected_points) == size(pixel_locations)
        return ( sum(norm.(projected_points[1:2].-pixel_locations[1:2])) * feature_mask[1]  # front corners
               + sum(norm.(projected_points[3:4].-pixel_locations[3:4])) * feature_mask[2]  # back corners
               # + 0.0*(!isnothing(rhos)   ? norm(rhos   - [ρ[:lhs]; ρ[:rhs]]) : 0) * feature_mask[2]
               # tranform angles to imaginary numbers on unit circle before comparison.
               # avoid problems with e.g. dist(-0.9pi, 0.9pi)
               + 1/10*(!isnothing(thetas) ? sum(norm.(exp.(im.*thetas) .- exp.(im.*[θ[:lhs]; θ[:rhs]]))) : 0) * feature_mask[3]
               )
    end
    return f
end

Optim.minimizer(lsr::LeastSquaresResult) = lsr.minimizer
Optim.converged(lsr::LeastSquaresResult) = LeastSquaresOptim.converged(lsr)
function pnp(world_pts, pixel_locations;
             rhos=nothing,
             thetas=nothing,
             feature_mask=[1;1;1],
             gt_rot=Rotations.IdentityMap(),
             initial_guess = Point3f([-100, 0, 30]),
             opt_traces=nothing)
    f = build_pnp_objective(world_pts, pixel_locations;
                            rhos=rhos,thetas=thetas,feature_mask=feature_mask,gt_rot=gt_rot)

    initial_guess[3] = max(initial_guess[3], 1.0)
    presolve = Optim.optimize(f,
                   Array(initial_guess),
                   NewtonTrustRegion(),
                   Optim.Options(f_tol=1e-7),
                   autodiff=:forward,
                   )
    sol = LeastSquaresOptim.optimize(f,
                   Optim.minimizer(presolve),
                   LevenbergMarquardt();
                   lower=[-Inf, -Inf, 0],
                   autodiff=:forward,
                   g_tol=1e-7,
                   iterations=1_000,
                   )
    @assert f(Optim.minimizer(sol)) < 1e1 (sol, Optim.minimizer(sol))
    # (!isnothing(opt_traces) && push!(opt_traces, sol))
    # @debug sol
    return sol
end

"Hough transform."
function compute_rho_theta(p1, p2, p3)
    p4(λ) = p1 + λ*(p2-p1)
    λ = dot((p2-p1), (p3-p1)) / norm(p2-p1)^2
    @assert isapprox(dot(p2-p1, p4(λ)-p3), 0.; atol=1e-6) "$(dot(p2-p1, p4(λ)-p3))"
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
