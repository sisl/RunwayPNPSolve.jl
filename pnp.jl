using Rotations
using CoordinateTransformations
using LinearAlgebra: dot, norm
using Optim
using ReTest
using Tau
using Roots
include("typedefs.jl")

function build_pnp_objective(
             world_pts, pixel_locations;
             rhos=nothing,
             thetas=nothing,
             feature_mask=[1;1;1],
             gt_rot=Rotations.IdentityMap())

    f(C_t) = let
        R_t_true = RotY{Float32}(π/2)
        projected_points = project_points(AffineMap(R_t_true, C_t), world_pts)
        ρ, θ = hough_transform(projected_points)
               # tranform angles to imaginary numbers on unit circle before comparison.
               # avoid problems with e.g. dist(-0.9pi, 0.9pi)
               + (!isnothing(thetas) ? sum(norm.(exp.(im.*thetas) .- exp.(im.*[θ_gt_lhs; θ_gt_rhs]))) : 0) * feature_mask[3]
               )
    end
    return f
end

function pnp(world_pts, pixel_locations;
             rhos=nothing,
             thetas=nothing,
             feature_mask=[1;1;1],
             gt_rot=Rotations.IdentityMap(),
             initial_guess = Point3f([-100, 0, 30]),
             opt_traces=nothing)
    f = build_pnp_objective(world_pts, pixel_locations;
                            rhos=rhos,thetas=thetas,feature_mask=feature_mask,gt_rot=gt_rot)

    sol = optimize(f, Array(initial_guess),
                   Optim.NewtonTrustRegion(), Optim.Options(x_tol=1e-4, f_tol=1e-6);
                   autodiff=:forward)
    @assert f(Optim.minimizer(sol)) < 1e8 (sol, Optim.minimizer(sol))
    (!isnothing(opt_traces) && push!(opt_traces, sol))
    @debug sol
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
