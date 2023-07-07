using Rotations
using CoordinateTransformations
using LinearAlgebra: dot, norm
using Optim
using ReTest
using Tau
using Roots
include("typedefs.jl")
function pnp(world_pts, pixel_locations;
             gt_rot=Rotations.IdentityMap(),
             initial_guess = Point3f([-100, 0, 30]))

    # C_t_true = Point3f([-100, 0, 30]) ./ 10
    rotXtoZ = RotY{Float64}(π/2)

    f(C_t) = begin
        # Cam_translation = AffineMap(rotXtoZ ∘ gt_rot, Point3f(C_t))
        Cam_translation = AffineMap(rotXtoZ, C_t)
        cam_transform = PerspectiveMap() ∘ inv(Cam_translation)
        projected_points = map(cam_transform, world_pts)
        # projected_points_global = map(Cam_translation ∘ AffineMap(I(3)[:, 1:2], Float32[0;0;1]),
        #                               projected_points)
        return sum(norm.(projected_points .- pixel_locations))
    end

    sol = optimize(f, Array(initial_guess), Optim.NewtonTrustRegion(), Optim.Options(f_tol=1e-7);
                   autodiff=:forward)
    @assert f(Optim.minimizer(sol)) < 1e8 (sol, Optim.minimizer(sol))
    @debug sol
    return Optim.minimizer(sol)
end

"Hough transform."
function compute_rho_theta(p1, p2, p3)
    p4(λ) = p1 + λ*(p2-p1)
    λ = fzero(λ->dot(p2-p1, p4(λ)-p3), 0.)
    @debug λ, p4(λ)
    ρ = norm(p4(λ) - p3)
    θ = acos( dot([1;0], p4(λ)-p3)/ρ ) * sign((p4(λ)-p3)[2])
    return ρ, θ
end
# @testset "compute_rho_theta"
ρ, θ = compute_rho_theta(Point2d(-2, 0), Point2d(0, -2), Point2d(0, 0))
@test all((ρ, θ) .≈ (√(2), -3/8*τ))
