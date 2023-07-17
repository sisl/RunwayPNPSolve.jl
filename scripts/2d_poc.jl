# in emacs, run (setq ns-right-alternate-modifier 'meta) to use right option + ] to set accents.
"""
Proof of concept implementation of Bayesian PnP solving with known attitude (rotation) from a set of points.
Simplified 2d version, but embedded in 3d space.

The function signature looks conceptually like this
Inputs:
- 3d point in world coords
- true attitude / rotation, given as LinearMap
- image correspondance (x, y) in image coords (x facing downwards)
- image correspondance standard deviations (σ_x, σ_y) in image coords (x facing downwards)
- (optional) distance prior
Outputs:
- Normal Distribution for the pose in world coordinates.

Note: There are three coordinate systems.
(1) The world coordinate system, which is centered in the landing strip entry point, with x alongtrack and z up (as in SVMP).
(2) The camera coordinate system, i.e. world points rotated and translated such that they are given as seen from the camera.
(3) The image coordinate system, i.e. correspondances of world points onto the camera lens.
We denote points either without, or with one or two apostrophes.
I.e., p''_1 would be a point in the image frame.

Conceptual steps:
(1) Fix a distance x=const. Then, find a value (y, z) for the pose that correctly projects the world point p onto the image correspondance p''.
Call this pose guess P_0 (as given in world coords).
(2) Construct the ray (P_0 -> p). We assume that the true pose lies along this ray, with some disturbance given by the Gaussian noise.
(3) Project the standard deviations onto an orthogonal projection of the ray.
"""

using LinearAlgebra, StaticArraysCore
using Rotations, CoordinateTransformations, GeometryBasics
using Tau  # τ = 2*π
using Roots, ForwardDiff; D(f) = x->ForwardDiff.derivative(f, x)
# using Makie, GLMakie
using Plots, StatsPlots
using Distributions
include("../src/mv_normal_prod.jl")
include("plot.jl")

using GeometryBasics
Point2d = Point2{Float64}
Vec2d = Vec2{Float64}
Point3d = Point3{Float64}
Vec3d = Vec3{Float64}

# Note: Type p′ as p+\prime<TAB>
p = [10.; 0; 8]
cam_rot = LinearMap(RotY(-τ/4))  # to point the camera z axis forward
cam_pos = Translation([-10.; 0; 5])
cam_pose = cam_pos ∘ cam_rot
make_projection_map(cam_pose) = PerspectiveMap() ∘ inv(cam_pose)
p′′ = make_projection_map(cam_pose)(p)

function compute_bayesian_pose_estimate(
    p :: Point3d,
    cam_rot :: LinearMap,
    p′′ :: Point2d,
    σ′′ :: SVector{2, Float64};
    x_guess :: Float64 = -30.
    ) :: MvNormal

    # Step 1
    P_0 = let x_guess = -30.
        projection_error(z) = let P = Translation([x_guess; 0; z])
            cam_pose = P ∘ cam_rot
            projection_map = make_projection_map(cam_pose)
            sq = x->x.^2
            return sum(sq, projection_map(p) - p′′)
        end
        f = projection_error
        z_0 = Roots.find_zero((f, D(f)),
                        0., Roots.Newton())
        Translation([x_guess; 0; z_0])
    end

    # Step 2
    ray_vec = p - P_0.translation

    # Step 3
    σ_z = let σ_z_pre = cam_rot([σ′′[2];0;0]),  # pre projection
            ray_vec = normalize(ray_vec)
        σ_z_pre - dot(σ_z_pre, ray_vec) * ray_vec
    end
    @assert abs(dot(σ_z, ray_vec)) < 1e-6

    # # Step 5
    μ = P_0.translation
    Σ_ = let
        # Step 4
        Λ = diagm([1e3; 10; norm(σ_z)])
        vecs  = [normalize(ray_vec) [0.;1;0] normalize(σ_z)]
        vecs * Λ * vecs'
    end
    Σ = 1/2 * (Σ_ + Σ_')  # make sure it's hermetian, overcoming numerical errors
    @assert all(abs.(Σ - Σ') .< 1e-6)

    MvNormal(μ, Σ)
end
