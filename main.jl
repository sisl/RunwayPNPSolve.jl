using StaticArrays
# using CameraModels
using LinearAlgebra
using Rotations
using CoordinateTransformations
using Makie, GLMakie
using Optim
"""
Some definitions:

All units in meters. Coordinate system is located in the center front of the runway.
x-axis along runway.
y-axis to the left.
z-axis up.

For now we assume that all runway points are in the x-y plane.
"""
Pos3 = SVector{3, Float64}
Rot3 = SVector{3, Float64}

runway_corners = Point3f[
    [  0, -5, 0],
    [  0,  5, 0],
    [100, -5, 0],
    [100,  5, 0]] ./ 10

C_t_true = Point3f([-100, 0, 30]) ./ 10
R_t_true = RotY{Float32}(π/2)

Cam_translation = AffineMap(R_t_true, C_t_true)
cam_transform = PerspectiveMap() ∘ inv(Cam_translation)
projected_points = map(cam_transform, runway_corners)
projected_points_global = map(Cam_translation ∘ AffineMap(I(3)[:, 1:2], Float32[0;0;1]), projected_points)

# fig = Figure(; size=(1600, 1600))
# ax = Axis3(fig; aspect=(1, 1, 1), limits=((-550, 150), (-10, 10), (-5, 50)))

scene = Scene(backgroundcolor=:gray)
cam3d!(scene)
lines!(scene, runway_corners[[1, 2, 4, 3, 1]])
# arrows!(scene, [Point3f(C_t_true), ], [Vec3f([1., 0, 0]), ]; normalize=true, lengthscale=0.5)
arrows!(scene,
        fill(Point3f(0, 0, 0), 3),
        Vec3f[[1,0,0,],[0,1,0],[0,0,1]]./5;
        arrowsize=Vec3f(0.1, 0.1, 0.2)
        )
surface!(scene, getindex.(runway_corners, 1),
                getindex.(runway_corners, 2),
                getindex.(runway_corners, 3))
for p in runway_corners
    lines!(scene, [p, C_t_true])
end
scatter!(scene, projected_points_global)
update_cam!(scene, Float32[-10, 0, 3], Float32[0, 0, 0])
scene

add_dim(x::AbstractArray) = reshape(x, (size(x)...,1))
add_dim2(x::AbstractArray) = reshape(x, size(x)[1], 1, size(x)[2])
OpenCV.solvePnP(stack(Array.(runway_corners)) |> add_dim2,
                stack(Array.(projected_points_global)) |> add_dim2,
                Matrix{Float32}(I(3)) |> add_dim,
                zeros(Float32, 4, 1, 1)
                )


scene = Scene(backgroundcolor=:gray)
lines!(scene, Rect2f(-1, -1, 2, 2), linewidth=5, color=:black)
cam3d!(scene)
scene


# OpenCV.solveP3P(rand(Float32, 3, 3, 3),
#                 rand(Float32, 2, 2, 2),
#                 Matrix{Float32}(I(3)) |> add_dim,
#                 zeros(Float32, 4, 1, 1),
#                 OpenCV.SOLVEPNP_P3P)


"""
the function we ultimately want is the following:
input:
- real object locations (3d)
- pixel locations (2d)
- rotation
- initial guess
- (distance from runway?)
output:
- 3d pose estimate

We can either minimize in the pixel space, or in the world space. (Maybe it doesn't matter either way?)
"""


# function pnp(world_pts, pixel_locations;
#              gt_rot=Rotations.IdentityMap(),
#              initial_guess = Point3f([-100, 0, 30]))

#     # C_t_true = Point3f([-100, 0, 30]) ./ 10
#     rotXtoZ = RotY{Float32}(π/2)

#     f(C_t) = begin
#         # Cam_translation = AffineMap(rotXtoZ ∘ gt_rot, Point3f(C_t))
#         Cam_translation = AffineMap(rotXtoZ, Point3f(C_t))
#         cam_transform = PerspectiveMap() ∘ inv(Cam_translation)
#         projected_points = map(cam_transform, world_pts)
#         # projected_points_global = map(Cam_translation ∘ AffineMap(I(3)[:, 1:2], Float32[0;0;1]),
#         #                               projected_points)
#         return sum(norm.(projected_points .- pixel_locations))
#     end

#     sol = optimize(f, Array(initial_guess))
#     return sol.minimizer
# end
pos_est = pnp(runway_corners, projected_points;
              initial_guess = Array(C_t_true)+1*randn(3))

perturb_x1(δ) = begin
    global projected_points
    projected_points_ = copy(projected_points)
    # projected_points_[2] += Vec2f(δ, 0)
    projected_points_[2] += Vec2f(0, δ)
    pos_est = pnp(runway_corners, projected_points_;
                  initial_guess = Array(C_t_true)+0.0*randn(3))
    @debug δ, pos_est
    return pos_est
end
central_fdm(5, 1; factor=1e-8/eps())(perturb_x1, 0)



# measure x-sensititity for back left point
# using Finite Difference


# sol.minimizer


# experiment 1
