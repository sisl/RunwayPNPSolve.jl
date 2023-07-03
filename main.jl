using Revise
includet("pnp.jl")
using StaticArrays
# using CameraModels
using LinearAlgebra
using Rotations
using CoordinateTransformations
using Makie, GLMakie
using Optim
using ThreadsX
using IntervalSets

"""
Some definitions:

All units in meters. Coordinate system is located in the center front of the runway.
x-axis along runway.
y-axis to the left.
z-axis up.

For now we assume that all runway points are in the x-y plane.
"""
# Pos3 = SVector{3, Float64}
# Rot3 = SVector{3, Float64}
Point2d = Point2{Float64}
Vec2d = Vec2{Float64}
Point3d = Point3{Float64}
Vec3d = Vec3{Float64}

runway_corners = Point3d[
    [  0, -5, 0],
    [  0,  5, 0],
    [100, -5, 0],
    [100,  5, 0]] ./ 10

R_t_true = RotY{Float32}(π/2)

fig = Figure()
scene = LScene(fig[1, 1], show_axis=false, scenekw = (backgroundcolor=:gray, clear=true))
slidergrid =  SliderGrid(fig[2, 1],
    (label="Error scale [log]", range = -10:0.1:-1, startvalue=-3, format=x->string(round(exp(x); sigdigits=2)))
)
rhs_grid = GridLayout(fig[1, 2]; tellheight=false)
toggles = [Toggle(rhs_grid[i, 1]; active=true) for i in 1:4]
toggle_labels = let labels = ["Front left", "Front right", "Back left", "Back right"]
    [Label(rhs_grid[i, 2], labels[i]) for i in 1:4]
end
scenario_menu = Menu(rhs_grid[5, 1:2]; options=["near", "mid", "far"], default="mid")
C_t_true = lift(scenario_menu.selection) do menu
    menu == "near" && return Point3d([-10, 0, 10])
    menu == "mid"  && return Point3d([-100, 0, 10])
    menu == "far"  && return Point3d([-500, 0, 10])
end
Cam_translation = lift(C_t_true) do C_t_true; AffineMap(R_t_true, C_t_true) end
cam_transform = lift(Cam_translation) do Cam_translation; PerspectiveMap() ∘ inv(Cam_translation) end
projected_points = lift(cam_transform) do cam_transform; map(cam_transform, runway_corners) end
projected_points_global = lift(Cam_translation, projected_points) do Cam_translation, projected_points
    map(Cam_translation ∘ AffineMap(I(3)[:, 1:2], Float64[0;0;1]), projected_points)
end
# rhs_grid[1, 1] = grid!(hcat(toggles, toggle_labels); tellheight=false, tellwidth=true)
#
#
cam3d!(scene; far=1e9)
lines!(scene, runway_corners[[1, 2, 4, 3, 1]])
# arrows!(scene, [Point3f(C_t_true), ], [Vec3f([1., 0, 0]), ]; normalize=true, lengthscale=0.5)
arrows!(scene,
        fill(Point3d(0, 0, 0), 3),
        Vec3f[[1,0,0,],[0,1,0],[0,0,1]]./5;
        arrowsize=Vec3f(0.1, 0.1, 0.2)
        )
surface!(scene, getindex.(runway_corners, 1),
                getindex.(runway_corners, 2),
                getindex.(runway_corners, 3))
corner_lines = [lift(C_t_true) do C_t_true
                    [p, C_t_true]
                    end for p in runway_corners]
for l in corner_lines
    lines!(scene, l)
end
scatter!(scene, projected_points_global)
# update_cam!(scene.scene, Array(C_t_true), Float32[0, 0, 0])
perturbation_mask = lift(toggles[1].active, toggles[2].active, toggles[3].active, toggles[4].active) do a, b, c, d;
    Int[a;b;c;d]
end
perturbed_locations = lift(projected_points, slidergrid.sliders[1].value, perturbation_mask) do projected_points, σ, mask
    pts = Point3d.([perturb_x1(projected_points, exp(σ); mask=mask) for _ in 1:100])
    filter(p -> (p[2] ∈ 0±30) && (p[3] ∈ 0..50) && (p[1] ∈ -150..0),
           pts) |> collect
end
scatter!(scene, perturbed_locations; color=:red)
scene

# add_dim(x::AbstractArray) = reshape(x, (size(x)...,1))
# add_dim2(x::AbstractArray) = reshape(x, size(x)[1], 1, size(x)[2])
# OpenCV.solvePnP(stack(Array.(runway_corners)) |> add_dim2,
#                 stack(Array.(projected_points_global)) |> add_dim2,
#                 Matrix{Float32}(I(3)) |> add_dim,
#                 zeros(Float32, 4, 1, 1)
#                 )


# scene = Scene(backgroundcolor=:gray)
# lines!(scene, Rect2f(-1, -1, 2, 2), linewidth=5, color=:black)
# cam3d!(scene)
# scene


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

# central_fdm(5, 1; factor=1e-8/eps())(perturb_x1, 0)



# measure x-sensititity for back left point
# using Finite Difference


# sol.minimizer


# experiment 1
