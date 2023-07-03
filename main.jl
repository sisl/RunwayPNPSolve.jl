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
toggle_grid = GridLayout(rhs_grid[1, 1])
toggles = [Toggle(toggle_grid[i, 1]; active=true) for i in 1:4]
toggle_labels = let labels = ["Front left", "Front right", "Back left", "Back right"]
    [Label(toggle_grid[i, 2], labels[i]) for i in 1:4]
end
scenario_menu = Menu(toggle_grid[5, 1:2]; options=["near", "mid", "far"], default="mid")
#
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
#
projected_points_rect = lift(projected_points) do projected_points
    projected_points[[1, 2, 4, 3, 1]]
end
cam_view_ax = Axis(rhs_grid[2, 1])
cam_view = lines!(cam_view_ax, projected_points_rect)
#
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
