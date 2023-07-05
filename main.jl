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

# Point2/3f already exists, also define for double precision
Point2d = Point2{Float64}
Vec2d = Vec2{Float64}
Point3d = Point3{Float64}
Vec3d = Vec3{Float64}

runway_corners = Point3d[
    [  0, -5, 0],
    [  0,  5, 0],
    [100, -5, 0],
    [100,  5, 0]]

R_t_true = RotY{Float32}(π/2)

fig = Figure()
scene = LScene(fig[1, 1], show_axis=false, scenekw = (backgroundcolor=:gray, clear=true))
slidergrid =  SliderGrid(fig[2, 1],
    (label="Error scale [log]", range = -10:0.1:-1, startvalue=-7, format=x->string(round(exp(x); sigdigits=2)))
)
σ = lift(slidergrid.sliders[1].value) do x; exp(x) end
rhs_grid = GridLayout(fig[1, 2]; tellheight=false)
toggle_grid = GridLayout(rhs_grid[1, 1])
# Set up noise toggles, scenario menu, num pose estimates
toggles = [Toggle(toggle_grid[i, 2]; active=true) for i in 1:4]
toggle_labels = let labels = ["Noise front left:", "Noise front right:", "Noise back left:", "Noise back right:"]
    [Label(toggle_grid[i, 1], labels[i]) for i in 1:4]
end
## Set up scenario, which affects cam position and therefore all the projections
Label(toggle_grid[5, 1], "Scenario:")
scenario_menu = Menu(toggle_grid[5, 2]; options=["near (10m)", "mid (100m)", "far (500m)"], default="mid (100m)")
#
C_t_true = lift(scenario_menu.selection) do menu
    menu == "near (10m)" && return Point3d([-10, 0, 10])
    menu == "mid (100m)"  && return Point3d([-100, 0, 10])
    menu == "far (500m)"  && return Point3d([-500, 0, 10])
end
Cam_translation = lift(C_t_true) do C_t_true; AffineMap(R_t_true, C_t_true) end
cam_transform = lift(Cam_translation) do Cam_translation; PerspectiveMap() ∘ inv(Cam_translation) end
projected_points = lift(cam_transform) do cam_transform; map(cam_transform, runway_corners) end
projected_points_global = lift(Cam_translation, projected_points) do Cam_translation, projected_points
    map(Cam_translation ∘ AffineMap(I(3)[:, 1:2], Float64[0;0;1]), projected_points)
end
## plot points projected onto 2D camera plane
projected_points_rect = lift(projected_points) do projected_points
    pts = map(p->typeof(p)(-p[2], -p[1]),
              projected_points) |> collect
    pts[[1, 2, 4, 3, 1]]
end
projected_points_2d = [lift(projected_points_rect) do rect; rect[i] end
                       for i in 1:4]
cam_view_ax = Axis(rhs_grid[2, 1], width=500, aspect=DataAspect(), limits=(-1,1,-1,1)./8)
cam_view = lines!(cam_view_ax, projected_points_rect)
# plot 1std of Gaussian noise
meshscatter!(cam_view_ax, projected_points_2d[1], marker=Makie.Circle(Point2d(0,0), 1.0), markersize=σ)
meshscatter!(cam_view_ax, projected_points_2d[2], marker=Makie.Circle(Point2d(0,0), 1.0), markersize=σ)
meshscatter!(cam_view_ax, projected_points_2d[3], marker=Makie.Circle(Point2d(0,0), 1.0), markersize=σ)
meshscatter!(cam_view_ax, projected_points_2d[4], marker=Makie.Circle(Point2d(0,0), 1.0), markersize=σ)
## Set up camera
cam3d!(scene; near=0.01, far=1e9, rotation_center=:eyeposition, cad=true, zoom_shift_lookat=false,
       mouse_rotationspeed = 5f-1,
       mouse_translationspeed = 0.1f0,
       mouse_zoomspeed = 5f-1,
       )
## Draw runway and coordinate system
lines!(scene, runway_corners[[1, 2, 4, 3, 1]])
# arrows!(scene, [Point3f(C_t_true), ], [Vec3f([1., 0, 0]), ]; normalize=true, lengthscale=0.5)
arrows!(scene,
        fill(Point3d(0, 0, 0), 3),
        Vec3f[[1,0,0,],[0,1,0],[0,0,1]]./5;
        arrowsize=Vec3f(0.1, 0.1, 0.2)
        )
arrows!(scene,  # larger coordinate system
        fill(Point3d(0, -15, 0), 3),
        Vec3f[[1,0,0,],[0,1,0],[0,0,1]]*5;
        arrowsize=Vec3f(2, 2, 3)
        )
# Plot runway surface
surface!(scene, getindex.(runway_corners, 1),
                getindex.(runway_corners, 2),
                getindex.(runway_corners, 3))
# Draw lines from corners to camera
corner_lines = [lift(C_t_true) do C_t_true
                    [p, C_t_true]
                    end for p in runway_corners]
for l in corner_lines
    lines!(scene, l)
end
# Draw Projected points
scatter!(scene, projected_points_global)
# Set cam position
update_cam!(scene.scene, Array(C_t_true[]).-[20.,0,0], Float32[0, 0, 0])
# Compute pose estimates
perturbation_mask = lift(toggles[1].active, toggles[2].active, toggles[3].active, toggles[4].active) do a, b, c, d;
    Int[a;b;c;d]
end
Label(toggle_grid[6, 1], "Num pose estimates: ")
num_pose_est_box = Textbox(toggle_grid[6, 2], stored_string = "100",
                       validator = Int, tellwidth = false)
perturbed_locations = lift(projected_points, σ, perturbation_mask, num_pose_est_box.stored_string) do projected_points, σ, mask, num_pose_est
    num_pose_est = tryparse(Int, num_pose_est)
    pts = Point3d.([perturb_x1(projected_points, σ; mask=mask) for _ in 1:num_pose_est])
    # filter(p -> (p[2] ∈ 0±30) && (p[3] ∈ 0..50) && (p[1] ∈ -150..0),
    #        pts) |> collect
    pts
end
scatter!(scene, perturbed_locations; color=:red)
fig
