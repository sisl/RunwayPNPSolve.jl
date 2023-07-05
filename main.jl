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
using StatsBase
using Unzip

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
# Error slider
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
                end
                for p in runway_corners]
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
num_pose_est = lift(num_pose_est_box.stored_string) do str
    tryparse(Int, str)
end
perturbed_pose_estimates = lift(projected_points,
                                σ,
                                perturbation_mask,
                                num_pose_est,
                                C_t_true) do projected_points, σ, mask, num_pose_est, C_t_true
    pts = Point3d.([pnp(runway_corners, projected_points .+ σ*mask.*[randn(2) for _ in 1:4];
                        initial_guess = Array(C_t_true)+10.0*randn(3))
                    for _ in 1:num_pose_est])
    # may filter to contain pose outliers
    # filter(p -> (p[2] ∈ 0±30) && (p[3] ∈ 0..50) && (p[1] ∈ -150..0),
    #        pts) |> collect
    pts
end
scatter!(scene, perturbed_pose_estimates; color=:red)
#
## construct pose estimate errors
errors_obs = lift(C_t_true, projected_points) do C_t_true, projected_points
    local num_pose_est = 100
    log_errs = LinRange(-10:0.5:-3)
    errs = exp.(log_errs)
    function compute_means_and_stds(σ)
        pts = Point3d.([pnp(runway_corners, projected_points .+ σ.*[randn(2) for _ in 1:4];
                            initial_guess = Array(C_t_true)+10.0*randn(3))
                        for _ in 1:num_pose_est])
        Δ = (pts .- C_t_true)
        Δ = map(p->abs.(p), Δ)
        μ_x, μ_y, μ_z = mean.([getindex.(Δ, i) for i in 1:3])
        std_x, std_y, std_z = std.([getindex.(Δ, i) for i in 1:3])
        q5_x, q5_y, q5_z = quantile.([getindex.(Δ, i) for i in 1:3], 0.05)
        q95_x, q95_y, q95_z = quantile.([getindex.(Δ, i) for i in 1:3], 0.95)
        (; x=μ_x, y=μ_y, z=μ_z), (; x=std_x, y=std_y, z=std_z), (; x=q5_x, y=q5_y, z=q5_z), (; x=q95_x, y=q95_y, z=q95_z)
    end
    means, stds, q5s, q95s = begin
        means_, stds_, q5s_, q95s_ = unzip(ThreadsX.map(σ->compute_means_and_stds(σ), errs))  # Array of tuples{:x,:y,:z}
        # means_, stds_ = unzip(compute_means_and_stds.(errs))  # Array of tuples{:x,:y,:z}
        StructArray(means_), StructArray(stds_), StructArray(q5s_), StructArray(q95s_)  # tuple{:x,:y,:z} of arrays (view)
    end
    (; σ=errs, means, stds, q5s, q95s)
end
errors = (;σ=lift(errors_obs) do errors_obs; errors_obs.σ end,
           means=(;x=lift(errors_obs) do errors_obs; errors_obs.means.x end,
                   y=lift(errors_obs) do errors_obs; errors_obs.means.y end,
                   z=lift(errors_obs) do errors_obs; errors_obs.means.z end),
           stds =(;x=lift(errors_obs) do errors_obs; errors_obs.stds.x end,
                   y=lift(errors_obs) do errors_obs; errors_obs.stds.y end,
                   z=lift(errors_obs) do errors_obs; errors_obs.stds.z end),
           q5s=(;x=lift(errors_obs) do errors_obs; errors_obs.q5s.x end,
                   y=lift(errors_obs) do errors_obs; errors_obs.q5s.y end,
                   z=lift(errors_obs) do errors_obs; errors_obs.q5s.z end),
           q95s=(;x=lift(errors_obs) do errors_obs; errors_obs.q95s.x end,
                   y=lift(errors_obs) do errors_obs; errors_obs.q95s.y end,
                   z=lift(errors_obs) do errors_obs; errors_obs.q95s.z end),
          )
#
error_plots_grid = rhs_grid[3, 1]
err_axes = (; x=Axis(error_plots_grid[1, 1]; xscale=log, title="Errors x direction"),
              y=Axis(error_plots_grid[1, 2]; xscale=log, title="Errors y direction"),
              z=Axis(error_plots_grid[1, 3]; xscale=log, title="Errors z direction"))
hideydecorations!(err_axes.y, ticks=false, ticklabels=true)
hideydecorations!(err_axes.z, ticks=false, ticklabels=true)
Makie.linkaxes!(err_axes...)
errorbars!(err_axes.x, errors.σ, errors.means.x, errors.q5s.x, errors.q95s.x)
lines!(err_axes.x, errors.σ, errors.means.x)
errorbars!(err_axes.y, errors.σ, errors.means.y, errors.q5s.y, errors.q95s.y)
lines!(err_axes.y, errors.σ, errors.means.y)
errorbars!(err_axes.z, errors.σ, errors.means.z, errors.q5s.z, errors.q95s.z)
lines!(err_axes.z, errors.σ, errors.means.z)
on(C_t_true) do
    reset_limits!(err_axes.x)  # y and z are linked automatically
end
#
fig
