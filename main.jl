using Revise
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
using StructArrays

"""
Some definitions:

All units in meters. Coordinate system is located in the center front of the runway.
x-axis along runway.
y-axis to the left.
z-axis up.

For now we assume that all runway points are in the x-y plane.
"""
# Point2/3f already exists, also define for double precision
includet("pnp.jl")


runway_corners = Point3d[
    [  0, -5, 0],
    [  0,  5, 0],
    [100, -5, 0],
    [100,  5, 0]]
runway_corners_far = [10*(runway_corners[3] - runway_corners[1])+runway_corners[1],
                      10*(runway_corners[4] - runway_corners[2])+runway_corners[2]]

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
Cam_translation = @lift AffineMap(R_t_true, $C_t_true)
cam_transform = @lift PerspectiveMap() ∘ inv($Cam_translation)
projected_points = @lift map($cam_transform, runway_corners)
projected_points_global = @lift map($Cam_translation ∘ AffineMap(I(3)[:, 1:2], Float64[0;0;1]), $projected_points)
## plot points projected onto 2D camera plane
flip_coord_system(p) = typeof(p)(-p[2], -p[1])
projected_points_rect = lift(projected_points) do projected_points
    pts = flip_coord_system.(projected_points)
    pts[[1, 2, 4, 3, 1]]
end
cam_view_ax = Axis(rhs_grid[2, 1], width=800, aspect=DataAspect(), limits=(-1,1,-1,1)./8)
lines!(cam_view_ax, projected_points_rect)
# plot far points in 2d
projected_points_far = @lift map($cam_transform, runway_corners_far)
projected_lines_far = (@lift(flip_coord_system.([$(projected_points)[1], $(projected_points_far)[1]])),
                       @lift(flip_coord_system.([$(projected_points)[2], $(projected_points_far)[2]])))
lines!.(cam_view_ax, projected_lines_far; color=:gray, linestyle=:dot)
# plot 1std of Gaussian noise
projected_points_2d = [lift(projected_points_rect) do rect; rect[i] end
                       for i in 1:4]
meshscatter!.(cam_view_ax, projected_points_2d, marker=Makie.Circle(Point2d(0,0), 1.0), markersize=σ)
# Compute and plot line estimates
ρ_θ_lhs = lift(projected_points) do ppts
    compute_rho_theta(ppts[1], ppts[3], (ppts[1]+ppts[2])/2)
end
ρ_θ_rhs = lift(projected_points) do ppts
    compute_rho_theta(ppts[2], ppts[4], (ppts[1]+ppts[2])/2)
end
ρ_θ_line_lhs = lift(projected_points, ρ_θ_lhs) do ppts, (ρ, θ)
    p0 = (ppts[1]+ppts[2])/2
    flip_coord_system.([p0, p0 + ρ*[cos(θ); sin(θ)]])
end
ρ_θ_line_rhs = lift(projected_points, ρ_θ_rhs) do ppts, (ρ, θ)
    p0 = (ppts[1]+ppts[2])/2
    flip_coord_system.([p0, p0 + ρ*[cos(θ); sin(θ)]])
end
lines!(cam_view_ax, ρ_θ_line_lhs)
lines!(cam_view_ax, ρ_θ_line_rhs)
## Set up camera
cam3d!(scene; near=0.01, far=1e9, rotation_center=:eyeposition, cad=true, zoom_shift_lookat=false,
       mouse_rotationspeed = 5f-1,
       mouse_translationspeed = 0.1f0,
       mouse_zoomspeed = 5f-1,
       )
## Draw runway and coordinate system
# Normal runway rectangle
lines!(scene, runway_corners[[1, 2, 4, 3, 1]])
# Draw 3d runway lines into the distance
lines!(scene, [runway_corners[1], runway_corners_far[1]]; color=:blue)
lines!(scene, [runway_corners[2], runway_corners_far[2]]; color=:blue)
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
                                ρ_θ_lhs,
                                ρ_θ_rhs,
                                σ,
                                perturbation_mask,
                                num_pose_est,
                                C_t_true) do projected_points, (ρ_lhs, θ_lhs), (ρ_rhs, θ_rhs), σ, mask, num_pose_est, C_t_true
    pts = Point3d.([pnp(runway_corners, projected_points .+ σ*mask.*[randn(2) for _ in 1:4];
                        rhos=[ρ_lhs; ρ_rhs].+σ.*randn(2),
                        thetas=[θ_lhs; θ_rhs].+σ.*randn(2),
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
errors_obs = lift(C_t_true, projected_points, ρ_θ_lhs, ρ_θ_rhs) do C_t_true, projected_points, (ρ_lhs, θ_lhs), (ρ_rhs, θ_rhs)
    local num_pose_est = 100
    log_errs = LinRange(-10:0.5:-5)
    errs = exp.(log_errs)
    function compute_means_and_stds(σ)
        pts = Point3d.([pnp(runway_corners, projected_points .+ σ.*[randn(2) for _ in 1:4];
                            rhos=[ρ_lhs; ρ_rhs].+σ.*randn(2),
                            thetas=[θ_lhs; θ_rhs].+σ.*randn(2),
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
# Actually plot all the errors
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
