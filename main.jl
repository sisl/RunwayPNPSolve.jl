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
includet("debug.jl")


runway_corners = Point3d[
    [  0, -5, 0],
    [  0,  5, 0],
    [100, -5, 0],
    [100,  5, 0]]
runway_corners_far = [3*(runway_corners[3] - runway_corners[1])+runway_corners[1],
                      3*(runway_corners[4] - runway_corners[2])+runway_corners[2]]

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
Label(toggle_grid[1, 2], "Use feature"); Label(toggle_grid[1, 3], "Add noise");
# Set up noise toggles, scenario menu, num pose estimates
feature_toggles = [Toggle(toggle_grid[1+i, 2]; active=true) for i in 1:3]
noise_toggles = [Toggle(toggle_grid[1+i, 3]; active=true) for i in 1:3]
toggle_labels = let labels = ["Near corners", "Far corners", "Edges"]
    [Label(toggle_grid[1+i, 1], labels[i]) for i in 1:3]
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
function project_points(cam_pose::AbstractAffineMap, points::Vector{Point3d})
    cam_transform = PerspectiveMap() ∘ inv(cam_pose)
    projected_points = map(cam_transform, points)
end
dims_2d_to_3d = LinearMap(1.0*I(3)[:, 1:2])
dims_3d_to_2d = LinearMap(1.0*I(3)[1:2, :])
cam_pose_gt = @lift AffineMap(R_t_true, $C_t_true)

"Allow `a, b = Observable((1, 2)) |> unzip_obs`"
function unzip_obs(obs::Observable{<:Tuple})
   Tuple([@lift $obs[i]
          for i in eachindex(obs[])])
end
"Map function over each element in Observable, i.e. mapeach(x->2*x, Observable([1, 2, 3])) == Observable([2, 4, 6])"
mapeach(f, obs::Observable{<:Vector}) = map(el->map(f, el), obs)
# cam_transform = @lift PerspectiveMap() ∘ inv($Cam_translation)
# projected_points = @lift map($cam_transform, runway_corners)
# projected_points_global = @lift map($Cam_translation ∘ AffineMap(dims_2d_to_3d, Float64[0;0;1]), $projected_points)
## plot points projected onto 2D camera plane
function make_perspective_plot(plt_pos, cam_pose::Observable{<:AffineMap})
    # Util
    projected_coords_to_plotting_coords = ∘(
        Point2d,
        dims_3d_to_2d,
        LinearMap(RotZ(1/4*τ)),
        LinearMap(RotY(1/2*τ)),
        dims_2d_to_3d)

    cam_view_ax = Axis(plt_pos, width=250, aspect=DataAspect(), limits=(-1,1,-1,1)./8)

    projective_transform = @lift PerspectiveMap() ∘ inv($cam_pose)
    projected_points = @lift map($projective_transform, runway_corners)
    projected_points_rect = @lift $projected_points[[1, 2, 4, 3, 1]]

    lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, projected_points_rect))
    # plot far points in 2d
    projected_points_far = @lift map($projective_transform, runway_corners_far)
    projected_lines_far = (@lift([$projected_points[1], $projected_points_far[1]]),
                           @lift([$projected_points[2], $projected_points_far[2]]))
    lines!.(cam_view_ax, mapeach.(projected_coords_to_plotting_coords, projected_lines_far);
            color=:gray, linestyle=:dot)
    # plot 1std of Gaussian noise
    meshscatter!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, projected_points),
                  marker=Makie.Circle(Point2d(0,0), 1.0), markersize=σ)
    # Compute and plot line estimates
    ρ, θ = @lift(hough_transform($projected_points)) |> unzip_obs
    # Notice the negative angle, due to the orientatin of the coord system.
    ρ_θ_line_lhs = lift(projected_points, ρ, θ) do ppts, ρ, θ
            p0 = (ppts[1]+ppts[2])/2
            [p0, p0 + ρ[:lhs]*[cos(-θ[:lhs]); sin(-θ[:lhs])]]
    end
    ρ_θ_line_rhs = lift(projected_points, ρ, θ) do ppts, ρ, θ
            p0 = (ppts[1]+ppts[2])/2
            [p0, p0 + ρ[:rhs]*[cos(-θ[:rhs]); sin(-θ[:rhs])]]
    end
    lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, ρ_θ_line_lhs))
    lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, ρ_θ_line_rhs))
end
projections_grid = GridLayout(rhs_grid[2, 1])
make_perspective_plot(projections_grid[1, 1], cam_pose_gt)
pose_guess = Observable(AffineMap(R_t_true, C_t_true[]))
make_perspective_plot(projections_grid[1, 2], pose_guess)
## and of projection
## Set up camera
cam3d!(scene; near=0.01, far=1e9, rotation_center=:eyeposition, cad=true, zoom_shift_lookat=false,
       mouse_rotationspeed = 5f-1,
       mouse_translationspeed = 0.1f0,
       mouse_zoomspeed = 5f-1,
       )
# inspector = DataInspector(scene)
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
projected_points_global = @lift map($cam_pose_gt ∘ Translation([0;0;1]) ∘ dims_2d_to_3d,
                                    project_points($cam_pose_gt, runway_corners))
scatter!(scene, projected_points_global)
# Set cam position
update_cam!(scene.scene, Array(C_t_true[]).-[20.,0,0], Float32[0, 0, 0])
# Compute pose estimates
feature_mask = lift(feature_toggles[1].active, feature_toggles[2].active, feature_toggles[3].active) do a, b, c
    Int[a;b;c]
end
noise_mask = lift(noise_toggles[1].active, noise_toggles[2].active, noise_toggles[3].active) do a, b, c
    Int[a;b;c]
end
Label(toggle_grid[6, 1], "Num pose estimates: ")
num_pose_est_box = Textbox(toggle_grid[6, 2], stored_string = "100",
                       validator = Int, tellwidth = false)
num_pose_est = lift(num_pose_est_box.stored_string) do str
    tryparse(Int, str)
end
opt_traces = []
perturbed_pose_estimates = lift(cam_pose_gt,
                                σ,
                                feature_mask,
                                noise_mask,
                                num_pose_est) do cam_pose_gt, σ, feature_mask, noise_mask, num_pose_est
    projected_points = project_points(cam_pose_gt, runway_corners)
    ρ, θ = hough_transform(projected_points)
    sols = ThreadsX.collect(pnp(runway_corners, projected_points .+ σ*noise_mask[[1,1,2,2]].*[randn(2) for _ in 1:4];
                        rhos  =[ρ[:lhs]; ρ[:rhs]].+σ*noise_mask[3].*randn(2),
                        thetas=[θ[:lhs]; θ[:rhs]].+σ*noise_mask[3].*randn(2),
                        feature_mask=feature_mask,
                        initial_guess = Array(cam_pose_gt.translation)+10.0*randn(3),
                        )
                    for _ in 1:num_pose_est)
    global opt_traces = sols
    pts = (Point3d∘Optim.minimizer).(sols)
    # may filter to contain pose outliers
    # filter(p -> (p[2] ∈ 0±30) && (p[3] ∈ 0..50) && (p[1] ∈ -150..0),
    #        pts) |> collect
    pts
end
pose_samples = scatter!(scene, perturbed_pose_estimates;
                        color=map(x->(x ? :blue : :red), Optim.converged.(opt_traces)))
#
## construct pose estimate errors
errors_obs = lift(cam_pose_gt) do cam_pose_gt
    projected_points = project_points(cam_pose_gt, runway_corners)
    ρ, θ = hough_transform(projected_points)
    local num_pose_est = 100
    log_errs = LinRange(-10:0.5:-5)
    errs = exp.(log_errs)
    function compute_means_and_stds(σ)
        sols = [pnp(runway_corners, projected_points .+ σ.*[randn(2) for _ in 1:4];
                    rhos  =[ρ[:lhs]; ρ[:rhs]].+σ.*randn(2),
                    thetas=[θ[:lhs]; θ[:rhs]].+σ.*randn(2),
                    initial_guess = Array(cam_pose_gt.translation)+10.0*randn(3),
                    )
                for _ in 1:num_pose_est]
        pts = (Point3d∘Optim.minimizer).(filter(Optim.converged, sols))
        Δ = (pts .- cam_pose_gt.translation)
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
on(events(scene).mousebutton, priority = 2) do event
    if event.button == Mouse.left && event.action == Mouse.press
        plt, i = pick(scene)
        @show plt, i
        if !isnothing(plt)
            @show opt_traces[i]
            pose_guess[] = AffineMap(R_t_true, Point3d(Optim.minimizer(opt_traces[i])))
        end
    end
    return Consume(false)
end


# display(GLMakie.Screen(), fig);
# display(GLMakie.Screen(), make_fig_pnp_obj());
fig
