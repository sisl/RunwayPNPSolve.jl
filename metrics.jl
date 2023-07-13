## construct pose estimate errors
function make_error_bars_plot(error_plots_grid)
    errors_obs = lift(cam_pose_gt) do cam_pose_gt
        projected_points = project_points(cam_pose_gt, runway_corners)
        ρ, θ = hough_transform(projected_points)
        local num_pose_est = 100
        log_errs = LinRange(-10:0.5:-5)
        errs = exp.(log_errs)
        function compute_means_and_stds(σ)
            sols = [pnp(runway_corners, projected_points .+ σ.*[randn(2) for _ in 1:4];
                        rhos  =[ρ[:lhs]; ρ[:rhs]].+σ.*randn(2),
                        thetas=[θ[:lhs]; θ[:rhs]].+σ_angle.*randn(2),
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
end
