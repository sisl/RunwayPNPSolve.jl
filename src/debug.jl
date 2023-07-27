
function make_fig_pnp_obj()
    fig = Figure()
    projected_points = @lift project_points($cam_pose_gt, runway_corners)
    ρ, θ = @lift(hough_transform($projected_points))|> unzip_obs
    pnp_obj = @lift build_pnp_objective(
                        runway_corners, $projected_points .+ $σ*$noise_mask[[1,1,2,2]].*[randn(2) for _ in 1:4];
                        rhos  =[$ρ[:lhs]; $ρ[:rhs]].+$σ*$noise_mask[3].*randn(2),
                        thetas=[$θ[:lhs]; $θ[:rhs]].+$σ*$noise_mask[3].*randn(2),
                        feature_mask=$feature_mask,
                    )
    ax = LScene(fig[1, 1], show_axis=true)
    #
    xs = @lift $cam_pose_gt.translation[1] .+ LinRange(-10, 15, 1001)
    ys = @lift $cam_pose_gt.translation[2] .+ LinRange(-40, 40, 1001)
    zs = @lift $cam_pose_gt.translation[3] .+ LinRange( -20,  20, 1001)
    #
    # vol = @lift [$pnp_obj([x, y, z]) for x∈$xs, y∈$ys, z∈$zs];
    # plt = contour!(ax, xs, ys, zs, vol;
    #                levels=10,
    #                transparency=true)
    # vol = @lift [$pnp_obj([x, y, $cam_pose_gt.translation[3]]) for x∈$xs, y∈$ys];
    # plt = contour!(ax, xs, ys, vol;
    #                levels=10,
    #                transparency=true)
    vol = @lift [$pnp_obj([$cam_pose_gt.translation[1], y, z]) for y∈$ys, z∈$zs];
    plt = surface!(ax, ys, zs, @lift log.($vol);
                   levels=50,
                   transparency=true)
    fig
end
