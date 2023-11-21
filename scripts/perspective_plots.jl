## plot points projected onto 2D camera plane
function make_perspective_plot(plt_pos, cam_pose::Observable{<:AffineMap}; kwargs...)
    # Util
    projected_coords_to_plotting_coords = ∘(
        ImgProj,
        (p->ustrip.(pxl, p)),
        dims_3d_to_2d,
        # LinearMap(RotZ(1 / 4 * τ)),
        # LinearMap(RotY(1 / 2 * τ)),
        dims_2d_to_3d,
    )

    # https://docs.google.com/spreadsheets/d/1r2neGh5YUa2e5Ufr7xOfkrC9kr5bqifN5rn2pktkGS0/edit#gid=760597346
    CAM_WIDTH_PX, CAM_HEIGHT_PX = 4096, 3000
    cam_view_ax = Axis(
        plt_pos,
        width = 750 / 2;
        aspect = DataAspect(),
        xlabel="pxl",
        ylabel="pxl",
        limits = (
            -CAM_WIDTH_PX // 2,
            CAM_WIDTH_PX // 2,
            -CAM_HEIGHT_PX // 2,
            CAM_HEIGHT_PX // 2,
        ),
        kwargs...
    )

    # projective_transform = @lift PerspectiveMap() ∘ inv($cam_pose)
    # projected_points = @lift map($projective_transform, runway_corners)
    projected_points = @lift project.([$cam_pose], runway_corners)
    projected_points_rect = @lift $projected_points[[1, 2, 3, 4, 1]]

    lines!(cam_view_ax, @map(projected_coords_to_plotting_coords.(&projected_points_rect)))
    # lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, projected_points_rect))
    # plot far points in 2d
    projected_points_far = @lift project.([$cam_pose], runway_corners_far)
    projected_lines_far = (
        @lift([$projected_points[1], $projected_points_far[1]]),
        @lift([$projected_points[2], $projected_points_far[2]])
    )
    lines!.(
        cam_view_ax,
        mapeach.(projected_coords_to_plotting_coords, projected_lines_far);
        color = :gray,
        linestyle = :dot,
    )
    # plot 1std of Gaussian noise
    meshscatter!(
        cam_view_ax,
        mapeach(projected_coords_to_plotting_coords, projected_points),
        marker = Makie.Circle(Point2(0., 0.), 1.0),
        markersize = σ,
    )
    # Compute and plot line estimates
    (; ρ, θ) = @lift(hough_transform($projected_points)) |> unzip_obs
    # Notice the negative angle, due to the orientatin of the coord system.
    ρ_θ_line_lhs = lift(projected_points, ρ, θ) do ppts, ρ, θ
        p0 = (ppts[1] + ppts[2]) / 2
        [p0, p0 + ρ[:lhs] * [cos(-θ[:lhs]); sin(-θ[:lhs])]]
    end
    ρ_θ_line_rhs = lift(projected_points, ρ, θ) do ppts, ρ, θ
        p0 = (ppts[1] + ppts[2]) / 2
        [p0, p0 + ρ[:rhs] * [cos(-θ[:rhs]); sin(-θ[:rhs])]]
    end
    lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, ρ_θ_line_lhs))
    lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, ρ_θ_line_rhs))
    return cam_view_ax
end
