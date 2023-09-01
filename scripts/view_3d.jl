using Revise
using PNPSolve
using PNPSolve.LsqPnP: pnp2
using PNPSolve.LsqPnP.RunwayLib: ImgProj, make_projection_fn
# using CameraModels
using LinearAlgebra
using Rotations
using CoordinateTransformations
using Makie: Pixel as MakiePixel, px as Makie_px, convert as Makie_convert
import Makie.Observables: @map
using GLMakie
using Optim
using ThreadsX
using IntervalSets
using StatsBase
using Unzip
using StructArrays
using Unitful
using Tau
import Unitful: mm, rad
using Geodesy
using GeodesyXYZExt
import Distributions: Chisq, MvNormal

# import Unitful: convert, Unit, Quantity
# Unitful.convert(t::Type{Quantity{T, D, U}}, u::Unit) where {T, D, U} = Unitful.convert(t, convert(Number, u))

"""
Some definitions:

All units in meters. Coordinate system is located in the center front of the runway.
x-axis along runway.
y-axis to the left.
z-axis up.

For now we assume that all runway points are in the x-y plane.
"""
# Point2/3f already exists, also define for double precision
# includet("pnp.jl")
# includet("debug.jl")
# includet("metrics.jl")

const PX_SIZE = 0.00345 * 1e-3  # [m / px]
# SFO measurements
const Δx = 3500.0m
const Δy = 61.0m

runway_corners =
    XYZ{Meters}[[0m, -Δy / 2, 0m], [0m, Δy / 2, 0m], [Δx, -Δy / 2, 0m], [Δx, Δy / 2, 0m]]
runway_corners_far = [
    3 * (runway_corners[3] - runway_corners[1]) + runway_corners[1],
    3 * (runway_corners[4] - runway_corners[2]) + runway_corners[2],
]

R_t_true = RotY(0.0)

fig = Figure()
scene =
    LScene(fig[1, 1], show_axis = false, scenekw = (backgroundcolor = :gray, clear = true))
# Error slider
slidergrid = SliderGrid(
    fig[2, 1],
    (
        label = "Error scale [pxl]",
        range = 0.0:0.25:5,
        startvalue = 1.0,
        format = x -> string(x, " pixels"),
    ),
    (
        label = "Error scale [°]",
        range = 0.0:0.25:5,
        startvalue = 0.5,
        format = x -> string(x, " degrees"),
    ),
)
σ = slidergrid.sliders[1].value
σ_angle = @lift deg2rad($(slidergrid.sliders[2].value))
rhs_grid = GridLayout(fig[1, 2]; tellheight = false)
toggle_grid = GridLayout(rhs_grid[1, 1])
Label(toggle_grid[1, 2], "Use feature");
Label(toggle_grid[1, 3], "Add noise");
# Set up noise toggles, scenario menu, num pose estimates
feature_toggles_ = [Toggle(toggle_grid[1+i, 2]; active = true) for i = 1:3]
noise_toggles_ = [Toggle(toggle_grid[1+i, 3]; active = true) for i = 1:3]
toggle_labels = let labels = ["Near corners", "Far corners", "Edges"]
    [Label(toggle_grid[1+i, 1], labels[i]) for i = 1:3]
end
## Set up scenario, which affects cam position and therefore all the projections
Label(toggle_grid[5, 1], "Scenario:")
scenario_menu = Menu(
    toggle_grid[5, 2];
    options = ["near (300m)", "mid (1000m)", "far (6000m)"],
    default = "far (6000m)",
)
#
C_t_true = lift(scenario_menu.selection) do menu
    menu == "near (300m)" && return XYZ([-300.0m, 0m, 125.0 / 18 * 1m])
    menu == "mid (1000m)" && return XYZ([-1000.0m, 0m, 125.0 / 6 * 1m])
    menu == "far (6000m)" && return XYZ([-6000.0m, 0m, 123.0m])
end
# function project_points(
#     cam_pose::AbstractAffineMap,
#     points::Vector{XYZ{T}},
# ) where {T<:Union{Float64,Meters}}
#     focal_length = 25mm
#     pixel_size = 0.00345mm
#     scale = focal_length / pixel_size |> upreferred  # solve units, i.e. [mm] / [m]
#     cam_transform = cameramap(scale) ∘ LinearMap(RotY(-τ / 4)) ∘ inv(cam_pose)
#     projected_points = map(ImgProj{Float64} ∘ cam_transform ∘ Point3{T}, points)
# end
dims_2d_to_3d = LinearMap(1.0 * I(3)[:, 1:2])
dims_3d_to_2d = LinearMap(1.0 * I(3)[1:2, :])
cam_pose_gt = @lift AffineMap(R_t_true, $C_t_true)

"Allow `a, b = Observable((1, 2)) |> unzip_obs`"
function unzip_obs(obs::Observable{<:Tuple})
    Tuple([@lift $obs[i] for i in eachindex(obs[])])
end
"Map function over each element in Observable, i.e. mapeach(x->2*x, Observable([1, 2, 3])) == Observable([2, 4, 6])"
mapeach(f, obs::Observable{<:AbstractVector}) = map(el -> map(f, el), obs)
# cam_transform = @lift PerspectiveMap() ∘ inv($Cam_translation)
# projected_points = @lift map($cam_transform, runway_corners)
# projected_points_global = @lift map($Cam_translation ∘ AffineMap(dims_2d_to_3d, Float64[0;0;1]), $projected_points)
## plot points projected onto 2D camera plane
function make_perspective_plot(plt_pos, cam_pose::Observable{<:AffineMap}; title = nothing)
    # Util
    projected_coords_to_plotting_coords = ∘(
        ImgProj,
        (p->ustrip.(pxl, p)),
        dims_3d_to_2d,
        LinearMap(RotZ(1 / 4 * τ)),
        LinearMap(RotY(1 / 2 * τ)),
        dims_2d_to_3d,
    )

    # https://docs.google.com/spreadsheets/d/1r2neGh5YUa2e5Ufr7xOfkrC9kr5bqifN5rn2pktkGS0/edit#gid=760597346
    CAM_WIDTH_PX, CAM_HEIGHT_PX = 4096, 3000
    cam_view_ax = Axis(
        plt_pos,
        width = 750 / 2;
        aspect = DataAspect(),
        title = title,
        limits = (
            -CAM_WIDTH_PX // 2,
            CAM_WIDTH_PX // 2,
            -CAM_HEIGHT_PX // 2,
            CAM_HEIGHT_PX // 2,
        ),
    )

    # projective_transform = @lift PerspectiveMap() ∘ inv($cam_pose)
    # projected_points = @lift map($projective_transform, runway_corners)
    projected_points = @lift make_projection_fn($cam_pose).(runway_corners)
    projected_points_rect = @lift $projected_points[[1, 2, 4, 3, 1]]

    lines!(cam_view_ax, @map(projected_coords_to_plotting_coords.(&projected_points_rect)))
    # lines!(cam_view_ax, mapeach(projected_coords_to_plotting_coords, projected_points_rect))
    # plot far points in 2d
    projected_points_far = @lift make_projection_fn($cam_pose).(runway_corners_far)
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
        marker = Makie.Circle(Point2d(0, 0), 1.0),
        markersize = σ,
    )
    # Compute and plot line estimates
    ρ, θ = @lift(hough_transform($projected_points)) |> unzip_obs
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
projections_grid = GridLayout(rhs_grid[2, 1])

pose_guess = Observable(AffineMap(R_t_true, C_t_true[]))
let
    pplot1 = make_perspective_plot(
        projections_grid[1, 1],
        cam_pose_gt;
        title = "Ground truth perspective",
    )
    pplot2 = make_perspective_plot(
        projections_grid[1, 2],
        pose_guess;
        title = "Perturbed perspective",
    )
    linkaxes!(pplot1, pplot2)
end
## and of projection
## Set up camera
cam3d!(
    scene;
    near = 0.01,
    far = 1e9,
    rotation_center = :eyeposition,
    cad = true,
    zoom_shift_lookat = false,
    mouse_rotationspeed = 5.0f-1,
    mouse_translationspeed = 0.1f0,
    mouse_zoomspeed = 5.0f-1,
)
# inspector = DataInspector(scene)
## Draw runway and coordinate system
# Normal runway rectangle
lines!(scene, map(p -> ustrip.(m, p), runway_corners[[1, 2, 4, 3, 1]]))
# Draw 3d runway lines into the distance
# lines!(scene, map(p->ustrip.(m, p), [runway_corners[1], runway_corners_far[1]]); color=:blue)
# lines!(scene, map(p->ustrip.(m, p), [runway_corners[2], runway_corners_far[2]]); color=:blue)
# arrows!(scene, [Point3f(C_t_true), ], [Vec3f([1., 0, 0]), ]; normalize=true, lengthscale=0.5)
arrows!(
    scene,
    fill(Point3d(0, 0, 0), 3),
    Vec3f[[1, 0, 0], [0, 1, 0], [0, 0, 1]] ./ 5;
    arrowsize = Vec3f(0.1, 0.1, 0.2),
)
arrows!(
    scene,  # larger coordinate system
    fill(Point3d(0, -15, 0), 3),
    Vec3f[[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 5;
    arrowsize = Vec3f(2, 2, 3),
)
# Plot runway surface
surface!(
    scene,
    ustrip.(m, getindex.(runway_corners, 1)),
    ustrip.(m, getindex.(runway_corners, 2)),
    ustrip.(m, getindex.(runway_corners, 3)),
)
# Draw lines from corners to camera
corner_lines = [
    lift(C_t_true) do C_t_true
        [p, C_t_true]
    end for p in runway_corners
]
for l in corner_lines
    lines!(scene, mapeach(p -> ustrip.(m, p), l), color = (:black, 0.3))
end
# Draw Projected points
# projected_points_global = @lift map($cam_pose_gt ∘ Translation([0;0;0.0025]) ∘ dims_2d_to_3d,
#                                     project_points($cam_pose_gt, runway_corners)) * PX_SIZE
# scatter!(scene, projected_points_global)
# Set cam position
update_cam!(scene.scene, ustrip.(m, C_t_true[] .- XYZ(20.0m, 0m, 0m)), Float32[0, 0, 0])
# Compute pose estimates
feature_toggles = lift(
    feature_toggles_[1].active,
    feature_toggles_[2].active,
    feature_toggles_[3].active,
) do near, far, angle
    Bool[near, far, angle]
end
noise_toggles = lift(
    noise_toggles_[1].active,
    noise_toggles_[2].active,
    noise_toggles_[3].active,
) do near, far, angle
    Bool[near, far, angle]
end
Label(toggle_grid[6, 1], "Num pose estimates: ")
num_pose_est_box =
    Textbox(toggle_grid[6, 2], stored_string = "100", validator = Int, tellwidth = false)
num_pose_est = lift(num_pose_est_box.stored_string) do str
    tryparse(Int, str)
end
Label(toggle_grid[7, 1], "Correlated noise")
corr_noise_toggle = Toggle(toggle_grid[7, 2]; active = false)
# corr_noise = Observable(false)
Label(toggle_grid[8, 1], "Mean err")
Label(toggle_grid[9, 1], "Std err")
mean_box = Label(toggle_grid[8, 2], "")
std_box = Label(toggle_grid[9, 2], "")
# connect!(corr_noise_toggle.active, corr_noise)

scatter!(scene.scene, mapeach(ustrip, C_t_true); color = :green, markersize = 27)

offdiag_indices(M::AbstractMatrix) = [ι for ι in CartesianIndices(M) if ι[1] ≠ ι[2]]
opt_traces = []
perturbed_pose_estimates = lift(
    cam_pose_gt,
    σ,
    σ_angle,
    feature_toggles,
    noise_toggles,
    num_pose_est,
    corr_noise_toggle.active,
) do cam_pose_gt, σ, σ_angle, feature_toggles, noise_toggles, num_pose_est, corr_noise
    projected_points::Vector{ImgProj{Pixels}} = make_projection_fn(cam_pose_gt).(runway_corners)
    ρ, θ = hough_transform(ustrip.(projected_points))
    # @show rad2deg.([θ[:lhs], θ[:rhs]])
    # @show rad2deg.(θ[:lhs] - θ[:rhs] - τ/2)

    feature_mask = feature_toggles[[1, 1, 2, 2]]
    noise_mask = noise_toggles[[1, 1, 2, 2]]
    make_corr_matrix(dim, offdiag_val) = begin
        Σ = Matrix{Float64}(I(dim))
        Σ[offdiag_indices(Σ)] .= offdiag_val
        Σ
    end

    sample_measurement_noise() =
        let Σ = (corr_noise ? make_corr_matrix(4, 0.9) : Matrix{Float64}(I(4))),
            D = MvNormal(zeros(4), Σ)

            noise_mask .* σ^2 .* Point2.(eachrow([rand(D) rand(D)])) .* 1pxl
        end
    sample_angular_noise() = σ_angle * 1rad * randn()
    sample_pos_noise() = 10 * randn(3) .* 1m
    sols = ThreadsX.collect(
        LsqPnP.pnp(
            runway_corners[feature_mask],
            (projected_points .+ sample_measurement_noise())[feature_mask],
            # (θ[:lhs] - θ[:rhs] - τ / 2) * 1rad + sample_angular_noise(),
            RotY(0.0);
            initial_guess = cam_pose_gt.translation + sample_pos_noise(),
        ) for _ = 1:num_pose_est
    )
    global opt_traces = sols
    pts = Optim.minimizer.(sols)
    # may filter to contain pose outliers
    # filter(p -> (p[2] ∈ 0±30) && (p[3] ∈ 0..50) && (p[1] ∈ -150..0),
    #        pts) |> collect
    pts
end
pose_samples = scatter!(
    scene,
    mapeach(p -> ustrip.(m, p), perturbed_pose_estimates);
    color = map(x -> (x ? :blue : :red), Optim.converged.(opt_traces)),
)
lift(perturbed_pose_estimates, C_t_true) do ps, pos
    μ = mean(getindex.(ps, 1) .- pos[1])
    mean_box.text = "$(round(m, μ; digits=3))"
    # Compute 95% confidence interval for std. Assumes variables are from Normal distribution.
    # See https://en.wikipedia.org/wiki/Standard_deviation#Confidence_interval_of_a_sampled_standard_deviation
    s = std(getindex.(ps, 1))
    n = length(ps)
    k = n - 1
    D = Chisq(k)
    α = 0.05
    lo = √(k * s^2 / quantile(D, 1 - α / 2))
    hi = √(k * s^2 / quantile(D, α / 2))
    CI = Interval(round(m, lo, digits = 3), round(m, hi, digits = 3))
    val = round(m, s, digits = 3)
    std_box.text = "$val ($CI)"
end
#
on(events(scene).mousebutton, priority = 2) do event
    if event.button == Mouse.left && event.action == Mouse.press
        plt, i = pick(scene)
        # @show plt, plt==pose_samples, i
        if !isnothing(plt) && plt == pose_samples
            @info opt_traces[i]
            pose_guess[] =
                AffineMap(R_t_true, XYZ{Meters}(Optim.minimizer(opt_traces[i]) .* 1m))
        end
    end
    return Consume(false)
end
# PNPSolve.make_error_bars_plot(rhs_grid[3, 1])


# display(GLMakie.Screen(), make_fig_pnp_obj());
# display(GLMakie.Screen(), fig);
fig
