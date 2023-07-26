using GeometryBasics
using StatsBase
using Match
using Tau
using Makie
using GLMakie
using CoordinateTransformations, Rotations
using Revise
using Distributions
using Plots, StatsPlots
using Random: randn!
includet("../BayesianPnP/scripts/2d_poc.jl")
includet("/Users/romeovalentin/Documents/PnPExperiments/runway_utils.jl")
includet("/Users/romeovalentin/Documents/PnPExperiments/pnp.jl")

runways_df::DataFrame = get_unique_runways("KABQ")
origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)
function compute_thresholds_and_corners_in_ENU(
        runways_df::DataFrame,
        origin::LLA)::Tuple{Vector{ENU{Meters}}, Vector{ENU{Meters}}}
    thresholds::Vector{ENU{Meters}} = [
        let
            thres = ENUfromLLA(origin, DATUM)(
                LLA(row[["THR Lat", "THR Long", "THR Elev"]]...)
            )
            ENU([thres...]m)  # assign unit "meter"
        end
        for row in eachrow(runways_df)
    ]

    corners = vcat([
        let
            construct_runway_corners(thres, width, bearing)
        end
        for (thres, width, bearing) in zip(thresholds,
                                        runways_df[:, "RWY Width (m)"]m,
                                        runways_df[:, "True Bearing" ]°)
    ]...)

    return thresholds, corners
end
thresholds, corners = compute_thresholds_and_corners_in_ENU(runways_df, origin)

camera_pose = let distance_to_runway=1000m
    bearing = angle_to_ENU(runways_df[1, "True Bearing"]°)
    AffineMap(RotZ(bearing), RotZ(bearing)*[-distance_to_runway;0.0m;50m])
end

projected_corners′′::Vector{Point{2, Pixels}} = project_points(camera_pose, corners)

make_sample(; σ::Pixels=5pxl,
              representation::Representation=NEAR_CORNERS) = let
    idx = @match representation begin
        NEAR_CORNERS => 1:2
        NEAR_FAR_CORNERS => 1:4,
        ALL_CORNERS => eachindex(corners)
    end
    noise = [σ*randn(2) for _ in eachindex(projected_corners′′)]
    pnp(corners[idx], projected_corners′′[idx] + noise[idx], camera_pose.linear;
        initial_guess = ENU(-100.0m, -100m, 0m),
        method=NelderMead()) |> Optim.minimizer
end

import Statistics
import Statistics: corzm, unscaled_covzm, _conj, corzm, cov2cor!
function Statistics.corzm(x::Matrix{T}, vardim::Int=1) where {T}
    c = unscaled_covzm(x, vardim) / oneunit(T)^2
    return cov2cor!(c, collect(sqrt(c[i,i]) for i in 1:min(size(c)...)))
end

for repr in instances(Representation),
    noise in [5.0pxl, 10.0pxl]
    #
    error_samples = inv(LinearMap(camera_pose.linear)).(
        [make_sample(; σ=noise, representation=repr)*1m - camera_pose.translation for _ in 1:1000]);
    @show repr, noise
    rnd(x) = round(x; sigdigits=3)
    rnd(x::Length) = round(Meters, x; sigdigits=3)
    display(mean(error_samples) .|> rnd)
    display(std(error_samples) .|> rnd)
    display(cov(stack(error_samples; dims=1)))
    display(cor(stack(error_samples; dims=1)))
end

rnd(x) = round(x; sigdigits=3)
rnd(x::Length) = round(Meters, x; sigdigits=3)
for repr in instances(Representation),
    noise in [5.0pxl]
    #
    @show repr, noise
    error_samples = inv(LinearMap(camera_pose.linear)).(
        [make_sample(; σ=noise, representation=repr)*1m - camera_pose.translation for _ in 1:1000]);
    display(std(error_samples) .|> rnd)
end

function make_bayesian_sample(; σ::Pixels=5pxl,
                                representation::Representation=NEAR_CORNERS)
    idx = @match representation begin
        NEAR_CORNERS => 1:2
        NEAR_FAR_CORNERS => 1:4,
        ALL_CORNERS => eachindex(corners)
    end
    noise = [Point2(σ*randn(2)...) for _ in eachindex(projected_corners′′)]
    prod([
        compute_bayesian_pose_estimate(corners[i],
                                       camera_pose.linear,
                                       projected_corners′′[i],
                                       noise[i])
        for i in idx])
end

for repr in instances(Representation),
    noise in [5.0pxl, 10.0pxl]
    #
    res = make_bayesian_sample(; σ=noise, representation=repr) - (camera_pose.translation |> ustrip)
    @show repr, noise
    display(res)
end

res = make_bayesian_sample(; σ=5.0pxl)
inv(LinearMap(camera_pose.linear))(res.μ)
let M = inv(camera_pose.linear)
    (M*res).Σ
end

# compare two approaches
let noise=5.0pxl, repr=NEAR_CORNERS
    error_samples = let
        samples = [make_sample(; σ=noise, representation=repr)*1m
                   for _ in 1:1_000]
        projected_samples = inv(camera_pose).(samples)
        stack(projected_samples; dims=1)
    end
    error_bayesian = let D = make_bayesian_sample(; σ=noise),
                         camera_pose = ustrip(camera_pose),
                         D′ = inv(camera_pose)(D)
        inv(camera_pose)(D)
    end
    plts =
       [let
            p = Plots.plot(; title="Axis=$ax")
            Plots.histogram!(p, error_samples[:, i]; normalize=true)
            Plots.plot!(p, Distributions.Gaussian(0, √(error_bayesian.Σ[i, i])))
            Plots.plot!(p, Distributions.Gaussian(0, 5*√(error_bayesian.Σ[i, i])))
            p
        end
        for (i, ax) in enumerate(["x", "y", "z"])
    ];
    Plots.plot(plts...)
end



using BenchmarkTools
for method in [NelderMead(), NewtonTrustRegion()],
    use_noise = [false, true]
    noise = use_noise * [5*randn(2) for _ in eachindex(projected_corners′′)]
    println("Use noise: $use_noise. Method: $method.")
    display(
      @benchmark pnp(corners, projected_corners′′ + noise, camera_pose.linear;
                     initial_guess = ENU(-100.0m, -100m, 0m),
                     method=$method)
    )
end

# for 2d plotting
using Permutations
Base.:*(p::Permutation, arr::AbstractArray) = Matrix(p)*arr
projection_to_plotting_map = LinearMap(Permutation([2;1])) ∘ LinearMap(RotMatrix2(τ/2))
projections_in_image_pane = projection_to_plotting_coords.(projected_corners′′)
fig = Figure();
CAM_WIDTH_PX, CAM_HEIGHT_PX = 4096, 3000
cam_view_ax = Axis(fig[1, 1], width=2000;
                          aspect=DataAspect(),
                          limits=(-CAM_WIDTH_PX//2, CAM_WIDTH_PX//2, -CAM_HEIGHT_PX//2, CAM_HEIGHT_PX//2))
Makie.scatter!(cam_view_ax, Point2d.(projections_in_image_pane))
fig


# area = (
#     minlat = 51.5015, minlon = -0.0921, # bottom left corner
#     maxlat = 51.5154, maxlon = -0.0662 # top right corner
# )

