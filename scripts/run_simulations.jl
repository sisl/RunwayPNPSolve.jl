using PNPSolve, RunwayLib
using GeodesyXYZExt
using Rotations
using DataFrames
using ThreadsX
using LinearAlgebra: I, normalize
import ProgressBars: ProgressBar
import Base: Fix1
# using GeometryBasics
# using StatsBase
# using Match
# using Tau
# using Makie
using Distributions
# using Random: randn!
using CoordinateTransformations
using Geodesy
using Unitful, Unitful.DefaultSymbols

using AlgebraOfGraphics, CairoMakie

function setup_runway!(; ICAO="KABQ", approach_idx=1)
    # Note: This sets a global variable in the XYZ coordinate system.
    runways_df::DataFrame = let df = load_runways()
        filter(:ICAO => ==(ICAO), df)
    end
    sort!(runways_df, Symbol("True Bearing"); by=x->x%180)  # pair up runway "forward" and "backward" direction
    origin::LLA = LLA(runways_df[approach_idx, ["THR Lat", "THR Long", "THR Elev"]]...)
    runway_bearing = angle_to_ENU(runways_df[approach_idx, "True Bearing"]°)


    # Fix coordinate system to enable ENU->XYZ translation. Global singleton variable!
    GeodesyXYZExt.fixdatum!(Geodesy.wgs84)
    GeodesyXYZExt.fixorigin!(origin)
    GeodesyXYZExt.fixbearing!(ustrip(rad, runway_bearing))

    thresholds::Vector{XYZ{Meters}}, corners::Vector{XYZ{Meters}} = begin
        ts, cs = PNPSolve.compute_thresholds_and_corners_in_ENU(runways_df, origin)
        XYZ.(ts), XYZ.(cs)
    end

    Nc = length(corners)

    return (; thresholds, corners, Nc, origin, )
end


# camera_pos = let distance_to_runway=6000m,
#                   vertical_angle=1.2°,
#                   crosstrack_angle=0°,
#                   crosstrack=atan(crosstrack_angle)*distance_to_runway,
#                   height=atan(vertical_angle)*distance_to_runway

#     XYZ(-distance_to_runway, crosstrack, height)
# end

# camera_rot = RotY(0.)
# camera_pose = AffineMap(camera_rot, camera_pos)
sample_pos_noise() = 50 * randn(3) .* 1m
sample_measurement_noise(N; σ=1.0pxl) =
    let Σ = Matrix{Float64}(I(N)),
        D = MvNormal(zeros(N), Σ)
        σ .* Point2.(eachrow([rand(D) rand(D)]))
    end
"Randomizes rotation axis direction and samples normally distributed angle around that axis."
function sample_random_rotation(; σ_rot::Angle=1.0°)
    n = normalize(rand(SVector{3, Float64}))
    angle = σ_rot*randn() |> Fix1(ustrip, rad)
    RotationVec((angle.*n)...)
end

# # feature_mask = [1, 2]
# feature_mask = 1:Nc
# sols = ThreadsX.collect(let camera_rot = sample_random_rotation(; σ_rot=10.0°),
#                             projection_fn = make_projection_fn(AffineMap(camera_rot,
#                                                                          camera_pos)),
#                             pixel_locs = projection_fn.(corners)
#     pnp(corners[feature_mask],
#         (pixel_locs+sample_measurement_noise(length(corners)))[feature_mask],
#         camera_rot;
#         initial_guess = camera_pos + sample_pos_noise()).pos
#     end
#     for _ = 1:1_000
# )
# std.(eachcol(stack(sols, dims=1))) |> display

# Experiment 1: Alongtrack distance
function make_alongtrack_distance_df(; N_measurements=1000,
                                       feature_mask=1:2,
                                       kwargs...)
    distances = (300.0:100:6000.0).*1m
    colnames = [:alongtrack_distance, :err_x, :err_y, :err_z]

    @show kwargs
    (; corners) = setup_runway!(; kwargs...)

    function solve_sample(alongtrack_distance)
        camera_pos = let distance_to_runway=alongtrack_distance,
                        vertical_angle=1.2°,
                        crosstrack_angle=0°,
                        crosstrack=atan(crosstrack_angle)*distance_to_runway,
                        height=atan(vertical_angle)*distance_to_runway
            XYZ(-distance_to_runway, crosstrack, height)
        end

        camera_rot = RotY(0.)
        projection_fn = make_projection_fn(AffineMap(camera_rot, camera_pos))
        pixel_locs = projection_fn.(corners)

        pnp(corners[feature_mask],
            (pixel_locs+sample_measurement_noise(length(corners)))[feature_mask],
            camera_rot;
            initial_guess = camera_pos + sample_pos_noise()).pos - camera_pos
    end

    results = ThreadsX.map(distances) do d
        sols = [solve_sample(d) for _ = 1:N_measurements]
        sols_mat = stack(sols, dims=1) .|> x->ustrip(m, x)
        DataFrame(
            hcat(-ustrip(m, d)*ones(length(sols)), sols_mat),
            colnames
        )
    end |> splat(vcat)
    return results
end

const alongtrack_reqs = (;
    x = [-6_000, 0],
    y = [370, 10])
const height_reqs = (;
    x = [-6000, -4500, -1450, -860, -280],
    y = [33.41, 25.46, 9.30, 6.07, 2.26])
const crosstrack_reqs = (;
    x = [-6000 , -4500 , -1450 , -860  , -280  ],
    y = [66.55 , 55.5  , 33.0  , 18.4  , 11.0  ])

function plot_alongtrack_distance_errors(; features=(feature_mask=(1:2), feature_str="Near only"),
                                           N_measurements=100, draw_requirements=false,
                                           kwargs...)
    seed!(1)
    (;feature_mask, feature_str) = features
    fig = Figure()
    df = make_alongtrack_distance_df(; N_measurements, feature_mask, kwargs...)
    df = stack(df, [:err_x, :err_y, :err_z];
               variable_name=:err_axis, value_name=:err_value)
    plt = data(df)
    plt *= mapping(:alongtrack_distance => "Alongtrack distance [m]", :err_value => "Estimation error [m]",
                   color=:err_axis, row=:err_axis)
    plt *= visual(BoxPlot; width=100, markersize=5)
    draw!(fig, plt;
         facet=(; linkyaxes = :none),
         axis=(yminorgridvisible=true, ygridvisible=true, yminorgridcolor=(:gray, 0.5), ygridcolor=(:black, 0.5),
               xminorgridvisible=true, xgridvisible=true, xgridcolor=(:gray, 0.5),
               yticks=Makie.LinearTicks(5),
               ))
    if draw_requirements
        for (row, data) in enumerate([alongtrack_reqs, crosstrack_reqs, height_reqs])
            lines!(fig[row, 1], data.x, data.y; color=:green)
            lines!(fig[row, 1], data.x, -data.y; color=:green)
        end
    end

    # fig = draw(plt;
    #      facet=(; linkyaxes = :none),
    #      axis=(yminorgridvisible=true, ygridvisible=true, yminorgridcolor=(:red, 0.5), ygridcolor=(:gray, 0.5),
    #            xminorgridvisible=true, xgridvisible=true,
    #            ))
    Label(fig[0, 1:1], "Estimation errors over alongtrack distance ($(feature_str))";
            fontsize=16, font=:bold)
    features_str = lowercase(replace(feature_str, ' '=>'_'))
    save("figs/distance_variation_$(feature_str).png", fig; px_per_unit = 3)
    save("figs/distance_variation_$(feature_str).svg", fig)
    fig
end
