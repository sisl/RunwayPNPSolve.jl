using PNPSolve, RunwayLib
using GeodesyXYZExt
using Rotations
using DataFrames
using ThreadsX
using LinearAlgebra: I
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

runway="KABQ";
runways_df::DataFrame = get_unique_runways(
    runway,
    runway_file="/Users/romeovalentin/Documents/PNPSolve/data/2307 A3 Reference Data_v2.xlsx")
sort!(runways_df, Symbol("True Bearing");
      by=x->x%180)  # pair up runway "forward" and "backward" direction
origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)
runway_bearing = angle_to_ENU(runways_df[1, "True Bearing"]°)
GeodesyXYZExt.fixdatum!(Geodesy.wgs84)
GeodesyXYZExt.fixorigin!(origin)
GeodesyXYZExt.fixbearing!(ustrip(rad, runway_bearing))

thresholds::Vector{XYZ{Meters}}, corners::Vector{XYZ{Meters}} = begin
    ts, cs = PNPSolve.compute_thresholds_and_corners_in_ENU(runways_df, origin)
    XYZ.(ts), XYZ.(cs)
end


camera_pos = let distance_to_runway=6000m,
                  vertical_angle=1.2°,
                  crosstrack_angle=0°,
                  crosstrack=atan(crosstrack_angle)*distance_to_runway,
                  height=atan(vertical_angle)*distance_to_runway

    XYZ(-distance_to_runway, crosstrack, height)
end
camera_rot = RotY(0.)
camera_pose = AffineMap(camera_rot, camera_pos)
projection_fn = make_projection_fn(camera_pose)

pixel_locs = projection_fn.(corners)
sample_pos_noise() = 10 * randn(3) .* 1m
sample_measurement_noise(N; σ=1.0pxl) =
    let Σ = Matrix{Float64}(I(N)),
        D = MvNormal(zeros(N), Σ)
        σ .* Point2.(eachrow([rand(D) rand(D)]))
    end

Nc = length(corners)
# feature_mask = [1, 2]
feature_mask = 1:Nc
sols = ThreadsX.collect(
    pnp(corners[feature_mask],
        (pixel_locs+sample_measurement_noise(length(corners)))[feature_mask],
        camera_rot;
        initial_guess = camera_pos + sample_pos_noise()).pos
    for _ = 1:1_000
)
@info std.(eachcol(stack(sols, dims=1)))
