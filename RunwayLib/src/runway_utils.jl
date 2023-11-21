using DataFrames, XLSX
using Rotations, CoordinateTransformations, Geodesy
using Unitful, Unitful.DefaultSymbols
import Unitful: Length, ustrip, uconvert
import StaticArraysCore: StaticVector
import Base: zero  # to get the additive identity of any type
const DATUM=wgs84

@enum Representation begin
  NEAR_CORNERS
  NEAR_AND_FAR_CORNERS
  ALL_CORNERS
end

function compute_LLA_rectangle(origin::LLA{<:Real}, rect::@NamedTuple{x::Tuple{T, T},
                                                                      y::Tuple{T, T}}) where T<:Length
    rect = let transform_back = LLAfromENU(origin, DATUM),
               origin = ENUfromLLA(origin, DATUM)

        # type takes care of unit conversion
        bottom_left = ENU{typeof(1.0m)}(rect.x[1], rect.y[1], 0m) |> ustrip
        top_right   = ENU{typeof(1.0m)}(rect.x[2], rect.y[2], 0m) |> ustrip
        transform_back.([bottom_left, top_right])
    end
    return (; minlat=rect[1].lat, minlon=rect[1].lon,
              maxlat=rect[2].lat, maxlon=rect[2].lon)
end

function load_runways(runway_file=joinpath(pkgdir(RunwayLib), "data", "2307 A3 Reference Data_v2.xlsx"))
    XLSX.readxlsx(runway_file)["Sheet1"] |> XLSX.eachtablerow |> DataFrame
end

"""Takes an angle given as 'bearing', i.e. with zero pointing north and positive direction clockwise,
 and translates it to an ENU angle, i.e. with zero pointing east and positive direction counter-clockwise."""
angle_to_ENU(θ::AngularQuantity) = let
    θ = -θ  # flip orientation
    θ = θ + 90°  # orient from x axis (east)
end

function construct_runway_corners(threshold::ENU{T}, width::Length, bearing::AngularQuantity) where T<:Length
    # Bearing is defined in degrees, clockwise, from north.
    # We want it in rad, counterclockwise (i.e. in accordance to z axis), from east (x axis).
    bearing = angle_to_ENU(bearing)

    front_left  = threshold + width/2 * [cos(bearing+90°); sin(bearing+90°); 0]
    front_right = threshold + width/2 * [cos(bearing-90°); sin(bearing-90°); 0]

    return ENU{Meters}[front_left, front_right]  #, back_left, back_right]
end

function compute_thresholds_and_corners_in_ENU(
        runways_df::DataFrame,
        origin::LLA)::Tuple{Vector{ENU{Meters}}, Vector{ENU{Meters}}}
    thresholds::Vector{ENU{Meters}} = [
        thres = ENUfromLLA(origin, DATUM)(
            LLA(row[["THR Lat", "THR Long", "THR Elev"]]...)
        ) * 1m
        for row in eachrow(runways_df)
    ]

    corners::Vector{ENU{Meters}} = vcat([
        construct_runway_corners(thres, width, bearing)
        for (thres, width, bearing) in zip(thresholds,
                                        runways_df[:, "RWY Width (m)"]m,
                                        runways_df[:, "True Bearing" ]°)
    ]...)

    return thresholds, corners
end
