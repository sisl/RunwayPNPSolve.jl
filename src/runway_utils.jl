using DataFrames, XLSX
using Rotations, CoordinateTransformations, Geodesy
using Unitful, Unitful.DefaultSymbols
import Unitful: Length, ustrip, uconvert
import StaticArrays: StaticVector
import Base: zero
# Angle = Union{typeof(1.0°), typeof(1.0rad)};
# Meters = typeof(1.0m)
# ustrip(vec::ENU{Q}) where Q <: Length =
#     ENU{Q.types[1]}(map(ustrip, vec)...)
# ustrip(u::Q, vec::ENU{Q′}) where {Q, Q′<:Unitful.Units} =
#     ENU{Q.types[1]}(map(x->ustrip(Q, x), vec)...)
# # ustrip(pos::ENU{Length}) = ustrip.([pos...]) |> ENU
# # uconvert(u::Unitful.Units, pos::ENU{<:Length}) = uconvert.(u, pos)
# uconvert(u::Unitful.Units, pos::ENU{<:Quantity}) = uconvert.(u, pos)
# uconvert(u::Unitful.Units, map::AffineMap) = AffineMap(map.linear, uconvert.(u, map.translation))
# uconvert(u::Unitful.Units, pt::Point) = uconvert.(u, pt)
# ustrip(map::AffineMap) = AffineMap(map.linear, ustrip.(map.translation))
# # Base.zero(u::T) where T <: Quantity = T(0)
const DATUM=wgs84
# @unit pxl "px" Pixel 0.00345mm false
# Pixels = typeof(1.0pxl)

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

get_unique_runways(runway_identifier;
                   runway_file="./data/2307 A3 Reference Data_v2.xlsx") =
    let df = XLSX.readxlsx(runway_file)["Sheet1"] |> XLSX.eachtablerow |> DataFrame
        df[df.ICAO .== runway_identifier, :]
    end

angle_to_ENU(θ::Angle) = let
    θ = -θ  # flip orientation
    θ = θ + 90°  # orient from x axis (east)
end
function construct_runway_corners(threshold::ENU{T}, width::Length, bearing::Angle;
                                  length=1000m) where T<:Length
    # Bearing is defined in degrees, clockwise, from north.
    # We want it in rad, counterclockwise (i.e. in accordance to z axis), from east (x axis).
    bearing = angle_to_ENU(bearing)

    threshold_far = threshold + length * [cos(bearing); sin(bearing); 0];

    front_left  = threshold     + width/2 * [cos(bearing+90°); sin(bearing+90°); 0]
    front_right = threshold     + width/2 * [cos(bearing-90°); sin(bearing-90°); 0]
    # back_left   = threshold_far + width/2 * [cos(bearing+90°); sin(bearing+90°); 0]
    # back_right  = threshold_far + width/2 * [cos(bearing-90°); sin(bearing-90°); 0]

    return ENU{typeof(1.0m)}[front_left, front_right]  #, back_left, back_right]
end

function project_points(cam_pose::AffineMap{<:Rotation{3, Float64}, <:StaticVector{3, T}},
                        points::Vector{ENU{T}}) where T<:Union{Length, Float64}
    # projection expects z axis to point forward, so we rotate accordingly
    focal_length = 25mm
    pixel_size = 0.00345mm
    scale = focal_length / pixel_size |> upreferred  # solve units, e.g. [mm] / [m]
    cam_transform = cameramap(scale) ∘ inv(LinearMap(RotY(τ/4))) ∘ inv(cam_pose)
    projected_points = map(Point2d ∘ cam_transform, points)
    projected_points = (T <: Quantity ? projected_points .* 1pxl : projected_points)
end

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

    corners::Vector{ENU{Meters}} = vcat([
        let
            construct_runway_corners(thres, width, bearing; length=3000m)
        end
        for (thres, width, bearing) in zip(thresholds,
                                        runways_df[:, "RWY Width (m)"]m,
                                        runways_df[:, "True Bearing" ]°)
    ]...)

    return thresholds, corners
end
