using DataFrames, XLSX
using Geodesy
using Unitful, Unitful.DefaultSymbols
import Unitful: Length, ustrip
Angle = Union{typeof(1.0°), typeof(1.0rad)};

using LightOSM
using OSMMakie
using GLMakie

fname = "./data/2307 A3 Reference Data_v2 (1).xlsx"
df = XLSX.readxlsx(fname)["Sheet1"] |> XLSX.eachtablerow |> DataFrame
df = df[df.ICAO.=="KABQ", :]

p1 = LLA(df[1, "THR Lat"], df[1, "THR Long"], df[1, "THR Elev"])
p2 = LLA(df[2, "THR Lat"], df[2, "THR Long"], df[2, "THR Elev"])
p3 = LLA(df[3, "THR Lat"], df[3, "THR Long"], df[3, "THR Elev"])

# tested, empirically seems correct.
trans = ENUfromLLA(p1, wgs84)
trans(p2)
trans(p3)

const DATUM=wgs84
function compute_LLA_rectangle(origin::LLA{<:Real}, rect::@NamedTuple{x::Tuple{T, T},
                                                                      y::Tuple{T, T}}) where T<:Real
    rect = let transform_back = LLAfromENU(origin, DATUM),
               origin = ENUfromLLA(origin, DATUM)

        bottom_left = ENU(rect.x[1], rect.y[1], 0)
        top_right = ENU(rect.x[2], rect.y[2], 0)
        transform_back.([bottom_left, top_right])
    end
    return (; minlat=rect[1].lat, minlon=rect[1].lon,
              maxlat=rect[2].lat, maxlon=rect[2].lon)
end

function get_unique_runways(runway_identifier; runway_file="./data/2307 A3 Reference Data_v2.xlsx")
    runways = let
        df = XLSX.readxlsx(runway_file)["Sheet1"] |> XLSX.eachtablerow |> DataFrame
        df[df.ICAO .== runway_identifier, :]
    end
    # Most runways are provided in the two opposite directions.
    # We only want one direction.
    runways_unique_direction = let
        bearings_180 = runways[:, "True Bearing"] .% 180.  # "project" e.g. left-to-right and right-to-left onto the same orientation
        unique_indices = unique(i->round(bearings_180[i]; digits=0),
                                eachindex(bearings_180))
        runways[unique_indices, :]
    end
end
runways_df::DataFrame = get_unique_runways("KABQ")
origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)
thresholds::Vector{ENU{Length}} = [
    let
        thres = ENUfromLLA(origin, DATUM)(
            LLA(row[["THR Lat", "THR Long", "THR Elev"]]...)
        )
        ENU([thres...]*1m)  # assign unit "meter"
    end
    for row in eachrow(runways_df)
]

function construct_runway_corners(threshold::ENU{Length}, width::Length, bearing::Angle;
                                  length=1000m)
    # Bearing is defined in degrees, clockwise, from north.
    # We want it in rad, counterclockwise (i.e. in accordance to z axis), from east (x axis).
    angle_to_ENU(θ::Angle) = let
        θ = -θ  # flip orientation
        θ = θ + 90°  # orient from x axis (east)
    end
    bearing = angle_to_ENU(bearing)

    threshold_far = threshold + length * [cos(bearing); sin(bearing); 0];

    front_left  = threshold     + width/2 * [cos(bearing+90°); sin(bearing+90°); 0]
    front_right = threshold     + width/2 * [cos(bearing-90°); sin(bearing-90°); 0]
    back_left   = threshold_far + width/2 * [cos(bearing+90°); sin(bearing+90°); 0]
    back_right  = threshold_far + width/2 * [cos(bearing-90°); sin(bearing-90°); 0]

    return [front_left, front_right, back_left, back_right]
end
all_corners = vcat([
        construct_runway_corners(thres, width, bearing)
        for (thres, width, bearing) in zip(thresholds, df[:, "RWY Width (m)"].*1m, df[:, "True Bearing"].*1°)
    ]...)
ustrip(pos::ENU{Length}) = ustrip(pos) |> ENU

area = compute_LLA_rectangle(p1, (; x=(-500, 3000), y=(-500, 3000)))
# area = (
#     minlat = 51.5015, minlon = -0.0921, # bottom left corner
#     maxlat = 51.5154, maxlon = -0.0662 # top right corner
# )


download_osm_network(:bbox; # rectangular area
    area..., # splat previously defined area boundaries
    network_type = :all, # download motorways
    save_to_file_location = "KABQ_airport.json"
);
download_osm_buildings(:bbox;
    area...,
    metadata = true,
    download_format = :osm,
    save_to_file_location = "KABQ_buildings.osm",
);
buildings = buildings_from_file("KABQ_buildings.osm");

# load as OSMGraph
osm = graph_from_file("KABQ_airport.json";
    graph_type = :light, # SimpleDiGraph
    weight_type = :distance
)

# load as Buildings Dict

osm_london = graph_from_file("london.json";
    graph_type = :light, # SimpleDiGraph
    weight_type = :distance
)

# use min and max latitude to calculate approximate aspect ratio for map projection
autolimitaspect = map_aspect(area.minlat, area.maxlat)

# plot it
# fig, ax, plot = osmplot(osm)
fig, ax, plot = osmplot!(osm; axis = (; autolimitaspect))

loc = Rect2f(area.minlon, area.minlat, area.maxlon-area.minlon, area.maxlat-area.minlat)
tyler = Tyler.Map(loc)
osmplot!(tyler.axis, osm)
