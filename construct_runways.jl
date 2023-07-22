using GeometryBasics
using StatsBase
using Match
using Tau
using LightOSM
using Makie
using OSMMakie
using GLMakie
using Tyler
using CoordinateTransformations, Rotations

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

runways_df::DataFrame = get_unique_runways("KABQ")
origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)
thresholds::Vector{ENU{Meters}} = [
    let
        thres = ENUfromLLA(origin, DATUM)(
            LLA(row[["THR Lat", "THR Long", "THR Elev"]]...)
        )
        ENU([thres...]m)  # assign unit "meter"
    end
    for row in eachrow(runways_df)
]

all_corners = vcat([
    let
        construct_runway_corners(thres, width, bearing)
    end
    for (thres, width, bearing) in zip(thresholds,
                                       runways_df[:, "RWY Width (m)"]m,
                                       runways_df[:, "True Bearing" ]°)
]...)

camera_pose = let distance_to_runway=1000m
    bearing = angle_to_ENU(runways_df[1, "True Bearing"]°)
    AffineMap(RotZ(bearing), RotZ(bearing)*[-distance_to_runway;0.0m;50m])
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
# function project_points(cam_pose::AffineMap{R, <:StaticVector{3, Float64}},
#                         points::Vector{<:StaticVector{3, Float64}}) where R
#     # projection expects z axis to point forward, so we rotate accordingly
#     focal_length = 25mm
#     pixel_size = 0.00345mm
#     scale = focal_length / pixel_size |> upreferred  # solve units if necessary, i.e. [mm] / [m]
#     cam_transform = cameramap(scale) ∘ inv(LinearMap(RotY(τ/4))) ∘ inv(cam_pose)
#     projected_points = map(Point2d ∘ cam_transform, points)
# end
projected_corners = project_points(camera_pose, all_corners)

using BenchmarkTools
for method in [NelderMead(), NewtonTrustRegion()],
    use_noise = [false, true]
    noise = use_noise * [5*randn(2) for _ in eachindex(projected_corners)]
    println("Use noise: $use_noise. Method: $method.")
    display(
      @benchmark pnp(all_corners, projected_corners + noise, camera_pose.linear;
                     initial_guess = ENU(-100.0m, -100m, 0m),
                     method=$method)
    )
end

@enum Representation begin
  NEAR_CORNERS
  NEAR_AND_FAR_CORNERS
  ALL_CORNERS
end

make_sample(; σ::Pixels=5pxl,
              representation::Representation=NEAR_CORNERS) = let
    idx = @match representation begin
        NEAR_CORNERS => 1:2
        NEAR_FAR_CORNERS => 1:4,
        ALL_CORNERS => eachindex(all_corners)
    end
    # idx = eachindex(all_corners)
    noise = [σ*randn(2) for _ in eachindex(projected_corners)]
    pnp(all_corners[idx], projected_corners[idx] + noise[idx], camera_pose.linear;
        initial_guess = ENU(-100.0m, -100m, 0m),
        method=NelderMead()) |> Optim.minimizer
end

for repr in instances(Representation),
    noise in [5.0pxl, 10.0pxl]
    samples = inv(LinearMap(camera_pose.linear)).(
        [make_sample(; σ=noise, representation=repr)*1m - camera_pose.translation for _ in 1:1000]);
    @show repr, noise
    rnd(x) = round(x; sigdigits=3)
    rnd(x::Length) = round(Meters, x; sigdigits=3)
    display(mean(samples) .|> rnd)
    display(std(samples) .|> rnd)
end


# for 2d plotting
using Permutations
Base.:*(p::Permutation, arr::AbstractArray) = Matrix(p)*arr
projection_to_plotting_coords() = LinearMap(Permutation([2;1])) ∘ LinearMap(RotMatrix2(τ/2))
projections_in_image_pane = projection_to_plotting_coords().(projected_corners)
fig = Figure();
CAM_WIDTH_PX, CAM_HEIGHT_PX = 4096, 3000
cam_view_ax = Axis(fig[1, 1], width=2000;
                          aspect=DataAspect(),
                          limits=(-CAM_WIDTH_PX//2, CAM_WIDTH_PX//2, -CAM_HEIGHT_PX//2, CAM_HEIGHT_PX//2))
Makie.scatter!(cam_view_ax, Point2d.(projections_in_image_pane))
fig


area = compute_LLA_rectangle(origin, (; x=(-500.0m, 3000.0m),
                                        y=(-500.0m, 3000.0m)))
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
fig, ax, plot = osmplot(osm; axis = (; autolimitaspect))

loc = Rect2f(area.minlon, area.minlat, area.maxlon-area.minlon, area.maxlat-area.minlat)
using Tyler
import Tyler.TileProviders
# tyler = Tyler.Map(loc)
tyler = Tyler.Map(loc; provider=TileProviders.Google())
# osmplot!(tyler.axis, osm)

import MapTiles
tyler_corners = LLAfromENU(origin, DATUM).(all_corners .|> ustrip) .|> LatLon;
tyler_corners_latlon = map(c->Point2f(c.lon, c.lat), tyler_corners)
tyler_corners_web_mercator = MapTiles.project.(tyler_corners_latlon, [MapTiles.wgs84], [MapTiles.web_mercator])
scatter!(tyler.axis, tyler_corners_web_mercator)
