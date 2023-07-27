### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 88ff9edf-045b-4273-8398-32acac3b0ebf
# ╠═╡ show_logs = false
import Pkg; Pkg.activate(); using Revise # don't use Pluto environment

# ╔═╡ 3f9c4bfe-d617-4e7f-8b48-35f4dda75e97
begin
using DataFrames
using GeometryBasics
using StatsBase
using Match
using Tau
using Makie
using GLMakie
using CoordinateTransformations, Rotations
# using Revise
using Distributions
using Plots, StatsPlots
using Random: randn!
using Geodesy
using Unitful, Unitful.DefaultSymbols
using Optim
("Some other imports (hidden).")
end

# ╔═╡ 34e85343-c69d-4fe9-a1e8-ba1afe87482a
import PNPSolve: get_unique_runways, construct_runway_corners, angle_to_ENU,
				 pnp, compute_LLA_rectangle,
				 project_points, Representation,
				 Meters, m, Pixels, pxl, °, DATUM, Length

# ╔═╡ 45178bc8-3d71-4050-a60a-374fb72b54b4
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
            construct_runway_corners(thres, width, bearing)
        end
        for (thres, width, bearing) in zip(thresholds,
                                        runways_df[:, "RWY Width (m)"]m,
                                        runways_df[:, "True Bearing" ]°)
    ]...)

    return thresholds, corners
end

# ╔═╡ 7b870a89-0ced-45f9-8d4e-d8574c122b89
runway="KABQ";

# ╔═╡ c2cb059e-35a4-4cc8-b619-f4b584e37df1
begin 
runways_df::DataFrame = get_unique_runways(
	runway,
	runway_file="/Users/romeovalentin/Documents/PNPSolve/data/2307 A3 Reference Data_v2.xlsx")
origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)

thresholds::Vector{ENU{Meters}}, corners::Vector{ENU{Meters}} =
	compute_thresholds_and_corners_in_ENU(runways_df, origin)
end

# ╔═╡ 64af8590-e4fd-4dae-805f-ecd672ff31ca
# ╠═╡ show_logs = false
begin
    # import PNPSolve: compute_LLA_rectangle
	using LightOSM
	using OSMMakie
	using Tyler

area = compute_LLA_rectangle(origin, (; x=(-1500.0m, 3000.0m),
                                        y=(-1500.0m, 3000.0m)))

if !isfile("KABQ_airport.json")
	download_osm_network(:bbox; # rectangular area
	    area..., # splat previously defined area boundaries
	    network_type = :all, # download motorways
	    save_to_file_location = "KABQ_airport.json"
	);
end

# load as OSMGraph
osm = graph_from_file("KABQ_airport.json";
    graph_type = :light, # SimpleDiGraph
    weight_type = :distance
)

# use min and max latitude to calculate approximate aspect ratio for map projection
autolimitaspect = map_aspect(area.minlat, area.maxlat)

# plot it
fig, ax, plot = osmplot(osm; axis = (; autolimitaspect))

loc = Rect2f(area.minlon, area.minlat, 
	         area.maxlon-area.minlon, area.maxlat-area.minlat)
using Tyler
import Tyler.TileProviders
# tyler = Tyler.Map(loc)
tyler = Tyler.Map(loc; provider=TileProviders.Google())
# osmplot!(tyler.axis, osm)

import MapTiles: project, wgs84, web_mercator, WGS84, WebMercator
project(p::LatLon, from::WGS84, to::WebMercator) = 
	project(Point2f(p.lon, p.lat), wgs84, web_mercator)
function project(p::ENU, origin::LLA, to::WebMercator)
	LatLon2WebMerc(p::LatLon) = project(p, wgs84, web_mercator)
	proj = LatLon2WebMerc ∘ LatLon ∘ LLAfromENU(origin, DATUM)
	proj(p)
end


import MapTiles
let corners = ustrip.(corners)

	Makie.scatter!(tyler.axis, project.(corners, [origin], [web_mercator]);
			   	   color=:red)
end
tyler
end

# ╔═╡ eda68c0e-b64e-4668-922f-4a8ac86b6f3b
md"""
Let's plot the computed corner points into the Google satelite view.
(Note that we don't know the runway length...)
"""

# ╔═╡ 4cf5468e-3bae-4abc-994e-e29c0c10667a
md"""
Let's define the camera pose and compute the camera projections.  
"""

# ╔═╡ e5a830b0-9a02-4d60-8e6c-8211fa97312d
begin
camera_pose = let distance_to_runway=1000m,
                  vertical_angle=1.2°,
				  height=atan(vertical_angle)*distance_to_runway
    @info "height=$(round(Meters, height; digits=2))"
    bearing = angle_to_ENU(runways_df[1, "True Bearing"]°)
    AffineMap(RotZ(bearing), RotZ(bearing)*[-distance_to_runway;0.0m;height])
end

projected_corners′′::Vector{Point{2, Pixels}} = project_points(camera_pose, corners)

make_pnp_sample(; σ::Pixels=5pxl,
                  representation::Representation=NEAR_CORNERS) = let
    idx = @match representation begin
        NEAR_CORNERS => 1:2
        NEAR_FAR_CORNERS => 1:4,
        ALL_CORNERS => eachindex(corners)
    end
    noise = [σ*randn(2) for _ in eachindex(projected_corners′′)]
    pnp(corners[idx], projected_corners′′[idx] + noise[idx], camera_pose.linear;
        initial_guess = camera_pose.linear*ENU(-100.0m, 0m, 10m),
        method=NelderMead()) |> Optim.minimizer
end
end

# ╔═╡ 7a8d633f-8d57-48c6-a681-9ebb1a4b8f90
camera_pose.linear - RotZ(-45°)

# ╔═╡ 3aaf57d2-81bf-4ffe-aeda-c8aeac3cc98f
begin
	import Statistics
import Statistics: corzm, unscaled_covzm, _conj, corzm, cov2cor!
function Statistics.corzm(x::Matrix{T}, vardim::Int=1) where {T}
    c = unscaled_covzm(x, vardim) / oneunit(T)^2
    return cov2cor!(c, collect(sqrt(c[i,i]) for i in 1:min(size(c)...)))
end
end

# ╔═╡ bd5c5c55-a28b-4b77-8f83-96e489c4784f
begin
pos = []
for repr in instances(Representation),
    noise in [5.0pxl, 10.0pxl]
    #
    samples = [make_pnp_sample(; σ=noise, representation=repr)*1m
		       for _ in 1:100]
	error_samples = inv(camera_pose).(samples)
    @show repr, noise
    rnd(x) = round(x; sigdigits=3)
    rnd(x::Length) = round(Meters, x; sigdigits=3)
    display(mean(error_samples) .|> rnd)
    display(std(error_samples) .|> rnd)
    display(cov(stack(error_samples; dims=1)))
    display(cor(stack(error_samples; dims=1)))
	push!(pos, samples)
end
end

# ╔═╡ 484584e2-db47-415e-a138-e24dfdac9af1
ENU(pos[1][1]) |> display

# ╔═╡ fbced025-7241-4aa7-9e6c-76634e6e0b53
begin
Makie.scatter!(tyler.axis, project.(ENU.(pos[5]) .|> ustrip,
								    [origin], [web_mercator]);
               color=:black)
Makie.scatter!(tyler.axis, project.(ENU.(pos[6]) .|> ustrip,
	                                [origin], [web_mercator]);
			   color=:turquoise)
tyler
end

# ╔═╡ Cell order:
# ╠═88ff9edf-045b-4273-8398-32acac3b0ebf
# ╟─3f9c4bfe-d617-4e7f-8b48-35f4dda75e97
# ╠═34e85343-c69d-4fe9-a1e8-ba1afe87482a
# ╟─45178bc8-3d71-4050-a60a-374fb72b54b4
# ╠═7b870a89-0ced-45f9-8d4e-d8574c122b89
# ╠═c2cb059e-35a4-4cc8-b619-f4b584e37df1
# ╠═7a8d633f-8d57-48c6-a681-9ebb1a4b8f90
# ╟─eda68c0e-b64e-4668-922f-4a8ac86b6f3b
# ╟─64af8590-e4fd-4dae-805f-ecd672ff31ca
# ╟─4cf5468e-3bae-4abc-994e-e29c0c10667a
# ╠═e5a830b0-9a02-4d60-8e6c-8211fa97312d
# ╟─3aaf57d2-81bf-4ffe-aeda-c8aeac3cc98f
# ╠═bd5c5c55-a28b-4b77-8f83-96e489c4784f
# ╠═484584e2-db47-415e-a138-e24dfdac9af1
# ╠═fbced025-7241-4aa7-9e6c-76634e6e0b53
