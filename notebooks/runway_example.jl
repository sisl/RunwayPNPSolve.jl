### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 88ff9edf-045b-4273-8398-32acac3b0ebf
# ╠═╡ show_logs = false
import Pkg; Pkg.activate("RunwayExample", shared=true);  # don't use Pluto environment

# ╔═╡ 30728793-341c-4898-8ff4-c2cf547c40fc
Pkg.develop(path="/Users/romeovalentin/Documents/PNPSolve"); using Revise

# ╔═╡ 3f9c4bfe-d617-4e7f-8b48-35f4dda75e97
begin
using DataFrames
# using GeometryBasics
using StatsBase
using Match
using Tau
using Makie
using GLMakie
using CoordinateTransformations, Rotations
# using Revise
using Distributions
#using Plots, StatsPlots
using Random: randn!
using Geodesy
using Unitful, Unitful.DefaultSymbols
using Optim
("Some other imports (hidden).")
end

# ╔═╡ ad33c971-4068-42a5-9ed9-b04a09617981
using TileProviders

# ╔═╡ 012d1b6b-34c9-41e5-92b1-edaddb40e2b1
Pkg.status()

# ╔═╡ 53891808-62ad-467b-b1d6-3ef33931aa4a
import Tyler

# ╔═╡ 08d4b51f-6550-4082-9a28-950a8b16a553
import MapTiles: project, wgs84, web_mercator, WGS84, WebMercator; import MapTiles

# ╔═╡ 34e85343-c69d-4fe9-a1e8-ba1afe87482a
import PNPSolve: get_unique_runways, construct_runway_corners, angle_to_ENU,
				 pnp, compute_LLA_rectangle, compute_thresholds_and_corners_in_ENU,
				 project_points, Representation,
				 Meters, m, Pixels, pxl, °, DATUM, Length


# ╔═╡ d17f6850-521a-4af2-954c-38c6606919f3
md"""
Let's start by selecting the airport.
"""

# ╔═╡ 7b870a89-0ced-45f9-8d4e-d8574c122b89
runway="KABQ";

# ╔═╡ 75fd79b1-a39f-4a99-b0a7-6e24dd339feb
md"""
Next, we select one of the runways, and compute the origin, bearing, and all the runway corners.
"""

# ╔═╡ c2cb059e-35a4-4cc8-b619-f4b584e37df1
begin 
runways_df::DataFrame = get_unique_runways(
	runway,
	runway_file="/Users/romeovalentin/Documents/PNPSolve/data/2307 A3 Reference Data_v2.xlsx")
origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)
runway_bearing = angle_to_ENU(runways_df[1, "True Bearing"]°)
thresholds::Vector{ENU{Meters}}, corners::Vector{ENU{Meters}} =
	compute_thresholds_and_corners_in_ENU(runways_df, origin)
end

# ╔═╡ eda68c0e-b64e-4668-922f-4a8ac86b6f3b
md"""
Let's plot the computed corner points into the Google satelite view.
(Note that we don't know the runway length...)
"""

# ╔═╡ 64af8590-e4fd-4dae-805f-ecd672ff31ca
# ╠═╡ show_logs = false
begin
#using LightOSM
#using OSMMakie

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
#osm = graph_from_file("KABQ_airport.json";
#    graph_type = :light, # SimpleDiGraph
#    weight_type = :distance
#)

# use min and max latitude to calculate approximate aspect ratio for map projection
#autolimitaspect = map_aspect(area.minlat, area.maxlat)

# plot it
# fig, ax, plot = osmplot(osm; axis = (; autolimitaspect))

loc = Rect2f(area.minlon, area.minlat, 
	         area.maxlon-area.minlon, area.maxlat-area.minlat)
#using Tyler
#import Tyler.TileProviders
# tyler = Tyler.Map(loc)
tyler = Tyler.Map(loc; provider=TileProviders.Google())
# osmplot!(tyler.axis, osm)

#import MapTiles: project, wgs84, web_mercator, WGS84, WebMercator
MapTiles.project(p::LatLon, from::WGS84, to::WebMercator) = 
	MapTiles.project(Point2f(p.lon, p.lat), wgs84, web_mercator)
function MapTiles.project(p::ENU, origin::LLA, to::WebMercator)
	LatLon2WebMerc(p::LatLon) = project(p, wgs84, web_mercator)
	proj = LatLon2WebMerc ∘ LatLon ∘ LLAfromENU(origin, DATUM)
	proj(p)
end

let corners = ustrip.(corners)
	Makie.scatter!(tyler.axis, project.(corners, [origin], [web_mercator]);
				   color=:red)
end

#import MapTiles
tyler
end

# ╔═╡ 4cf5468e-3bae-4abc-994e-e29c0c10667a
md"""
Let's define the camera pose in ENU (i.e. wrt the origin).
"""

# ╔═╡ e5a830b0-9a02-4d60-8e6c-8211fa97312d
camera_pose = let distance_to_runway=6000m,
                  vertical_angle=1.2°,
				  crosstrack_angle=3°,
				  crosstrack=atan(crosstrack_angle)*distance_to_runway,
				  height=atan(vertical_angle)*distance_to_runway
    @info "height=$(round(Meters, height; digits=2))"
	@info "crosstrack=$(round(Meters, crosstrack; digits=2))"
    AffineMap(RotZ(runway_bearing), 
		      RotZ(runway_bearing)*[-distance_to_runway;crosstrack;height])
end;

# ╔═╡ 73541eb9-6d34-4a54-88f2-8292fa633aa7
md"""
Next, let's set up a function that takes pixel error and choice of representation, and outputs a single pose estimate.
"""

# ╔═╡ 1b48ae08-c4eb-4d6d-abae-bb09d95c522b
make_pnp_sample(; σ::Pixels=5pxl,
                  representation::Representation=NEAR_CORNERS) = let
    idx = @match representation begin
        NEAR_CORNERS => 1:2
        NEAR_FAR_CORNERS => 1:4,
        ALL_CORNERS => eachindex(corners)
    end
	projected_corners′′::Vector{Point{2, Pixels}} = 
		project_points(camera_pose, corners)
    noise = [σ*randn(2) for _ in eachindex(projected_corners′′)]
    pnp(corners[idx], projected_corners′′[idx] + noise[idx], camera_pose.linear;
        initial_guess = camera_pose.linear*ENU(-1000.0m, 0m, 10m),
        method=NelderMead()) |> Optim.minimizer
	#pnp2(ustrip.([m], corners[idx]), ustrip.(pxl, projected_corners′′[idx] + noise[idx]), camera_pose.linear)
end

# ╔═╡ 3aaf57d2-81bf-4ffe-aeda-c8aeac3cc98f
begin
	import Statistics
import Statistics: corzm, unscaled_covzm, _conj, corzm, cov2cor!
function Statistics.corzm(x::Matrix{T}, vardim::Int=1) where {T}
    c = unscaled_covzm(x, vardim) / oneunit(T)^2
    return cov2cor!(c, collect(sqrt(c[i,i]) for i in 1:min(size(c)...)))
end
md"""
(Hidden.) To compute the statistics, we need to fix a "bug" in the Statistics stdlib.
"""
end

# ╔═╡ bd5c5c55-a28b-4b77-8f83-96e489c4784f
begin
pos = []
for repr in instances(Representation),
    noise in [1.0pxl] #, 10.0pxl]
    #
    samples = [make_pnp_sample(; σ=noise, representation=repr)*1m
		       for _ in 1:100]
	error_samples = inv(camera_pose).(samples)
    @show repr, noise
    rnd(x) = round(x; sigdigits=3)
    rnd(x::Length) = round(Meters, x; sigdigits=3)
    @info (mean(error_samples) .|> rnd)
    @info (std(error_samples) .|> rnd)
    #@info (cov(stack(error_samples; dims=1)))
    #@info (cor(stack(error_samples; dims=1)))
	push!(pos, samples)
end
end

# ╔═╡ fbced025-7241-4aa7-9e6c-76634e6e0b53
begin
Makie.scatter!(tyler.axis, project.(ENU.(pos[1]) .|> ustrip,
								    [origin], [web_mercator]);
               color=:blue)
Makie.scatter!(tyler.axis, project.(ENU.(pos[2]) .|> ustrip,
	                                [origin], [web_mercator]);
			   color=:green)
	Makie.scatter!(tyler.axis, project.(ENU.(pos[3]) .|> ustrip,
	                                [origin], [web_mercator]);
			   color=:orange)

tyler.figure
end

# ╔═╡ 2ed36ce3-67a6-4935-b956-080ec1d35f14
display(tyler.figure)

# ╔═╡ Cell order:
# ╠═88ff9edf-045b-4273-8398-32acac3b0ebf
# ╠═012d1b6b-34c9-41e5-92b1-edaddb40e2b1
# ╠═30728793-341c-4898-8ff4-c2cf547c40fc
# ╠═3f9c4bfe-d617-4e7f-8b48-35f4dda75e97
# ╠═53891808-62ad-467b-b1d6-3ef33931aa4a
# ╠═08d4b51f-6550-4082-9a28-950a8b16a553
# ╟─34e85343-c69d-4fe9-a1e8-ba1afe87482a
# ╟─d17f6850-521a-4af2-954c-38c6606919f3
# ╠═7b870a89-0ced-45f9-8d4e-d8574c122b89
# ╟─75fd79b1-a39f-4a99-b0a7-6e24dd339feb
# ╠═c2cb059e-35a4-4cc8-b619-f4b584e37df1
# ╟─eda68c0e-b64e-4668-922f-4a8ac86b6f3b
# ╠═ad33c971-4068-42a5-9ed9-b04a09617981
# ╠═64af8590-e4fd-4dae-805f-ecd672ff31ca
# ╟─4cf5468e-3bae-4abc-994e-e29c0c10667a
# ╠═e5a830b0-9a02-4d60-8e6c-8211fa97312d
# ╟─73541eb9-6d34-4a54-88f2-8292fa633aa7
# ╠═1b48ae08-c4eb-4d6d-abae-bb09d95c522b
# ╟─3aaf57d2-81bf-4ffe-aeda-c8aeac3cc98f
# ╠═bd5c5c55-a28b-4b77-8f83-96e489c4784f
# ╠═fbced025-7241-4aa7-9e6c-76634e6e0b53
# ╠═2ed36ce3-67a6-4935-b956-080ec1d35f14
