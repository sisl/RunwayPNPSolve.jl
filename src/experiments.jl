using RunwayPNPSolve, RunwayLib               # Here's where the magic happens
using GeodesyXYZExt                           # My library to build a local xyz frame relative to longitude, latitude
using LinearAlgebra: I, normalize             # ------------------
import ProgressBars: ProgressBar              #     ...
import Base: Fix1                             #     ...
import Random: seed!                          #     ...
using Logging                                 #     ...
using DataFrames                              # ------------------
using Rotations                               # deal with rotation matrices
using Transducers                             # parallelize some operations
using Distributions                           # Gaussian, Cauchy, pdf, cdf, quantile, ...
using CoordinateTransformations               # work with coordinate systems, image projections
using Geodesy                                 # work with Longitude, Latitude etc
using Unitful, Unitful.DefaultSymbols         # work with statically typed SI units
import StaticArraysCore: SVector              # Fixed-sized vectors
using StatsBase: winsor                       # replace outliers
import Serialization: serialize, deserialize  # store results from long computations
import Tau: τ                                 # τ = 2π. more reasonable way to count rotations. see https://tauday.com/tau-manifesto
import ComponentArrays: ComponentVector       # indexing into long array by component. a bit special purpose.

import Rotations: RotY
import RunwayLib: AngularQuantity

using AlgebraOfGraphics, CairoMakie

function setup_runway!(; ICAO="KABQ", approach_idx=1)
    # Note: This sets a global variable in the XYZ coordinate system.
    runways_df::DataFrame = let df = load_runways()
        filter(:ICAO => ==(ICAO), df)
    end
    # pair up runway "forward" and "backward" direction
    sort!(runways_df, Symbol("True Bearing"); by=x->x%180)
    # Rotate df rows such that approach_idx comes first. See https://stackoverflow.com/a/59436296.
    @assert approach_idx <= nrow(runways_df)
    rot_indices = mod1.(first(axes(runways_df)) .+ (approach_idx-1),
                        nrow(runways_df))
    runways_df = runways_df[rot_indices, :]

    origin::LLA = LLA(runways_df[1, ["THR Lat", "THR Long", "THR Elev"]]...)
    runway_bearing::AngularQuantity = angle_to_ENU(runways_df[1, "True Bearing"]°)

    # Fix coordinate system to enable ENU->XYZ translation. Global singleton variable!
    GeodesyXYZExt.fixdatum!(Geodesy.wgs84)
    GeodesyXYZExt.fixorigin!(origin)
    GeodesyXYZExt.fixbearing!(ustrip(rad, runway_bearing))

    thresholds::Vector{XYZ{Meters}}, corners::Vector{XYZ{Meters}} = begin
        ts, cs = RunwayLib.compute_thresholds_and_corners_in_ENU(runways_df, origin)
        XYZ.(ts), XYZ.(cs)
    end
    Nc = length(corners)

    return (; thresholds, corners, Nc, origin, )
end


sample_pos_noise() = [100;100;50] .* randn(3) *1m
offdiag_indices(M::AbstractMatrix) = [ι for ι in CartesianIndices(M) if ι[1] ≠ ι[2]]
make_corr_matrix(dim, offdiag_val) = begin
    Σ = Matrix{Float64}(I(dim))
    Σ[offdiag_indices(Σ)] .= offdiag_val
    Σ
end
sample_measurement_noise(N; σ=1.0pxl, correlated_noise=false) =
    let Σ = (correlated_noise ? make_corr_matrix(N, 0.9) : Matrix{Float64}(I(N))),
        D = MvNormal(zeros(N), Σ)
        σ .* Point2.(eachrow([rand(D) rand(D)]))
    end
sample_angular_noise(; σ_angle=deg2rad(1rad)) = σ_angle * randn()
"Randomizes rotation axis direction and samples normally distributed angle around that axis."
function sample_random_rotation(; σ_rot::AngularQuantity=1.0°)
    n = normalize(rand(SVector{3, Float64}))
    angle = σ_rot*randn() |> Fix1(ustrip, rad)
    RotationVec((angle.*n)...)
end

# We need this to pretty-print in a logging call.
# I.e. `@info rand(3,3 )` is ugly, but `@info tostr(rand(3, 3))` will be pretty.
# See https://stackoverflow.com/a/76393774
function tostr(obj)
    io = IOBuffer()
    show(io, "text/plain", obj)
    String(take!(io))
end

# Experiment 1: Alongtrack distance
function make_alongtrack_distance_df(; N_measurements=1000,
                                       distances = (300.0:100:6000.0).*1m,
                                       feature_mask=1:2,
                                       parallel=true,
                                       σ_pxl=1pxl,
                                       σ_rot=1.0°,
                                       correlated_noise=false,
                                       sample_rotations=false,
                                       rotation_noise=false,
                                       runway_args = (;),
                                       kwargs...)
    # distances = (300.0:100:6000.0).*1m
    colnames = [:alongtrack_distance, :err_x, :err_y, :err_z]

    # unpack only the corners
    (; corners) = setup_runway!(; runway_args...)

    function solve_sample(alongtrack_distance)
        camera_pos = let distance_to_runway=alongtrack_distance,
                        vertical_angle=1.2°,
                        crosstrack_angle=0°,
                        crosstrack=atan(crosstrack_angle)*distance_to_runway,
                        height=atan(vertical_angle)*distance_to_runway
            XYZ(-distance_to_runway, crosstrack, height)
        end

        true_camera_rot = (sample_rotations ? sample_random_rotation(; σ_rot) : RotY(0.))
        pred_camera_rot = (rotation_noise ? sample_random_rotation(; σ_rot) * true_camera_rot : true_camera_rot)
        projection_fn = make_projection_fn(AffineMap(true_camera_rot, camera_pos))
        true_pixel_locs = projection_fn.(corners)
        pred_pixel_locs = (
              true_pixel_locs
            + sample_measurement_noise(length(corners); σ=σ_pxl, correlated_noise)
        )[feature_mask]

        pos_estimate = pnp(
            corners,
            pred_pixel_locs,
            feature_mask,
            pred_camera_rot;
            initial_guess = camera_pos + 1.0*sample_pos_noise()).pos
        return pos_estimate - camera_pos
    end

    map_fn = map # (parallel ? Transducers.map : map)
    collect_fn = (parallel ? Transducers.tcollect : collect)
    results = map_fn(distances) do d
        sols = collect_fn(solve_sample(d) for _ = 1:N_measurements)
        sols_mat = stack(sols, dims=1) .|> x->ustrip(m, x)
        DataFrame(
            hcat(-ustrip(m, d)*ones(length(sols)), sols_mat),
            colnames
        )
    end |> splat(vcat)
    @info "[mean ; std] for x,y,z:"
    # for printing only
    let remove_outliers = xs->winsor(xs; prop=0.01),
        compute_mean_std = xs->(mean(xs), std(xs))
        processed_results = map((compute_mean_std ∘ remove_outliers),
                                eachcol(results)[[:err_x, :err_y, :err_z]])
        @info tostr(stack(processed_results))
    end
    return results
end

# basically the same as above but with sideline angles. could be refactored.
function make_alongtrack_distance_df_including_angles( ;
    N_measurements=1000,
    distances = (300.0:100:6000.0).*1m,
    feature_mask=1:2,
    parallel=true,
    σ_pxl=1pxl,
    σ_rot=1.0°,
    σ_angle=1.0°,
    correlated_noise=false,
    sample_rotations=false,
    rotation_noise=false,
    runway_args = (;),
    kwargs...)::DataFrame

    # distances = (300.0:100:6000.0).*1m
    colnames = [:alongtrack_distance, :err_x, :err_y, :err_z]

    # unpack only the corners
    (; corners) = setup_runway!(; runway_args...)

    function solve_sample(alongtrack_distance)
        camera_pos = let distance_to_runway=alongtrack_distance,
                        vertical_angle=1.2°,
                        crosstrack_angle=0°,
                        crosstrack=atan(crosstrack_angle)*distance_to_runway,
                        height=atan(vertical_angle)*distance_to_runway
            XYZ(-distance_to_runway, crosstrack, height)
        end

        true_camera_rot = (!sample_rotations ? RotY(0.) : sample_random_rotation(; σ_rot))
        pred_camera_rot = (!rotation_noise ? true_camera_rot : RotY(0.))
        projection_fn = make_projection_fn(AffineMap(true_camera_rot, camera_pos))
        true_pixel_locs = projection_fn.(corners)
        pred_pixel_locs = (
              true_pixel_locs
            + sample_measurement_noise(length(corners); σ=σ_pxl, correlated_noise)
        )[feature_mask]
        true_angles = let
            (; lhs, rhs) = hough_transform(true_pixel_locs[1:4])[:θ]
            ComponentVector(γ=[lhs, rhs], β=[(rhs+(τ/4)rad)-(lhs-(τ/4)rad), ])
        end
        pred_angles = true_angles[(:γ, )] + ComponentVector(γ=[sample_angular_noise(; σ_angle) for _ in 1:2])

        pos_estimate = pnp(
            corners,
            pred_pixel_locs,
            feature_mask,
            pred_camera_rot;
            angles = pred_angles,
            initial_guess = camera_pos + 1.0*sample_pos_noise(),
            components=[:x, :y, :γ]
        ).pos

        return pos_estimate - camera_pos
    end

    map_fn = map # (parallel ? ThreadsX.map : map)
    collect_fn = (parallel ? ThreadsX.collect : collect)
    results = map_fn(distances) do d
        sols = collect_fn(solve_sample(d) for _ = 1:N_measurements)
        sols_mat = stack(sols, dims=1) .|> x->ustrip(m, x)
        DataFrame(
            hcat(-ustrip(m, d)*ones(length(sols)), sols_mat),
            colnames
        )
    end |> splat(vcat)
    @info "[mean ; std] for x,y,z:"
    # for printing only
    let remove_outliers = xs->winsor(xs; prop=0.01),
        compute_mean_std = xs->(mean(xs), std(xs))
        processed_results = map((compute_mean_std ∘ remove_outliers),
                                eachcol(results)[[:err_x, :err_y, :err_z]])
        @info tostr(stack(processed_results))
    end
    return results
end

# requirements found in MPVS: https://docs.google.com/document/d/1W5QZ-fLEq-X_ftKTutadUVN_TLgp3MP7EX2913VAR9M/edit#heading=h.3ylonubdepc
const alongtrack_reqs = (;
    x = [-6000, 0],
    y = [370, 10])
const height_reqs = (;
    x = [-6000, -4500, -1450, -860, -280],
    y = [33.41, 25.46, 9.30, 6.07, 2.26])
const crosstrack_reqs = (;
    x = [-6000 , -4500 , -1450 , -860  , -280  ],
    y = [66.55 , 55.5  , 33.0  , 18.4  , 11.0  ])

function plot_alongtrack_distance_errors(; features=(feature_mask=(1:2), feature_str="Near only"),
                                           N_measurements=100, draw_requirements=false,
                                           extra_fname="",
                                           df=nothing,  # can either provide or recompute
                                           savefig=false,
                                           runway_args=(;),
                                           kwargs...)
    seed!(1)
    (;feature_mask, feature_str) = features
    fig = Figure()
    df = (isnothing(df) ? make_alongtrack_distance_df(; N_measurements, feature_mask, runway_args, kwargs...) : df)
    df = stack(df, [:err_x, :err_y, :err_z];
               variable_name=:err_axis, value_name=:err_value)
    plt = data(df)
    plt *= mapping(:alongtrack_distance => "Alongtrack distance [m]", :err_value => "Estimation error [m]",
                   color=:err_axis, row=:err_axis)
    plt *= visual(BoxPlot; width=100, markersize=5, show_outliers=false)

    # use `draw!` to avoid drawing legend
    draw!(fig, plt;
         facet=(; linkyaxes = :none),
         axis=(yminorgridcolor=(:gray, 0.5), ygridcolor=(:black, 0.5), xgridcolor=(:gray, 0.5),
               yticks=Makie.LinearTicks(5))
    )
    if draw_requirements
        for (row, data) in enumerate([alongtrack_reqs, crosstrack_reqs, height_reqs])
            lines!(fig[row, 1], data.x, data.y; color=:green)
            lines!(fig[row, 1], data.x, -data.y; color=:green)
        end
    end

    # See https://github.com/MakieOrg/AlgebraOfGraphics.jl/issues/331#issuecomment-1654825941
    Label(fig[-1, 1:1], "Estimation errors over alongtrack distance";
          fontsize=16, font=:bold)
    Label(fig[0, 1:1], "Features=($(feature_str))";
          fontsize=16, font=:italic)
    resizetocontent!(fig)

    feature_str = let
        feature_str = lowercase(replace(feature_str, ' '=>'_'))
        if (:approach_idx in keys(kwargs))
            feature_str = string(feature_str, "_approach=$(kwargs[:approach_idx])")
        end
        feature_str = string(feature_str, "_", extra_fname)
    end
    if savefig
        save("figs/distance_variation_$(feature_str).png", fig; px_per_unit = 3)
        save("figs/distance_variation_$(feature_str).svg", fig)
        @info "figs/distance_variation_$(feature_str).png"
    end
    fig
end

# this is pretty crappy, only used during processing by hand...
function fetch_data_from_info_stream(str)
    exp = r"\[ Info: ([0-9]+\.0°)\n.+\n2×3 Matrix{Float64}:\n\s*(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)\n\s?(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)"

    df = DataFrame(σ_angle=String[], σ_x=Float64[], σ_y=Float64[], σ_z=Float64[])

    for str_ ∈ Base.Iterators.partition(split(str, '\n'), 5)
       str_ = join(str_, '\n')
       m = match(exp, str_)
       if isnothing(m)
           @warn str_
           continue
       else
           let m = m.captures
               push!(df, [m[1],
                   parse(Float64, m[5]),
                   parse(Float64, m[6]),
                   parse(Float64, m[7])])
           end
       end
    end
    df[:, :id] .= axes(df)[1]
    dfs = stack(df, [:σ_angle, :σ_x, :σ_y, :σ_z])
    # dfs = stack(df)
    unstack(dfs, :variable, :id, :value)
end


function rank_all_runways(; features=(; feature_mask=(:), feature_string="(:)"),
                            limit::Union{Nothing, Int}=nothing)
    all_runways_df::DataFrame = load_runways()
    icaos = unique(all_runways_df.ICAO)
    if !isnothing(limit)
      icaos=icaos[1:limit]
    end

    min_errs = Dict()
    max_errs = Dict()

    for icao in ProgressBar(icaos)
        try
            n_approaches = nrow(filter(:ICAO => ==(icao),  all_runways_df))
            all_approaches = map(1:n_approaches) do idx
                runway_args = (; ICAO=icao, approach_idx = idx)
                results_df = with_logger(SimpleLogger(Logging.Warn)) do  # ignore the info loggings
                    make_alongtrack_distance_df(;
                        feature_mask,
                        N_measurements=1_000,
                        distances=(6000:6000).*1m,
                        runway_args)
                end
                std.(eachcol(results_df))[2:end]
            end |> stack  # results in 3xn_approaches matrix
            min_errs[icao] = minimum(abs.(all_approaches); dims=2)[:]
            max_errs[icao] = maximum(abs.(all_approaches); dims=2)[:]
        catch e
           if isa(e, InterruptException)
               @warn "Interrupted"; break
           else
              @warn "Couldn't process $icao"
           end
        end
    end

    serialize("./min_max_errors_results_$(features.feature_string)", (; min_errs, max_errs))
end

robuststd(xs; prop=0.01) = std(winsor(xs; prop))

function experiment_with_using_sideline_angles()
    errs = [0.0, 0.01, 0.1, 0.3, 0.5, 1.0] .* 1°
    results = map(errs) do σ_angle
        df = make_alongtrack_distance_df_including_angles(;
            distances=(6000:500:6000).*1m,
            N_measurements=1000,
            σ_angle)
        errs = robuststd.(eachcol(df)[[:err_x, :err_y, :err_z]])
        DataFrame([Quantity[σ_angle], [errs[1]], [errs[2]], [errs[3]]],
                  [:σ_angle, :σ_x, :σ_y, :σ_z])
    end |> splat(vcat)
    results[:, :id] .= axes(results)[1]
    results = stack(results, [:σ_angle, :σ_x, :σ_y, :σ_z])
    # resultss = stack(results)
    unstack(results, :variable, :id, :value)
end

function experiment_with_attitude_noise(; feature_mask=(1:2))
    errs = [0.0, 0.01, 0.1, 0.3, 0.5, 1.0] .* 1°
    results = map(errs) do σ_angle
        df = make_alongtrack_distance_df(;
            feature_mask,
            distances=(6000:500:6000).*1m,
            N_measurements=2000,
            sample_rotations=false,
            rotation_noise=true,
            σ_rot=σ_angle
            )
        errs = robuststd.(eachcol(df)[[:err_x, :err_y, :err_z]])
        DataFrame([Quantity[σ_angle], [errs[1]], [errs[2]], [errs[3]]],
                  [:σ_angle, :σ_x, :σ_y, :σ_z])
    end |> splat(vcat)
    results[:, :id] .= axes(results)[1]
    results = stack(results, [:σ_angle, :σ_x, :σ_y, :σ_z])
    # resultss = stack(results)
    unstack(results, :variable, :id, :value)
end
