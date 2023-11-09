module RunwayPNPSolve
using Reexport
using Unitful
using Geodesy, GeodesyXYZExt
using Makie: Pixel as MakiePixel, px as Makie_px, convert as Makie_convert
import Makie: @lift
@reexport using LsqPnP

include("metrics.jl")
include("debug.jl")
include("experiments.jl")

export load_runways, construct_runway_corners, angle_to_ENU
export project_points, Representation

export setup_runway!,
    make_alongtrack_distance_df,
    make_alongtrack_distance_df_including_angles,
    plot_alongtrack_distance_errors,
    rank_all_runways,
    experiment_with_using_sideline_angles,
    experiment_with_attitude_noise
export sample_pos_noise,
    offdiag_indices,
    make_corr_matrix,
    sample_measurement_noise,
    sample_angular_noise,
    sample_random_rotation

end # module RunwayPNPSolve
