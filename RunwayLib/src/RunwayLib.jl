module RunwayLib
using CoordinateTransformations
import CoordinateTransformations: cameramap
using Rotations
using StaticArraysCore
import StaticArraysCore: StaticVector, FieldVector
import StaticArraysCore: similar_type, Size
using GeometryBasics
using GeodesyXYZExt
using Unitful, Unitful.DefaultSymbols
import Unitful: Units, ustrip, register, promote_unit, unit
using GeometryBasics
import LinearAlgebra: UniformScaling
include("typedefs.jl")
include("projections.jl")
include("runway_utils.jl")

export ImgProj, RunwayCorners, project, project_line
export CamTransform
export pxl
export Point2, Point3
export angle_to_ENU, load_runways, compute_LLA_rectangle, compute_thresholds_and_corners_in_ENU

function __init__()
    Unitful.register(RunwayLib)
end

end # module RunwayLib
