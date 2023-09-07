module LsqPnP
using RunwayLib
using Rotations
using ComponentArrays
using CoordinateTransformations, Geodesy, GeodesyXYZExt
using LinearAlgebra: dot, norm, I, normalize
using Optim
using ReTest
using Tau
using Roots
using LsqFit
using StaticArrays: StaticVector, MVector, SVector, MArray, FieldVector
using Unitful: ustrip, Length
using Unitful, Unitful.DefaultSymbols
import LsqFit.DiffResults: DiffResult, MutableDiffResult
import LsqFit.ForwardDiff: Dual
import StatsBase: mean
import Base: Fix1
import IntervalSets: (..)

# include("typedefs.jl")
include("derivatives.jl")
include("hough_transform.jl")
include("pnp.jl")
include("pnp_others.jl")

export pnp, hough_transform
export Meters, m, Pixels, pxl, Angle, °, DATUM, Length
export Point2d, Point3d

end # module LsqPnP
