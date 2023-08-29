module LsqPnP
using RunwayLib
using Unitful
include("typedefs.jl")
include("derivatives.jl")
include("pnp.jl")

export pnp, hough_transform
export Meters, m, Pixels, pxl, Angle, °, DATUM, Length
export Point2d, Point3d

# function __init__()
#     Unitful.register(LsqPnP)
# end

end # module LsqPnP
