using Unitful, Unitful.DefaultSymbols
import Unitful: Units, ustrip
using GeometryBasics
import Geodesy: ENU
Point2d = Point2{Float64}
Vec2d = Vec2{Float64}
Point3d = Point3{Float64}
Vec3d = Vec3{Float64}

Angle = Union{typeof(1.0°), typeof(1.0rad)};
Meters = typeof(1.0m)
@unit pxl "px" Pixel 0.00345mm false
Pixels = typeof(1.0pxl)

Unitful.ustrip(pos::ENU{Q}) where Q <: Quantity =
    ENU{Q.types[1]}(ustrip.(pos))
Unitful.ustrip(u::Units, pos::ENU{Q}) where Q <: Quantity =
    ENU{Q.types[1]}(ustrip.(u, pos))

Unitful.ustrip(pos::Point{N, Q}) where {N, Q <: Quantity} =
    Point{N, Q.types[1]}(ustrip.(pos))
Unitful.ustrip(u::Units, pos::Point{N, Q}) where {N, Q <: Quantity} =
    Point{N, Q.types[1]}(ustrip.(u, pos))
