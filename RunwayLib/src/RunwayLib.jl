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
import Unitful: Units, ustrip, register
using GeometryBasics
import LinearAlgebra: UniformScaling

Angle = Union{typeof(1.0°),typeof(1.0rad)}
Meters = typeof(1.0m)
@unit pxl "pxl" Pixel 0.00345mm false
Pixels = typeof(1.0pxl)
"""
    ProjectionMap{N}

    Redefine PerspectiveMap to set which dimension is "pointing outwards".
    In CoordinateTransformations, it's the z-axis (i.e. N=3), but we usually want N=1.

    We generate a projection for each of the axes pointing forward. E.g. we can call
    ```julia
    pm = ProjectionMap{1}()
    relative_point = SVector(1.,2., 3.)
    pm(relaltive_point)
    ```
    to get the projection with x-axis forward.
"""
struct ProjectionMap{N} end
getaxis(::ProjectionMap{N}) where {N} = N
function (pmap::ProjectionMap)(svec::StaticVector{3,T}) where {T}
    N = getaxis(pmap)
    idx = filter(!=(N), axes(svec)[1])
    proj = (svec[idx]) * inv(svec[N])
    return ImgProj(proj)
end
cameramap(::Val{N}) where {N} = ProjectionMap{N}()
cameramap(::Val{N}, scale::Number) where {N} =
    LinearMap(UniformScaling(scale)) ∘ ProjectionMap{N}()


struct ImgProj{T} <: FieldVector{2, T}
    x
    y
end
similar_type(::Type{A}, ::Type{T}, s::Size{S}) where {A<:ImgProj, T, S} = ImgProj{T}
# const ImgProj{T} = Point2{T}
# function (Point2)(p::Point2{T}) where {T}
#     ImgProj{T}(p)
# end

"Function to produce a NamedTuple type.

Similar to a struct, but call with () instead of {} for type param. Benefit: We can broadcast over `values(tpl)`.
"
RunwayCorners(::Type{T}) where {T<:Union{XYZ{Meters},ImgProj{Pixels}}} = @NamedTuple begin
    near_left::T
    near_right::T
    far_left::T
    far_right::T
end

function make_projection_fn(cam_pose::AffineMap{<:Rotation{3,Float64},<:XYZ})
    scale = let focal_length = 25mm, pixel_size = 0.00345mm / 1pxl
        focal_length / pixel_size
    end
    cam_transform = cameramap(Val(1), scale) ∘ inv(cam_pose)
    function proj(p::XYZ{T})::ImgProj where T
        (ImgProj ∘ cam_transform ∘ Point3{T})(p)
    end
    function proj(rc::RunwayCorners(XYZ{Meters}))::RunwayCorners(ImgProj{Pixels})
        RunwayCorners(ImgProj)(proj.(values(rc)))
    end
    return proj
end


function __init__()
    Unitful.register(RunwayLib)
end

export ImgProj, RunwayCorners, make_projection_fn
export Angle, Meters, Pixels, pxl
end # module RunwayLib
