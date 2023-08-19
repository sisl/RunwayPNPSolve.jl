module RunwayLib
using CoordinateTransformations
import CoordinateTransformations: cameramap
using Rotations
using StaticArraysCore
import StaticArraysCore: StaticVector
using GeometryBasics
using GeodesyXYZExt
using Unitful, Unitful.DefaultSymbols
import Unitful: Units, ustrip
using GeometryBasics
import LinearAlgebra: UniformScaling

Angle = Union{typeof(1.0°),typeof(1.0rad)}
Meters = typeof(1.0m)
@unit px "px" Pixel 0.00345mm false
Pixels = typeof(1.0px)
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
for i = 1:3
    idx = filter(!=(i), axes(SVector{3})[1])
    @eval function project(::ProjectionMap{$i}, svec::StaticVector{3,T}) where {T}
        return SVector{2,T}(svec[$idx]) * inv(svec[$i])
    end
end
(pmap::ProjectionMap)(svec::StaticVector{3,T}) where {T} = project(pmap, svec)
cameramap(::Val{N}) where {N} = ProjectionMap{N}()
cameramap(::Val{N}, scale::Number) where {N} =
    LinearMap(UniformScaling(scale)) ∘ ProjectionMap{N}()

const ImgProj{T} = Point2{T}


function make_projection_fn(cam_pose::AffineMap{<:Rotation{3,Float64},XYZ{Meters}})
    scale = let focal_length = 25mm, pixel_size = 0.00345mm / 1px
        focal_length / pixel_size
    end
    cam_transform = cameramap(Val(1), scale) ∘ inv(cam_pose)
    proj(p::XYZ{Meters})::ImgProj{Pixels} =
        (ImgProj{Pixels} ∘ cam_transform ∘ Point3{Meters})(p)
    return proj
end

export ImgProj, make_projection_fn
end # module RunwayLib
