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
function projectionmap_(::Val{N}, svec::StaticVector{3,T}) where {N, T}
    idx = filter(!=(N), first(axes(svec)))
    proj = svec[idx] * inv(svec[N])
    return proj
end
function (pmap::ProjectionMap)(svec::StaticVector{3,T}) where {T}
    N = getaxis(pmap)
    projectionmap_(Val(N), svec)
end
cameramap(::Val{N}) where {N} = ProjectionMap{N}()
cameramap(::Val{N}, scale::Number) where {N} =
    LinearMap(UniformScaling(scale)) ∘ ProjectionMap{N}()


function make_projection_fn(cam_pose::AffineMap{<:Rotation{3,Float64},<:XYZ})
    scale = let focal_length = 25mm, pixel_size = 0.00345mm / 1pxl
        focal_length / pixel_size
    end
    cam_transform = cameramap(Val(1), scale) ∘ inv(cam_pose)
    function proj(p::XYZ{T})::ImgProj where T
        (ImgProj ∘ cam_transform)(p)
    end
    function proj(rc::RunwayCorners(XYZ{Meters}))::RunwayCorners(ImgProj{Pixels})
        RunwayCorners(ImgProj)(proj.(values(rc)))
    end
    return proj
end
