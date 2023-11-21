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
    idx = filter(!=(N), axes(svec, 1))
    proj = svec[idx] * inv(svec[N])
    #or, (about 50% faster)
    # proj = svec[vcat(1:(N-1), (N+1):end)] * inv(svec[N])
    return proj
end
function (pmap::ProjectionMap)(svec::StaticVector{3,T}) where {T}
    N = getaxis(pmap)
    projectionmap_(Val(N), svec)
end
cameramap(::Val{N}) where {N} = ProjectionMap{N}()
cameramap(::Val{N}, scale::Number) where {N} =
    LinearMap(UniformScaling(scale)) ∘ ProjectionMap{N}()

CamTransform = AffineMap

function project(cam_pose::CamTransform{<:Rotation{3}, <:XYZ{<:WithUnits(m)}},
                 world_point::XYZ{<:WithUnits(m)})::ImgProj
    scale = let focal_length = 25mm, pixel_size = 0.00345mm / 1pxl
        focal_length / pixel_size
    end
    cam_transform = cameramap(Val(1), scale) ∘ inv(cam_pose)  # first axis aligned with direction of view
    orient_coord_sys =  LinearMap([-1 0 ; 0 1])  # flip x, leave y
    transform = (ImgProj ∘ orient_coord_sys ∘ cam_transform)
    transform(world_point)
end

# # get derivatives of pose
# derivative((@_ ustrip(pxl, norm(__)) ∘ project(pose, pt + XYZ(__*m, 0.0m, 0.0m))), 1.0)
# gradient((@_ ustrip(pxl, norm(__)) ∘ project(pose, pt + XYZ(__*m))), zeros(3))
# jacobian((@_ ustrip.(pxl, __) ∘ project(pose, pt + XYZ(__*m))), zeros(3))
#
# # get derivatives of orientation
# let new_pose(θ) = AffineMap(RotZ(θ)*pose.linear, pose.translation)
#     derivative((@_ ustrip(pxl, norm(__)) ∘ project(new_pose(__), pt)), 0.)
# end
# let new_pose(θs) = AffineMap(RotYZ(θs...)*pose.linear, pose.translation)
#     gradient((@_ ustrip(pxl, norm(__)) ∘ project(new_pose(__), pt)), zeros(2))
# end
# let new_pose(θs) = AffineMap(RotYZ(θs...)*pose.linear, pose.translation)
#   jacobian((@_ ustrip.(pxl, __) ∘ project(new_pose(__), pt)), zeros(2))
# end
#
#
# pose = AffineMap(RotY(0.), XYZ(-100.0m, 0.0m, 0.0m))
# pts = [XYZ(1.0m, 2.0m, 3.0m), XYZ(1.0m, -2.0m, 3.0m)];
# projections = project.([pose], pts);
# measurements = projections .+ [3*randn(2)pxl for _ in eachindex(projections)]
# loss(loc) = begin
#   simulated_projections = project.([AffineMap(pose.linear, XYZ(loc*m))], pts)
#   res = sum((lhs, rhs)::Tuple->norm(lhs - rhs), zip(measurements, simulated_projections))
#   ustrip(pxl, res)
# end
# loss(loc, p) = loss(loc);
#
# using Optimization
# using OptimizationOptimJL
# prob = OptimizationProblem(OptimizationFunction(loss, Optimization.AutoForwardDiff()), MVector(-10., 0., 0.))
# solve(prob, NewtonTrustRegion())


# using Optimization
# using OptimizationOptimJL
# using NonlinearSolve
function solve_once(pt2_x::T) where T
    pose = AffineMap(RotY(0.), XYZ(-100.0m, 0.0m, 0.0m))
    pts = [XYZ(1.0m, 2.0m, 3.0m) + XYZ(pt2_x*m, 0.0m, 0.0m),
           XYZ(1.0m, -2.0m, 3.0m)];
    projections = project.([pose], pts);
    measurements = projections .+ [3*randn(2)pxl for _ in eachindex(projections)]
    loss(loc) = begin
        simulated_projections = project.([AffineMap(pose.linear, XYZ(loc*m))], pts)
        res = sum((lhs, rhs)::Tuple->norm(lhs - rhs), zip(measurements, simulated_projections))
        ustrip(pxl, res)
    end
    loss(loc, p) = loss(loc);

    func = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    # prob = NonlinearLeastSquaresProblem(func, MVector{3, T}(-90. * one(T), 0. * one(T), 0. * one(T)))
    prob = OptimizationProblem(func, MVector{3, T}(-90. * one(T), 0. * one(T), 0. * one(T)))
    # solve(prob, LevenbergMarquardt())
    solve(prob, ConjugateGradient(), reltol=1e-14, allow_f_increases=true, maxiters=10_000)
end
# ForwardDiff.derivative(solve_once, 0.0)
