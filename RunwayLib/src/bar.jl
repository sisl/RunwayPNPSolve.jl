# using Optimization
# using OptimizationOptimJL
# run ] dev Unitful GeodesyXYZExt .
# run ] add SimpleNonlinearSolve ForwardDiff CoordinateTransformations Rotations StaticArraysCore ForwardDiff
using Rotations, CoordinateTransformations, NonlinearSolve, Unitful, Unitful.DefaultSymbols, GeodesyXYZExt, RunwayLib, StaticArraysCore
using SimpleNonlinearSolve
import SciMLBase: ReturnCode
import RunwayLib: CamTransform, project
import LinearAlgebra: norm
function solve_once(pt2_x::T) where T
    pose = AffineMap(RotY(0.), XYZ(-100.0m, 0.0m, 0.0m))
    pts = [XYZ(1.0m, 2.0m, 3.0m) + XYZ(pt2_x*m, 0.0m, 0.0m),
           XYZ(1.0m, -2.0m, 3.0m)];
    projections = project.([pose], pts);
    measurements = projections .+ 0 .*[3*randn(2)pxl for _ in eachindex(projections)]
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

function solve_once_nlls(pt2_y::T, solver=SimpleNewtonRaphson()) where T
    cam_rot = RotY(0.); cam_loc = XYZ(-6000.0m, 0.0m, 125.0m)
    pose = CamTransform(cam_rot, cam_loc)
    pts = [XYZ(1.0m, 25.0m, 3.0m),
           XYZ(1.0m, -25.0m, 3.0m)];
    projections = project.([pose], pts);
    # seed!(1)
    # measurements = projections .+ 0 .*[3*randn(2)pxl for _ in eachindex(projections)]
    measurements = projections .+ [ImgProj(0.0pxl, 0.0pxl), ImgProj(pt2_y*pxl, 0.0pxl)]
    loss(loc, (;cam_rot, pts, measurements)) = begin
        simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], pts)
        res = map((lhs, rhs)::Tuple->norm(lhs - rhs),
                  zip(measurements, simulated_projections))
        ustrip.(pxl, res)
    end
    # loss(loc, p) = loss(loc)

    initial_guess = MVector{3, T}([-3000., 0., 0.].*one(T))  # T may be, for example, Dual{Float64}
    prob = NonlinearLeastSquaresProblem(loss, initial_guess, (; cam_rot, pts, measurements))
    res = solve(prob, solver; maxiters=1_000_000)
    @assert res.retcode == ReturnCode.Success
    res.u
end
# This works!!
# ForwardDiff.derivative(x->solve_once_nlls(x, LevenbergMarquardt(linsolve=LS.GenericFactorization(lu))), 0.0)
# compare: mydiff(f, x; e=1e-4) = (f(x+e) - f(x-e))/(2*e)
# mydiff(x->solve_once_nlls(x, LevenbergMarquardt()), 0.0; e=0.00001)
#
# works also easier with SimpleNewtonRaphson, but that one doesn't converge under more harsh conditions.
# ForwardDiff.derivative(x->solve_once_nlls(x, SimpleNewtonRaphson()), 0.0)
