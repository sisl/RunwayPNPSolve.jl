# run ] dev Unitful GeodesyXYZExt .
# run ] add SimpleNonlinearSolve ForwardDiff CoordinateTransformations Rotations StaticArraysCore
using Rotations, CoordinateTransformations, NonlinearSolve, Unitful, Unitful.DefaultSymbols, GeodesyXYZExt, RunwayLib, StaticArraysCore
using SimpleNonlinearSolve
import SciMLBase: ReturnCode
import RunwayLib: CamTransform, project
import LinearAlgebra: norm

function solve_once_nlls(pt2_y::T, solver=SimpleNewtonRaphson()) where T
    cam_rot = RotY(0.); cam_loc = XYZ(-6000.0m, 0.0m, 125.0m)
    pose = CamTransform(cam_rot, cam_loc)
    pts = [XYZ(1.0m, 25.0m, 3.0m),
           XYZ(1.0m, -25.0m, 3.0m)];
    projections = project.([pose], pts);
    measurements = projections .+ [ImgProj(0.0pxl, 0.0pxl), ImgProj(pt2_y*pxl, 0.0pxl)]
    loss(loc, (;cam_rot, pts, measurements)) = begin
        simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], pts)
        res = (reduce(vcat, measurements) - reduce(vcat, simulated_projections))
        # res = norm.(measurements .- simulated_projections)  # <- this version is about 1_000_000 times slower somehow...
        ustrip.(pxl, res) |> SVector{2*length(pts)}
    end

    initial_guess = MVector{3, T}([-3000., 0., 0.].*one(T))  # T may be, for example, Dual{Float64}
    prob = NonlinearLeastSquaresProblem(loss, initial_guess, (; cam_rot, pts, measurements))
    res = solve(prob, solver; maxiters=1_000_000)
    # @assert res.retcode == ReturnCode.Success
    res.u
end
# This works!!
# ForwardDiff.derivative(x->solve_once_nlls(x, LevenbergMarquardt(linsolve=LS.GenericFactorization(lu))), 0.0)
# compare: mydiff(f, x; e=1e-4) = (f(x+e) - f(x-e))/(2*e)
# mydiff(x->solve_once_nlls(x, LevenbergMarquardt()), 0.0; e=0.00001)
#
# works also easier with SimpleNewtonRaphson, but that one doesn't converge under more harsh conditions.
# ForwardDiff.derivative(x->solve_once_nlls(x, SimpleNewtonRaphson()), 0.0)
# actually it does converge if we're not super close!
