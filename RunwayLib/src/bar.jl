using Optimization
using OptimizationOptimJL
using NonlinearSolve
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


function solve_once_nlls(pt2_y::T, solver=LevenbergMarquardt()) where T
    pose = AffineMap(RotY(0.), XYZ(-6000.0m, 0.0m, 125.0m))
    pts = [XYZ(1.0m, 25.0m, 3.0m),
           XYZ(1.0m, -25.0m, 3.0m)];
    projections = project.([pose], pts);
    seed!(1)
    # measurements = projections .+ 0 .*[3*randn(2)pxl for _ in eachindex(projections)]
    measurements = projections .+ [ImgProj(0.0pxl, 0.0pxl), ImgProj(pt2_y*pxl, 0.0pxl)]
    loss(loc) = begin
        simulated_projections = project.([AffineMap(pose.linear, XYZ(loc*m))], pts)
        res = map((lhs, rhs)::Tuple->norm(lhs - rhs), zip(measurements, simulated_projections)) |> collect |> SVector{length(pts)}
        ustrip.(pxl, res)
    end
    loss(loc, p) = loss(loc);
    # maybe make this in-place, but there seem to be dimensionality problems...
    loss!(buf, loc) = begin
        simulated_projections = project.([AffineMap(pose.linear, XYZ(loc*m))], pts)
        # THIS DOESN'T WORK FOR MORE POINTS. THERE MUST BE A BUG (?)
        buf[1:length(pts)] .= map((lhs, rhs)::Tuple->norm(lhs - rhs),
                                  zip(measurements, simulated_projections)) |> collect .|> x->ustrip(pxl, x)
    end
    loss!(buf, loc, p) = begin
        loss!(buf, loc)
    end

    prob = NonlinearLeastSquaresProblem(loss, MVector{3, T}(-3000. * one(T), 0. * one(T), 0. * one(T)))
    # prob = NonlinearLeastSquaresProblem{true}(loss!, MVector{3, T}(-50. * one(T), 0. * one(T), 0. *one(T)))
    # prob = NonlinearLeastSquaresProblem(loss, [-3000. * one(T), 0. * one(T), 100. * one(T)])
    res = solve(prob, solver; maxiters=1_000_000)
    @assert res.retcode == Optimization.ReturnCode.Success
    res.u
end
# This works!!
# ForwardDiff.derivative(x->solve_once_nlls(x, LevenbergMarquardt(linsolve=LS.GenericFactorization(lu))), 0.0)
# compare: mydiff(f, x; e=1e-4) = (f(x+e) - f(x-e))/(2*e)
# mydiff(x->solve_once_nlls(x, LevenbergMarquardt()), 0.0; e=0.00001)
#
# works also easier with SimpleNewtonRaphson, but that one doesn't converge under more harsh conditions.
# ForwardDiff.derivative(x->solve_once_nlls(x, SimpleNewtonRaphson()), 0.0)
