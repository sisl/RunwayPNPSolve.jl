import RunwayLib: project, CamTransform, project_line
import Base.Iterators: flatten
import RunwayLib: AngularQuantity
import LinearAlgebra: cholesky
import NonlinearSolve

function pnp(world_pts::AbstractVector{<:XYZ{<:WithUnits(m)}},
             measurements::AbstractVector{<:ImgProj{<:WithUnits(pxl)}},
             cam_rot::Union{Rotation{3}, Nothing};
             measurement_covariance::AbstractMatrix{<:Real} = I(2*length(measurements)), # defined as [x_1, y_1, x_2...]
             world_pts_lines::AbstractVector{<:NTuple{2, <:XYZ}} = NTuple{2, XYZ{typeof(1.0m)}}[],
             measured_lines::AbstractVector{<:Tuple} = Tuple{typeof(1.0pxl), typeof(1.0m)}[], #[(0.004489402014602179 * pxl, -1.8384693995160628 * rad)],
             lines_covariance::AbstractMatrix{<:Real} = I(2*length(measured_lines)),  # defined as [rho_1, theta_1, rho_2, ...]
             runway_pts::AbstractArray{<:XYZ{<:WithUnits(m)}}=world_pts,
             initial_guess_loc::XYZ{<:WithUnits(m)} = XYZ(-3000.0m, 0m, 30m),
             initial_guess_rot::AbstractVector{<:Real} = [0.0, 0.0, 0.0], # roll, pitch, yaw (radians)
             components=[:x, :y],
             solver=SimpleNewtonRaphson(), #NonlinearSolve.LevenbergMarquardt() #SimpleNewtonRaphson()
             )
    isempty(components) && return initial_guess
    N = length(world_pts); @assert N == length(measurements)

    loss(pose, (; cam_rot, world_pts, measurements, measured_lines, runway_pts, use_lines)) = begin
        loc = pose[1:3]
        cam_rot = isnothing(cam_rot) ? RotXYZ(roll=pose[4], pitch=pose[5], yaw=pose[6]) : cam_rot
        simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], world_pts)
        results = []
        # process points
        if (:x in components || :y in components)
            inv_weights = cholesky(measurement_covariance).U'
            res_pts = let measurements_ = SVector(flatten(measurements)...),
                        simulated_projections_ = SVector(flatten(simulated_projections)...)
                res = measurements_ .- simulated_projections_
                inv_weights \ ustrip.(pxl, res)
            end
            append!(results, res_pts)
        end

        # process lines
        if use_lines
            # Convert measurement covariance to line projection (add extra elements)
            cov_ext = extend_covariance(lines_covariance)
            inv_weights = cholesky(cov_ext).U'
            # Project lines
            simulated_lines = project_line.([CamTransform(cam_rot, XYZ(loc*m))], [ImgProj(0 * pxl, 0 * pxl)], world_pts_lines)
            p = project_line_onto_unit_circle
            res_lines = let 
                measurements_ = SVector(flatten(p.(measured_lines))...)
                simulated_lines_ = SVector(flatten(p.(simulated_lines))...)
                res = measurements_ - simulated_lines_
                inv_weights \ res
            end
            for elem in res_lines
                append!(results, elem)
            end
        end

        T = promote_type(eltype.(results)...)
        return reduce(vcat, results; init=T[])
    end

    # Run first without edges to get good initial guess
    if isnothing(cam_rot)
        initial_guess = MVector{6}(vcat(ustrip.(m, initial_guess_loc), initial_guess_rot))
    else
        initial_guess = MVector{3}(ustrip.(m, initial_guess_loc))
    end
    ps = (; cam_rot, world_pts, measurements, measured_lines, runway_pts, use_lines=false)
    prob = NonlinearLeastSquaresProblem(loss, initial_guess, ps)
    res = solve(prob, solver; maxiters=1_000, abstol=1e-2)
    @assert res.retcode == ReturnCode.Success res
    if !(:α in components)
        return isnothing(cam_rot) ? (XYZ(res.u[1:3]*m), res.u[4:6]) : XYZ(res.u*m)
    end

    # Run with edges with good initial guess
    new_initial_guess = isnothing(cam_rot) ? MVector{6}(res.u) : MVector{3}(res.u)
    # println(new_initial_guess)
    ps = (; cam_rot, world_pts, measurements, measured_lines, runway_pts, use_lines=true)
    prob = NonlinearLeastSquaresProblem(loss, new_initial_guess, ps)
    res = solve(prob, solver; maxiters=1_000, abstol=1e-2)
    @assert res.retcode == ReturnCode.Success res
    # println(res)
    return isnothing(cam_rot) ? (XYZ(res.u[1:3]*m), res.u[4:6]) : XYZ(res.u*m)
end

project_angle_onto_unit_circle(angle::AngularQuantity) = Point2(cos(angle), sin(angle))

function project_line_onto_unit_circle(line::Tuple)
    angle = line[2]
    return SVector{3}([ustrip(pxl, line[1]), ustrip(rad, cos(angle)), ustrip(rad, sin(angle))])
end

function extend_covariance(cov::AbstractMatrix{<:Real})
    n = convert(Int64, size(cov, 1) / 2)
    cov_ext = zeros(3n, 3n)
    for i in 1:n
        cov_ext[3*(i-1)+1:3*(i-1)+2, 3*(i-1)+1:3*(i-1)+2] = cov[2*(i-1)+1:2*(i-1)+2, 2*(i-1)+1:2*(i-1)+2]
        cov_ext[3i, 3i] = cov[2i, 2i]
    end
    return cov_ext
end
