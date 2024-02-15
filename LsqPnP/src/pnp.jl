import RunwayLib: project, CamTransform, project_line
import Base.Iterators: flatten
import RunwayLib: AngularQuantity
import LinearAlgebra: cholesky
function pnp(world_pts::AbstractVector{<:XYZ{<:WithUnits(m)}},
             measurements::AbstractVector{<:ImgProj{<:WithUnits(pxl)}},
             cam_rot::Rotation{3};
             measurement_covariance::AbstractMatrix{<:Real} = I(2*length(measurements)), # defined as [x_1, y_1, x_2...]
             world_pts_lines::AbstractVector{<:NTuple{2, <:XYZ}} = NTuple{2, XYZ{typeof(1.0m)}}[],
             measured_lines::AbstractVector{<:Tuple} = Tuple{typeof(1.0pxl), typeof(1.0m)}[], #[(0.004489402014602179 * pxl, -1.8384693995160628 * rad)],
             lines_covariance::AbstractMatrix{<:Real} = I(2*length(measured_lines)),  # defined as [rho_1, theta_1, rho_2, ...]
             runway_pts::AbstractArray{<:XYZ{<:WithUnits(m)}}=world_pts,
             initial_guess::XYZ{<:WithUnits(m)} = XYZ(-3000.0m, 0m, 30m),
             components=[:x, :y],
             solver=SimpleNewtonRaphson(),
             )
    isempty(components) && return initial_guess
    N = length(world_pts); @assert N == length(measurements)
    mask_pts = repeat([:x in components, :y in components], outer=N) |> SVector{2*N}
    # mask_angles = repeat([:α in components], outer=4) |> SVector{4}

    loss(loc, (; cam_rot, world_pts, measurements, measured_lines, runway_pts)) = begin
        simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], world_pts)
        # res = norm.(measurements .- simulated_projections)  # <- this version is about 1_000_000 times slower somehow...
        results = []
        # process points
        if (:x in components || :y in components)
            inv_weights = cholesky(measurement_covariance).U'
            res_pts = let measurements_ = SVector(flatten(measurements)...),
                        simulated_projections_ = SVector(flatten(simulated_projections)...)
                res = measurements_ .- simulated_projections_
                inv_weights \ ustrip.(pxl, res).*mask_pts
            end
            # @info typeof(res_pts)
            # push!(results, res_pts)
            append!(results, res_pts)
        end

        # process lines
        if (:α in components)
            simulated_lines = project_line.([CamTransform(cam_rot, XYZ(loc*m))], [ImgProj(0 * pxl, 0 * pxl)], world_pts_lines)
            p = project_line_onto_unit_circle
            res_lines = let 
                measurements_ = p.(measured_lines)
                simulated_lines_ = p.(simulated_lines)
                measurements_ - simulated_lines_
            end
            # @info typeof(res_lines)
            # push!(results, res_lines)
            for elem in res_lines
                append!(results, elem)
            end
            # append!(results, res_lines)

            # @assert length(runway_pts) == 4 "Provide runway corners to use angle."
            # all_simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], runway_pts)
            # simulated_angles = hough_transform(all_simulated_projections)[:θ]
            # p = project_angle_onto_unit_circle
            # f = (x->SVector(flatten(x)...))
            # res_angles = let measurements_ = f(p.([measured_angles[:lhs], measured_angles[:rhs]])),
            #                 simulated_angles_ = f(p.([simulated_angles[:lhs], simulated_angles[:rhs]]))
            #     scaling = deg2rad(1)*2
            #     (measurements_ - simulated_angles_) .* mask_angles / scaling
            # end
            # push!(results, res_angles)
        end

        # @info typeof(results)
        T = promote_type(eltype.(results)...)
        return reduce(vcat, results; init=T[])
    end

    initial_guess = MVector{3}(ustrip.(m, initial_guess))
    ps = (; cam_rot, world_pts, measurements, measured_lines, runway_pts)
    prob = NonlinearLeastSquaresProblem(loss, initial_guess, ps)
    res = solve(prob, solver; maxiters=100_000)
    @assert res.retcode == ReturnCode.Success res
    XYZ(res.u*m)
end
project_angle_onto_unit_circle(angle::AngularQuantity) = Point2(cos(angle), sin(angle))
function project_line_onto_unit_circle(line)
    angle = line[2]
    return SVector{3}([ustrip(pxl, line[1]), ustrip(rad, cos(angle)), ustrip(rad, sin(angle))])
end
