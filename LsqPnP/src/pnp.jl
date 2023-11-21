import RunwayLib: project, CamTransform
import Base.Iterators: flatten
import RunwayLib: AngularQuantity
function pnp(world_pts::AbstractVector{<:XYZ{<:WithUnits(m)}},
             measurements::AbstractVector{<:ImgProj{<:WithUnits(pxl)}},
             cam_rot::Rotation{3};
             measured_angles::ComponentVector{<:AngularQuantity} = ComponentVector(lhs=0.0rad, rhs=0.0rad),
             runway_pts::AbstractArray{<:XYZ{<:WithUnits(m)}}=world_pts,
             initial_guess::XYZ{<:WithUnits(m)} = XYZ(-3000.0m, 0m, 30m),
             components=[:x, :y],
             solver=SimpleNewtonRaphson(),
             )
    isempty(components) && return initial_guess
    N = length(world_pts); @assert N == length(measurements)
    mask_pts = repeat([:x in components, :y in components], outer=N) |> SVector{2*N}
    mask_angles = repeat([:α in components], outer=4) |> SVector{4}

    loss(loc, (; cam_rot, world_pts, measurements, measured_angles, runway_pts)) = begin
        simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], world_pts)
        # res = norm.(measurements .- simulated_projections)  # <- this version is about 1_000_000 times slower somehow...
        results = []
        # process corner projections
        if (:x in components || :y in components)
            res_pts = let measurements_ = SVector(flatten(measurements)...),
                        simulated_projections_ = SVector(flatten(simulated_projections)...)
                res = measurements_ .- simulated_projections_
                ustrip.(pxl, res).*mask_pts
            end
            push!(results, res_pts)
        end

        # process sideline angles
        if (:α in components)
            @assert length(runway_pts) == 4 "Provide runway corners to use angle."
            all_simulated_projections = project.([CamTransform(cam_rot, XYZ(loc*m))], runway_pts)
            simulated_angles = hough_transform(all_simulated_projections)[:θ]
            p = project_angle_onto_unit_circle
            f = (x->SVector(flatten(x)...))
            res_angles = let measurements_ = f(p.([measured_angles[:lhs], measured_angles[:rhs]])),
                            simulated_angles_ = f(p.([simulated_angles[:lhs], simulated_angles[:rhs]]))
                scaling = deg2rad(1)*2
                (measurements_ - simulated_angles_) .* mask_angles / scaling
            end
            push!(results, res_angles)
        end

        T = promote_type(eltype.(results)...)
        return reduce(vcat, results; init=T[])
    end

    initial_guess = MVector{3}(ustrip.(m, initial_guess))
    ps = (; cam_rot, world_pts, measurements, measured_angles, runway_pts)
    prob = NonlinearLeastSquaresProblem(loss, initial_guess, ps)
    res = solve(prob, solver; maxiters=100_000)
    @assert res.retcode == ReturnCode.Success
    res.u*m
end
project_angle_onto_unit_circle(angle::AngularQuantity) = Point2(cos(angle), sin(angle))
