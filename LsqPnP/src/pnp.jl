import RunwayLib: project, CamTransform
import Base.Iterators: flatten
function pnp(world_pts::SVector{N, <:XYZ{<:WithUnits(m)}},
             measurements::SVector{N, <:ImgProj{<:WithUnits(pxl)}},
             # pixel_feature_mask::Union{UnitRange{Int64}, Colon, <:AbstractVector{Bool}},
             cam_rot::Rotation{3};
             # angles = ComponentVector{Angle}(β=[], ᵞ=[]),
             initial_guess::XYZ{<:WithUnits(m)} = XYZ(-3000.0m, 0m, 30m),
             components=[:x, :y],
             solver=SimpleNewtonRaphson(),
             ) where {N}
    mask = repeat([:x in components, :y in components], outer=N) |> SVector{2*N}

    loss(loc, (; cam_rot, world_pts, measurements)) = begin
        simulated_projections::SVector{N, ImgProj} = project.([CamTransform(cam_rot, XYZ(loc*m))], world_pts)
        res = SVector(flatten(measurements)...) - SVector(flatten(simulated_projections)...)
        # res = norm.(measurements .- simulated_projections)  # <- this version is about 1_000_000 times slower somehow...
        ustrip.(pxl, res).*mask
    end

    initial_guess = MVector{3}(ustrip.(m, initial_guess))
    ps = (; cam_rot, world_pts, measurements)
    prob = NonlinearLeastSquaresProblem(loss, initial_guess, ps)
    res = solve(prob, solver; maxiters=1_000_000)
    @assert res.retcode == ReturnCode.Success
    res.u*m
end

### Solution using Levenberg-Marquardt algo from LsqFit.jl
# I've tried before to use LeastSquaresOptim but got bad results.
# LsqFit expects vector valued in- and outputs, so we have to flatten/unflatten the vector of points.
# Also, we overload the Optim.minimizer/converged functions for easier back-and-forth between Optim.jl and LsqFit.jl.
# flatten_points(pts::Vector{<:StaticVector}) = stack(pts, dims=1)[:]
# flatten_points(pts::Vector{<:FieldVector{2, T}}) where {T} = begin
#     data = (length(pts) > 0 ? stack(pts, dims=1) : zeros(T, 0, 2))
#     ComponentVector((; x=data[:, 1],
#                        y=data[:, 2]))
# end
# flatten_points(pts::Vector{<:FieldVector{3, T}}) where {T} = begin
#     data = (length(pts) > 0 ? stack(pts, dims=1) : zeros(T, 0, 3))
#     ComponentVector((; x=data[:, 1],
#                        y=data[:, 2],
#                        z=data[:, 3]))
# end
# unflatten_points(P::Type{<:StaticVector}, pts::AbstractVector{<:Number}) = P.(eachrow(reshape(pts, :, length(P))))
# in_camera_img(p::ImgProj{Pixels}) = all(p .∈ [(-3000÷2*1pxl) .. (3000÷2*1pxl);
#                                               (-4096÷2*1pxl) .. (4096÷2*1pxl)])
#
# expand_angles(vec::ComponentVector) =
#     2 * ComponentVector(NamedTuple(k=>[sin.(vec[k]); cos.(vec[k])] for k in keys(vec)))
#
# # We need to overload DiffResults to support mutable static arrays, see https://github.com/JuliaDiff/DiffResults.jl/issues/25
# DiffResult(value::MArray, derivs::Tuple{Vararg{MArray}}) = MutableDiffResult(value, derivs)
# DiffResult(value::Union{Number, AbstractArray}, derivs::Tuple{Vararg{MVector}}) = MutableDiffResult(value, derivs)
# function pnp(world_pts::Vector{XYZ{Meters}},
#              pixel_locations::Vector{ImgProj{Pixels}},
#              pixel_feature_mask::Union{UnitRange{Int64}, Colon, <:AbstractVector{Bool}},
#              cam_rotation::Rotation{3, Float64};
#              angles = ComponentVector{Angle}(β=[], ᵞ=[]),
#              initial_guess::XYZ{Meters} = XYZ(-100.0m, 0m, 30m),
#              components=[:x, :y],
#              )
#     N_mask = if typeof(pixel_feature_mask) == UnitRange{Int64}
#         length(pixel_feature_mask)
#     elseif typeof(pixel_feature_mask) == Colon
#         length(pixel_locations)
#     elseif typeof(pixel_feature_mask) <: AbstractVector{Bool}
#         sum(pixel_feature_mask)
#     else
#         error("???")
#     end
#     @assert length(pixel_locations) == N_mask
#     # early exit if no points given
#     (length(pixel_locations) + length(angles) == 0) && return PNP3Sol((pos=initial_guess, converged=false,))
#
#     in_camera_img_mask = in_camera_img.(pixel_locations)
#     # world_pts = world_pts[in_camera_img_mask]
#     pixel_locations = pixel_locations[in_camera_img_mask]
#     (length(world_pts)+length(angles) == 0) && error("No runway points in camera frame during pnp and no angles available.")
#
#     function model(world_pts_flat, pos::StaticVector{3, <:Real})
#         proj = make_projection_fn(AffineMap(cam_rotation, XYZ(pos*1m)))
#         projected_points_given_pos = proj.(unflatten_points(XYZ, world_pts_flat).*1m)
#         angles_given_pos = let
#             (; lhs, rhs) = hough_transform(projected_points_given_pos[1:4])[:θ]
#             ComponentVector(β=[(rhs+(τ/4)rad)-(lhs-(τ/4)rad), ], γ=[lhs, rhs])
#         end
#
#         res = let
#             pix = ustrip.(pxl, flatten_points(projected_points_given_pos[pixel_feature_mask][in_camera_img_mask]))
#             # @assert eltype(angles_given_pos) <: Angle
#             ang = ustrip.(°, angles_given_pos)
#             ang = expand_angles(ang)
#             vcat(pix, ang)[components]
#         end |> collect
#     end
#
#     xs = ustrip.(m, flatten_points(world_pts))
#     ys = let
#         pix = ustrip.(pxl, flatten_points(pixel_locations))
#         @assert eltype(angles) <: Angle "$angles"
#         ang = ustrip.(°, angles)
#         ang = expand_angles(ang)
#         vcat(pix, ang)[components]
#     end |> collect
#     init = ustrip.(MVector(initial_guess))
#     fit = try
#         curve_fit(model, xs, ys, init;
#                   autodiff=:forward)
#     catch e
#         # @warn components
#         # @warn angles
#         # @warn ys
#         # @warn in_camera_img_mask
#         @warn e
#         @warn "Failed to converge"
#         return PNP3Sol((pos=initial_guess, converged=false,))
#     end
#     return PNP3Sol((pos=XYZ(fit.param)*1m, converged=true))
# end
#
#
# PNP3Sol = @NamedTuple begin
#   pos::XYZ{Meters}
#   converged::Bool
# end
# Optim.minimizer(sol::PNP3Sol) = sol.pos
# Optim.converged(sol::PNP3Sol) = true
