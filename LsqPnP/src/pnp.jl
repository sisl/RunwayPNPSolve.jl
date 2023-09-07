### Solution using Levenberg-Marquardt algo from LsqFit.jl
# I've tried before to use LeastSquaresOptim but got bad results.
# LsqFit expects vector valued in- and outputs, so we have to flatten/unflatten the vector of points.
# Also, we overload the Optim.minimizer/converged functions for easier back-and-forth between Optim.jl and LsqFit.jl.
flatten_points(pts::Vector{<:StaticVector}) = stack(pts, dims=1)[:]
flatten_points(pts::Vector{<:FieldVector{2, T}}) where {T} = begin
    data = stack(pts, dims=1)
    ComponentVector((; x=data[:, 1],
                       y=data[:, 2]))
end
flatten_points(pts::Vector{<:FieldVector{3, T}}) where {T} = begin
    data = stack(pts, dims=1)
    ComponentVector((; x=data[:, 1],
                       y=data[:, 2],
                       z=data[:, 3]))
end
unflatten_points(P::Type{<:StaticVector}, pts::AbstractVector{<:Number}) = P.(eachrow(reshape(pts, :, length(P))))
in_camera_img(p::ImgProj{Pixels}) = all(p .∈ [(-3000÷2*1pxl) .. (3000÷2*1pxl);
                                                (-4096÷2*1pxl) .. (4096÷2*1pxl)])

# We need to overload DiffResults to support mutable static arrays, see https://github.com/JuliaDiff/DiffResults.jl/issues/25
DiffResult(value::MArray, derivs::Tuple{Vararg{MArray}}) = MutableDiffResult(value, derivs)
DiffResult(value::Union{Number, AbstractArray}, derivs::Tuple{Vararg{MVector}}) = MutableDiffResult(value, derivs)
function pnp(world_pts::Vector{XYZ{Meters}},
             pixel_locations::Vector{ImgProj{Pixels}},
             cam_rotation::Rotation{3, Float64};
             initial_guess::XYZ{Meters} = XYZ(-100.0m, 0m, 30m),
             components=[:x, :y],
             )
    # early exit if no points given
    (length(world_pts) == 0) && return PNP3Sol((pos=initial_guess, ))

    in_camera_img_mask = in_camera_img.(pixel_locations)
    world_pts = world_pts[in_camera_img_mask]
    pixel_locations = pixel_locations[in_camera_img_mask]
    (length(world_pts) == 0) && error("No runway points in camera frame during pnp.")

    # strip units for Optim.jl package. See https://github.com/JuliaNLSolvers/Optim.jl/issues/695.
    world_pts = map(p->ustrip.(m, p), world_pts) |> collect
    N_p = length(world_pts)
    pixel_locations = map(p->ustrip.(pxl, p), pixel_locations) |> collect
    initial_guess = ustrip.(m, initial_guess)


    function model(world_pts_flat, pos::StaticVector{3, <:Real})
        proj = make_projection_fn(AffineMap(cam_rotation, XYZ(pos*1m)))
        unflatten_to_xyz(pts) = unflatten_points(XYZ, pts*1m)
        f = flatten_points ∘ Fix1(broadcast, proj) ∘ unflatten_to_xyz

        res = f(world_pts_flat) .|> p′->ustrip.(pxl, p′)
        res[components] |> collect
    end

    fit = curve_fit(model,
                    flatten_points(world_pts),
                    flatten_points(pixel_locations)[components] |> collect,
                    MVector(initial_guess);
                    autodiff=:forward,  )
    return PNP3Sol((pos=XYZ(fit.param)*1m,))
end


PNP3Sol = @NamedTuple begin
  pos::XYZ{Meters}
end
Optim.minimizer(sol::PNP3Sol) = sol.pos
Optim.converged(sol::PNP3Sol) = true
