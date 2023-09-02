using GeometryBasics
function compute_Δy(p_lhs::XYZ{T}, p_rhs::XYZ{T})::T where T
    abs(p_rhs[2] - p_lhs[2])
end
function compute_Δx(runway_corners::Vector{XYZ{T}})::T where T
    Δx = (  mean([runway_corners[3].x, runway_corners[4].x])
          - mean([runway_corners[1].x, runway_corners[2].x]) )
end

function compute_Δy′(p′_lhs::StaticVector{2, T}, p′_rhs::StaticVector{2, T}) where T
    abs(p′_rhs[2] - p′_lhs[2])
end
function compute_Δx′(p′_mid_near::StaticVector{2, T}, p′_mid_far::StaticVector{2, T}) where T
    abs(p′_mid_far[1] - p′_mid_near[1])
end

flatten_points(pts::Pair{T, T}) where T<:StaticVector = stack(pts, dims=1)[:]
function pnp_alongtrack_from_threshold_width(
        world_pts::Pair{T, T}, # near / far center threshold
        pixel_locations::Pair{Point2{Pixels}, Point2{Pixels}},
        Y::Meters, Z::Meters,
        cam_rotation::Union{LinearMap{<:Rotation{3, Float64}}, Rotation{3, Float64}};
        initial_guess::T = XYZ(-100.0m, 0m, 30m),
        ) where {T<:XYZ{Meters}}
    # strip units for Optim.jl package. See https://github.com/JuliaNLSolvers/Optim.jl/issues/695.
    world_pts = map(p->ustrip.(m, p), world_pts) |> collect
    pixel_locations = map(p->ustrip.(pxl, p), pixel_locations) |> collect
    initial_guess = ustrip.(m, initial_guess)
    cam_rotation = (cam_rotation isa LinearMap ? cam_rotation.linear : cam_rotation)
    Y, Z = ustrip.(m, [Y, Z])

    (length(world_pts) == 0) && return LsqFit.LsqFitResult(initial_guess, 0*similar(initial_guess), [], false, [], [])

    function model(world_pts_flat, X::StaticVector{1, <:Real})
        pos = SVector(X[1], Y, Z)
        let proj = make_projection_fn(AffineMap(cam_rotation, XYZ(pos*1m))),
            unflatten_to_xyz(pts) = unflatten_points(XYZ, pts*1m),
            f = splat(compute_Δy′) ∘ Fix1(broadcast, proj) ∘ unflatten_to_xyz

            MVector(f(world_pts_flat), ) .|> x->ustrip(m, x)
        end
    end

    fit = curve_fit(model,
                    flatten_points(world_pts),
                    SVector(compute_Δy′(pixel_locations[1], pixel_locations[2]), ),
                    MVector(initial_guess.x, );
                    autodiff=:forwarddiff, store_trace=true)
    return fit.param[1]*1m
end

function pnp_alongtrack_from_runway_length(
        world_pts::Pair{XYZ{Meters}, XYZ{Meters}},
        pixel_locations::Pair{Point2{Pixels}, Point2{Pixels}},
        Y::Meters, Z::Meters,
        cam_rotation::Union{LinearMap{<:Rotation{3, Float64}}, Rotation{3, Float64}};
        initial_guess::T = XYZ(-100.0m, 0m, 30m),
        ) where {T<:XYZ{Meters}}
    # @show "hello"
    # @show initial_guess
    # strip units for Optim.jl package. See https://github.com/JuliaNLSolvers/Optim.jl/issues/695.
    # world_pts = map(p->ustrip.(m, p), world_pts) |> collect
    # pixel_locations = map(p->ustrip.(pxl, p), pixel_locations) |> collect
    # initial_guess = ustrip.(m, initial_guess)
    cam_rotation = (cam_rotation isa LinearMap ? cam_rotation.linear : cam_rotation)
    # Y, Z = ustrip.(m, [Y, Z])

    (length(world_pts) == 0) && return LsqFit.LsqFitResult(initial_guess, 0*similar(initial_guess), [], false, [], [])

    function model(world_pts_flat::Vector, X::StaticVector{1, T}) where {T<:Number}
        pos = SVector(X[1]*1m, Y, Z)
        project(pts) = project_points(AffineMap(cam_rotation, pos), pts)
        unflatten_to_xyz(pts) = unflatten_points(XYZ{T}, pts)
        get_width(p_lhs::Point2, p_rhs::Point2) = abs(p_rhs[2] - p_lhs[2])  # recall positive y axis goes left...
        f = splat(compute_Δy′) ∘ project ∘ unflatten_to_xyz
        MVector(f(world_pts_flat), ) .|> x->ustrip(m, x)
    end

    fit = curve_fit(model,
                    flatten_points(world_pts),
                    SVector(ustrip.(m, compute_Δy′(pixel_locations[1], pixel_locations[2])), ),
                    MVector(initial_guess.x, ) .|> x->ustrip(m, x);
                    autodiff=:forward, store_trace=true)
    return fit.param[1]*1m
end

function pnp_height_from_angle(
        angle::Angle,
        world_pts::Vector{T}) :: Meters where {T<:Union{ENU{Meters}, XYZ{Meters}}}
    Δy = compute_Δy(world_pts[1], world_pts[2])
    H = 1/2 * Δy / tan(angle/2)
    return H
end

function pnp3(world_pts::Vector{T},
              pixel_locations::Vector{Point2{Pixels}},
              sideline_angle::Angle,
              cam_rotation::Union{LinearMap{<:Rotation{3, Float64}}, Rotation{3, Float64}};
              initial_guess::T = XYZ(-100.0m, 0m, 30m)
              ) where {T<:Union{ENU{Meters}, XYZ{Meters}}}
    Y = 0.0m
    Z = pnp_height_from_angle(
        sideline_angle,
        world_pts
    )

    X_from_near= pnp_alongtrack_from_threshold_width(
            Pair(world_pts[1], world_pts[2]),
            Pair(pixel_locations[1], pixel_locations[2]),
            Y, Z,
            cam_rotation;
            initial_guess)
    X_from_far = pnp_alongtrack_from_threshold_width(
            Pair(world_pts[3], world_pts[4]),
            Pair(pixel_locations[3], pixel_locations[4]),
            Y, Z,
            cam_rotation;
            initial_guess)
    # X_from_length = pnp_alongtrack_from_runway_length(
    #         Pair(mean([world_pts[1], world_pts[2]]), mean([world_pts[3], world_pts[4]])),
    #         Pair(mean([pixel_locations[1], pixel_locations[2]]), mean([pixel_locations[3], pixel_locations[4]])),
    #         Y, Z,
    #         cam_rotation;
    #         initial_guess)


    X_agg = let Δx = compute_Δx(world_pts),
                Δy = compute_Δy(world_pts[1], world_pts[2]),
                Δx′ = uconvert(m, compute_Δx′(mean([pixel_locations[1], pixel_locations[2]]),
                                  mean([pixel_locations[3], pixel_locations[4]]))),
                Δy′_near = uconvert(m, compute_Δy′(pixel_locations[1], pixel_locations[2])),
                Δy′_far = uconvert(m, compute_Δy′(pixel_locations[3], pixel_locations[4]))
        w_near = abs(1/∂x_∂Δy′(Δy′_near, Δy, Z))
        w_far = abs(1/∂x_∂Δy′(Δy′_far, Δy, Z))
        w_length = abs(1/∂x_∂Δx′(Δx′, Δx, Z))*0
        C = w_near + w_far + w_length
        # @show w_near/C, w_far/C, w_length/C
        (w_near * X_from_near + w_far * X_from_far) / C
    end

    return PNP3Sol((XYZ(X_agg, Y, Z), ))
end
