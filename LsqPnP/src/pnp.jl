using Rotations
using ComponentArrays
using CoordinateTransformations, Geodesy, GeodesyXYZExt
using LinearAlgebra: dot, norm, I, normalize
using Optim
using ReTest
using Tau
using Roots
using LeastSquaresOptim
using LsqFit
using Unitful: Length
using StaticArrays: StaticVector, MVector, SVector, MArray, FieldVector
using Unitful: ustrip
import LsqFit.DiffResults: DiffResult, MutableDiffResult
import LsqFit.ForwardDiff: Dual
import StatsBase: mean
import Base: Fix1

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

Optim.minimizer(lsr::LsqFit.LsqFitResult) = lsr.param
Optim.converged(lsr::LsqFit.LsqFitResult) = lsr.converged
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
                    autodiff=:forward, store_trace=true)
    return PNP3Sol((pos=XYZ(fit.param)*1m,))
end


"Hough transform."
function compute_rho_theta(p1::StaticVector{2, T}, p2::StaticVector{2, T}, p3::StaticVector{2, T}) where T
    p4(λ) = p1 + λ*(p2-p1)
    λ = dot((p2-p1), (p3-p1)) / norm(p2-p1)^2
    @assert isapprox(dot(p2-p1, p4(λ)-p3)/oneunit(T)^2, 0.; atol=1e-4) "$(dot(p2-p1, p4(λ)-p3))"
    @debug λ, p4(λ)
    ρ = norm(p4(λ) - p3)

    vec1 = Point2d(1, 0)
    vec2 = normalize(p4(λ) - p3)
    y = vec1 - vec2
    x = vec1 + vec2
    θ = 2*atan(norm(y), norm(x)) * -sign(vec2[2])
    return ρ, θ
end
@testset "compute_rho_theta" begin
    ρ, θ = compute_rho_theta(Point2d(-2, 0), Point2d(0, -2), Point2d(0, 0))
    @test all((ρ, θ) .≈ (√(2), 3/8*τ))

    ρ, θ = compute_rho_theta(Point2d(0, 2), Point2d(-2, 2), Point2d(0, 0))
    @test all((ρ, θ) .≈ (2, -2/8*τ))

    ρ, θ = compute_rho_theta(Point2d(0, -2), Point2d(-1, -3), Point2d(0, 0))
    @test all((ρ, θ) .≈ (sqrt(2), 1/8*τ))

    ρ, θ = compute_rho_theta(Point2d(0, 2), Point2d(-1, 3), Point2d(0, 0))
    @test all((ρ, θ) .≈ (sqrt(2), -1/8*τ))
end

function hough_transform(projected_points)  # front left, front right, back left, back right
    ppts = projected_points
    ρ_θ_lhs = compute_rho_theta(ppts[1], ppts[3], (ppts[1]+ppts[2])/2)
    ρ_θ_rhs = compute_rho_theta(ppts[2], ppts[4], (ppts[1]+ppts[2])/2)
    ρ = (; lhs=ρ_θ_lhs[1], rhs=ρ_θ_rhs[1])
    θ = (; lhs=ρ_θ_lhs[2]*1rad, rhs=ρ_θ_rhs[2]*1rad)
    ρ, θ
end



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

PNP3Sol = @NamedTuple begin
  pos::XYZ{Meters}
end
Optim.minimizer(sol::PNP3Sol) = sol.pos
Optim.converged(sol::PNP3Sol) = true
