"Hough transform."
function compute_rho_theta(p1::StaticVector{2, T}, p2::StaticVector{2, T′}, p3::StaticVector{2, T′′}) where {T, T′, T′′}
    p4(λ) = p1 + λ*(p2-p1)
    λ = dot((p2-p1), (p3-p1)) / norm(p2-p1)^2
    # @assert isapprox(dot(p2-p1, p4(λ)-p3)/oneunit(T)^2, 0.; atol=1e-3) "$(dot(p2-p1, p4(λ)-p3))"
    ρ = norm(p4(λ) - p3)

    vec1 = Point2(1., 0.)
    vec2 = normalize(p4(λ) - p3)
    y = vec1 - vec2
    x = vec1 + vec2
    θ = 2*atan(norm(y), norm(x)) * -sign(vec2[2])
    return ρ, θ
end
@testset "compute_rho_theta" begin
    ρ, θ = compute_rho_theta(Point2(-2, 0), Point2{Float64}(0, -2), Point2{Float64}(0, 0))
    @test all((ρ, θ) .≈ (√(2), 3/8*τ))

    ρ, θ = compute_rho_theta(Point2(0, 2), Point2{Float64}(-2, 2), Point2{Float64}(0, 0))
    @test all((ρ, θ) .≈ (2, -2/8*τ))

    ρ, θ = compute_rho_theta(Point2{Float64}(0, -2), Point2{Float64}(-1, -3), Point2{Float64}(0, 0))
    @test all((ρ, θ) .≈ (sqrt(2), 1/8*τ))

    ρ, θ = compute_rho_theta(Point2{Float64}(0, 2), Point2{Float64}(-1, 3), Point2{Float64}(0, 0))
    @test all((ρ, θ) .≈ (sqrt(2), -1/8*τ))
end

function hough_transform(projected_points)  # front left, front right, back left, back right
    ppts = projected_points
    ρ_θ_lhs = compute_rho_theta(ppts[1], ppts[4], (ppts[1]+ppts[2])/2)
    ρ_θ_rhs = compute_rho_theta(ppts[2], ppts[3], (ppts[1]+ppts[2])/2)
    ρ = (; lhs=ρ_θ_lhs[1], rhs=ρ_θ_rhs[1])
    θ = (; lhs=ρ_θ_lhs[2]*1rad, rhs=ρ_θ_rhs[2]*1rad)
    (; ρ, θ)
end
