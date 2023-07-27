function Base.prod(Ns::AbstractVector{<:MvNormalCanon})
    MvNormalCanon(sum(N->N.h, Ns), sum(N->N.J, Ns))
end
function Base.prod(Ns::AbstractVector{<:MvNormal})
    MvNormal(prod(MvNormalCanon.(Ns)))
end
function Base.:*(N1::MvNormalCanon, N2::MvNormalCanon)
    Base.prod([N1, N2])
end
MvNormalCanon(N::MvNormal) = MvNormalCanon(N.Σ\N.μ, inv(N.Σ))
MvNormal(N::MvNormalCanon) = MvNormal(N.μ, inv(N.J))
