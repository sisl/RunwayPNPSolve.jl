@unit pxl "pxl" Pixel 0.00345mm false

AngularQuantity = Union{WithUnits(°), WithUnits(rad)};
# without this there's a bug when adding different angular units...
Unitful.promote_unit(lhs::T, rhs::S) where {T<:AngularQuantity, S<:AngularQuantity} = rad


"""
    ImgProj{T}

Coordinate system in camera display, i.e. what the camera sees.
x points right, y points up.
"""
struct ImgProj{T} <: FieldVector{2, T}
    x :: T
    y :: T
end
# enable broadcasting and other functionality
similar_type(::Type{<:ImgProj}, ::Type{T}, s::Size{S}) where {T, S} = ImgProj{T}
