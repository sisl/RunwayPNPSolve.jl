Angle = Union{typeof(1.0°),typeof(1.0rad)}
Meters = typeof(1.0m)
@unit pxl "pxl" Pixel 0.00345mm false
Pixels = typeof(1.0pxl)

AngularUnits = Union{typeof(unit(1.0°)), typeof(unit(1.0rad))};
# without this there's a bug when adding different angular units...
Unitful.promote_unit(lhs::T, rhs::S) where {T<:AngularUnits, S<:AngularUnits} = rad


" x points up, y points left "
struct ImgProj{T} <: FieldVector{2, T}
    x
    y
end
similar_type(::Type{<:ImgProj}, ::Type{T}, s::Size{S}) where {T, S} = ImgProj{T}

"Function to produce a NamedTuple type.
Similar to a struct, but call with () instead of {} for type param. Benefit: We can broadcast over `values(tpl)`.
"
RunwayCorners(::Type{T}) where {T<:Union{XYZ{Meters},ImgProj{Pixels}}} = @NamedTuple begin
    near_left::T
    near_right::T
    far_left::T
    far_right::T
end
