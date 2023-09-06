Angle = Union{typeof(1.0°),typeof(1.0rad)}
Meters = typeof(1.0m)
@unit pxl "pxl" Pixel 0.00345mm false
Pixels = typeof(1.0pxl)

" x points up, y points left "
struct ImgProj{T} <: FieldVector{2, T}
    x
    y
end
similar_type(::Type{A}, ::Type{T}, s::Size{S}) where {A<:ImgProj, T, S} = ImgProj{T}

"Function to produce a NamedTuple type.
Similar to a struct, but call with () instead of {} for type param. Benefit: We can broadcast over `values(tpl)`.
"
RunwayCorners(::Type{T}) where {T<:Union{XYZ{Meters},ImgProj{Pixels}}} = @NamedTuple begin
    near_left::T
    near_right::T
    far_left::T
    far_right::T
end
