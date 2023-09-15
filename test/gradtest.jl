using LsqFit.ForwardDiff
using StaticArrays

function f(x::StaticVector{3, T}) where T
    return sum(x) / oneunit(T)
end

proj_pts2 = [ImgProj(1.0pxl, 1.0pxl), ImgProj(1.0pxl, -1.0pxl), ImgProj(3.0pxl, -2.5pxl), ImgProj(3.0pxl, 2.5pxl)];
function f3(p)
    (;lhs, rhs) = hough_transform([proj_pts2[1:3]; [p*1pxl]])[:θ]
    # lhs = ustrip(rad, lhs); rhs=ustrip(rad, rhs)
    # cv = ComponentVector(β=[(rhs-2*lhs + 1rad), ], γ=[lhs, rhs])
    # cv = ComponentVector(β=[(rhs+(τ/4))-(lhs-(τ/4)), ], γ=[lhs, rhs])
    cv = ComponentVector(β=[(rhs+(τ/4)rad)-2*(lhs-(τ/4)rad), ], γ=[lhs, rhs])
    ustrip.(rad, sum(cv))
end
x = ustrip.(pxl, proj_pts2[4])
f3(x), (f3(x + sqrt(eps()) * ImgProj(0., 1.)) - f3(x)) / sqrt(eps())
ForwardDiff.gradient(f3, x)


function f2(x)
    x = x*1rad
    x = x + 0.5rad
    ustrip.(rad, x)^2
end
f2(0.1)
ForwardDiff.derivative(f2, 0.1)
