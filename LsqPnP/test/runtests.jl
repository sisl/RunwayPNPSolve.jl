using LsqPnP
using Tests

@testset "flatten / unflatten" begin
    ps = [XYZ(1.0m, 2.0m, 3.0m), XYZ(10.0m, 20.0m, 30.0m)]
    @testall(ps .== unflatten_points(XYZ{Meters}, flatten_points(ps)))
end
