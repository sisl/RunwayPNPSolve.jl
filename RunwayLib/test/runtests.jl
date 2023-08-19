using RunwayLib
using RunwayLib.GeodesyXYZExt: XYZ
using RunwayLib: Meters, m, Pixels, px
using RunwayLib: ProjectionMap, cameramap
using StaticArraysCore
using Test
using Rotations
using CoordinateTransformations

@testset "Projection axis alignments" begin
    @test ProjectionMap{1}()(SVector(1.0, 0.0, 0.0)) ≈ SVector(0.0, 0.0)
    @test ProjectionMap{2}()(SVector(0.0, 1.0, 0.0)) ≈ SVector(0.0, 0.0)
    @test ProjectionMap{3}()(SVector(0.0, 0.0, 1.0)) ≈ SVector(0.0, 0.0)
    @test cameramap(Val(1))(SVector(1.0, 0.0, 0.0)) ≈ SVector(0.0, 0.0)
    @test cameramap(Val(2))(SVector(0.0, 1.0, 0.0)) ≈ SVector(0.0, 0.0)
    @test cameramap(Val(3))(SVector(0.0, 0.0, 1.0)) ≈ SVector(0.0, 0.0)


    proj = make_projection_fn(AffineMap(RotY(0.0), XYZ(-1.0m, 0.0m, 0.0m)))
    @test proj(XYZ(1.0m, 0.0m, 0.0m)) == ImgProj{Pixels}(0.0px, 0.0px)

    rc = RunwayCorners(XYZ{Meters})((
        XYZ(1.0m, 0.0m, 0.0m),
        XYZ(1.0m, 0.0m, 0.0m),
        XYZ(1.0m, 0.0m, 0.0m),
        XYZ(1.0m, 0.0m, 0.0m),
    ))
    @test proj(rc) == RunwayCorners(ImgProj{Pixels})((
        ImgProj(0.0px, 0.0px),
        ImgProj(0.0px, 0.0px),
        ImgProj(0.0px, 0.0px),
        ImgProj(0.0px, 0.0px),
    ))
end
