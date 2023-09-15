using PNPSolve, RunwayLib
using GeodesyXYZExt, Rotations, CoordinateTransformations
using BenchmarkTools
using ThreadsX

const Δx = 3500.0m
const Δy = 61.0m
runway_corners =
    XYZ{Meters}[[0m, -Δy / 2, 0m], [0m, Δy / 2, 0m], [Δx, +Δy / 2, 0m], [Δx, -Δy / 2, 0m]]

R_t_true = RotY(0.0)
C_t_true = XYZ([-6000.0m, 0m, 123.0m])
cam_pose_gt = AffineMap(R_t_true, C_t_true)
projected_points = make_projection_fn(cam_pose_gt).(runway_corners)

corner_feature_mask=(1:4)

num_pose_est = 1_000
function run(num_pose_est = 1_000; collect_fn=collect)
    collect_fn(
        LsqPnP.pnp(
            runway_corners,
            (projected_points .+ sample_measurement_noise(length(projected_points)))[corner_feature_mask],
            corner_feature_mask,
            RotY(0.0);
            initial_guess = cam_pose_gt.translation + sample_pos_noise(),
        ) for _ = 1:num_pose_est
    )
end

@info "Serial:"
res = @benchmark run()
display(res)
println("$(floor(Int, 1_000/(median(res).time / 1e9))) iters / s")
@info "Parallel:"
res2 = @benchmark run(; collect_fn=ThreadsX.collect)
display(res2)
println("$(floor(Int, 1_000/(median(res2).time / 1e9))) iters / s")
