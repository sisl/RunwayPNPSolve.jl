using StaticArrays


# Plot everything
no_y(p::Point) = p[[1, 3]] |> Point2f
no_y(p::Vector) = p[[1, 3]] |> Point2f
no_y(p::SVector) = p[[1, 3]] |> Point2f
no_y(p::Matrix) = p[[1, 3], [1, 3]]

ps = Point3d.([
      [10.; 0; 1],
      [10.; 3; 5],
      [10.; -3; 8]])
cam_rot = LinearMap(RotY(-τ/4))  # to point the camera z axis forward
cam_pos = Translation([-10.; 0; 5])
cam_pose = cam_pos ∘ cam_rot
make_projection_map(cam_pose) = PerspectiveMap() ∘ inv(cam_pose)
projection_map = make_projection_map(cam_pose)
p′′s = map(projection_map, ps)

Ns = map(p->compute_bayesian_pose_estimate(
             p,
             cam_rot,
             projection_map(p),
             @SVector [1.0; 1.0];
             x_guess = -10.),
         ps)
N_total = prod(Ns)


plt = Plots.plot(; aspect_ratio=1)
scatter!(plt, ps.|>no_y)
scatter!(plt, [cam_pos.translation |> Point3d |> no_y])
for N in Ns
    covellipse!(plt, N.μ|>no_y, N.Σ|>Matrix|>no_y)
end
covellipse!(plt, N_total.μ|>no_y, N_total.Σ|>Matrix|>no_y)
plt
