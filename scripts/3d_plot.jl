using Makie, GLMakie
using StaticArrays
using StatsBase


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
             @SVector [1.0e-1; 1.0e-1];
             x_guess = -20.),
         ps)
N_total = prod(Ns)

fig = Makie.Figure();
scene = Makie.LScene(fig[1, 1])
Makie.scatter!(scene, ps)
Makie.scatter!(scene, [cam_pos.translation |> Point3d])
#
xs = cam_pos.translation[1] .+ LinRange(-30, 30, 101)/10
ys = cam_pos.translation[2] .+ LinRange(-40, 40, 101)/10
zs = cam_pos.translation[3] .+ LinRange( -20,  20, 101)/10
individual_plots = []
for N in Ns
    vol = [pdf(N, [x, y, z]) for x∈xs, y∈ys, z∈zs];
    push!(individual_plots,
          Makie.contour!(scene, xs, ys, zs, vol; levels=5, alpha=0.3))
end
let N = N_total
    vol = [pdf(N, [x, y, z]) for x∈xs, y∈ys, z∈zs];
    Makie.contour!(scene, xs, ys, zs, vol; levels=5, alpha=1.0)
end
cam3d!(scene)
viz_toggle = Toggle(fig[2, 1]; tellwidth=false)
connect!(individual_plots[1].visible, viz_toggle.active)
connect!(individual_plots[2].visible, viz_toggle.active)
connect!(individual_plots[3].visible, viz_toggle.active)
fig
