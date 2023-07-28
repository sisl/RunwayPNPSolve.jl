### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ a7d8c731-e529-435f-ad30-aefa0732ed1d
# ╠═╡ show_logs = false
begin
import Pkg; Pkg.activate();
using CSV, DataFrames, DataFramesMeta
using Unitful; using Unitful.DefaultSymbols; const Meters = typeof(1.0m)
using IntervalSets
using StatsBase, Statistics
md"""Some imports. (hidden)"""
end

# ╔═╡ c54e4a78-ee13-4b83-aec3-911fd29d157e
begin
import Makie, AlgebraOfGraphics;
import WGLMakie, CairoMakie;
CairoMakie.activate!()  # choose between WGL and Cairo.
	                    # Cairo => export
	                    # WGL => interactivity
AoG = AlgebraOfGraphics;
md"""Set up graphics backend. (hidden)"""
end

# ╔═╡ 76e85dc3-974a-40ec-8a69-e33eb43afa66
md"""
## Investigating the model's corner error distribution

#### Motivation
- During the error sensitivity analysis of the pose estimation system, we have found that for a corner-pixel-based pose estimation, 1 pixel of error corresponds to approximately σ=150m alongtrack error at 6000m distance.
- In this notebook, we investigate how much pixel error we actually have to expect.

#### Service volume.
The MPVS specifies a service volume with limits on the alongtrack distance, and the horizontal and vertical angles.
Specifially, the limits are (-6000m..-300m) $\times$ (-3°..3°) $\times$ (1.2°..6°).
We will therefore filter out samples not in the service volume.

Let's start by defining this requirement in code.
"""

# ╔═╡ 6c0bfb80-e907-46c1-90d9-c10b8ac2414b
in_service_volume(row) = 
  row.gt_translation_x ∈ -6000m..(-300m) &&
  row.horizontal_angle ∈ -3°..3° &&
  row.vertical_angle   ∈ 1.2°..6°

# ╔═╡ 5e15a03e-3063-406e-ac55-6275a1c8a75a
md"""
We want to consider some real, recent data.
Let's load and slightly wrangle it.
"""


# ╔═╡ 89123779-3cca-4540-955e-3ab03750a671
datafile = (false ? 
	"/tmp/trl_resultsKSJC.csv" :      # cluster2:/home/matt.sorgenfrei/vnv_results/trl_resultsKSJC.csv
	"/tmp/26_trajectories_concat.csv" # /mnt/data1/vnv/staging/romeo/csv_v2/26_trajectories_concat.csv
)

# ╔═╡ 2ca9aebe-95f5-49a8-ada2-a6e56712a8ab
df_full = let df = CSV.read(datafile, DataFrame)
    transform(df,
        # make "unitful"
        :gt_translation_x=>ByRow(Meters),
        :gt_translation_y=>ByRow(Meters),
        :gt_translation_z=>ByRow(Meters),
        # compute horizontal and vertical angle
        [:gt_translation_x, :gt_translation_y] => ByRow(
          (x, y) -> atan(y / abs(x))rad
        ) => :horizontal_angle,
        [:gt_translation_x, :gt_translation_z] => ByRow(
          (x, z) -> atan(z / abs(x))rad
        ) => :vertical_angle;
        renamecols=false  # basically "in-place"
    )
end;

# ╔═╡ c3f2d8f8-04fd-4f45-af0d-eaceb849bee5
df_in_service_volume = filter(in_service_volume, df_full);

# ╔═╡ 1cd77f49-5617-408b-a7b1-bc16f502b885
@info "num samples in service volume = $(nrow(df_in_service_volume))"

# ╔═╡ 25c74665-4c6d-4872-8e4e-86cdbc76ebf6
function unfold_df(df)
"""Bring dataframe into stacked form for better processing.

The dataframe by default has columns `gt_vertices_runway_top_left_corner_x` etc., aggregating many predictions in one row.
For easier processing, we want one prediction error per row, with columns e.g. `loc_top_bottom=top`, `loc_left_right=left`, `axis=x`, etc.

Executing this takes usually <1s (0.3s for 20_000 samples).
"""
  vcat([
  select(df, [Symbol("gt_vertices_runway_$(loc_tb)_$(loc_lr)_corner_$(ax)"),
            Symbol("pred_vertices_$(loc_tb)_$(loc_lr)_$(ax)")]
         => ByRow((val_gt, val_pred)->[val_pred-val_gt, ax, loc_tb, loc_lr])
         => [:err_val, :axis, :loc_near_far, :loc_left_right])
  for loc_tb in ["top", "bottom"],
      loc_lr in ["left", "right"],
    ax in ["x", "y"]
  ]...) |> dropmissing
end

# ╔═╡ 204e5e13-c1d1-412f-9cf2-c1037a1e5cf6
function error_plots(df_unfolded;
                     linkxaxes=:all,
                     datalimits=(arr->2*iqr(arr).*(-1,1)),
                     title=nothing)
  plt = AoG.data(df_unfolded)
  plt *= AoG.mapping(:err_val => "Error [px]",
                     col=:loc_left_right,
                     row=:loc_near_far => AoG.renamer(["top"=>"far",
                                                       "bottom"=>"near"]),
                     color=:axis)
  plt *= AoG.density(; datalimits)
  fig = AoG.draw(plt, facet = (; linkxaxes, linkyaxes = :none))
  if !isnothing(title); Makie.Label(fig.figure[0, 1:2], title;
                                    font=:bold, fontsize=20); end
  fig
end

# ╔═╡ 169e4d42-fe69-49d3-a2e6-c76d6818bf93
md"""
Now we're ready to investigate the actual distribution of location estimation errors.
We have one plot for each of the for corners, and plot error in x- and y-direction.
"""

# ╔═╡ 7afdbacc-d3f0-4309-aeca-9d3b355396e1
 error_plots(df_in_service_volume |> unfold_df;
             title="Corner estimation errors across full service volume.")

# ╔═╡ f804765c-8e8f-4c80-a98f-e15a6af4e370
md"""
We see that the errors are actually way above 1 pixel for all corners.
In fact, consider the following fact:
"""

# ╔═╡ 95145748-e937-44c5-a732-13c36540454c
@info quantile.([(df_in_service_volume |> unfold_df).err_val], [0.25, 0.75])

# ╔═╡ 1e558af0-b1c9-458d-9ae8-1e2e407b53ad
md"""
In other words, more than 50% of the sensor measurements have an error larger than 2 pixels.
"""

# ╔═╡ 7890895d-a6a7-467a-a22c-7b34375c906d
md"""
### Considering the extreme cases only.
Next, we want to look at only samples that lie at the "extremes" of this service volume.
Notice however that there are practically no samples that lie at all extremes at once.
We therefore consider samples that lie on one of the three extremes only.
"""

# ╔═╡ 2848f3aa-257f-4dba-bac0-74fad2449041
md"""
Let's define the "extreme" service volume and count how many samples lie in each extreme.
"""


# ╔═╡ 9f55e07a-fae1-408e-8c20-c204250aafdb
in_extreme_service_volume(row) = 
  row.gt_translation_x ∈ (-6000m .. -5000m) ||
  abs(row.horizontal_angle) ∈ 1°..3° ||
  row.vertical_angle ∈ 1.2°..3.0°

# ╔═╡ 7cdd671a-867a-4fe6-9c59-777e1711e182
begin
@info filter(row->row.gt_translation_x ∈ (-6000m .. -5000m), df_full) |> nrow
@info filter(row->abs(row.horizontal_angle) ∈ 1°..3°, df_full) |> nrow
@info filter(row->row.vertical_angle ∈ 1.2°..3.0°, df_full) |> nrow
end

# ╔═╡ d5a9d032-2f0e-4b7e-b1a1-b4a427c3a475
md"""
Notice that we picked the definition of "extreme" such that we still have a significant number of points.
Next, let's look again at the distributions, and at the "worse" 50% range.
"""

# ╔═╡ 06b25714-a11b-4a77-a029-d3d9fe00d0f5
df_extreme = filter(in_extreme_service_volume, df_in_service_volume);

# ╔═╡ af292295-4fca-43ee-a8ff-3a5513803713
error_plots(unfold_df(df_extreme);
		    title="Corner estimation errors in extreme service volume.")

# ╔═╡ bfc22f95-820a-4507-bc72-f753444cfeda
@info quantile.([(df_extreme |> unfold_df).err_val], [0.25, 0.75])

# ╔═╡ e8533e42-c021-4778-97b8-fd98d9b91789
md"""
We can see that in the "extreme" case it got slightly worse still, however not that much.
"""

# ╔═╡ Cell order:
# ╟─a7d8c731-e529-435f-ad30-aefa0732ed1d
# ╟─c54e4a78-ee13-4b83-aec3-911fd29d157e
# ╟─76e85dc3-974a-40ec-8a69-e33eb43afa66
# ╠═6c0bfb80-e907-46c1-90d9-c10b8ac2414b
# ╟─5e15a03e-3063-406e-ac55-6275a1c8a75a
# ╟─89123779-3cca-4540-955e-3ab03750a671
# ╠═2ca9aebe-95f5-49a8-ada2-a6e56712a8ab
# ╠═c3f2d8f8-04fd-4f45-af0d-eaceb849bee5
# ╠═1cd77f49-5617-408b-a7b1-bc16f502b885
# ╟─25c74665-4c6d-4872-8e4e-86cdbc76ebf6
# ╟─204e5e13-c1d1-412f-9cf2-c1037a1e5cf6
# ╟─169e4d42-fe69-49d3-a2e6-c76d6818bf93
# ╠═7afdbacc-d3f0-4309-aeca-9d3b355396e1
# ╟─f804765c-8e8f-4c80-a98f-e15a6af4e370
# ╠═95145748-e937-44c5-a732-13c36540454c
# ╟─1e558af0-b1c9-458d-9ae8-1e2e407b53ad
# ╟─7890895d-a6a7-467a-a22c-7b34375c906d
# ╟─2848f3aa-257f-4dba-bac0-74fad2449041
# ╠═9f55e07a-fae1-408e-8c20-c204250aafdb
# ╠═7cdd671a-867a-4fe6-9c59-777e1711e182
# ╟─d5a9d032-2f0e-4b7e-b1a1-b4a427c3a475
# ╠═06b25714-a11b-4a77-a029-d3d9fe00d0f5
# ╠═af292295-4fca-43ee-a8ff-3a5513803713
# ╠═bfc22f95-820a-4507-bc72-f753444cfeda
# ╟─e8533e42-c021-4778-97b8-fd98d9b91789
