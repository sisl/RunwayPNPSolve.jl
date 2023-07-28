### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ d755b7f7-c0fc-400d-9e87-21032bbe1348
# ╠═╡ show_logs = false
import Pkg; Pkg.activate();

# ╔═╡ 6b1e6cb2-2cb8-11ee-10f4-c7d88b502e1c
using CSV, DataFrames, DataFramesMeta

# ╔═╡ a7d8c731-e529-435f-ad30-aefa0732ed1d
begin
using Unitful; using Unitful.DefaultSymbols
using IntervalSets
using StatsBase, Statistics
md"""Imports... (hidden)"""
end

# ╔═╡ c54e4a78-ee13-4b83-aec3-911fd29d157e
begin
import Makie, AlgebraOfGraphics;
import WGLMakie, CairoMakie;
CairoMakie.activate!()
AoG = AlgebraOfGraphics;
end

# ╔═╡ 6c0bfb80-e907-46c1-90d9-c10b8ac2414b
in_service_volume(row) = 
  row.gt_translation_x ∈ (-6000m .. -300m) &&
  atan(row.gt_translation_y / abs(row.gt_translation_x))rad ∈ -3°..3° &&
  atan(row.gt_translation_z / abs(row.gt_translation_x))rad ∈ 1.2°..6°

# ╔═╡ 2ca9aebe-95f5-49a8-ada2-a6e56712a8ab
df_full = @chain CSV.read("/tmp/trl_resultsKSJC.csv", DataFrame) begin
    transform(_, :gt_translation_x=>x->x*1m,  # make "unitful"
                 :gt_translation_y=>y->y*1m,
                 :gt_translation_z=>z->z*1m,
				 [:gt_translation_x, :gt_translation_y] => ByRow(
					 (x, y) -> atan(y / abs(x))rad
				 ) => :horizontal_angle,
				 [:gt_translation_x, :gt_translation_z] => ByRow(
					 (x, z) -> atan(z / abs(x))rad
				 ) => :vertical_angle;
              renamecols=false)
    filter(in_service_volume, _)
end;

# ╔═╡ c3f2d8f8-04fd-4f45-af0d-eaceb849bee5
df_in_service_volume = filter(in_service_volume, df_full);

# ╔═╡ 1cd77f49-5617-408b-a7b1-bc16f502b885
@info "num samples in service volume = $(nrow(df_service_volume))"

# ╔═╡ 25c74665-4c6d-4872-8e4e-86cdbc76ebf6
"Bring dataframe into stacked form for better processing."
unfold_df(df) = vcat([
	select(df, [Symbol("gt_vertices_runway_","$(loc_tb)_$(loc_lr)","_corner_",ax),
		        Symbol("pred_vertices_","$(loc_tb)_$(loc_lr)","_",ax)] 		   
			   => ByRow((val_gt, val_pred)->[val_pred-val_gt, ax, loc_tb, loc_lr])
			   => [:err_val, :axis, :loc_near_far, :loc_left_right])
	for loc_tb in ["top", "bottom"],
	    loc_lr in ["left", "right"],
		ax in ["x", "y"]
]...) |> dropmissing;

# ╔═╡ 204e5e13-c1d1-412f-9cf2-c1037a1e5cf6
function error_plots(df_unfolded)
  plt = AoG.data(df_unfolded)
  plt *= AoG.mapping(:err_val => "Error [px]",
                     col=:loc_left_right,
                     row=:loc_near_far => AoG.renamer(["top"=>"far",
                                                       "bottom"=>"near"]),
                     color=:axis)
  plt *= AoG.density(; datalimits=(arr->2*iqr(arr).*(-1, 1)))
  AoG.draw(plt, facet = (; linkxaxes = :all, linkyaxes = :none))
end; error_plots(df_in_service_volume |> unfold_df)

# ╔═╡ 9f55e07a-fae1-408e-8c20-c204250aafdb
in_extreme_service_volume(row) = 
  row.gt_translation_x ∈ (-6000m .. -5000m) ||
  abs(row.horizontal_angle) ∈ 2.0°..3° ||
  row.vertical_angle ∈ 1.2°..3.0°

# ╔═╡ 7cdd671a-867a-4fe6-9c59-777e1711e182
let df = df_full
@info filter(row->row.gt_translation_x ∈ (-6000m .. -5000m), df) |> nrow
@info filter(row->abs(row.horizontal_angle) ∈ 2.0°..3°, df) |> nrow
@info filter(row->row.vertical_angle ∈ 1.2°..3.0°, df) |> nrow
end

# ╔═╡ 06b25714-a11b-4a77-a029-d3d9fe00d0f5
df_extreme = filter(in_extreme_service_volume, df_full);

# ╔═╡ af292295-4fca-43ee-a8ff-3a5513803713
error_plots(unfold_df(df_extreme))

# ╔═╡ Cell order:
# ╠═d755b7f7-c0fc-400d-9e87-21032bbe1348
# ╠═6b1e6cb2-2cb8-11ee-10f4-c7d88b502e1c
# ╟─a7d8c731-e529-435f-ad30-aefa0732ed1d
# ╟─c54e4a78-ee13-4b83-aec3-911fd29d157e
# ╠═6c0bfb80-e907-46c1-90d9-c10b8ac2414b
# ╠═2ca9aebe-95f5-49a8-ada2-a6e56712a8ab
# ╠═c3f2d8f8-04fd-4f45-af0d-eaceb849bee5
# ╠═1cd77f49-5617-408b-a7b1-bc16f502b885
# ╠═25c74665-4c6d-4872-8e4e-86cdbc76ebf6
# ╠═204e5e13-c1d1-412f-9cf2-c1037a1e5cf6
# ╠═9f55e07a-fae1-408e-8c20-c204250aafdb
# ╠═7cdd671a-867a-4fe6-9c59-777e1711e182
# ╠═06b25714-a11b-4a77-a029-d3d9fe00d0f5
# ╠═af292295-4fca-43ee-a8ff-3a5513803713
