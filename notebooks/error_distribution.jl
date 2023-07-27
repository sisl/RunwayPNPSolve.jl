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
end

# ╔═╡ 6c0bfb80-e907-46c1-90d9-c10b8ac2414b
in_service_volume(row) = 
  row.gt_translation_x ∈ (-6000m .. -300m) &&
  atan(row.gt_translation_y / abs(row.gt_translation_x))rad ∈ -3°..3° &&
  atan(row.gt_translation_z / abs(row.gt_translation_x))rad ∈ 1.2°..6°

# ╔═╡ 2ca9aebe-95f5-49a8-ada2-a6e56712a8ab
df = @chain CSV.read("/tmp/trl_resultsKSJC.csv", DataFrame) begin
    transform(_, :gt_translation_x=>x->x*1m,
                 :gt_translation_y=>y->y*1m,
                 :gt_translation_z=>z->z*1m;
              renamecols=false)
    #filter(in_service_volume, _)
end;

# ╔═╡ c54e4a78-ee13-4b83-aec3-911fd29d157e
begin
import Makie, WGLMakie, AlgebraOfGraphics;
AoG = AlgebraOfGraphics;
end

# ╔═╡ f5c0d527-e425-4ade-ad46-5529961e7b17
begin
df2 = DataFrame((err_x=[1;2;3], err_y=[10;20;30], err_z=[-1;-2;-3], err_k=[-10;-20;-30]))
#AoG.data(df2)*AoG.mapping([:err_x, :err_y],col=AoG.dims(1))*AoG.mapping([:err_z, :err_k], row=AoG.dims(1))*AoG.visual(AoG.Scatter);
AoG.data(df2)*AoG.mapping([:err_x, :err_y, :err_z, :err_k], col=AoG.dims(1))*
              AoG.visual(AoG.Scatter) |> AoG.draw
end

# ╔═╡ 087bf3b3-b7f0-4671-bb00-9e1c78a65bca
begin
df3 = DataFrame((err_x=[1;2;3], err_y=[10;20;30], err_z=[-1;-2;-3], err_k=[-10;-20;-30]))
transform(df3, :err_x=>ByRow(x->[x^2,x^3])=>[:lhs, :rhs])
end

# ╔═╡ 25c74665-4c6d-4872-8e4e-86cdbc76ebf6
unfold_df(df) = vcat([
	select(df, [Symbol("gt_vertices_runway_","$(loc_tb)_$(loc_lr)","_corner_",ax),
		        Symbol("pred_vertices_","$(loc_tb)_$(loc_lr)","_",ax)] 		   
			   => ByRow((val_gt, val_pred)->[val_pred-val_gt, ax, loc_tb, loc_lr])
			   => [:err_val, :axis, :loc_near_far, :loc_left_right])
	for loc_tb in ["top", "bottom"],
	    loc_lr in ["left", "right"],
		ax in ["x", "y"]
]...) |> dropmissing

# ╔═╡ c225016c-8a7e-4c39-94b2-0eac9ce84fff
df_unfolded = unfold_df(df)

# ╔═╡ 204e5e13-c1d1-412f-9cf2-c1037a1e5cf6
begin
  compute_lims =
  plt = AoG.data(df_unfolded)
  plt *= AoG.mapping(:err_val => "Error [px]",
                     col=:loc_left_right,
                     row=:loc_near_far => AoG.renamer(["top"=>"far",
                                                       "bottom"=>"near"]),
                     color=:axis)
  plt *= AoG.density(; datalimits=(arr->1.2*iqr(arr).*(-1, 1)))
  AoG.draw(plt, facet = (; linkxaxes = :all, linkyaxes = :none))
end

# ╔═╡ Cell order:
# ╠═d755b7f7-c0fc-400d-9e87-21032bbe1348
# ╠═6b1e6cb2-2cb8-11ee-10f4-c7d88b502e1c
# ╠═a7d8c731-e529-435f-ad30-aefa0732ed1d
# ╠═6c0bfb80-e907-46c1-90d9-c10b8ac2414b
# ╠═2ca9aebe-95f5-49a8-ada2-a6e56712a8ab
# ╠═c54e4a78-ee13-4b83-aec3-911fd29d157e
# ╠═f5c0d527-e425-4ade-ad46-5529961e7b17
# ╠═087bf3b3-b7f0-4671-bb00-9e1c78a65bca
# ╠═25c74665-4c6d-4872-8e4e-86cdbc76ebf6
# ╠═c225016c-8a7e-4c39-94b2-0eac9ce84fff
# ╠═204e5e13-c1d1-412f-9cf2-c1037a1e5cf6
