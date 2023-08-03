using DataFrames, CSV
using DataFramesMeta
using Plots, StatsPlots

fname = joinpath("/tmp", "trl_resultsKSJC.csv")
df = @chain CSV.read(fname, DataFrame) begin
    @subset -6000 .< :gt_translation_x .< 300
end

location = ["top_left", "top_right", "bottom_left", "bottom_right"]
for loc in location, ax in ["x", "y"]
    error = Symbol("$(loc)_$(ax)_pixel_error")
    pred, gt = Symbol("pred_vertices_$(loc)_$(ax)"), Symbol("gt_vertices_runway_$(loc)_corner_$(ax)")
    @transform!(df, $error = $pred - $gt)
    dropmissing!(df, error)
end


# observation: x error is distributed with "Cauchy" distribution (heavy-tailed)
p, d = let vals = df[!, :bottom_left_x_pixel_error]
    p = plot(; xlims=quantile.([vals], [0.03, 0.95]))
    int = Interval(quantile.([vals], [0.08, 0.92])...)
    # d = fit_mle(Normal, vals)
    d = fit(Cauchy, vals)
    # d = fit(TDist, vals)
    @show d
    histogram!(p, vals; normalize=true)
    plot!(p, d; label="Cauchy");
    p, d
end
plot!(p, Normal(d.μ, d.σ); label="Normal direct.")
plot!(p, Normal(median(vals), iqr(vals)); label="Normal robust.")
p

let vals = df[!, :top_left_x_pixel_error]
    p = plot(; xlims=(0, quantile(vals, 0.99)))
    # int = Interval(quantile.([vals], [0.08, 0.92])...)
    # d = fit_mle(Normal, vals, float(vals.∈[int]))
    # plot!(p, d);
    histogram!(p, abs.(vals); normalize=true)
    p
end

[:gt_vertices_runway_top_left_corner_x,:pred_vertices_top_left_x] =>
                ByRow((val_gt, val_pred)->[val_pred-val_gt, :x, :far, :left]) =>
                [:err_val, :axis, :loc_near_far, :loc_left_right]
