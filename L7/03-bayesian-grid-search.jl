#---- Imports ----
using CairoMakie
using LinearAlgebra
using Printf
using Statistics

#---- Shared hard-case setup ----
include(joinpath(@__DIR__, "..", "L6", "05-problem-setup-hard.jl"))
include(joinpath(@__DIR__, "..", "L6", "06-data-generation-hard.jl"))

#---- Figure output ----
const FIGURE_DIR = joinpath(@__DIR__, "Figures")
mkpath(FIGURE_DIR)

#---- Posterior summaries ----
function weighted_quantile(values, weights, q)
    order = sortperm(values)
    sorted_values = values[order]
    sorted_weights = weights[order] ./ sum(weights)
    cumulative = cumsum(sorted_weights)
    index = searchsortedfirst(cumulative, q)
    return sorted_values[clamp(index, firstindex(sorted_values), lastindex(sorted_values))]
end

function marginal_summary(grid, probabilities)
    mean_value = sum(grid .* probabilities)
    q05 = weighted_quantile(grid, probabilities, 0.05)
    q50 = weighted_quantile(grid, probabilities, 0.50)
    q95 = weighted_quantile(grid, probabilities, 0.95)
    return mean_value, q05, q50, q95
end

function posterior_grid(t, d, sigma_d, lower, upper)
    m1_grid = collect(range(lower[1], upper[1], length = 90))
    m2_grid = collect(range(lower[2], upper[2], length = 90))
    m3_grid = collect(range(lower[3], upper[3], length = 70))

    # Evaluate the posterior on a coarse grid so the idea stays transparent.
    logp = Array{Float64}(undef, length(m1_grid), length(m2_grid), length(m3_grid))
    best_logp = -Inf
    best_index = CartesianIndex(1, 1, 1)

    for i in eachindex(m1_grid), j in eachindex(m2_grid), k in eachindex(m3_grid)
        model = [m1_grid[i], m2_grid[j], m3_grid[k]]
        value = -0.5 * chi2_d_hard(model, t, d, sigma_d)
        logp[i, j, k] = value
        if value > best_logp
            best_logp = value
            best_index = CartesianIndex(i, j, k)
        end
    end

    weights = exp.(logp .- maximum(logp))
    posterior = weights ./ sum(weights)
    map_model = [m1_grid[best_index[1]], m2_grid[best_index[2]], m3_grid[best_index[3]]]

    return m1_grid, m2_grid, m3_grid, posterior, map_model
end

#---- Main demo ----
function main()
    set_theme!(merge(theme_light(), Theme(fontsize = 18, Axis = (xgridvisible = false, ygridvisible = false, xlabelsize = 21, ylabelsize = 21, xticklabelsize = 17, yticklabelsize = 17, xlabelcolor = :gray10, ylabelcolor = :gray10, xticklabelcolor = :gray20, yticklabelcolor = :gray20, titlecolor = :gray10))))

    # Reuse the Lecture 6 hard problem so the only new idea is probability.
    (; t, mtrue, mref, lower, upper, sigma_d, epsilon2, noise_seed) = let case = generate_hard_geophysical_case()
        (
            t = case.t,
            mtrue = case.mtrue,
            mref = case.mref,
            lower = case.lower,
            upper = case.upper,
            sigma_d = case.σd,
            epsilon2 = case.ϵ2,
            noise_seed = case.noise_seed,
        )
    end
    d = generate_hard_case_observations(mtrue, t, sigma_d, noise_seed)

    m1_grid, m2_grid, m3_grid, posterior, map_model = posterior_grid(t, d, sigma_d, lower, upper)

    p_m1_m2 = dropdims(sum(posterior, dims = 3), dims = 3)
    p_m1 = dropdims(sum(posterior, dims = (2, 3)), dims = (2, 3))
    p_m2 = dropdims(sum(posterior, dims = (1, 3)), dims = (1, 3))
    p_m3 = dropdims(sum(posterior, dims = (1, 2)), dims = (1, 2))

    mean_m1, m1_q05, m1_q50, m1_q95 = marginal_summary(m1_grid, p_m1)
    mean_m2, m2_q05, m2_q50, m2_q95 = marginal_summary(m2_grid, p_m2)
    mean_m3, m3_q05, m3_q50, m3_q95 = marginal_summary(m3_grid, p_m3)
    posterior_mean = [mean_m1, mean_m2, mean_m3]

    fig = Figure(size = (1600, 980))

    # Left side: one joint slice of the posterior.
    ax_heat = Axis(fig[1:2, 1], title = "Joint posterior", xlabel = "m1", ylabel = "m2", aspect = DataAspect())
    hm = heatmap!(ax_heat, m1_grid, m2_grid, p_m1_m2, colormap = :viridis)
    contour!(ax_heat, m1_grid, m2_grid, p_m1_m2, levels = 8, color = (:gray10, 0.85), linewidth = 1.2)
    scatter!(ax_heat, [mtrue[1]], [mtrue[2]], color = :white, strokecolor = :black, strokewidth = 1.5, markersize = 18, label = "true")
    scatter!(ax_heat, [map_model[1]], [map_model[2]], color = :firebrick4, marker = :xcross, markersize = 20, label = "MAP")
    axislegend(ax_heat, position = :rt)
    Colorbar(fig[1:2, 2], hm, label = "probability")

    # Right side: one-parameter summaries extracted from the same posterior grid.
    ax_m1 = Axis(fig[1, 3], title = "m1", xlabel = "m1", ylabel = "probability")
    lines!(ax_m1, m1_grid, p_m1, color = :midnightblue, linewidth = 3)
    vlines!(ax_m1, [mtrue[1]], color = :gray15, linestyle = :dash, linewidth = 2)
    vlines!(ax_m1, [map_model[1]], color = :firebrick4, linewidth = 2)
    band!(ax_m1, [m1_q05, m1_q95], [0.0, 0.0], [maximum(p_m1), maximum(p_m1)], color = (:midnightblue, 0.10))

    ax_m2 = Axis(fig[2, 3], title = "m2", xlabel = "m2", ylabel = "probability")
    lines!(ax_m2, m2_grid, p_m2, color = :midnightblue, linewidth = 3)
    vlines!(ax_m2, [mtrue[2]], color = :gray15, linestyle = :dash, linewidth = 2)
    vlines!(ax_m2, [map_model[2]], color = :firebrick4, linewidth = 2)
    band!(ax_m2, [m2_q05, m2_q95], [0.0, 0.0], [maximum(p_m2), maximum(p_m2)], color = (:midnightblue, 0.10))

    ax_m3 = Axis(fig[3, 3], title = "m3", xlabel = "m3", ylabel = "probability")
    lines!(ax_m3, m3_grid, p_m3, color = :midnightblue, linewidth = 3)
    vlines!(ax_m3, [mtrue[3]], color = :gray15, linestyle = :dash, linewidth = 2)
    vlines!(ax_m3, [map_model[3]], color = :firebrick4, linewidth = 2)
    band!(ax_m3, [m3_q05, m3_q95], [0.0, 0.0], [maximum(p_m3), maximum(p_m3)], color = (:midnightblue, 0.10))

    # Bottom-left: compare observed data with the response of the MAP model.
    ax_data = Axis(fig[3, 1:2], title = "Data fit", xlabel = "t", ylabel = "d")
    scatter!(ax_data, t, d, color = :gray10, markersize = 12, label = "observed")
    lines!(ax_data, t, gg_hard(mtrue, t), color = :gray25, linestyle = :dash, linewidth = 3, label = "true response")
    lines!(ax_data, t, gg_hard(map_model, t), color = :firebrick4, linewidth = 3, label = "MAP response")
    axislegend(ax_data, position = :rt)

    Label(fig[0, :], "Grid posterior", fontsize = 30, font = :bold, color = :gray10)
    save(joinpath(FIGURE_DIR, "L7_03_bayesian_grid_search.png"), fig, px_per_unit = 2)

    println("Bayesian grid search")
    println("=" ^ 72)
    println("Prior          : bounded uniform over Lecture 6 hard-case bounds")
    println("Likelihood     : Gaussian, Cd = sigma_d^2 I")
    println("Reference mref : ", format_vector(mref), " (not used by the uniform-prior grid posterior)")
    println("Regularization epsilon^2 from Lecture 6: ", @sprintf("%.3e", epsilon2))
    println("True model     : ", format_vector(mtrue))
    println("MAP model      : ", format_vector(map_model))
    println("Posterior mean : ", format_vector(posterior_mean))
    @printf("m1 90%% interval : [%.3f, %.3f], median %.3f\n", m1_q05, m1_q95, m1_q50)
    @printf("m2 90%% interval : [%.3f, %.3f], median %.3f\n", m2_q05, m2_q95, m2_q50)
    @printf("m3 90%% interval : [%.3f, %.3f], median %.3f\n", m3_q05, m3_q95, m3_q50)
    println("Saved figure: Figures/L7_03_bayesian_grid_search.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
