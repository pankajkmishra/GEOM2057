#---- Imports ----
using CairoMakie
using LinearAlgebra
using Printf
using Random
using Statistics

#---- Figure output ----
const FIGURE_DIR = joinpath(@__DIR__, "Figures")
mkpath(FIGURE_DIR)

#---- Density helpers ----
function bivariate_normal_density(x, y, mu, Sigma)
    delta = [x - mu[1], y - mu[2]]
    return exp(-0.5 * dot(delta, Sigma \ delta)) / (2pi * sqrt(det(Sigma)))
end

function correlated_samples(mu, Sigma, nsamples)
    factor = cholesky(Symmetric(Sigma)).L
    noise = factor * randn(2, nsamples)
    return mu .+ noise
end

#---- Main demo ----
function main()
    Random.seed!(20260427)
    set_theme!(merge(theme_light(), Theme(fontsize = 18, Axis = (xgridvisible = false, ygridvisible = false, xlabelsize = 22, ylabelsize = 22, xticklabelsize = 18, yticklabelsize = 18, xlabelcolor = :gray10, ylabelcolor = :gray10, xticklabelcolor = :gray20, yticklabelcolor = :gray20, titlecolor = :gray10))))

    # Create a simple population where height and weight tend to increase together.
    mu = [172.0, 74.0]
    Sigma = [7.5^2 0.72 * 7.5 * 11.0; 0.72 * 7.5 * 11.0 11.0^2]
    nsamples = 2_500
    samples = correlated_samples(mu, Sigma, nsamples)

    # Evaluate the smooth joint density on a grid for the contour panel.
    height_grid = collect(range(148.0, 198.0, length = 180))
    weight_grid = collect(range(40.0, 112.0, length = 180))
    density = [bivariate_normal_density(h, w, mu, Sigma) for h in height_grid, w in weight_grid]

    fig = Figure(size = (1500, 720))

    # Left panel: raw samples make the positive correlation easy to see.
    ax_samples = Axis(
        fig[1, 1],
        title = "Samples",
        xlabel = "height (cm)",
        ylabel = "weight (kg)",
    )
    scatter!(ax_samples, samples[1, :], samples[2, :], color = (:gray10, 0.20), markersize = 5)
    lines!(ax_samples, [mu[1] - 18, mu[1] + 18], [mu[2] - 0.72 * 18 * 11.0 / 7.5, mu[2] + 0.72 * 18 * 11.0 / 7.5], color = :midnightblue, linewidth = 2.5)

    # Right panel: the joint PDF shows where height-weight pairs are most likely.
    ax_joint = Axis(
        fig[1, 2],
        title = "Joint PDF",
        xlabel = "height (cm)",
        ylabel = "weight (kg)",
    )
    contourf!(ax_joint, height_grid, weight_grid, density, levels = 14, colormap = :Greys)
    contour!(ax_joint, height_grid, weight_grid, density, levels = 7, color = (:gray10, 0.85), linewidth = 1.1)

    Label(fig[0, :], "Joint PDF of height and weight", fontsize = 30, font = :bold, color = :gray10)
    save(joinpath(FIGURE_DIR, "L7_02_height_weight_joint_pdf.png"), fig, px_per_unit = 2)

    correlation = cor(samples[1, :], samples[2, :])
    println("Height-weight joint PDF")
    println("=" ^ 72)
    @printf("Sample mean height/weight : %.3f / %.3f\n", mean(samples[1, :]), mean(samples[2, :]))
    @printf("Sample correlation        : %.3f\n", correlation)
    println("Saved figure: Figures/L7_02_height_weight_joint_pdf.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end