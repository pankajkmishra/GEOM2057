#---- Imports ----
using CairoMakie
using Printf
using Random
using Statistics

#---- Figure output ----
const FIGURE_DIR = joinpath(@__DIR__, "Figures")
mkpath(FIGURE_DIR)

#---- Probability helpers ----
function normal_pdf(x, mu, sigma)
    return exp(-0.5 * ((x - mu) / sigma)^2) / (sigma * sqrt(2pi))
end

function trapezoid_integral(x, y)
    return sum(0.5 .* (y[1:(end - 1)] .+ y[2:end]) .* diff(x))
end

#---- Main demo ----
function main()
    Random.seed!(20260427)
    set_theme!(merge(theme_light(), Theme(fontsize = 18, Axis = (xgridvisible = false, ygridvisible = false, xlabelsize = 22, ylabelsize = 22, xticklabelsize = 18, yticklabelsize = 18, xlabelcolor = :gray10, ylabelcolor = :gray10, xticklabelcolor = :gray20, yticklabelcolor = :gray20, titlecolor = :gray10))))

    # Build a simple synthetic height population so the PDF idea stays familiar.
    mu = 172.0
    sigma = 7.5
    nsamples = 8_000
    heights = mu .+ sigma .* randn(nsamples)

    # Use a smooth Gaussian only as an intuitive reference curve.
    xs = collect(range(145.0, 200.0, length = 500))
    pdf_values = [normal_pdf(x, mu, sigma) for x in xs]

    # Highlight one interval so students can connect probability with area.
    interval_low = 170.0
    interval_high = 175.0
    interval_x = collect(range(interval_low, interval_high, length = 150))
    interval_y = [normal_pdf(x, mu, sigma) for x in interval_x]
    interval_probability = trapezoid_integral(interval_x, interval_y)

    fig = Figure(size = (1500, 720))

    # Left panel: sampled heights and the smooth PDF on top of them.
    ax_hist = Axis(
        fig[1, 1],
        title = "Height samples",
        xlabel = "height (cm)",
        ylabel = "density",
    )
    hist!(ax_hist, heights, bins = 28, normalization = :pdf, color = (:slategray4, 0.30), strokewidth = 0)
    lines!(ax_hist, xs, pdf_values, color = :midnightblue, linewidth = 3)
    vlines!(ax_hist, [mu], color = :gray15, linestyle = :dash, linewidth = 1.5)

    # Right panel: the shaded area is the probability of landing in the interval.
    ax_pdf = Axis(
        fig[1, 2],
        title = "Area gives probability",
        xlabel = "height (cm)",
        ylabel = "p(h)",
    )
    lines!(ax_pdf, xs, pdf_values, color = :midnightblue, linewidth = 3)
    band!(ax_pdf, interval_x, zeros(length(interval_x)), interval_y, color = (:midnightblue, 0.18))
    vlines!(ax_pdf, [interval_low, interval_high], color = :gray15, linestyle = :dot, linewidth = 1.2)
    text!(
        ax_pdf,
        147.5,
        maximum(pdf_values) * 0.93,
        text = @sprintf("P(170 <= H <= 175) = %.3f", interval_probability),
        align = (:left, :center),
        fontsize = 20,
        color = :gray10,
    )

    Label(fig[0, :], "Heights to PDF", fontsize = 30, font = :bold, color = :gray10)
    save(joinpath(FIGURE_DIR, "L7_01_height_pdf_intuition.png"), fig, px_per_unit = 2)

    println("Height PDF intuition")
    println("=" ^ 72)
    @printf("Sample mean/std          : %.3f / %.3f\n", mean(heights), std(heights))
    @printf("P(170 <= H <= 175)       : %.3f\n", interval_probability)
    println("Saved figure: Figures/L7_01_height_pdf_intuition.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
