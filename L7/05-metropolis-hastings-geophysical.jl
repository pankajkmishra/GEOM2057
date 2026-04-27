#---- Imports ----
using CairoMakie
using LinearAlgebra
using Printf
using Random
using Statistics

#---- Shared hard-case setup ----
include(joinpath(@__DIR__, "..", "L6", "05-problem-setup-hard.jl"))
include(joinpath(@__DIR__, "..", "L6", "06-data-generation-hard.jl"))

#---- Figure output ----
const FIGURE_DIR = joinpath(@__DIR__, "Figures")
mkpath(FIGURE_DIR)

#---- Posterior helpers ----
function inside_bounds(model, lower, upper)
    return all(model .>= lower) && all(model .<= upper)
end

function logposterior(model, t, d, sigma_d, lower, upper)
    if !inside_bounds(model, lower, upper)
        return -Inf
    end
    return -0.5 * chi2_d_hard(model, t, d, sigma_d)
end

function run_metropolis_hastings(t, d, sigma_d, lower, upper, start_model, proposal_scale; nsamples = 35_000, seed = 20260426)
    Random.seed!(seed)

    nmodel = length(start_model)
    chain = Matrix{Float64}(undef, nsamples, nmodel)

    current = copy(start_model)
    current_logp = logposterior(current, t, d, sigma_d, lower, upper)
    accepted = 0

    for i in 1:nsamples
        # Propose a nearby model and compare how plausible it is.
        proposal = current .+ proposal_scale .* randn(nmodel)
        proposal_logp = logposterior(proposal, t, d, sigma_d, lower, upper)
        log_alpha = proposal_logp - current_logp

        if log(rand()) < min(0.0, log_alpha)
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        end

        chain[i, :] .= current
    end

    return chain, accepted / nsamples
end

function column_quantiles(matrix, probabilities)
    result = Matrix{Float64}(undef, length(probabilities), size(matrix, 2))
    for j in axes(matrix, 2)
        sorted = sort(matrix[:, j])
        for (i, p) in enumerate(probabilities)
            index = clamp(round(Int, p * (length(sorted) - 1)) + 1, firstindex(sorted), lastindex(sorted))
            result[i, j] = sorted[index]
        end
    end
    return result
end

function histogram_peak(values; nbins = 45)
    edges = collect(range(minimum(values), maximum(values), length = nbins + 1))
    counts = zeros(Int, nbins)
    for value in values
        index = searchsortedlast(edges, value)
        index = clamp(index, 1, nbins)
        counts[index] += 1
    end
    return maximum(counts)
end

function predictive_quantiles(samples, t, probabilities)
    predictions = Matrix{Float64}(undef, size(samples, 1), length(t))
    for i in axes(samples, 1)
        predictions[i, :] .= gg_hard(vec(samples[i, :]), t)
    end
    return column_quantiles(predictions, probabilities)
end

#---- Main demo ----
function main()
    set_theme!(merge(theme_light(), Theme(fontsize = 17, Axis = (xgridvisible = false, ygridvisible = false, xlabelsize = 20, ylabelsize = 20, xticklabelsize = 16, yticklabelsize = 16, xlabelcolor = :gray10, ylabelcolor = :gray10, xticklabelcolor = :gray20, yticklabelcolor = :gray20, titlecolor = :gray10))))

    # Reuse the same hard synthetic problem from Lecture 6.
    case = generate_hard_geophysical_case()
    t = case.t
    mtrue = case.mtrue
    m0 = case.m0
    lower = case.lower
    upper = case.upper
    sigma_d = case.σd
    noise_seed = case.noise_seed
    d = generate_hard_case_observations(mtrue, t, sigma_d, noise_seed)

    # Use small Gaussian steps so the chain explores locally but still moves.
    proposal_scale = [0.12, 0.035, 0.045]
    chain, acceptance_rate = run_metropolis_hastings(t, d, sigma_d, lower, upper, m0, proposal_scale)

    # Ignore the early part of the run before the chain settles into the target region.
    burnin = 5_000
    posterior_samples = chain[(burnin + 1):end, :]
    posterior_mean = vec(mean(posterior_samples, dims = 1))
    posterior_std = vec(std(posterior_samples, dims = 1))
    posterior_q = column_quantiles(posterior_samples, [0.05, 0.50, 0.95])
    predictive_q = predictive_quantiles(posterior_samples[1:20:end, :], t, [0.05, 0.50, 0.95])

    fig = Figure(size = (1450, 1350))

    labels = ["m1", "m2", "m3"]
    colors = [:midnightblue, :saddlebrown, :darkgreen]

    # Left column: how the chain moves through parameter space.
    for j in 1:3
        ax = Axis(fig[j, 1], title = "$(labels[j]) trace", xlabel = "iteration", ylabel = labels[j])
        lines!(ax, 1:size(chain, 1), chain[:, j], color = colors[j], linewidth = 1.0)
        vlines!(ax, [burnin], color = :firebrick4, linestyle = :dash, linewidth = 2)
        hlines!(ax, [mtrue[j]], color = :gray15, linestyle = :dot, linewidth = 2)
    end

    # Right column: what values remain common after burn-in.
    for j in 1:3
        ax = Axis(fig[j, 2], title = "$(labels[j]) distribution", xlabel = labels[j], ylabel = "count")
        hist!(ax, posterior_samples[:, j], bins = 45, color = (colors[j], 0.50), strokewidth = 0)
        vlines!(ax, [mtrue[j]], color = :gray15, linestyle = :dot, linewidth = 2)
        vlines!(ax, [posterior_q[2, j]], color = :firebrick4, linewidth = 2)
        peak_count = histogram_peak(posterior_samples[:, j])
        text!(
            ax,
            minimum(posterior_samples[:, j]),
            0.92 * peak_count,
            text = @sprintf("90%% interval: [%.2f, %.2f]", posterior_q[1, j], posterior_q[3, j]),
            align = (:left, :center),
            fontsize = 17,
            color = :gray10,
        )
    end

    # Bottom panel: uncertainty in the predicted data after propagating the samples.
    ax_predictive = Axis(fig[4, 1:2], title = "Predictive uncertainty", xlabel = "t", ylabel = "d")
    band!(ax_predictive, t, predictive_q[1, :], predictive_q[3, :], color = (:midnightblue, 0.18), label = "90% predictive band")
    lines!(ax_predictive, t, predictive_q[2, :], color = :midnightblue, linewidth = 3, label = "posterior median")
    scatter!(ax_predictive, t, d, color = :gray10, markersize = 11, label = "observed")
    lines!(ax_predictive, t, gg_hard(mtrue, t), color = :gray25, linestyle = :dash, linewidth = 2.5, label = "true response")
    axislegend(ax_predictive, position = :rt)

    Label(fig[0, :], "MCMC traces, parameter uncertainty, and predictive uncertainty", fontsize = 29, font = :bold, color = :gray10)
    save(joinpath(FIGURE_DIR, "L7_05_metropolis_hastings_geophysical.png"), fig, px_per_unit = 2)

    println("Metropolis-Hastings geophysical inversion")
    println("=" ^ 72)
    println("Prior          : bounded uniform over Lecture 6 hard-case bounds")
    println("Likelihood     : Gaussian, Cd = sigma_d^2 I")
    println("Proposal scale : ", format_vector(proposal_scale))
    println("Samples        : ", size(chain, 1), " (burn-in ", burnin, ")")
    @printf("Acceptance rate: %.3f\n", acceptance_rate)
    println("True model     : ", format_vector(mtrue))
    println("Posterior mean : ", format_vector(posterior_mean))
    println("Posterior std  : ", format_vector(posterior_std))
    for j in 1:3
        @printf("%s mean +/- std : %.3f +/- %.3f\n", labels[j], posterior_mean[j], posterior_std[j])
        @printf("%s 90%% interval : [%.3f, %.3f], median %.3f\n", labels[j], posterior_q[1, j], posterior_q[3, j], posterior_q[2, j])
    end
    println("Saved figure: Figures/L7_05_metropolis_hastings_geophysical.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
