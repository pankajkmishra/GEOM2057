#---- Imports ----
using CairoMakie
using Printf
using Random
using Statistics

#---- Figure output ----
const FIGURE_DIR = joinpath(@__DIR__, "Figures")
mkpath(FIGURE_DIR)

#---- Markov-chain helpers ----
function simulate_chain(P, start_index, nsteps)
    states = Vector{Int}(undef, nsteps + 1)
    states[1] = start_index
    for i in 1:nsteps
        cumulative = cumsum(P[states[i], :])
        states[i + 1] = searchsortedfirst(cumulative, rand())
    end
    return states
end

function stationary_distribution(P; niter = 10_000)
    pi = fill(1.0 / size(P, 1), size(P, 1))
    for _ in 1:niter
        pi = vec(pi' * P)
    end
    return pi ./ sum(pi)
end

function running_frequencies(chain, nstates)
    frequencies = zeros(length(chain), nstates)
    counts = zeros(nstates)
    for i in eachindex(chain)
        counts[chain[i]] += 1
        frequencies[i, :] .= counts ./ i
    end
    return frequencies
end

function recent_days(index, window)
    start_index = max(1, index - window + 1)
    return collect((start_index:index) .- 1), start_index:index
end

#---- Main demo ----
function main()
    Random.seed!(20260426)
    set_theme!(merge(theme_light(), Theme(fontsize = 18, Axis = (xgridvisible = false, ygridvisible = false, xlabelsize = 21, ylabelsize = 21, xticklabelsize = 17, yticklabelsize = 17, xlabelcolor = :gray10, ylabelcolor = :gray10, xticklabelcolor = :gray20, yticklabelcolor = :gray20, titlecolor = :gray10))))

    # A tiny weather model is enough to show the idea of long-run behavior.
    names = ["Sunny", "Cloudy", "Rainy"]
    P = [
        0.7 0.2 0.1
        0.3 0.4 0.3
        0.2 0.3 0.5
    ]

    row_sums = vec(sum(P, dims = 2))
    if any(abs.(row_sums .- 1.0) .> 1.0e-12)
        error("Transition matrix rows must sum to 1.")
    end

    pi_stationary = stationary_distribution(P)
    nsteps = 900
    preview_days = 45
    animation_stride = 10
    chain = simulate_chain(P, 2, nsteps)
    frequencies = running_frequencies(chain, length(names))
    final_frequency = frequencies[end, :]
    state_counts = [count(==(i), chain) for i in 1:length(names)]
    preview_x, preview_index = recent_days(preview_days + 1, preview_days + 1)

    fig = Figure(size = (1500, 900))

    # Top panel: one realization of the chain.
    ax_path = Axis(fig[1, 1:2], title = "One weather history", xlabel = "day", ylabel = "state", yticks = (1:3, names))
    colors = [:goldenrod4, :gray35, :midnightblue]
    lines!(ax_path, preview_x, chain[preview_index], color = :gray10, linewidth = 2.5)
    scatter!(ax_path, preview_x, chain[preview_index], color = [colors[i] for i in chain[preview_index]], markersize = 11)

    # Bottom-left: the running frequencies settle toward a stable split.
    ax_run = Axis(fig[2, 1], title = "Frequencies settle", xlabel = "day", ylabel = "frequency")
    for i in 1:3
        lines!(ax_run, 0:nsteps, frequencies[:, i], color = colors[i], linewidth = 2.5, label = names[i])
        hlines!(ax_run, [pi_stationary[i]], color = colors[i], linestyle = :dash, linewidth = 1.5)
    end
    axislegend(ax_run, position = :rt)

    # Bottom-right: compare the final sample frequencies with the stationary values.
    ax_bar = Axis(fig[2, 2], title = "Final frequencies", xticks = (1:3, names), ylabel = "probability")
    barplot!(ax_bar, 1:3, final_frequency, color = colors)
    scatter!(ax_bar, 1:3, pi_stationary, color = :gray10, marker = :circle, markersize = 16, label = "stationary")
    axislegend(ax_bar, position = :rt)

    Label(fig[0, :], "Why a Markov chain settles", fontsize = 30, font = :bold, color = :gray10)
    save(joinpath(FIGURE_DIR, "L7_04_markov_chain.png"), fig, px_per_unit = 2)

    frame_index = Observable(preview_days + 1)
    anim_title = lift(frame_index) do index
        return "Weather chain convergence: day $(index - 1)"
    end
    path_x = lift(frame_index) do index
        x, _ = recent_days(index, preview_days + 1)
        x
    end
    path_index = lift(frame_index) do index
        _, idx = recent_days(index, preview_days + 1)
        idx
    end
    run_x = lift(frame_index) do index
        collect(0:(index - 1))
    end
    current_freq = lift(frame_index) do index
        vec(frequencies[index, :])
    end

    fig_anim = Figure(size = (1500, 900))
    Label(fig_anim[0, :], anim_title, fontsize = 30, font = :bold, color = :gray10)

    ax_path_anim = Axis(fig_anim[1, 1:2], title = "Recent weather", xlabel = "day", ylabel = "state", yticks = (1:3, names))
    lines!(ax_path_anim, path_x, lift(indexes -> chain[indexes], path_index), color = :gray10, linewidth = 2.5)
    scatter!(ax_path_anim, path_x, lift(indexes -> chain[indexes], path_index), color = lift(indexes -> [colors[i] for i in chain[indexes]], path_index), markersize = 11)

    ax_run_anim = Axis(fig_anim[2, 1], title = "Running frequencies", xlabel = "day", ylabel = "frequency")
    for i in 1:3
        lines!(ax_run_anim, run_x, lift(index -> frequencies[1:index, i], frame_index), color = colors[i], linewidth = 2.5, label = names[i])
        hlines!(ax_run_anim, [pi_stationary[i]], color = colors[i], linestyle = :dash, linewidth = 1.5)
    end
    axislegend(ax_run_anim, position = :rt)

    ax_bar_anim = Axis(fig_anim[2, 2], title = "Current frequencies", xticks = (1:3, names), ylabel = "probability")
    barplot!(ax_bar_anim, 1:3, current_freq, color = colors)
    scatter!(ax_bar_anim, 1:3, pi_stationary, color = :gray10, marker = :circle, markersize = 16, label = "stationary")
    axislegend(ax_bar_anim, position = :rt)

    frame_indices = collect((preview_days + 1):animation_stride:length(chain))
    record(fig_anim, joinpath(FIGURE_DIR, "L7_04_markov_chain.gif"), frame_indices; framerate = 10) do index
        frame_index[] = index
    end

    println("Markov chain demo")
    println("=" ^ 72)
    println("Chain length: ", nsteps, " days")
    println("Rows sum to: ", join([@sprintf("%.1f", value) for value in row_sums], ", "))
    println("All transition probabilities are positive, so the finite chain is irreducible and aperiodic.")
    println("Long-chain state counts:")
    for i in 1:3
        @printf("  %-6s %d\n", names[i], state_counts[i])
    end
    println("Stationary distribution:")
    for i in 1:3
        @printf("  %-6s %.4f\n", names[i], pi_stationary[i])
    end
    println("Saved figure: Figures/L7_04_markov_chain.png")
    println("Saved animation: Figures/L7_04_markov_chain.gif")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
