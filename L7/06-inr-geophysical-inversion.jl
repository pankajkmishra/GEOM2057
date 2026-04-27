#---- Imports ----
using CairoMakie
using LinearAlgebra
using Printf
using Random
using Statistics

#---- Figure output ----
const FIGURE_DIR = joinpath(@__DIR__, "Figures")
mkpath(FIGURE_DIR)

#---- Synthetic Earth model ----
function true_property(x, z)
    background = 2.1 + 0.35z
    layer = 0.45 / (1.0 + exp(-(z - 0.58) / 0.035))
    anomaly = -0.62 * exp(-((x - 0.63)^2 / (2 * 0.08^2) + (z - 0.42)^2 / (2 * 0.07^2)))
    return background + layer + anomaly
end

function feature_row(x, z, freqs)
    values = Float64[1.0, x, z, x * z, x^2, z^2]
    for (kx, kz) in freqs
        phase = 2pi * (kx * x + kz * z)
        push!(values, sin(phase))
        push!(values, cos(phase))
    end
    return values
end

function design_matrix(points, freqs)
    nfeatures = length(feature_row(points[1][1], points[1][2], freqs))
    A = Matrix{Float64}(undef, length(points), nfeatures)
    for i in eachindex(points)
        A[i, :] .= feature_row(points[i][1], points[i][2], freqs)
    end
    return A
end

function fit_inr(points, values, freqs; lambda = 2.0e-4)
    A = design_matrix(points, freqs)
    return (A' * A + lambda * I) \ (A' * values)
end

function evaluate_inr(theta, x, z, freqs)
    return dot(theta, feature_row(x, z, freqs))
end

function nearest_voxel_value(x, z, voxel_values, xcenters, zcenters)
    ix = argmin(abs.(xcenters .- x))
    iz = argmin(abs.(zcenters .- z))
    return voxel_values[ix, iz]
end

#---- Main demo ----
function main()
    Random.seed!(20260426)
    set_theme!(merge(theme_light(), Theme(fontsize = 17, Axis = (xgridvisible = false, ygridvisible = false, xlabelsize = 20, ylabelsize = 20, xticklabelsize = 16, yticklabelsize = 16, xlabelcolor = :gray10, ylabelcolor = :gray10, xticklabelcolor = :gray20, yticklabelcolor = :gray20, titlecolor = :gray10))))

    # Create a smooth synthetic property field as the target we want to represent.
    xfine = collect(range(0, 1, length = 180))
    zfine = collect(range(0, 1, length = 150))
    true_grid = [true_property(x, z) for x in xfine, z in zfine]

    # First build a coarse voxel version of the same field.
    nx_voxel = 12
    nz_voxel = 8
    xcenters = collect(range(0.5 / nx_voxel, 1 - 0.5 / nx_voxel, length = nx_voxel))
    zcenters = collect(range(0.5 / nz_voxel, 1 - 0.5 / nz_voxel, length = nz_voxel))
    voxel_values = [true_property(x, z) for x in xcenters, z in zcenters]
    voxel_grid = [nearest_voxel_value(x, z, voxel_values, xcenters, zcenters) for x in xfine, z in zfine]

    # Then build an INR that maps coordinates directly to property values.
    freqs = [
        (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0),
        (2.0, 0.0), (0.0, 2.0), (2.0, 1.0), (1.0, 2.0),
        (3.0, 0.0), (0.0, 3.0), (2.0, -1.0), (3.0, 2.0),
        (4.0, 1.0), (1.0, 4.0), (3.0, -2.0), (4.0, -1.0),
    ]

    training_points = [(x, z) for x in xcenters for z in zcenters]
    training_values = [true_property(x, z) for (x, z) in training_points]
    theta = fit_inr(training_points, training_values, freqs)
    inr_grid = [evaluate_inr(theta, x, z, freqs) for x in xfine, z in zfine]

    # Compare the two representations along one horizontal slice.
    zsection = 0.43
    true_section = [true_property(x, zsection) for x in xfine]
    voxel_section = [nearest_voxel_value(x, zsection, voxel_values, xcenters, zcenters) for x in xfine]
    inr_section = [evaluate_inr(theta, x, zsection, freqs) for x in xfine]

    # Add source, receivers, and collocation points to suggest the physics-informed idea.
    collocation_x = rand(180)
    collocation_z = rand(180)
    receiver_x = collect(range(0.08, 0.92, length = 9))
    receiver_z = fill(0.02, length(receiver_x))
    source_x = [0.50]
    source_z = [0.03]

    fig = Figure(size = (1700, 1050))
    colormap = :viridis
    colorrange = (minimum(true_grid), maximum(true_grid))

    # Top row: compare the reference field, a voxel model, and the INR field.
    ax_true = Axis(fig[1, 1], title = "Reference", xlabel = "x", ylabel = "z", aspect = DataAspect(), yreversed = true)
    heatmap!(ax_true, xfine, zfine, true_grid, colormap = colormap, colorrange = colorrange)

    ax_voxel = Axis(fig[1, 2], title = "Voxel model", xlabel = "x", ylabel = "z", aspect = DataAspect(), yreversed = true)
    heatmap!(ax_voxel, xfine, zfine, voxel_grid, colormap = colormap, colorrange = colorrange)
    vlines!(ax_voxel, range(0, 1, length = nx_voxel + 1), color = (:white, 0.6), linewidth = 0.8)
    hlines!(ax_voxel, range(0, 1, length = nz_voxel + 1), color = (:white, 0.6), linewidth = 0.8)

    ax_inr = Axis(fig[1, 3], title = "INR model", xlabel = "x", ylabel = "z", aspect = DataAspect(), yreversed = true)
    hm = heatmap!(ax_inr, xfine, zfine, inr_grid, colormap = colormap, colorrange = colorrange)
    Colorbar(fig[1, 4], hm, label = "property")

    # Bottom row: compare one profile, the physics-point idea, and unknown counts.
    ax_section = Axis(fig[2, 1], title = @sprintf("Section z = %.2f", zsection), xlabel = "x", ylabel = "property")
    lines!(ax_section, xfine, true_section, color = :gray15, linewidth = 3, label = "reference")
    lines!(ax_section, xfine, voxel_section, color = :saddlebrown, linewidth = 3, linestyle = :dash, label = "voxel")
    lines!(ax_section, xfine, inr_section, color = :midnightblue, linewidth = 3, label = "INR")
    axislegend(ax_section, position = :lb)

    ax_piml = Axis(fig[2, 2], title = "Physics points", xlabel = "x", ylabel = "z", aspect = DataAspect(), yreversed = true)
    heatmap!(ax_piml, xfine, zfine, true_grid, colormap = colormap, colorrange = colorrange)
    scatter!(ax_piml, collocation_x, collocation_z, color = (:white, 0.55), markersize = 5, label = "physics residual")
    scatter!(ax_piml, receiver_x, receiver_z, color = :gray15, marker = :utriangle, markersize = 14, label = "receivers")
    scatter!(ax_piml, source_x, source_z, color = :firebrick4, marker = :star5, markersize = 24, label = "source")
    axislegend(ax_piml, position = :rb)

    ax_params = Axis(fig[2, 3], title = "Unknown counts", xticks = (1:2, ["voxel cells", "INR weights"]), ylabel = "count")
    barplot!(ax_params, 1:2, [length(voxel_values), length(theta)], color = [:saddlebrown, :midnightblue])
    text!(ax_params, 1, length(voxel_values) + 3, text = string(length(voxel_values)), align = (:center, :bottom), fontsize = 20, color = :gray10)
    text!(ax_params, 2, length(theta) + 3, text = string(length(theta)), align = (:center, :bottom), fontsize = 20, color = :gray10)

    Label(fig[0, :], "Voxel vs INR", fontsize = 30, font = :bold, color = :gray10)
    save(joinpath(FIGURE_DIR, "L7_06_inr_geophysical_inversion.png"), fig, px_per_unit = 2)

    voxel_rmse = sqrt(mean((vec(voxel_grid) .- vec(true_grid)) .^ 2))
    inr_rmse = sqrt(mean((vec(inr_grid) .- vec(true_grid)) .^ 2))

    println("Implicit neural representation demo")
    println("=" ^ 72)
    println("Voxel unknowns : ", length(voxel_values))
    println("INR weights    : ", length(theta))
    @printf("Voxel RMSE on fine grid: %.4f\n", voxel_rmse)
    @printf("INR RMSE on fine grid  : %.4f\n", inr_rmse)
    println("Saved figure: Figures/L7_06_inr_geophysical_inversion.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
