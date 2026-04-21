# ================================================================
# Script 05: Hard Problem Setup
# ================================================================

# This script defines a harder comparison case for the stochastic-search part of the lecture.
# The goal is to keep the same overall inverse-problem ideas while making the forward response
# more non-linear and the data more sparse and noisy.
#
# Variable guide
# - t is the sparse vector of sample locations / times.
# - mtrue is the true hard-case model.
# - m0 is the initial model used by PSO and VFSA.
# - gn_start is the deliberately poor starting model for hard-case Gauss-Newton.
# - mref is the reference model used in Φm_hard.
# - lower and upper are bound vectors for the search space.
# - σd is the hard-case data standard deviation.
# - ϵ2 is the regularization weight.
# - noise_seed, pso_seed, and vfsa_seed fix the random sequences for reproducibility.
# - pso_particles and pso_iterations control the PSO search.
# - vfsa_initial_candidates, vfsa_iterations, vfsa_moves_per_iteration, T0, and cooling_rate control VFSA.
# - gn_maxiter and gn_gtol control hard-case Gauss-Newton.
# - model is a generic current model vector in the hard-case helper functions.
#
# Functions defined in this file
# - clamp_model(model, lower, upper): keeps a model inside the allowed bounds.
# - gg_hard(m, t): harder forward model with stronger curvature in m2.
# - G_hard(m, t): Jacobian of the harder forward model.
# - generate_hard_geophysical_case(): returns the full hard-case setup and algorithm settings.
# - hard_data_residual(model, t, d): hard-case residual vector d - g_hard(m).
# - chi2_d_hard(model, t, d, σd): weighted hard-case chi-squared.
# - reduced_chi2_d_hard(model, t, d, σd): hard-case chi-squared per datum.
# - rms_d_hard(model, t, d): hard-case RMS residual.
# - Φm_hard(model, mref): model regularization term for the hard case.
# - objective_hard(model, t, d, σd, mref, ϵ2): total hard-case objective.
# - gradient_hard(model, t, d, σd, mref, ϵ2): gradient of the hard-case objective.
# - H_gauss_newton_hard(model, t, σd, ϵ2): Gauss-Newton Hessian approximation for the hard case.
# - print_hard_case_summary(...): prints the hard-case setup summary.
# - print_fit_summary_hard(...): prints final hard-case fit diagnostics.
# - print_stochastic_objective_note(): explains why PSO and VFSA use Φ(m).
# - main(): prints the standalone hard-case setup summary.

#------ Let's load the baseline helper functions if they are not loaded yet ----------

if !@isdefined(print_summary)
    include("01-problem-setup-derivatives-etc.jl")
end

using LinearAlgebra
using Printf

#------ Let's define a small helper to keep models inside the allowed bounds ----------

clamp_model(model, lower, upper) = min.(max.(model, lower), upper)

#------ Let's define the harder forward map and its Jacobian ----------
# Compared with script 01, m2 now appears inside a squared denominator.
# This sharper curvature makes local linearization more fragile.

function gg_hard(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, m3 = m
    return m1 .* exp.(-(t ./ m2) .^ 2) .+ m3 # note the square in the exponential, in earlier problem it was exp(-(t/m2)) 
end

function G_hard(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, _ = m
    e = exp.(-(t ./ m2) .^ 2)
    J = zeros(length(t), length(m))
    J[:, 1] .= e
    J[:, 2] .= m1 .* e .* (2.0 .* t .^ 2 ./ m2 .^ 3)
    J[:, 3] .= 1.0
    return J
end

#------ Let's package the full hard-case setup in one named tuple ----------
# The tuple contains the true model, starting models, noise level, bounds,
# and method-specific settings for GN, PSO, and VFSA.

function generate_hard_geophysical_case()
    return (
        t = collect(range(0.05, 2.0, length = 8)),
        mtrue = [3.0, 0.35, 0.15],
        m0 = [1.8, 0.55, 0.35],
        gn_start = [0.85, 1.05, 0.80],
        mref = [1.0, 0.8, 0.4],
        lower = [0.5, 0.1, 0.0],
        upper = [5.0, 1.2, 1.0],
        σd = 0.08,
        ϵ2 = 1.0e-4,
        noise_seed = 7,
        pso_seed = 11,
        pso_particles = 24,
        pso_iterations = 40,
        vfsa_seed = 23,
        vfsa_initial_candidates = 48,
        vfsa_iterations = 150,
        vfsa_moves_per_iteration = 30,
        T0 = 5.0,
        cooling_rate = 0.05,
        gn_maxiter = 12,
        gn_gtol = 1.0e-6,
    )
end

#------ Let's define residuals and objective-function pieces for the hard case ----------
# These mirror the same ideas used in script 01, but now for the harder forward model.

hard_data_residual(model, t, d) = d - gg_hard(model, t)

function chi2_d_hard(model, t, d, σd)
    δd = hard_data_residual(model, t, d)
    return dot(δd, δd) / σd^2
end

reduced_chi2_d_hard(model, t, d, σd) = chi2_d_hard(model, t, d, σd) / length(d)

function rms_d_hard(model, t, d)
    δd = hard_data_residual(model, t, d)
    return sqrt(dot(δd, δd) / length(d))
end

function Φm_hard(model, mref)
    δm = mref - model
    return 0.5 * dot(δm, δm)
end

function objective_hard(model, t, d, σd, mref, ϵ2)
    return 0.5 * chi2_d_hard(model, t, d, σd) + ϵ2 * Φm_hard(model, mref)
end

function gradient_hard(model, t, d, σd, mref, ϵ2)
    δd = hard_data_residual(model, t, d)
    δm = mref - model
    return -(G_hard(model, t)' * δd) / σd^2 - ϵ2 .* δm
end

function H_gauss_newton_hard(model, t, σd, ϵ2)
    Gm = G_hard(model, t)
    return (Gm' * Gm) / σd^2 + ϵ2 .* Matrix{Float64}(I, length(model), length(model))
end

#------ Let's print short teaching summaries for the hard case ----------

function print_hard_case_summary(t, mtrue, start_model, mref, lower, upper, σd, ϵ2)
    println("Forward operator: d_i^pre = m1 * exp(-(t_i / m2)^2) + m3")
    println("Sparse, noisy data and stronger curvature in m2 make the local objective much harder for Gauss-Newton.")
    println()
    print_summary(mtrue, start_model, mref, σd, ϵ2)
    println("Number of samples : ", length(t))
    println("Time range        : ", format_number(first(t)), " to ", format_number(last(t)))
    println("Search bounds     : ", format_vector(lower), " to ", format_vector(upper))
    println()
end

function print_fit_summary_hard(title, model, t, d, σd, mref, ϵ2)
    println(title)
    println("-" ^ 72)
    println("mest         = ", format_vector(model))
    println("chi^2_d      = ", @sprintf("%.3f", chi2_d_hard(model, t, d, σd)))
    println("red. chi^2_d = ", @sprintf("%.3f", reduced_chi2_d_hard(model, t, d, σd)))
    println("RMS          = ", @sprintf("%.3f", rms_d_hard(model, t, d)))
    println("Φm(mest)     = ", @sprintf("%.3f", Φm_hard(model, mref)))
    println("Objective    = ", @sprintf("%.3f", objective_hard(model, t, d, σd, mref, ϵ2)))
end

function print_stochastic_objective_note()
    println("Objective used by PSO and VFSA")
    println("-" ^ 72)
    println("They minimize Φ(m) = Φd(m) + ϵ² Φm(m), not only the data misfit χ²d.")
    println("Φd keeps the predicted data close to the observations under Cd = σd² I.")
    println("Φm keeps models from drifting to implausible parts of the bounded search space when the data are sparse and noisy.")
    println("Using the same Φ(m) as Gauss-Newton makes the comparison about search strategy, not about changing the target function.")
    println()
end

#------ Let's run the standalone hard-case setup script ----------

function main()
    (; t, mtrue, m0, gn_start, mref, lower, upper, σd, ϵ2, pso_particles, pso_iterations, vfsa_iterations, vfsa_moves_per_iteration) = generate_hard_geophysical_case()
    println("Hard inverse-problem setup")
    println("=" ^ 72)
    print_hard_case_summary(t, mtrue, gn_start, mref, lower, upper, σd, ϵ2)
    println("Gauss-Newton comparison start : ", format_vector(gn_start))
    println("PSO initial model             : ", format_vector(m0))
    println("PSO iterations / particles    : ", pso_iterations, " / ", pso_particles)
    println("VFSA iterations / moves       : ", vfsa_iterations, " / ", vfsa_moves_per_iteration)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end