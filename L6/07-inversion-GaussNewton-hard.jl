# ================================================================
# Script 07: Gauss-Newton on the Hard Case
# ================================================================

# This script applies Gauss-Newton to the harder inverse problem so students can see
# how a local method behaves when the objective surface is more difficult.
#
# Variable guide
# - t is the sparse vector of sample locations / times.
# - d is the hard-case observed data vector.
# - gn_start is the poor initial model for this local solve.
# - m is the current model estimate during the iteration loop.
# - mest is the final hard-case Gauss-Newton estimate.
# - mref is the reference model used in Φm_hard.
# - lower and upper are the search bounds enforced after each step.
# - σd is the assumed data standard deviation.
# - ϵ2 is the regularization weight.
# - Δm is the Gauss-Newton model update.
# - α is the backtracking step length.
# - gn_maxiter and gn_gtol are the stopping controls.
# - Φ_history stores the hard-case objective values.
# - ∇Φ_history stores the gradient norms.
# - dpred is the final predicted dataset gg_hard(mest, t).
#
# Functions defined in this file
# - α_backtracking_hard(...): chooses a safe bounded step length for hard-case Gauss-Newton.
# - solve_gauss_newton_hard(...): runs the hard-case Gauss-Newton inversion.
# - main(): loads the hard case, runs the solve, and prints diagnostics.

#------ Let's load the hard-case setup and shared noisy observations ----------

using LinearAlgebra
using Printf

include("05-problem-setup-hard.jl")
include("06-data-generation-hard.jl")

#------ Let's solve the hard case with local Gauss-Newton iterations ----------
# The algorithm is the same basic idea as script 04, but now the harder forward map,
# sparse data, and poor starting model make convergence more sensitive.

#------ Let's compute a safe step length with backtracking line search ----------

function α_backtracking_hard(model, Δm, ∇Φᵢ, t, d, σd, mref, ϵ2, lower, upper)
    α = 1.0
    c = 1.0e-4
    Φᵢ = objective_hard(model, t, d, σd, mref, ϵ2)
    while α > 1.0e-8
        mtrial = clamp_model(model .+ α .* Δm, lower, upper)
        if objective_hard(mtrial, t, d, σd, mref, ϵ2) <= Φᵢ + c * α * dot(∇Φᵢ, Δm)
            return α
        end
        α *= 0.5
    end
    return α
end

function solve_gauss_newton_hard(t, d, mstart, σd, mref, ϵ2, lower, upper, maxiter, gtol)
    m = copy(mstart)
    Φ_history = Float64[objective_hard(m, t, d, σd, mref, ϵ2)]
    ∇Φ_history = Float64[]

    println("Gauss-Newton on the hard case")
    println("-" ^ 72)
    for i in 0:(maxiter - 1)
        ∇Φᵢ = gradient_hard(m, t, d, σd, mref, ϵ2)
        push!(∇Φ_history, norm(∇Φᵢ))
        @printf(
            "iter=%02d | Objective=%.3f | χ²d=%.3f | Φm=%.3f | m=%s\n",
            i,
            Φ_history[end],
            chi2_d_hard(m, t, d, σd),
            Φm_hard(m, mref),
            format_vector(m),
        )

        if norm(∇Φᵢ) < gtol
            println("Converged in $(i) Gauss-Newton iterations.")
            return m, Φ_history, ∇Φ_history
        end

        H = H_gauss_newton_hard(m, t, σd, ϵ2)
        Δm = -(H \ ∇Φᵢ)
        α = α_backtracking_hard(m, Δm, ∇Φᵢ, t, d, σd, mref, ϵ2, lower, upper)
        m = clamp_model(m .+ α .* Δm, lower, upper)
        push!(Φ_history, objective_hard(m, t, d, σd, mref, ϵ2))
    end

    println("Stopped after $(maxiter) Gauss-Newton iterations.")
    return m, Φ_history, ∇Φ_history
end

#------ Let's run the hard-case Gauss-Newton comparison ----------

function main()
    (; t, mtrue, gn_start, mref, lower, upper, σd, ϵ2, noise_seed, gn_maxiter, gn_gtol) = generate_hard_geophysical_case()
    d = generate_hard_case_observations(mtrue, t, σd, noise_seed)

    println()
    println("Hard-case comparison for local Gauss-Newton")
    println("=" ^ 72)
    print_hard_case_summary(t, mtrue, gn_start, mref, lower, upper, σd, ϵ2)
    println("Gauss-Newton is started from a poor local model to show how the local linearization can get trapped.")
    println()

    mest, _, ∇Φ_history = solve_gauss_newton_hard(
        t,
        d,
        gn_start,
        σd,
        mref,
        ϵ2,
        lower,
        upper,
        gn_maxiter,
        gn_gtol,
    )
    dpred = gg_hard(mest, t)

    println()
    print_fit_summary_hard("Final Gauss-Newton estimate", mest, t, d, σd, mref, ϵ2)
    println("Iterations    = ", length(∇Φ_history) - (last(∇Φ_history) < gn_gtol ? 1 : 0))
    println()
    print_data_table(t, d, dpred)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end