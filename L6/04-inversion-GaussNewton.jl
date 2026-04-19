# ================================================================
# Script 04: Gauss-Newton Inversion
# ================================================================

# This script solves the deterministic inverse problem with Gauss-Newton iterations and prints the final data fit.
#
# Variable guide
# - t is the vector of sample locations / times.
# - d is the observed data vector.
# - m0 is the initial model for Gauss-Newton iterations.
# - m is the current model estimate during the iteration loop.
# - mest is the final Gauss-Newton estimate.
# - mref is the reference model used in Φm.
# - σd is the assumed data standard deviation.
# - ϵ2 is the regularization weight.
# - Δm is the Gauss-Newton model update.
# - α is the backtracking step length.
# - maxiter and gtol are the stopping controls.
# - Φ_history stores the objective values through the iterations.
# - ∇Φ_history stores the gradient norms through the iterations.
# - dpred is the final predicted dataset gg(mest, t).
#
# Functions defined in this file
# - α_backtracking(...): chooses a safe Gauss-Newton step length by line search.
# - solve_gauss_newton(...): runs the Gauss-Newton inversion and returns the estimate and iteration histories.
# - main(): loads the teaching case, runs Gauss-Newton, and prints diagnostics.

#------ Let's load linear-algebra tools and the shared lecture functions ----------

using LinearAlgebra
using Printf

include("01-problem-setup-derivatives-etc.jl")
include("02-data-generation.jl")

#------ Let's solve the inverse problem with Gauss-Newton ----------
# Gauss-Newton keeps only the G'G part of the Hessian plus regularization.
# This is often simpler and cheaper than Newton's method, but it can be less robust
# when the forward problem is strongly non-linear away from the solution.

#------ Let's compute a safe step length with backtracking line search ----------

function α_backtracking(m, Δm, ∇Φᵢ, t, d, σd, mref, ϵ2)
    α = 1.0
    c = 1.0e-4
    Φᵢ = Φ(m, t, d, σd, mref, ϵ2)
    while α > 1.0e-8
        mtrial = m .+ α .* Δm
        if Φ(mtrial, t, d, σd, mref, ϵ2) <= Φᵢ + c * α * dot(∇Φᵢ, Δm)
            return α
        end
        α *= 0.5
    end
    return α
end

function solve_gauss_newton(t, d, m0, σd, mref, ϵ2, maxiter, gtol)
    m = copy(m0)
    Φ_history = Float64[Φ(m, t, d, σd, mref, ϵ2)]
    ∇Φ_history = Float64[]

    println("Gauss-Newton inversion")
    println("-" ^ 72)
    for i in 0:(maxiter - 1)
        ∇Φᵢ = ∇Φ(m, t, d, σd, mref, ϵ2)
        push!(∇Φ_history, norm(∇Φᵢ))
        @printf(
            "iter=%02d | Φ=%.3f | Φd=%.3f | Φm=%.3f | m=%s\n",
            i,
            Φ_history[end],
            Φd(m, t, d, σd),
            Φm(m, mref),
            format_vector(m),
        )

        if norm(∇Φᵢ) < gtol
            println("Converged in $(i) Gauss-Newton iterations.")
            return m, Φ_history, ∇Φ_history
        end

        H = H_gauss_newton(m, t, d, σd, mref, ϵ2)
        Δm = -(H \ ∇Φᵢ)
        α = α_backtracking(m, Δm, ∇Φᵢ, t, d, σd, mref, ϵ2)
        m .+= α .* Δm
        push!(Φ_history, Φ(m, t, d, σd, mref, ϵ2))
    end

    println("Stopped after $(maxiter) Gauss-Newton iterations.")
    return m, Φ_history, ∇Φ_history
end

#------ Let's run the Gauss-Newton example and print the final diagnostics ----------

function main()
    (; t, d, mtrue, m0, mref, σd, ϵ2, maxiter, gtol) = generate_synthetic_geophysical_case()
    print_summary(mtrue, m0, mref, σd, ϵ2)
    mest, _, ∇Φ_history = solve_gauss_newton(t, d, m0, σd, mref, ϵ2, maxiter, gtol)
    dpred = gg(mest, t)

    println()
    print_fit_summary("Final Gauss-Newton estimate", mest, t, d, σd, mref, ϵ2)
    println("Iterations    = ", length(∇Φ_history) - (last(∇Φ_history) < gtol ? 1 : 0))
    println()
    print_data_table(t, d, dpred)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end