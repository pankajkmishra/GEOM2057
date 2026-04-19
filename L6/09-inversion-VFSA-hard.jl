# ================================================================
# Script 09: VFSA on the Hard Case
# ================================================================

# This script solves the same hard inverse problem with Very Fast Simulated Annealing (VFSA).
# Students should read it as one stochastic search chain that proposes random moves and
# sometimes accepts worse models to avoid getting trapped too early.
#
# Variable guide
# - t is the sparse vector of sample locations / times.
# - d is the hard-case observed data vector.
# - m0 is the initial model inserted into the initial VFSA candidate set.
# - mref is the reference model used in the objective.
# - lower and upper are the search bounds for all proposals.
# - σd is the assumed data standard deviation.
# - ϵ2 is the regularization weight.
# - current is the current VFSA model in the Markov chain.
# - best is the best model found so far.
# - proposal is a candidate model generated from the current model.
# - Φcurrent, Φproposal, and Φbest are the corresponding objective values.
# - T is the current temperature.
# - vfsa_seed, vfsa_initial_candidates, vfsa_iterations, vfsa_moves_per_iteration, T0, and cooling_rate control the search.
# - uphill_accepts and max_uphill_ΔΦ track accepted worse moves for teaching purposes.
# - Φ_history stores the best-so-far objective values.
# - dpred is the final predicted dataset gg_hard(mest, t).
#
# Functions defined in this file
# - describe_vfsa_hard_case(...): prints the hard-case setup and VFSA controls before the search.
# - vfsa_perturbation(T): generates one heavy-tailed random perturbation scaled by temperature.
# - propose_vfsa_model(current, lower, upper, T): builds one bounded VFSA proposal.
# - solve_vfsa_hard(...): runs the VFSA search and returns the best model found.
# - main(): loads the hard case, runs VFSA, and prints diagnostics.

#------ Let's load the hard-case setup and shared noisy observations ----------

using Printf
using Random

include("05-problem-setup-hard.jl")
include("06-data-generation-hard.jl")

#------ Let's explain the VFSA case before the search starts ----------

function describe_vfsa_hard_case(t, mtrue, m0, mref, lower, upper, σd, ϵ2, vfsa_initial_candidates, vfsa_iterations, vfsa_moves_per_iteration, T0)
    println()
    println("Hard inverse problem for VFSA")
    println("=" ^ 72)
    print_hard_case_summary(t, mtrue, m0, mref, lower, upper, σd, ϵ2)
    print_stochastic_objective_note()
    println("Initial candidates : ", vfsa_initial_candidates)
    println("VFSA iterations    : ", vfsa_iterations)
    println("VFSA moves/iter    : ", vfsa_moves_per_iteration)
    println("Initial T          : ", format_number(T0))
    println("VFSA can accept models with larger Φ(m) early in the run so it can escape local minima.")
    println()
end

#------ Let's define the VFSA proposal mechanism ----------
# The perturbation is heavy-tailed, so the method can still make occasional large jumps,
# especially while the temperature is high.

function vfsa_perturbation(T)
    Tscaled = max(T, 1.0e-6)
    u = rand()
    return sign(u - 0.5) * Tscaled * ((1.0 + 1.0 / Tscaled)^abs(2.0 * u - 1.0) - 1.0)
end

function propose_vfsa_model(current, lower, upper, T)
    proposal = similar(current)
    for i in eachindex(current)
        proposal[i] = current[i] + vfsa_perturbation(T) * (upper[i] - lower[i])
    end
    return clamp_model(proposal, lower, upper)
end

#------ Let's run the VFSA search ----------
# At each iteration the temperature is reduced, making the search gradually less willing
# to accept worse models and more focused on refinement.

function solve_vfsa_hard(t, d, m0, mref, lower, upper, σd, ϵ2, vfsa_seed, vfsa_initial_candidates, vfsa_iterations, vfsa_moves_per_iteration, T0, cooling_rate)
    Random.seed!(vfsa_seed)

    candidates = [copy(m0)]
    append!(
        candidates,
        [lower .+ rand(length(m0)) .* (upper - lower) for _ in 1:(vfsa_initial_candidates - 1)],
    )

    Φcandidates = [objective_hard(candidate, t, d, σd, mref, ϵ2) for candidate in candidates]
    current = copy(candidates[argmin(Φcandidates)])
    Φcurrent = objective_hard(current, t, d, σd, mref, ϵ2)
    Φcurrent_previous = Φcurrent
    best = copy(current)
    Φbest = Φcurrent
    Φ_history = Float64[Φbest]

    println("VFSA inversion")
    println("-" ^ 72)
    println("Iteration log: current and best Φ, uphill acceptances, and the best model.")
    for iter in 1:vfsa_iterations
        T = max(T0 / (1.0 + cooling_rate * (iter - 1)), 1.0e-3)
        uphill_accepts = 0
        max_uphill_ΔΦ = 0.0
        for _ in 1:vfsa_moves_per_iteration
            proposal = propose_vfsa_model(current, lower, upper, T)
            Φproposal = objective_hard(proposal, t, d, σd, mref, ϵ2)
            ΔΦ = Φproposal - Φcurrent

            if ΔΦ <= 0.0 || rand() < exp(-ΔΦ / T)
                if ΔΦ > 0.0
                    uphill_accepts += 1
                    max_uphill_ΔΦ = max(max_uphill_ΔΦ, ΔΦ)
                end
                current = proposal
                Φcurrent = Φproposal
            end

            if Φcurrent < Φbest
                best = copy(current)
                Φbest = Φcurrent
            end
        end

        push!(Φ_history, Φbest)
        ΔΦcurrent = Φcurrent - Φcurrent_previous
        @printf(
            "iter=%03d | T=%.3f | Φcurrent=%.3f | ΔΦcurrent=%+.3f | uphill=%02d | maxΔΦup=%.3f | Φbest=%.3f | χ²d(best)=%.3f | mbest=%s\n",
            iter,
            T,
            Φcurrent,
            ΔΦcurrent,
            uphill_accepts,
            max_uphill_ΔΦ,
            Φbest,
            chi2_d_hard(best, t, d, σd),
            format_vector(best),
        )
        Φcurrent_previous = Φcurrent
    end

    println("Stopped after $(vfsa_iterations) VFSA iterations.")
    return best, Φ_history
end

#------ Let's run the hard-case VFSA example and print the final fit ----------

function main()
    (; t, mtrue, m0, mref, lower, upper, σd, ϵ2, noise_seed, vfsa_seed, vfsa_initial_candidates, vfsa_iterations, vfsa_moves_per_iteration, T0, cooling_rate) = generate_hard_geophysical_case()
    d = generate_hard_case_observations(mtrue, t, σd, noise_seed)
    describe_vfsa_hard_case(t, mtrue, m0, mref, lower, upper, σd, ϵ2, vfsa_initial_candidates, vfsa_iterations, vfsa_moves_per_iteration, T0)

    mest, _ = solve_vfsa_hard(
        t,
        d,
        m0,
        mref,
        lower,
        upper,
        σd,
        ϵ2,
        vfsa_seed,
        vfsa_initial_candidates,
        vfsa_iterations,
        vfsa_moves_per_iteration,
        T0,
        cooling_rate,
    )
    dpred = gg_hard(mest, t)

    println()
    print_fit_summary_hard("Final VFSA estimate", mest, t, d, σd, mref, ϵ2)
    println("Iterations    = ", vfsa_iterations)
    println()
    print_data_table(t, d, dpred)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end