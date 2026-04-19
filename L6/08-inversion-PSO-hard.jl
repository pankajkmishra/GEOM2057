# ================================================================
# Script 08: PSO on the Hard Case
# ================================================================

# This script solves the hard inverse problem with Particle Swarm Optimisation (PSO).
# Students should read it as a derivative-free search over many candidate models.
#
# Variable guide
# - t is the sparse vector of sample locations / times.
# - d is the hard-case observed data vector.
# - m0 is the initial model inserted into the swarm.
# - mref is the reference model used in the objective.
# - lower and upper are the search bounds for all particles.
# - σd is the assumed data standard deviation.
# - ϵ2 is the regularization weight.
# - positions stores the current swarm particle locations.
# - velocities stores the particle velocities.
# - Φparticles stores each particle's current objective value.
# - personal_best_positions and personal_best_values store each particle's best-so-far model and objective.
# - global_best and global_best_value store the best-so-far model found by the whole swarm.
# - ω, c1, and c2 are the PSO inertia and attraction coefficients.
# - pso_seed, pso_particles, and pso_iterations control the stochastic search.
# - Φ_history stores the all-time global-best objective values.
# - dpred is the final predicted dataset gg_hard(mest, t).
#
# Functions defined in this file
# - describe_pso_hard_case(...): prints the hard-case setup and PSO controls before the search.
# - swarm_statistics(positions): computes simple mean and spread summaries for the swarm.
# - solve_pso_hard(...): runs the PSO search and returns the best model found.
# - main(): loads the hard case, runs PSO, and prints diagnostics.

#------ Let's load the hard-case setup and shared noisy observations ----------

using Printf
using Random
using Statistics

include("05-problem-setup-hard.jl")
include("06-data-generation-hard.jl")

#------ Let's explain the PSO case before the search starts ----------

function describe_pso_hard_case(t, mtrue, m0, mref, lower, upper, σd, ϵ2, pso_particles, pso_iterations)
    println()
    println("Hard inverse problem for PSO")
    println("=" ^ 72)
    print_hard_case_summary(t, mtrue, m0, mref, lower, upper, σd, ϵ2)
    print_stochastic_objective_note()
    println("PSO particles     : ", pso_particles)
    println("PSO iterations    : ", pso_iterations)
    println("The current swarm-best Φ can increase from one iteration to the next because particles keep exploring.")
    println("The all-time global best Φ cannot increase because PSO stores the best model found so far.")
    println()
end

#------ Let's compute simple swarm statistics for reporting ----------
# The mean is not used in the update, but the spread σm is useful in class because it
# shows whether the swarm is still exploring widely or has contracted to a small region.

function swarm_statistics(positions)
    M = reduce(hcat, positions)
    return vec(mean(M, dims = 2)), vec(std(M, dims = 2, corrected = false))
end

#------ Let's run the PSO search ----------
# Each particle remembers its own best model, and the swarm also remembers the best
# model found by any particle at any time.

function solve_pso_hard(t, d, m0, mref, lower, upper, σd, ϵ2, pso_seed, pso_particles, pso_iterations)
    Random.seed!(pso_seed)

    nmodel = length(m0)
    Δv = 0.20 .* (upper - lower)
    positions = [lower .+ rand(nmodel) .* (upper - lower) for _ in 1:pso_particles]
    velocities = [Δv .* (2.0 .* rand(nmodel) .- 1.0) for _ in 1:pso_particles]
    positions[1] = copy(m0)

    Φparticles = [objective_hard(position, t, d, σd, mref, ϵ2) for position in positions]
    personal_best_positions = deepcopy(positions)
    personal_best_values = copy(Φparticles)

    global_index = argmin(personal_best_values)
    global_best = copy(personal_best_positions[global_index])
    global_best_value = personal_best_values[global_index]
    Φ_history = Float64[global_best_value]
    Φparticle_previous = minimum(Φparticles)

    ω = 0.70
    c1 = 1.50
    c2 = 1.50

    println("PSO inversion")
    println("-" ^ 72)
    println("Iteration log: current/global best weighted fits, swarm spread, and whether the current swarm-best got worse.")
    for iter in 1:pso_iterations
        for particle in eachindex(positions)
            r1 = rand(nmodel)
            r2 = rand(nmodel)
            velocities[particle] =
                ω .* velocities[particle] .+
                c1 .* r1 .* (personal_best_positions[particle] - positions[particle]) .+
                c2 .* r2 .* (global_best - positions[particle])

            positions[particle] = clamp_model(positions[particle] + velocities[particle], lower, upper)
            value = objective_hard(positions[particle], t, d, σd, mref, ϵ2)
            Φparticles[particle] = value

            if value < personal_best_values[particle]
                personal_best_values[particle] = value
                personal_best_positions[particle] = copy(positions[particle])
            end

            if value < global_best_value
                global_best_value = value
                global_best = copy(positions[particle])
            end
        end

        push!(Φ_history, global_best_value)
        _, σm = swarm_statistics(positions)
        particle_index = argmin(Φparticles)
        particle_best = positions[particle_index]
        Φparticle_best = Φparticles[particle_index]
        ΔΦparticle = Φparticle_best - Φparticle_previous
        @printf(
            "iter=%02d | Φparticle=%.3f | ΔΦparticle=%+.3f | χ²particle=%.3f | Φglobal=%.3f | χ²global=%.3f | σm=%s\n",
            iter,
            Φparticle_best,
            ΔΦparticle,
            chi2_d_hard(particle_best, t, d, σd),
            global_best_value,
            chi2_d_hard(global_best, t, d, σd),
            format_vector(σm),
        )
        Φparticle_previous = Φparticle_best
    end

    println("Stopped after $(pso_iterations) PSO iterations.")
    return global_best, Φ_history
end

#------ Let's run the hard-case PSO example and print the final fit ----------

function main()
    (; t, mtrue, m0, mref, lower, upper, σd, ϵ2, noise_seed, pso_seed, pso_particles, pso_iterations) = generate_hard_geophysical_case()
    d = generate_hard_case_observations(mtrue, t, σd, noise_seed)
    describe_pso_hard_case(t, mtrue, m0, mref, lower, upper, σd, ϵ2, pso_particles, pso_iterations)

    mest, _ = solve_pso_hard(
        t,
        d,
        m0,
        mref,
        lower,
        upper,
        σd,
        ϵ2,
        pso_seed,
        pso_particles,
        pso_iterations,
    )
    dpred = gg_hard(mest, t)

    println()
    print_fit_summary_hard("Final PSO estimate", mest, t, d, σd, mref, ϵ2)
    println("Iterations    = ", pso_iterations)
    println()
    print_data_table(t, d, dpred)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end