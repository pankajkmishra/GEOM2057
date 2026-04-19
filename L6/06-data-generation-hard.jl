# ================================================================
# Script 06: Hard-Case Data Generation
# ================================================================

# This script generates the sparse and noisy observations used by the hard-case
# Gauss-Newton, PSO, and VFSA comparison scripts.
#
# Variable guide
# - t is the sparse vector of sample locations / times.
# - mtrue is the true hard-case model.
# - m0 is the displayed starting model for the stochastic methods.
# - mref is the hard-case reference model.
# - lower and upper are the bound vectors displayed in the summary.
# - σd is the hard-case data standard deviation.
# - ϵ2 is the regularization weight displayed in the summary.
# - noise_seed fixes the random noise so every algorithm uses the same data.
# - d is the noisy hard-case observed dataset.
#
# Functions defined in this file
# - generate_hard_case_observations(mtrue, t, σd, noise_seed): creates the reproducible noisy hard-case data.
# - main(): prints the hard-case observations when the file is run directly.

#------ Let's load the shared hard-case setup ----------

if !@isdefined(generate_hard_geophysical_case)
    include("05-problem-setup-hard.jl")
end

using Printf
using Random

#------ Let's generate one repeatable noisy dataset for the hard case ----------
# A fixed random seed is used so all inversion methods see the same observations.

function generate_hard_case_observations(mtrue, t, σd, noise_seed)
    Random.seed!(noise_seed)
    return gg_hard(mtrue, t) .+ σd .* randn(length(t))
end

#------ Let's print the hard-case observations when run directly ----------

function main()
    (; t, mtrue, m0, mref, lower, upper, σd, ϵ2, noise_seed) = generate_hard_geophysical_case()
    d = generate_hard_case_observations(mtrue, t, σd, noise_seed)

    println("Hard-case data generation")
    println("=" ^ 72)
    print_hard_case_summary(t, mtrue, m0, mref, lower, upper, σd, ϵ2)
    println("Observed data")
    println("sample\tt\td_obs")
    for i in eachindex(d)
        @printf("%d\t%.3f\t%.3f\n", i, t[i], d[i])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end