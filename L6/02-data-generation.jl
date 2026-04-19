# ================================================================
# Script 02: Data Generation
# ================================================================

# This script generates the deterministic synthetic dataset and prints the observed data table when run directly.
#
# Variable guide
# - t is the vector of sample locations / times.
# - mtrue is the true model used to generate clean synthetic data.
# - m0 is the starting model used later by the inversion scripts.
# - mref is the reference model used in regularization.
# - dclean is the noise-free predicted data gg(mtrue, t).
# - d is the noisy observed data.
# - σd is the noise standard deviation.
# - ϵ2 is the regularization weight passed to later inversion scripts.
# - maxiter and gtol are the default deterministic inversion controls.
#
# Functions defined in this file
# - generate_synthetic_geophysical_case(): builds and returns the full baseline teaching dataset.
# - main(): prints the generated observations when the file is run directly.

#------ Let's load the shared forward model and helper-printing functions ----------

if !@isdefined(gg)
    include("01-problem-setup-derivatives-etc.jl")
end

using Printf
using Random

#------ Let's build one deterministic teaching case ----------
# This bundles the sampling, the true model, the starting model, the prior model,
# and the noisy observations into one named tuple that later scripts can reuse.

function generate_synthetic_geophysical_case()
    Random.seed!(2)

    t = collect(range(0.05, 2.0, length = 30))
    mtrue = [3.0, 0.35, 0.15]
    m0 = [1.5, 0.2, 0.5]
    mref = [1.0, 0.8, 0.4]

    σd = 0.02
    ϵ2 = 1.0e-4
    maxiter = 15
    gtol = 1.0e-6

    dclean = gg(mtrue, t)
    d = dclean .+ σd .* randn(length(t))

    return (
        t = t,
        mtrue = mtrue,
        m0 = m0,
        mref = mref,
        dclean = dclean,
        d = d,
        σd = σd,
        ϵ2 = ϵ2,
        maxiter = maxiter,
        gtol = gtol,
    )
end

#------ Let's print the generated data when this script is run directly ----------

function main()
    (; t, mtrue, m0, mref, d, σd, ϵ2) = generate_synthetic_geophysical_case()

    println("Synthetic data generation")
    println("=" ^ 72)
    print_summary(mtrue, m0, mref, σd, ϵ2)
    println("Number of samples : ", length(t))
    println("Time range        : ", format_number(first(t)), " to ", format_number(last(t)))
    println()
    println("Observed data")
    println("sample\tt\td_obs")
    for i in eachindex(d)
        @printf("%d\t%.3f\t%.3f\n", i, t[i], d[i])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end