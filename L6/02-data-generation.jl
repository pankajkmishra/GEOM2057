# ================================================================
# Script 02: Data Generation
# ================================================================

if !@isdefined(gg)
    include("01-problem-description.jl")
end

using Printf
using Random

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

function main()
    case = generate_synthetic_geophysical_case()

    println("Synthetic data generation")
    println("=" ^ 72)
    print_summary(case.mtrue, case.m0, case.mref, case.σd, case.ϵ2)
    println("Number of samples : ", length(case.t))
    println("Time range        : ", format_number(first(case.t)), " to ", format_number(last(case.t)))
    println()
    println("Observed data")
    println("sample\tt\td_obs")
    for i in eachindex(case.d)
        @printf("%d\t%.3f\t%.3f\n", i, case.t[i], case.d[i])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end