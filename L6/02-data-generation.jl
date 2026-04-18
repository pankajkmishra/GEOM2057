# ================================================================
# Script 02: Data Generation
# ================================================================

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