using LinearAlgebra
using Printf

include("01-problem-description.jl")
include("02-data-generation.jl")

function solve_newton(case)
    m = copy(case.m0)
    Φ_history = Float64[Φ(m, case.t, case.d, case.σd, case.mref, case.ϵ2)]
    ∇Φ_history = Float64[]

    println("Newton inversion")
    println("-" ^ 72)
    for i in 0:(case.maxiter - 1)
        ∇Φᵢ = ∇Φ(m, case.t, case.d, case.σd, case.mref, case.ϵ2)
        push!(∇Φ_history, norm(∇Φᵢ))
        @printf(
            "iter=%02d | Φ=%.3f | Φd=%.3f | Φm=%.3f | m=%s\n",
            i,
            Φ_history[end],
            Φd(m, case.t, case.d, case.σd),
            Φm(m, case.mref),
            format_vector(m),
        )

        if norm(∇Φᵢ) < case.gtol
            println("Converged in $(i) Newton iterations.")
            return m, Φ_history, ∇Φ_history
        end

        H = H_newton(m, case.t, case.d, case.σd, case.mref, case.ϵ2)
        Δm = -(H \ ∇Φᵢ)
        α = α_backtracking(m, Δm, ∇Φᵢ, case.t, case.d, case.σd, case.mref, case.ϵ2)
        m .+= α .* Δm
        push!(Φ_history, Φ(m, case.t, case.d, case.σd, case.mref, case.ϵ2))
    end

    println("Stopped after $(case.maxiter) Newton iterations.")
    return m, Φ_history, ∇Φ_history
end

function main()
    case = generate_synthetic_geophysical_case()
    describe_problem()
    print_summary(case.mtrue, case.m0, case.mref, case.σd, case.ϵ2)
    mest, Φ_history, ∇Φ_history = solve_newton(case)
    dpred = gg(mest, case.t)

    println()
    println("Final Newton estimate")
    println("-" ^ 72)
    println("mest         = ", format_vector(mest))
    println("Φd(mest)     = ", @sprintf("%.3f", Φd(mest, case.t, case.d, case.σd)))
    println("Φm(mest)     = ", @sprintf("%.3f", Φm(mest, case.mref)))
    println("Φ(mest)      = ", @sprintf("%.3f", Φ(mest, case.t, case.d, case.σd, case.mref, case.ϵ2)))
    println("Iterations    = ", length(∇Φ_history) - (last(∇Φ_history) < case.gtol ? 1 : 0))
    println()
    print_data_table(case.t, case.d, dpred)
end

main()