using LinearAlgebra
using Printf

function describe_problem()
    println()
    println("Toy geophysical inverse problem")
    println("=" ^ 72)
    println("Intuitive description")
    println("- Imagine a hypothetical experiment where the Earth can be described by three model parameters m = [m1, m2, m3].")
    println("- You go to the field with a geophysical device and measure some observations; let us call them data d.")
    println("- Because this observed response is caused by the Earth, the observed data d are a function of the model vector m.")
    println("- The inversion task is to find m such that the predicted data dpred(m) are as close as possible to the observed data d.")
    println()
    println("Forward model and physics")
    println("- The forward model gives the predicted data for a chosen model m.")
    println("- This forward model is the physics that relates the model parameters to the data.")
    println()
    println("Forward operator: dpre = g(m)")
    println("Model vector    : m = [m1, m2, m3]")
    println("Predicted datum : dᵢ^pre = m1 * exp(-tᵢ / m2) + m3")
    println("Data misfit     : Φd(m) = 0.5 * (d - dpre)' * Cd⁻¹ * (d - dpre)")
    println("Model misfit    : Φm(m) = 0.5 * (m - mref)' * (m - mref)")
    println("Total misfit    : Φ(m) = Φd(m) + ϵ² * Φm(m)")
    println()
    println("Why is this problem non-linear?")
    println("- The model parameter m2 sits inside exp(-t / m2).")
    println("- Because of that, the forward map is not of the form G * m.")
    println("- The Jacobian G(m) changes with the current model m.")
    println()
    println("Newton versus Gauss-Newton")
    println("- Newton uses H = G' * Cd⁻¹ * G - Σᵢ [δdᵢ * ∇²gᵢ(m)] / σd² + ϵ² I")
    println("- Gauss-Newton uses H = G' * Cd⁻¹ * G + ϵ² I")
    println("- So Gauss-Newton drops the residual-weighted second-derivative term.")
    println()
    println("High-level Newton algorithm")
    println("- Choose a starting model m₀.")
    println("- Compute predicted data dpre = g(mᵢ).")
    println("- Compute the gradient ∇Φ(mᵢ) and the full Hessian H(mᵢ).")
    println("- Solve H(mᵢ) Δm = -∇Φ(mᵢ).")
    println("- Update mᵢ₊₁ = mᵢ + α Δm and repeat until convergence.")
    println()
    println("High-level Gauss-Newton algorithm")
    println("- Choose a starting model m₀.")
    println("- Compute predicted data dpre = g(mᵢ).")
    println("- Compute the gradient ∇Φ(mᵢ) and the Gauss-Newton Hessian approximation H(mᵢ) ≈ G' * Cd⁻¹ * G + ϵ² I.")
    println("- Solve H(mᵢ) Δm = -∇Φ(mᵢ).")
    println("- Update mᵢ₊₁ = mᵢ + α Δm and repeat until convergence.")
    println("=" ^ 72)
end

function gg(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, m3 = m
    return m1 .* exp.(-t ./ m2) .+ m3
end

function G(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, _ = m
    e = exp.(-t ./ m2)
    G = zeros(length(t), length(m))
    G[:, 1] .= e
    G[:, 2] .= m1 .* e .* (t ./ m2^2)
    G[:, 3] .= 1.0
    return G
end

function Hgg(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, _ = m
    e = exp.(-t ./ m2)
    ndata = length(t)
    nmodel = length(m)
    Hgg_array = zeros(ndata, nmodel, nmodel)
    mixed = e .* (t ./ m2^2)
    curvature = m1 .* e .* ((t .^ 2 ./ m2^4) .- (2.0 .* t ./ m2^3))
    Hgg_array[:, 1, 2] .= mixed
    Hgg_array[:, 2, 1] .= mixed
    Hgg_array[:, 2, 2] .= curvature
    return Hgg_array
end

function Φd(m, t, d, σd)
    δd = d - gg(m, t)
    return 0.5 * dot(δd, δd) / σd^2
end

function Φm(m, mref)
    δm = m - mref
    return 0.5 * dot(δm, δm)
end

function Φ(m, t, d, σd, mref, ϵ2)
    return Φd(m, t, d, σd) + ϵ2 * Φm(m, mref)
end

function ∇Φ(m, t, d, σd, mref, ϵ2)
    δd = d - gg(m, t)
    return -(G(m, t)' * δd) / σd^2 + ϵ2 .* (m - mref)
end

function H_newton(m, t, d, σd, mref, ϵ2)
    δd = d - gg(m, t)
    Gm = G(m, t)
    H = (Gm' * Gm) / σd^2
    Hgg_array = Hgg(m, t)
    for i in eachindex(t)
        H .-= (δd[i] / σd^2) .* Hgg_array[i, :, :]
    end
    H .+= ϵ2 .* Matrix{Float64}(I, length(m), length(m))
    return H
end

function H_gauss_newton(m, t, d, σd, mref, ϵ2)
    Gm = G(m, t)
    return (Gm' * Gm) / σd^2 + ϵ2 .* Matrix{Float64}(I, length(m), length(m))
end

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

function print_summary(mtrue, m0, mref, σd, ϵ2)
    println("True model     : ", mtrue)
    println("Starting model : ", m0)
    println("Reference model: ", mref)
    println("σd             : ", σd)
    println("ϵ²             : ", ϵ2)
    println()
end

function main()
    describe_problem()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end