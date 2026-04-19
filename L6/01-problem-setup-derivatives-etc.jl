# This script contains the shared core functions for the toy non-linear
# inverse problem used throughout Lecture 6.
#
# Problem setup
# - The Earth is represented by a three-parameter model vector
#   m = [m1, m2, m3].
# - The synthetic observed data are generated from the non-linear forward map
#   g(m), with predicted datum
#       d_i^pre = m1 * exp(-t_i / m2) + m3.
# - The parameter m2 appears inside the exponential denominator, so the
#   forward problem is non-linear and the Jacobian changes with the model.
#
# Objective function used in the scripts
# - Data misfit:
#       Φd(m) = 0.5 * (dobs - g(m))' * Cd^-1 * (dobs - g(m))
# - Model/prior misfit:
#       Φm(m) = 0.5 * (f(mref) - f(m))' * (f(mref) - f(m))
# - Total objective:
#       Φ(m) = Φd(m) + ϵ² * Φm(m)
#
# Classroom simplifications
# - The scripts use Cd = σd² I, so the data misfit reduces to a weighted
#   least-squares / chi-squared form.
# - The scripts use f(m) = m, so F = ∂f/∂m = I and the prior term reduces to
#   a simple distance from the reference model mref.
#
# Variable guide
# - m = [m1, m2, m3] is the model vector.
# - t is the vector of sample locations / times.
# - d is the observed data vector.
# - mtrue is the true model used to create synthetic data.
# - m0 is the starting model for inversion.
# - mref is the reference model used in the regularization term.
# - σd is the assumed data standard deviation.
# - ϵ2 is the regularization weight.
# - δd = d - g(m) is the data residual.
# - δm = mref - m is the model residual.
# - Cd = σd² I is the data covariance.
# - Φd, Φm, Φ are the data misfit, model misfit, and total objective.
#
# Functions defined in this file
# - gg(m, t): forward model g(m).
# - G(m, t): Jacobian of the forward model.
# - Hgg(m, t): second-derivative tensor of the forward model.
# - data_residual(m, t, d): residual vector d - g(m).
# - ff(m): regularization map, here equal to the identity.
# - FF(m): Jacobian of the regularization map, here the identity matrix.
# - model_residual(m, mref): model-side residual mref - m.
# - chi2_d(m, t, d, σd): weighted chi-squared data misfit.
# - reduced_chi2_d(m, t, d, σd): chi-squared per datum.
# - rms_d(m, t, d): root-mean-square data residual.
# - Φd(m, t, d, σd): scalar data objective.
# - Φm(m, mref): scalar model objective.
# - Φ(m, t, d, σd, mref, ϵ2): total objective.
# - ∇Φ(m, t, d, σd, mref, ϵ2): gradient of the total objective.
# - H_newton(m, t, d, σd, mref, ϵ2): full Newton Hessian.
# - H_gauss_newton(m, t, d, σd, mref, ϵ2): Gauss-Newton Hessian approximation.
# - format_number(value): short numeric formatter.
# - format_vector(values): short vector formatter.
# - print_summary(...): prints the core inverse-problem setup.
# - print_fit_summary(...): prints final fit diagnostics.
# - print_model_table(...): prints a model comparison table.
# - print_data_table(...): prints observed and predicted data together.
# - main(): runs this setup file directly as a small demonstration.
using LinearAlgebra
using Printf

#------ Let's define the forward model g(m) ----------

function gg(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, m3 = m
    return m1 .* exp.(-t ./ m2) .+ m3
end

#------ Let's compute the Jacobian G(m) of the forward model ----------

function G(m::Vector{Float64}, t::Vector{Float64})
    m1, m2, _ = m
    e = exp.(-t ./ m2)
    G = zeros(length(t), length(m))
    G[:, 1] .= e
    G[:, 2] .= m1 .* e .* (t ./ m2^2)
    G[:, 3] .= 1.0
    return G
end

#------ Let's compute the second-derivative tensor of g(m) for Newton's method ----------

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

# Commented teaching extension: same three-parameter structure, but stronger curvature in m2.
# Keep this inactive so the core examples stay simple.
#
# Predicted datum : dᵢ^pre = m1 * exp(-(tᵢ / m2)^2) + m3
# Why harder for Gauss-Newton?
# - m2 now appears inside a squared denominator, so the curvature changes more sharply.
# - The residual-weighted second-derivative term becomes more important away from the solution.
# - Newton can benefit more from the full Hessian, while Gauss-Newton can struggle from a poor start.
#
# function gg_harder(m::Vector{Float64}, t::Vector{Float64})
#     m1, m2, m3 = m
#     return m1 .* exp.(-(t ./ m2) .^ 2) .+ m3
# end
#
# function G_harder(m::Vector{Float64}, t::Vector{Float64})
#     m1, m2, _ = m
#     e = exp.(-(t ./ m2) .^ 2)
#     G = zeros(length(t), length(m))
#     G[:, 1] .= e
#     G[:, 2] .= m1 .* e .* (2.0 .* t .^ 2 ./ m2 .^ 3)
#     G[:, 3] .= 1.0
#     return G
# end
#
# function Hgg_harder(m::Vector{Float64}, t::Vector{Float64})
#     m1, m2, _ = m
#     e = exp.(-(t ./ m2) .^ 2)
#     ndata = length(t)
#     nmodel = length(m)
#     Hgg_array = zeros(ndata, nmodel, nmodel)
#     mixed = e .* (2.0 .* t .^ 2 ./ m2 .^ 3)
#     curvature = m1 .* e .* ((2.0 .* t .^ 2 ./ m2 .^ 3) .^ 2 .- 6.0 .* t .^ 2 ./ m2 .^ 4)
#     Hgg_array[:, 1, 2] .= mixed
#     Hgg_array[:, 2, 1] .= mixed
#     Hgg_array[:, 2, 2] .= curvature
#     return Hgg_array
# end

#------ Let's compute the data residual d - g(m) ----------

data_residual(m, t, d) = d - gg(m, t)

#------ Let's define the model-side regularization map f(m) and its Jacobian F ----------
# In this classroom example, f(m) = m, so F = I.

#------ Let's define the regularization map f(m) ----------

ff(m::Vector{Float64}) = copy(m)

#------ Let's compute the Jacobian F(m) of the regularization map ----------

function FF(m::Vector{Float64})
    return Matrix{Float64}(I, length(m), length(m))
end

#------ Let's compute the model residual f(mref) - f(m) ----------

model_residual(m, mref) = ff(mref) - ff(m)

#------ Let's compute the weighted chi-squared data misfit ----------

function chi2_d(m, t, d, σd)
    δd = data_residual(m, t, d)
    return dot(δd, δd) / σd^2
end

#------ Let's compute the reduced chi-squared data misfit ----------

reduced_chi2_d(m, t, d, σd) = chi2_d(m, t, d, σd) / length(d)

#------ Let's compute the RMS data residual ----------

function rms_d(m, t, d)
    δd = data_residual(m, t, d)
    return sqrt(dot(δd, δd) / length(d))
end

#------ Let's compute the data-misfit contribution Phi_d(m) ----------

function Φd(m, t, d, σd)
    return 0.5 * chi2_d(m, t, d, σd)
end

#------ Let's compute the model-misfit contribution Phi_m(m) ----------

function Φm(m, mref)
    δm = model_residual(m, mref)
    return 0.5 * dot(δm, δm)
end

#------ Let's compute the total objective function Phi(m) ----------

function Φ(m, t, d, σd, mref, ϵ2)
    return Φd(m, t, d, σd) + ϵ2 * Φm(m, mref)
end

#------ Let's compute gradients that we will use in Newton and Gauss-Newton ----------

function ∇Φ(m, t, d, σd, mref, ϵ2)
    δd = data_residual(m, t, d)
    δm = model_residual(m, mref)
    return -(G(m, t)' * δd) / σd^2 - ϵ2 .* (FF(m)' * δm)
end

#------ Let's compute Hessians that we will use later ----------
# H_newton keeps the full second-derivative information.
# H_gauss_newton drops the residual-weighted second-derivative term.

function H_newton(m, t, d, σd, mref, ϵ2)
    δd = data_residual(m, t, d)
    Gm = G(m, t)
    H = (Gm' * Gm) / σd^2
    Hgg_array = Hgg(m, t)
    for i in eachindex(t)
        H .-= (δd[i] / σd^2) .* Hgg_array[i, :, :]
    end
    H .+= ϵ2 .* (FF(m)' * FF(m))
    return H
end

function H_gauss_newton(m, t, d, σd, mref, ϵ2)
    Gm = G(m, t)
    return (Gm' * Gm) / σd^2 + ϵ2 .* (FF(m)' * FF(m))
end

#------ Let's format one number for printing ----------

format_number(value) = @sprintf("%.3f", value)

#------ Let's format one vector for printing ----------

function format_vector(values)
    return "[" * join([@sprintf("%.3f", value) for value in values], ", ") * "]"
end

#------ Let's print a short summary of the case setup ----------

function print_summary(mtrue, m0, mref, σd, ϵ2)
    println("True model     : ", format_vector(mtrue))
    println("Starting model : ", format_vector(m0))
    println("Reference model: ", format_vector(mref))
    println("σd             : ", format_number(σd))
    println("ϵ²             : ", @sprintf("%.3e", ϵ2))
    println("Cd             : σd² I")
    println("Cm             : I")
    println()
end

#------ Let's print the final fit summary after inversion ----------

function print_fit_summary(title, mest, t, d, σd, mref, ϵ2)
    println(title)
    println("-" ^ 72)
    println("mest         = ", format_vector(mest))
    println("chi^2_d      = ", @sprintf("%.3f", chi2_d(mest, t, d, σd)))
    println("red. chi^2_d = ", @sprintf("%.3f", reduced_chi2_d(mest, t, d, σd)))
    println("RMS          = ", @sprintf("%.3f", rms_d(mest, t, d)))
    println("Φd(mest)     = ", @sprintf("%.3f", Φd(mest, t, d, σd)))
    println("Φm(mest)     = ", @sprintf("%.3f", Φm(mest, mref)))
    println("Φ(mest)      = ", @sprintf("%.3f", Φ(mest, t, d, σd, mref, ϵ2)))
end

#------ Let's print a table comparing true, start, prior, and estimated models ----------

function print_model_table(mtrue, m0, mref, mest)
    println("Model summary")
    println("parameter\ttrue\tstart\tprior\testimated")
    for i in eachindex(mtrue)
        @printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\n", i, mtrue[i], m0[i], mref[i], mest[i])
    end
    println()
end

#------ Let's print a table comparing observed and predicted data ----------

function print_data_table(t, d, dpred)
    println("Data comparison")
    println("sample\tt\td_obs\td_pred(mest)")
    for i in eachindex(d)
        @printf("%d\t%.3f\t%.3f\t%.3f\n", i, t[i], d[i], dpred[i])
    end
    println()
end

#------ Let's run the standalone core-function script ----------

function main()
    println("Core inverse-problem functions loaded. See the header comments in this file for the full problem description.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end