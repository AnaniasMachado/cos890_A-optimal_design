using LinearAlgebra

function dual_variable_fixing(UB::Float64, Lambda::AbstractMatrix, tau::Float64, mu::AbstractVector, nu::AbstractVector, k::Integer)
    zeta = dual_objective(Lambda, tau, mu, nu, k)
    gap = UB - zeta

    fixed_zero = findall(mu .> gap)
    fixed_one = findall(nu .> gap)

    return fixed_zero, fixed_one
end

function primal_variable_fixing(UB::Float64, x::AbstractVector, A::AbstractMatrix, k::Integer)
    M = information_matrix(x, A)
    g = gradient(M, A)
    ordering = sortperm(g)
    g_sorted = g[ordering]

    LB = objective(M) - dot(g, x) + sum(g_sorted[1:k])
    gap = UB - LB

    fixed_one = findall(g_sorted[k + 1] .- g .> gap)
    fixed_zero = findall(g .- g_sorted[k] .> gap)

    return fixed_zero, fixed_one
end