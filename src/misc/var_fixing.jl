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

function primal_variable_fixing(UB::Float64, x::AbstractVector, A::AbstractMatrix, k::Integer, fixed_one::AbstractVector{<:Integer}=Int[])
    remaining = k - length(fixed_one)
    free = setdiff(collect(eachindex(x)), fixed_one)

    remaining < 0 && error("More than k variables are fixed to one.")
    remaining > length(free) && error("Not enough free variables remain.")

    if remaining == 0 || remaining == length(free)
        return Int[], Int[]
    end

    M = information_matrix(x, A)
    g = gradient(M, A)

    ordering = sortperm(g[free])
    sorted_free = free[ordering]
    g_sorted = g[sorted_free]

    LB = objective(M) - dot(g, x) + sum(g[fixed_one]) + sum(g_sorted[1:remaining])

    gap = UB - LB

    fixed_one_new = findall(j -> g_sorted[remaining + 1] - g[j] > gap, free)
    fixed_zero_new = findall(j -> g[j] - g_sorted[remaining] > gap, free)

    return free[fixed_zero_new], free[fixed_one_new]
end