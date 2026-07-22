using LinearAlgebra

function construct_dual(A::AbstractMatrix, x::AbstractVector, k::Integer, fixed_one::AbstractVector{<:Integer}=Int[])
    m, n = size(A)

    fixed_one = sort(unique(Int.(fixed_one)))

    all(index -> 1 <= index <= n, fixed_one) || error("fixed_one contains an invalid index.")
    length(fixed_one) <= k || error("More than k variables are fixed to one.")

    remaining = k - length(fixed_one)
    free = setdiff(collect(1:n), fixed_one)

    remaining >= 0 || error("The remaining cardinality is negative.")
    remaining <= length(free) || error("Not enough free variables remain.")

    M = information_matrix(x, A)
    F = cholesky(Symmetric(M))
    M_inv = F \ I(m)
    Lambda = Symmetric(M_inv * M_inv)

    d = vec(sum(A .* (Lambda * A), dims=1))

    if remaining == 0
        tau = maximum(d[free]; init=minimum(d[fixed_one]))
    elseif remaining == length(free)
        tau = minimum(d[free])
    else
        ordering = sortperm(d[free], rev=true)

        upper_index = free[ordering[remaining]]
        lower_index = free[ordering[remaining + 1]]

        tau = (d[upper_index] + d[lower_index]) / 2
    end

    mu = max.(tau .- d, 0.0)
    nu = max.(d .- tau, 0.0)

    return Lambda, tau, mu, nu
end

function dual_objective(Lambda::AbstractMatrix, tau::Real, mu::AbstractVector, nu::AbstractVector, k::Integer)
    return 2 * tr(sqrt(Symmetric(Lambda))) - k * tau - sum(nu)
end