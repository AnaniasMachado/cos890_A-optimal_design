using LinearAlgebra

function construct_dual(A::AbstractMatrix, x::AbstractVector, k::Integer)
    m, n = size(A)

    M = information_matrix(x, A)
    F = cholesky(Symmetric(M))
    M_inv = F \ I(m)
    Lambda = Symmetric(M_inv * M_inv)

    d = vec(sum(A .* (Lambda * A), dims=1))
    ordering = sortperm(d, rev=true)
    tau = (d[ordering[k]] + d[ordering[k + 1]]) / 2

    mu = max.(tau .- d, 0.0)
    nu = max.(d .- tau, 0.0)

    return Lambda, tau, mu, nu
end

function dual_objective(Lambda::AbstractMatrix, tau::Real, mu::AbstractVector, nu::AbstractVector, k::Integer)
    return 2 * tr(sqrt(Symmetric(Lambda))) - k * tau - sum(nu)
end