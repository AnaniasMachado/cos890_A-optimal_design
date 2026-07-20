using LinearAlgebra

function information_matrix(x::AbstractVector, A::AbstractMatrix)
    return A * Diagonal(x) * A'
end

function objective(M::AbstractMatrix)
    m = size(M, 1)
    F = cholesky(Symmetric(M))
    return tr(F \ I(m))
end

function gradient(M::AbstractMatrix, A::AbstractMatrix)
    Y = cholesky(Symmetric(M)) \ A
    return -vec(sum(abs2, Y; dims = 1))
end