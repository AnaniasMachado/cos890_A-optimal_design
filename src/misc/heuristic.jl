using Random
using LinearAlgebra

function greedy(A::AbstractMatrix, k::Integer)
    m, n = size(A)

    @assert m <= k <= n

    initial = randperm(n)[1:m]
    M = A[:, initial] * A[:, initial]'

    while !isposdef(Symmetric(M))
        initial = randperm(n)[1:m]
        M = A[:, initial] * A[:, initial]'
    end

    selected = falses(n)
    selected[initial] .= true

    while count(selected) < k
        best_index = 0
        best_value = Inf

        for j in 1:n
            if selected[j]
                continue
            end

            aj = A[:, j]
            M_trial = M + aj * aj'
            value = objective(M_trial)

            if value < best_value
                best_value = value
                best_index = j
            end
        end

        aj = A[:, best_index]
        M = M + aj * aj'
        selected[best_index] = true
    end

    x = Float64.(selected)

    return x, objective(M)
end