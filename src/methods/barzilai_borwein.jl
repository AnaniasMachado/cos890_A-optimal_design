using LinearAlgebra

function fixed_projection(x::AbstractVector, k::Integer, free::AbstractVector{<:Integer}, fixed_one::AbstractVector{<:Integer}, proj_eps::Float64)
    n = length(x)

    x_projected = copy(x)
    x_projected[fixed_one] .= 1.0
    x_projected[free] = dykstra_projection(x[free], k - length(fixed_one), proj_eps)

    return x_projected
end

function barzilai_borwein(A::AbstractMatrix, k::Integer, x0::AbstractVector, fixed_one::AbstractVector{<:Integer}, eps::Float64, proj_eps::Float64, step_size::String)
    @assert step_size == "BB1" || step_size == "BB2"

    free = setdiff(1:length(x0), fixed_one)
    x = fixed_projection(x0, k, free, fixed_one, proj_eps)

    M = information_matrix(x, A)
    g = gradient(M, A)

    alpha = 1.0

    while true
        x_opt = fixed_projection(x - g, k, free, fixed_one, proj_eps)

        if norm(x[free] - x_opt[free]) <= eps
            return x
        end

        x_new = fixed_projection(x - alpha * g, k, free, fixed_one, proj_eps)

        M_new = information_matrix(x_new, A)
        g_new = gradient(M_new, A)

        s = x_new[free] - x[free]
        y = g_new[free] - g[free]

        if step_size == "BB1"
            alpha = dot(s, s) / dot(s, y)
        else
            alpha = dot(s, y) / dot(y, y)
        end

        x = x_new
        g = g_new
    end
end