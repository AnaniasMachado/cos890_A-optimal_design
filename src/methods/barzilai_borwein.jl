using LinearAlgebra

function barzilai_borwein(A::AbstractMatrix, k::Integer, x0::AbstractVector, eps::Float64, proj_eps::Float64, step_size::String)
    @assert step_size == "BB1" || step_size == "BB2"

    x = dykstra_projection(x0, k, proj_eps)

    M = information_matrix(x, A)
    g = gradient(M, A)

    alpha = 1.0

    while true
        x_opt = dykstra_projection(x - g, k, proj_eps)

        if norm(x - x_opt) <= eps
            return x
        end

        x_new = dykstra_projection(x - alpha * g, k, proj_eps)

        M_new = information_matrix(x_new, A)
        g_new = gradient(M_new, A)

        s = x_new - x
        y = g_new - g

        if step_size == "BB1"
            alpha = dot(s, s) / dot(s, y)
        else
            alpha = dot(s, y) / dot(y, y)
        end

        x = x_new
        g = g_new
    end
end