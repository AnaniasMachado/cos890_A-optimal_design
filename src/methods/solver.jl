using JuMP
import Ipopt

function continuous_relaxation(A::AbstractMatrix, k::Integer, fixed_one::AbstractVector{<:Integer}; x0=nothing, tol=1e-8)
    n = size(A, 2)

    function trace_inverse_function(xs...)
        x = collect(xs)
        return objective(information_matrix(x, A))
    end

    function trace_inverse_gradient(grad, xs...)
        x = collect(xs)
        grad .= gradient(information_matrix(x, A), A)
        return
    end

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    set_optimizer_attribute(model, "tol", tol)
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")

    @variable(model, 0 <= x[1:n] <= 1)
    @constraint(model, sum(x) == k)

    for i in fixed_one
        fix(x[i], 1.0; force=true)
    end

    if x0 === nothing
        free = setdiff(1:n, fixed_one)
        x_start = zeros(n)
        x_start[fixed_one] .= 1.0
        x_start[free] .= (k - length(fixed_one)) / length(free)
        set_start_value.(x, x_start)
    else
        set_start_value.(x, x0)
    end

    @operator(model, trace_inverse, n, trace_inverse_function, trace_inverse_gradient)
    @objective(model, Min, trace_inverse(x...))

    optimize!(model)

    return value.(x)
end