using JuMP
using Ipopt
using Hypatia

const MOI = JuMP.MOI

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

function continuous_relaxation_with_conflict_cuts(A::AbstractMatrix, k::Integer, fixed_one::AbstractVector{<:Integer}, cuts; x0=nothing, tol=1e-8)
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

    for cut in cuts
        all(i -> 1 <= i <= n, cut.positive) || error("Conflict cut contains an invalid positive index.")
        all(i -> 1 <= i <= n, cut.negative) || error("Conflict cut contains an invalid negative index.")

        @constraint(
            model,
            sum(x[i] for i in cut.positive) -
            sum(x[i] for i in cut.negative)
            <= cut.rhs
        )
    end

    if x0 === nothing
        free = setdiff(1:n, fixed_one)
        x_start = zeros(Float64, n)
        x_start[fixed_one] .= 1.0

        if !isempty(free)
            x_start[free] .= (k - length(fixed_one)) / length(free)
        end

        set_start_value.(x, x_start)
    else
        length(x0) == n || error("x0 has the wrong dimension.")
        set_start_value.(x, x0)
    end

    @operator(model, trace_inverse, n, trace_inverse_function, trace_inverse_gradient)
    @objective(model, Min, trace_inverse(x...))

    optimize!(model)

    status = termination_status(model)

    if status == MOI.INFEASIBLE || status == MOI.LOCALLY_INFEASIBLE
        return nothing
    end

    if status == MOI.NUMERICAL_ERROR
        return continuous_relaxation(
            A,
            k,
            fixed_one;
            x0=x0,
            tol=tol,
        )
    end

    status in (
        MOI.OPTIMAL,
        MOI.LOCALLY_SOLVED,
        MOI.ALMOST_OPTIMAL,
        MOI.ALMOST_LOCALLY_SOLVED,
    ) || error("Relaxation was not solved successfully. Status: $status")

    return Vector{Float64}(value.(x))
end

function solve_lr_sdp(A::AbstractMatrix, k::Int, fixed_one::Vector{Int}; tol::Float64=1e-6, time_limit::Real=60.0)
    m, n = size(A)

    tol > 0.0 || error("tol must be positive.")
    time_limit > 0.0 || error("time_limit must be positive.")

    fixed_one = sort(unique(fixed_one))

    fixed_mask = falses(n)
    fixed_mask[fixed_one] .= true
    free = findall(!, fixed_mask)

    remaining = k - length(fixed_one)

    1 <= remaining < length(free) || error("The nontrivial LR SDP must satisfy 1 <= remaining < number of free variables.")

    model = Model(() -> Hypatia.Optimizer(
        verbose=false,
        tol_rel_opt=tol,
        tol_abs_opt=tol,
    ))

    set_time_limit_sec(model, Float64(time_limit))

    @variable(model, Lambda[1:m, 1:m], Symmetric)
    @variable(model, W[1:m, 1:m], Symmetric)
    @variable(model, tau)
    @variable(model, s[free] >= 0.0)

    @expression(model, q[i=1:n], sum(A[r, i] * A[c, i] * Lambda[r, c] for r in 1:m, c in 1:m))

    for i in free
        @constraint(model, s[i] >= q[i] - tau)
    end

    identity_matrix = Matrix{Float64}(I, m, m)

    @constraint(model, [Lambda W; W identity_matrix] in PSDCone())

    fixed_term = isempty(fixed_one) ? 0.0 : sum(q[i] for i in fixed_one)
    free_slack_term = sum(s[i] for i in free)

    @objective(model, Max, 2.0 * sum(W[j, j] for j in 1:m) - fixed_term - remaining * tau - free_slack_term)

    optimize!(model)

    status = termination_status(model)
    pstatus = primal_status(model)

    has_primal_point = pstatus == MOI.FEASIBLE_POINT || pstatus == MOI.NEARLY_FEASIBLE_POINT

    Lambda_value = has_primal_point ? Matrix{Float64}(value.(Lambda)) : nothing

    return (
        Lambda=Lambda_value,
        status=status,
        primal_status=pstatus,
        time_limit_hit=status == MOI.TIME_LIMIT,
    )
end