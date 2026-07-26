using LinearAlgebra
using Printf
using DataStructures: BinaryMinHeap

function _normalize_lr_step_rule(step_rule::Symbol)
    step_rule in (:constant, :diminishing, :polyak) || error("lr_step_rule must be :constant, :diminishing, or :polyak.")
    return step_rule
end

function _project_psd_lr(Z::AbstractMatrix)
    decomposition = eigen(Symmetric(Z))
    eigenvalues = max.(decomposition.values, 0.0)
    projected = decomposition.vectors * Diagonal(eigenvalues) * decomposition.vectors'
    return Matrix(Symmetric(projected))
end

function _lr_matrix_terms(Lambda::AbstractMatrix, inverse_eps::Float64)
    decomposition = eigen(Symmetric(Lambda))
    eigenvalues_psd = max.(decomposition.values, 0.0)
    eigenvalues_inverse = max.(eigenvalues_psd, inverse_eps)

    Lambda_sqrt = decomposition.vectors * Diagonal(sqrt.(eigenvalues_psd)) * decomposition.vectors'
    Lambda_invsqrt = decomposition.vectors * Diagonal(1.0 ./ sqrt.(eigenvalues_inverse)) * decomposition.vectors'

    return Matrix(Symmetric(Lambda_sqrt)), Matrix(Symmetric(Lambda_invsqrt))
end

function _lr_coefficients(A::AbstractMatrix, Lambda::AbstractMatrix)
    return vec(sum(A .* (Lambda * A); dims=1))
end

function _lr_binary_solution(coefficients::Vector{Float64}, k::Int, fixed_one::Vector{Int})
    n = length(coefficients)
    remaining = k - length(fixed_one)

    fixed_mask = falses(n)
    fixed_mask[fixed_one] .= true

    free = findall(!, fixed_mask)

    0 <= remaining <= length(free) || error("The LR cardinality subproblem is infeasible.")

    selected_free = Int[]

    if remaining > 0
        positions = partialsortperm(coefficients[free], 1:remaining; rev=true)
        selected_free = free[positions]
    end

    selected = sort(vcat(fixed_one, selected_free))

    x = zeros(Float64, n)
    x[selected] .= 1.0

    return x, selected
end

function _lr_value(Lambda_sqrt::AbstractMatrix, coefficients::Vector{Float64}, selected::Vector{Int})
    return 2.0 * tr(Lambda_sqrt) - sum(coefficients[selected])
end

function _lr_feasible_value(A::AbstractMatrix, selected::Vector{Int})
    value = try
        _exact_subset_value(A, selected)
    catch
        Inf
    end

    return value
end

function _lr_step_size(step_rule::Symbol, alpha0::Float64, iteration::Int, lr_value::Float64, node_UB::Float64, norm_H::Float64)
    if norm_H <= eps(Float64)
        return 0.0
    elseif step_rule == :constant
        return alpha0 / max(1.0, norm_H)
    elseif step_rule == :diminishing
        return alpha0 / (sqrt(iteration) * max(1.0, norm_H))
    elseif isfinite(node_UB)
        return alpha0 * max(node_UB - lr_value, 0.0) / (norm_H^2)
    else
        return alpha0 / (sqrt(iteration) * max(1.0, norm_H))
    end
end

function _solve_lr_node(A::AbstractMatrix, k::Int, fixed_one::Vector{Int}, deadline::Float64; iter_lr::Int, lr_step_rule::Symbol, alpha0::Float64, inverse_eps::Float64, Lambda0=nothing)
    iter_lr >= 1 || error("iter_lr must be at least 1.")
    alpha0 > 0.0 || error("alpha0 must be positive.")
    inverse_eps > 0.0 || error("inverse_eps must be positive.")

    lr_step_rule = _normalize_lr_step_rule(lr_step_rule)

    p, n = size(A)

    Lambda = if Lambda0 === nothing
        Matrix{Float64}(I, p, p)
    else
        size(Lambda0) == (p, p) || error("Lambda0 has the wrong dimensions.")
        _project_psd_lr(Matrix{Float64}(Lambda0))
    end

    best_lb = -Inf
    best_Lambda = copy(Lambda)

    x_last = zeros(Float64, n)
    coefficients_last = zeros(Float64, n)
    selected_last = Int[]

    iterations = 0
    time_limit_hit = false

    for iteration in 1:iter_lr
        if time() >= deadline
            time_limit_hit = true
            break
        end

        Lambda_sqrt, Lambda_invsqrt = _lr_matrix_terms(Lambda, inverse_eps)

        coefficients = _lr_coefficients(A, Lambda)
        x, selected = _lr_binary_solution(coefficients, k, fixed_one)

        value = _lr_value(Lambda_sqrt, coefficients, selected)

        isfinite(value) || error("The LR lower bound is not finite.")

        if value > best_lb
            best_lb = value
            best_Lambda .= Lambda
        end

        x_last .= x
        coefficients_last .= coefficients
        selected_last = copy(selected)
        iterations = iteration

        Mx = information_matrix(x, A)
        supergradient = Lambda_invsqrt - Mx
        norm_H = norm(supergradient)

        norm_H <= eps(Float64) && break

        node_UB = lr_step_rule == :polyak ? _lr_feasible_value(A, selected) : Inf
        alpha = _lr_step_size(lr_step_rule, alpha0, iteration, value, node_UB, norm_H)

        alpha > 0.0 || continue

        Lambda = _project_psd_lr(Lambda + alpha * supergradient)
    end

    if iterations == 0
        Lambda_sqrt, _ = _lr_matrix_terms(Lambda, inverse_eps)

        coefficients_last .= _lr_coefficients(A, Lambda)
        x_last, selected_last = _lr_binary_solution(coefficients_last, k, fixed_one)

        best_lb = _lr_value(Lambda_sqrt, coefficients_last, selected_last)
        best_Lambda .= Lambda
    end

    isfinite(best_lb) || error("The LR lower bound is not finite.")

    return (
        lb=best_lb,
        x=x_last,
        selected=selected_last,
        coefficients=coefficients_last,
        Lambda_lr=best_Lambda,
        iterations=iterations,
        time_limit_hit=time_limit_hit,
    )
end

function _bound_node_lr(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, deadline::Float64; iter_lr::Int, lr_step_rule::Symbol, alpha0::Float64, inverse_eps::Float64, Lambda0=nothing)
    n = size(A, 2)

    F1 = sort(unique(F1))
    F0 = sort(unique(F0))

    isempty(intersect(F1, F0)) || error("F1 and F0 must be disjoint.")

    keep = setdiff(collect(1:n), F0)

    if length(F1) > k || length(keep) < k
        return (
            determined=false,
            infeasible=true,
            lb=Inf,
            x=Float64[],
            coefficients=Float64[],
            keep=keep,
            Lambda=nothing,
            Lambda_lr=nothing,
            tau=NaN,
            mu=Float64[],
            nu=Float64[],
            lr_iterations=0,
            time_limit_hit=false,
        )
    end

    if length(F1) == k || length(keep) == k
        determined = _determined_node(A, k, F1, F0)

        return (
            determined=true,
            infeasible=determined.infeasible,
            lb=determined.lb,
            x=determined.x,
            coefficients=zeros(Float64, length(determined.keep)),
            keep=determined.keep,
            Lambda=nothing,
            Lambda_lr=nothing,
            tau=NaN,
            mu=Float64[],
            nu=Float64[],
            lr_iterations=0,
            time_limit_hit=false,
        )
    end

    A_reduced = A[:, keep]
    fixed_one = _local_fix1_indices(keep, F1)

    node_Lambda0 = nothing

    if Lambda0 !== nothing && size(Lambda0) == (size(A, 1), size(A, 1))
        node_Lambda0 = Lambda0
    end

    lr = _solve_lr_node(A_reduced, k, fixed_one, deadline; iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, Lambda0=node_Lambda0)

    length(lr.x) == length(keep) || error("The LR solution has the wrong length.")
    length(lr.coefficients) == length(keep) || error("The LR coefficient vector has the wrong length.")

    Lambda, tau, mu, nu = construct_dual(A_reduced, lr.x, k, fixed_one)

    mu = Vector{Float64}(mu)
    nu = Vector{Float64}(nu)

    length(mu) == length(keep) || error("construct_dual returned mu with the wrong length.")
    length(nu) == length(keep) || error("construct_dual returned nu with the wrong length.")

    fixing_lb = dual_objective(Lambda, tau, mu, nu, k) + sum(mu[fixed_one])

    isfinite(fixing_lb) || error("The fixing-dual lower bound is not finite.")

    lb = max(lr.lb, fixing_lb)

    return (
        determined=false,
        infeasible=false,
        lb=lb,
        x=lr.x,
        coefficients=lr.coefficients,
        keep=keep,
        Lambda=Lambda,
        Lambda_lr=lr.Lambda_lr,
        tau=tau,
        mu=mu,
        nu=nu,
        lr_iterations=lr.iterations,
        time_limit_hit=lr.time_limit_hit,
    )
end

function _branch_variable_lr(keep::Vector{Int}, coefficients::Vector{Float64}, F1::Vector{Int}, k::Int)
    length(coefficients) == length(keep) || error("The LR coefficient vector has the wrong length.")

    fixed_set = Set(F1)

    free_local = [j for j in eachindex(keep) if !(keep[j] in fixed_set)]
    remaining = k - length(F1)

    1 <= remaining < length(free_local) || return 0

    order = sortperm(coefficients[free_local]; rev=true)
    branch_local = free_local[order[remaining]]

    return keep[branch_local]
end

function _push_node_lr!(open, F1::Vector{Int}, F0::Vector{Int}, r, UB::Float64, depth::Int, tol::Float64)
    r.determined && return false
    r.infeasible && return false
    r.lb >= UB - tol && return false

    branch_index = _branch_variable_lr(r.keep, r.coefficients, F1, length(F1) + count(index -> !(index in Set(F1)), r.keep))

    node = AOPTNode(copy(F1), copy(F0), r.lb, copy(r.coefficients), copy(r.keep), depth)
    push!(open, node)

    return true
end

function _update_incumbent_lr!(state::Base.RefValue, A::AbstractMatrix, k::Int, F1::Vector{Int}, r, tol::Float64)
    return _update_incumbent!(state, A, k, F1, r, tol)
end

function _solve_and_fix_node_lr(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent_lb::Float64, parent_Lambda, state::Base.RefValue, counters::Base.RefValue, deadline::Float64; fixing_rule::Symbol, resolve::Int, iter_lr::Int, lr_step_rule::Symbol, alpha0::Float64, inverse_eps::Float64, tol::Float64)
    resolve >= 1 || error("resolve must be at least 1.")

    F1_current = sort(unique(copy(F1)))
    F0_current = sort(unique(copy(F0)))

    inherited_lb = parent_lb
    Lambda_start = parent_Lambda
    nresolve = 0

    while true
        nresolve += 1

        r_raw = _bound_node_lr(A, k, F1_current, F0_current, deadline; iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, Lambda0=Lambda_start)

        counters[] = (
            nodes=counters[].nodes + 1,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
            lr_iterations=counters[].lr_iterations + r_raw.lr_iterations,
        )

        r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))
        inherited_lb = r.lb
        Lambda_start = r.Lambda_lr

        r.infeasible && return F1_current, F0_current, r

        _update_incumbent_lr!(state, A, k, F1_current, r, tol)

        r.determined && return F1_current, F0_current, r
        r.lb >= state[].UB - tol && return F1_current, F0_current, r
        fixing_rule == :none && return F1_current, F0_current, r
        time() >= deadline && return F1_current, F0_current, r

        F1_new, F0_new = _apply_variable_fixing(F1_current, F0_current, r, A, k, state[].UB, fixing_rule)

        if F1_new == F1_current && F0_new == F0_current
            return F1_current, F0_current, r
        end

        added_fix1 = length(F1_new) - length(F1_current)
        added_fix0 = length(F0_new) - length(F0_current)

        added_fix1 >= 0 || error("The number of variables fixed to one decreased.")
        added_fix0 >= 0 || error("The number of variables fixed to zero decreased.")

        counters[] = (
            nodes=counters[].nodes,
            nfix0=counters[].nfix0 + added_fix0,
            nfix1=counters[].nfix1 + added_fix1,
            lr_iterations=counters[].lr_iterations,
        )

        F1_current = F1_new
        F0_current = F0_new

        if nresolve >= resolve
            r_raw = _bound_node_lr(A, k, F1_current, F0_current, deadline; iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, Lambda0=Lambda_start)

            counters[] = (
                nodes=counters[].nodes + 1,
                nfix0=counters[].nfix0,
                nfix1=counters[].nfix1,
                lr_iterations=counters[].lr_iterations + r_raw.lr_iterations,
            )

            r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))

            if !r.infeasible
                _update_incumbent_lr!(state, A, k, F1_current, r, tol)
            end

            return F1_current, F0_current, r
        end
    end
end

function _process_child_lr!(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent::AOPTNode, parent_Lambda, open, multipliers, state::Base.RefValue, counters::Base.RefValue, deadline::Float64; fixing_rule::Symbol, resolve::Int, iter_lr::Int, lr_step_rule::Symbol, alpha0::Float64, inverse_eps::Float64, tol::Float64)
    F1_final, F0_final, r = _solve_and_fix_node_lr(A, k, F1, F0, parent.lb, parent_Lambda, state, counters, deadline; fixing_rule=fixing_rule, resolve=resolve, iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, tol=tol)

    if !r.determined && !r.infeasible && r.lb < state[].UB - tol
        node = AOPTNode(copy(F1_final), copy(F0_final), r.lb, copy(r.coefficients), copy(r.keep), parent.depth + 1)
        push!(open, node)
        multipliers[node] = r.Lambda_lr
    end

    return nothing
end

function _print_root_lr(fixing_rule::Symbol, lr_step_rule::Symbol, iter_lr::Int, UB::Float64, LB::Float64)
    @printf("root: fixing = %s  LR = %s  iter_lr = %d  UB = %.10f  LB = %.10f  gap = %.10f\n", String(fixing_rule), String(lr_step_rule), iter_lr, UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_progress_lr(nodes::Int, lr_iterations::Int, open, nfix0::Int, nfix1::Int, UB::Float64)
    LB = isempty(open) ? UB : min(UB, first(open).lb)
    @printf("nodes = %d  open = %d  LR iterations = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f\n", nodes, length(open), lr_iterations, nfix0, nfix1, UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_final_lr(status::String, fixing_rule::Symbol, lr_step_rule::Symbol, iter_lr::Int, nodes::Int, open_count::Int, lr_iterations::Int, nfix0::Int, nfix1::Int, UB::Float64, LB::Float64, wall_time::Float64)
    @printf("done [%s]: fixing = %s  LR = %s  iter_lr = %d  nodes = %d  open = %d  LR iterations = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f  time = %.6f\n", status, String(fixing_rule), String(lr_step_rule), iter_lr, nodes, open_count, lr_iterations, nfix0, nfix1, UB, LB, max(UB - LB, 0.0), wall_time)
    flush(stdout)
end

function solve_bnb_lr(A::AbstractMatrix, k::Int; iter_lr::Int=500, lr_step_rule::Symbol=:polyak, alpha0::Float64=1.0, fixing_rule::Symbol=:none, resolve::Int=1, time_limit::Real=3600.0, verbose::Bool=true, inverse_eps::Float64=1e-10, proj_eps::Union{Nothing,Float64}=nothing, tol::Float64=1e-6, report_every::Int=1000)
    start_time = time()
    deadline = start_time + Float64(time_limit)
    n = size(A, 2)

    if proj_eps !== nothing
        inverse_eps = proj_eps
    end

    1 <= k <= n || error("k must satisfy 1 <= k <= size(A, 2).")
    iter_lr >= 1 || error("iter_lr must be at least 1.")
    resolve >= 1 || error("resolve must be at least 1.")
    time_limit > 0.0 || error("time_limit must be positive.")
    alpha0 > 0.0 || error("alpha0 must be positive.")
    inverse_eps > 0.0 || error("inverse_eps must be positive.")
    report_every >= 1 || error("report_every must be at least 1.")

    fixing_rule = _normalize_fixing_rule(fixing_rule)
    lr_step_rule = _normalize_lr_step_rule(lr_step_rule)

    greedy_x, greedy_value = greedy(A, k)

    length(greedy_x) == n || error("greedy returned a vector with the wrong length.")
    isfinite(greedy_value) || error("greedy returned a non-finite objective value.")

    state = Ref((UB=greedy_value, x_best=greedy_x))
    counters = Ref((nodes=0, nfix0=0, nfix1=0, lr_iterations=0))

    open = BinaryMinHeap{AOPTNode}()
    multipliers = IdDict{AOPTNode,Any}()

    root_F1, root_F0, root_result = _solve_and_fix_node_lr(A, k, Int[], Int[], -Inf, nothing, state, counters, deadline; fixing_rule=fixing_rule, resolve=resolve, iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, tol=tol)

    root_lb = root_result.lb

    verbose && _print_root_lr(fixing_rule, lr_step_rule, iter_lr, state[].UB, min(state[].UB, root_lb))

    if !root_result.determined && !root_result.infeasible && root_result.lb < state[].UB - tol
        root_node = AOPTNode(copy(root_F1), copy(root_F0), root_result.lb, copy(root_result.coefficients), copy(root_result.keep), 0)
        push!(open, root_node)
        multipliers[root_node] = root_result.Lambda_lr
    end

    next_report = report_every
    time_limit_hit = root_result.time_limit_hit || time() >= deadline

    while !isempty(open) && !time_limit_hit
        if time() >= deadline
            time_limit_hit = true
            break
        end

        node = pop!(open)
        parent_Lambda = pop!(multipliers, node, nothing)

        node.lb >= state[].UB - tol && continue

        branch_index = _branch_variable_lr(node.keep, node.x, node.F1, k)
        branch_index == 0 && continue

        F1_child = sort(unique(vcat(node.F1, branch_index)))

        _process_child_lr!(A, k, F1_child, node.F0, node, parent_Lambda, open, multipliers, state, counters, deadline; fixing_rule=fixing_rule, resolve=resolve, iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, tol=tol)

        if time() >= deadline
            time_limit_hit = true
            break
        end

        F0_child = sort(unique(vcat(node.F0, branch_index)))

        _process_child_lr!(A, k, node.F1, F0_child, node, parent_Lambda, open, multipliers, state, counters, deadline; fixing_rule=fixing_rule, resolve=resolve, iter_lr=iter_lr, lr_step_rule=lr_step_rule, alpha0=alpha0, inverse_eps=inverse_eps, tol=tol)

        if verbose && counters[].nodes >= next_report
            _print_progress_lr(counters[].nodes, counters[].lr_iterations, open, counters[].nfix0, counters[].nfix1, state[].UB)

            while next_report <= counters[].nodes
                next_report += report_every
            end
        end
    end

    wall_time = time() - start_time
    tree_exhausted = isempty(open) && !time_limit_hit

    if tree_exhausted
        status = "OPTIMAL (exhausted)"
        final_lb = state[].UB
    else
        status = "TIME LIMIT"
        final_lb = isempty(open) ? min(state[].UB, root_lb) : min(state[].UB, first(open).lb)
    end

    verbose && _print_final_lr(status, fixing_rule, lr_step_rule, iter_lr, counters[].nodes, length(open), counters[].lr_iterations, counters[].nfix0, counters[].nfix1, state[].UB, final_lb, wall_time)

    stats = (
        status=status,
        nodes=counters[].nodes,
        open_nodes=length(open),
        lr_iterations=counters[].lr_iterations,
        nfix0=counters[].nfix0,
        nfix1=counters[].nfix1,
        UB=state[].UB,
        LB=final_lb,
        gap=max(state[].UB - final_lb, 0.0),
        root_LB=min(state[].UB, root_lb),
        wall_time=wall_time,
        time_limit_hit=time_limit_hit,
        tree_exhausted=tree_exhausted,
        fixing_rule=fixing_rule,
        iter_lr=iter_lr,
        lr_step_rule=lr_step_rule,
        alpha0=alpha0,
        inverse_eps=inverse_eps,
    )

    return state[].x_best, stats
end