using LinearAlgebra
using Printf
using DataStructures: BinaryMinHeap

struct AOPTCut
    alpha::Float64
    beta::Vector{Float64}
end

struct AOPTOANode
    F1::Vector{Int}
    F0::Vector{Int}
    lb::Float64
    x::Vector{Float64}
    keep::Vector{Int}
    cuts::Vector{AOPTCut}
    depth::Int
end

Base.isless(a::AOPTOANode, b::AOPTOANode) = a.lb < b.lb

function _oa_cut(A::AbstractMatrix, x::AbstractVector)
    M = information_matrix(x, A)
    f = objective(M)
    g = gradient(M, A)

    return AOPTCut(f - dot(g, x), Vector{Float64}(g))
end

function _reduced_oa_cuts(cuts::Vector{AOPTCut}, keep::Vector{Int})
    return [AOPTCut(cut.alpha, cut.beta[keep]) for cut in cuts]
end

function _node_oa_cut(A::AbstractMatrix, keep::Vector{Int}, x::Vector{Float64})
    x_global = zeros(Float64, size(A, 2))
    x_global[keep] .= x

    return _oa_cut(A, x_global)
end

function _bound_node_oa(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, cuts::Vector{AOPTCut}; eps::Float64=1e-6)
    n = size(A, 2)

    F1 = sort(unique(F1))
    F0 = sort(unique(F0))
    keep = setdiff(collect(1:n), F0)

    if length(F1) > k || length(keep) < k
        return (
            determined=false,
            infeasible=true,
            lb=Inf,
            x=Float64[],
            keep=keep,
            Lambda=nothing,
            tau=NaN,
            mu=Float64[],
            nu=Float64[],
        )
    end

    if length(F1) == k || length(keep) == k
        return _determined_node(A, k, F1, F0)
    end

    A_reduced = A[:, keep]
    fixed_one = _local_fix1_indices(keep, F1)
    x0 = _initial_relaxation_point(length(keep), k, fixed_one)
    reduced_cuts = _reduced_oa_cuts(cuts, keep)

    result = outer_approximation_relaxation(A_reduced, k, fixed_one, reduced_cuts; x0=x0, tol=eps)

    x = Vector{Float64}(result.x)
    lb = Float64(result.lb)

    length(x) == length(keep) || error("outer_approximation_relaxation returned a vector with the wrong length.")
    isfinite(lb) || error("The OA lower bound is not finite.")

    Lambda, tau, mu, nu = construct_dual(A_reduced, x, k, fixed_one)

    mu = Vector{Float64}(mu)
    nu = Vector{Float64}(nu)

    length(mu) == length(keep) || error("construct_dual returned mu with the wrong length.")
    length(nu) == length(keep) || error("construct_dual returned nu with the wrong length.")

    return (
        determined=false,
        infeasible=false,
        lb=lb,
        x=x,
        keep=keep,
        Lambda=Lambda,
        tau=tau,
        mu=mu,
        nu=nu,
    )
end

function _set_incumbent_oa!(state::Base.RefValue, A::AbstractMatrix, S::Vector{Int}, value::Float64)
    isfinite(value) || return false
    value >= state[].UB && return false

    x_best = zeros(Float64, size(A, 2))
    x_best[S] .= 1.0

    state[] = (UB=value, x_best=x_best)

    return true
end

function _update_incumbent_oa!(state::Base.RefValue, A::AbstractMatrix, k::Int, F1::Vector{Int}, r, tol::Float64)
    improved = false

    integer_value, integer_set = _integer_subset(A, k, r.keep, r.x, F1, tol)

    if isfinite(integer_value)
        improved |= _set_incumbent_oa!(state, A, integer_set, integer_value)
    end

    rounded_value, rounded_set = _rounded_subset(A, k, r.keep, r.x, F1)

    if isfinite(rounded_value)
        improved |= _set_incumbent_oa!(state, A, rounded_set, rounded_value)
    end

    return improved
end

function _cuts_for_children(A::AbstractMatrix, cuts::Vector{AOPTCut}, r, counters::Base.RefValue)
    child_cuts = copy(cuts)

    if !r.determined && !r.infeasible
        push!(child_cuts, _node_oa_cut(A, r.keep, r.x))

        counters[] = (
            nodes=counters[].nodes,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
            ncuts=counters[].ncuts + 1,
        )
    end

    return child_cuts
end

function _push_node_oa!(open, F1::Vector{Int}, F0::Vector{Int}, r, cuts::Vector{AOPTCut}, UB::Float64, depth::Int, tol::Float64)
    r.determined && return false
    r.infeasible && return false
    _is_integer_point(r.x, tol) && return false
    r.lb >= UB - tol && return false

    node = AOPTOANode(copy(F1), copy(F0), r.lb, copy(r.x), copy(r.keep), copy(cuts), depth)
    push!(open, node)

    return true
end

function _solve_and_fix_node_oa(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent_lb::Float64, cuts::Vector{AOPTCut}, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, resolve::Int, eps::Float64, tol::Float64)
    resolve >= 1 || error("resolve must be at least 1.")

    F1_current = sort(unique(copy(F1)))
    F0_current = sort(unique(copy(F0)))
    inherited_lb = parent_lb

    nresolve = 0

    while true
        nresolve += 1

        r_raw = _bound_node_oa(A, k, F1_current, F0_current, cuts; eps=eps)

        counters[] = (
            nodes=counters[].nodes + 1,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
            ncuts=counters[].ncuts,
        )

        r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))
        inherited_lb = r.lb

        r.infeasible && return F1_current, F0_current, r

        _update_incumbent_oa!(state, A, k, F1_current, r, tol)

        r.determined && return F1_current, F0_current, r
        _is_integer_point(r.x, tol) && return F1_current, F0_current, r
        r.lb >= state[].UB - tol && return F1_current, F0_current, r
        fixing_rule == :none && return F1_current, F0_current, r

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
            ncuts=counters[].ncuts,
        )

        F1_current = F1_new
        F0_current = F0_new

        if nresolve >= resolve
            r_raw = _bound_node_oa(A, k, F1_current, F0_current, cuts; eps=eps)

            counters[] = (
                nodes=counters[].nodes + 1,
                nfix0=counters[].nfix0,
                nfix1=counters[].nfix1,
                ncuts=counters[].ncuts,
            )

            r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))

            r.infeasible && return F1_current, F0_current, r

            _update_incumbent_oa!(state, A, k, F1_current, r, tol)

            return F1_current, F0_current, r
        end
    end
end

function _process_child_oa!(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent::AOPTOANode, open, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, resolve::Int, eps::Float64, tol::Float64)
    F1_final, F0_final, r = _solve_and_fix_node_oa(A, k, F1, F0, parent.lb, parent.cuts, state, counters; fixing_rule=fixing_rule, resolve=resolve, eps=eps, tol=tol)

    child_cuts = _cuts_for_children(A, parent.cuts, r, counters)

    _push_node_oa!(open, F1_final, F0_final, r, child_cuts, state[].UB, parent.depth + 1, tol)

    return nothing
end

function _print_root_oa(fixing_rule::Symbol, UB::Float64, LB::Float64, nfix0::Int, nfix1::Int, ncuts::Int)
    @printf("root: fixing = %s  cuts = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f\n", String(fixing_rule), ncuts, nfix0, nfix1, UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_progress_oa(nodes::Int, open, nfix0::Int, nfix1::Int, ncuts::Int, UB::Float64)
    LB = isempty(open) ? UB : min(UB, first(open).lb)
    @printf("nodes = %d  open = %d  fix0 = %d  fix1 = %d  cuts = %d  UB = %.10f  LB = %.10f  gap = %.10f\n", nodes, length(open), nfix0, nfix1, ncuts, UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_final_oa(status::String, fixing_rule::Symbol, nodes::Int, open_count::Int, nfix0::Int, nfix1::Int, ncuts::Int, UB::Float64, LB::Float64, wall_time::Float64)
    @printf("done [%s]: fixing = %s  nodes = %d  open = %d  fix0 = %d  fix1 = %d  cuts = %d  UB = %.10f  LB = %.10f  gap = %.10f  time = %.6f\n", status, String(fixing_rule), nodes, open_count, nfix0, nfix1, ncuts, UB, LB, max(UB - LB, 0.0), wall_time)
    flush(stdout)
end

function solve_bnb_oa(A::AbstractMatrix, k::Int; fixing_rule::Symbol=:none, resolve::Int=1, time_limit::Real=3600.0, verbose::Bool=true, eps::Float64=1e-6, tol::Float64=1e-6, report_every::Int=1000)
    start_time = time()
    n = size(A, 2)

    1 <= k <= n || error("k must satisfy 1 <= k <= size(A, 2).")
    time_limit > 0 || error("time_limit must be positive.")
    report_every >= 1 || error("report_every must be at least 1.")

    fixing_rule = _normalize_fixing_rule(fixing_rule)

    greedy_x, greedy_value = greedy(A, k)

    length(greedy_x) == n || error("greedy returned a vector with the wrong length.")
    isfinite(greedy_value) || error("greedy returned a non-finite objective value.")

    state = Ref((UB=greedy_value, x_best=greedy_x))
    counters = Ref((nodes=0, nfix0=0, nfix1=0, ncuts=0))
    open = BinaryMinHeap{AOPTOANode}()

    root_cuts = AOPTCut[_oa_cut(A, greedy_x)]

    counters[] = (
        nodes=0,
        nfix0=0,
        nfix1=0,
        ncuts=1,
    )

    root_F1, root_F0, root_result = _solve_and_fix_node_oa(A, k, Int[], Int[], -Inf, root_cuts, state, counters; fixing_rule=fixing_rule, resolve=resolve, eps=eps, tol=tol)

    root_lb = root_result.lb
    root_child_cuts = _cuts_for_children(A, root_cuts, root_result, counters)

    verbose && _print_root_oa(fixing_rule, state[].UB, min(state[].UB, root_lb), counters[].nfix0, counters[].nfix1, counters[].ncuts)

    _push_node_oa!(open, root_F1, root_F0, root_result, root_child_cuts, state[].UB, 0, tol)

    next_report = report_every
    time_limit_hit = false

    while !isempty(open)
        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        node = pop!(open)

        node.lb >= state[].UB - tol && continue

        branch_index = _branch_variable(node.keep, node.x, node.F1, tol)
        branch_index == 0 && continue

        F1_child = sort(vcat(node.F1, branch_index))

        _process_child_oa!(A, k, F1_child, node.F0, node, open, state, counters; fixing_rule=fixing_rule, resolve=resolve, eps=eps, tol=tol)

        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        F0_child = sort(vcat(node.F0, branch_index))

        _process_child_oa!(A, k, node.F1, F0_child, node, open, state, counters; fixing_rule=fixing_rule, resolve=resolve, eps=eps, tol=tol)

        if verbose && counters[].nodes >= next_report
            _print_progress_oa(counters[].nodes, open, counters[].nfix0, counters[].nfix1, counters[].ncuts, state[].UB)

            while next_report <= counters[].nodes
                next_report += report_every
            end
        end
    end

    tree_exhausted = isempty(open)
    wall_time = time() - start_time

    if tree_exhausted
        status = "OPTIMAL (exhausted)"
        final_lb = state[].UB
    else
        status = "TIME LIMIT"
        final_lb = min(state[].UB, first(open).lb)
    end

    verbose && _print_final_oa(status, fixing_rule, counters[].nodes, length(open), counters[].nfix0, counters[].nfix1, counters[].ncuts, state[].UB, final_lb, wall_time)

    stats = (
        status=status,
        nodes=counters[].nodes,
        open_nodes=length(open),
        nfix0=counters[].nfix0,
        nfix1=counters[].nfix1,
        ncuts=counters[].ncuts,
        UB=state[].UB,
        LB=final_lb,
        gap=max(state[].UB - final_lb, 0.0),
        root_LB=min(state[].UB, root_lb),
        wall_time=wall_time,
        time_limit_hit=time_limit_hit,
        tree_exhausted=tree_exhausted,
        fixing_rule=fixing_rule,
    )

    return state[].x_best, stats
end