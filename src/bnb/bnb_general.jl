using Printf
using DataStructures: BinaryMinHeap

function _set_incumbent!(state::Base.RefValue, A::AbstractMatrix, S::Vector{Int}, value::Float64)
    isfinite(value) || return false
    value >= state[].UB && return false

    x_best = zeros(Float64, size(A, 2))
    x_best[S] .= 1.0
    state[] = (UB=value, x_best=x_best)

    return true
end

function _update_incumbent!(state::Base.RefValue, A::AbstractMatrix, k::Int, F1::Vector{Int}, r, tol::Float64)
    improved = false

    integer_value, integer_set = _integer_subset(A, k, r.keep, r.x, F1, tol)

    if isfinite(integer_value)
        improved |= _set_incumbent!(state, A, integer_set, integer_value)
    end

    rounded_value, rounded_set = _rounded_subset(A, k, r.keep, r.x, F1)

    if isfinite(rounded_value)
        improved |= _set_incumbent!(state, A, rounded_set, rounded_value)
    end

    return improved
end

function _push_node!(open, F1::Vector{Int}, F0::Vector{Int}, r, UB::Float64, depth::Int, tol::Float64)
    r.determined && return false
    _is_integer_point(r.x, tol) && return false
    r.lb >= UB - tol && return false

    node = AOPTNode(copy(F1), copy(F0), r.lb, copy(r.x), copy(r.keep), depth)
    push!(open, node)

    return true
end

function _solve_and_fix_node(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent_lb::Float64, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, eps::Float64, proj_eps::Float64, step_size::String, tol::Float64)
    F1_current = sort(unique(copy(F1)))
    F0_current = sort(unique(copy(F0)))
    inherited_lb = parent_lb

    while true
        r_raw = _bound_node(A, k, F1_current, F0_current; eps=eps, proj_eps=proj_eps, step_size=step_size)

        counters[] = (
            nodes=counters[].nodes + 1,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
        )

        r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))
        inherited_lb = r.lb

        _update_incumbent!(state, A, k, F1_current, r, tol)

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
        )

        F1_current = F1_new
        F0_current = F0_new
    end
end

function _process_child!(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent::AOPTNode, open, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, eps::Float64, proj_eps::Float64, step_size::String, tol::Float64)
    F1_final, F0_final, r = _solve_and_fix_node(A, k, F1, F0, parent.lb, state, counters; fixing_rule=fixing_rule, eps=eps, proj_eps=proj_eps, step_size=step_size, tol=tol)

    _push_node!(open, F1_final, F0_final, r, state[].UB, parent.depth + 1, tol)

    return nothing
end

function _print_root(fixing_rule::Symbol, UB::Float64, LB::Float64)
    @printf("root: fixing = %s  UB = %.10f  LB = %.10f  gap = %.10f\n", String(fixing_rule), UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_progress(nodes::Int, open, nfix0::Int, nfix1::Int, UB::Float64)
    LB = isempty(open) ? UB : min(UB, first(open).lb)
    @printf("nodes = %d  open = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f\n", nodes, length(open), nfix0, nfix1, UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_final(status::String, fixing_rule::Symbol, nodes::Int, open_count::Int, nfix0::Int, nfix1::Int, UB::Float64, LB::Float64, wall_time::Float64)
    @printf("done [%s]: fixing = %s  nodes = %d  open = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f  time = %.6f\n", status, String(fixing_rule), nodes, open_count, nfix0, nfix1, UB, LB, max(UB - LB, 0.0), wall_time)
    flush(stdout)
end

function solve_bnb(A::AbstractMatrix, k::Int; fixing_rule::Symbol=:none, time_limit::Real=3600.0, verbose::Bool=true, eps::Float64=1e-6, proj_eps::Float64=1e-12, step_size::String="BB1", tol::Float64=1e-6, report_every::Int=1000)
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
    counters = Ref((nodes=0, nfix0=0, nfix1=0))
    open = BinaryMinHeap{AOPTNode}()

    root_F1, root_F0, root_result = _solve_and_fix_node(A, k, Int[], Int[], -Inf, state, counters; fixing_rule=fixing_rule, eps=eps, proj_eps=proj_eps, step_size=step_size, tol=tol)

    root_lb = root_result.lb
    verbose && _print_root(fixing_rule, state[].UB, min(state[].UB, root_lb))

    _push_node!(open, root_F1, root_F0, root_result, state[].UB, 0, tol)

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
        _process_child!(A, k, F1_child, node.F0, node, open, state, counters; fixing_rule=fixing_rule, eps=eps, proj_eps=proj_eps, step_size=step_size, tol=tol)

        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        F0_child = sort(vcat(node.F0, branch_index))
        _process_child!(A, k, node.F1, F0_child, node, open, state, counters; fixing_rule=fixing_rule, eps=eps, proj_eps=proj_eps, step_size=step_size, tol=tol)

        if verbose && counters[].nodes >= next_report
            _print_progress(counters[].nodes, open, counters[].nfix0, counters[].nfix1, state[].UB)

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

    verbose && _print_final(status, fixing_rule, counters[].nodes, length(open), counters[].nfix0, counters[].nfix1, state[].UB, final_lb, wall_time)

    stats = (
        status=status,
        nodes=counters[].nodes,
        open_nodes=length(open),
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
    )

    return state[].x_best, stats
end