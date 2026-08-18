using LinearAlgebra
using Printf
using DataStructures: BinaryMinHeap


struct AOPTConflictCut
    F1::Vector{Int}
    F0::Vector{Int}
end


struct AOPTReducedConflictCut
    positive::Vector{Int}
    negative::Vector{Int}
    rhs::Int
end


struct AOPTBnCNode
    F1::Vector{Int}
    F0::Vector{Int}
    lb::Float64
    x::Vector{Float64}
    keep::Vector{Int}
    depth::Int
end


Base.isless(a::AOPTBnCNode, b::AOPTBnCNode) = a.lb < b.lb


function _reduced_conflict_cuts(cuts::Vector{AOPTConflictCut}, keep::Vector{Int})
    position = Dict(index => local_index for (local_index, index) in enumerate(keep))

    reduced = AOPTReducedConflictCut[]

    for cut in cuts
        positive = Int[]
        negative = Int[]

        for index in cut.F1
            if haskey(position, index)
                push!(positive, position[index])
            end
        end

        for index in cut.F0
            if haskey(position, index)
                push!(negative, position[index])
            end
        end

        push!(
            reduced,
            AOPTReducedConflictCut(
                positive,
                negative,
                length(cut.F1) - 1,
            ),
        )
    end

    return reduced
end


function _same_conflict_cut(a::AOPTConflictCut, b::AOPTConflictCut)
    return a.F1 == b.F1 && a.F0 == b.F0
end


function _add_global_conflict_cut!(cuts::Vector{AOPTConflictCut}, F1::Vector{Int}, F0::Vector{Int}, max_cuts::Int)
    F1_cut = sort(unique(F1))
    F0_cut = sort(unique(F0))

    isempty(F1_cut) && isempty(F0_cut) && return false

    isempty(intersect(F1_cut, F0_cut)) ||
        error("Conflict cut fixes the same variable to zero and one.")

    candidate = AOPTConflictCut(
        F1_cut,
        F0_cut,
    )

    for cut in cuts
        _same_conflict_cut(cut, candidate) && return false
    end

    push!(cuts, candidate)

    if length(cuts) > max_cuts
        deleteat!(
            cuts,
            1:(length(cuts) - max_cuts),
        )
    end

    return true
end


function _bound_node_bnc(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, cuts::Vector{AOPTConflictCut}; eps::Float64=1e-6)
    n = size(A, 2)

    F1 = sort(unique(F1))
    F0 = sort(unique(F0))
    keep = setdiff(collect(1:n), F0)

    if !isempty(intersect(F1, F0)) || length(F1) > k || length(keep) < k
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
        return _determined_node(
            A,
            k,
            F1,
            F0,
        )
    end

    A_reduced = A[:, keep]
    fixed_one = _local_fix1_indices(keep, F1)
    x0 = _initial_relaxation_point(length(keep), k, fixed_one)

    reduced_cuts = _reduced_conflict_cuts(
        cuts,
        keep,
    )

    x = continuous_relaxation_with_conflict_cuts(
        A_reduced,
        k,
        fixed_one,
        reduced_cuts;
        x0=x0,
        tol=eps,
    )

    if x === nothing
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

    x = Vector{Float64}(x)

    length(x) == length(keep) ||
        error("continuous_relaxation_with_conflict_cuts returned a vector with the wrong length.")

    Lambda, tau, mu, nu = construct_dual(
        A_reduced,
        x,
        k,
        fixed_one,
    )

    mu = Vector{Float64}(mu)
    nu = Vector{Float64}(nu)

    length(mu) == length(keep) ||
        error("construct_dual returned mu with the wrong length.")

    length(nu) == length(keep) ||
        error("construct_dual returned nu with the wrong length.")

    lb =
        dual_objective(
            Lambda,
            tau,
            mu,
            nu,
            k,
        ) +
        sum(
            mu[fixed_one];
            init=0.0,
        )

    isfinite(lb) ||
        error("The dual lower bound is not finite.")

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


function _set_incumbent_bnc!(state::Base.RefValue, A::AbstractMatrix, S::Vector{Int}, value::Float64)
    isfinite(value) || return false
    value >= state[].UB && return false

    x_best = zeros(Float64, size(A, 2))
    x_best[S] .= 1.0

    state[] = (
        UB=value,
        x_best=x_best,
    )

    return true
end


function _update_incumbent_bnc!(state::Base.RefValue, A::AbstractMatrix, k::Int, F1::Vector{Int}, r, tol::Float64)
    improved = false

    integer_value, integer_set = _integer_subset(
        A,
        k,
        r.keep,
        r.x,
        F1,
        tol,
    )

    if isfinite(integer_value)
        improved |= _set_incumbent_bnc!(
            state,
            A,
            integer_set,
            integer_value,
        )
    end

    rounded_value, rounded_set = _rounded_subset(
        A,
        k,
        r.keep,
        r.x,
        F1,
    )

    if isfinite(rounded_value)
        improved |= _set_incumbent_bnc!(
            state,
            A,
            rounded_set,
            rounded_value,
        )
    end

    return improved
end


function _learn_conflict_cut!(cuts::Vector{AOPTConflictCut}, F1::Vector{Int}, F0::Vector{Int}, counters::Base.RefValue, max_cuts::Int)
    added = _add_global_conflict_cut!(
        cuts,
        F1,
        F0,
        max_cuts,
    )

    if added
        counters[] = (
            nodes=counters[].nodes,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
            ncuts=counters[].ncuts + 1,
        )
    end

    return added
end


function _push_node_bnc!(open, F1::Vector{Int}, F0::Vector{Int}, r, UB::Float64, depth::Int, tol::Float64)
    r.determined && return false
    r.infeasible && return false
    _is_integer_point(r.x, tol) && return false
    r.lb >= UB - tol && return false

    node = AOPTBnCNode(
        copy(F1),
        copy(F0),
        r.lb,
        copy(r.x),
        copy(r.keep),
        depth,
    )

    push!(open, node)

    return true
end


function _solve_and_fix_node_bnc(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent_lb::Float64, cuts::Vector{AOPTConflictCut}, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, resolve::Int, eps::Float64, tol::Float64, max_cuts::Int)
    resolve >= 1 ||
        error("resolve must be at least 1.")

    F1_current = sort(unique(copy(F1)))
    F0_current = sort(unique(copy(F0)))

    inherited_lb = parent_lb
    nresolve = 0

    while true
        nresolve += 1

        r_raw = _bound_node_bnc(
            A,
            k,
            F1_current,
            F0_current,
            cuts;
            eps=eps,
        )

        counters[] = (
            nodes=counters[].nodes + 1,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
            ncuts=counters[].ncuts,
        )

        r = merge(
            r_raw,
            (
                lb=max(
                    inherited_lb,
                    r_raw.lb,
                ),
            ),
        )

        inherited_lb = r.lb

        if r.infeasible
            _learn_conflict_cut!(
                cuts,
                F1_current,
                F0_current,
                counters,
                max_cuts,
            )

            return F1_current, F0_current, r
        end

        _update_incumbent_bnc!(
            state,
            A,
            k,
            F1_current,
            r,
            tol,
        )

        if r.lb >= state[].UB - tol
            _learn_conflict_cut!(
                cuts,
                F1_current,
                F0_current,
                counters,
                max_cuts,
            )

            return F1_current, F0_current, r
        end

        if r.determined
            return F1_current, F0_current, r
        end

        if _is_integer_point(r.x, tol)
            return F1_current, F0_current, r
        end

        fixing_rule == :none &&
            return F1_current, F0_current, r

        F1_new, F0_new = _apply_variable_fixing(
            F1_current,
            F0_current,
            r,
            A,
            k,
            state[].UB,
            fixing_rule,
        )

        if F1_new == F1_current &&
           F0_new == F0_current

            return F1_current, F0_current, r
        end

        added_fix1 =
            length(F1_new) -
            length(F1_current)

        added_fix0 =
            length(F0_new) -
            length(F0_current)

        added_fix1 >= 0 ||
            error("The number of variables fixed to one decreased.")

        added_fix0 >= 0 ||
            error("The number of variables fixed to zero decreased.")

        counters[] = (
            nodes=counters[].nodes,
            nfix0=counters[].nfix0 + added_fix0,
            nfix1=counters[].nfix1 + added_fix1,
            ncuts=counters[].ncuts,
        )

        if nresolve >= resolve
            r_raw = _bound_node_bnc(
                A,
                k,
                F1_new,
                F0_new,
                cuts;
                eps=eps,
            )

            counters[] = (
                nodes=counters[].nodes + 1,
                nfix0=counters[].nfix0,
                nfix1=counters[].nfix1,
                ncuts=counters[].ncuts,
            )

            r = merge(
                r_raw,
                (
                    lb=max(
                        inherited_lb,
                        r_raw.lb,
                    ),
                ),
            )

            if r.infeasible
                _learn_conflict_cut!(
                    cuts,
                    F1_new,
                    F0_new,
                    counters,
                    max_cuts,
                )

                return F1_new, F0_new, r
            end

            _update_incumbent_bnc!(
                state,
                A,
                k,
                F1_new,
                r,
                tol,
            )

            if r.lb >= state[].UB - tol
                _learn_conflict_cut!(
                    cuts,
                    F1_new,
                    F0_new,
                    counters,
                    max_cuts,
                )
            end

            return F1_new, F0_new, r
        end

        F1_current = F1_new
        F0_current = F0_new
    end
end


function _process_child_bnc!(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent::AOPTBnCNode, open, cuts::Vector{AOPTConflictCut}, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, resolve::Int, eps::Float64, tol::Float64, max_cuts::Int)
    F1_final, F0_final, r = _solve_and_fix_node_bnc(
        A,
        k,
        F1,
        F0,
        parent.lb,
        cuts,
        state,
        counters;
        fixing_rule=fixing_rule,
        resolve=resolve,
        eps=eps,
        tol=tol,
        max_cuts=max_cuts,
    )

    _push_node_bnc!(
        open,
        F1_final,
        F0_final,
        r,
        state[].UB,
        parent.depth + 1,
        tol,
    )

    return nothing
end


function _print_root_bnc(fixing_rule::Symbol, UB::Float64, LB::Float64, nfix0::Int, nfix1::Int, ncuts::Int)
    @printf(
        "root: fixing = %s  cuts = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f\n",
        String(fixing_rule),
        ncuts,
        nfix0,
        nfix1,
        UB,
        LB,
        max(UB - LB, 0.0),
    )

    flush(stdout)
end


function _print_progress_bnc(nodes::Int, open, nfix0::Int, nfix1::Int, ncuts::Int, UB::Float64)
    LB =
        isempty(open) ?
        UB :
        min(
            UB,
            first(open).lb,
        )

    @printf(
        "nodes = %d  open = %d  fix0 = %d  fix1 = %d  cuts = %d  UB = %.10f  LB = %.10f  gap = %.10f\n",
        nodes,
        length(open),
        nfix0,
        nfix1,
        ncuts,
        UB,
        LB,
        max(UB - LB, 0.0),
    )

    flush(stdout)
end


function _print_final_bnc(status::String, fixing_rule::Symbol, nodes::Int, open_count::Int, nfix0::Int, nfix1::Int, ncuts::Int, UB::Float64, LB::Float64, wall_time::Float64)
    @printf(
        "done [%s]: fixing = %s  nodes = %d  open = %d  fix0 = %d  fix1 = %d  cuts = %d  UB = %.10f  LB = %.10f  gap = %.10f  time = %.6f\n",
        status,
        String(fixing_rule),
        nodes,
        open_count,
        nfix0,
        nfix1,
        ncuts,
        UB,
        LB,
        max(UB - LB, 0.0),
        wall_time,
    )

    flush(stdout)
end


function solve_bnc(A::AbstractMatrix, k::Int; fixing_rule::Symbol=:none, resolve::Int=1, time_limit::Real=3600.0, verbose::Bool=true, eps::Float64=1e-6, tol::Float64=1e-6, report_every::Int=1000, max_cuts::Int=20)
    start_time = time()
    n = size(A, 2)

    1 <= k <= n ||
        error("k must satisfy 1 <= k <= size(A, 2).")

    time_limit > 0 ||
        error("time_limit must be positive.")

    report_every >= 1 ||
        error("report_every must be at least 1.")

    max_cuts >= 1 ||
        error("max_cuts must be at least 1.")

    fixing_rule = _normalize_fixing_rule(fixing_rule)

    greedy_x, greedy_value = greedy(A, k)

    length(greedy_x) == n ||
        error("greedy returned a vector with the wrong length.")

    isfinite(greedy_value) ||
        error("greedy returned a non-finite objective value.")

    state = Ref(
        (
            UB=greedy_value,
            x_best=greedy_x,
        ),
    )

    counters = Ref(
        (
            nodes=0,
            nfix0=0,
            nfix1=0,
            ncuts=0,
        ),
    )

    cuts = AOPTConflictCut[]

    open = BinaryMinHeap{AOPTBnCNode}()

    root_F1, root_F0, root_result = _solve_and_fix_node_bnc(
        A,
        k,
        Int[],
        Int[],
        -Inf,
        cuts,
        state,
        counters;
        fixing_rule=fixing_rule,
        resolve=resolve,
        eps=eps,
        tol=tol,
        max_cuts=max_cuts,
    )

    root_lb = root_result.lb

    verbose &&
        _print_root_bnc(
            fixing_rule,
            state[].UB,
            min(
                state[].UB,
                root_lb,
            ),
            counters[].nfix0,
            counters[].nfix1,
            counters[].ncuts,
        )

    _push_node_bnc!(
        open,
        root_F1,
        root_F0,
        root_result,
        state[].UB,
        0,
        tol,
    )

    next_report = report_every
    time_limit_hit = false

    while !isempty(open)
        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        node = pop!(open)

        node.lb >= state[].UB - tol &&
            continue

        branch_index = _branch_variable(
            node.keep,
            node.x,
            node.F1,
            tol,
        )

        branch_index == 0 &&
            continue

        F1_child = sort(
            vcat(
                node.F1,
                branch_index,
            ),
        )

        _process_child_bnc!(
            A,
            k,
            F1_child,
            node.F0,
            node,
            open,
            cuts,
            state,
            counters;
            fixing_rule=fixing_rule,
            resolve=resolve,
            eps=eps,
            tol=tol,
            max_cuts=max_cuts,
        )

        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        F0_child = sort(
            vcat(
                node.F0,
                branch_index,
            ),
        )

        _process_child_bnc!(
            A,
            k,
            node.F1,
            F0_child,
            node,
            open,
            cuts,
            state,
            counters;
            fixing_rule=fixing_rule,
            resolve=resolve,
            eps=eps,
            tol=tol,
            max_cuts=max_cuts,
        )

        if verbose &&
           counters[].nodes >= next_report

            _print_progress_bnc(
                counters[].nodes,
                open,
                counters[].nfix0,
                counters[].nfix1,
                counters[].ncuts,
                state[].UB,
            )

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

        final_lb = min(
            state[].UB,
            first(open).lb,
        )
    end

    verbose &&
        _print_final_bnc(
            status,
            fixing_rule,
            counters[].nodes,
            length(open),
            counters[].nfix0,
            counters[].nfix1,
            counters[].ncuts,
            state[].UB,
            final_lb,
            wall_time,
        )

    stats = (
        status=status,
        nodes=counters[].nodes,
        open_nodes=length(open),
        nfix0=counters[].nfix0,
        nfix1=counters[].nfix1,
        ncuts=counters[].ncuts,
        active_cuts=length(cuts),
        UB=state[].UB,
        LB=final_lb,
        gap=max(
            state[].UB - final_lb,
            0.0,
        ),
        root_LB=min(
            state[].UB,
            root_lb,
        ),
        wall_time=wall_time,
        time_limit_hit=time_limit_hit,
        tree_exhausted=tree_exhausted,
        fixing_rule=fixing_rule,
        max_cuts=max_cuts,
    )

    return state[].x_best, stats
end