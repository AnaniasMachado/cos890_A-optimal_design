using LinearAlgebra
using Printf
using DataStructures: BinaryMinHeap

# ============================================================
# LR matrix operations
# ============================================================

function _project_psd_lr(Z::AbstractMatrix)
    decomposition = eigen(Symmetric(Z))
    eigenvalues = max.(decomposition.values, 0.0)
    projected = decomposition.vectors * Diagonal(eigenvalues) * decomposition.vectors'
    return Matrix(Symmetric(projected))
end

function _lr_matrix_sqrt(Lambda::AbstractMatrix)
    decomposition = eigen(Symmetric(Lambda))
    eigenvalues = max.(decomposition.values, 0.0)
    Lambda_sqrt = decomposition.vectors * Diagonal(sqrt.(eigenvalues)) * decomposition.vectors'
    return Matrix(Symmetric(Lambda_sqrt))
end

function _lr_coefficients(A::AbstractMatrix, Lambda::AbstractMatrix)
    return vec(sum(A .* (Lambda * A); dims=1))
end

# ============================================================
# Binary LR solution
# ============================================================

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

# ============================================================
# LR dual variables for variable fixing
# ============================================================

function _lr_dual_variables(coefficients::Vector{Float64}, k::Int, fixed_one::Vector{Int})
    n = length(coefficients)
    remaining = k - length(fixed_one)

    fixed_mask = falses(n)
    fixed_mask[fixed_one] .= true
    free = findall(!, fixed_mask)

    1 <= remaining < length(free) || error("The nontrivial LR dual construction requires 1 <= remaining < number of free variables.")

    ordered = sort(coefficients[free]; rev=true)
    tau = 0.5 * (ordered[remaining] + ordered[remaining + 1])

    mu = zeros(Float64, n)
    nu = zeros(Float64, n)

    for i in free
        mu[i] = max(tau - coefficients[i], 0.0)
        nu[i] = max(coefficients[i] - tau, 0.0)
    end

    return tau, mu, nu, free
end

function _lr_dual_bound(Lambda::AbstractMatrix, coefficients::Vector{Float64}, k::Int, fixed_one::Vector{Int}, tau::Float64, nu::Vector{Float64}, free::Vector{Int})
    remaining = k - length(fixed_one)
    fixed_term = isempty(fixed_one) ? 0.0 : sum(coefficients[i] for i in fixed_one)
    return 2.0 * tr(_lr_matrix_sqrt(Lambda)) - fixed_term - remaining * tau - sum(nu[i] for i in free)
end

# ============================================================
# Solve LR relaxation
# ============================================================

function _solve_lr_node_sdp(A::AbstractMatrix, k::Int, fixed_one::Vector{Int}, tol::Float64, sdp_timelimit::Float64)
    m, n = size(A)

    fixed_one = sort(unique(fixed_one))

    fixed_mask = falses(n)
    fixed_mask[fixed_one] .= true
    free = findall(!, fixed_mask)

    remaining = k - length(fixed_one)

    1 <= remaining < length(free) || error("The nontrivial LR SDP must satisfy 1 <= remaining < number of free variables.")

    sdp = solve_lr_sdp(A, k, fixed_one; tol=tol, time_limit=sdp_timelimit)

    Lambda_value = if sdp.Lambda === nothing
        Matrix{Float64}(I, m, m)
    else
        _project_psd_lr(sdp.Lambda)
    end

    coefficients = _lr_coefficients(A, Lambda_value)
    x, selected = _lr_binary_solution(coefficients, k, fixed_one)
    tau_value, mu, nu, free = _lr_dual_variables(coefficients, k, fixed_one)
    lb = _lr_dual_bound(Lambda_value, coefficients, k, fixed_one, tau_value, nu, free)

    isfinite(lb) || error("The LR lower bound is not finite.")

    return (
        lb=lb,
        x=x,
        selected=selected,
        coefficients=coefficients,
        Lambda_lr=Lambda_value,
        tau=tau_value,
        mu=mu,
        nu=nu,
        sdp_time_limit_hit=sdp.time_limit_hit,
        sdp_status=sdp.status,
    )
end

# ============================================================
# Node bound
# ============================================================

function _bound_node_lr(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, tol::Float64, sdp_timelimit::Float64)
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
            fixing_lb=Inf,
            x=Float64[],
            coefficients=Float64[],
            keep=keep,
            Lambda_lr=nothing,
            tau=NaN,
            mu=Float64[],
            nu=Float64[],
            sdp_time_limit_hit=false,
            sdp_status=MOI.OPTIMAL,
        )
    end

    if length(F1) == k || length(keep) == k
        determined = _determined_node(A, k, F1, F0)

        return (
            determined=true,
            infeasible=determined.infeasible,
            lb=determined.lb,
            fixing_lb=determined.lb,
            x=determined.x,
            coefficients=zeros(Float64, length(determined.keep)),
            keep=determined.keep,
            Lambda_lr=nothing,
            tau=NaN,
            mu=Float64[],
            nu=Float64[],
            sdp_time_limit_hit=false,
            sdp_status=MOI.OPTIMAL,
        )
    end

    A_reduced = A[:, keep]
    fixed_one = _local_fix1_indices(keep, F1)

    lr = _solve_lr_node_sdp(A_reduced, k, fixed_one, tol, sdp_timelimit)

    length(lr.x) == length(keep) || error("The LR solution has the wrong length.")
    length(lr.coefficients) == length(keep) || error("The LR coefficient vector has the wrong length.")
    length(lr.mu) == length(keep) || error("The LR mu vector has the wrong length.")
    length(lr.nu) == length(keep) || error("The LR nu vector has the wrong length.")

    return (
        determined=false,
        infeasible=false,
        lb=lr.lb,
        fixing_lb=lr.lb,
        x=lr.x,
        coefficients=lr.coefficients,
        keep=keep,
        Lambda_lr=lr.Lambda_lr,
        tau=lr.tau,
        mu=lr.mu,
        nu=lr.nu,
        sdp_time_limit_hit=lr.sdp_time_limit_hit,
        sdp_status=lr.sdp_status,
    )
end

# ============================================================
# LR dual variable fixing
# ============================================================

function _apply_lr_dual_fixing(F1::Vector{Int}, F0::Vector{Int}, r, UB::Float64, tol::Float64)
    gap = UB - r.fixing_lb

    isfinite(gap) || return F1, F0

    F1_new = copy(F1)
    F0_new = copy(F0)

    fixed_one_set = Set(F1)
    fixed_zero_set = Set(F0)

    for local_index in eachindex(r.keep)
        global_index = r.keep[local_index]

        global_index in fixed_one_set && continue
        global_index in fixed_zero_set && continue

        if r.mu[local_index] > gap + tol
            push!(F0_new, global_index)
        elseif r.nu[local_index] > gap + tol
            push!(F1_new, global_index)
        end
    end

    return sort(unique(F1_new)), sort(unique(F0_new))
end

# ============================================================
# Branching
# ============================================================

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

    node = AOPTNode(copy(F1), copy(F0), r.lb, copy(r.coefficients), copy(r.keep), depth)
    push!(open, node)

    return true
end

# ============================================================
# Incumbent
# ============================================================

function _update_incumbent_lr!(state::Base.RefValue, A::AbstractMatrix, k::Int, F1::Vector{Int}, r, tol::Float64)
    return _update_incumbent!(state, A, k, F1, r, tol)
end

# ============================================================
# Solve and fix node
# ============================================================

function _solve_and_fix_node_lr(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent_lb::Float64, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, resolve::Int, tol::Float64, sdp_timelimit::Float64)
    resolve >= 1 || error("resolve must be at least 1.")

    F1_current = sort(unique(copy(F1)))
    F0_current = sort(unique(copy(F0)))

    inherited_lb = parent_lb
    nresolve = 0

    while true
        nresolve += 1

        r_raw = _bound_node_lr(A, k, F1_current, F0_current, tol, sdp_timelimit)

        counters[] = (
            nodes=counters[].nodes + 1,
            nfix0=counters[].nfix0,
            nfix1=counters[].nfix1,
            sdp_solves=counters[].sdp_solves + (!r_raw.determined && !r_raw.infeasible ? 1 : 0),
        )

        r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))
        inherited_lb = r.lb

        r.infeasible && return F1_current, F0_current, r

        _update_incumbent_lr!(state, A, k, F1_current, r, tol)

        r.determined && return F1_current, F0_current, r
        r.lb >= state[].UB - tol && return F1_current, F0_current, r
        fixing_rule == :none && return F1_current, F0_current, r

        F1_new, F0_new = _apply_lr_dual_fixing(F1_current, F0_current, r, state[].UB, tol)

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
            sdp_solves=counters[].sdp_solves,
        )

        F1_current = F1_new
        F0_current = F0_new

        if nresolve >= resolve
            r_raw = _bound_node_lr(A, k, F1_current, F0_current, tol, sdp_timelimit)

            counters[] = (
                nodes=counters[].nodes + 1,
                nfix0=counters[].nfix0,
                nfix1=counters[].nfix1,
                sdp_solves=counters[].sdp_solves + (!r_raw.determined && !r_raw.infeasible ? 1 : 0),
            )

            r = merge(r_raw, (lb=max(inherited_lb, r_raw.lb),))

            if !r.infeasible
                _update_incumbent_lr!(state, A, k, F1_current, r, tol)
            end

            return F1_current, F0_current, r
        end
    end
end

# ============================================================
# Process child
# ============================================================

function _process_child_lr!(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}, parent::AOPTNode, open, state::Base.RefValue, counters::Base.RefValue; fixing_rule::Symbol, resolve::Int, tol::Float64, sdp_timelimit::Float64)
    F1_final, F0_final, r = _solve_and_fix_node_lr(A, k, F1, F0, parent.lb, state, counters; fixing_rule=fixing_rule, resolve=resolve, tol=tol, sdp_timelimit=sdp_timelimit)

    if !r.determined && !r.infeasible && r.lb < state[].UB - tol
        node = AOPTNode(copy(F1_final), copy(F0_final), r.lb, copy(r.coefficients), copy(r.keep), parent.depth + 1)
        push!(open, node)
    end

    return nothing
end

# ============================================================
# Printing
# ============================================================

function _print_root_lr(fixing_rule::Symbol, UB::Float64, LB::Float64, sdp_status)
    @printf("root: fixing = %s  LR = SDP/Hypatia  UB = %.10f  LB = %.10f  gap = %.10f  SDP status = %s\n", String(fixing_rule), UB, LB, max(UB - LB, 0.0), string(sdp_status))
    flush(stdout)
end

function _print_progress_lr(nodes::Int, sdp_solves::Int, open, nfix0::Int, nfix1::Int, UB::Float64)
    LB = isempty(open) ? UB : min(UB, first(open).lb)
    @printf("nodes = %d  open = %d  SDP solves = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f\n", nodes, length(open), sdp_solves, nfix0, nfix1, UB, LB, max(UB - LB, 0.0))
    flush(stdout)
end

function _print_final_lr(status::String, fixing_rule::Symbol, nodes::Int, open_count::Int, sdp_solves::Int, nfix0::Int, nfix1::Int, UB::Float64, LB::Float64, wall_time::Float64)
    @printf("done [%s]: fixing = %s  LR = SDP/Hypatia  nodes = %d  open = %d  SDP solves = %d  fix0 = %d  fix1 = %d  UB = %.10f  LB = %.10f  gap = %.10f  time = %.6f\n", status, String(fixing_rule), nodes, open_count, sdp_solves, nfix0, nfix1, UB, LB, max(UB - LB, 0.0), wall_time)
    flush(stdout)
end

# ============================================================
# Main LR branch-and-bound
# ============================================================

function solve_bnb_lr(A::AbstractMatrix, k::Int; fixing_rule::Symbol=:none, resolve::Int=1, time_limit::Real=3600.0, sdp_timelimit::Real=60.0, verbose::Bool=true, tol::Float64=1e-6, report_every::Int=1000)
    start_time = time()
    n = size(A, 2)

    1 <= k <= n || error("k must satisfy 1 <= k <= size(A, 2).")
    resolve >= 1 || error("resolve must be at least 1.")
    time_limit > 0.0 || error("time_limit must be positive.")
    sdp_timelimit > 0.0 || error("sdp_timelimit must be positive.")
    tol > 0.0 || error("tol must be positive.")
    report_every >= 1 || error("report_every must be at least 1.")
    fixing_rule in (:none, :dual) || error("fixing_rule must be :none or :dual.")

    greedy_x, greedy_value = greedy(A, k)

    length(greedy_x) == n || error("greedy returned a vector with the wrong length.")
    isfinite(greedy_value) || error("greedy returned a non-finite objective value.")

    state = Ref((UB=greedy_value, x_best=greedy_x))
    counters = Ref((nodes=0, nfix0=0, nfix1=0, sdp_solves=0))

    open = BinaryMinHeap{AOPTNode}()

    root_F1, root_F0, root_result = _solve_and_fix_node_lr(A, k, Int[], Int[], -Inf, state, counters; fixing_rule=fixing_rule, resolve=resolve, tol=tol, sdp_timelimit=Float64(sdp_timelimit))

    root_lb = root_result.lb

    verbose && _print_root_lr(fixing_rule, state[].UB, min(state[].UB, root_lb), root_result.sdp_status)

    if !root_result.determined && !root_result.infeasible && root_result.lb < state[].UB - tol
        root_node = AOPTNode(copy(root_F1), copy(root_F0), root_result.lb, copy(root_result.coefficients), copy(root_result.keep), 0)
        push!(open, root_node)
    end

    next_report = report_every
    time_limit_hit = time() - start_time >= time_limit

    while !isempty(open) && !time_limit_hit
        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        node = pop!(open)

        node.lb >= state[].UB - tol && continue

        branch_index = _branch_variable_lr(node.keep, node.x, node.F1, k)
        branch_index == 0 && continue

        F1_child = sort(unique(vcat(node.F1, branch_index)))

        _process_child_lr!(A, k, F1_child, node.F0, node, open, state, counters; fixing_rule=fixing_rule, resolve=resolve, tol=tol, sdp_timelimit=Float64(sdp_timelimit))

        if time() - start_time >= time_limit
            time_limit_hit = true
            break
        end

        F0_child = sort(unique(vcat(node.F0, branch_index)))

        _process_child_lr!(A, k, node.F1, F0_child, node, open, state, counters; fixing_rule=fixing_rule, resolve=resolve, tol=tol, sdp_timelimit=Float64(sdp_timelimit))

        if verbose && counters[].nodes >= next_report
            _print_progress_lr(counters[].nodes, counters[].sdp_solves, open, counters[].nfix0, counters[].nfix1, state[].UB)

            while next_report <= counters[].nodes
                next_report += report_every
            end
        end

        time_limit_hit = time() - start_time >= time_limit
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

    verbose && _print_final_lr(status, fixing_rule, counters[].nodes, length(open), counters[].sdp_solves, counters[].nfix0, counters[].nfix1, state[].UB, final_lb, wall_time)

    stats = (
        status=status,
        nodes=counters[].nodes,
        open_nodes=length(open),
        sdp_solves=counters[].sdp_solves,
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
        lr_method=:sdp_hypatia,
        sdp_timelimit=Float64(sdp_timelimit),
        tol=tol,
    )

    return state[].x_best, stats
end