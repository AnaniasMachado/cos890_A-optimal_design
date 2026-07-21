using LinearAlgebra

struct AOPTNode
    F1::Vector{Int}
    F0::Vector{Int}
    lb::Float64
    x::Vector{Float64}
    keep::Vector{Int}
    depth::Int
end

Base.isless(a::AOPTNode, b::AOPTNode) = a.lb < b.lb

function _normalize_fixing_rule(fixing_rule::Symbol)
    fixing_rule in (:none, :dual, :primal) || error("fixing_rule must be :none, :dual, or :primal.")
    return fixing_rule
end

function _is_integer_point(x::AbstractVector, tol::Real=1e-6)
    isempty(x) && return false
    return all(xi -> xi <= tol || xi >= 1.0 - tol, x)
end

function _local_fix1_indices(keep::Vector{Int}, F1::Vector{Int})
    position = Dict(index => local_index for (local_index, index) in enumerate(keep))
    return sort([position[index] for index in F1])
end

function _initial_relaxation_point(n::Int, k::Int, fixed_one::Vector{Int})
    x = zeros(Float64, n)
    x[fixed_one] .= 1.0

    remaining = k - length(fixed_one)
    free = setdiff(1:n, fixed_one)

    if remaining > 0
        isempty(free) && error("No free variables remain, but the cardinality constraint is not satisfied.")
        x[free] .= remaining / length(free)
    end

    return x
end

function _exact_subset_value(A::AbstractMatrix, S::Vector{Int})
    isempty(S) && return Inf
    return Float64(objective(information_matrix(ones(Float64, length(S)), A[:, S])))
end

function _rounded_subset(A::AbstractMatrix, k::Int, keep::Vector{Int}, x::Vector{Float64}, F1::Vector{Int})
    length(F1) > k && return Inf, Int[]
    length(keep) < k && return Inf, Int[]
    length(x) == length(keep) || return Inf, Int[]

    fixed_set = Set(F1)
    free_local = [j for j in eachindex(keep) if !(keep[j] in fixed_set)]
    remaining = k - length(F1)

    remaining > length(free_local) && return Inf, Int[]

    chosen_local = remaining == 0 ? Int[] : partialsortperm(x[free_local], 1:remaining; rev=true)
    chosen_global = remaining == 0 ? Int[] : keep[free_local[chosen_local]]
    S = sort(vcat(F1, chosen_global))

    length(S) == k || return Inf, Int[]
    length(unique(S)) == k || return Inf, Int[]

    value = try
        _exact_subset_value(A, S)
    catch
        Inf
    end

    return value, S
end

function _integer_subset(A::AbstractMatrix, k::Int, keep::Vector{Int}, x::Vector{Float64}, F1::Vector{Int}, tol::Real)
    _is_integer_point(x, tol) || return Inf, Int[]

    selected_local = findall(xi -> xi >= 1.0 - tol, x)
    S = sort(keep[selected_local])

    length(S) == k || return Inf, Int[]
    all(index -> index in S, F1) || return Inf, Int[]

    value = try
        _exact_subset_value(A, S)
    catch
        Inf
    end

    return value, S
end

function _branch_variable(keep::Vector{Int}, x::Vector{Float64}, F1::Vector{Int}, tol::Real)
    fixed_set = Set(F1)
    best_local = 0
    best_fractionality = tol

    for j in eachindex(keep)
        keep[j] in fixed_set && continue

        fractionality = min(x[j], 1.0 - x[j])

        if fractionality > best_fractionality
            best_fractionality = fractionality
            best_local = j
        end
    end

    return best_local == 0 ? 0 : keep[best_local]
end

function _determined_node(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int})
    keep = setdiff(collect(1:size(A, 2)), F0)
    S = length(F1) == k ? sort(copy(F1)) : sort(copy(keep))

    selected = Set(S)
    x = [keep[j] in selected ? 1.0 : 0.0 for j in eachindex(keep)]
    value = _exact_subset_value(A, S)

    return (
        determined=true,
        lb=value,
        x=x,
        keep=keep,
        Lambda=nothing,
        tau=NaN,
        mu=Float64[],
        nu=Float64[],
    )
end

function _bound_node(A::AbstractMatrix, k::Int, F1::Vector{Int}, F0::Vector{Int}; eps::Float64=1e-6, proj_eps::Float64=1e-12, step_size::String="BB1")
    n = size(A, 2)
    F1 = sort(unique(F1))
    F0 = sort(unique(F0))
    keep = setdiff(collect(1:n), F0)

    if length(F1) == k || length(keep) == k
        return _determined_node(A, k, F1, F0)
    end

    A_reduced = A[:, keep]
    fixed_one = _local_fix1_indices(keep, F1)
    x0 = _initial_relaxation_point(length(keep), k, fixed_one)

    x = Vector{Float64}(barzilai_borwein(A_reduced, k, x0, fixed_one, eps, proj_eps, step_size))

    length(x) == length(keep) || error("barzilai_borwein returned a vector with the wrong length.")

    Lambda, tau, mu, nu = construct_dual(A_reduced, x, k)

    mu = Vector{Float64}(mu)
    nu = Vector{Float64}(nu)

    length(mu) == length(keep) || error("construct_dual returned mu with the wrong length.")
    length(nu) == length(keep) || error("construct_dual returned nu with the wrong length.")

    lb = Float64(dual_objective(Lambda, tau, mu, nu, k) + sum(mu[fixed_one]))

    isfinite(lb) || error("The dual lower bound is not finite.")

    return (
        determined=false,
        lb=lb,
        x=x,
        keep=keep,
        Lambda=Lambda,
        tau=Float64(tau),
        mu=mu,
        nu=nu,
    )
end

function _dual_fixing(F1::Vector{Int}, F0::Vector{Int}, r, k::Int, UB::Float64)
    fixed_one = _local_fix1_indices(r.keep, F1)
    free = setdiff(collect(eachindex(r.keep)), fixed_one)

    adjusted_UB = UB - sum(r.mu[fixed_one])
    candidate_fix0, candidate_fix1 = dual_variable_fixing(adjusted_UB, r.Lambda, r.tau, r.mu, r.nu, k)

    local_fix0 = sort(unique(intersect(Int.(candidate_fix0), free)))
    local_fix1 = sort(unique(intersect(Int.(candidate_fix1), free)))

    valid_local = Set(eachindex(r.keep))
    all(index -> index in valid_local, local_fix0) || error("dual_variable_fixing returned an invalid local index.")
    all(index -> index in valid_local, local_fix1) || error("dual_variable_fixing returned an invalid local index.")
    isempty(intersect(local_fix0, local_fix1)) || error("dual_variable_fixing fixed the same variable to zero and one.")

    global_fix0 = r.keep[local_fix0]
    global_fix1 = r.keep[local_fix1]

    F1_new = sort(unique(vcat(F1, global_fix1)))
    F0_new = sort(unique(vcat(F0, global_fix0)))

    return F1_new, F0_new
end

function _primal_fixing(F1::Vector{Int}, F0::Vector{Int}, r, A::AbstractMatrix, k::Int, UB::Float64)
    fixed_one = _local_fix1_indices(r.keep, F1)

    if !isempty(fixed_one)
        return sort(unique(F1)), sort(unique(F0))
    end

    free = collect(eachindex(r.keep))
    candidate_fix0, candidate_fix1 = primal_variable_fixing(UB, r.x, A[:, r.keep], k)

    local_fix0 = sort(unique(intersect(Int.(candidate_fix0), free)))
    local_fix1 = sort(unique(intersect(Int.(candidate_fix1), free)))

    valid_local = Set(eachindex(r.keep))
    all(index -> index in valid_local, local_fix0) || error("primal_variable_fixing returned an invalid local index.")
    all(index -> index in valid_local, local_fix1) || error("primal_variable_fixing returned an invalid local index.")
    isempty(intersect(local_fix0, local_fix1)) || error("primal_variable_fixing fixed the same variable to zero and one.")

    global_fix0 = r.keep[local_fix0]
    global_fix1 = r.keep[local_fix1]

    F1_new = sort(unique(vcat(F1, global_fix1)))
    F0_new = sort(unique(vcat(F0, global_fix0)))

    return F1_new, F0_new
end

function _apply_variable_fixing(F1::Vector{Int}, F0::Vector{Int}, r, A::AbstractMatrix, k::Int, UB::Float64, fixing_rule::Symbol)
    fixing_rule = _normalize_fixing_rule(fixing_rule)

    if fixing_rule == :none
        return sort(unique(F1)), sort(unique(F0))
    elseif fixing_rule == :dual
        F1_new, F0_new = _dual_fixing(F1, F0, r, k, UB)
    else
        F1_new, F0_new = _primal_fixing(F1, F0, r, A, k, UB)
    end

    isempty(intersect(F1_new, F0_new)) || error("Variable fixing produced overlapping F0 and F1.")
    length(F1_new) <= k || error("Variable fixing fixed more than k variables to one.")
    size(A,2) - length(F0_new) >= k || error("Variable fixing removed too many variables.")

    return F1_new, F0_new
end