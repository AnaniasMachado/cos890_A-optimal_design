using Random
using LinearAlgebra
using Printf

include("../methods/projection.jl")
include("../methods/barzilai_borwein.jl")

include("../misc/util.jl")
include("../misc/heuristic.jl")

Random.seed!(1)

m = 10
n = 100
k = 30

eps = 1e-6
proj_eps = 1e-12
step_size = "BB1"

A = randn(m, n)
x0 = fill(k / n, n)

bb_runtime = @elapsed begin
    x_relaxed = barzilai_borwein(
        A,
        k,
        x0,
        eps,
        proj_eps,
        step_size
    )
end

greedy_runtime = @elapsed begin
    x_greedy, upper_bound = greedy(A, k)
end

M_relaxed = information_matrix(x_relaxed, A)
lower_bound = objective(M_relaxed)

gap = upper_bound - lower_bound

@printf("BB runtime:       %.6f seconds\n", bb_runtime)
@printf("Greedy runtime:   %.6f seconds\n", greedy_runtime)
@printf("Lower bound:      %.10f\n", lower_bound)
@printf("Upper bound:      %.10f\n", upper_bound)
@printf("Gap:              %.10f\n", gap)