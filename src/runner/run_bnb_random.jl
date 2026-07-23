using Random
using LinearAlgebra
using Printf

include("../methods/projection.jl")
include("../methods/barzilai_borwein.jl")

include("../misc/util.jl")
include("../misc/heuristic.jl")
include("../misc/dual.jl")
include("../misc/var_fixing.jl")

include("../bnb/bnb_util.jl")
include("../bnb/bnb_general.jl")

Random.seed!(1)

m = 10
n = 100
k = 30

A = randn(m, n)

runtime = @elapsed begin
    x_best, stats = solve_bnb(A, k; fixing_rule=:dual, verbose=false)
end

objective_value = objective(information_matrix(x_best, A))

println("AOPT branch-and-bound results")
println("-----------------------------")
@printf("fixing rule       : %s\n", String(stats.fixing_rule))
@printf("status            : %s\n", stats.status)
@printf("runtime           : %.6f seconds\n", runtime)
@printf("solver wall time  : %.6f seconds\n", stats.wall_time)
@printf("objective / UB    : %.10f\n", objective_value)
@printf("reported UB       : %.10f\n", stats.UB)
@printf("lower bound       : %.10f\n", stats.LB)
@printf("gap               : %.10f\n", stats.gap)
@printf("root lower bound  : %.10f\n", stats.root_LB)
@printf("nodes processed   : %d\n", stats.nodes)
@printf("open nodes        : %d\n", stats.open_nodes)
@printf("fixed to zero     : %d\n", stats.nfix0)
@printf("fixed to one      : %d\n", stats.nfix1)
@printf("time limit hit    : %s\n", string(stats.time_limit_hit))
@printf("tree exhausted    : %s\n", string(stats.tree_exhausted))