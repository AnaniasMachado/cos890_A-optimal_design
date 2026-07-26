using Random
using LinearAlgebra
using Printf
using DelimitedFiles

include("../methods/projection.jl")
include("../methods/barzilai_borwein.jl")

include("../misc/util.jl")
include("../misc/heuristic.jl")
include("../misc/dual.jl")
include("../misc/var_fixing.jl")

include("../bnb/bnb_util.jl")
include("../bnb/bnb_general.jl")
include("../bnb/bnb_lr.jl")

Random.seed!(1)

instance = "airfoil_normalized"

A = Matrix{Float64}(readdlm(
    "data/aopt_matrices/A_$(instance).csv",
    ',',
))

m, n = size(A)

k = ceil(Int, 0.20 * n)   # 20% of the observations

runtime = @elapsed begin
    x_best, stats = solve_bnb_lr(A, k;
        iter_lr=50, lr_step_rule=:polyak, alpha0=0.1,
        fixing_rule=:dual, verbose=true)
end

objective_value = objective(information_matrix(x_best, A))

println("AOPT LR branch-and-bound results")
println("--------------------------------")
@printf("fixing rule       : %s\n", String(stats.fixing_rule))
@printf("LR step rule      : %s\n", String(stats.lr_step_rule))
@printf("LR iterations/node: %d\n", stats.iter_lr)
@printf("LR alpha0         : %.6f\n", stats.alpha0)
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
@printf("total LR iterations: %d\n", stats.lr_iterations)
@printf("fixed to zero     : %d\n", stats.nfix0)
@printf("fixed to one      : %d\n", stats.nfix1)
@printf("time limit hit    : %s\n", string(stats.time_limit_hit))
@printf("tree exhausted    : %s\n", string(stats.tree_exhausted))