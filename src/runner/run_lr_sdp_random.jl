using Random
using LinearAlgebra
using Printf

include("../methods/projection.jl")
include("../methods/barzilai_borwein.jl")
include("../methods/solver.jl")
include("../misc/util.jl")
include("../misc/heuristic.jl")
include("../misc/dual.jl")
include("../misc/var_fixing.jl")
include("../bnb/bnb_util.jl")
include("../bnb/bnb_general.jl")
include("../bnb/bnb_lr_sdp.jl")

Random.seed!(1)

m = 10
n = 100
k = 30

A = randn(m, n)

tol = 1e-6
time_limit = 3600.0
sdp_timelimit = 600.0

runtime = @elapsed begin
    x_best, stats = solve_bnb_lr(
        A,
        k;
        fixing_rule=:dual,
        time_limit=time_limit,
        sdp_timelimit=sdp_timelimit,
        tol=tol,
        verbose=true,
        report_every=100,
    )
end

objective_value = objective(information_matrix(x_best, A))

println("AOPT LR-SDP branch-and-bound results")
println("-----------------------------------")
@printf("fixing rule      : %s\n", String(stats.fixing_rule))
@printf("BnB time limit   : %.2f seconds\n", time_limit)
@printf("SDP time limit   : %.2f seconds\n", sdp_timelimit)
@printf("status           : %s\n", stats.status)
@printf("runtime          : %.6f seconds\n", runtime)
@printf("objective / UB   : %.10f\n", objective_value)
@printf("lower bound      : %.10f\n", stats.LB)
@printf("gap              : %.10f\n", stats.gap)
@printf("root lower bound : %.10f\n", stats.root_LB)
@printf("nodes processed  : %d\n", stats.nodes)
@printf("SDP solves       : %d\n", stats.sdp_solves)
@printf("fixed to zero    : %d\n", stats.nfix0)
@printf("fixed to one     : %d\n", stats.nfix1)
@printf("time limit hit   : %s\n", string(stats.time_limit_hit))
@printf("tree exhausted   : %s\n", string(stats.tree_exhausted))