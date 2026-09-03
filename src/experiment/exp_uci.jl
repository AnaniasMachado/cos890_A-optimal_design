using LinearAlgebra
using Random
using Printf
using Dates
using CSV
using DataFrames
using DelimitedFiles

include("../methods/projection.jl")
include("../methods/barzilai_borwein.jl")
include("../methods/solver.jl")
include("../misc/util.jl")
include("../misc/heuristic.jl")
include("../misc/dual.jl")
include("../misc/var_fixing.jl")
include("../bnb/bnb_util.jl")

const METHOD = :bnc
# Options: :bnb, :lr, :bnc

if METHOD == :bnb
    include("../bnb/bnb_general.jl")
elseif METHOD == :lr
    include("../bnb/bnb_general.jl")
    include("../bnb/bnb_lr_sdp.jl")
elseif METHOD == :bnc
    include("../bnb/bnc.jl")
else
    error("METHOD must be :bnb, :lr, or :bnc.")
end

const SEED = 1

const INSTANCE_NAMES = [
    "airfoil",
    "auto_mpg",
    "concrete",
    "forest_fires",
    "wine_red",
    "wine_white",
    "yacht",
]

const PREPROCESSINGS = [
    "raw",
    "normalized",
    "standardized",
]

const K_OVER_N_VALUES = [0.1]
const FIXING_RULES = [:none, :dual, :primal]
# const FIXING_RULES = [:none, :dual]

const TIME_LIMIT = 3600.0
const SDP_TIMELIMIT = 600.0
const RESOLVE = 1
const EPS = 1e-6
const PROJ_EPS = 1e-9
const MAX_CUTS = 20

const DATA_FOLDER = "data/aopt_matrices"
const OUTPUT_FILE = "results/uci_benchmark.csv"

function experiment_key(method::Symbol, instance_name::String, preprocessing::String, k_over_n::Float64, fixing_rule::Symbol)
    if method == :bnc
        return (
            method,
            instance_name,
            preprocessing,
            k_over_n,
            fixing_rule,
            MAX_CUTS,
        )
    end

    return (
        method,
        instance_name,
        preprocessing,
        k_over_n,
        fixing_rule,
        0,
    )
end

function completed_experiments(filename::String)
    completed = Set()

    if !isfile(filename) || filesize(filename) == 0
        return completed
    end

    df = CSV.read(filename, DataFrame)

    for row in eachrow(df)
        method = Symbol(row.method)
        fixing_rule = Symbol(row.fixing_rule)
        max_cuts = method == :bnc ? Int(row.max_cuts) : 0

        push!(
            completed,
            (
                method,
                String(row.instance_name),
                String(row.preprocessing),
                Float64(row.k_over_n),
                fixing_rule,
                max_cuts,
            ),
        )
    end

    return completed
end

function append_result(filename::String, row)
    df = DataFrame([row])

    if isfile(filename) && filesize(filename) > 0
        CSV.write(filename, df; append=true, writeheader=false)
    else
        CSV.write(filename, df)
    end

    return nothing
end

function read_instance(instance_name::String, preprocessing::String)
    filename = "A_$(instance_name)_$(preprocessing).csv"
    path = joinpath(DATA_FOLDER, filename)

    isfile(path) || error("Instance file not found: $path")

    return Matrix{Float64}(readdlm(path, ','))
end

function solve_instance(A::AbstractMatrix, k::Int, fixing_rule::Symbol)
    if METHOD == :bnb
        runtime = @elapsed begin
            x_best, stats = solve_bnb(
                A,
                k;
                fixing_rule=fixing_rule,
                resolve=RESOLVE,
                time_limit=TIME_LIMIT,
                eps=EPS,
                proj_eps=PROJ_EPS,
                verbose=false,
            )
        end

    elseif METHOD == :lr
        runtime = @elapsed begin
            x_best, stats = solve_bnb_lr(
                A,
                k;
                fixing_rule=fixing_rule,
                resolve=RESOLVE,
                time_limit=TIME_LIMIT,
                sdp_timelimit=SDP_TIMELIMIT,
                tol=EPS,
                verbose=false,
            )
        end

    elseif METHOD == :bnc
        runtime = @elapsed begin
            x_best, stats = solve_bnc(
                A,
                k;
                fixing_rule=fixing_rule,
                resolve=RESOLVE,
                max_cuts=MAX_CUTS,
                time_limit=TIME_LIMIT,
                eps=EPS,
                verbose=false,
            )
        end

    else
        error("Invalid METHOD.")
    end

    return x_best, stats, runtime
end

function build_result_row(instance_name::String, preprocessing::String, m::Int, n::Int, k_over_n::Float64, k::Int, fixing_rule::Symbol, start_time::DateTime, end_time::DateTime, runtime::Float64, objective_value::Float64, stats)
    return (
        method=String(METHOD),
        instance_name=instance_name,
        preprocessing=preprocessing,
        m=m,
        n=n,
        n_over_m=n / m,
        k_over_n=k_over_n,
        k=k,
        fixing_rule=String(fixing_rule),
        max_cuts=METHOD == :bnc ? MAX_CUTS : missing,
        start_time=Dates.format(start_time, dateformat"yyyy-mm-dd HH:MM:SS"),
        end_time=Dates.format(end_time, dateformat"yyyy-mm-dd HH:MM:SS"),
        runtime_seconds=runtime,
        solver_wall_time_seconds=stats.wall_time,
        objective_ub=objective_value,
        reported_ub=stats.UB,
        lower_bound=stats.LB,
        gap=stats.gap,
        root_lower_bound=stats.root_LB,
        nodes_processed=stats.nodes,
        open_nodes=stats.open_nodes,
        cuts_generated=METHOD == :bnc ? stats.ncuts : missing,
        fixed_to_zero=stats.nfix0,
        fixed_to_one=stats.nfix1,
        time_limit_hit=stats.time_limit_hit,
        tree_exhausted=stats.tree_exhausted,
        status=stats.status,
    )
end

function run_real_benchmark()
    completed = completed_experiments(OUTPUT_FILE)

    total_cases =
        length(INSTANCE_NAMES) *
        length(PREPROCESSINGS) *
        length(K_OVER_N_VALUES) *
        length(FIXING_RULES)

    println("Real-world AOPT benchmark")
    println("-------------------------")
    @printf("method          : %s\n", String(METHOD))
    @printf("configured runs : %d\n", total_cases)

    run_number = 0

    for instance_name in INSTANCE_NAMES
        for preprocessing in PREPROCESSINGS
            A = read_instance(instance_name, preprocessing)
            m, n = size(A)

            for k_over_n in K_OVER_N_VALUES
                k = ceil(Int, k_over_n * n)

                for fixing_rule in FIXING_RULES
                    run_number += 1

                    key = experiment_key(
                        METHOD,
                        instance_name,
                        preprocessing,
                        k_over_n,
                        fixing_rule,
                    )

                    if key in completed
                        @printf(
                            "[%d/%d] SKIP  instance=%s  preprocessing=%s  k=%d  fixing=%s\n",
                            run_number,
                            total_cases,
                            instance_name,
                            preprocessing,
                            k,
                            String(fixing_rule),
                        )
                        continue
                    end

                    start_time = now()

                    println()
                    println("============================================================")
                    @printf("[%d/%d] START\n", run_number, total_cases)
                    @printf("OS time           : %s\n", Dates.format(start_time, dateformat"yyyy-mm-dd HH:MM:SS"))
                    @printf("method            : %s\n", String(METHOD))
                    @printf("instance          : %s\n", instance_name)
                    @printf("preprocessing     : %s\n", preprocessing)
                    @printf("m                 : %d\n", m)
                    @printf("n                 : %d\n", n)
                    @printf("n/m               : %.3f\n", n / m)
                    @printf("k/n               : %.3f\n", k_over_n)
                    @printf("k                 : %d\n", k)
                    @printf("fixing rule       : %s\n", String(fixing_rule))
                    flush(stdout)

                    local x_best
                    local stats
                    local runtime

                    Random.seed!(SEED)

                    try
                        x_best, stats, runtime = solve_instance(A, k, fixing_rule)

                        end_time = now()

                        objective_value = objective(
                            information_matrix(x_best, A),
                        )

                        row = build_result_row(
                            instance_name,
                            preprocessing,
                            m,
                            n,
                            k_over_n,
                            k,
                            fixing_rule,
                            start_time,
                            end_time,
                            runtime,
                            objective_value,
                            stats,
                        )

                        append_result(OUTPUT_FILE, row)
                        push!(completed, key)

                        @printf("finish OS time    : %s\n", Dates.format(end_time, dateformat"yyyy-mm-dd HH:MM:SS"))
                        @printf("runtime           : %.6f seconds\n", runtime)
                        @printf("final gap         : %.10f\n", stats.gap)
                        @printf("status            : %s\n", stats.status)
                        flush(stdout)

                    catch err
                        println()
                        @printf(
                            "ERROR: instance=%s preprocessing=%s k=%d fixing=%s\n",
                            instance_name,
                            preprocessing,
                            k,
                            String(fixing_rule),
                        )

                        showerror(stderr, err)
                        println(stderr)

                        GC.gc(true)
                        rethrow(err)
                    end

                    x_best = nothing
                    stats = nothing

                    GC.gc(true)
                end
            end

            A = nothing
            GC.gc(true)
        end
    end

    println()
    println("============================================================")
    println("Real-world benchmark completed.")
    @printf("Results saved to: %s\n", OUTPUT_FILE)
end

run_real_benchmark()