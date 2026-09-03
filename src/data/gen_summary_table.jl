using CSV
using DataFrames
using Statistics
using Printf

const INPUT_FILE = "results/uci_benchmark.csv"

function fixing_name(rule)
    rule == "none" && return "None"
    rule == "dual" && return "Dual"
    rule == "primal" && return "Primal"
    return string(rule)
end

function method_name(method)
    method == "bnb" && return "BnB"
    method == "lr" && return "LR"
    method == "bnc" && return "BnC"
    return string(method)
end

function write_average_table(df::DataFrame, method::String, output_file::String)
    method_df = filter(row -> row.method == method, df)

    summary = combine(
        groupby(method_df, [:preprocessing, :fixing_rule]),
        :gap => mean => :avg_gap,
        :nodes_processed => mean => :avg_nodes,
        [:fixed_to_zero, :fixed_to_one] => ((z, o) -> mean(z .+ o)) => :avg_fixed,
        :runtime_seconds => mean => :avg_runtime,
        :status => (s -> count(x -> occursin("OPTIMAL", uppercase(string(x))), s)) => :solved,
    )

    preprocessing_order = Dict("raw" => 1, "normalized" => 2, "standardized" => 3)
    fixing_order = Dict("none" => 1, "dual" => 2, "primal" => 3)

    sort!(summary, [:preprocessing, :fixing_rule],
        order=[
            Base.Order.By(x -> get(preprocessing_order, string(x), 99)),
            Base.Order.By(x -> get(fixing_order, string(x), 99))
        ])

    open(output_file, "w") do io
        println(io, "\\begin{table}[!ht]")
        println(io, "    \\centering")
        println(io, "    \\scriptsize")
        println(io, "    \\setlength{\\tabcolsep}{3pt}")
        println(io, "    \\renewcommand{\\arraystretch}{0.95}")
        println(io, "    \\caption{Average $(method_name(method)) results over the UCI benchmark instances.}")
        println(io, "    \\label{tab:$(method)_uci_average_results}")
        println(io, "    \\begin{tabular}{ll|rrrrr}")
        println(io, "    \\hline")
        println(io, "    \\textbf{Preprocessing} & \\textbf{Variable fixing} & \\textbf{Avg. final gap} & \\textbf{Avg. nodes} & \\textbf{Avg. fixed vars.} & \\textbf{Avg. runtime} & \\textbf{Solved} \\\\")
        println(io, "    \\hline")

        for row in eachrow(summary)
            preprocessing = replace(string(row.preprocessing), "_" => "\\_")
            fixing = fixing_name(string(row.fixing_rule))

            @printf(io, "    %s & %s & %.6f & %.1f & %.1f & %.3f & %d \\\\\n",
                preprocessing,
                fixing,
                Float64(row.avg_gap),
                Float64(row.avg_nodes),
                Float64(row.avg_fixed),
                Float64(row.avg_runtime),
                Int(row.solved),
            )
        end

        println(io, "    \\hline")
        println(io, "    \\end{tabular}")
        println(io, "\\end{table}")
    end
end

df = CSV.read("results/uci_benchmark.csv", DataFrame)

write_average_table(df, "bnb", "results/bnb_average_results_table.txt")
write_average_table(df, "lr", "results/lr_average_results_table.txt")
write_average_table(df, "bnc", "results/bnc_average_results_table.txt")