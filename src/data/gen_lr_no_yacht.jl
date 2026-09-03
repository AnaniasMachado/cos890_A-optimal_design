using CSV
using DataFrames
using Statistics
using Printf

df = CSV.read("results/uci_benchmark.csv", DataFrame)

lr = filter(row -> row.method == "lr" && row.instance_name != "yacht", df)

summary = combine(
    groupby(lr, [:preprocessing, :fixing_rule]),
    :gap => mean => :avg_gap,
    :nodes_processed => mean => :avg_nodes,
    [:fixed_to_zero, :fixed_to_one] => ((z, o) -> mean(z .+ o)) => :avg_fixed,
    :runtime_seconds => mean => :avg_runtime,
    :status => (s -> count(x -> occursin("OPTIMAL", uppercase(string(x))), s)) => :solved,
)

preprocessing_order = Dict("raw" => 1, "normalized" => 2, "standardized" => 3)
fixing_order = Dict("none" => 1, "dual" => 2)

sort!(summary, [:preprocessing, :fixing_rule],
    order=[
        Base.Order.By(x -> get(preprocessing_order, string(x), 99)),
        Base.Order.By(x -> get(fixing_order, string(x), 99))
    ])

output_file = "results/lr_average_results_without_yacht_table.txt"

open(output_file, "w") do io
    println(io, "\\begin{table}[!ht]")
    println(io, "    \\centering")
    println(io, "    \\scriptsize")
    println(io, "    \\setlength{\\tabcolsep}{3pt}")
    println(io, "    \\renewcommand{\\arraystretch}{0.95}")
    println(io, "    \\caption{Average LR results over the UCI benchmark instances excluding Yacht Hydrodynamics.}")
    println(io, "    \\label{tab:lr_uci_average_results_without_yacht}")
    println(io, "    \\begin{tabular}{ll|rrrrr}")
    println(io, "    \\hline")
    println(io, "    \\textbf{Preprocessing} & \\textbf{Variable fixing} & \\textbf{Avg. final gap} & \\textbf{Avg. nodes} & \\textbf{Avg. fixed vars.} & \\textbf{Avg. runtime} & \\textbf{Solved} \\\\")
    println(io, "    \\hline")

    for row in eachrow(summary)
        fixing = row.fixing_rule == "none" ? "None" : "Dual"

        @printf(io, "    %s & %s & %.6f & %.1f & %.1f & %.3f & %d \\\\\n",
            string(row.preprocessing),
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