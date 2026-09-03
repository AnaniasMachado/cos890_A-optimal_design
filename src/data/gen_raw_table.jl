using CSV
using DataFrames
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

function write_method_table(df::DataFrame, method::String, output_file::String)
    method_df = filter(row -> row.method == method, df)

    open(output_file, "w") do io
        println(io, "\\begin{table}[!ht]")
        println(io, "    \\centering")
        println(io, "    \\scriptsize")
        println(io, "    \\setlength{\\tabcolsep}{3pt}")
        println(io, "    \\renewcommand{\\arraystretch}{0.95}")
        println(io, "    \\caption{$(method_name(method)) results on the UCI benchmark instances.}")
        println(io, "    \\label{tab:$(method)_uci_results}")
        println(io, "    \\begin{tabular}{lll|rrrr}")
        println(io, "    \\hline")
        println(io, "    \\textbf{Instance} & \\textbf{Preprocessing} & \\textbf{Variable fixing} & \\textbf{Final gap} & \\textbf{Nodes} & \\textbf{Fixed vars.} & \\textbf{Runtime} \\\\")
        println(io, "    \\hline")

        for row in eachrow(method_df)
            final_gap = Float64(row.gap)
            fixed_vars = Int(row.fixed_to_zero) + Int(row.fixed_to_one)

            instance = replace(string(row.instance_name), "_" => "\\_")
            preprocessing = replace(string(row.preprocessing), "_" => "\\_")
            fixing = fixing_name(string(row.fixing_rule))

            @printf(io, "    %s & %s & %s & %.6f & %d & %d & %.3f \\\\\n",
                instance,
                preprocessing,
                fixing,
                final_gap,
                Int(row.nodes_processed),
                fixed_vars,
                Float64(row.runtime_seconds),
            )
        end

        println(io, "    \\hline")
        println(io, "    \\end{tabular}")
        println(io, "\\end{table}")
    end
end

df = CSV.read(INPUT_FILE, DataFrame)

write_method_table(df, "bnb", "results/bnb_results_table.txt")
write_method_table(df, "lr", "results/lr_results_table.txt")
write_method_table(df, "bnc", "results/bnc_results_table.txt")