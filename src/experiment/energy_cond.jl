A = Matrix{Float64}(readdlm(
    "data/aopt_matrices/A_energy_raw.csv",
    ',',
))

m, n = size(A)

println("m = ", m)
println("rank(A) = ", rank(A))

G = Symmetric(A * A')

println("isposdef(A*A') = ", isposdef(G))
println("eigenvalues(A*A') = ", eigvals(G))
println("cond(A*A') = ", cond(Matrix(G)))