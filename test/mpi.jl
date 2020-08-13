using GCMAES
using Zygote
using Test

rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))

D = 5000
x0 = fill(0.3f0, D)
σ0 = 0.2f0
lo = fill(-5.12f0, D)
hi = fill(5.12f0, D)

@mpirun GCMAES.minimize(x0, σ0, lo, hi, λ = 3, maxiter = 200) do x
    GCMAES.allmean(rastrigin(x), GCMAES.localcomm())
end

@mpirun GCMAES.minimize(rastrigin, x0, σ0, lo, hi, λ = 200, maxiter = 200)

@mpirun GCMAES.minimize(rastrigin, x0, σ0, lo, hi, λ = 200, maxiter = 200, autodiff = true)

rm("CMAES.bson", force = true)