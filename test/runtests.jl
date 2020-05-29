using GCMAES
using Zygote
using Distributed
using Test

@everywhere rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))

D = 2000
x0 = fill(0.3, D)
σ0 = 0.2
lo = fill(-5.12, D)
hi = fill(5.12, D)

GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200)
GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200, autodiff = true)
GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200, constr = true, α = 2)
GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200, constr = true, α = 2)
GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200, constr = BoxConstraint(lo, hi, 2))
GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200, constr = NormConstraint(2))

rm("CMAES.bson", force = true)