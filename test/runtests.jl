using GCMAES, ForwardDiff

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@everywhere begin
    rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))
    ∇rastrigin(x) =  ForwardDiff.gradient(rastrigin, x)
    rastrigin′ = x -> (rastrigin(x), ∇rastrigin(x))
end

N = 2000; x0, σ0, lo, hi = 0.3ones(N), 0.2, fill(-5.12, N), fill(5.12, N)
xmin, fmin, = GCMAES.minimize(rastrigin, x0, σ0, lo, hi; grad = false, maxiter = 3000)
xmin, fmin, = GCMAES.minimize(rastrigin′, x0, σ0, lo, hi; grad = true, maxiter = 50, ν = 2)

rm("CMAES.jld")