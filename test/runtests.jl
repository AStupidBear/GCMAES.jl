using GCMAES, ForwardDiff

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@everywhere begin
    rastrigin(x) = -(10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x)))
    ∇rastrigin(x) =  ForwardDiff.gradient(rastrigin, x)
    f = x -> (rastrigin(x), ∇rastrigin(x))
end

N = 100; x0, σ0, lo, hi = 0.3ones(N), 0.2, fill(-5.12, N), fill(5.12, N)
xmin, fmin, = GCMAES.maximize(f, x0, σ0, lo, hi; maxiter = 50, ν = 0)
xmin, fmin, = GCMAES.maximize(f, x0, σ0, lo, hi; maxiter = 50, ν = 2)

rm("CMAES.jld")