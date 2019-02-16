using GCMAES, ForwardDiff, Test, Distributed

@everywhere begin
    rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))
    ∇rastrigin(x) =  ForwardDiff.gradient(rastrigin, x)
end

N = 2000; x0, σ0, lo, hi = 0.3ones(N), 0.2, fill(-5.12, N), fill(5.12, N)
xmin, fmin, = GCMAES.minimize(rastrigin, x0, σ0, lo, hi; maxiter = 200);
xmin, fmin, = GCMAES.minimize((rastrigin, ∇rastrigin), x0, σ0, lo, hi; maxiter = 200);

rm("CMAES.bson")