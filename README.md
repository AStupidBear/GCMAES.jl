# Gradient-based Covariance Matrix Adaptation Evolutionary Strategy for Real Blackbox  Optimization

## Installation

```julia
using Pkg
pkg"add GCMAES"
```

## Usage

```julia
using GCMAES
D = 2000            # dimension of x
x0 = fill(0.3, D)   # initial x
σ0 = 0.2            # initial search variance
lo = fill(-5.12, D) # lower bound for each dimension
hi = fill(5.12, D)  # upper bound for each dimension
```

Minimize a blackbox function

```julia
rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))
xmin, fmin, status = GCMAES.minimize(rastrigin, x0, σ0, lo, hi, maxiter = 200)
```

If the optimization terminate prematurely before `maxiter` is reached, `status` will be `1`, otherwise `0`.

You can speed up the optimization process by providing additional gradient infomation if the loss function is differentialble but noisy. The evolution part can help escaping local minima while the gradient part can speed up convergence in non-noisy regions.

```julia
using ForwardDiff
∇rastrigin(x) = ForwardDiff.gradient(rastrigin, x)
GCMAES.minimize((rastrigin, ∇rastrigin), x0, σ0, lo, hi, maxiter = 200)
```

A checkpoint file named `CMAES.bson` will be created in the current working directory during optimization, which will be loaded back to initilize `CMAESOpt` if dimensions are equal.