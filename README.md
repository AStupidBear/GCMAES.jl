# Gradient-based Covariance Matrix Adaptation Evolutionary Strategy for Real Blackbox  Optimization

[![Build Status](https://github.com/AStupidBear/GCMAES.jl/workflows/CI/badge.svg)](https://github.com/AStupidBear/GCMAES.jl/actions)
[![Coverage](https://codecov.io/gh/AStupidBear/GCMAES.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AStupidBear/GCMAES.jl)

## Installation

```julia
using Pkg
pkg"add GCMAES"
```

## Features

- use low level BLAS operations to ensure performance
- use `Elemental` to do distributed eigendecomposition, which is crutial for high dimensional (>10000) problem
- compatible with `julia`'s native parallelism
- compatible with `MPI.jl`, therefore suitable to be run on clusters without good TCP connections
- handling constraints and transformations

## Basic Usage

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

A checkpoint file named `CMAES.bson` will be created in the current working directory during optimization, which will be loaded back to initilize `CMAESOpt` if dimensions are equal.

## Incoporating Gradient

You can speed up the optimization process by providing additional gradient infomation if the loss function is differentialble but noisy. The evolution part can help escaping local minima while the gradient part can speed up convergence in non-noisy regions.

```julia
using ForwardDiff
∇rastrigin(x) = ForwardDiff.gradient(rastrigin, x)
GCMAES.minimize((rastrigin, ∇rastrigin), x0, σ0, lo, hi, maxiter = 200)
```

You can also enable `autodiff` and then `GCMAES` will internally use `Zygote` to do the gradient calculation

```julia
using Zygote
GCMAES.minimize((rastrigin, ∇rastrigin), x0, σ0, lo, hi, maxiter = 200, autodiff = true)
```

## Parallel Usage

Just simply add `@mpirun` before `GCMAES.minimize`

```julia
# ....
@mpirun GCMAES.minimize(...)
# ....
```

Then you can use `mpirun -n N julia ...` or `julia -p N ...` to start your job.