using GCMAES
using LinearAlgebra
using Test

BLAS.set_num_threads(1)
@mpirun using Elemental

for d in (1000, 2000, 5000, 10000)
    x = rand(Float32, d, d)
    copyto!(x, Symmetric(x, :U))
    t = @elapsed GCMAES.deigen(x)
    @master println("dimension=$d, deigen=$t")
    t = @master @elapsed eigen(Symmetric(x, :U))
    @master println("dimension=$d, eigen=$t")
end
