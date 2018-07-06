minibatch(x, b) = [x[i:min(end, i + b - 1)] for i in 1:b:max(1, length(x) - b + 1)]

sample(lo, hi) = lo .+ rand(size(lo)) .* (hi .- lo)

function ptp(x)
    xmin, xmax = extrema(x)
    xmax - xmin
end

call(f, xs...) = f(xs...)

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)