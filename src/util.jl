minibatch(x, b) = [x[i:min(end, i + b - 1)] for i in 1:b:max(1, length(x) - b + 1)]

sample(lo, hi) = lo .+ rand(size(lo)) .* (hi .- lo)

function ptp(x)
    xmin, xmax = extrema(x)
    xmax - xmin
end

runall(f) = f
runall(fs::AbstractVector) = (xs...) -> last([f(xs...) for f in fs])

function throttle(f, timeout; leading = true)
    lasttime = time()
    leading && (lasttime -= timeout)
    function throttled(args...; kwargs...)
        result = nothing
        if time() >= lasttime + timeout
            result = f(args...; kwargs...)
            lasttime = time()
        end
        return result
    end
end