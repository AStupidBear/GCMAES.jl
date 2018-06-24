minibatch(x, b) = [x[i:min(end, i + b - 1)] for i in 1:b:max(1, length(x) - b + 1)]

sample(lo, hi) = lo .+ rand(size(lo)) .* (hi .- lo)

function ptp(x)
    xmin, xmax = extrema(x)
    xmax - xmin
end

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

function throttle(f, timeout; leading=true, trailing=false)
  cooldown = true
  later = nothing
  result = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        result = f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @schedule try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> (result = f(args...; kwargs...))
    end

    return result
  end
end