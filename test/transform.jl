using GCMAES, Plots

xs = Float32.(collect(-6:0.1:6))
lo = fill(1f0, size(xs))
hi = fill(2f0, size(xs))
trans = BoxLinQuadTransform(lo, hi)
plot(xs, GCMAES.transform(trans, xs))
plot(xs, GCMAES.inverse(trans, xs))
