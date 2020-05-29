export BoxLinQuadTransform

abstract type Transform end

transform(trans::Transform, x) = x

inverse(trans::Transform, x) = x

struct NoTransform <: Transform
end

struct BoxLinQuadTransform{T}
    lo::Vector{T}
    hi::Vector{T}
end

function transform(trans::BoxLinQuadTransform, x)
    # TODO
    return x
end

function inverse(trans::BoxLinQuadTransform, x)
    # TODO
    return x
end