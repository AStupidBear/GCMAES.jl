export BoxLinQuadTransform

abstract type Transform end

transform(trans::Transform, x) = x

inverse(trans::Transform, x) = x

struct NoTransform <: Transform
end

struct BoxLinQuadTransform{T}
    lb::Vector{T}
    ub::Vector{T}
    al::Vector{T}
    au::Vector{T}
end

function BoxLinQuadTransform(lb, ub)
    al = @. min((ub - lb) / 2, (1 + abs(lb)) / 20)
    au = @. min((ub - lb) / 2, (1 + abs(ub)) / 20)
    BoxLinQuadTransform(lb, ub, al, au)
end

function transform(trans::BoxLinQuadTransform, x)
    map(x, trans.lb, trans.ub, trans.al, trans.au) do y, lb, ub, al, au
        yl = lb - 2 * al - (ub - lb) / 2
        yu = ub + 2 * au + (ub - lb) / 2
        z, r = y, 2 * (ub - lb + al + au)
        y = ifelse(y < yl, y + r * ceil((yl - y) / r),
            ifelse(y > yu, y - r * ceil((y - yu) / r), y))
        y = ifelse(y < lb - al, y + 2 * (lb - al - y),
            ifelse(y > ub + au, y - 2 * (y - ub - au), y))
        y = ifelse(y < lb + al, lb + (y - (lb - al))^2 / 4 / al,
            ifelse(y > ub - au, ub -(y - (ub + au))^2 / 4 / au, y))
        oftype(z, y)
    end
end

function inverse(trans::BoxLinQuadTransform, x)
    map(x, trans.lb, trans.ub, trans.al, trans.au) do y, lb, ub, al, au
        z, y = y, clamp(y, lb, ub)
        y = ifelse(y < lb + al, (lb - al) + 2 * sqrt(al * (y - lb)),
            ifelse(y > ub - au, (ub + au) - 2 * sqrt(au * (ub - y)), y))
        oftype(z, y)
    end
end