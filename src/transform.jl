export BoxLinQuadTransform

abstract type Transform end

transform(t::Transform, x) = x

inverse(t::Transform, x) = x

struct NoTransform <: Transform
end

struct BoxLinQuadTransform{T}
    lb::Vector{T}
    ub::Vector{T}
    al::Vector{T}
    au::Vector{T}
    scaling::Bool
end

function BoxLinQuadTransform(lb, ub; scaling = false)
    al = @. min((ub - lb) / 2, (1 + abs(lb)) / 20)
    au = @. min((ub - lb) / 2, (1 + abs(ub)) / 20)
    BoxLinQuadTransform(lb, ub, al, au, scaling)
end

function transform(tr::BoxLinQuadTransform, x)
    map(tr.scaling ? tr.lb .+ (tr.ub .- tr.lb) .* x : x,
        tr.lb, tr.ub, tr.al, tr.au) do y, lb, ub, al, au
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

function inverse(tr::BoxLinQuadTransform, x)
    y = map(x, tr.lb, tr.ub, tr.al, tr.au) do y, lb, ub, al, au
        z, y = y, clamp(y, lb, ub)
        y = ifelse(y < lb + al, (lb - al) + 2 * sqrt(al * (y - lb)),
            ifelse(y > ub - au, (ub + au) - 2 * sqrt(au * (ub - y)), y))
        oftype(z, y)
    end
    tr.scaling ? (y .- tr.lb) ./ (tr.ub .- tr.lb) : y
end
