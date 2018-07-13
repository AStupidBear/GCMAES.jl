abstract type Constraint end

transform(c::Constraint, x) = x

getpenalty(c::Constraint, x) = 0

# NoConstraint
mutable struct NoConstraint <: Constraint
end

# RangeConstraint lo <= x <= hi
mutable struct RangeConstraint{T} <: Constraint
    lo::Vector{T}
    hi::Vector{T}
    λ::T # penalty scaling factor
end

transform(c::RangeConstraint, x) = ifelse.(x .< c.lo, c.lo, ifelse.(x .> c.hi, c.hi, x))

function getpenalty(c::RangeConstraint, x)
    xt = transform(c, x)
    penalty = c.λ * maximum(abs.(x .- xt) ./ (c.hi .- c.lo))
end

# NormConstraint norm(x, p) <= maxnorm
mutable struct NormConstraint{T} <: Constraint
    p::Int
    maxnorm::T
    λ::T # penalty scaling factor
end

function transform(c::NormConstraint, x)
    n = norm(x, c.p)
    n > c.maxnorm ? x ./ n .* c.maxnorm : x
end

function getpenalty(c::NormConstraint, x)
    xt = transform(c, x)
    penalty = c.λ * vecnorm(x .- xt, c.p) / vecnorm(xt, c.p)
end

# MaxNormConstraint
mutable struct MaxNormConstraint{T1, T2} <: Constraint
    weight_inds::Vector{T1}
    norm_constraint::NormConstraint{T2}
    allnorm::Bool # true: use all weight indices for transform
    λ::T2 # penalty scaling factor
end

MaxNormConstraint(weight_inds, maxnorm::Real, allnorm, λ) =  
    MaxNormConstraint(weight_inds, NormConstraint(2, maxnorm, λ), allnorm, λ)

function transform(c::MaxNormConstraint, x)
    y = copy(x)
    if c.allnorm
        ind = vcat(c.weight_inds)
        setindex!(y, transform(c.norm_constraint, x[ind]), ind)
    else
        for ind in c.weight_inds
            setindex!(y, transform(c.norm_constraint, x[ind]), ind)
        end
    end
    return y
end

function getpenalty(c::MaxNormConstraint, x)
    penalty = zero(eltype(x))
    for ind in c.weight_inds
        w = x[ind]
        wt = transform(c.norm_constraint, w)
        penalty += c.λ * vecnorm(w .- wt) / vecnorm(wt)
    end
    return penalty
end