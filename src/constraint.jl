abstract type Constraint end

transform(c::Constraint, x) = x

getpenalty(c::Constraint, x) = zero(eltype(x))

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
    p::Int # p-norm
    maxnorm::T
    λ::T # penalty scaling factor
end

function transform(c::NormConstraint, x)
    n = norm(x, c.p)
    n > c.maxnorm ? x ./ n .* c.maxnorm : x
end

function getpenalty(c::NormConstraint, x)
    xt = transform(c, x)
    penalty = c.λ * vecnorm(x .- xt, c.p) / (vecnorm(xt, c.p) + eps(eltype(x)))
end

# MaxNormConstraint
mutable struct MaxNormConstraint{T1, T2} <: Constraint
    weight_inds::Vector{T1}
    norm_constraint::NormConstraint{T2}
    allnorm::Bool # true: use all weight indices for transform
end

MaxNormConstraint(weight_inds, maxnorm::Real, λ, allnorm) =  
    MaxNormConstraint(weight_inds, NormConstraint(2, maxnorm, λ), allnorm)

function transform(c::MaxNormConstraint, x)
    y = copy(x)
    if c.allnorm
        ind = vcat(c.weight_inds...)
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
    inds = c.allnorm ? [vcat(c.weight_inds...)] : c.weight_inds
    for ind in inds
        penalty += getpenalty(c.norm_constraint, x[ind])
    end
    return penalty
end

mutable struct NormPenalty{T} <: Constraint
    p::Int # p-norm
    λ::T # penalty scaling factor
end

getpenalty(c::NormPenalty, x) =  c.λ * vecnorm(x, c.p)

mutable struct MaxNormPenalty{T1, T2} <: Constraint
    weight_inds::Vector{T1}
    norm_penalty::NormPenalty{T2}
end

MaxNormPenalty(weight_inds, p, λ) = MaxNormPenalty(weight_inds, NormPenalty(p, λ))

getpenalty(c::MaxNormPenalty, x) = getpenalty(c.norm_penalty, x[vcat(c.weight_inds...)])