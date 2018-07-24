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
    penalty = c.λ * maximum(abs.(x .- xt))
end

# NormConstraint norm(x, p) <= θ
mutable struct NormConstraint{T} <: Constraint
    p::Int # p-norm
    θ::T # max norm
    λ::T # penalty scaling factor
end

function transform(c::NormConstraint, x)
    n = norm(x, c.p)
    n > c.θ ? x ./ n .* c.θ : x
end

function getpenalty(c::NormConstraint, x)
    xt = transform(c, x)
    penalty = c.λ * vecnorm(x .- xt, c.p)
end

# MaxNormConstraint
mutable struct MaxNormConstraint{T1, T2} <: Constraint
    weight_inds::Vector{T1}
    norm_constraint::NormConstraint{T2}
    allnorm::Bool # true: use all weight indices for transform
end

MaxNormConstraint(weight_inds, θ::Real, λ, allnorm) = MaxNormConstraint(weight_inds, NormConstraint(2, θ, λ), allnorm)

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

mutable struct LpPenalty{T} <: Constraint
    p::Int # p-norm
    λ::T # penalty scaling factor
    θ::T # penalty margin margin
end

getpenalty(c::LpPenalty, x) =  c.λ * max(0, vecnorm(x, c.p) - θ)^c.p

mutable struct LpWeightPenalty{T1, T2} <: Constraint
    weight_inds::Vector{T1}
    lp_penalty::LpPenalty{T2}
end

LpWeightPenalty(weight_inds, p, λ, θ) = LpWeightPenalty(weight_inds, LpPenalty(p, λ, θ))

getpenalty(c::LpWeightPenalty, x) = getpenalty(c.lp_penalty, x[vcat(c.weight_inds...)])