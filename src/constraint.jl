abstract type Constraint end

transform!(c::Constraint, x) = x

transform(c::Constraint, x) = transform!(c, copy(x))

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

transform!(c::RangeConstraint, x) = clamp!(x, c.lo, c.hi)

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

function transform!(c::NormConstraint, x)
    n = norm(x, c.p)
    if n > c.θ
        rmul!(x, c.θ / n)
    end
    return x
end

function getpenalty(c::NormConstraint, x)
    xt = transform(c, x)
    penalty = c.λ * norm(x .- xt, c.p)
end

# MaxNormConstraint
mutable struct MaxNormConstraint{T1, T2} <: Constraint
    winds::Vector{T1}
    nrmconstr::NormConstraint{T2}
    allnorm::Bool # true: use all weight indices for transform
end

MaxNormConstraint(winds, θ::Real, λ, allnorm) = MaxNormConstraint(winds, NormConstraint(2, θ, λ), allnorm)

function transform!(c::MaxNormConstraint, x)
    if c.allnorm
        ind = vcat(c.winds...)
        transform!(c.nrmconstr, view(x, ind))
    else
        for ind in c.winds
            transform(c.nrmconstr, view(x, ind))
        end
    end
    return x
end

function getpenalty(c::MaxNormConstraint, x)
    penalty = zero(eltype(x))
    inds = c.allnorm ? [vcat(c.winds...)] : c.winds
    for ind in inds
        penalty += getpenalty(c.nrmconstr, x[ind])
    end
    return penalty
end

mutable struct LpPenalty{T} <: Constraint
    p::Int # p-norm
    λ::T # penalty scaling factor
    θ::T # penalty margin margin
end

getpenalty(c::LpPenalty, x) =  c.λ * max(0, norm(x, c.p) - c.θ)^c.p

mutable struct LpWeightPenalty{T1, T2} <: Constraint
    winds::Vector{T1}
    pnlty::LpPenalty{T2}
end

LpWeightPenalty(winds, p, λ, θ) = LpWeightPenalty(winds, LpPenalty(p, λ, θ))

getpenalty(c::LpWeightPenalty, x) = getpenalty(c.pnlty, x[vcat(c.winds...)])

mutable struct MultiConstraints <: Constraint
    constrs::Tuple
end

MultiConstraints(cs::Constraint...) = MultiConstraints(cs)

getpenalty(c::MultiConstraints, x) = sum(c -> getpenalty(c, x), c.constrs)