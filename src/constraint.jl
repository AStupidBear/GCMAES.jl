export BoxConstraint, NormConstraint

abstract type Constraint end

repair!(c::Constraint, x) = x

repair(c::Constraint, x) = repair!(c, copy(x))

getfitness(f, c::Constraint, x) = f(x)

# NoConstraint
mutable struct NoConstraint <: Constraint
end

# BoxConstraint lo <= x <= hi
mutable struct BoxConstraint{T} <: Constraint
    lo::Vector{T} # lower bound
    hi::Vector{T} # upper bound
    α::T          # penalty scaling factor
    BoxConstraint(lo, hi, α = one(eltype(lo))) = new{eltype(lo)}(lo, hi, α)
end

repair!(c::BoxConstraint, x) = map!(clamp, x, x, c.lo, c.hi)

function getfitness(f, c::BoxConstraint, x)
    x_repair = repair(c, x)
    violation = norm(x .- x_repair)
    f(x_repair) + c.α * violation
end

# NormConstraint norm(x, p) <= θ
mutable struct NormConstraint{T} <: Constraint
    θ::T        # max norm
    p::Int      # p-norm
    α::T        # penalty scaling factor
    NormConstraint(θ, p = 2, α = one(θ)) = new{typeof(θ)}(θ, p, α)
end

function repair!(c::NormConstraint, x)
    nrm = norm(x, c.p)
    if nrm > c.θ
        rmul!(x, c.θ / nrm)
    end
    return x
end

function getfitness(f, c::NormConstraint, x)
    x_repair = repair(c, x)
    violation = norm(x .- x_repair, c.p)
    f(x_repair) + c.α * violation
end