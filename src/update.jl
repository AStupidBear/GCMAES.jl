using Base: axpy!

mutable struct Sgd
    lr::Float64
    gclip::Float64
end

const SGDLR = 0.001

Sgd(;lr = SGDLR, gclip = 1) = Sgd(lr, gclip)

mutable struct Momentum
    lr::Float64
    gclip::Float64
    gamma::Float64
    velocity::Vector{Float64}
end

Momentum(;lr = 0.001, gclip = 1, gamma = 0.9) = Momentum(lr, gclip, gamma, [])

mutable struct Rmsprop
    lr::Float64
    gclip::Float64
    rho::Float64
    eps::Float64
    G::Vector{Float64}
end

Rmsprop(;lr = 0.001, gclip = 1, rho = 0.9, eps = 1e-6) = Rmsprop(lr, gclip, rho, eps, [])

mutable struct Adam
    lr::Float64
    gclip::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    t::Int
    fstm::Vector{Float64}
    scndm::Vector{Float64}
end

Adam(;lr = 0.001, gclip = 1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) = 
    Adam(lr, gclip, beta1, beta2, eps, 0, [], [])

function update!(w, g, p::Sgd)
    gclip!(g, p.gclip)
    axpy!(-p.lr, g, w)
end

function update!(w, g, p::Momentum)
    gclip!(g, p.gclip)
    if isempty(p.velocity); p.velocity = zeros(w); end
    scale!(p.gamma, p.velocity)
    axpy!(p.lr, g, p.velocity)
    axpy!(-1, p.velocity, w)
end

function update!(w, g, p::Adam)
    gclip!(g, p.gclip)
    if isempty(p.fstm); p.fstm = zeros(w); p.scndm = zeros(w); end
    p.t += 1
    scale!(p.beta1, p.fstm)
    axpy!(1 - p.beta1, g, p.fstm)
    scale!(p.beta2, p.scndm)
    axpy!(1 - p.beta2, g .* g, p.scndm)
    fstm_corrected = p.fstm ./ (1 - p.beta1 ^ p.t)
    scndm_corrected = p.scndm ./ (1 - p.beta2 ^ p.t)
    axpy!(-p.lr, (fstm_corrected ./ (sqrt.(scndm_corrected) .+ p.eps)), w)
end

function update!(w, g, p::Rmsprop)
    gclip!(g, p.gclip)
    if isempty(p.G); p.G = zeros(w); end
    scale!(p.rho, p.G)
    axpy!(1 - p.rho, g .* g, p.G)
    axpy!(-p.lr, g ./ sqrt.(p.G .+ p.eps), w)
end

function gclip!(g, gclip)
    gclip == 0 && return g
    gnorm = vecnorm(g)
    gnorm > gclip && scale!(gclip / gnorm, g)
    return g
end