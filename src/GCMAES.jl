__precompile__(true)

module GCMAES

using Printf, Distributed, LinearAlgebra
using Dates, Random, Statistics, Pkg
using Requires, BSON, FastClosures

include("util.jl")
include("constraint.jl")
include("transform.jl")

function __init__()
    @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
    @require Elemental="902c3f28-d1ec-5e7e-8399-a24c3845ee38" include("elemental.jl")
end

mutable struct CMAESOpt{T, F, G, CO, TR}
    # fixed hyper-parameters
    f::F
    g::G
    N::Int
    σ0::T
    lo::Vector{T}
    hi::Vector{T}
    constr::CO
    trans::TR
    rng::MersenneTwister
    # strategy parameter setting: selection
    λ::Int
    μ::Int
    w::Vector{T}
    μeff::T
    χₙ::T
    # strategy parameter setting: adaptation
    σ::T
    cc::T
    cσ::T
    c1::T
    cμ::T
    dσ::T
    # dynamic (internal) strategy parameters and constants
    x̄::Vector{T}
    pc::Vector{T}
    pσ::Vector{T}
    D::Vector{T}
    B::Matrix{T}
    BD::Matrix{T}
    C::Matrix{T}
    arx::Matrix{T}
    ary::Matrix{T}
    arz::Matrix{T}
    arfitness::Vector{T}
    arindex::Vector{Int}
    # recordings
    xmin::Vector{T}
    fmin::T
    fmins::Vector{T} # history of best fitness
    fmeds::Vector{T} # history of median fitness
    feqls::Vector{T} # history of equal fitness
    # report
    last_report_time::Float64
    pmap_time::Float64
    grad_time::Float64
    ls_time::Float64
    ls_dec::T
    file::String
end

function CMAESOpt(f, g, x0, σ0, lo, hi;
            λ = 0, constr = NoConstraint(), trans = false, 
            file = "CMAES.bson", rng = MersenneTwister())
    trans = trans == true ? BoxLinQuadTransform(lo, hi) :
            trans == false ? NoTransform() : trans
    x0′ = inverse(trans, x0)
    f′ = @closure z -> getfitness(f, constr, transform(trans, z))
    g′ = @closure z -> g(transform(trans, z))
    N, x̄, xmin, fmin, σ = length(x0), copy(x0′), copy(x0′), f′(x0′), σ0
    # strategy parameter setting: selection
    λ = λ == 0 ? round(Int, 4 + 3log(N)) : max(4, λ)
    μ = ceil(Int, λ / 2)                   # number of parents/points for recombination
    w = log(μ + 1/2) .- log.(1:μ)          # μXone array for weighted recombination
    normalize!(w, 1)                       # normalize recombination w array
    μeff = 1 / sum(abs2, w)                # variance-effectiveness of sum w_i x_i
    # strategy parameter setting: adaptation
    cc = 4 / (N + 4)                       # time constant for cumulation for C
    cσ = (μeff + 2) / (N + μeff + 3)       # t-const for cumulation for σ control
    c1 = 2 / ((N + 1.3)^2 + μeff)          # learning rate for rank-one update of C
    cμ = min(1 - c1, 2(μeff - 2 + 1 / μeff) / ((N + 2)^2 + μeff) )   # and for rank-μ update
    dσ = 1 + 2 * max(0, sqrt((μeff - 1) / (N + 1)) - 1) + cσ # damping for σ, usually close to 1
    # initialize dynamic (internal) strategy parameters and constants
    pc = zeros(N); pσ = zeros(N)            # evolution paths for C and σ
    D = fill(σ, N)                          # diagonal matrix D defines the scaling
    B = Matrix(I, N, N)                     # B defines the coordinate system
    BD = B .* reshape(D, 1, N)              # B*D for speed up only
    C = diagm(0 => D.^2)                    # covariance matrix == BD*(BD)'
    χₙ = sqrt(N) * (1 -1 / 4N + 1 / 21N^2)   # expectation of  ||N(0,I)|| == norm(randn(N,1))
    # init a few things
    arx, ary, arz = zeros(N, λ), zeros(N, λ), zeros(N, λ)
    arfitness, arindex = zeros(λ), ones(λ)
    @master @printf("%i-%i CMA-ES\n", λ, μ)
    # gradient
    T, F, G = eltype(x̄), typeof(f′), typeof(g′)
    CO, TR = typeof(constr), typeof(trans)
    return CMAESOpt{T, F, G, CO, TR}(
            f′, g′, N, σ0, lo, hi, 
            constr, trans, rng,
            λ, μ, w, μeff, χₙ, 
            σ, cc, cσ, c1, cμ, dσ,
            x̄, pc, pσ, D, B, BD, C,
            arx, ary, arz, arfitness, arindex,
            xmin, fmin, T[], T[], T[],
            time(), 0, 0, 0, 0, file)
end

function update_candidates!(opt::CMAESOpt)
    # generate and evaluate λ offspring
    randn!(opt.rng, opt.arz) # resample
    opt.ary = opt.BD * opt.arz
    opt.arx .= opt.x̄ .+ opt.σ .* opt.ary
    arx_cols = [opt.arx[:, k] for k in 1:opt.λ]
    @assert allequal(sum(opt.arx))
    opt.pmap_time = @elapsed opt.arfitness .= pmap(opt.f, arx_cols)
    # sort by fitness and compute weighted mean into x̄
    sortperm!(opt.arindex, opt.arfitness)
    opt.arfitness = opt.arfitness[opt.arindex] # minimization
    # store the best candidate
    if opt.arfitness[1] < opt.fmin
        opt.xmin, opt.fmin = opt.arx[:, opt.arindex[1]], opt.arfitness[1]
    end
    push!(opt.fmins, opt.arfitness[1])
    push!(opt.fmeds, opt.arfitness[ceil(Int, opt.λ / 2)])
    feql = opt.λ > 1 ? opt.arfitness[1] == opt.arfitness[ceil(Int, 1.1 + opt.λ / 4)] : 0
    push!(opt.feqls, feql)
end

function linesearch(f, x0::Array{T}, Δ) where T
    nrm = norm(Δ)
    nrm == 0 && return x0, typemax(T), zero(T)
    rmul!(Δ, 1 / nrm)
    αs = T[0.0; 2.0.^(2 - worldsize():0)]
    xs = [x0 .+ α .* Δ for α in αs]
    fs = pmap(f, xs)
    fx, i = findmin(fs)
    return xs[i], fx, fs[1] - fx
end

function update_mean!(opt::CMAESOpt)
    get(ENV, "GCMAES_USE_GRAD", "1") == "0" && return
    opt.grad_time = @elapsed Δ = -opt.g(opt.x̄)
    opt.ls_time = @elapsed opt.x̄, fx, opt.ls_dec = linesearch(opt.f, opt.x̄, Δ)
    if fx < opt.fmin
        opt.xmin .= opt.x̄
        opt.fmin = fx
    end
end

function update_parameters!(opt::CMAESOpt, iter, lazydecomp)
    indμ = opt.arindex[1:opt.μ]
    # calculate new x̄, this is selection and recombination
    x̄old = copy(opt.x̄)                                # for speed up of Eq. (2) and (3)
    mul!(opt.x̄, opt.arx[:, indμ], opt.w)
    # use parallel line search to determin the best α s.t. x += α * Δ achieves minimum
    update_mean!(opt)
    # calculate new z̄
    z̄ = opt.arz[:, indμ] * opt.w                       # == D^-1 * B' * (x̄ - x̄old) / σ
    # cumulation: update evolution paths
    BLAS.gemv!('N', sqrt(opt.cσ * (2 - opt.cσ) * opt.μeff), opt.B, z̄, 1 - opt.cσ, opt.pσ)  # i.e. pσ = (1 - cσ) * pσ + sqrt(cσ * (2 - cσ) * μeff) * (B * z̄) Eq.(4)
    hsig = norm(opt.pσ) / sqrt(1 - (1 - opt.cσ)^2iter) / opt.χₙ < 1.4 + 2 / (opt.N + 1)
    rmul!(opt.pc, 1 - opt.cc)
    BLAS.axpy!(hsig * sqrt(opt.cc * (2 - opt.cc) * opt.μeff) / opt.σ, opt.x̄ - x̄old, opt.pc) # i.e. pc = (1 - cc) * pc + (hsig * sqrt(cc * (2 - cc) * μeff) / σ) * (x̄ - x̄old)
    # adapt covariance matrix C
    rmul!(opt.C, (1 - opt.c1 - opt.cμ + (1 - hsig) * opt.c1 * opt.cc * (2 - opt.cc)))      # discard old C
    BLAS.syr!('U', opt.c1, opt.pc, opt.C)               # rank 1 update C += c1 * pc * pc'
    artmp = opt.ary[:, indμ]                            # μ difference vectors
    artmp = (artmp .* reshape(opt.w, 1, opt.μ)) * transpose(artmp)
    BLAS.axpy!(opt.cμ, artmp, opt.C)
    # adapt step size σ
    opt.σ *= exp((norm(opt.pσ) / opt.χₙ - 1) * opt.cσ / opt.dσ)  #Eq. (5)
    # update B and D from C
    if !lazydecomp || mod(iter, 1 / (opt.c1 + opt.cμ) / opt.N / 10) < 1 # if counteval - eigeneval > λ / (c1 + cμ) / N / 10  # to achieve O(N^2)
        (opt.D, opt.B) = deigen(opt.C)                  # eigen decomposition, B == normalized eigenvectors
        opt.D .= sqrt.(max.(opt.D, 0f0))                # D contains standard deviations now
        opt.BD .= opt.B .* reshape(opt.D, 1, opt.N)     # O(n^2)
    end
end

function terminate(opt::CMAESOpt, equal_best = Inf)
    histiter = 10 + round(Int, 30opt.N / opt.λ)
    stagiter = round(Int, 0.2 * length(opt.fmins) + 120 + 30opt.N / opt.λ)
    stagiter = min(stagiter, 20000)
    hist = opt.fmins[max(1, end - histiter):end]
    condition = Dict{String, Bool}()
    # FlatFitness: warn if 70% candidates' fitnesses are identical
    if opt.arfitness[1] == opt.arfitness[ceil(Int, 0.7opt.λ)]
        opt.σ *= exp(0.2 + opt.cσ / opt.dσ)
        @master println("warning: flat fitness, consider reformulating the objective")
    end
    # Stop conditions:
    # MaxIter: the maximal number of iterations in each run of CMA-ES
    length(opt.fmins) >= 100 + 50(opt.N + 3)^2 / sqrt(opt.λ) && get!(condition, "MaxIter", true)
    # TolHistFun: the range of the best function values during the last
    #             10 + ⌈30N/λ⌉ iterations is smaller than TolHistFun = 1e-13
    length(hist) > histiter && ptp(hist) < 1e-13 && get!(condition, "TolHistFun", true)
    # EqualFunVals: in more than 1/3rd of the last D iterations the objective function value of the best and
    #               the k-th best solution are identical, that is f(x1:λ) = f(xk:λ), where k = 1 + ⌈0.1 + λ/4⌉
    length(opt.fmins) > opt.N && mean(opt.feqls[end - opt.N:end]) > 0.33 && get!(condition, "EqualFunVals", true)
    # TolX: all components of pc and all square roots of diagonal components of C,
    #       multiplied by σ/σ0, are smaller than TolX = 1e-12.
    sqrt(maximum(abs, diag(opt.C))) * opt.σ / opt.σ0 < 1e-11 &&
    maximum(abs, opt.pc) * opt.σ / opt.σ0 < 1e-11 &&
    get!(condition, "TolX", true)
    # TolUpSigma: σ/σ0 >= maximum(D) * TolUpSigma = 1e20
    opt.σ / opt.σ0 > 1e20 * maximum(opt.D) && get!(condition, "TolUpSigma", true)
    # Stagnation: we track a history of the best and the median fitness in each iteration
    #             over the last 20% but at least 120 + 30n/λ and no more than 20000 iterations.
    #             We stop, if in both histories the median of the last (most recent) 30% values
    #             is not better than the median of the first 30%
    length(opt.fmins) > stagiter &&
    median(opt.fmins[end - 20:end]) >= median(opt.fmins[end - stagiter:end - stagiter + 20]) &&
    median(opt.fmeds[end - 20:end]) >= median(opt.fmeds[end - stagiter:end - stagiter + 20]) &&
    get!(condition, "Stagnation", true)
    # ConditionCov: the condition number of C exceeds 1e14
    maximum(opt.D) > 1e7 * minimum(opt.D) && get!(condition, "ConditionCov", true)
    # TolFun: stop if the range of the best objective function values of the last 10+⌈30n/λ⌉
    #         generations and all function values of the recent generation is below TolFun = 1e-12.
    length(hist) > histiter && ptp(vcat(hist, opt.arfitness)) < 1e-11 && get!(condition, "TolFun", true)
    # EqualBest: terminate is the range of the fmins of the last EqualBest iterations is smaller than 1e-12
    length(opt.fmins) > equal_best &&
    ptp(accumulate(min, opt.fmins)[end - equal_best:end]) < 1e-12 &&
    get!(condition, "EqualBest", true)
    # NoEffectAxis: stop if adding a 0.1-standard deviation vector in any principal axis
    #               direction of C does not change m.
    # NoEffectCoor: stop if adding 0.2-standard deviations in any single coordinate does not
    #               change m (i.e. mᵢ equals mᵢ + 0.2σcᵢᵢ, for any i)
    # TolFacUpX: terminate when step-size increases by TolFacUpx = 1e3 (diverges). That is,
    #            the initial step-size was chosen far too small and better solutions were found
    #            far away from the initial solution x0.
    # https://github.com/DEAP/deap/blob/8b8c82186083af4b78bf67d9bb3508cf1489a4be/examples/es/cma_bipop.py
    # http://cma.gforge.inria.fr/cmaes.m
    # Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed
    termination = false
    for (k, v) in condition
        @master v && printstyled("Termination Condition Satisfied: ", k, '\n', color = :red)
        termination = termination | v
    end
    return termination
end

function restart(opt::CMAESOpt)
    @master @printf("restarting...\n")
    opt′ = CMAESOpt(opt.f, sample(opt.lo, opt.hi), opt.σ0, opt.lo, opt.hi; 
                    λ = 2opt.λ, constr = opt.constr, trans = opt.constr)
    opt′.xmin, opt′.fmin = opt.xmin, opt.fmin
    return opt′
end

function trace_state(opt::CMAESOpt, iter, fcount)
    elapsed_time = time() - opt.last_report_time
    # display some information every iteration
    @master @printf("time:%s  iter:%d  elapsed-time:%.2f  ", Time(now()), iter, elapsed_time)
    @master @printf("pmap-time:%.2f  grad-time:%.2f  ls-time:%.2f ls-dec:%2.2e\n",
            opt.pmap_time, opt.grad_time, opt.ls_time, opt.ls_dec)
    @master @printf("fcount:%d  fval:%2.2e  fmin:%2.2e  ", fcount, opt.arfitness[1], opt.fmin)
    @master @printf("norm:%2.2e  axis-ratio:%2.2e  free-mem:%.2fGB\n",
            norm(opt.arx[:, opt.arindex[1]]), maximum(opt.D) / minimum(opt.D), Sys.free_memory() / 1024^3)
    opt.last_report_time = time()
    return nothing
end

function load!(opt::CMAESOpt, resume)
    (resume == false || !isfile(opt.file)) && return
    data = BSON.load(opt.file)
    data[:N] != opt.N && return
    loadvars = [:σ, :cc, :cσ, :c1, :cμ, :dσ, :x̄, :pc, :pσ, :D, :B, :BD, :C]
    resume == :full && append!(loadvars, [:xmin, :fmin, :fmins, :fmeds, :feqls])
    for s in loadvars
        haskey(data, s) && setfield!(opt, s, data[s])
    end
end

function save(opt::CMAESOpt, saveall = false)
    isempty(opt.file) && return
    data = Dict{Symbol, Any}(:N => opt.N)
    savevars = [:σ, :cc, :cσ, :c1, :cμ, :dσ, :x̄, :pc, :pσ, :D, :B, :BD, :C, :xmin, :fmin, :fmins, :fmeds, :feqls]
    if !saveall && Base.summarysize(opt) >= 1024^2 * 20
        savevars = setdiff(savevars, [:D, :B, :BD, :C])
    end
    for s in savevars
        data[s] = copy(getfield(opt, s))
    end
    @master BSON.bson(opt.file, data)
end

function minimize(fg, x0, a...; maxfevals = 0, gcitr = false, maxiter = 0, maxtime::Number=Inf,
                resume = false, cb = [], seed = nothing, autodiff = false, 
                saveall = false, lazydecomp = false, equal_best = Inf, ka...)
    rng = bcast(MersenneTwister(seed))
    f, g = fg isa Tuple ? fg : autodiff ? (fg, fg') : (fg, zero)
    opt = CMAESOpt(f, g, bcast(x0), a...; rng = rng, ka...)
    cb = runall([throttle(x -> save(opt, saveall), 300); cb])
    maxfevals = (maxfevals == 0) ? 1e3 * length(x0)^2 : maxfevals
    maxfevals = maxiter != 0 ? maxiter * opt.λ : maxfevals
    load!(opt, resume)
    iter = length(opt.fmins)
    fcount = iter * opt.λ
    status = 0
    t0 = time()
    t_run = 0.0
    while (fcount < maxfevals) & (t_run <= maxtime)
        iter += 1; fcount += opt.λ
        if opt.λ == 1
            update_mean!(opt)
            trace_state(opt, iter, fcount)
            continue
        end
        update_candidates!(opt)
        update_parameters!(opt, iter, lazydecomp)
        trace_state(opt, iter, fcount)
        gcitr && @everywhere GC.gc(true)
        cb(opt.xmin) == :stop && break
        t_run = time() - t0
        terminate(opt, equal_best) && (status = 1; break)
        # if terminate(opt, equal_best) opt, iter = restart(opt), 0 end
    end
    xmin = inverse(opt.trans, opt.xmin)
    return xmin, opt.fmin, status
end

function maximize(fg, args...; kws...)
    f, g = fg isa Tuple ? fg : (fg, zero)
    fg′ = (x -> -f(x), x -> -g(x))
    xmin, fmin, status = minimize(fg′, args...; kws...)
    return xmin, -fmin, status
end

end
