__precompile__(true)

module GCMAES

using JLD, FileIO

include("util.jl")
include("constraint.jl")
include("update.jl")

mutable struct CMAESOpt{F, C, O}
    # fixed hyper-parameters
    f::F
    N::Int
    σ0::Float64
    lo::Vector{Float64}
    hi::Vector{Float64}
    constraint::C
    # strategy parameter setting: selection
    λ::Int
    μ::Int
    w::Vector{Float64}
    μeff::Float64
    # strategy parameter setting: adaptation
    σ::Float64
    cc::Float64
    cσ::Float64
    c1::Float64
    cμ::Float64
    dσ::Float64
    # dynamic (internal) strategy parameters and constants
    x̄::Vector{Float64}
    pc::Vector{Float64}
    pσ::Vector{Float64}
    D::Vector{Float64}
    B::Matrix{Float64}
    BD::Matrix{Float64}
    C::Matrix{Float64}
    χₙ::Float64
    arx::Matrix{Float64}
    ary::Matrix{Float64}
    arz::Matrix{Float64}
    arfitness::Vector{Float64}
    arpenalty::Vector{Float64}
    arindex::Vector{Int}
    # recordings
    xmin::Vector{Float64}
    fmin::Float64
    fmins::Vector{Float64} # history of best fitness
    fmeds::Vector{Float64} # history of median fitness
    feqls::Vector{Float64} # history of equal fitness
    # gradient parameters
    ν::Int # number of gradients to save
    argrad::Matrix{Float64}
    arx′::Matrix{Float64}
    gradopt::O
    gradopts::Vector{O}
    # report
    last_report_time::Float64
    file::String
    equal_best::Int
end

function CMAESOpt(f, x0, σ0, lo, hi; λ = 0, equal_best = 10^10, constraint = NoConstraint(), 
                            grad = true, ν = 0, lr = 1e-3, gclip = 0.5, gradopt = :Sgd, o...)
    if grad == false
        ν = 0
        g = f
        f = x -> (y = g(x); (y, zeros(x)))
    end
    N, x̄, xmin, (fmin,), σ = length(x0), x0, x0, f(x0), σ0
    # strategy parameter setting: selection
    λ = λ == 0 ? round(Int, 4 + 3log(N)) : λ
    μ = λ ÷ 2                    # number of parents/points for recombination
    w = log(μ + 1/2) .- log.(1:μ) # μXone array for weighted recombination
    normalize!(w, 1)             # normalize recombination w array
    μeff = 1 / sum(abs2, w)     # variance-effectiveness of sum w_i x_i
    # strategy parameter setting: adaptation
    cc = 4 / (N + 4) # time constant for cumulation for C
    cσ = (μeff + 2) / (N + μeff + 3)  # t-const for cumulation for σ control
    c1 = 2 / ((N + 1.3)^2 + μeff)    # learning rate for rank-one update of C
    cμ = min(1 - c1, 2(μeff - 2 + 1 / μeff) / ((N + 2)^2 + μeff) )   # and for rank-μ update
    dσ = 1 + 2 * max(0, sqrt((μeff - 1) / (N + 1)) - 1) + cσ # damping for σ, usually close to 1
    # initialize dynamic (internal) strategy parameters and constants
    pc = zeros(N); pσ = zeros(N)   # evolution paths for C and σ
    D = fill(σ, N)                # diagonal matrix D defines the scaling
    B = eye(N, N)                 # B defines the coordinate system
    BD = B .* reshape(D, 1, N)    # B*D for speed up only
    C = diagm(D.^2)                    # covariance matrix == BD*(BD)'
    χₙ = sqrt(N) * (1 -1 / 4N + 1 / 21N^2)  # expectation of  ||N(0,I)|| == norm(randn(N,1))
    # init a few things
    arx, ary, arz = zeros(N, λ), zeros(N, λ), zeros(N, λ)
    arfitness = zeros(λ); arpenalty = zeros(λ); arindex = zeros(λ)
    @printf("%i-%i CMA-ES\n", λ, μ)
    # gradient
    argrad = zeros(N, λ)
    arx′ = zeros(N, ν)
    gradopt = gradopt == :Sgd ? Sgd(lr = lr, gclip = gclip) :
              gradopt == :Adam ? Adam(lr = lr, gclip = gclip) :
              gradopt == :Rmsprop ? Rmsprop(lr = lr, gclip = gclip) : 
              error("unknown optimizer")
    gradopts = [deepcopy(gradopt) for k in 1:ν]
    F, C′, O = typeof(f), typeof(constraint), typeof(gradopt)
    return CMAESOpt{F, C′, O}(
            f, N, σ0, lo, hi, constraint,
            λ, μ, w, μeff,
            σ, cc, cσ, c1, cμ, dσ,
            x̄, pc, pσ, D, B, BD, C, χₙ,
            arx, ary, arz, arfitness, arpenalty, arindex, 
            xmin, fmin, Float64[], Float64[], Float64[],
            ν, argrad, arx′, gradopt, gradopts,
            time(), "CMAES.jld", equal_best)
end

function update_candidates!(opt::CMAESOpt, pool)
    # generate and evaluate λ offspring
    randn!(opt.arz) # resample
    opt.ary = opt.BD * opt.arz
    opt.arx .= opt.x̄ .+ opt.σ .* opt.ary
    length(opt.fmins) > 0 && (opt.arx[:, 1:opt.ν] = opt.arx′)
    arx_cols = [opt.arx[:, k] for k in 1:opt.λ]
    results = pmap(WorkerPool(pool), opt.f, arx_cols)
    for k in 1:opt.λ
       opt.arfitness[k] = results[k][1]
       opt.argrad[:, k] = results[k][2]
    end
    opt.arpenalty .=  getpenalty.(opt.constraint, arx_cols)
    opt.arfitness .+= opt.arpenalty
    # sort by fitness and compute weighted mean into x̄
    sortperm!(opt.arindex, opt.arfitness)
    opt.arfitness = opt.arfitness[opt.arindex] # minimization
    # SGD with grad clipping for the best μ candidates
    indν = opt.arindex[1:opt.ν]
    opt.gradopts = [1 <= i <= opt.ν ? opt.gradopts[i] : deepcopy(opt.gradopt) for i in indν]
    for k in 1:opt.ν
        w = view(opt.arx′, :, k)
        g = view(opt.argrad, :, indν[k])
        p = opt.gradopts[k]
        update!(w, g, p)
    end
    # store the best candidate
    if opt.arfitness[1] < opt.fmin
        opt.xmin, opt.fmin = opt.arx[:, opt.arindex[1]], opt.arfitness[1]
    end
    push!(opt.fmins, opt.arfitness[1])
    push!(opt.fmeds, opt.arfitness[ceil(Int, opt.λ / 2)])
    feql = opt.λ > 1 ? opt.arfitness[1] == opt.arfitness[ceil(Int, 1.1 + opt.λ / 4)] : 0
    push!(opt.feqls, feql)
end

function update_parameters!(opt::CMAESOpt, iter)
    indμ = opt.arindex[1:opt.μ]
    # calculate new x̄, this is selection and recombination
    x̄old = copy(opt.x̄) # for speed up of Eq. (2) and (3)
    opt.x̄ = transform(opt.constraint, opt.arx[:, indμ] * opt.w)
    z̄ = opt.arz[:, indμ] * opt.w # ==D^-1*B'*(x̄-x̄old)/σ
    # cumulation: update evolution paths
    BLAS.gemv!('N', sqrt(opt.cσ * (2 - opt.cσ) * opt.μeff), opt.B, z̄, 1 - opt.cσ, opt.pσ)
    #  i.e. pσ = (1 - cσ) * pσ + sqrt(cσ * (2 - cσ) * μeff) * (B * z̄) # Eq. (4)
    hsig = norm(opt.pσ) / sqrt(1 - (1 - opt.cσ)^2iter) / opt.χₙ < 1.4 + 2 / (opt.N + 1)
    BLAS.scale!(opt.pc, 1 - opt.cc)
    BLAS.axpy!(hsig * sqrt(opt.cc * (2 - opt.cc) * opt.μeff) / opt.σ, opt.x̄ - x̄old, opt.pc)
    # i.e. pc = (1 - cc) * pc + (hsig * sqrt(cc * (2 - cc) * μeff) / σ) * (x̄ - x̄old)
    # adapt covariance matrix C
    scale!(opt.C, (1 - opt.c1 - opt.cμ + (1 - hsig) * opt.c1 * opt.cc * (2 - opt.cc))) # discard old C
    BLAS.syr!('U', opt.c1, opt.pc, opt.C) # rank 1 update C += c1 * pc * pc'
    artmp = opt.ary[:, indμ]  # μ difference vectors
    artmp = (artmp .* reshape(opt.w, 1, opt.μ)) * artmp.'
    BLAS.axpy!(opt.cμ, artmp, opt.C)
    # adapt step size σ
    opt.σ *= exp((norm(opt.pσ) / opt.χₙ - 1) * opt.cσ / opt.dσ)  #Eq. (5)
    # update B and D from C
    # if counteval - eigeneval > λ / (c1 + cμ) / N / 10  # to achieve O(N^2)
    if mod(iter, 1 / (opt.c1 + opt.cμ) / opt.N / 10) < 1
        (opt.D, opt.B) = eig(Symmetric(opt.C, :U)) # eigen decomposition, B==normalized eigenvectors
        opt.D .= sqrt.(opt.D)                   # D contains standard deviations now
        opt.BD .= opt.B .* reshape(opt.D, 1, opt.N)     # O(n^2)
    end
end

function terminate(opt::CMAESOpt)
    histiter = 10 + round(Int, 30opt.N / opt.λ)
    stagiter = round(Int, 0.2 * length(opt.fmins) + 120 + 30opt.N / opt.λ)
    stagiter = min(stagiter, 20000)
    hist = opt.fmins[max(1, end - histiter):end]
    condition = Dict{String, Bool}()
    # FlatFitness: warn if 70% candidates' fitnesses are identical
    if opt.arfitness[1] == opt.arfitness[ceil(Int, 0.7opt.λ)]
        opt.σ *= exp(0.2 + opt.cσ / opt.dσ)
        println("warning: flat fitness, consider reformulating the objective")
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
    length(opt.fmins) > opt.equal_best && 
    ptp(accumulate(min, opt.fmins)[end - opt.equal_best:end]) < 1e-12 &&  
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
        v && print_with_color(:red, "Termination Condition Satisfied: ", k, '\n')
        termination = termination | v
    end
    return termination
end

function restart(opt::CMAESOpt)
    @printf("restarting...\n")
    optnew = CMAESOpt(opt.f, sample(opt.lo, opt.hi), opt.σ0, opt.lo, opt.hi; :λ => 2opt.λ)
    optnew.xmin, optnew.fmin = opt.xmin, opt.fmin
    return optnew
end

function trace_state(opt::CMAESOpt, iter, fcount)
    elapsed_time = time() - opt.last_report_time
    save(opt)
    # display some information every iteration
    @printf("time: %s iter: %d  elapsed-time: %.2f fcount: %d  fval: %2.2e  fmin: %2.2e  penalty: %2.2e  axis-ratio: %2.2e free-mem: %.2fGB\n",
            now(), iter, elapsed_time, fcount, opt.arfitness[1], opt.fmin, median(opt.arpenalty), maximum(opt.D) / minimum(opt.D), Sys.free_memory() / 1024^3)
    opt.last_report_time = time()
    return nothing
end

function load!(opt::CMAESOpt, resume)
    (resume == "false" || !isfile(opt.file)) && return
    d = load(File(format"JLD", opt.file))
    get(d, "N", opt.N) != opt.N && return
    loadvars = ["σ", "cc", "cσ", "c1", "cμ", "dσ", "x̄", "pc", "pσ", "D", "B", "BD", "C", "χₙ"]
    resume == "full" && append!(loadvars, ["xmin", "fmin", "fmins", "fmeds", "feqls", "gradopt", "gradopts"])
    for s in loadvars
        setfield!(opt, Symbol(s), get(d, s, getfield(opt, Symbol(s))))
    end
end

function save(opt::CMAESOpt)
    JLD.jldopen(opt.file, "w") do fid
        for s in fieldnames(opt)
            s != :f && write(fid, string(s), getfield(opt, s))            
        end
    end
end

function minimize(f::Function, x0, σ0, lo, hi; pool = workers(), maxfevals = 0, 
            gcitr = true, maxiter = 0, resume = "false", cb = (xs...) -> (), o...)
    cb = runall(cb)
    opt = CMAESOpt(f, x0, σ0, lo, hi; o...)
    maxfevals = (maxfevals == 0) ? 1e3 * length(x0)^2 : maxfevals
    maxfevals = maxiter != 0 ? maxiter * opt.λ : maxfevals
    load!(opt, resume)
    fcount = iter = 0; status = 0
    while fcount < maxfevals
        gcitr && @everywhere gc(true)
        iter += 1; fcount += opt.λ
        update_candidates!(opt, pool)
        update_parameters!(opt, iter)
        trace_state(opt, iter, fcount)
        terminate(opt) && (status = 1; break)
        cb(opt.xmin) == :stop && break
        # if terminate(opt) opt, iter = restart(opt), 0 end
    end
    return opt.xmin, opt.fmin, status
end

function maximize(f, args...; kwargs...)
    xmin, fmin, status = minimize(x -> .-f(x), args...; kwargs...)
    return xmin, -fmin, status
end

end