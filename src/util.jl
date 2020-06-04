export @mpirun, @master

minibatch(x, b) = [x[i:min(end, i + b - 1)] for i in 1:b:max(1, length(x) - b + 1)]

sample(lo, hi) = lo .+ rand(size(lo)) .* (hi .- lo)

function ptp(x)
    xmin, xmax = extrema(x)
    xmax - xmin
end

runall(f) = f
runall(fs::AbstractVector) = (xs...) -> last([f(xs...) for f in fs])

function throttle(f, timeout; leading = true)
    lasttime = time()
    leading && (lasttime -= timeout)
    function throttled(args...; kwargs...)
        result = nothing
        if time() >= lasttime + timeout
            result = f(args...; kwargs...)
            lasttime = time()
        end
        return result
    end
end

function pmap(f, xs)
    if @isdefined(MPI) && MPI.Initialized()
        allgather(map(f, part(xs)))
    else
        Distributed.pmap(f, xs)
    end
end

macro master(ex)
    :(if myrank() == 0
        $(esc(ex))
    end)
end

worldsize() = @isdefined(MPI) && MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : nworkers()

function processname(pid)
    @static if Sys.iswindows()
        split(read(`wmic process where processid=$pid get executablepath`, String))[end]
    else
        strip(read(`ps -p $pid -o comm=`, String))
    end
end

function pstree(pid = getpid())
    pids = Int[]
    while !isnothing(pid)
        push!(pids, pid)
        pid = getppid(pid)
    end
    pop!(pids)
    return pids
end

function inmpi()
    try
        @static if Sys.iswindows()
            occursin("mpi", join(processname.(pstree())))
        else
            ps = read(`pstree -s $(getpid())`, String)
            occursin("mpi", ps) || occursin("slurm", ps)
        end
    catch
        false
    end
end

macro mpirun(ex)
    !inmpi() && return esc(ex)
    quote
        @eval using MPI
        if "Elemental" in keys(Pkg.installed())
            @eval using Elemental
        end
        !MPI.Initialized() && MPI.Init()
        MPI.Barrier(MPI.COMM_WORLD)
        $(esc(ex))
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

deigen(x) = eigen(Symmetric(x, :U))

myrank() = myid() - 1

bcast(x) = x