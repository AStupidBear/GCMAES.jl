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
        barrier()
        result = nothing
        if time() >= lasttime + timeout
            result = f(args...; kwargs...)
            lasttime = time()
        end
        return result
    end
end

function getppid(pid = getpid())
    try
        @static if Sys.iswindows()
            cmd = `wmic process where processid=$pid get parentprocessid`
            str = read(pipeline(cmd, stderr = devnull), String)
            parse(Int, match(r"\d+", str).match)
        else
            cmd = `ps -o ppid= -p $pid`
            str = read(pipeline(cmd, stderr = devnull), String)
            parse(Int, strip(str))
        end
    catch
        nothing
    end
end

function getcpids(pid = getpid())
    try
        cmd = `ps -o pid= --ppid $pid`
        str = read(pipeline(cmd, stderr = devnull), String)
        parse.(Int, split(str, '\n', keepempty = false))
    catch
        nothing
    end
end

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
            occursin(r"mpi|slurm|srun|salloc", ps)
        end
    catch
        false
    end
end

deigen(x) = eigen(Symmetric(x, :U))

worldsize() = nworkers()

myrank() = myid() - 1

bcast(x, root = 0) = x

allequal(x) = length(unique(x)) == 1

barrier() = nothing

macro barrier(ex) :(barrier(); res = $(esc(ex)); barrier(); res) end

macro mpirun(ex)
    !inmpi() && return esc(ex)
    quote
        @eval using MPI
        !MPI.Initialized() && MPI.Init()
        MPI.Barrier(MPI.COMM_WORLD)
        res = $(esc(ex))
        MPI.Barrier(MPI.COMM_WORLD)
        res
    end
end

macro master(ex)
    quote
        @barrier if myrank() == 0
            res = $(esc(ex))
            bcast(res, 0)
        else
            bcast(nothing, 0)
        end
    end
end

function part(x::AbstractArray{T, N}, rank, wsize, dims) where {T, N}
    dims = clamp(dims > 0 ? dims : N + dims + 1, 1, N)
    dsize = size(x, dims)
    if dsize >= wsize
        q, r = divrem(dsize, wsize)
        splits = cumsum([i <= r ? q + 1 : q for i in 1:wsize])
        pushfirst!(splits, 0)
        is = (splits[rank + 1] + 1):splits[rank + 2]
        view(x, ntuple(x -> x == dims ? is : (:), N)...)
    else
        @debug @warn "rank=$rank: dsize=$dsize < wsize=$wsize"
        is = (rank + 1):min(rank + 1, dsize)
        view(x, ntuple(x -> x == dims ? is : (:), N)...)
    end
end

function part(x, comm = nothing; dims = -1)
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        rank = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
        wsize = parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"])
        x = part(x, rank, wsize, dims)
    end
    return x
end
