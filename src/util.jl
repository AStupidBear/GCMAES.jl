export @mpirun, @master, @mpiman

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

function pstreeids(pid = getpid())
    pids = Int[]
    while !isnothing(pid)
        push!(pids, pid)
        pid = getppid(pid)
    end
    pop!(pids)
    return pids[end:-1:1]
end

pstree(pid = getpid()) = join(processname.(pstreeids(pid)), "--")

function inmpi()
    try
        isfile("/usr/bin/yhrun") ?
        occursin(r"mpi|orte|hydra|slurm|srun|salloc", pstree()) :
        occursin(r"mpi|orte|hydra", pstree())
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
        MPI.Barrier(worldcomm())
        res = $(esc(ex))
        MPI.Barrier(worldcomm())
        res
    end
end

macro mpiman(ex)
    !inmpi() && return esc(ex)
    man = gensym()
    quote
        @eval using MPIClusterManagers
        $man = MPIClusterManagers.start_main_loop(MPIClusterManagers.MPI_TRANSPORT_ALL)
        res = $ex
        MPIClusterManagers.stop_main_loop($man)
    end |> esc
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
    wsize = something(wsize, dsize)
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

function partjob(x; dims = -1)
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        rank = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
        wsize = parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"])
        part(x, rank, wsize, dims)
    elseif haskey(ENV, "PBS_ARRAYID")
        rank = parse(Int, ENV["PBS_ARRAYID"])
        part(x, rank, nothing, dims)
    else
        return x
    end
end

part(x, comm = nothing; dims = -1) = x

function limit_julia_procs(n)
    njulia = parse(Int, read(pipeline(`pgrep julia`, `wc -l`), String)) - 1
    if njulia > n
        println("njulia > $n, exiting...")
        return true
    end
    return false
end

function limit_mem_per_cpu(mem)
    mem = @eval let MB = 1024^2, M = MB, G = GB = 1024^3
        $(Meta.parse(mem))
    end
    n = floor(Int, Sys.total_memory() / mem)
    limit_julia_procs(n)
end

function make_virtual_jobarray(n)
    if !haskey(ENV, "SLURM_ARRAY_TASK_ID")
        ENV["SLURM_ARRAY_TASK_ID"] = 0
        ENV["SLURM_ARRAY_TASK_COUNT"] = 1
    end
    taskid = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    taskcount = parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"])
    rank, wsize = myrank(), worldsize()
    q, r = divrem(wsize, n)
    splits = cumsum([i <= r ? q + 1 : q for i in 1:n])
    splits = [0; splits[1:end-1]]
    color = searchsortedlast(splits, rank) - 1
    comm = MPI.Comm_split(MPI.COMM_WORLD, color, rank)
    setglobalcomm!(comm)
    ENV["SLURM_ARRAY_TASK_ID"] = taskid * n + color
    ENV["SLURM_ARRAY_TASK_COUNT"] = taskcount * n
    return nothing
end
