worldcomm(comm = nothing) = something(comm, globalcomm())

selfcomm() = MPI.COMM_SELF

worldsize(comm = nothing) = MPI.Initialized() ? MPI.Comm_size(worldcomm(comm)) : nworkers()

myrank(comm = nothing) = MPI.Initialized() ? MPI.Comm_rank(worldcomm(comm)) : myid() - 1

bcast(x, root = 0, comm = nothing) =  MPI.Initialized() ? MPI.bcast(x, root, worldcomm(comm)) : x

allunique(x, comm = nothing) = sort(unique(allgather(unique(x), comm)))

allequal(x, comm = nothing) = length(unique(allgather(x, comm))) == 1

function allmean(x, comm = nothing)
    if MPI.Initialized()
        x = MPI.Allreduce(x, MPI.SUM, worldcomm(comm))
        x = x / worldsize(comm)
    end
    return x
end

allmean(x::DateTime, comm = nothing) = unix2datetime(allmean(datetime2unix(x), comm))

function allsum(x, comm = nothing)
    if MPI.Initialized()
        x = MPI.Allreduce(x, MPI.SUM, worldcomm(comm))
    end
    return x
end

function allmin(x, comm = nothing)
    if MPI.Initialized()
        x = MPI.Allreduce(x, MPI.SUM, worldcomm(comm))
    end
    return x
end

function allmax(x, comm = nothing)
    if MPI.Initialized()
        x = MPI.Allreduce(x, MPI.MAX, worldcomm(comm))
    end
    return x
end

barrier(comm = nothing) = MPI.Initialized() ? MPI.Barrier(worldcomm(comm)) : nothing

function part(x, comm = nothing; dims = -1)
    !MPI.Initialized() && return x
    rank = myrank(comm)
    wsize = worldsize(comm)
    part(x, rank, wsize, dims)
end

function allgather(x::Union{Number, AbstractArray{<:Number}}, comm = nothing; dims = 1)
    if MPI.Initialized()
        x = isa(x, Number) ? [x] : x
        counts = MPI.Allgather(Cint(length(x)), worldcomm(comm))
        recvbuf = MPI.Allgatherv(vec(x), counts, worldcomm(comm))
        ranges = zip(cumsum([1; counts[1:end - 1]]), cumsum(counts))
        shape = ntuple(i -> i == dims || dims < 1 ? (:) : size(x, i), ndims(x))
        xs = [reshape(view(recvbuf, i:j), shape) for (i, j) in ranges]
        @assert sum(length, xs) == sum(counts)
        return dims > 0 ? cat(xs..., dims = dims) : xs
    else
        return dims > 0 ? x : [x]
    end
end

allgather(x, comm = nothing; dims = 1) = cat(MPI.deserialize.(allgather(MPI.serialize(x), comm; dims = 0))..., dims = dims)

function gather(x::Union{Number, AbstractArray{<:Number}}, root, comm = nothing; dims = 1)
    if MPI.Initialized()
        x = isa(x, Number) ? [x] : x
        counts = MPI.Allgather(Cint(length(x)), worldcomm(comm))
        recvbuf = MPI.Gatherv(vec(x), counts, root, worldcomm(comm))
        if myrank(comm) == root
            ranges = zip(cumsum([1; counts[1:end - 1]]), cumsum(counts))
            shape = ntuple(i -> i == dims || dims < 1 ? (:) : size(x, i), ndims(x))
            xs = [reshape(view(recvbuf, i:j), shape) for (i, j) in ranges]
            @assert sum(length, xs) == sum(counts)
            return dims > 0 ? cat(xs..., dims = dims) : xs
        else
            return nothing
        end
    else
        return dims > 0 ? x : [x]
    end
end

function gather(x, root, comm = nothing; dims = 1)
    xs = gather(MPI.serialize(x), root, comm; dims = 0)
    isnothing(x) && return
    cat(MPI.deserialize.(xs)..., dims = dims)
end

function localsize()
    host = gethostname()
    sendbuf = Vector{UInt8}(host * "\n")
    recvbuf = allgather(sendbuf)
    hosts = split(String(recvbuf))
    count(hosts .== gethostname())
end

function pmap(f, xs, comm = nothing)
    isempty(xs) && return map(f, xs)
    if @isdefined(MPI) && MPI.Initialized()
        allgather(map(f, part(xs, comm)), comm)
    else
        Distributed.pmap(f, xs)
    end
end

function splitcomm(comm, n)
    wsize, rank = worldsize(comm), myrank(comm)
    if wsize > n
        glbl = MPI.Comm_split(comm, rank % n, rank)
        lcl = MPI.Comm_split(comm, rank ÷ n, rank)
    else
        glbl, lcl = comm, MPI.COMM_SELF
    end
    return glbl, lcl
end

function splitcomm!(comm, n)
    glbl, lcl = splitcomm(comm, n)
    GCMAES.set_globalcomm!(glbl)
    GCMAES.set_localcomm!(lcl)
    return glbl, lcl
end

const _localcomm = Ref{MPI.Comm}(MPI.COMM_SELF)
const _globalcomm = Ref{MPI.Comm}(MPI.COMM_WORLD)

set_localcomm!(comm) = _localcomm[] = comm
set_globalcomm!(comm) = _globalcomm[] = comm

localcomm() = _localcomm[]
globalcomm() = _globalcomm[]
