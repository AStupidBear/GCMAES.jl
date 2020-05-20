function part(x::AbstractArray{T, N}, dim = -2) where {T, N}
    !MPI.Initialized() && return x
    dim = clamp(dim > 0 ? dim : N + dim + 1, 1, N)
    dsize, rank, wsize = size(x, dim), myrank(), worldsize()
    @assert wsize <= dsize
    q, r = divrem(dsize, wsize)
    splits = cumsum([i <= r ? q + 1 : q for i in 1:wsize])
    pushfirst!(splits, 0)
    is = (splits[rank + 1] + 1):splits[rank + 2]
    view(x, ntuple(x -> x == dim ? is : (:), N)...)
end

function allgather(x, dim = 1)
    if MPI.Initialized()
        x = isa(x, Number) ? [x] : x
        counts = MPI.Allgather(Cint(length(x)), MPI.COMM_WORLD)
        recvbuf = MPI.Allgatherv(vec(x), counts, MPI.COMM_WORLD)
        ranges = zip(cumsum([1; counts[1:end - 1]]), cumsum(counts))
        shape = ntuple(i -> i == dim ? (:) : size(x, i), ndims(x))
        xs = [reshape(view(recvbuf, i:j), shape) for (i, j) in ranges]
        @assert sum(length, xs) == sum(counts)
        return cat(xs..., dims = dim)
    else
        return x
    end
end

myrank() = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0

bcast(x, root = 0) = MPI.bcast(x, root, MPI.COMM_WORLD)