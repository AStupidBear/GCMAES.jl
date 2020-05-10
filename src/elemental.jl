function deigen(x)
    copyto!(x, Symmetric(x, :U))
    xd = Elemental.DistMatrix(eltype(x))
    Elemental.zeros!(xd, size(x)...)
    copy_local_symm!(xd, x)
    @debug @assert x ≈ Array(xd)
    w, Z = Elemental.eigHermitian(Elemental.UPPER, xd)
    vals = vec(Array(w))
    vecs = copy_local_symm!(fill!(similar(x), 0), Z)
    copyto!(vecs, Symmetric(vecs, :L))
    @debug @assert vecs ≈ Array(Z)
    return vals, vecs
end

function copy_local_symm!(xd::Elemental.DistMatrix, x)
    is = local_indices(size(x, 1))
    for i in is, j in 1:size(x, 2)
        Elemental.queueUpdate(xd, i, j, x[i, j])
    end
    Elemental.processQueues(xd)
    return xd
end

function copy_local_symm!(x, xd::Elemental.DistMatrix)
    is = local_indices(size(x, 1))
    xl = view(x, :, is)
    for i in is, j in 1:size(x, 2)
        Elemental.queuePull(xd, i, j)
    end
    Elemental.processPullQueue(xd, xl)
    counts = MPI.Allgather(Cint(length(xl)), MPI.COMM_WORLD)
    MPI.Allgatherv!(vec(x), counts, MPI.COMM_WORLD)
    return x
end

function local_indices(dsize)
    rank, wsize = myrank(), worldsize()
    @assert wsize <= dsize
    q, r = divrem(dsize, wsize)
    splits = cumsum([i <= r ? q + 1 : q for i in 1:wsize])
    pushfirst!(splits, 0)
    is = (splits[rank + 1] + 1):splits[rank + 2]
end