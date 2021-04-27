using LinearAlgebra

function deigen(x)
    copyto!(x, Symmetric(x, :U))
    xd = Elemental.DistMatrix(eltype(x))
    Elemental.zeros!(xd, size(x)...)
    copy_local_symm!(xd, x)
    @debug @assert x ≈ Array(xd)
    w, Z = Elemental.eigHermitian(Elemental.UPPER, xd)
    vals = vec(Array(w))
    vecs = fill!(similar(x), 0)
    copy_local_symm!(vecs, Z)
    vecs = permutedims(vecs)
    @debug @assert vecs ≈ Array(Z)
    return vals, vecs
end

function copy_local_symm!(xd::Elemental.DistMatrix, x)
    is = part(1:size(x, 1))
    for i in is, j in 1:size(x, 2)
        Elemental.queueUpdate(xd, i, j, x[i, j])
    end
    Elemental.processQueues(xd)
    return xd
end

function copy_local_symm!(x, xd::Elemental.DistMatrix)
    is = part(1:size(x, 1))
    xl = view(x, :, is)
    for i in is, j in 1:size(x, 2)
        Elemental.queuePull(xd, i, j)
    end
    Elemental.processPullQueue(xd, unsafe_wrap(Array, pointer(xl), size(xl)))
    counts = MPI.Allgather(Cint(length(xl)), worldcomm())
    MPI.Allgatherv!(vec(x), counts, worldcomm())
    return x
end
