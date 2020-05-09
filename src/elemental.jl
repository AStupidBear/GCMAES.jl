function deigen(x)
    xd = Elemental.DistMatrix(eltype(x))
    Elemental.copy!(xd, x)
    w, Z = Elemental.eigHermitian(Elemental.UPPER, xd)
    vec(Array(w)), Array(Z)
end