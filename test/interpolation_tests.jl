#=
Test for polynomial interpolation for uniformly spaced data. Should reproduce the original
data points, as well as a polynomial of the given order "almost" exactly
=#
using GregoryIntegrators
using Test

# Check that interpolation reproduces the points
@testset "Reproduce points" begin
    f(t) = sin(t) * exp(t)
    k, Δt = 9, 0.1
    t  = Δt .* collect(0:k)
    fl = f.(t)
    Pᴷ = GregoryIntegrators.P⁽ᴷ⁾_weights(k)
    @test all(fl ≈ [GregoryIntegrators.Pᴷt(tval, Pᴷ, fl, Δt) for tval∈t])
end

@testset "Reproduce Polynomial" begin
    Δt = 0.1
    f(t, k) = t^k
    for k in 1:6
        tj = Δt .* collect(0:k) # k+1 points
        fl = f.(tj, k)
        Pᴷ = GregoryIntegrators.P⁽ᴷ⁾_weights(k)

        t_ref = 0:0.01:5 |> collect
        f_ref = f.(t_ref, k)
        fk = [GregoryIntegrators.Pᴷt(t, Pᴷ, fl, Δt) for t∈t_ref]

        @test f_ref ≈ fk
    end
end
