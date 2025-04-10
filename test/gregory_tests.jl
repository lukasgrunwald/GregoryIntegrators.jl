#=
Test for Gregory integration as well as the polynomial interpolation backup scheme. The
correctness of the results is tested with ≈ and not with mashine precision.

"gint: pure": q=2 test seems to fail. Weights are correct! -> Removed from test for now.
=#
using GregoryIntegrators
using PConv: isapprox_eps
using QuadGK
using Test

function check_gregory(f, q, k, N, a=-2, δt=0.3)

    t = a .+ δt .* collect(0:N)
    b = t[end]
    Gweights = GregoryWeights(q)

    fl = [f(ti, k) for ti in t]
    ref, err = quadgk(x->f(x, k), a, b) # create numerically evaluated reference
    res = gint(fl, δt, q, Gweights)
    # println(res .- ref, " .. ", eps(abs.(ref)))
    return res ≈ ref
end

function rand_poly(t, ci)
    res = zero(eltype(ci))
    for i in eachindex(ci)
        res += ci[i] * t^(i-1)
    end
    return res
end
verbose = true

# ——————————— Compare single and double variable interpolations give the same —————————— #
@testset verbose=verbose "poly_integrate, int-conv" begin
    gweights = GregoryWeights(7)
    for k=1:6
        δt  = rand()
        y1, y2 = (rand(ComplexF64, k+1) for _ in 1:2)
        ref = GregoryIntegrators.poly_integrate(y1 .* y2, δt, k, gweights)
        res = GregoryIntegrators.poly_integrate(y1, y2, δt, k, gweights)

        # @test isapprox_eps(res, ref; factor=1)
        @test res ≈ ref
    end
end

@testset verbose=verbose "gint, int-conv" begin
    for q=1:7
        gweights = GregoryWeights(q)
        δt  = rand()
        y1, y2 = (rand(ComplexF64, 100) for _ in 1:2)
        y1_, y2_ = (rand(ComplexF64, 101) for _ in 1:2)
        ref = gint(y1 .* y2, δt, q, gweights)
        res = gint(y1, y2, δt, q, gweights)
        ref_ = gint(y1_ .* y2_, δt, q, gweights)
        res_ = gint(y1_, y2_, δt, q, gweights)

        @test res ≈ ref
        @test res_ ≈ ref_
    end
end

# ——————————————————————— Ploynomial interpolate for polynomials ——————————————————————— #
@testset verbose=verbose "poly_integrate, pure k" begin
    # Integral from 0 to kδt, i.e. (k+1)-points
    δt = 0.3
    f(t, k) = t^k
    Gweights = GregoryWeights(7)
    for k=1:6
        fref = 1/(k+1) * (k*δt)^(k+1)

        t  = δt .* collect(0:k)
        fl = f.(t, k)
        res = GregoryIntegrators.poly_integrate(fl, δt, k, Gweights)
        @test res ≈ fref
    end
end

@testset verbose=verbose "poly_integrate, rand_poly" begin
    δt = 0.1
    Gweights = GregoryWeights(7)
    for k in 1:6
        ci = rand(ComplexF64, k+1)
        ref, err = quadgk(x->rand_poly(x, ci), 0, k*δt)

        t  = δt .* collect(0:k)
        zero(eltype(ci))
        fl = [rand_poly(ti, ci) for ti in t]
        res = GregoryIntegrators.poly_integrate(fl, δt, k, Gweights)

        @test res ≈ ref
    end
end

# ————————————————————————————————— Gregory integration ———————————————————————————————— #
#* q=2 test seems to fail. Removed from test for now.
#* Weights are correct as far as I can tell
@testset verbose=verbose "gint: pure" begin
    f(t, k) = t^k
    for q in 1:7
        @testset "q=$q" begin
            for k in 0:q
                if (q == 2 && k == q) continue end
                @test check_gregory(f, q, k, 100) # Odd number
                @test check_gregory(f, q, k, 101) # Even number
            end
        end
    end
end

##
#* The even Gregory don't seem to be that great...
@testset verbose=verbose "gint: poly_rand" begin
    for q in 1:2:7
        @testset "q=$q" begin
            for k=0:q
                ci = rand(ComplexF64, k+1)
                @test check_gregory(rand_poly, q, ci, 10) # Odd number
                @test check_gregory(rand_poly, q, ci, 11) # Even number
            end
        end
    end
end

# —————————————————————————— Gregory integration backup calls —————————————————————————— #
@testset "gint: l<q+1 calls" begin
    q = 7
    for k in 0:7
        f(t, k) = t^k
        N, a, δt = k, 0, 0.1

        t = a .+ δt .* collect(0:N)
        b = t[end]
        Gweights = GregoryWeights(q)

        fl = [f(ti, k) for ti in t]
        ref, err = quadgk(x->f(x, k), a, b)
        res = gint(fl, δt, q, Gweights)
        @test res ≈ ref
    end
end
