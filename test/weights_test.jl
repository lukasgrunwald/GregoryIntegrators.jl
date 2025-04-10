#=
Test for integrator weights. All the weight functions should sum to one!
=#
using GregoryIntegrators
using Test

@testset "AB-explicit_adams_weights" begin
    for i in 1:5
        bweigths = explicit_adams_weights(i)
        @test sum(bweigths) ≈ 1
    end
end

@testset "AM-implicit_adams_weights" begin
    for i in 1:5
        mweigths = implicit_adams_weights(i)
        @test sum(mweigths) ≈ 1
    end
end

@testset "explicit_stoermer_weights" begin
    for i in 1:5
        sweigths = explicit_stoermer_weights(i, false)
        @test sum(sweigths) ≈ 1
    end
end

@testset "implicit_stoermer_weights" begin
    for i in 1:5
        sweigths = implicit_stoermer_weights(i, false)
        @test sum(sweigths) ≈ 1
    end
end
