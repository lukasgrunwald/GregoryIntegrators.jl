#=
Tests for avx and simd enabled sum
=#
using GregoryIntegrators
using PConv: isapprox_eps
using Test

# —————————————————————————————————————— Sum tests ————————————————————————————————————— #
@testset "sum_avx-F64" begin
    a = rand(Float64, 249)
    ref = sum(a)
    simd = GregoryIntegrators.sum_simd(a)
    avx = GregoryIntegrators.sum_avx(a)

    @test isapprox_eps(ref, simd; factor = 10)
    @test isapprox_eps(ref, avx; factor = 10)
    @test isapprox_eps(simd, avx; factor = 10)
end

@testset "sum_avx-CF64" begin
    a = rand(ComplexF64, 249)
    ref = sum(a)
    simd = GregoryIntegrators.sum_simd(a)
    avx = GregoryIntegrators.sum_avx(a)

    @test isapprox_eps(ref, simd; factor = 10)
    @test isapprox_eps(ref, avx; factor = 10)
    @test isapprox_eps(simd, avx; factor = 10)
end

# —————————————————————————————————————— Dot tests ————————————————————————————————————— #
@testset "dot_avx-F64" begin
    a = rand(Float64, 249)
    b = rand(Float64, 249)

    ref = sum(a .* b)
    simd = GregoryIntegrators.dot_simd(a, b)
    avx = GregoryIntegrators.dot_avx(a, b)

    @test isapprox_eps(ref, simd; factor = 10)
    @test isapprox_eps(ref, avx; factor = 10)
    @test isapprox_eps(simd, avx; factor = 10)
end

@testset "dot_avx-CF64" begin
    a = rand(ComplexF64, 249)
    b = rand(ComplexF64, 249)

    ref = sum(a .* b)
    simd = GregoryIntegrators.dot_simd(a, b)
    avx = GregoryIntegrators.dot_avx(a, b)

    @test isapprox_eps(ref, simd; factor = 50)
    @test isapprox_eps(ref, avx; factor = 50)
    @test isapprox_eps(simd, avx; factor = 50)
end
