using SafeTestsets
using Test

@safetestset "avx_dot_sum" begin include("avx_dot_sum_tests.jl") end
@safetestset "weights_test" begin include("weights_test.jl") end
@safetestset "interpolations" begin include("interpolation_tests.jl") end
@safetestset "Gregory Integration" begin include("gregory_tests.jl") end
