#=
Benchmarks and comparison for different dot-product sums.
Compared are simd, avx and conventional sum implementations.
=#
using BenchmarkTools
using GregoryIntegrators

N = 249

println("Benchmark: sum_avx-F64")
a = rand(Float64, N); b = rand(Float64, N);
print("Std.:"); @btime sum($a)
print("SIMD:"); @btime GregoryIntegrators.sum_simd($a)
print("AVX :"); @btime GregoryIntegrators.sum_avx($a)

println("\nBenchmark: sum_avx-CF64")
a = rand(ComplexF64, N); b = rand(ComplexF64, N);
print("Std.:"); @btime sum($a)
print("SIMD:"); @btime GregoryIntegrators.sum_simd($a)
print("AVX :"); @btime GregoryIntegrators.sum_avx($a)

println("\nBenchmark: dot_avx-F64")
a = rand(Float64, N); b = rand(Float64, N);
print("Std.:"); @btime sum($a .* $b)
print("SIMD:"); @btime GregoryIntegrators.dot_simd($a, $b)
print("AVX :"); @btime GregoryIntegrators.dot_avx($a, $b)

# Complex
println("\nBenchmark: dot_avx-CF64")
a = rand(ComplexF64, N); b = rand(ComplexF64, N);
print("Std.:"); @btime sum($a .* $b)
print("SIMD:"); @btime GregoryIntegrators.dot_simd($a, $b)
print("AVX :"); @btime GregoryIntegrators.dot_avx($a, $b);

# Output
# Benchmark: sum_avx-F64
# Std.:  13.842 ns (0 allocations: 0 bytes)
# SIMD:  14.104 ns (0 allocations: 0 bytes)
# AVX :  9.840 ns (0 allocations: 0 bytes)

# Benchmark: sum_avx-CF64
# Std.:  53.694 ns (0 allocations: 0 bytes)
# SIMD:  52.138 ns (0 allocations: 0 bytes)
# AVX :  25.456 ns (0 allocations: 0 bytes)

# Benchmark: dot_avx-F64
# Std.:  226.909 ns (1 allocation: 2.06 KiB)
# SIMD:  27.372 ns (0 allocations: 0 bytes)
# AVX :  16.843 ns (0 allocations: 0 bytes)

# Benchmark: dot_avx-CF64
# Std.:  337.029 ns (1 allocation: 4.06 KiB)
# SIMD:  107.960 ns (0 allocations: 0 bytes)
# AVX :  48.055 ns (0 allocations: 0 bytes)
