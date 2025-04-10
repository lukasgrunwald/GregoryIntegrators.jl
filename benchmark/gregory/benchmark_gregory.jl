#=
Benchmark and comparison for different Gregory integral implementations. gint_old implements
the Gregory scheme with a direct appraoch, while the exported gint uses avx instructions!
=#

using BenchmarkTools
using GregoryIntegrators

function GregoryWeightsOld()

    weights = zeros(Float64, 7, 7)

    # O(h^2)
    weights[1, 1] = -1/2

    # O(h^3)
    weights[1, 2] = -7/12
    weights[2, 2] = +1/12

    # O(h^4)
    weights[1, 3] = -5/8
    weights[2, 3] = +1/6
    weights[3, 3] = -1/24

    # O(h^5)
    weights[1, 4] = -469/720
    weights[2, 4] = +59/240
    weights[3, 4] = -29/240
    weights[4, 4] = +19/720

    # O(h^6)
    weights[1, 5] = -193/288
    weights[2, 5] = +77/240
    weights[3, 5] = -7/30
    weights[4, 5] = +73/720
    weights[5, 5] = -3/160

    # O(h^7)
    weights[1, 6] = -41393/60480
    weights[2, 6] = +23719/60480
    weights[3, 6] = -11371/30240
    weights[4, 6] = +7381/30240
    weights[5, 6] = -5449/60480
    weights[6, 6] = +863/60480

    # O(h^8)
    weights[1, 7] = -12023/17280
    weights[2, 7] = +6961/15120
    weights[3, 7] = -66109/120960
    weights[4, 7] = +33/70
    weights[5, 7] = -31523/120960
    weights[6, 7] = +1247/15120
    weights[7, 7] = -275/24192

    return weights
end

function gint_old(y1, y2, δt, q, weights)::ComplexF64

    l = length(y1)
    (l == 1) && (return 0)
    (l < q+1) && (q = l-1)

    s::ComplexF64 = 0.0
    @inbounds @simd for i=1:l
        s += δt * y1[i] * y2[i]
    end

    # Boundary corrections
    @inbounds @simd for i=1:q
        s += δt * weights[i, q] * (y1[i] * y2[i] + y1[l-(i-1)] * y2[l-(i-1)])
    end
    return s
end

q = 7
N = 1001
gweights = GregoryWeights(q)
gweights_old = GregoryWeightsOld()

fl1, fl2  = (rand(ComplexF64, N) for _ in 1:2)
δt = 0.01

@btime gint($fl1, $fl2, $δt, $q, $gweights)
@btime gint_old($fl1, $fl2, $δt, $q, $gweights_old)
@btime $δt * GregoryIntegrators.dot_avx($fl1, $fl2)

# N = 1001 Float64 (q=1)
# 36.409 ns (0 allocations: 0 bytes)
# 73.614 ns (0 allocations: 0 bytes)
# 30.814 ns (0 allocations: 0 bytes)

# N = 1000 ComplexF64 (q=1)
# 208.964 ns (0 allocations: 0 bytes)
# 428.588 ns (0 allocations: 0 bytes)
# 203.964 ns (0 allocations: 0 bytes)

# N = 1000 ComplexF64 (q=7)
# 237.188 ns (0 allocations: 0 bytes)
# 439.227 ns (0 allocations: 0 bytes)
# 204.939 ns (0 allocations: 0 bytes)
