#=
Implementation of avx and simd enabeled sum functions that can handle complex numbers
using the LoopVectorization @turbo command.
=#

# ————————————————————————————————— Sum implementations ———————————————————————————————— #
function sum_simd(x::AbstractVector{T}) where {T<:Number}
    s = zero(T)
    @fastmath @simd for i ∈ eachindex(x)
        @inbounds s += x[i]
    end
    return s
end

function sum_avx(a::AbstractVector{T})::T where {T<:Real}
    s = zero(T)
    @turbo for i ∈ eachindex(a)
        @inbounds s += a[i]
    end
    return s
end

function sum_avx(ca::AbstractVector{Complex{T}})::Complex{T} where {T<:Real}
    a = reinterpret(reshape, T, ca)
    s_re = zero(T)
    s_im = zero(T)
    @turbo for i ∈ axes(a, 2)
        s_re += a[1, i]
        s_im += a[2, i]
    end
    return Complex{T}(s_re, s_im)
end

# ————————————————————————————————— Dot implementations ———————————————————————————————— #
#= Implementation without the conventional complex conjugate, since needed in this form
for the gint functions
=#
function dot_simd(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    s = zero(T)
    @fastmath @simd for i ∈ eachindex(a)
        @inbounds s += a[i] * b[i]
    end
    return s
end

function dot_avx(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real}
    s = zero(T)
    @turbo for i ∈ eachindex(a)
        s += a[i] * b[i]
    end
    return s
end

function dot_avx(ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}) where {T}
    a = reinterpret(reshape, T, ca)
    b = reinterpret(reshape, T, cb)
    s_re = zero(T)
    s_im = zero(T)
    @turbo for i ∈ axes(a, 2)
        s_re += a[1, i] * b[1, i] - a[2, i] * b[2, i]
        s_im += a[1, i] * b[2, i] + a[2, i] * b[1, i]
    end
    return Complex{T}(s_re, s_im)
end
