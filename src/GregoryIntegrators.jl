module GregoryIntegrators

export GregoryWeights, gint
export explicit_adams_weights, explicit_stoermer_weights
export implicit_adams_weights, implicit_stoermer_weights
export sum_step

using LoopVectorization

include("avx_dot_sum.jl")
include("weights.jl")
include("polynomials.jl")

# ————————————————————————————————— Gregory integrators ———————————————————————————————— #
"""
    gint(y, δt, q, weights)
    gint(y1, y2, δt, q, weights)

Calculate ``∫ₐᵇ y(x)`` with `y[1]=a, y[end]=b` using Gregory integration.
"""
function gint(y::AbstractVector{T}, δt::Real, q::Integer,
              weights::GregoryWeights)::T where {T<:Number}

    l = length(y)
    (l < q+1) && (return poly_integrate(y, δt, l-1, weights))
    res = sum_avx(y)

    # Boundary corrections
    μG = weights.μG
    @inbounds @simd for i in 1:q
        res += μG[i] * (y[i] + y[l-(i-1)])
    end

    return δt * res
end

function gint(y1::AbstractVector{T}, y2::AbstractVector{T}, δt::Real,
              q::Integer, weights::GregoryWeights)::T where {T<:Number}
    l = length(y1)
    (l < q + 1) && (return poly_integrate(y1, y2, δt, l - 1, weights))

    res = dot_avx(y1, y2)

    # Boundary corrections
    μG = weights.μG
    @inbounds @simd for i in 1:q
        res += μG[i] * (y1[i] * y2[i] + y1[l - (i - 1)] * y2[l - (i - 1)])
    end

    return δt * res
end


# ——————————————————————————— Apply implicit or explicit step —————————————————————————— #
"""
    sum_step(pf::AbstractArray{T}, p::Integer, ex_weights::AbstractArray) where T
    sum_step(cf::T, pf::AbstractArray{T}, p::Integer, im_weights::AbstractArray) where T

Sum up contributions from explicit (no `cf`) or implicit ODE step without bounds checking.
"""
function sum_step(pf::AbstractArray{T}, p::Integer, ex_weights::AbstractArray) where T
    # @inbounds @simd
    res = zero(T)
    @simd for i in 1:p
        @inbounds res += pf[i] * ex_weights[i]
    end
    return res
end

function sum_step(cf::T, pf::AbstractArray{T}, p::Integer,
                  im_weights::AbstractArray) where T
    # @inbounds @simd
    res = cf * im_weights[1]
    @simd for i in 1:p
        @inbounds res += pf[i] * im_weights[i+1]
    end
    return res
end

end # module
