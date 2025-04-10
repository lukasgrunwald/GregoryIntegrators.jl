#=
Polynomal interpolation used when gregory integration does not have enough points.
Derivations and idea from: https://www.sciencedirect.com/science/article/pii/S0010465520302277
In order to use q-th order Gregory scheme, one needs at least q + 1 sample points. If there
are only k < q + 1 points, we use this polynomial interpolation.
=#

"""
    P⁽ᴷ⁾_weights(k::Integer)

Universal polynomial interpolation coefficients for polynomial of order k. This is independent
of f(x) and δx!
"""
function P⁽ᴷ⁾_weights(k::Integer)
    (k<1) && error("P⁽ᴷ⁾_weights: Invalid order")

    A = Matrix{Rational}(undef, k + 1, k + 1)

    @inbounds for j in 1:k+1
        for i in 1:k+1
            A[i, j] = (i - 1)^(j - 1)
        end
    end
    return inv(A)
end

"""
    Pᴷt(t, Pᴷ, fl, Δt)

Evaluate interpolation polynomial at time-argument t.
"""
function Pᴷt(t, Pᴷ, fl, Δt)

    k = length(fl)-1
    res = zero(eltype(fl))
    for a = 1:k+1, l = 1:k+1
        res += (t / Δt)^(a - 1) * Pᴷ[a, l] * fl[l]
    end
    return res
end

"""
    I⁽ᴷ⁾_weights(k::Integer)

Polynomial integration weights of order k. Can be used to evaluate the integral as
∫ₐᵇ ds f(s) ≈ δt ∑_{l=0}^k f_k I⁽ᴷ⁾_l
"""
function I⁽ᴷ⁾_weights(k::Integer)
    Pᴷ = P⁽ᴷ⁾_weights(k)
    Il = zeros(Rational, k + 1)

    @inbounds for l = 1:k+1
        for a = 1:k+1
            Il[l] += Pᴷ[a, l] * k^a / a
        end
    end

    return Float64.(Il)
end

"""
    poly_integrate(y::AbstractVector{T}, δt::Real, k::Integer, Gweights::GregoryWeights)
    poly_integrate(y1::AbstractVector{T}, y2::AbstractVector{T}, δt::Real, k::Integer, Gweights::GregoryWeights)

Polynomial integration function, used if not enough points for using Gregory Integration.
"""
function poly_integrate(y::AbstractVector{T}, δt::Real, k::Integer,
                        Gweights::GregoryWeights) where {T}
    (k < 1) && return zero(T)
    In = Gweights.In[k]

    res = zero(T)
    @inbounds @simd for i in 1:(k + 1)
        res += y[i] * In[i]
    end
    return δt * res
end

function poly_integrate(y1::AbstractVector{T}, y2::AbstractVector{T}, δt::Real, k::Integer,
                        Gweights::GregoryWeights) where {T}
    (k < 1) && return zero(T)

    In = Gweights.In[k]

    res = zero(T)
    @inbounds @simd for i in 1:(k + 1)
        res += y1[i] * y2[i] * In[i]
    end
    return δt * res
end
