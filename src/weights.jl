#=
Implementation of integrator weights for Gregory Integration (evaluation of integrals),
Adams-Moulten integrator (predictor-corrector scheme) and stroemer integrators.
=#

struct GregoryWeights
    μG::Vector{Float64}
    In::Vector{Vector{Float64}}
end

"""
    GregoryWeights(q::Integer)

GregoryWeights object for q-th order schema, containing `μᴳ` and `In`.
"""
function GregoryWeights(q::Integer)
    # Accuracy O(h^(q+1)) and q-th order schema needs q+1 points
    !(0 < q < 8) && error("GregoryWeights: Invalid order!")
    weights = Vector{Float64}(undef, q)

    if q == 1
        weights[1] = -1 / 2
    elseif q == 2
        weights[1] = -7 / 12
        weights[2] = +1 / 12
    elseif q == 3
        weights[1] = -5 / 8
        weights[2] = +1 / 6
        weights[3] = -1 / 24
    elseif q == 4
        weights[1] = -469 / 720
        weights[2] = +59 / 240
        weights[3] = -29 / 240
        weights[4] = +19 / 720
    elseif q == 5
        weights[1] = -193 / 288
        weights[2] = +77 / 240
        weights[3] = -7 / 30
        weights[4] = +73 / 720
        weights[5] = -3 / 160
    elseif q == 6
        weights[1] = -41393 / 60480
        weights[2] = +23719 / 60480
        weights[3] = -11371 / 30240
        weights[4] = +7381 / 30240
        weights[5] = -5449 / 60480
        weights[6] = +863 / 60480
    elseif q == 7
        weights[1] = -12023 / 17280
        weights[2] = +6961 / 15120
        weights[3] = -66109 / 120960
        weights[4] = +33 / 70
        weights[5] = -31523 / 120960
        weights[6] = +1247 / 15120
        weights[7] = -275 / 24192
    end

    # Generate boundary corrections
    In = Vector{Vector{Float64}}()

    for k = 1:q-1 # Order of polynomial is #points-1
        push!(In, I⁽ᴷ⁾_weights(k))
    end

    weights = GregoryWeights(weights, In)

    return weights
end

"""
    explicit_adams_weights(p::Integer)
Weights for the p-th order explicit Adams-Bashforth scheme.

Weights are saved as in ``yₘ₊₁ = yₘ + δt \\sum_{x=1}^p μₓ yₘ₋₍ₓ₋₁₎``. The p-th order scheme
has local error ``O(δtᵖ⁺¹)`` and global error ``O(δtᵖ)``.
"""
function explicit_adams_weights(p::Integer)
    !(0 < p < 6) && error("explicit_adams_weights: Invalid order!")
    weights = Vector{Float64}(undef, p)

    if p == 1
        weights[1] = 1.0
    elseif p == 2
        weights[1] = +3 / 2
        weights[2] = -1 / 2
    elseif p == 3
        weights[1] = +23 / 12
        weights[2] = -4 / 3
        weights[3] = +5 / 12
    elseif p == 4
        weights[1] = +55 / 24
        weights[2] = -59 / 24
        weights[3] = +37 / 24
        weights[4] = -3 / 8
    elseif p == 5
        weights[1] = +1901 / 720
        weights[2] = -1387 / 360
        weights[3] = +109 / 30
        weights[4] = -637 / 360
        weights[5] = +251 / 720
    end

    return weights
end

"""
    implicit_adams_weights(p::Integer)
Weights for the p-th order Adams-Moulton scheme.

Weights are saved as in ``yₘ₊₁ = yₘ + δt \\sum_{x=0}^p μₓ yₘ₋ₓ``. The p-th order scheme
has local error ``O(δtᵖ⁺²)`` and global error ``O(δtᵖ⁺¹)``.
"""
function implicit_adams_weights(p::Integer)
    !(0 < p < 6) && error("implicit_adams_weights: Invalid order!")
    weights = Vector{Float64}(undef, p + 1)

    if p == 1 # Heuns method
        weights[1] = +1 / 2
        weights[2] = +1 / 2
    elseif p == 2
        weights[1] = +5 / 12
        weights[2] = +2 / 3
        weights[3] = -1 / 12
    elseif p == 3
        weights[1] = +3 / 8
        weights[2] = +19 / 24
        weights[3] = -5 / 24
        weights[4] = +1 / 24
    elseif p == 4
        weights[1] = +251 / 720
        weights[2] = +323 / 360
        weights[3] = -11 / 30
        weights[4] = +53 / 360
        weights[5] = -19 / 720
    elseif p == 5
        weights[1] = +95 / 288
        weights[2] = +1427 / 1440
        weights[3] = -133 / 240
        weights[4] = +241 / 720
        weights[5] = -173 / 1440
        weights[6] = +3 / 160
    end

    return weights
end

"""
    explicit_stoermer_weights(p::Integer, warning::Bool=true)
Weights for the p-th order explicit stoermer scheme.

Weights are saved as in ``yₘ₊₁ = 2yₘ - yₘ₋₁ + δt² \\sum_{x=1}^p μₓ yₘ₋₍ₓ₋₁₎``. The p-th order
scheme has local error ``O(δtᵖ⁺¹)`` and global error ``O(δtᵖ)``.
"""
function explicit_stoermer_weights(p::Integer, warning::Bool=true)
    !(0 < p < 6) && error("explicit_stoermer_weights: Invalid order!")
    # weights = Vector{Float64}(undef, p!=2 ? p : p-1)
    weights = Vector{Float64}(undef, p)
    # Scheme of order 1,2 are exactly the same!!

    if p == 1
        weights[1] = +1.0
    elseif p == 2
        weights[1] = +1.0
        weights[2] = +0.0
        warning && printstyled("Warning: Called explicit_stoermer_weights with p=2!\n",
                               color=:red)
    elseif p == 3
        weights[1] = +13 / 12
        weights[2] = -1 / 6
        weights[3] = +1 / 12
    elseif p == 4
        weights[1] = +7 / 6
        weights[2] = -5 / 12
        weights[3] = +1 / 3
        weights[4] = -1 / 12
    elseif p == 5
        weights[1] = +299 / 240
        weights[2] = -11 / 15
        weights[3] = +97 / 120
        weights[4] = -2 / 5
        weights[5] = +19 / 240
    end

    return weights
end

"""
    implicit_stoermer_weights(p::Integer, warning::Bool=true)
Weights for the p-th order implicit stoermer scheme.

Weights are saved as in ``yₘ₊₁ = 2yₘ - yₘ₋₁ + δt² \\sum_{x=0}^p μₓ yₘ₋₍ₓ₋₁₎``. The p-th order
scheme has local error ``O(δtᵖ⁺²)`` and global error ``O(δtᵖ⁺¹)``.
"""
function implicit_stoermer_weights(p::Integer, warning::Bool=true)
    !(0 < p < 6) && error("implicit_stroemer_weights: Invalid order!")
    # weights = Vector{Float64}(undef, p!=3 ? p : p-1)
    weights = Vector{Float64}(undef, p+1)

    if p == 1
        weights[1] = 1.0
        weights[2] = 0.0
        warning && printstyled("Warning: Called implicit_stroemer_weights with p=1!\n",
                               color=:orange)
    elseif p == 2
        weights[1] = +1 / 12
        weights[2] = +10 / 12
        weights[3] = +1 / 12
    elseif p == 3
        weights[1] = +1 / 12
        weights[2] = +10 / 12
        weights[3] = +1 / 12
        weights[4] = +0.0
        warning && printstyled("Warning: Called implicit_stroemer_weights with p=3!\n",
                               color=:orange)
    elseif p == 4
        weights[1] = +19 / 240
        weights[2] = +17 / 20
        weights[3] = +7 / 120
        weights[4] = +1 / 60
        weights[5] = -1 / 240
    elseif p == 5
        weights[1] = +18 / 240
        weights[2] = +209 / 240
        weights[3] = +1 / 60
        weights[4] = +7 / 120
        weights[5] = -1 / 40
        weights[6] = +1 / 240
    end

    return weights
end
