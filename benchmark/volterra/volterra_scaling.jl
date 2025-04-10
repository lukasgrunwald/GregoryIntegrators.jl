#=
Application of my designed scheme to scalar integro differential equations that can be solved
analytically (see mathematica notebook for derivation of analytical results). We compare the
time dependence and provide a scaling analysis for 1st and 2nd order equations.
=#
using GregoryIntegrators
using Parameters
using PConv
using Printf
using LsqFit
using MyPlot
using SpecialFunctions
include("1d_volterra_functions.jl")


mutable struct VolterraParameters
    tmin::Float64
    tmax::Float64
    δt::Float64
    q::Int64
    p::Int64
    miter::Int64
    ϵ::Float64
    initial::Float64
    start::Int64
end

# ————————————————————— Application of implicit explicit time step ————————————————————— #
function step_explicit!(f, i, u, t, p, q, weights, miter=0, ϵ=0)
    ex_weights = weights[1]
    gweights = weights[2]
    δt = t[2] - t[1]
    pf = Vector{Float64}(undef, p)

    for j in 1:p # Bashforth predictor step
        pf[j] = f(i - (j - 1), u, t, q, gweights)
    end

    u[i+1] = u[i] + δt * sum_step(pf, p, ex_weights)
    return nothing
end

function step_explicit_2nd!(f, i, u, t, p, q, weights, miter=0, ϵ=0)
    ex_weights = weights[1]
    gweights = weights[2]
    δt = t[2] - t[1]
    pf = Vector{Float64}(undef, p)

    for j in 1:p # Predictor step
        pf[j] = f(i - (j - 1), u, t, q, gweights)
    end

    u[i+1] = 2u[i] - u[i-1] + δt^2 * sum_step(pf, p, ex_weights)
    return nothing
end

function step_implicit!(f, i, u, t, p, q, weights, miter=100, ϵ=1e-8)
    ex_weights = weights[1]
    im_weights = weights[3]
    gweights = weights[2]
    δt = t[2] - t[1]
    pf = Vector{Float64}(undef, p)

    for j in 1:p # Bashforth predictor step
        pf[j] = f(i - (j - 1), u, t, q, gweights)
    end

    u[i+1] = u[i] + δt * sum_step(pf, p, ex_weights)
    up = u[i+1]

    for j in 1:miter
        cf = f(i + 1, u, t, q, gweights)
        u[i+1] = u[i] + δt * sum_step(cf, pf, p, im_weights)

        (abs(up - u[i+1]) < ϵ) && break
        up = u[i+1]
    end

    return nothing
end

function step_implicit_2nd!(f, i, u, t, p, q, weights, miter=100, ϵ=1e-10)
    ex_weights = weights[1]
    im_weights = weights[3]
    gweights = weights[2]
    δt = t[2] - t[1]
    pf = Vector{Float64}(undef, p)
    for j in 1:p # Bashforth predictor step
        pf[j] = f(i - (j - 1), u, t, q, gweights)
    end

    u[i+1] = 2u[i] - u[i-1] + δt^2 * sum_step(pf, p, ex_weights)
    up = u[i+1]

    for j in 1:miter
        cf = f(i + 1, u, t, q, gweights)
        u[i+1] = 2u[i] - u[i-1] + δt^2 * sum_step(cf, pf, p, im_weights)

        (abs(up - u[i+1]) < ϵ) && break
        up = u[i+1]
    end

    return nothing
end

# ——————————————————————————— Integrator for scalar equations —————————————————————————— #
function integrate_volterra(f, fe, stepper, param)
    @unpack tmin, tmax, δt = param
    @unpack q, p, miter, ϵ = param
    @unpack start, initial = param

    t = tmin:δt:tmax
    N = length(t)

    u = Vector{Float64}(undef, N)
    gweights = GregoryWeights(q)
    ex_weights = explicit_adams_weights(p)
    im_weights = implicit_adams_weights(p)

    # Setting initial values
    u[1] = initial
    u[1:start] = fe.(t[1:start])

    for i in start:N-1
        stepper(f, i, u, t, p, q, (ex_weights, gweights, im_weights), miter, ϵ)
    end

    return t, u
end

function integrate_volterra_2nd(f, fe, stepper, param)
    @unpack tmin, tmax, δt = param
    @unpack q, p, miter, ϵ = param
    @unpack start, initial = param

    t = tmin:δt:tmax
    N = length(t)

    u = Vector{Float64}(undef, N)
    gweights = GregoryWeights(q)
    ex_weights = explicit_stoermer_weights(p, false)
    im_weights = implicit_stoermer_weights(p, false)

    # Setting initial values
    u[1] = initial
    u[1:start] = fe.(t[1:start])

    for i in start:N-1
        stepper(f, i, u, t, p, q, (ex_weights, gweights, im_weights), miter, ϵ)
    end

    return t, u
end

# —————————————————————————————————— Scaling analysis —————————————————————————————————— #
function scaling_volterra(p, stepper, f, f_exact, tmin, tmax, δts, miter=1, ϵ=1e-10)
    l = length(δts)
    start = 7
    initial = 0.0
    q = 7
    obj = VolterraParameters(tmin, tmax, δts[1], q, p, miter, ϵ, initial, start)

    Δmax = Vector{Float64}(undef, l)
    for i in eachindex(δts)
        obj.δt = δts[i]
        t, u = integrate_volterra(f, f_exact, stepper, obj)
        ue = f_exact.(t)

        Δmax[i] = maximum(@. abs(ue - u))
    end
    return δts, Δmax
end

function scaling_volterra_2nd(p, stepper, f, f_exact, tmin, tmax, δts, miter=100, ϵ=1e-10)
    l = length(δts)
    start = 7
    initial = 0.0
    q = 7
    obj = VolterraParameters(tmin, tmax, δts[1], q, p, miter, ϵ, initial, start)

    Δmax = Vector{Float64}(undef, l)
    for i in eachindex(δts)
        obj.δt = δts[i]
        t, u = integrate_volterra_2nd(f, f_exact, stepper, obj)
        ue = f_exact.(t)

        Δmax[i] = maximum(@. abs(ue - u))
    end
    return δts, Δmax
end

function scaling_plot_volterra(f, f_exact, f_string; tmax=10.0, save=false)
    δts = vcat(0.0005, 0.001, 0.002, 0.0025:0.0025:0.075)
    xd = δts[1]:0.001:δts[end]+0.025

    fit_param_ex, fit_param_im = (Matrix{Float64}(undef, 2, 5) for _ in 1:2)
    max_diff_ex, max_diff_im = (Matrix{Float64}(undef, length(δts), 5) for _ in 1:2)

    for p in 1:5
        δts, Δmax = scaling_volterra(p, step_explicit!, f, f_exact, 0, tmax, δts)
        max_diff_ex[:, p] .= Δmax

        δts, Δmax = scaling_volterra(p, step_implicit!, f, f_exact, 0, tmax, δts, 1, 1e-10)
        max_diff_im[:, p] .= Δmax
    end

    start, cut = 2, 15
    for p in 1:5
        p == 5 && (start = 5)
        x_temp = @view δts[start:(end-cut)]
        y_temp_ex = @view max_diff_ex[start:(end-cut), p]
        y_temp_im = @view max_diff_im[start:(end-cut), p]
        fit_param_ex[:, p] = curve_fit(m, log.(x_temp), log.(y_temp_ex), [1.0, 1.0]).param
        fit_param_im[:, p] = curve_fit(m, log.(x_temp), log.(y_temp_im), [1.0, 1.0]).param
    end

    fig, ax = plt.subplots(ncols=2)
    fig.suptitle(f_string)
    [(it.set_xscale("log"), it.set_yscale("log")) for it in ax]

    ax[1].set_title("Explicit (Bashforth)")
    ax[2].set_title("Implicit (Moulton)")

    for p in 1:5
        p_fit_ex = @sprintf "%.2f" fit_param_ex[1, p]
        p_fit_im = @sprintf "%.2f" fit_param_im[1, p]

        base, = ax[1].plot(δts, max_diff_ex[:, p]; label=L"p=%$p", markero...)
        ax[1].plot(xd, m_plot(xd, fit_param_ex[:, p]); color=base.get_color(), linestyle="--",
            label=L"p_{fit}=%$p_fit_ex", zorder=1)

        base, = ax[2].plot(δts, max_diff_im[:, p]; label=L"p=%$p", markero...)
        ax[2].plot(xd, m_plot(xd, fit_param_im[:, p]); color=base.get_color(), linestyle="--",
            label=L"p_{fit}=%$p_fit_im", zorder=1)
    end

    # it.set_ylim(5 * eps())
    [(it.set_xlabel(L"\delta t"), it.set_ylabel(L"\Delta_{max}"),
        it.legend(shadow=true)) for it in ax]
    plt.tight_layout()
    path = versionized_path("scaling_ex_im.png", abspath=@__DIR__)
    save && plt.savefig(path)
    plt.show()
end

function scaling_plot_volterra_2nd(f, f_exact, f_string; tmax=10.0, save=false)
    δts = vcat(0.001, 0.002, 0.0025:0.0025:0.2)
    xd = δts[1]:0.001:δts[end]+0.025

    fit_param_ex, fit_param_im = (Matrix{Float64}(undef, 2, 4) for _ in 1:2)
    max_diff_ex = Matrix{Float64}(undef, length(δts), 4)
    max_diff_im = Matrix{Float64}(undef, length(δts), 3)

    # Calculate differences
    for (i,p) in enumerate([1, 3, 4, 5])
        δts, Δmax = scaling_volterra_2nd(p, step_explicit_2nd!, f, f_exact, 0, tmax, δts)
        max_diff_ex[:, i] .= Δmax
    end
    for (i,p) in enumerate([2, 4, 5])
        δts, Δmax = scaling_volterra_2nd(p, step_implicit_2nd!, f, f_exact, 0, tmax, δts)
        max_diff_im[:, i] .= Δmax
    end

    # Fit log-log lines
    start, cut = 5,  10
    for p in 1:size(max_diff_ex)[2]
        x_temp = @view δts[start:(end-cut)]
        y_temp = @view max_diff_ex[start:(end-cut), p]
        fit_param_ex[:, p] = curve_fit(m, log.(x_temp), log.(y_temp), [1.0, 1.0]).param
    end

    start, cut = 5, 10
    for p in 1:size(max_diff_im)[2]
        x_temp = @view δts[start:(end-cut)]
        y_temp = @view max_diff_im[start:(end-cut), p]
        fit_param_im[:, p] = curve_fit(m, log.(x_temp), log.(y_temp), [1.0, 1.0]).param
    end

    # Plotting
    fig, ax = plt.subplots(ncols=2)
    fig.suptitle(f_string)
    [(it.set_xscale("log"), it.set_yscale("log")) for it in ax]

    ax[1].set_title("Explicit (Stoermer)")
    ax[2].set_title("Implicit (Stoermer)")

    for (i, p) in enumerate([1, 3, 4, 5])
        p_fit_ex = @sprintf "%.2f" fit_param_ex[1, i]
        # p_fit_im = @sprintf "%.2f" fit_param_im[1, p]
        p_string = p == 1 ? L"p=1,2" : L"p=%$p"
        base, = ax[1].plot(δts, max_diff_ex[:, i]; label=p_string, markero...)
        ax[1].plot(xd, m_plot(xd, fit_param_ex[:, i]); color=base.get_color(),
         linestyle="--", label=L"p_{fit}=%$p_fit_ex", zorder=1)
    end

    ax[2]._get_lines.get_next_color()
    for (i, p) in enumerate([2, 4, 5])
        # p_fit_ex = @sprintf "%.2f" fit_param_ex[1, i]
        p_fit_im = @sprintf "%.2f" fit_param_im[1, i]
        p_string = p == 2 ? L"p=2,3" : L"p=%$p"
        base, = ax[2].plot(δts, max_diff_im[:, i]; label=p_string, markero...)
        ax[2].plot(xd, m_plot(xd, fit_param_im[:, i]); color=base.get_color(), linestyle="--",
                label=L"p_{fit}=%$p_fit_im", zorder=1)
    end

    # it.set_ylim(5 * eps())
    [(it.set_xlabel(L"\delta t"), it.set_ylabel(L"\Delta_{max}"), it.set_ylim(5 * eps()),
        it.legend(shadow=true)) for it in ax]
    plt.tight_layout()
    path = versionized_path("2nd_scaling_ex_im.png", abspath=@__DIR__)
    save && plt.savefig(path)
    plt.show()
end

# ———————————————————————————————— Comparison functions ———————————————————————————————— #
function comparison_consistency_1step(f, f_exact, f_string; tmax=10.0, save=false)

    δts = vcat(0.0005, 0.001, 0.002, 0.0025:0.0025:0.075)
    xd = δts[1]:0.001:δts[end]+0.025

    fit_param = Matrix{Float64}(undef, 2, 5)
    max_diff_SC = Matrix{Float64}(undef, length(δts), 5)
    max_diff_AM = Matrix{Float64}(undef, length(δts), 5)

    for p in 1:5
        δts, Δmax = scaling_volterra(p, step_implicit!, f, f_exact, 0, tmax, δts, 100, 1e-10)
        max_diff_SC[:, p] .= Δmax

        δts, Δmax = scaling_volterra(p, step_implicit!, f, f_exact, 0, tmax, δts, 1, 1e-10)
        max_diff_AM[:, p] .= Δmax
    end

    fig, ax = plt.subplots()
    plt.suptitle(L"Comparison self-consistency vs 1 AM-step ($\epsilon=10^{-10}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f_string)
    for p in 1:5
        base, = ax.plot(δts, max_diff_SC[:, p]; label=L"p=%$p", markero..., linestyle=":")
        ax.plot(δts, max_diff_AM[:, p]; color=base.get_color(), markerx...)
    end

    ax.scatter(0.1, 0; label="1-Step", marker="x", color="black")
    ax.scatter(0.1, 0; label="SC", marker=".", color="black")
    ax.set_xlabel(L"\delta t")
    ax.set_ylabel(L"\Delta_{max}")

    plt.ylim(5 * eps())
    plt.legend()
    path = versionized_path("comparison_sc_am.png", abspath=@__DIR__)
    save && plt.savefig(path)
    plt.show()
end

function comparison_consistency_1step_2nd(f, f_exact, f_string; tmax=10.0, save=false)
    δts = vcat(0.001, 0.002, 0.0025:0.0025:0.2)
    xd = δts[1]:0.001:δts[end]+0.025

    max_diff_SC = Matrix{Float64}(undef, length(δts), 3)
    max_diff_AM = Matrix{Float64}(undef, length(δts), 3)

    for (i,p) in enumerate([2, 4, 5])
        δts, Δmax = scaling_volterra_2nd(p, step_implicit_2nd!, f, f_exact, 0, tmax, δts,
                                         1, 0)
        max_diff_AM[:, i] .= Δmax

        δts, Δmax = scaling_volterra_2nd(p, step_implicit_2nd!, f, f_exact, 0, tmax, δts,
                                         100, 1e-10)
        max_diff_SC[:, i] .= Δmax
    end

    fig, ax = plt.subplots()
    plt.suptitle(L"Comparison self-consistency vs 1 AM-step 2nd ($\epsilon=10^{-10}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f_string)
    for (i, p) in enumerate([2, 4, 5])
        p_string = p == 2 ? L"p=2,3" : L"p=%$p"
        base, = ax.plot(δts, max_diff_SC[:, i]; label=p_string, markero..., linestyle=":")
        ax.plot(δts, max_diff_AM[:, i]; color=base.get_color(), markerx...)
    end

    ax.scatter(0.1, 0; label="1-Step", marker="x", color="black")
    ax.scatter(0.1, 0; label="SC", marker=".", color="black")
    ax.set_xlabel(L"\delta t")
    ax.set_ylabel(L"\Delta_{max}")

    plt.ylim(5 * eps())
    plt.legend()
    path = versionized_path("2nd_comparison_sc_am.png", abspath=@__DIR__)
    save && plt.savefig(path)
    plt.show()
end

# —————————————————————— Single correction vs SC for Adams-Moulten ————————————————————— #
# comparison_consistency_1step(f, f_exact, f_string; tmax=10, save=false)
# comparison_consistency_1step(h, h_exact, h_string; tmax = 3, save = false)
# comparison_consistency_1step(g, g_exact, g_string; tmax = 10, save = false)

# ———————————————————————— Single correction vs SC for Stroemer ———————————————————————— #
# comparison_consistency_1step_2nd(f2, f2_exact, f2_string; tmax=10.0, save=false)
# comparison_consistency_1step_2nd(g2, g2_exact, f2_string; tmax=10.0, save=false)
# comparison_consistency_1step_2nd(h2, h2_exact, h2_string; tmax=10.0, save=false)

# ———————————————————————————————————— Scaling plots ——————————————————————————————————— #
# scaling_plot_volterra(f, f_exact, f_string; save=false)
# scaling_plot_volterra_2nd(f2, f2_exact, g2_string; tmax=10.0, save=false)
# scaling_plot_volterra_2nd(g2, g2_exact, g2_string; tmax=10.0, save=false)
