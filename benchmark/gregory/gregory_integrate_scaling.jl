#=
Compare numerical evaluation of integrals with numerical Gregory integration.
Check δt-sclaing of the implemented method with the theoretical prediction.
=#

using Distributions
using GregoryIntegrators
using LsqFit
using MyPlot
using PConv
using Printf

function compare_exact(save=false)
    xlower, xupper, δx = -5, -2, 0.05
    x = (xupper+δx):δx:10
    f1(x) = exp(im * x)
    f2(x) = cos(2x) * sin(x)
    f3(x) = x / (1 + x^2)^(3 / 2)
    # f4(x) = tanh(x)

    I1 = @. -im * (exp(im * x) - exp(xlower * im))
    I2 = @. 1/2 * cos(x) - 1/6 * cos(3x) - (1/2 * cos(xlower) - 1/6 * cos(3xlower))
    I3 = @. - 1 / sqrt(x^2 + 1) + 1/sqrt(xlower^2 + 1)
    # I4 = @. log(cosh(x)) - log(cosh(xlower))

    I1g = Matrix{ComplexF64}(undef, length(x), 7)
    I2g = Matrix{ComplexF64}(undef, length(x), 7)
    I3g = Matrix{ComplexF64}(undef, length(x), 7)

    for q=1:7
        gweights = GregoryWeights(q)
        for i in eachindex(x)
            δt = 0.05
            xi = xlower:δt:round(x[i], digits = 5)

            y1i = f1.(xi)
            y2i = f2.(xi)
            y3i = f3.(xi)

            I1g[i, q] = gint(y1i, δt, q, gweights)
            I2g[i, q] = gint(y2i, δt, q, gweights)
            I3g[i, q] = gint(y3i, δt, q, gweights)
        end
    end

    fig, ax = plt.subplots(ncols=3, figsize=(8, 4), dpi=200)
    [(it.set_yscale("log"), it.set_xlabel("t")) for it in ax]
    ax[1].set_title(L"\int_{-5}^t dx \, e^{ix}")
    ax[2].set_title(L"\int_{-5}^t dx \, \cos(2x) \cdot \sin(x)")
    ax[3].set_title(L"\int_{-5}^t dx \,  x \cdot (1 + x^2)^{-3/2}")

    for q=1:7
        ax[1].plot(x, abs.(I1 - I1g[:, q]), marker=".", markersize=3 ,label="q=$(q)")
        ax[2].plot(x, abs.(I2 - I2g[:, q]), marker=".", markersize=3 ,label="q=$(q)")
        ax[3].plot(x, abs.(I3 - I3g[:, q]), marker=".", markersize=3 ,label="q=$(q)")
    end

    [it.set_ylim(1e-14) for it in ax]
    ax[1].legend(ncol=2, )
    plt.tight_layout()
    path = versionized_path("pictures/comp_exact.png"; abspath=dirname(@__DIR__))
    save && savefig(path)
    plt.show()
end

function gregory_error_scaling(save=false)
    δts = [0.0125, 0.025, 0.05, 0.1, 0.125, 0.2, 0.25]
    xmax = 20

    # Test functions
    f(x) = exp(im * x)
    fe(x) = -im * exp(im * x)

    # Other options
    # f(x) = 1 / (1 + x^2 + x)^(3/2); fe(x) = -2/3*(2+x) / sqrt(x^2 + x + 1)
    # f(x) = tanh(x); fe(x) = log(cosh(x))

    # Get the maximum and mean differences
    dme = Matrix{Float64}(undef, length(δts), 7)
    dmx = Matrix{Float64}(undef, length(δts), 7)

    for i in eachindex(δts)
        δt = δts[i]
        xlower, xupper = -5, 0
        x = (xupper+δt):δt:xmax
        ye = @. fe(x) - fe(xlower) # exact solution

        temp = Matrix{ComplexF64}(undef, length(x), 7)
        for q in 1:7
            gweights = GregoryWeights(q)

            for j in eachindex(x)
                xj = xlower:δt:round(x[j], digits = 5)
                yt = f.(xj)

                temp[j, q] = gint(yt, δt, q, gweights)
            end

            # Extract the mean and maximum difference
            dme[i, q] = @views mean(abs.(ye .- temp[:, q]))
            dmx[i, q] = @views maximum(abs.(ye .- temp[:, q]))
        end
    end

    # Doing the fits for all curves
    m(x, p) = @. x * p[1] + p[2]
    pm, px = (Matrix{Float64}(undef, 2, 7) for _=1:2)
    xd = minimum(δts):0.001:maximum(δts)+0.05

    for q in 1:7
        start = 1
        (q==7 || q == 6) && (start = 2)

        pm[:, q] = curve_fit(m, log.(δts[start:end]),
                             log.(dme[start:end, q]), [1.0, 1.0]).param
        px[:, q] = curve_fit(m, log.(δts[start:end]),
                             log.(dmx[start:end, q]), [1.0, 1.0]).param
    end

    fig, ax = plt.subplots(ncols=2, figsize=(15CM, 8CM), dpi=200)
    fig.suptitle("Scaling analysis for Gregory Integration")

    for q in 1:7
        spm = @sprintf "%.2f" pm[1, q]
        spx = @sprintf "%.2f" px[1, q]

        base, = ax[1].plot(δts, dme[:, q], marker=".", markersize=5, label=L"q=%$spm")
        ax[1].plot(xd, @. exp(pm[2, q]) * xd^pm[1, q]; color=base.get_color(),
                   linestyle="--", zorder=1)

        base, = ax[2].plot(δts, dmx[:, q], marker=".", markersize=5, label=L"q=%$spx")
        ax[2].plot(xd, @. exp(px[2, q]) * xd^px[1, q]; color=base.get_color(),
                   linestyle="--", zorder=1)
    end

    [(it.set_yscale("log"), it.set_xscale("log"), it.set_xlabel(L"\delta t")) for it in ax]
    ax[1].set_ylabel("mean-error")
    ax[2].set_ylabel("max-error")
    plt.tight_layout()
    # [it.legend() for it in ax]
    [it.legend(shadow=true, loc="upper center", bbox_to_anchor=(0.84, 0.6)) for it ∈ ax]
    path = versionized_path("pictures/scaling_gregory.png"; abspath=dirnames(2, @__DIR__))
    save && savefig(path)
    plt.show()
end

# compare_exact(false)
# gregory_error_scaling(false)
