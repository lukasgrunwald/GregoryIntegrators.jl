#=
Test equations and analytical solution for integro differential equations of the type
encountered in the (Y)SYK model.
=#
using GregoryIntegrators
using SpecialFunctions

# ———————————————————————————————————— Fit functions ——————————————————————————————————— #
m(x, p) = @. x * p[1] + p[2]
m_plot(x, p) = @. exp(p[2]) * x^p[1]

# ———————————————————————————————— First order functions ——————————————————————————————— #
# To be solved with the Adams-Moulten scheme
f_exact(t) = 1 / 2 * exp(-t) * sin(2t) * heaviside(t)
f_string = L"u'(t) = -2u(x) + 1 - 5\int_0^t ds \; u(s); \quad u(0) = 0"
function f(i, u, t, q, gweights)
    algeb = -2 * u[i] + heaviside(t[i])
    Ii = -5 * gint((@view u[1:i]), t[2] - t[1], q, gweights)
    return algeb + Ii
end

g_exact(t) = 2 / 5 * (sin(2t) + 3sinh(t))
g_string = L"u'(t) = 1 + \cos(2t) + \int_0^t ds \; u(s); \quad u(0) = 0"
function g(i, u, t, q, gweights)
    algeb = 1 + cos(2t[i])
    Ii = gint((@view u[1:i]), t[2] - t[1], q, gweights)
    return algeb + Ii
end

h_exact(t) = √(π / 2) * erfi(t / √2)
h_string = L"u'(t) = 1 + t \cdot u(t) - \int_0^t ds \; u(s); \quad u(0) = 0"
function h(i, u, t, q, gweights)
    algeb = 1 + t[i] * u[i]
    Ii = -gint((@view u[1:i]), t[2] - t[1], q, gweights)
    return algeb + Ii
end

# ——————————————————————————————— Second order functions ——————————————————————————————— #
# To be solved with the stroemer scheme (predictor-corrector)
f2_exact(t) = (5 / 2)^(2 / 5) * sqrt(t) * gamma(7 / 5) * besselj(2 / 5, 4 / 5 * t^(5 / 4))
f2_string = L"u''(t) = -u(t)"
function f2(i, u, t, q, gweights)
    return - u[i] * sqrt(t[i])
end

g2_exact(t) = 3^(-1 / 3) * t * besselj(2 / 3, 2 / 3 * t^(3 / 2)) * gamma(2 / 3)
g2_string = L"u''(t) = 1 - t \cdot u(t) - \int_0^t ds \; u(s); \quad u(0)=u'(0)=0"
function g2(i, u, t, q, gweights)
    return 1 - t[i] * u[i] - gint((@view u[1:i]), t[2] - t[1], q, gweights)
end

h2_exact(t) = 1 / 3 * (exp(-t) + 2exp(t / 2) * cos(√3 * t / 2))
h2_string = L"u''(t) = -\int_0^t ds \; u(s); \quad u(0)=1, u'(0)=0"
function h2(i, u, t, q, gweights)
    return -gint((@view u[1:i]), t[2] - t[1], q, gweights)
end
