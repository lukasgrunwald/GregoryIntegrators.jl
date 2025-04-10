# GregoryIntegrators.jl

This package implements integrator weights for various constant step size schemes. It further provides efficient functions for the numerical evaluation of integrals using Gregory integration
$$
\int_a^b d t \; f(t) \approx \delta t \sum_{n=1}^N f(t_n) + \delta t \sum_{l=1}^{q} \mu_l^{G} \Big[ f\big(t_l\big) + f\big(t_{N-(l-1)}\big) \Big].
$$

See Appendix of my master thesis on more details on the theory.

We export all the functions for generating the integrator weights, as well as the `gint` function, which can be used to calculate integrals and convolutions efficiently.
