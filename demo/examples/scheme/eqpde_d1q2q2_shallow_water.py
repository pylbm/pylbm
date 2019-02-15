

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Example of a two velocities scheme for the shallow water system

 d_t(h) + d_x(q)    = 0, t > 0, 0 < x < 1,
 d_t(q) + d_x(q^2/h+gh^2/2) = 0, t > 0, 0 < x < 1,
"""
import sympy as sp
import pylbm

# pylint: disable=invalid-name

H, Q, X = sp.symbols('h, q, X')
LA, G = sp.symbols('lambda, g', constants=True)
SIGMA_H, SIGMA_Q = sp.symbols('sigma_1, sigma_2', constants=True)

scheme_cfg = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2],
            'conserved_moments': H,
            'polynomials': [1, X],
            'relaxation_parameters': [0, 1/(.5+SIGMA_H)],
            'equilibrium': [H, Q],
        },
        {
            'velocities': [1, 2],
            'conserved_moments': Q,
            'polynomials': [1, X],
            'relaxation_parameters': [0, 1/(.5+SIGMA_Q)],
            'equilibrium': [Q, Q**2/H+.5*G*H**2],
        },
    ],
    'parameters': {
        LA: 1.,
        G: 9.81,
        SIGMA_H: 1/1.8-.5,
        SIGMA_Q: 1/1.2-.5,
    },
}

scheme = pylbm.Scheme(scheme_cfg)
eq_pde = pylbm.EquivalentEquation(scheme)

print(eq_pde)
