

"""
 Stability analysis of the
 D1Q2 solver for the advection equation

 d_t(u)   + c d_x(u) = 0
"""

import sympy as sp
import pylbm

# pylint: disable=invalid-name

# symbolic variables
U, X = sp.symbols('U, X')

# symbolic parameters
LA, C, SIGMA = sp.symbols('lambda, c, sigma', constants=True)
symb_s = 1/(.5+SIGMA)  # symbolic relaxation parameter

# numerical parameters
la = 1.  # velocity of the scheme
s = 1.9  # relaxation parameters
c = 0.5  # velocity of the advection equation

dico = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2],
            'conserved_moments': U,
            'polynomials': [1, X],
            'relaxation_parameters': [0, symb_s],
            'equilibrium': [U, C*U],
        },
    ],
    'parameters': {
        LA: la,
        SIGMA: 1/s-.5,
        C: c,
    },
}

scheme = pylbm.Scheme(dico, formal=True)
stab = pylbm.Stability(scheme)

stab.visualize()
