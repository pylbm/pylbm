

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
LA, C, S = sp.symbols('lambda, c, s', constants=True)

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
            'relaxation_parameters': [0, S],
            'equilibrium': [U, C*U],
        },
    ],
    'parameters': {
        LA: la,
        S: s,
        C: c,
    },
}

scheme = pylbm.Scheme(dico)
stab = pylbm.Stability(scheme)

stab.visualize({
    'parameters': {
        C: {
            'range': [0, 1.5],
            'init': c,
            'step': 0.01,
        },
        S: {
            'range': [0, 2],
            'init': s,
            'step': 0.01,
        },
    },
})
