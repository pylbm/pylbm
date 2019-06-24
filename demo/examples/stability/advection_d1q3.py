

"""
 Stability analysis of the
 D1Q3 solver for the advection equation

 d_t(u)   + c d_x(u) = 0
"""

import sympy as sp
import pylbm

# pylint: disable=invalid-name

# symbolic variables
U, X = sp.symbols('U, X')

# symbolic parameters
LA, C = sp.symbols('lambda, c', constants=True)
# parameters for the energy equilibrium
ALPHA = sp.symbols('alpha', constants=True)
S_1, S_2 = sp.symbols('s1, s2', constants=True)

# numerical parameters
la = 1.              # velocity of the scheme
s_1, s_2 = 1.5, 1.0  # relaxation parameters
c = 0.75             # velocity of the advection equation
alpha = .7           # equilibrium parameter for energy

dico = {
    'dim': 1,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': list(range(3)),
            'conserved_moments': U,
            'polynomials': [1, X, X**2],
            'relaxation_parameters': [0, S_1, S_2],
            'equilibrium': [U, C*U, (ALPHA*LA**2+(1-ALPHA)*C**2)*U],
        },
    ],
    'parameters': {
        LA: la,
        S_1: s_1,
        S_2: s_2,
        C: c,
        ALPHA: alpha,
    },
    'relative_velocity': [C],
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
        ALPHA: {
            'range': [0, 1],
            'init': alpha,
            'step': 0.01,
        },
        S_1: {
            'name': r"$s_1$",
            'range': [0, 2],
            'init': s_1,
            'step': 0.01,
        },
        S_2: {
            'name': r"$s_2$",
            'range': [0, 2],
            'init': s_2,
            'step': 0.01,
        },
    },
})
