

"""
 Stability analysis of the
 D2Q4 solver for the advection equation

 d_t(u)   + c_x d_x(u) + c_y d_y(u) = 0
"""

import sympy as sp
import pylbm

# pylint: disable=invalid-name

# symbolic variables
U, X, Y = sp.symbols('U, X, Y')

# symbolic parameters
LA, CX, CY = sp.symbols('lambda, cx, cy', constants=True)
S_1, S_2 = sp.symbols('s1, s2', constants=True)

# numerical parameters
la = 1.               # velocity of the scheme
s_1, s_2 = 2., 1.    # relaxation parameters
c_x, c_y = 0.5, 0.25  # velocity of the advection equation

dico = {
    'dim': 2,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2, 3, 4],
            'conserved_moments': U,
            'polynomials': [1, X, Y, X**2-Y**2],
            'relaxation_parameters': [0, S_1, S_1, S_2],
            'equilibrium': [
                U,
                CX*U, CY*U,
                (CX**2-CY**2)*U
            ],
        },
    ],
    'parameters': {
        LA: la,
        S_1: s_1,
        S_2: s_2,
        CX: c_x,
        CY: c_y,
    },
    'relative_velocity': [CX, CY],
}

scheme = pylbm.Scheme(dico)
stab = pylbm.Stability(scheme)
stab.visualize({
    'parameters': {
        CX: {
            'range': [0, 1],
            'init': c_x,
            'step': 0.01,
        },
        CY: {
            'range': [0, 1],
            'init': c_y,
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
    'number_of_wave_vectors': 4096,
})
