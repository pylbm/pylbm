

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
SIGMA_1, SIGMA_2 = sp.symbols('sigma_1, sigma_2', constants=True)
symb_s1 = 1/(.5+SIGMA_1)  # symbolic relaxation parameter
symb_s2 = 1/(.5+SIGMA_2)  # symbolic relaxation parameter

# numerical parameters
la = 1.               # velocity of the scheme
s_1, s_2 = 1.8, 1.    # relaxation parameters
c_x, c_y = 0.5, 0.25  # velocity of the advection equation

dico = {
    'dim': 2,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2, 3, 4],
            'conserved_moments': U,
            'polynomials': [1, X, Y, X**2-Y**2],
            'relaxation_parameters': [0, symb_s1, symb_s1, symb_s2],
            'equilibrium': [U, CX*U, CY*U, 0],
        },
    ],
    'parameters': {
        LA: la,
        SIGMA_1: 1/s_1-.5,
        SIGMA_2: 1/s_2-.5,
        CX: c_x,
        CY: c_y,
    },
    'relative_velocity': [CX, CY],
}

scheme = pylbm.Scheme(dico, formal=True)
stab = pylbm.Stability(scheme)
stab.visualize({'number_of_wave_vectors': 4096})
