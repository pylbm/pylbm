

"""
 Stability analysis of the
 D2Q5 solver for the advection equation

 d_t(u)   + c_x d_x(u) + c_y d_y(u) = 0
"""

import sympy as sp
import pylbm

# pylint: disable=invalid-name

# symbolic variables
U, X, Y = sp.symbols('U, X, Y')

# symbolic parameters
LA, CX, CY = sp.symbols('lambda, cx, cy', constants=True)
# parameters for the energy equilibrium
ALPHA = sp.symbols('alpha', constants=True)
S_1, S_2, S_3 = sp.symbols(
    's1, s2, s3',
    constants=True
)

# numerical parameters
la = 1.                         # velocity of the scheme
s_1, s_2, s_3 = 1.9, 1.5, 1.0   # relaxation parameters
c_x, c_y = 0.5, 0.25            # velocity of the advection equation
alpha = .8                      # equilibrium parameter for energy

dico = {
    'dim': 2,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': list(range(5)),
            'conserved_moments': U,
            'polynomials': [1, X, Y, X**2+Y**2, X**2-Y**2],
            'relaxation_parameters': [0, S_1, S_1, S_2, S_3],
            'equilibrium': [
                U, CX*U, CY*U,
                (ALPHA*LA**2 + (1-ALPHA)*(CX**2+CY**2))*U,
                (CX**2-CY**2)*U
            ],
        },
    ],
    'parameters': {
        LA: la,
        S_1: s_1,
        S_2: s_2,
        S_3: s_3,
        CX: c_x,
        CY: c_y,
        ALPHA: alpha,
    },
    'relative_velocity': [CX, CY],
}

scheme = pylbm.Scheme(dico)
stab = pylbm.Stability(scheme)
stab.visualize({
    'number_of_wave_vectors': 4096,
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
        S_3: {
            'name': r"$s_3$",
            'range': [0, 2],
            'init': s_2,
            'step': 0.01,
        },
    },
})
