

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
SIGMA_1, SIGMA_2, SIGMA_3 = sp.symbols(
    'sigma_1, sigma_2, sigma_3',
    constants=True
)
symb_s1 = 1/(.5+SIGMA_1)  # symbolic relaxation parameter
symb_s2 = 1/(.5+SIGMA_2)  # symbolic relaxation parameter
symb_s3 = 1/(.5+SIGMA_3)  # symbolic relaxation parameter

# numerical parameters
la = 1.                         # velocity of the scheme
s_1, s_2, s_3 = 1.8, 1.5, 1.    # relaxation parameters
c_x, c_y = 0.5, 0.25            # velocity of the advection equation
alpha = 1.                      # equilibrium parameter for energy

dico = {
    'dim': 2,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': list(range(5)),
            'conserved_moments': U,
            'polynomials': [1, X, Y, X**2+Y**2, X**2-Y**2],
            'relaxation_parameters': [0, symb_s1, symb_s1, symb_s2, symb_s3],
            'equilibrium': [
                U, CX*U, CY*U,
                (ALPHA*LA**2 + (1-ALPHA)*(CX**2+CY**2))*U, 0
            ],
        },
    ],
    'parameters': {
        LA: la,
        SIGMA_1: 1/s_1-.5,
        SIGMA_2: 1/s_2-.5,
        SIGMA_3: 1/s_3-.5,
        CX: c_x,
        CY: c_y,
        ALPHA: alpha,
    },
    # 'relative_velocity': [CX, CY],
}

scheme = pylbm.Scheme(dico, formal=True)
stab = pylbm.Stability(scheme)
stab.visualize({'number_of_wave_vectors': 4096})
