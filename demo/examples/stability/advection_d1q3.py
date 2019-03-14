

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
SIGMA_1, SIGMA_2 = sp.symbols('sigma_1, sigma_2', constants=True)
symb_s1 = 1/(.5+SIGMA_1)  # symbolic relaxation parameter
symb_s2 = 1/(.5+SIGMA_2)  # symbolic relaxation parameter

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
            'relaxation_parameters': [0, symb_s1, symb_s2],
            'equilibrium': [U, C*U, (ALPHA*LA**2+(1-ALPHA)*C**2)*U],
        },
    ],
    'parameters': {
        LA: la,
        SIGMA_1: 1/s_1-.5,
        SIGMA_2: 1/s_2-.5,
        C: c,
        ALPHA: alpha,
    },
    # 'relative_velocity': [C],
}

scheme = pylbm.Scheme(dico, formal=True)
stab = pylbm.Stability(scheme)

stab.visualize()
