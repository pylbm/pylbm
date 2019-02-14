

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 4 velocities scheme for the advection equation in 2D
"""
import sympy as sp
import pylbm

# pylint: disable=invalid-name

U, X, Y = sp.symbols('u, X, Y')
CX, CY = sp.symbols('c_0, c_1', constants=True)
LA, SIGMA = sp.symbols('lambda, sigma', constants=True)

s = 1/(.5+SIGMA)
s_num = 1.9

scheme_cfg = {
    'parameters': {
        LA: 1.,
        CX: 0.1,
        CY: 0.,
        SIGMA: 1/s_num - .5,
    },
    'dim': 2,
    'scheme_velocity': LA,
    'schemes': [
        {
            'velocities': [1, 2, 3, 4],
            'conserved_moments': U,
            'polynomials': [1, X, Y, X**2-Y**2],
            'equilibrium': [U, CX*U, CY*U, 0.],
            'relaxation_parameters': [0., s, s, 1.],
        },
    ],
}

scheme = pylbm.Scheme(scheme_cfg)
eq_pde = pylbm.EquivalentEquation(scheme)

print(eq_pde)
