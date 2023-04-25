# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D2Q4 for advection
"""
import sympy as sp
import pylbm

u, X, Y = sp.symbols("u, X, Y")

d = {
    "dim": 2,
    "scheme_velocity": 1.0,
    "schemes": [
        {
            "velocities": [1, 2, 3, 4],
            "conserved_moments": u,
            "polynomials": [1, X, Y, X**2 - Y**2],
            "equilibrium": [u, 0.1 * u, 0.2 * u, 0.0],
            "relaxation_parameters": [0.0, 1.9, 1.9, 1.4],
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
