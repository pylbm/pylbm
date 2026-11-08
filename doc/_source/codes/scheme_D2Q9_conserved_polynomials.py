# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D2Q9 TRT scheme for the Stokes equations with a body force,
whose conserved moments are not rows of the basis
"""

import sympy as sp
import pylbm

rho, qx, qy, X, Y = sp.symbols("rho, qx, qy, X, Y")

s_plus, s_minus = 1.2, 1.7  # TRT relaxation parameters
fx = 1.0e-5  # body force along x

E = 3 * (X**2 + Y**2) - 4
QX = 3 * X * (X**2 + Y**2) - 5 * X
QY = 3 * Y * (X**2 + Y**2) - 5 * Y

d = {
    "dim": 2,
    "scheme_velocity": 1.0,
    "schemes": [
        {
            "velocities": list(range(9)),
            "conserved_moments": [rho, qx, qy],
            "conserved_polynomials": [1, X, Y],
            "polynomials": [
                1 + E,
                X + QX,
                Y + QY,
                E,
                (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) + 8) / 2,
                QX,
                QY,
                X**2 - Y**2,
                X * Y,
            ],
            "relaxation_parameters": [
                s_plus,
                s_minus,
                s_minus,
                s_plus,
                s_plus,
                s_minus,
                s_minus,
                s_plus,
                s_plus,
            ],
            "equilibrium": [
                -rho,
                0.0,
                0.0,
                -2 * rho,
                rho,
                -(qx + fx / s_minus),
                -qy,
                0.0,
                0.0,
            ],
        },
    ],
}
s = pylbm.Scheme(d)
print(s)
