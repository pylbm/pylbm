from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a vectorial D2Q444 for shallow water
"""
import sympy as sp
import pylbm

X, Y, LA, g = sp.symbols('X, Y, LA, g')
h, qx, qy = sp.symbols('h, qx, qy')

# parameters
la = 4 # velocity of the scheme
s_h  = [0., 2.,  2.,  1.5]
s_q  = [0., 1.5, 1.5, 1.2]

vitesse = [1,2,3,4]
polynomes = [1, LA*X, LA*Y, X**2-Y**2]

d = {
    'dim': 2,
    'scheme_velocity': la,
    'schemes':[
        {
            'velocities': vitesse,
            'conserved_moments': h,
            'polynomials': polynomes,
            'relaxation_parameters': s_h,
            'equilibrium': [h, qx, qy, 0.],
        },
        {
            'velocities': vitesse,
            'conserved_moments': qx,
            'polynomials': polynomes,
            'relaxation_parameters': s_q,
            'equilibrium': [qx, qx**2/h + 0.5*g*h**2, qx*qy/h, 0.],
        },
        {
            'velocities': vitesse,
            'conserved_moments': qy,
            'polynomials': polynomes,
            'relaxation_parameters': s_q,
            'equilibrium': [qy, qy*qx/h, qy**2/h + 0.5*g*h**2, 0.],
        },
    ],
    'parameters': {LA: la, g: 1.},
}

s = pylbm.Scheme(d)
print(s)
