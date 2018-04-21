from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a D1Q3 for the wave equation
"""
import numpy as np
import sympy as sp
import pylbm
u, v, X, c, LA = sp.symbols('u, v, X, c, LA')

def init_u(x):
    return np.sin(2*np.pi/3*x)

def bc_in(f, m, x):
    m[u] = 1.

d = {
    'box': {'x': [0., 3.], 'label': 0},
    'scheme_velocity':LA,
    'space_step': 0.01,
    'parameters': {LA: 1., c: .5},
    'schemes':[
        {
            'velocities': [0, 1, 2],
            'conserved_moments':[u, v],
            'polynomials': [1, X, 0.5*X**2],
            'equilibrium': [u, v, .5*c**2*u],
            'relaxation_parameters': [0., 0., 1.9],
            'init': {v: (init_u,), u: 0.},
        },
    ],
    'boundary_conditions':{
        0:{'method': {0: pylbm.bc.bounce_back}, 'value': None},
    },
}
#s = pylbm.Scheme(d)

"""
import matplotlib.pyplot as plt
sol = pylbm.Simulation(d)

viewer = pylbm.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]
xmin, xmax, ymin, ymax = 0., 3., -2.2, 2.2
ax.axis(xmin, xmax, ymin, ymax)

x = sol.domain.x
l1 = ax.plot(x, sol.m[u], width=2, color='b', label='u')[0]
l2 = ax.plot(x, sol.m[v], width=2, color='r', label='v')[0]

def update(iframe):
    if sol.t < 5.:                 # time loop
        sol.one_time_step()      # increment the solution of one time step
        l1.set_data(x, sol.m[u])
        l2.set_data(x, sol.m[v])
        ax.title = 'solution at t = {0:f}'.format(sol.t)
        ax.legend()

fig.animate(update)
fig.show()
"""
