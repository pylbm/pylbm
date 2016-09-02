from __future__ import print_function, division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
D2Q5 to solve the stationary state of the heat equation in 2D
"""
from six.moves import range
import numpy as np
import sympy as sp
import pyLBM

u, X, Y = sp.symbols('u, X, Y')

# parameters
xmin, xmax, ymin, ymax = 0., 1., 0., 1.
N = 256
mu = 1.
Tf = .1
dx = (xmax-xmin)/N # spatial step
la = 1.
s1 = 2./(1+4*mu)
s2 = 1.

xm, ym = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
xl, xr = 0.75*xmin+0.25*xmax, 0.25*xmin+0.75*xmax
e, l, L = 0.01, 0.1, xmax-xmin
yw1, yw2, dxw = 0.75*ymin+0.25*ymax, 0.25*ymin+0.75*ymax, 0.15
stove1 = pyLBM.Parallelogram([xm-l, ymin + e], [2*l, 0.], [0.,  e], label=1)
stove2 = pyLBM.Parallelogram([xl-l, ymax - e], [2*l, 0.], [0., -e], label=1)
stove3 = pyLBM.Parallelogram([xr-l, ymax - e], [2*l, 0.], [0., -e], label=1)
wall1 = pyLBM.Parallelogram([xmin, ym-e], [ 0.3*L, 0.], [0., 2*e], label=0)
wall2 = pyLBM.Parallelogram([xmax, ym-e], [-0.3*L, 0.], [0., 2*e], label=0)
wall3 = pyLBM.Parallelogram([xm-0.1*L, ym-e], [0.2*L, 0.], [0., 2*e], label=0)
wall4 = pyLBM.Parallelogram([xm-e, ym], [2*e, 0.], [0., 0.5*(ymax-ymin)], label=0)
window1 = pyLBM.Parallelogram([xmin, yw1-dxw], [e, 0], [0, 2*dxw], label=2)
window2 = pyLBM.Parallelogram([xmax, yw1-dxw], [-e, 0], [0, 2*dxw], label=2)
window3 = pyLBM.Parallelogram([xmin, yw2-dxw], [e, 0], [0, 2*dxw], label=2)
window4 = pyLBM.Parallelogram([xmax, yw2-dxw], [-e, 0], [0, 2*dxw], label=2)

def bc_stove(f, m, x, y):
    m[u] = 1.

dico = {
    'box': {'x': [xmin, xmax], 'y': [ymin, ymax], 'label': 0},
    'elements':[
        stove1, stove2, stove3,
        wall1, wall2, wall3, wall4,
        window1, window2, window3, window4
    ],
    'space_step': dx,
    'scheme_velocity': la,
    'schemes': [
        {
            'velocities': list(range(5)),
            'conserved_moments': u,
            'polynomials': [1, X, Y, (X**2+Y**2)/2, (X**2-Y**2)/2],
            'equilibrium': [u, 0., 0., .5*u, 0.],
            'relaxation_parameters': [0., s1, s1, s2, s2],
            'init': {u: 0.},
        }
    ],
    'boundary_conditions': {
        0: {'method': {0: pyLBM.bc.Neumann,}, 'value': None},
        1: {'method': {0: pyLBM.bc.Bouzidi_anti_bounce_back,}, 'value': bc_stove},
        2: {'method': {0: pyLBM.bc.Bouzidi_anti_bounce_back,}, 'value': None},
    },
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)

x = sol.domain.x
y = sol.domain.y

viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()

ax = fig[0]
image = ax.image(sol.m[u].T, cmap='cubehelix', clim=[0, 1.])

def update(iframe):
    nrep = 32
    for i in range(nrep):
         sol.one_time_step()
    image.set_data(sol.m[u].T)
    ax.title = "Solution t={0:f}".format(sol.t)

# run the simulation
fig.animate(update, interval=1)
fig.show()
