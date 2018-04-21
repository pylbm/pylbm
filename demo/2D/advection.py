from __future__ import print_function
from __future__ import division
"""
 Solver D2Q4 for the advection equation on the 2D-torus

 d_t(u) + cx d_x(u) + cy d_y(u) = 0, t > 0, 0 < x,y < 1,
 u(t=0,x,y) = u0(x,y),
 u(t,x=0,y) = u(t,x=1,y) 0 < y < 1,
 u(t,x,y=0) = u(t,x,y=1) 0 < x < 1,

 the solution is
 u(t,x,y) = u0(x-cx*t,y-cy*t)

 test: True
"""
import numpy as np
import sympy as sp
from six.moves import range

import pylbm

u, X, Y, LA = sp.symbols('u, X, Y, LA')

compt = 0

def u0(x, y, xmin, xmax, ymin, ymax):
    return np.ones((x.shape[0], y.shape[0]), dtype='float64') \
           + .5 * ((x-0.25*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.01)

def run(dx, Tf, generator="cython", sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    sorder: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    xmin, xmax, ymin, ymax = 0., 1., 0., 1. # bounds of the domain
    cx, cy = 0.2, 0.5                       # velocity of the advection
    la = 2.                                 # scheme velocity
    sigma_qx = 1./np.sqrt(12)
    sigma_xy = sigma_qx
    s_qx = 1./(0.5+sigma_qx)
    s_xy = 1./(0.5+sigma_xy)
    s  = [0., s_qx, s_qx, s_xy]             # relaxation parameters

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':LA,
        'schemes':[
            {
                'velocities':list(range(1,5)),
                'conserved_moments':u,
                'polynomials':[1, X, Y, X**2-Y**2],
                'relaxation_parameters':s,
                'equilibrium':[u, cx*u, cy*u, 0],
                'init':{u:(u0,(xmin, xmax, ymin, ymax))},
            },
        ],
        'generator': generator,
        'parameters':{LA:la},
        #'relative_velocity': [cx, cy],
        }

    sol = pylbm.Simulation(dico, sorder=sorder)

    if withPlot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]
        im = ax.image(sol.m[u].transpose())
        ax.title = 'solution at t = {0:f}'.format(sol.t)

        def update(iframe):
            nrep = 128
            for i in range(nrep):
                sol.one_time_step()
            im.set_data(sol.m[u].transpose())
            ax.title = 'solution at t = {0:f}'.format(sol.t)

        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < Tf:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./256
    Tf = 10.
    run(dx, Tf)
