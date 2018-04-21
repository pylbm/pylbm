from __future__ import print_function
from __future__ import division
"""
 Solver D1Q2 for the advection equation on the 1D-torus

 d_t(u) + c d_x(u) = 0, t > 0, 0 < x < 1, (c=1/4)
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1)

 the solution is
 u(t,x) = u0(x-ct).

 test: True
"""
import numpy as np
import sympy as sp
import pylbm

X, LA, u = sp.symbols('X, LA, u')

def u0(x, xmin, xmax):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    return 1.0/largeur**10 * (x%1-milieu-largeur)**5 * (milieu-x%1-largeur)**5 * (abs(x%1-milieu)<=largeur)

def run(dx, Tf, generator="numpy", sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    store: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    xmin, xmax = 0., 1.   # bounds of the domain
    la = 1.               # scheme velocity (la = dx/dt)
    c = 0.25              # velocity of the advection
    s = 1.99              # relaxation parameter

    # dictionary of the simulation
    dico = {
        'box':{'x':[xmin, xmax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':LA,
        'schemes':[
        {
            'velocities':[1,2],
            'conserved_moments':u,
            'polynomials':[1,LA*X],
            'relaxation_parameters':[0., s],
            'equilibrium':[u, c*u],
            'init':{u:(u0,(xmin, xmax))},
        },
        ],
        'generator': generator,
        'parameters': {LA: la},
    }

    # simulation
    sol = pylbm.Simulation(dico, sorder=sorder) # build the simulation

    if withPlot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]
        ymin, ymax = -.2, 1.2
        ax.axis(xmin, xmax, ymin, ymax)

        x = sol.domain.x
        l1 = ax.plot(x, sol.m[u], width=2, color='b', label='D1Q2')[0]
        l2 = ax.plot(x, u0(x-c*sol.t, xmin, xmax), width=2, color='k', label='exact')[0]

        def update(iframe):
            if sol.t < Tf:                 # time loop
                sol.one_time_step()      # increment the solution of one time step
                l1.set_data(x, sol.m[u])
                l2.set_data(x, u0(x-c*sol.t, xmin, xmax))
                ax.title = 'solution at t = {0:f}'.format(sol.t)
                ax.legend()

        fig.animate(update)
        fig.show()
    else:
        while sol.t < Tf:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 1.
    sol = run(dx, Tf)
