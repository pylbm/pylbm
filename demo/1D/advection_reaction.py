from __future__ import print_function
from __future__ import division
"""
 Solver D1Q2 for the advection reaction equation on the 1D-torus

 d_t(u) + c d_x(u) = mu u(1-u), t > 0, 0 < x < 1, (c=1/4)
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1)

 test: True
"""
from six.moves import range
import numpy as np
from scipy import stats
import sympy as sp
import pylbm

t, X, LA, u = sp.symbols('t, X, LA, u')
C, MU = sp.symbols('C, MU')


def u0(x, xmin, xmax):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    return 0.25 + .125/largeur**10 * (x%1-milieu-largeur)**5 * (milieu-x%1-largeur)**5 * (abs(x%1-milieu)<=largeur)

def solution(t, x, xmin, xmax, c, mu):
    dt = np.tanh(0.5*mu*t)
    ui = u0(x - c*t, xmin, xmax)
    return (dt+2*ui-(1-2*ui)*dt)/(2-2*(1-2*ui)*dt)

def run(dt, Tf, generator = 'numpy', sorder=None, withPlot=True):
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
    mu = 1.               # parameter of the source term
    s = 2.                # relaxation parameter
    dx = la*dt
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
            'equilibrium':[u, C*u],
            'source_terms':{u:MU*u*(1-u)},
            'init':{u:(u0,(xmin, xmax))},
        },
        ],
        'generator': generator,
        'parameters': {LA: la, C: c, MU: mu},
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
        l2 = ax.plot(x, solution(sol.t, x, xmin, xmax, c, mu), width=2, color='k', label='exact')[0]

        def update(iframe):
            if sol.t < Tf:                 # time loop
                sol.one_time_step()      # increment the solution of one time step
                l1.set_data(x, sol.m[u])
                l2.set_data(x, solution(sol.t, x, xmin, xmax, c, mu))
                ax.title = 'solution at t = {0:f}'.format(sol.t)
                ax.legend()

        fig.animate(update)
        fig.show()
    else:
        while sol.t < Tf:
            sol.one_time_step()

    sol.time_info()
    return np.linalg.norm(sol.m[u] - solution(sol.t, sol.domain.x, xmin, xmax, c, mu), np.inf)

if __name__ == '__main__':
    Tf = 2.
    run(1./128, 1.)
