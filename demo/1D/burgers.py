from __future__ import print_function
from __future__ import division
"""
 Solver D1Q2 and D1Q3 for the Burger's equation on [-1, 1]

 d_t(u) + d_x(u^2/2) = 0, t > 0, 0 < x < 1,
 u(t=0,x) = u0(x),
 d_t(u)(t,x=0) = d_t(u)(t,x=1) = 0

 the initial condition is a Riemann problem,
 that is a picewise constant function

 u0(x) = uL if x<0, uR if x>0.

 The solution is a shock wave if uL>uR and a linear rarefaction wave if uL<uR

 test: True
"""
from six.moves import range
import numpy as np
import sympy as sp

import pylbm

X, LA, u = sp.symbols('X, LA, u')

def u0(x, xmin, xmax, uL, uR): # initial condition
    xm = 0.5*(xmin+xmax)
    u = np.empty(x.shape)
    u[x < xm] = uL
    u[x == xm] = .5*(uL+uR)
    u[x > xm] = uR
    return u

def solution(t, x, xmin, xmax, uL, uR): # solution
    xm = 0.5*(xmin+xmax)
    u = np.empty(x.shape)
    if uL >= uR or t==0: # shock wave
        xD = xm + .5*t*(uL+uR)
        u[x < xD] = uL
        u[x == xD] = .5*(uL+uR)
        u[x > xD] = uR
    else: # rarefaction wave
        xL = xm + t*uL
        xR = xm + t*uR
        u[x < xL] = uL
        u[x>=xL and x<=xR] = (uL * (xR-x[ind_D]) + uR * (x[ind_D]-xL)) / (xR-xL)
        u[x > xR] = uR
    return u

def run(dx, Tf, generator="numpy", sorder=None, withPlot=True):
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
    xmin, xmax = -1., 1.  # bounds of the domain
    uL =  0.3             # left value
    uR =  0.0             # right value
    L = 0.2               # length of the middle area
    la = 1.               # scheme velocity (la = dx/dt)
    s = 1.8               # relaxation parameter

    # dictionary for the D1Q2
    dico1 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':LA,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':[u],
                'polynomials':[1,LA*X],
                'relaxation_parameters':[0., s],
                'equilibrium':[u, u**2/2],
                'init':{u:(u0, (xmin, xmax, uL, uR))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Neumann}},
        },
        'generator': generator,
        'parameters': {LA: la},
    }
    # dictionary for the D1Q3
    dico2 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':LA,
        'schemes':[
            {
                'velocities':list(range(3)),
                'conserved_moments':u,
                'polynomials':[1,LA*X,LA**2*X**2],
                'relaxation_parameters':[0., s, s],
                'equilibrium':[u, u**2/2, LA**2*u/3 + 2*u**3/9],
                'init':{u:(u0, (xmin, xmax, uL, uR))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Neumann}},
        },
        'generator': generator,
        'parameters': {LA: la},
    }
    # simulation
    sol = pylbm.Simulation(dico1, sorder=sorder) # build the simulation with D1Q2
    title = 'D1Q2'
    # sol = pylbm.Simulation(dico2, sorder=sorder) # build the simulation with D1Q3
    # title = 'D1Q3'

    if withPlot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]
        ymin, ymax = min([uL,uR])-.1*abs(uL-uR), max([uL,uR])+.1*abs(uL-uR)
        ax.axis(xmin, xmax, ymin, ymax)

        x = sol.domain.x
        l = ax.plot(x, sol.m[u], width=1, color='b', label=title)[0]
        le = ax.plot(x, solution(sol.t, x, xmin, xmax, uL, uR), width=1, color='k', label='exact')[0]

        def update(iframe):
            if sol.t<Tf:
                sol.one_time_step()
                l.set_data(x, sol.m[u])
                le.set_data(x, solution(sol.t, x, xmin, xmax, uL, uR))
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
    Tf = .5
    run(dx, Tf)
