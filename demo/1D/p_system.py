from __future__ import print_function
from __future__ import division
"""
 Solver D1Q2Q2 for the p-system on [0, 1]

 d_t(ua) - d_x(ub)    = 0, t > 0, 0 < x < 1,
 d_t(ub) - d_x(p(ua)) = 0, t > 0, 0 < x < 1,
 ua(t=0,x) = ua0(x), ub(t=0,x) = ub0(x),
 d_t(ua)(t,x=0) = d_t(ua)(t,x=1) = 0
 d_t(ub)(t,x=0) = d_t(ub)(t,x=1) = 0

 the initial condition is a picewise constant function
 in order to visualize the simulation of elementary waves

 test: True
"""

import sympy as sp
import numpy as np
import pylbm

ua, ub, X, LA = sp.symbols('ua, ub, X, LA')

def Riemann_pb(x, xmin, xmax, uL, uR):
    xm = 0.5*(xmin+xmax)
    u = np.empty(x.shape)
    u[x < xm] = uL
    u[x == xm] = .5*(uL+uR)
    u[x > xm] = uR
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
    gamma = 2./3.        # exponent in the p-function
    xmin, xmax = 0., 1.  # bounds of the domain
    la = 2.              # velocity of the scheme
    s = 1.7              # relaxation parameter

    uaL, uaR, ubL, ubR = 1.50, 1.25, 1.50, 1.00
    ymina, ymaxa, yminb, ymaxb = 1., 1.75, 1., 1.5

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':ua,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[ua, -ub],
                'init':{ua:(Riemann_pb, (xmin, xmax, uaL, uaR))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':ub,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[ub, ua**(-gamma)],
                'init':{ub:(Riemann_pb, (xmin, xmax, ubL, ubR))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Neumann, 1: pylbm.bc.Neumann}},
        },
        'generator': generator,
        'parameters':{LA:la},
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    if withPlot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig(2, 1)
        ax1 = fig[0]
        ax1.axis(xmin, xmax, .9*ymina, 1.1*ymaxa)
        ax2 = fig[1]
        ax2.axis(xmin, xmax, .9*yminb, 1.1*ymaxb)

        x = sol.domain.x
        l1 = ax1.plot(x, sol.m[ua])[0]
        l2 = ax2.plot(x, sol.m[ub])[0]

        def update(iframe):
            if sol.t<Tf:
                sol.one_time_step()
                l1.set_data(x, sol.m[ua])
                l2.set_data(x, sol.m[ub])
                ax1.title = r'$u_a$ at $t = {0:f}$'.format(sol.t)
                ax2.title = r'$u_b$ at $t = {0:f}$'.format(sol.t)

        fig.animate(update)
        fig.show()
    else:
        while sol.t < Tf:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./256
    Tf = .25
    run(dx, Tf)
