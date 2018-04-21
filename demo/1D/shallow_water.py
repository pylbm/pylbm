from __future__ import print_function
from __future__ import division
"""
 Solver D1Q2Q2 for the shallow water system on [0, 1]

 d_t(h) + d_x(q)    = 0, t > 0, 0 < x < 1,
 d_t(q) + d_x(q^2/h+gh^2/2) = 0, t > 0, 0 < x < 1,
 h(t=0,x) = h0(x), q(t=0,x) = q0(x),
 d_t(h)(t,x=0) = d_t(h)(t,x=1) = 0
 d_t(q)(t,x=0) = d_t(q)(t,x=1) = 0

 the initial condition is a picewise constant function
 in order to visualize the simulation of elementary waves

 test: True
"""
import sympy as sp
import numpy as np
import pylbm

h, q, X, LA, g = sp.symbols('h, q, X, LA, g')

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
    xmin, xmax = 0., 1.  # bounds of the domain
    la = 2.              # velocity of the scheme
    s = 1.5              # relaxation parameter

    hL, hR, qL, qR = 1., .25, 0.10, 0.10
    ymina, ymaxa, yminb, ymaxb = 0., 1., 0., .5

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':h,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[h, q],
                'init':{h:(Riemann_pb, (xmin, xmax, hL, hR))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':q,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[q, q**2/h+.5*g*h**2],
                'init':{q:(Riemann_pb, (xmin, xmax, qL, qR))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Neumann, 1: pylbm.bc.Neumann}},
        },
        'generator': generator,
        'parameters':{LA:la, g:1.},
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
        l1 = ax1.plot(x, sol.m[h], color='b')[0]
        l2 = ax2.plot(x, sol.m[q], color='r')[0]

        def update(iframe):
            if sol.t<Tf:
                sol.one_time_step()
                l1.set_data(x, sol.m[h])
                l2.set_data(x, sol.m[q])
                ax1.title = r'$h$ at $t = {0:f}$'.format(sol.t)
                ax2.title = r'$q$ at $t = {0:f}$'.format(sol.t)

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
