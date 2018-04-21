from __future__ import print_function
from __future__ import division
"""
test: True
"""
from six.moves import range
import numpy as np
import sympy as sp

import pylbm

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_in(f, m, x, y, width, height, max_velocity, grad_pressure):
    m[rho] = (x-0.5*width) * grad_pressure
    m[qx] = max_velocity * (1. - 4.*y**2/height**2)

def bc_out(f, m, x, y, width, grad_pressure):
    m[rho] = (x-0.5*width) * grad_pressure


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
    la = 1. # velocity of the scheme
    width = 2
    height = 1
    max_velocity = 0.1
    rhoo = 1.
    mu   = 0.00185
    zeta = 1.e-2
    xmin, xmax, ymin, ymax = 0.0, width, -0.5*height, 0.5*height
    grad_pressure = - max_velocity * 8.0 / (height)**2 * 3.0/(la**2*rhoo) * mu
    dummy = 3.0/(la*rhoo*dx)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)
    s  = [0.,0.,0.,s1,s1,s1,s1,s2,s2]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[2, 1, 0, 0]},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{'velocities':list(range(9)),
                    'polynomials':[1,
                             LA*X, LA*Y,
                             3*(X**2+Y**2)-4,
                             0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                             3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                             X**2-Y**2, X*Y],
                    'relaxation_parameters':s,
                    'equilibrium':[rho,
                              qx, qy,
                              -2*rho + 3*q2,
                              rho - 3*q2,
                              -qx/LA, -qy/LA,
                              qx2 - qy2, qxy],
                    'conserved_moments': [rho, qx, qy],
                    'init':{rho: 1.,
                            qx: 0.,
                            qy: 0.
                            },
                    }],
        'parameters':{'LA':la},
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Bouzidi_bounce_back}},
            1:{'method':{0: pylbm.bc.Neumann_x}},
            2:{'method':{0: pylbm.bc.Bouzidi_bounce_back},
               'value':(bc_in, (width, height, max_velocity, grad_pressure))}
        },
        'generator': generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    if withPlot:
        # init viewer
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]

        nt = int(sol.domain.shape_in[0]/2)
        y = sol.domain.y

        l1 = ax.plot(y, sol.m[qx][nt], color='r', marker='+', label='LBM')[0]
        l2 = ax.plot(y, rhoo*max_velocity * (1.-4.*y**2/height**2), color='k', label='exact')
        ax.title = 'Velocity at t = {0:f}'.format(sol.t)
        ax.legend()

        def update(iframe):
            sol.one_time_step()
            l1.set_data(y, sol.m[qx][nt])
            ax.title = 'Velocity at t = {0:f}'.format(sol.t)

        # run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < Tf:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 20
    run(dx, Tf)
