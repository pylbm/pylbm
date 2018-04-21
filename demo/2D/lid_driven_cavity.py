from __future__ import print_function
from __future__ import division
"""
test: True
"""
from six.moves import range
import numpy as np
import sympy as sp
import mpi4py.MPI as mpi
import pylbm

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_up(f, m, x, y, driven_velocity):
    m[qx] = driven_velocity

def vorticity(sol):
    #sol.f2m()
    qx_n = sol.m[qx]
    qy_n = sol.m[qy]
    vort = np.abs(qx_n[1:-1, 2:] - qx_n[1:-1, :-2]
                  - qy_n[2:, 1:-1] + qy_n[:-2, 1:-1])
    return vort.T

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
    la = 1.
    rhoo = 1.
    mu   = 1.e-4
    zeta = 1.e-4
    driven_velocity = 0.2 # velocity of the upper border
    dummy = 3.0/dx
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)
    s  = [0.,0.,0.,s1,s1,s1,s1,s2,s2]
    Tf = 10.
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy

    lid_cavity = {
        'parameters':{LA: la},
        'box':{'x':[0., 1.], 'y':[0., 1.], 'label':[0, 0, 0, 1]},
        'space_step': dx,
        'scheme_velocity':LA,
        'schemes':[
            {
                'velocities':list(range(9)),
                'polynomials':[
                    1, LA*X, LA*Y,
                    3*(X**2+Y**2)-4,
                    0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                    3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                    X**2-Y**2, X*Y
                ],
                'relaxation_parameters':s,
                'equilibrium':[
                    rho,
                    qx, qy,
                    -2*rho + 3*q2,
                    rho - 3*q2,
                    -qx/LA, -qy/LA,
                    qx2 - qy2, qxy
                ],
                'conserved_moments': [rho, qx, qy],
                'init': {rho: 1., qx: 0., qy: 0.},
            },
        ],
        #'relative_velocity': [qx/rho, qy/rho],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Bouzidi_bounce_back}},
            1:{'method':{0: pylbm.bc.Bouzidi_bounce_back}, 'value':(bc_up, (driven_velocity,))}
        },
        'generator': generator,
        'show_code': True,
    }

    sol = pylbm.Simulation(lid_cavity, sorder=sorder)

    if withPlot:
        # init viewer
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]
        image = ax.image(vorticity, (sol,), cmap='jet', clim=[0, .1])

        def update(iframe):
            nrep = 100
            for i in range(nrep):
                sol.one_time_step()

            image.set_data(vorticity(sol))
            ax.title = "Solution t={0:f}".format(sol.t)

        # run the simulation
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
