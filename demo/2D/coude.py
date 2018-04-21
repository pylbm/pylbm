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

def bc_in(f, m, x, y, rhoo, uo, ymin, ymax):
    m[rho] = rhoo
    m[qx] = rhoo*uo * (ymax-y)*(y-0.75*(ymax-ymin))*8**2

def vorticity(sol):
    qx_n = sol.m[qx]
    qy_n = sol.m[qy]
    vort = np.abs(qx_n[1:-1, 2:] - qx_n[1:-1, :-2]
                  - qy_n[2:, 1:-1] + qy_n[:-2, 1:-1])
    return vort.T

def norme_q(sol):
    qx_n = sol.m[qx]
    qy_n = sol.m[qy]
    nv = qx_n**2 + qy_n**2
    return nv.T

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
    xmin, xmax, ymin, ymax = 0., 1., 0., 1
    rayon = 0.25*(xmax-xmin)
    la = 1. # velocity of the scheme
    rhoo = 1.
    uo = 0.05
    mu   = 2.5e-5 #0.00185
    zeta = 10*mu
    dummy = 3.0/(la*rhoo*dx)
    s3 = 1.0/(0.5+zeta*dummy)
    s4 = s3
    s5 = s4
    s6 = s4
    s7 = 1.0/(0.5+mu*dummy)
    s8 = s7
    s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy
    xc = xmin + 0.75*(xmax-xmin)
    yc = ymin + 0.75*(ymax-ymin)

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[2, 0, 1, 0]},
        'elements':[pylbm.Parallelogram((xmin,ymin),(xc,ymin),(xmin,yc), label=0)],
        'scheme_velocity':la,
        'space_step': dx,
        'schemes':[{'velocities':list(range(9)),
                    'polynomials':[1,
                           X, Y,
                           3*(X**2+Y**2)-4,
                           0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                           3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                           X**2-Y**2, X*Y],
                    'relaxation_parameters':s,
                    'equilibrium':[rho, qx, qy,
                                -2*rho + 3*q2,
                                rho - 3*q2,
                                -qx, -qy,
                                qx2 - qy2, qxy],
                    'conserved_moments': [rho, qx, qy],
                    'init': {rho: rhoo, qx: 0., qy: 0.},
        }],
        'boundary_conditions':{
           0:{'method':{0: pylbm.bc.Bouzidi_bounce_back}},
           1:{'method':{0: pylbm.bc.Neumann_y}},
           2:{'method':{0: pylbm.bc.Bouzidi_bounce_back}, 'value':(bc_in, (rhoo, uo, ymin, ymax))}
        },
        'generator': generator,
        'parameters':{'LA':la},
      }

    sol = pylbm.Simulation(dico, sorder=sorder)

    if withPlot:
        # init viewer
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]
        image = ax.image(norme_q, (sol,), cmap='jet', clim=[0, uo**2])
        ax.polygon([[xmin/dx, ymin/dx],[xmin/dx, yc/dx], [xc/dx, yc/dx], [xc/dx, ymin/dx]], 'k')

        def update(iframe):
            nrep = 64
            for i in range(nrep):
                 sol.one_time_step()
            image.set_data(norme_q(sol))
            ax.title = "Solution t={0:f}".format(sol.t)

        # run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < Tf:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./512
    Tf = 1.
    run(dx, Tf)
