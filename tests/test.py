from __future__ import division
from six.moves import range
import sys
import cmath
from math import pi
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import pylbm

u, X, LA = sp.symbols('u, X, LA')

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = -1., 1.
    dx = 1./64 # spatial step
    la = 1. # velocity of the scheme
    ug, ud = 1., 0. # left and right state
    Tf = 1.

    dicoQ2 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'parameters':{LA:la},
        'space_step':2*dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'conserved_moments':u,
                'velocities':[2,1],
                'polynomials':[1,LA*X],
                'relaxation_parameters':[0.,1.0],
                'equilibrium':[u, 0.5*u],
                'init':{u:(Riemann_pb, (ug, ud))},
            }
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Neumann,},},
        },
    }

    dicoQ3 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'parameters':{LA:la},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'conserved_moments':u,
                'velocities':list(range(3)),
                'polynomials':[1,LA*X, (LA*X)**2/2],
                'relaxation_parameters':[0.,1.9, 1.9],
                'equilibrium':[u, 0.25*u, LA**2/2*u],
                'init':{u:(Riemann_pb, (ug, ud))},
            }
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Neumann,},},
        },
    }

    sol2 = pylbm.Simulation(dicoQ2)
    sol3 = pylbm.Simulation(dicoQ3)

    viewer = pylbm.viewer.matplotlibViewer
    fig = viewer.Fig()
    ax = fig[0]
    ymin, ymax = -.2, 1.2
    ax.axis(xmin, xmax, ymin, ymax)

    x2 = sol2.domain.x[0][1:-1]
    x3 = sol3.domain.x[0][1:-1]
    l1 = ax.plot(x2, sol2.m[u][1:-1], width=2, color='b', label=r'$D_1Q_2$')[0]
    l2 = ax.plot(x3, sol3.m[u][1:-1], width=2, color='r', label=r'$D_1Q_3$')[0]

    def update(iframe):
        if sol2.t < Tf:                 # time loop
            sol2.one_time_step()      # increment the solution of one time step
            l1.set_data(x2, sol2.m[u][1:-1])
            sol3.one_time_step()      # increment the solution of one time step
            l2.set_data(x3, sol3.m[u][1:-1])
            ax.title = 'solution at t = {0:f}'.format(sol2.t)
            ax.legend()

    fig.animate(update)
    fig.show()
