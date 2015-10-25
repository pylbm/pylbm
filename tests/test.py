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

import pyLBM
import pyLBM.geometry as pyLBMGeom
import pyLBM.simulation as pyLBMSimu
import pyLBM.domain as pyLBMDom
import pyLBM.scheme as pyLBMScheme

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in range(25)] for i in range(10)]

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = -1., 1.
    dx = 0.01 # spatial step
    la = 1. # velocity of the scheme
    ug, ud = 0.5, 0. # left and right state
    Tf = 1.
    NbImages = 2

    dicoQ2 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':2*dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,1],
                 'polynomials':Matrix([1,LA*X]),
                 'relaxation_parameters':[0.,1.0],
                 'equilibrium':Matrix([u[0][0], 0.5*u[0][0]]),
                 'init':{0:Riemann_pb},
                 'init_args':{0:(ug, ud)}
                 }
              }

    dicoQ3 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,0,1],
                 'polynomials':Matrix([1,LA*X,LA*LA*X**2/2]),
                 'relaxation_parameters':[0.,1.9,1.9],
                 'equilibrium':Matrix([u[0][0], 0.25*u[0][0], LA**2/2*u[0][0]]),
                 'init':{0:Riemann_pb},
                 'init_args':{0:(ug, ud)}
                 }
              }

    geom2 = pyLBMGeom.Geometry(dicoQ2)
    geom3 = pyLBMGeom.Geometry(dicoQ3)
    sol3 = pyLBMSimu.Simulation(dicoQ3, geom3)
    sol2 = pyLBMSimu.Simulation(dicoQ2, geom2)

    fig = plt.figure(0,figsize=(8, 8))
    fig.clf()
    plt.ion()
    plt.hold(True)
    plt.plot(sol3.Domain.x[0][1:-1], sol3.m[0][2][1:-1], 'r')
    #plt.plot(sol2.Domain.x[0][1:-1], sol2.m[0][0][1:-1], 'b')
    plt.hold(False)
    plt.title("Solution at t={0:.3f}".format(sol3.t),fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    compt = 0
    Ncompt = int(Tf/(NbImages*sol3.dt))
    im = 0
    while (sol3.t<=Tf):
        compt += 1
        #sol2.one_time_step()
        sol3.one_time_step()
        if (compt%Ncompt==0):
            im += 1
            fig.clf()
            plt.hold(True)
            plt.plot(sol3.Domain.x[0][1:-1], sol3.m[0][2][1:-1], 'r')
            #plt.plot(sol2.Domain.x[0][1:-1], sol2.m[0][0][1:-1], 'b')
            plt.hold(False)
            plt.title("Solution at t={0:.3f}".format(sol3.t), fontsize=14)
            plt.draw()
            plt.pause(1.e-3)
    plt.ioff()
    plt.show()
