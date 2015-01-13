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
import pyLBM.Geometry as pyLBMGeom
import pyLBM.Simulation as pyLBMSimu
import pyLBM.Domain as pyLBMDom
import pyLBM.Scheme as pyLBMScheme

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = -1., 1.
    dx = 0.1 # spatial step
    la = 1. # velocity of the scheme
    ug, ud = 0.5, 0. # left and right state
    Tf = 1.
    NbImages = 2
    
    dicoQ2 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,1],
                 'polynomials':Matrix([1,LA*X]),
                 'relaxation_parameters':[0.,1.],
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2]),
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
                 'relaxation_parameters':[0.,1.2,1.9],
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2, LA**2/2*u[0][0]]),
                 'init':{0:Riemann_pb},
                 'init_args':{0:(ug, ud)}
                 }
              }

    geom = pyLBMGeom.Geometry(dicoQ2)
    sol = pyLBMSimu.Simulation(dicoQ2, geom)

    while (sol.t<=Tf):
        sol.one_time_step()
    XQ2 = sol.Domain.x[0][1:-1].copy()
    m0Q2 = sol.m[0][0][1:-1].copy()
    m1Q2 = sol.m[0][1][1:-1] - m0Q2**2/2
    F0Q2 = sol.F[0][0][1:-1].copy()
    F1Q2 = sol.F[0][1][1:-1].copy()

    geom = pyLBMGeom.Geometry(dicoQ3)
    sol1 = pyLBMSimu.Simulation(dicoQ3, geom)

    while (sol1.t<=Tf):
        sol1.one_time_step()
    XQ3 = sol1.Domain.x[0][1:-1].copy()
    m0Q3 = sol1.m[0][0][1:-1].copy()
    m1Q3 = sol1.m[0][1][1:-1] - m0Q3**2/2
    m2Q3 = sol1.m[0][2][1:-1] - la**2/2*m0Q3
    F0Q3 = sol1.F[0][0][1:-1].copy()
    F1Q3 = sol1.F[0][1][1:-1].copy()
    F2Q3 = sol1.F[0][2][1:-1].copy()

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    #plt.ion()
    #plt.subplot(231)
    plt.hold(True)
    #plt.plot(XQ2, m0Q2, 'r+')
    #plt.plot(XQ3, m0Q3, 'bo')
    plt.plot(m0Q2-m0Q3, 'bo')
    plt.hold(False)
    print XQ2, XQ3
    plt.title("m1 at t={0:.3f}".format(Tf),fontsize=14)
    #plt.subplot(232)
    #plt.plot(XQ2, m1Q2, 'r', XQ3, m1Q3, 'b')
    #plt.title("m2 at t={0:.3f}".format(Tf),fontsize=14)
    #plt.subplot(233)
    #plt.plot(XQ3, m2Q3, 'b')
    #plt.title("m3 at t={0:.3f}".format(Tf),fontsize=14)
    #plt.subplot(234)
    #plt.plot(XQ2, F0Q2, 'r', XQ3, F0Q3, 'b')
    #plt.title("F1 at t={0:.3f}".format(Tf),fontsize=14)
    #plt.subplot(235)
    #plt.plot(XQ2, F1Q2, 'r', XQ3, F1Q3, 'b')
    #plt.title("F2 at t={0:.3f}".format(Tf),fontsize=14)
    #plt.subplot(236)
    #plt.plot(XQ3, F2Q3, 'b')
    #plt.title("F3 at t={0:.3f}".format(Tf),fontsize=14)
    plt.draw()
    plt.ioff()
    plt.show()
    

    """
    geom3 = pyLBMGeom.Geometry(dicoQ3)
    sol3 = pyLBMSimu.Simulation(dicoQ3, geom3)
    geom2 = pyLBMGeom.Geometry(dicoQ2)
    sol2 = pyLBMSimu.Simulation(dicoQ2, geom2)

    fig = plt.figure(0,figsize=(8, 8))
    fig.clf()
    plt.ion()
    plt.plot(sol3.Domain.x[0][1:-1], sol3.m[0][0][1:-1], 'r', sol2.Domain.x[0][1:-1], sol2.m[0][0][1:-1], 'b')
    plt.title("Solution at t={0:.3f}".format(sol2.t),fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol2.dt))
    im = 0
    while (sol2.t<=Tf):
        compt += 1
        sol2.one_time_step()
        sol3.one_time_step()
        print sol2.m[0]
        print sol3.m[0]
        if (compt%Ncompt==0):
            im += 1
            fig.clf()
            plt.plot(sol3.Domain.x[0][1:-1], sol3.m[0][0][1:-1], 'r', sol2.Domain.x[0][1:-1], sol2.m[0][0][1:-1], 'b')
            plt.title("Solution at t={0:.3f}".format(sol2.t), fontsize=14)
            plt.draw()
            plt.pause(1.e-3)
    plt.ioff()
    plt.show()
    """
