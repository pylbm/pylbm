import sys
import cmath
from math import pi, sqrt
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

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

if __name__ == "__main__":
    # init values
    try:
        numonde = int(sys.argv[1])
    except:
        numonde = 0
    # parameters
    g = 1. # exponent in the p-function
    dim = 1 # spatial dimension
    xmin, xmax = 0., 1.
    dx = 0.001 # spatial step
    la = 3. # velocity of the scheme
    if (numonde == 0): # 1-shock, 2-shock
        uag, uad, ubg, ubd = 1.9, 2., 2., 1.
    elif (numonde == 1): # 1-shock, 2-rarefaction
        uag, uad, ubg, ubd = 2., 4., 1., -1.
    elif (numonde == 2): # 1-rarefaction, 2-shock
        uag, uad, ubg, ubd = 4., 2., 2., -1.
    elif (numonde == 3): # 1-rarefaction, 2-rarefaction
        uag, uad, ubg, ubd = 3., 3., 1., 2.
    elif (numonde == 4): # 2-shock
        uag, uad, ubd = 1.00, 1.25, 1.25
        (ubg,dummy) = f2(uag, uad, ubd)
    elif (numonde == 5): # 2-rarefaction
        uag, uad, ubd = 1.25, 1.00, 1.25
        (ubg,dummy) = f2(uag, uad, ubd)
    elif (numonde == 6): # 1-shock
        uag, uad, ubg = 1.25, 1.00, 1.25
        (ubd,dummy) = f1(uad, uag, ubg)
    elif (numonde == 7): # 1-rarefaction
        uag, uad, ubg = 1.00, 1.25, 1.25
        (ubd,dummy) = f1(uad, uag, ubg)
    else:
        print "Odd initialization: numonde = " + str(numonde)
        sys.exit()
    Tf = 0.1 # final time
    NbImages = 10 # number of figures
    
    dico_geometry = {'dim':dim,
                     'box':{'x':[xmin, xmax], 'label':[0,0]},
                     'Elements':[]
                     }

    dico = {'dim':dim,
            'Geometry':dico_geometry,
            'space_step':dx,
            'scheme_velocity':la,
            'number_of_schemes':2,
            'init':'moments',
            0:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.5],
               'equilibrium':Matrix([u[0][0], u[1][0]]),
               'init':{0:Riemann_pb},
               'init_args':{0:(uag, uad)}
               },
            1:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.5],
               'equilibrium':Matrix([u[1][0], g*0.5*u[0][0]**2 + u[1][0]**2/u[0][0]]),
               'init':{0:Riemann_pb},
               'init_args':{0:(ubg, ubd)}
               }
            }

    geom = pyLBMGeom.Geometry(dico)
    sol = pyLBMSimu.Simulation(dico, geom)

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plt.subplot(121)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0][1:-1])
    plt.title("Solution mass at t={0:.3f}".format(sol.t),fontsize=14)
    plt.subplot(122)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[1][0][1:-1])
    plt.title("Solution velocity at t={0:.3f}".format(sol.t),fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0
    while (sol.t<=Tf):
        compt += 1
        sol.one_time_step()
        if (compt%Ncompt==0):
            im += 1
            fig.clf()
            plt.subplot(121)
            plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0][1:-1])
            plt.title("Solution mass at t={0:.3f}".format(sol.t), fontsize=14)
            plt.subplot(122)
            plt.plot(sol.Domain.x[0][1:-1],sol.m[1][0][1:-1])
            plt.title("Solution velocity at t={0:.3f}".format(sol.t), fontsize=14)
            plt.draw()
            plt.pause(1.e-3)
    plt.ioff()
    plt.show()
