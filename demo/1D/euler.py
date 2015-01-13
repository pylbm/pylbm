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

def myplot(sol):
    fig.clf()
    plt.subplot(231)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0,1:-1])
    plt.title("Solution mass at t={0:.3f}".format(sol.t),fontsize=14)
    plt.subplot(232)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[1][0,1:-1])
    plt.title("Solution momentum at t={0:.3f}".format(sol.t),fontsize=14)
    plt.subplot(233)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[2][0,1:-1])
    plt.title("Solution energy at t={0:.3f}".format(sol.t),fontsize=14)
    plt.subplot(235)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[1][0,1:-1]/sol.m[0][0,1:-1])
    plt.title("Solution velocity at t={0:.3f}".format(sol.t),fontsize=14)
    plt.subplot(236)
    plt.plot(sol.Domain.x[0][1:-1],(gamma-1.)*sol.m[2][0,1:-1]-sol.m[1][0,1:-1]**2/sol.m[0][0,1:-1])
    plt.title("Solution pressure at t={0:.3f}".format(sol.t),fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    return

if __name__ == "__main__":
    # init values
    try:
        numonde = int(sys.argv[1])
    except:
        numonde = 0
    # parameters
    gamma = 1.4
    dim = 1 # spatial dimension
    xmin, xmax = 0., 1.
    dx = 0.001 # spatial step
    la = 3. # velocity of the scheme
    if (numonde == 0): # Sod tube
        rhog, rhod, pg, pd, ug, ud = 1., 1./8., 1., 0.1, 0., 0.
        uag = rhog
        uad = rhod
        ubg = rhog*ug
        ubd = rhod*ud
        ucg = rhog*ug**2 + pg/(gamma-1.)
        ucd = rhod*ud**2 + pd/(gamma-1.)
    else:
        print "Odd initialization: numonde = " + str(numonde)
        sys.exit()
    Tf = 0.14 # final time
    NbImages = 10 # number of figures
    
    dico = {'dim':dim,
            'box':([xmin, xmax],),
            'space_step':dx,
            'scheme_velocity':la,
            'number_of_schemes':3,
            'init':'moments',
            0:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.9],
               'equilibrium':Matrix([u[0][0], u[1][0]]),
               'init':{0:Riemann_pb},
               'init_args':{0:(uag, uad)}
               },
            1:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.9],
               'equilibrium':Matrix([u[1][0], (gamma-1.)*u[2][0]+0.5*(3.-gamma)*u[1][0]**2/u[0][0]]),
               'init':{0:Riemann_pb},
               'init_args':{0:(ubg, ubd)}
               },
            2:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.9],
               'equilibrium':Matrix([u[2][0], gamma*u[2][0]*u[1][0]/u[0][0]-0.5*(gamma-1.)*u[1][0]**3/u[0][0]**2]),
               'init':{0:Riemann_pb},
               'init_args':{0:(ucg, ucd)}
               }
            }

    geom = pyLBMGeom.Geometry(dico)
    sol = pyLBMSimu.Simulation(dico, geom)

    fig = plt.figure(0,figsize=(16, 8))
    plt.ion()
    myplot(sol)
    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0
    while (sol.t<=Tf):
        compt += 1
        sol.one_time_step()
        if (compt%Ncompt==0):
            im += 1
            myplot(sol)
    plt.ioff()
    plt.show()
