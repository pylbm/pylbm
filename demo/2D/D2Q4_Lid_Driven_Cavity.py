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
from pyLBM.elements import *
import pyLBM.geometry as pyLBMGeom
import pyLBM.simulation as pyLBMSimu
import pyLBM.domain as pyLBMDom
import pyLBM.boundary as pyLBMBoundary
#import pyLBM.Scheme as pyLBMScheme

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    xx = x[:,np.newaxis]
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qx(x,y):
    yy = y[np.newaxis,:]
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def plot_vorticite(sol):
    V = sol.m[2][0][2:,1:-1] - sol.m[2][0][0:-2,1:-1] - sol.m[1][0][1:-1,2:] + sol.m[1][0][1:-1,0:-2]
    V /= 2*np.sqrt(V**2+1.e-5)
    V += 0.5
    plt.clf()
    plt.imshow(np.float32(V.transpose()), origin='lower')#, cmap=cm.gray)
    plt.title('Vorticity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

def plot_velocity(sol):
    """
    NV = (sol.m[0][1][1:-1,1:-1]**2 + sol.m[0][2][1:-1,1:-1]**2)
    plt.imshow(np.float32(NV.transpose()), origin='lower', cmap=cm.jet, interpolation='nearest')
    plt.title('Norme of the velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)
    """
    """
    pas = 4
    plt.clf()
    Y, X = np.meshgrid(sol.Domain.x[1][1:-1:pas], sol.Domain.x[0][1:-1:pas])
    #X = sol.Domain.x[0][1:-1:pas]
    #Y = sol.Domain.x[1][1:-1:pas]
    u = sol.m[1][0,1:-1:pas,1:-1:pas]
    v = sol.m[2][0,1:-1:pas,1:-1:pas]
    normu = np.sqrt(sol.m[1][0,1:-1,1:-1]**2+sol.m[2][0,1:-1,1:-1]**2).max()
    nv = u**2+v**2
    plt.quiver(X, Y, u, v, nv, pivot='mid', scale=normu*10)
    plt.title('Velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)
    """
    pas = 1
    plt.clf()
    n = sol.Domain.N[0]/4
    Y, X = np.meshgrid(sol.Domain.x[1][1:n:pas], sol.Domain.x[0][1:n:pas])
    u = sol.m[1][0,1:n:pas,1:n:pas]
    v = sol.m[2][0,1:n:pas,1:n:pas]
    normu = np.sqrt(sol.m[1][0,1:n,1:n]**2+sol.m[2][0,1:n,1:n]**2).max()
    nv = u**2+v**2
    plt.quiver(X, Y, u, v, nv, pivot='mid', scale=normu*10)
    plt.title('First Corner Velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

if __name__ == "__main__":
    # parameters
    dim = 2 # spatial dimension
    dx = 1./256 # spatial step
    la = 1. # velocity of the scheme
    Tf = 100
    Longueur = 1
    Largeur = 1
    rhoo = 1.
    murho = 1.e-3
    mu   = murho#0.00185
    xmin, xmax, ymin, ymax = 0.0, Longueur, 0.0, Largeur
    NbImages = 100 # number of figures
    sigmarho = 2.*murho/(la*dx)
    sigmaq  = 2.*mu/(la*dx)
    srho = [0, 1.0/(sigmarho+0.5), 1.0/(sigmarho+0.5), 1.0/(sigmarho+0.5)]
    sq  = [0, 1.0/(sigmaq+0.5), 1.0/(sigmaq+0.5), 1.0/(sigmaq+0.5)]

    dico_geometry = {'dim':dim,
                     'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,2,0,1]},
                     #'Elements':[]
                     }

    dico_scheme = {'dim':dim,
                   'number_of_schemes':3,
                   'scheme_velocity':la,
                   'init':'moments',
                   0:{'velocities':range(5,9),
                      'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
                      'relaxation_parameters':srho,
                      'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                      'init':{0:initialization_rho}
                      },
                   1:{'velocities':range(5,9),
                      'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
                      'relaxation_parameters':sq,
                      'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + u[0][0]/3, u[1][0]*u[2][0]/u[0][0], 0.]),
                      'init':{0:initialization_qx}
                      },
                   2:{'velocities':range(5,9),
                      'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
                      'relaxation_parameters':sq,
                      'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + u[0][0]/3, 0.]),
                      'init':{0:initialization_qy}
                      }
                    }

    dico   = {'dim':dim,
              'geometry':dico_geometry,
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':3,
              'init':'moments',
              0:{'velocities':range(5,9),
                 'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
                 'relaxation_parameters':srho,
                 'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                 'init':{0:initialization_rho}
                 },
              1:{'velocities':range(5,9),
                 'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
                 'relaxation_parameters':sq,
                 'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + u[0][0]/3, u[1][0]*u[2][0]/u[0][0], 0.]),
                 'init':{0:initialization_qx}
                 },
              2:{'velocities':range(5,9),
                 'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
                 'relaxation_parameters':sq,
                 'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + u[0][0]/3, 0.]),
                 'init':{0:initialization_qy}
                 }
            }
    

    geom = pyLBMGeom.Geometry(dico)
    #print geom
    #geom.visualize()
    dom = pyLBMDom.Domain(geom,dico)
    #pyLBMDom.visualize(dom,opt=1)
    #pyLBMDom.verification(dom)
    sol = pyLBMSimu.Simulation(dico, geom)
    #print sol.Scheme.Code_m2F
    #print sol.Scheme.Code_F2m
    #print sol.Scheme.Code_Transport
    #print sol.Scheme.Code_Relaxation
    
    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    #plot_vorticite(sol)
    plot_velocity(sol)
    #plot_coupe(sol)
  

    #"""
    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0
    while (sol.t<=Tf-0.5*sol.dt):
        sol.Scheme.relaxation(sol.m)
        sol.Scheme.m2f(sol.m, sol.F)
        # boundary conditions
        # mass: Neumann everywhere
        sol.F[0][:, 0,1:-1] = sol.F[0][:, 1,1:-1]
        sol.F[0][:,-1,1:-1] = sol.F[0][:,-2,1:-1]
        sol.F[0][:,1:-1, 0] = sol.F[0][:,1:-1, 1]
        sol.F[0][:,1:-1,-1] = sol.F[0][:,1:-1,-2]
        sol.F[0][:, 0, 0] = sol.F[0][:, 1, 1]
        sol.F[0][:,-1, 0] = sol.F[0][:,-2, 1]
        sol.F[0][:, 0,-1] = sol.F[0][:, 1,-2]
        sol.F[0][:,-1,-1] = sol.F[0][:,-2,-2]
        # momentum in the x direction
        sol.F[1][0, 0,0:-1] = - sol.F[1][2, 1,1:]
        sol.F[1][3, 0,1:]   = - sol.F[1][1, 1,0:-1]
        sol.F[1][1,-1,0:-1] = - sol.F[1][3,-2,1:]
        sol.F[1][2,-1,1:]   = - sol.F[1][0,-2,0:-1]
        sol.F[1][2,1:,-1]   = - sol.F[1][0,0:-1,-2] + 0.5*0.25
        sol.F[1][3,0:-1,-1] = - sol.F[1][1,1:,-2]   + 0.5*0.25
        sol.F[1][1,1:, 0]   = - sol.F[1][3,0:-1, 1]
        sol.F[1][0,0:-1, 0] = - sol.F[1][2,1:, 1]
        # momentum in the y direction
        sol.F[2][0, 0,0:-1] = - sol.F[2][2, 1,1:]
        sol.F[2][3, 0,1:]   = - sol.F[2][1, 1,0:-1]
        sol.F[2][1,-1,0:-1] =   sol.F[2][3,-2,1:]
        sol.F[2][2,-1,1:]   =   sol.F[2][0,-2,0:-1]
        sol.F[2][2,1:,-1]   = - sol.F[2][0,0:-1,-2]
        sol.F[2][3,0:-1,-1] = - sol.F[2][1,1:,-2]
        sol.F[2][1,1:, 0]   = - sol.F[2][3,0:-1, 1]
        sol.F[2][0,0:-1, 0] = - sol.F[2][2,1:, 1]
        sol.Scheme.transport(sol.F)
        sol.Scheme.f2m(sol.F, sol.m)
        sol.t += sol.dt
        print str((int)(100*sol.t/Tf)) + '%'
        compt += 1
        if (compt%Ncompt==0):
            im += 1
            #plot_vorticite(sol)
            plot_velocity(sol)
            #plot_coupe(sol)
        
    plt.ioff()
    plt.show()
    #"""
