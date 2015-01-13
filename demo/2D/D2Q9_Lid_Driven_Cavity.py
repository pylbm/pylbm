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
import pyLBM.stencil as pyLBMSten
import pyLBM.domain as pyLBMDom
import pyLBM.scheme as pyLBMScheme
import pyLBM.simulation as pyLBMSimu
import pyLBM.boundary as pyLBMBound

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qx(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def bc_up(f, m, x, y, scheme):
    m[0][0] = 0.
    m[0][1] = rhoo * driven_velocity
    m[0][2] = 0.
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def plot_quiver(sol):
    pas = 4
    plt.clf()
    X, Y = np.meshgrid(sol.domain.x[0][1:-1:pas], sol.domain.x[1][1:-1:pas])
    u = sol.m[0][1,1:-1:pas,1:-1:pas].transpose()
    v = sol.m[0][2,1:-1:pas,1:-1:pas].transpose()
    normu = np.sqrt(sol.m[0][1,1:-1,1:-1]**2+sol.m[0][2,1:-1,1:-1]**2).max()
    nv = u**2+v**2
    plt.quiver(X, Y, u, v, nv, pivot='mid', scale=normu*10)
    plt.title('Velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

def plot_stream(sol):
    pas = 1
    plt.clf()
    X, Y = np.meshgrid(sol.domain.x[0][1:-1:pas], sol.domain.x[1][1:-1:pas])
    u = sol.m[0][1,1:-1:pas,1:-1:pas].transpose()
    v = sol.m[0][2,1:-1:pas,1:-1:pas].transpose()
    normu = np.sqrt(u**2+v**2)
    nv = u**2+v**2
    plt.streamplot(X, Y, u, v, color=normu, linewidth=2, cmap=plt.cm.autumn)
    plt.title('Velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

def plot_vorticity(sol):
    V = sol.m[0][2][2:,1:-1] - sol.m[0][2][0:-2,1:-1] - sol.m[0][1][1:-1,2:] + sol.m[0][1][1:-1,0:-2]
    V /= np.sqrt(V**2+1.e-5)
    plt.imshow(np.float32(V.transpose()), origin='lower', cmap=cm.gray)
    plt.title('Vorticity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)


if __name__ == "__main__":
    # parameters
    NbImages = 80 # number of figures
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.
    dx = 1./256 # spatial step
    la = 1. # velocity of the scheme
    Tf = 20
    rhoo = 1.
    mu   = 1.e-4 #0.00185
    zeta = 1.e-4
    driven_velocity = 0.2 # velocity of the upper border
    dummy = 3.0/(la*rhoo*dx)
    s3 = 1.0/(0.5+zeta*dummy)
    s4 = s3
    s5 = s4
    s6 = s4
    s7 = 1.0/(0.5+mu*dummy)
    s8 = s7
    s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,0,1,0]},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        0:{'velocities':range(9),
           'polynomials':Matrix([1,
                                 LA*X, LA*Y,
                                 3*(X**2+Y**2)-4,
                                 0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                                 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                                 X**2-Y**2, X*Y]),
            'relaxation_parameters':s,
            'equilibrium':Matrix([u[0][0],
                                  u[0][1], u[0][2],
                                  -2*u[0][0] + 3*q2,
                                  u[0][0]+1.5*q2,
                                  -u[0][1]/LA, -u[0][2]/LA,
                                  qx2-qy2, qxy]),
        },
        'init':{'type':'moments', 0:{0:(initialization_rho,),
                                     1:(initialization_qx,),
                                     2:(initialization_qy,)
                                     }
        },
        'boundary_conditions':{
            0:{'method':{0: pyLBMBound.bouzidi_bounce_back}, 'value':None},
            1:{'method':{0: pyLBMBound.bouzidi_bounce_back}, 'value':bc_up}
        }
    }


    sol = pyLBMSimu.Simulation(dico)
    #print sol.domain.geom
    #print sol.domain.stencil
    #print sol.domain
    #print sol.scheme
    Re = rhoo*driven_velocity/mu
    print "Reynolds number {0:10.3e}".format(Re)

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plot_quiver(sol)
    #plot_stream(sol)
    #plot_vorticity(sol)

    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0
    while (sol.t<Tf):
        sol.one_time_step()
        compt += 1
        if (compt%Ncompt==0):
            im += 1
            plot_quiver(sol)
            #plot_stream(sol)
            #plot_vorticity(sol)
        
    plt.ioff()
    plt.show()
    
