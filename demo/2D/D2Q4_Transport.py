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
import pyLBM.simulation as pyLBMSimu
import pyLBM.boundary as pyLBMBound
import pyLBM.generator as pyLBMGen

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.01)

def plot(sol):
    plt.clf()
    plt.imshow(np.float32(sol.m[0][0].transpose()), origin='lower', cmap=cm.gray)
    plt.title('mass at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

def plot_coupe(sol):
    nx, ny = sol.domain.N
    plt.clf()
    plt.hold(True)    
    t = sol.domain.x[0][1:-1].copy()
    z = sol.m[0][0,1:-1,ny/2+1].copy()
    plt.plot(t, z)
    t = sol.domain.x[1][1:-1].copy()
    z = sol.m[0][0,nx/2+1,1:-1].copy()
    plt.plot(t, z)
    for k in xrange(t.shape[0]):
        t[k] = sol.domain.x[0][k+1]*sqrt(2)
        z[k] = sol.m[0][0,k+1,k+1]
    plt.plot(t, z)
    plt.title('mass at t = {0:f}'.format(sol.t))
    plt.axis((xmin, xmax, rhoo-deltarho, rhoo+2*deltarho))
    plt.hold(False)
    plt.draw()
    plt.pause(1.e-3)    

if __name__ == "__main__":
    # parameters
    vx, vy = 0.3, 0.1
    dim = 2 # spatial dimension
    dx = 1./128 # spatial step
    la = 1. # velocity of the scheme
    rhoo = 1.
    deltarho = 0.5
    Tf = 10
    Longueur = 1
    Largeur = 1
    NbImages = 100 # number of figures
    sigma_qx = 1.e-3
    sigma_xy = sigma_qx#(1.-8*sigma_qx**2)/(4*sigma_qx)
    s_qx = 1./(0.5+sigma_qx)
    s_xy = 1./(0.5+sigma_xy)
    s  = [0., s_qx, s_qx, s_xy]
    Taille = 1.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1,-1,-1,-1]},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        0:{'velocities':range(5,9),
           'polynomials':Matrix([1, LA*X, LA*Y, X*Y]),
           'relaxation_parameters':s,
           'equilibrium':Matrix([u[0][0], vx*u[0][0], vy*u[0][0], vx*vy/la**2*u[0][0]]),
           },
        'init':{'type':'moments',
                0:{0:(initialization_rho,)},
                },
        'generator': pyLBMGen.CythonGenerator,
        }

    sol = pyLBMSimu.Simulation(dico)
    
    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plot(sol)

    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0
    while (sol.t<Tf):
        sol.one_time_step()
        #print str((int)(100*sol.t/Tf)) + '%'
        compt += 1
        if (compt%Ncompt==0):
            im += 1
            plot(sol)
    plot_coupe(sol)
    plt.ioff()
    plt.show()
    
