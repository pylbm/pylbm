import sys
import os
import os.path

import cmath
from math import pi, sqrt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import pyLBM
from pyLBM.elements import *
import pyLBM.geometry as pyLBMGeom
import pyLBM.simulation as pyLBMSimu
import pyLBM.domain as pyLBMDom
import pyLBM.boundary as pyLBMBoundary
import pyLBM.generator as pyLBMGen
#import pyLBM.Scheme as pyLBMScheme

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.5**2)

def initialization_q(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def plot_radial(sol, num=0):
    plt.clf()
    plt.plot(sol.domain.x[0], sol._m[:, sol.domain.N[0]/2+1, 0], 'b',
             sol.domain.x[0]*sqrt(2), sol._m[:,:,0].diagonal(), 'r')
    #plt.plot(sol.domain.x[0], sol.m[0][0,:,sol.domain.N[0]/2+1], 'b-',
    #         sol.domain.x[0]*sqrt(2), sol.m[0][0].diagonal(), 'r-')
    plt.legend(['angle 0', 'angle Pi/4'])
    plt.title('depth h at t = {0:f}'.format(sol.t))
    plt.axis((xmin, xmax, rhoo-0.75*deltarho, rhoo+1.1*deltarho))
    plt.draw()
    plt.pause(1.e-3)    

def simu1():
    # parameters
    dx = 1./125 # spatial step
    la = 4 # velocity of the scheme
    g = 1.
    Tf = 1.5
    sigma = 1.e-4
    s_0qx = 2.#1./(0.5+sigma)
    s_0xy = 1.5
    s_1qx = 1.5
    s_1xy = 1.2    
    s0  = [0., s_0qx, s_0qx, s_0xy]
    s1  = [0., s_1qx, s_1qx, s_1xy]

    vitesse = range(1,5)
    polynomes = Matrix([1, LA*X, LA*Y, X**2-Y**2])

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1,-1,-1,-1]},
        'space_step':dx,
        'scheme_velocity':la,
        'number_of_schemes':3,
        'init':'moments',
        0:{'velocities':vitesse,
           'polynomials':polynomes,
           'relaxation_parameters':s0,
           'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
        },
        1:{'velocities':vitesse,
           'polynomials':polynomes,
           'relaxation_parameters':s1,
           'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + 0.5*g*u[0][0]**2, u[1][0]*u[2][0]/u[0][0], 0.]),
        },
        2:{'velocities':vitesse,
           'polynomials':polynomes,
           'relaxation_parameters':s1,
           'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + 0.5*g*u[0][0]**2, 0.]),
        },
        'init':{'type':'moments',
                0:{0:(initialization_rho,)},
                1:{0:(initialization_q,)},
                2:{0:(initialization_q,)},
                },
        'generator': pyLBMGen.CythonGenerator,
        }

    #sol = pyLBMSimu.Simulation(dico)
    sol = pyLBMSimu.Simulation(dico, nv_on_beg=False)

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()

    im = 0
    plot_radial(sol,im)

    while (sol.t<Tf):
        sol.one_time_step()
        im += 1
        #
        sol.scheme.f2m(sol._F, sol._m)
        plot_radial(sol,im)
    plt.ioff()
    plt.show()

def simu2():
    # parameters
    dx = 1./25 # spatial step
    la = 4 # velocity of the scheme
    g = 1.
    Tf = 1.5
    mu   = 5.e-3
    zeta = mu
    dummy = 3.0/(la*rhoo*dx)
    s3 = 1.0/(0.5+zeta*dummy)
    s4 = s3
    s5 = s4
    s6 = s4
    s7 = 1.0/(0.5+mu*dummy)
    s8 = s7
    s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
    dummy = 1./LA**2
    qx2 = dummy*u[0][1]**2/u[0][0]
    qy2 = dummy*u[0][2]**2/u[0][0]
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]/u[0][0]

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1,-1,-1,-1]},
        'space_step':dx,
        'scheme_velocity':la,
        'number_of_schemes':1,
        'init':'moments',
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
                                  3*dummy*g*u[0][0]**2 - 4*u[0][0] + 3*q2,
                                  u[0][0]+1.5*q2,
                                  -u[0][1]/LA, -u[0][2]/LA,
                                  qx2-qy2, qxy]),
        },
        'init':{'type':'moments', 0:{0:(initialization_rho,),
                                     1:(initialization_q,),
                                     2:(initialization_q,)
                                     }
        },
        'generator': pyLBMGen.CythonGenerator,
        }
    
    sol = pyLBMSimu.Simulation(dico)

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()

    im = 0
    plot_radial(sol,im)

    while (sol.t<Tf):
        sol.one_time_step()
        im += 1
        plot_radial(sol,im)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    rhoo = 1.
    deltarho = 1.
    Taille = 5.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille
    simu1()
