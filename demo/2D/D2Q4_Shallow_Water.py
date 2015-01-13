import sys
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
from mpl_toolkits.mplot3d import Axes3D


X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.5**2)

def initialization_q(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def plot(sol, num=0):
    plt.clf()
    plt.imshow(np.float32(sol.m[0][0,1:-1,1:-1].transpose()), origin='lower', cmap=cm.gray)
    plt.title('mass at t = {0:f}'.format(sol.t))
    plt.draw()
    #plt.savefig("sauvegarde_images/Shallow_Water_{0:04d}.pdf".format(num))
    plt.pause(1.e-3)

def plot_3D(sol):
    pas = 8
    plt.clf()
    ax = Axes3D(fig)
    ax.plot_surface(sol.Domain.x[0][1:-1:pas,np.newaxis],sol.Domain.x[1][np.newaxis,1:-1:pas],sol.m[0][0,1:-1:pas,1:-1:pas],\
                    cmap = matplotlib.cm.copper,linewidth=0,antialiased=True)
    #ax.set_zlim3d([-1.,-1.])
    plt.title('mass at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)
    
def plot_coupe(sol):
    nx, ny = sol.Domain.N
    plt.clf()
    plt.hold(True)    
    t = sol.Domain.x[0][1:-1].copy()
    z = sol.m[0][0,1:-1,ny/2+1].copy()
    plt.plot(t, z)
    t = sol.Domain.x[1][1:-1].copy()
    z = sol.m[0][0,nx/2+1,1:-1].copy()
    plt.plot(t, z)
    for k in xrange(t.shape[0]):
        t[k] = sol.Domain.x[0][k+1]*sqrt(2)
        z[k] = sol.m[0][0,k+1,k+1]
    plt.plot(t, z)
    plt.title('mass at t = {0:f}'.format(sol.t))
    plt.axis((xmin, xmax, rhoo-deltarho, rhoo+2*deltarho))
    plt.hold(False)
    plt.draw()
    plt.pause(1.e-3)    

if __name__ == "__main__":
    # parameters
    dim = 2 # spatial dimension
    dx = 1./128 # spatial step
    la = 16 # velocity of the scheme
    rhoo = 1.
    deltarho = 1.
    Tf = 1.
    Longueur = 1
    Largeur = 1
    NbImages = 100 # number of figures
    sigma_qx = 1.e-3
    sigma_xy = sigma_qx#(1.-8*sigma_qx**2)/(4*sigma_qx)
    s_qx = 1./(0.5+sigma_qx)
    s_xy = 1./(0.5+sigma_xy)
    s0  = [0., s_qx, s_qx, s_xy]
    s1  = 10*[0., s_qx, s_qx, s_xy]
    Taille = 2.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille

    ## D2Q4 twisted
    #vitesse = range(5,9)
    #polynomes = Matrix([1, LA*X, LA*Y, X*Y])
    # D2Q4
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
           'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + u[0][0]/3, u[1][0]*u[2][0]/u[0][0], 0.]),
        },
        2:{'velocities':vitesse,
           'polynomials':polynomes,
           'relaxation_parameters':s1,
           'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + u[0][0]/3, 0.]),
        },
        'init':{'type':'moments',
                0:{0:(initialization_rho,)},
                1:{0:(initialization_q,)},
                2:{0:(initialization_q,)},
                },
        'generator': pyLBMGen.CythonGenerator,
        }
    
    sol = pyLBMSimu.Simulation(dico)

    # fig = plt.figure(0,figsize=(16, 8))
    # fig.clf()
    # plt.ion()
    # im = 0
    # plot(sol,im)

    im = 0
    #compt = 0
    #Ncompt = (int)(Tf/(NbImages*sol.dt))
    import time
    t = time.time()
    while (sol.t<Tf):
        sol.one_time_step()
        #print str((int)(100*sol.t/Tf)) + '%'
        #compt += 1
        #if (compt%Ncompt==0):
        im += 1
        plot(sol,im)

    #plt.ioff()
    #plt.show()
