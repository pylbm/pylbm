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
import pyLBM.geometry as pyLBMGeom
import pyLBM.simulation as pyLBMSimu
import pyLBM.domain as pyLBMDom
import pyLBM.scheme as pyLBMScheme

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def minit_lisse(x):
    Nx = x.shape[0]
    u = np.zeros((Nx, ), dtype = 'float64')
    largeur = 0.25
    hauteur = 0.5
    milieu = 0.5*(xmin+xmax)
    for k in xrange(Nx):
        xx = x[k] - milieu
        if xx < -largeur:
            u[k] = -hauteur
        elif xx > largeur:
            u[k] = hauteur
        elif xx>=0:
            u[k] = hauteur * (1 + (xx/largeur-1)**3)
        else:
            u[k] = -hauteur * (1 + (-xx/largeur-1)**3)
    return u

def solution_lisse(t,x):
    Nx = x.shape[0]
    u = np.zeros((Nx, ), dtype = 'float64')
    largeur = 0.25
    hauteur = 0.5
    milieu = 0.5*(xmin+xmax)
    for k in xrange(Nx):
        xx = x[k] - milieu
        if xx <= -largeur-hauteur*t:
            u[k] = -hauteur
        elif xx >= largeur+hauteur*t:
            u[k] = hauteur
        else:
            if t>0:
                if xx>=0:
                    #xo =  (largeur+2*t*hauteur-sqrt((largeur+2*t*hauteur)**2-4*t*hauteur*xx))*largeur/(2*t*hauteur)
                    #u[k] = hauteur * (1 - (xo/largeur-1)**2)
                    xo = ((1./6)*((-108*t*hauteur-108*largeur+108*xx+12*sqrt(3)*sqrt((4*largeur**3+27*t**3*hauteur**3+54*t**2*hauteur**2*largeur-54*t**2*hauteur**2*xx+27*t*hauteur*largeur**2-54*t*hauteur*largeur*xx+27*t*hauteur*xx**2)/(t*hauteur)))*t**2*hauteur**2)**(1./3)/(t*hauteur)-2*largeur/((-108*t*hauteur-108*largeur+108*xx+12*sqrt(3)*sqrt((4*largeur**3+27*t**3*hauteur**3+54*t**2*hauteur**2*largeur-54*t**2*hauteur**2*xx+27*t*hauteur*largeur**2-54*t*hauteur*largeur*xx+27*t*hauteur*xx**2)/(t*hauteur)))*t**2*hauteur**2)**(1./3)+1)*largeur
                    u[k] = hauteur * (1 + (xo/largeur-1)**3)
                else:
                    #xo = -(largeur+2*t*hauteur-sqrt((largeur+2*t*hauteur)**2+4*t*hauteur*xx))*largeur/(2*t*hauteur)
                    #u[k] = -hauteur * (1 - (-xo/largeur-1)**2)
                    xo = ((1./6)*((108*t*hauteur+108*largeur+108*xx+12*sqrt(3)*sqrt((4*largeur**3+27*t**3*hauteur**3+54*t**2*hauteur**2*largeur+54*t**2*hauteur**2*xx+27*t*hauteur*largeur**2+54*t*hauteur*largeur*xx+27*t*hauteur*xx**2)/(t*hauteur)))*t**2*hauteur**2)**(1./3)/(t*hauteur)-2*largeur/((108*t*hauteur+108*largeur+108*xx+12*sqrt(3)*sqrt((4*largeur**3+27*t**3*hauteur**3+54*t**2*hauteur**2*largeur+54*t**2*hauteur**2*xx+27*t*hauteur*largeur**2+54*t*hauteur*largeur*xx+27*t*hauteur*xx**2)/(t*hauteur)))*t**2*hauteur**2)**(1./3)-1)*largeur
                    u[k] = -hauteur * (1 + (-xo/largeur-1)**3)
            else:
                if xx>=0:
                    #u[k] = hauteur * (1 - (xx/largeur-1)**2)
                    u[k] = hauteur * (1 + (xx/largeur-1)**3)
                else:
                    #u[k] = -hauteur * (1 - (-xx/largeur-1)**2)
                    u[k] = -hauteur * (1 + (-xx/largeur-1)**3)
    return u

def minit_Riemann(x):
    largeur = 0.2
    rhog = -0.3
    rhom =  0.2
    rhod =  0.0
    xm = 0.5*(xmin+xmax)
    return (rhog-rhom)*(x<xm-largeur) + rhom*(x<xm+largeur) + rhod*(x>=xm+largeur)

def solution_Riemann(t,x):
    Nx = x.shape[0]
    u = np.zeros(x.shape, dtype = 'float64')
    largeur = 0.2
    rhog = -0.3
    rhom =  0.2
    rhod =  0.0
    milieu = 0.5*(xmin+xmax)
    for k in xrange(Nx):
        xx = x[k] - milieu
        if xx <= -largeur+rhog*t:
            u[k] = rhog
        elif xx < -largeur+rhom*t:
            u[k] = (xx+largeur)/t
        elif xx<= largeur+0.5*(rhom+rhod)*t:
            u[k] = rhom
        else:
            u[k] = rhod
    return u

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = -1., 1.
    dx = 0.005 # spatial step
    Tf = 0.25
    NbImages = 20
    s = 1.99
    FINIT = minit_Riemann
    FSOL = solution_Riemann
    #FINIT = minit_lisse
    #FSOL = solution_lisse

    dico_geometry = {'dim':dim,
                     'box':{'x':[xmin, xmax], 'label':[0,0]},
                     #'Elements':[]
                     }
    
    dico1 = {'dim':dim,
              'geometry':dico_geometry,
              'space_step':dx,
              'scheme_velocity':1.,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,1],
                 'polynomials':Matrix([1,LA*X]),
                 'relaxation_parameters':[0.,s],
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2]),
                 'init':{0:FINIT}
                 }
              }
    """
    alpha = 0.45
    dico2 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':1.,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,0,1],
                 'polynomials':Matrix([1,LA*X,LA**2*X**2]),
                 'relaxation_parameters':s,
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2, alpha*LA**2*u[0][0]]),
                 'init':{0:FINIT}
                 }
              }
    """
    dico2 = {'dim':dim,
              'geometry':dico_geometry,
              'space_step':dx,
              'scheme_velocity':1.,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,0,1],
                 'polynomials':Matrix([1,LA*X,LA**2*X**2]),
                 'relaxation_parameters':[0., s, s],
                 #'equilibrium':Matrix([u[0][0], u[0][0]**2/2, LA*u[0][0]*sp.Abs(u[0][0])/2]),
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2, LA**2*u[0][0]/3 + 2*u[0][0]**3/9]),
                 'init':{0:FINIT}
                 }
              }
    
    fig = plt.figure(0,figsize=(8, 8))
    fig.clf()
    plt.ion()
    plt.hold(True)

    geom = pyLBMGeom.Geometry(dico1)
    sol = pyLBMSimu.Simulation(dico1, geom)
    while (sol.t<Tf-0.5*sol.dt):
        sol.one_time_step()
    plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0][1:-1],'r*')
    plt.title("Solution at t={0:.3f}".format(sol.t), fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    print 'final time for the first scheme: {0:f}, dt={1:f}'.format(sol.t, sol.dt)
    
    geom = pyLBMGeom.Geometry(dico2)
    sol = pyLBMSimu.Simulation(dico2, geom)
    while (sol.t<Tf-0.5*sol.dt):
        sol.one_time_step()
    plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0][1:-1],'bd')
    plt.title("Solution at t={0:.3f}".format(sol.t), fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    print 'final time for the second scheme: {0:f}, dt={1:f}'.format(sol.t, sol.dt)

    plt.plot(sol.Domain.x[0][1:-1],FSOL(sol.t,sol.Domain.x[0][1:-1]),'k-')
    plt.hold(False)
    plt.ioff()
    plt.show()
