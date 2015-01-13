import sys
import cmath
from math import pi, sqrt, log10
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

def Calcul_D1Q2(k, FINIT, FSOL, norm=1):
    dx = 2**(-k) # spatial step
    dicoQ2 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,1],
                 'polynomials':Matrix([1,LA*X]),
                 'relaxation_parameters':[0.,1.9],
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2]),
                 'init':{0:FINIT}
                 }
                }
    geom = pyLBMGeom.Geometry(dicoQ2)
    sol = pyLBMSimu.Simulation(dicoQ2, geom)
    while (sol.t<Tf):
        sol.one_time_step_half()
    exacte = FSOL(sol.t, sol.Domain.x[0][1:-1])
    if (norm == 1):
        Err = dx*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 1)
    elif (norm == 2):
        Err = sqrt(dx)*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 2)
    else:
        print 'Bad choice of norm'
        sys.exit()
    print 'Error for k={0:2d}: {1:10.3e}'.format(k,Err)
    return Err

def Calcul_D1Q3(k, FINIT, FSOL, norm=1):
    dx = 2**(-k) # spatial step
    s1 = 0.1
    sigma1 = 1./s1-0.5
    sigma2 = sqrt(sigma1**2+1./(64*sigma1**2)) - sigma1 + 1./(8*sigma1)
    sQ3 = [0., 1./(0.5+sigma1), 1./(0.5+sigma2)]
    dicoQ3 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,0,1],
                 'polynomials':Matrix([1,LA*X,LA**2*X**2]),
                 'relaxation_parameters':sQ3,
                 'equilibrium':Matrix([u[0][0], u[0][0]**2/2, u[0][0]**3/3]),
                 'init':{0:FINIT}
                 }
            }

    geom = pyLBMGeom.Geometry(dicoQ3)
    sol = pyLBMSimu.Simulation(dicoQ3, geom)
    while (sol.t<Tf):
        sol.one_time_step()
    exacte = FSOL(sol.t, sol.Domain.x[0][1:-1])
    if (norm == 1):
        Err = dx*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 1)
    elif (norm == 2):
        Err = sqrt(dx)*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 2)
    else:
        print 'Bad choice of norm'
        sys.exit()
    print 'Error for k={0:2d}: {1:10.3e}'.format(k,Err)
    return Err

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = 0., 1.
    #FINIT = minit_lisse
    #FSOL = solution_lisse
    FINIT = minit_Riemann
    FSOL = solution_Riemann
    la = 1. # velocity of the scheme
    Tf = 0.2
    KK = range(3, 15)
    EK2 = []
    EK3 = []
    for k in KK:
        EK2.append(log10(Calcul_D1Q2(k,FINIT,FSOL,norm=1)))
        EK3.append(log10(Calcul_D1Q3(k,FINIT,FSOL,norm=1)))
    slope2 = (EK2[-1]-EK2[-2]) / (log10(2**(-KK[-1]))-log10(2**(-KK[-2])))
    slope3 = (EK3[-1]-EK3[-2]) / (log10(2**(-KK[-1]))-log10(2**(-KK[-2])))
    fig = plt.figure(0,figsize=(8, 8))
    fig.clf()
    plt.ion()
    plt.plot(KK, EK2, 'b*', KK, EK3, 'rd')
    plt.title('Slope D1Q2 {0:5.3f}, D1Q3 {1:5.3f}'.format(slope2, slope3))
    plt.ioff()
    plt.draw()
    plt.show()
    

