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

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def Riemann_pb(x):
    ug, ud = 1.0, 0.0 # left and right state
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>=xm)

def Smooth(x):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    milieu -= 0.5*c*Tf
    return 1.0/largeur**10 * (x-milieu-largeur)**5 * (milieu-x-largeur)**5 * (abs(x-milieu)<=largeur)

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = -1., 1.
    dx = 0.001 # spatial step
    la = 1. # velocity of the scheme
    c = 0.25 # velocity of the advection
    Tf = 1.
    s = 1.99
    FINIT = Riemann_pb

    dico_Q2 = {
        'box':{'x':[xmin, xmax], 'label':-1},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        'schemes':[{
            'velocities':[2,1],
            'polynomials':Matrix([1,LA*X]),
            'relaxation_parameters':[0.,s],
            'equilibrium':Matrix([u[0][0], c*u[0][0]]),
            'init':{0:(FINIT,)},
            },],
        'generator': pyLBM.generator.NumpyGenerator,
        }

    #s1 = 1.9
    #sigma1 = 1./s1-0.5
    #sigma2 = sqrt(sigma1**2+1./(64*sigma1**2)) - sigma1 + 1./(8*sigma1)
    #sQ3 = [0., 1./(0.5+sigma1), 1./(0.5+sigma2)]
    sQ3 = [0., s, s]
    dico_Q3 = {
        'box':{'x':[xmin, xmax], 'label':-1},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        'schemes':[{
            'velocities':[2,0,1],
            'polynomials':Matrix([1,LA*X,(LA*X)**2]),
            'relaxation_parameters':sQ3,
            'equilibrium':Matrix([u[0][0], c*u[0][0], (2*c**2+LA**2)/3*u[0][0]]),
            'init':{0:(FINIT,)},
            },],
        'generator': pyLBM.generator.NumpyGenerator,
        }

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plt.hold(True)

    sol = pyLBM.Simulation(dico_Q2)
    plt.subplot(121)
    plt.plot(sol.domain.x[0][1:-1],sol.m[0][0][1:-1],'k-')
    plt.draw()
    plt.pause(1.e-3)
    while (sol.t<Tf):
        sol.one_time_step()
    plt.plot(sol.domain.x[0][1:-1],sol.m[0][0][1:-1],'r-')
    plt.draw()
    plt.subplot(122)
    plt.plot(sol.domain.x[0][1:-1],sol.m[0][0][1:-1]-FINIT(sol.domain.x[0][1:-1]-c*sol.t),'r-')
    plt.draw()
    plt.pause(1.e-3)

    sol = pyLBM.Simulation(dico_Q3)
    plt.subplot(121)
    while (sol.t<Tf):
        sol.one_time_step()
    plt.plot(sol.domain.x[0][1:-1],sol.m[0][0][1:-1],'b-')
    plt.draw()
    plt.subplot(122)
    plt.plot(sol.domain.x[0][1:-1],sol.m[0][0][1:-1]-FINIT(sol.domain.x[0][1:-1]-c*sol.t),'b-')
    plt.draw()
    plt.pause(1.e-3)

    plt.subplot(121)
    xx = np.linspace(xmin,xmax,10000)
    plt.plot(xx, FINIT(xx-c*sol.t),'k--')

    plt.legend(['initial','D1Q2','D1Q3','exact'])
    plt.title("Solution at t={0:.3f}".format(sol.t), fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    plt.hold(False)
    plt.ioff()
    plt.show()
