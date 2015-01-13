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
import pyLBM.stencil as pyLBMSten
import pyLBM.domain as pyLBMDom
import pyLBM.scheme as pyLBMScheme
import pyLBM.simulation as pyLBMSimu

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def FINIT(x):
    ug, ud = 1.0, 0.0 # left and right state
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>=xm)

def dictionnaires_separes():
    # separated dictionnary for each class
    dico_geometry = {'box':{'x':[xmin, xmax], 'label':[0,0]}, 'Elements':None} #dico_geometry = {'box':{'x':[xmin, xmax], 'label':0}}
    dico_stencil = {'dim':dim, 0:{'velocities':[2,1]}}
    dico_domain = {'space_step':dx}
    dico_scheme = {
        'scheme_velocity':la,
        0:{
            'polynomials':Matrix([1,LA*X]),
            'relaxation_parameters':[0.,s],
            'equilibrium':Matrix([u[0][0], c*u[0][0]])
            }
        }
    dico_simu = {'init':{'type':'moments', 0:{0:(FINIT,)}}}
    return (dico_geometry, dico_stencil, dico_domain, dico_scheme, dico_simu)

def dictionnaire_global():
    # global dictionnary used by each class
    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        0:{
            'velocities':[2,1],
            'polynomials':Matrix([1,LA*X]),
            'relaxation_parameters':[0.,s],
            'equilibrium':Matrix([u[0][0], c*u[0][0]])
            },
        'init':{'type':'moments', 0:{0:(FINIT,)}}
        }
    return dico    

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = -1., 1.
    dx = 0.001 # spatial step
    la = 1. # velocity of the scheme
    c = 0.25 # velocity of the advection
    Tf = 1.
    s = 1.99
    # global dictionnary: 0, separated dictionnary: 1
    dico_option = 0
    
    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plt.hold(True)

    if (dico_option == 1):
        dico_geometry, dico_stencil, dico_domain, dico_scheme, dico_simu = dictionnaires_separes()
        geom = pyLBMGeom.Geometry(dico_geometry)
        sten = pyLBMSten.Stencil(dico_stencil)
        dom = pyLBMDom.Domain(dico_domain, geometry=geom, stencil=sten)
        scheme = pyLBMScheme.Scheme(dico_scheme, stencil=sten)
        sol = pyLBMSimu.Simulation(dico_simu, domain=dom, scheme=scheme)
    elif (dico_option == 0):
        dico = dictionnaire_global()
        sol = pyLBMSimu.Simulation(dico)


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

    plt.subplot(121)
    xx = np.linspace(xmin,xmax,10000)
    plt.plot(xx, FINIT(xx-c*sol.t),'k--')

    plt.legend(['initial','D1Q2','exact'])
    plt.title("Solution at t={0:.3f}".format(sol.t), fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    plt.hold(False)
    plt.ioff()
    plt.show()

    
