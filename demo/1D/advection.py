import sys
import numpy as np
import sympy as sp
from sympy.matrices import Matrix

import pylab as plt
import matplotlib.cm as cm

import pyLBM

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def Riemann_pb(x):
    uu, ud = 1.0, 0.0 # left and right state
    xm = 0.5*(xmin+xmax)
    L = .125*(xmax-xmin)
    return ud + (uu-ud)*(x<xm+L)*(x>xm-L)

def Smooth(x):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    milieu -= 0.5*c*Tf
    return 1.0/largeur**10 * (x-milieu-largeur)**5 * (milieu-x-largeur)**5 * (abs(x-milieu)<=largeur)

def plot_init(num = 0):
    fig = plt.figure(num,figsize=(16, 8))
    plt.clf()
    l1 = plt.plot([], [], 'r', label=r'$D_1Q_2$')[0]
    l2 = plt.plot([], [], 'b', label=r'$D_1Q_3$')[0]
    plt.xlim(xmin, xmax)
    ymin, ymax = -.2, 1.2
    plt.ylim(ymin, ymax)
    plt.legend()
    return [l1, l2]

def plot(sol1, sol2, l):
    sol1.f2m()
    sol2.f2m()
    l[0].set_data(sol1.domain.x[0][1:-1], sol1.m[0][0][1:-1])
    l[1].set_data(sol2.domain.x[0][1:-1], sol2.m[0][0][1:-1])
    plt.title('solution at t = {0:f}'.format(sol1.t))
    plt.pause(1.e-3)

if __name__ == "__main__":
    # parameters
    xmin, xmax = -1., 1.
    dx = 1./256 # spatial step
    la = 1. # velocity of the scheme
    c = 0.25 # velocity of the advection
    Tf = 1.
    mu = 1.e-4 # numerical viscosity
    sigma2 = mu/(la**2-c**2)*la/dx
    sigma3 = 3*sigma2
    sigmap3 = .5*sigma3
    s2 = 1./(.5+sigma2)
    s3 = 1./(.5+sigma3)
    sp3 = 1./(.5+sigmap3)
    print s2, s3, sp3
    FINIT = Riemann_pb

    dico_Q2 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{
            'velocities':[2,1],
            'polynomials':Matrix([1,LA*X]),
            'relaxation_parameters':[0., s2],
            'equilibrium':Matrix([u[0][0], c*u[0][0]]),
            'init':{0:(FINIT,)},
            },],
        'generator': pyLBM.generator.CythonGenerator,
        'boundary_conditions':{
            0:{'method':{0:pyLBM.bc.neumann,},},
        }
    }

    # dico_Q3 = {
    #     'box':{'x':[xmin, xmax], 'label':0},
    #     'space_step':dx,
    #     'number_of_schemes':1,
    #     'scheme_velocity':la,
    #     'schemes':[{
    #         'velocities':[2,0,1],
    #         'polynomials':Matrix([1,LA*X,(LA*X)**2]),
    #         'relaxation_parameters':[0., s3, s3],
    #         'equilibrium':Matrix([u[0][0], c*u[0][0], (2*c**2+LA**2)/3*u[0][0]]),
    #         'init':{0:(FINIT,)},
    #         },],
    #     'generator': pyLBM.generator.CythonGenerator,
    #     'boundary_conditions':{
    #         0:{'method':{0:pyLBM.bc.neumann,},},
    #     }
    # }
    dico_Q3 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
        'schemes':[{
            'velocities':[2,0,1],
            'polynomials':Matrix([1,LA*X,(LA*X-c)**2]),
            'relaxation_parameters':[0., s3, sp3],
            'equilibrium':Matrix([u[0][0], c*u[0][0], (LA**2-c**2)/3*u[0][0]]),
            'init':{0:(FINIT,)},
            },],
        'generator': pyLBM.generator.CythonGenerator,
        'boundary_conditions':{
            0:{'method':{0:pyLBM.bc.neumann,},},
        }
    }

    sol1 = pyLBM.Simulation(dico_Q2)
    sol2 = pyLBM.Simulation(dico_Q3)
    l = plot_init(0)
    plot(sol1, sol2, l)

    while (sol1.t<Tf):
        sol1.one_time_step()
        sol2.one_time_step()
        plot(sol1, sol2, l)

    sol1.time_info()
    sol2.time_info()
    plt.show()
