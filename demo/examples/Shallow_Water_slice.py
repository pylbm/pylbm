import sys

import numpy as np
import sympy as sp
from sympy.matrices import Matrix

import pyLBM

import pylab as plt

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    #return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)
    return rhoo * np.ones((y.shape[0], x.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)

def initialization_q(x,y):
    #return np.zeros((x.shape[0], y.shape[0]), dtype='float64')
    return np.zeros((y.shape[0], x.shape[0]), dtype='float64')

def plot_radial(sol, num=0):
    plt.clf()
    plt.plot(sol.domain.x[0], sol.m[0][0][:, sol.domain.N[0]/2+1], 'b',
            sol.domain.x[0]*np.sqrt(2), sol.m[0][0][:,:].diagonal(), 'r')
    plt.legend(['angle 0', 'angle Pi/4'])
    plt.title('depth h at t = {0:f}'.format(sol.t))
    plt.axis((xmin, xmax, rhoo-0.75*deltarho, rhoo+1.1*deltarho))
    plt.draw()
    plt.pause(1.e-3)

@profile
def simu():
    # parameters
    dx = 1./125 # spatial step
    la = 4 # velocity of the scheme
    g = 1.
    Tf = 0.1
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
        'inittype':'moments',
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':s0,
                    'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                    'init':{0:(initialization_rho,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':s1,
                    'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + 0.5*g*u[0][0]**2, u[1][0]*u[2][0]/u[0][0], 0.]),
                    'init':{0:(initialization_q,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':s1,
                    'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + 0.5*g*u[0][0]**2, 0.]),
                    'init':{0:(initialization_q,)},
                    },
        ],
        'generator': pyLBM.generator.CythonGenerator,
        }

    sol = pyLBM.Simulation(dico)

    # fig = plt.figure(0,figsize=(16, 8))
    # fig.clf()
    # plt.ion()
    #
    # im = 0
    # plot_radial(sol,im)

    while (sol.t<Tf):
        sol.one_time_step()
        # im += 1
        sol.f2m()
    #     plot_radial(sol,im)
    # plt.ioff()
    # plt.show()
    sol.time_info()

if __name__ == "__main__":
    rhoo = 1.
    deltarho = 1.
    Taille = 5.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille
    simu()
