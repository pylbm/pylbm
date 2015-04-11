import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import pylab as plt

import mpi4py.MPI as mpi
import pyLBM

from pyLBM.interface import get_directions

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def test1D(comm):
    xmin, xmax = 0., 1.
    dx = (xmax-xmin)/128
    Tf = 0.5

    def init_u(x):
        milieu = 0.5*(xmin+xmax)
        largeur = 2*dx
        milieu -= 0.25*Tf
        return 1.0/largeur**10 * (x-milieu-largeur)**5 * (milieu-x-largeur)**5 * (abs(x-milieu)<=largeur)

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'comm': comm,
        'space_step':dx,
        'scheme_velocity':1.,
        'schemes':[{
            'velocities':range(1, 3),
            'polynomials':Matrix([1, X]),
            'relaxation_parameters':[0., 1.5],
            'equilibrium':Matrix([u[0][0], .5*u[0][0]]),
            'init':{0:(init_u,)},
        }],
        'generator': pyLBM.generator.CythonGenerator,
        'boundary_conditions':{
            0:{'method':{0:pyLBM.bc.bouzidi_bounce_back,}, 'value':None},
        },
    }

    sol = pyLBM.Simulation(dico)
    while sol.t < Tf:
        sol.one_time_step()
    sol.f2m()
    return sol.mglobal


def test2D_1(comm):
    dx = 1./16 # spatial step
    la = 4 # velocity of the scheme
    rhoo = 1.
    deltarho = 1.
    g = 1.
    Tf = 50*dx

    Taille = 1.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille

    vitesse = range(1,5)
    polynomes = Matrix([1, LA*X, LA*Y, X**2-Y**2])

    def initialization_rho(x,y):
        return rhoo * np.ones((y.shape[0], x.shape[0]), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)

    def initialization_q(x,y):
        return np.zeros((y.shape[0], x.shape[0]), dtype='float64')

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
        'comm': comm,
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 2., 2., 1.5],
                    'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                    'init':{0:(initialization_rho,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 1.5, 1.5, 1.2],
                    'equilibrium':Matrix([u[1][0], u[1][0]**2/u[0][0] + 0.5*g*u[0][0]**2, u[1][0]*u[2][0]/u[0][0], 0.]),
                    'init':{0:(initialization_q,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., 1.5, 1.5, 1.2],
                    'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0]/u[0][0], u[2][0]**2/u[0][0] + 0.5*g*u[0][0]**2, 0.]),
                    'init':{0:(initialization_q,)},
                    },
        ],
        'generator': pyLBM.generator.NumpyGenerator,
    }

    sol = pyLBM.Simulation(dico)
    while sol.t < Tf:
        sol.one_time_step()
    sol.f2m()
    return sol.mglobal

def test2D_2(comm):
    dx = 1./128 # spatial step
    la = 1. # velocity of the scheme
    rhoo = 1.
    driven_velocity = .1
    Tf = 10.

    Taille = 1.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille

    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]

    def initialization_rho(x,y):
        return np.zeros((x.size, y.size), dtype='float64')

    def initialization_qx(x,y):
        return np.zeros((x.size, y.size), dtype='float64')

    def initialization_qy(x,y):
        return np.zeros((x.size, y.size), dtype='float64')

    def bc_up(f, m, x, y, scheme):
        m[:,0] = 0.
        m[:,1] = rhoo * driven_velocity
        m[:,2] = 0.
        scheme.equilibrium(m)
        scheme.m2f(m, f)

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,0,1,0]},
        'comm': comm,
        'element':[pyLBM.Circle([.5*(xmin+xmax), .5*(ymin+ymax)], 0.1, label=0)],
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{
            'velocities':range(9),
            'polynomials':Matrix([1,
                                  LA*X, LA*Y,
                                  3*(X**2+Y**2)-4,
                                  0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                                  3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                                  X**2-Y**2, X*Y]),
            'relaxation_parameters':[0., 0., 0., 1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
            'equilibrium':Matrix([u[0][0],
                                  u[0][1], u[0][2],
                                  -2*u[0][0] + 3*q2,
                                  u[0][0]+1.5*q2,
                                  -u[0][1]/LA, -u[0][2]/LA,
                                  qx2-qy2, qxy]),
            'init':{
                0:(initialization_rho,),
                1:(initialization_qx,),
                2:(initialization_qy,)
            },
            },],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.bouzidi_bounce_back}, 'value':None},
            1:{'method':{0: pyLBM.bc.bouzidi_bounce_back}, 'value':bc_up}
        },
        'generator': pyLBM.generator.CythonGenerator,
    }

    sol = pyLBM.Simulation(dico)
    while sol.t < Tf:
        sol.one_time_step()
    sol.f2m()
    return sol.mglobal

if __name__ == "__main__":

    if mpi.COMM_WORLD.Get_rank() == 0:
        fig, axarr = plt.subplots(2, 2)

    for i in range(4):
        group = mpi.COMM_WORLD.Get_group()
        incl = np.arange(2**i)
        g = group.Incl(incl)
        newcomm = mpi.COMM_WORLD.Create(g)

        if mpi.COMM_WORLD.Get_rank() in incl:
            m = test1D(newcomm)
            if mpi.COMM_WORLD.Get_rank() == 0:
                print i
                axarr[i/2,i%2].plot(m[:, 1])
                axarr[i/2,i%2].set_title('{0} proc'.format(len(incl)))

    if mpi.COMM_WORLD.Get_rank() == 0:
        plt.show()


    if mpi.COMM_WORLD.Get_rank() == 0:
        fig, axarr = plt.subplots(2, 2)

    for i in range(4):
        group = mpi.COMM_WORLD.Get_group()
        incl = np.arange(2**i)
        g = group.Incl(incl)
        newcomm = mpi.COMM_WORLD.Create(g)

        if mpi.COMM_WORLD.Get_rank() in incl:
            m = test2D_2(newcomm)
            if mpi.COMM_WORLD.Get_rank() == 0:
                print i
                axarr[i/2,i%2].imshow(m[:, :, 1])
                axarr[i/2,i%2].set_title('{0} proc'.format(len(incl)))

    if mpi.COMM_WORLD.Get_rank() == 0:
        plt.show()
