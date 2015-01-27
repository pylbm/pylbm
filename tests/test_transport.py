import sys
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

def init_m(x,y,val):
    return val*np.ones((y.size, x.size), dtype='float64')

def init_un(x,y):
    uu = np.zeros((y.size, x.size), dtype='float64')
    uu[y.size/2, x.size/2] = 1.
    return uu

def bounce_back(sol):
    sten = sol.scheme.stencil
    ns = sol.scheme.nscheme
    for n in xrange(ns):
        for v in sten.v[n]:
            ksym = sten[n].num2index[v.get_symmetric().num]
            k = sten[n].num2index[v.num]
            num = sten.unum2index[v.num]
            vkx, vky = v.vx, v.vy
            ind_vk_y, ind_vk_x = np.where(sol.domain.distance[num,:,:]<sol.domain.valin)
            sol.F[n][ksym][ind_vk_y+vky, ind_vk_x+vkx] = sol.F[n][k][ind_vk_y, ind_vk_x]
    # f = sol.F[0]
    # f[1][1:-1, 0] = f[3][1:-1, 1]
    # f[2][0, 1:-1] = f[4][1, 1:-1]
    # f[3][1:-1, -1] = f[1][1:-1, -2]
    # f[4][-1, 1:-1] = f[2][-2, 1:-1]
    # f[5][:-2, 0] = f[7][1:-1, 1]
    # f[5][0, :-2] = f[7][1, 1:-1]
    # f[6][:-2, -1] = f[8][1:-1, -2]
    # f[6][0, 2:] = f[8][1, 1:-1]
    # f[7][2:, -1] = f[5][1:-1, -2]
    # f[7][-1, 2:] = f[5][-2, 1:-1]
    # f[8][2:, 0] = f[6][1:-1, 1]
    # f[8][-1, :-2] = f[6][-2, 1:-1]

def CL(sol):
    f = sol.F[0]
    f[1][1:-1, 0] = -f[3][1:-1, 1]
    f[2][0, 1:-1] = f[4][1, 1:-1]
    f[3][1:-1, -1] = -f[1][1:-1, -2]
    f[4][-1, 1:-1] = f[2][-2, 1:-1]
    f[5][:-2, 0] = -f[7][1:-1, 1]
    f[5][0, :-2] = f[7][1, 1:-1]
    f[6][:-2, -1] = -f[8][1:-1, -2]
    f[6][0, 2:] = f[8][1, 1:-1]
    f[7][2:, -1] = -f[5][1:-1, -2]
    f[7][-1, 2:] = f[5][-2, 1:-1]
    f[8][2:, 0] = -f[6][1:-1, 1]
    f[8][-1, :-2] = f[6][-2, 1:-1]

def periodique(sol):
    ns = sol.scheme.nscheme
    xb = sol.domain.indbe[0][0]
    xe = sol.domain.indbe[0][1]
    yb = sol.domain.indbe[1][0]
    ye = sol.domain.indbe[1][1]
    for n in xrange(ns):
        for l in xrange(sol.scheme.stencil.nv[n]):
            sol.F[n][l][yb:ye, 0 :xb] = sol.F[n][l][yb:ye,    xe-xb:xe] # E -> W
            sol.F[n][l][yb:ye, xe:  ] = sol.F[n][l][yb:ye,    xb:xb+xb] # W -> E
            sol.F[n][l][0 :yb, xb:xe] = sol.F[n][l][ye-yb:ye, xb:xe   ] # N -> S
            sol.F[n][l][ye:,   xb:xe] = sol.F[n][l][yb:yb+yb, xb:xe   ] # S -> N
            sol.F[n][l][0 :yb, 0 :xb] = sol.F[n][l][ye-yb:ye, xe-xb:xe] # NE -> SW
            sol.F[n][l][ye:,   0 :xb] = sol.F[n][l][yb:yb+yb, xe-xb:xe] # SE -> NW
            sol.F[n][l][0 :yb, xe:  ] = sol.F[n][l][ye-yb:ye, xb:xb+xb] # NW -> SE
            sol.F[n][l][ye:,   xe:  ] = sol.F[n][l][yb:yb+yb, xb:xb+xb] # SW -> NE


def plot_F(sol):
    Sten = sol.scheme.stencil
    vxm  = Sten.vmax[0]
    vym  = Sten.vmax[1]
    nx   = 1+2*vxm
    ny   = 1+2*vym
    num_scheme = 0
    for k in xrange(Sten.nv[num_scheme]):
        vx = (int)(Sten.vx[num_scheme][k])
        vy = (int)(Sten.vy[num_scheme][k])
        numim = ny*(vxm-vx) + vy+vym + 1
        plt.subplot(ny*100+nx*10+numim)
        plt.imshow(np.float32((sol.F[num_scheme][k][1:-1,1:-1])).transpose(), origin='lower', cmap=cm.jet, interpolation='nearest')
        plt.title('({1:d},{2:d}) at t = {0:f}'.format(sol.t, vx, vy))
    plt.draw()
    plt.pause(3.e0)

def plot_m(sol,valeq):
    for i in xrange(3):
        for j in xrange(3):
            k = 3*i+j
            plt.subplot(331+k)
            plt.plot(sol.t,sol.m[0][k,3,3],'k*',[sol.t,sol.t+sol.dt],[valeq[k],valeq[k]],'r')
            plt.title('m[{1:d}] at t = {0:f}'.format(sol.t, k))

def test_transport():
    # parameters
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = -0.5, 4.5, -0.5, 4.5
    dx = 1. # spatial step
    la = 1. # velocity of the scheme
    Tf = 5

    rhoo = 1.
    s  = [0.,0.,0.,1.,1.,1.,1.,1.,1.]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]

    vitesse = range(9)
    polynomes = Matrix([1,
                        LA*X, LA*Y,
                        3*(X**2+Y**2)-4,
                        0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                        3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                        X**2-Y**2, X*Y])
    equilibre = Matrix([u[0][0],
                        u[0][1], u[0][2],
                        -2*u[0][0] + 3*q2,
                        u[0][0]+1.5*q2,
                        u[0][1]/LA, u[0][2]/LA,
                        qx2-qy2, qxy])

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0, 1, 0, 1]},
        'space_step':dx,
        'scheme_velocity':la,
        'inittype':'distributions',
        'schemes':[{'velocities':vitesse,
                      'polynomials':polynomes,
                      'relaxation_parameters':s,
                      'equilibrium':equilibre,
                      'init':{0:(init_un,),
                              1:(init_un,),
                              2:(init_un,),
                              3:(init_un,),
                              4:(init_un,),
                              5:(init_un,),
                              6:(init_un,),
                              7:(init_un,),
                              8:(init_un,),
                              },
                    },
                    ],
        #'generator': pyLBM.generator.CythonGenerator,
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.bouzidi_bounce_back}, 'value':None},
            1:{'method':{0: pyLBM.bc.bouzidi_anti_bounce_back}, 'value':None},
        },
        }


    sol = pyLBM.Simulation(dico)
    #print sol._F
    #print sol.scheme.generator.code

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plot_F(sol)
    while (sol.t<Tf-0.5*sol.dt):
        # sol.scheme.m2f(sol._m, sol._F)
        # #periodique(sol)
        # bounce_back(sol)
        # sol.scheme.transport(sol._F)
        # sol.scheme.f2m(sol._F, sol._m)
        sol.m2f()
        #sol.boundary_condition()
        CL(sol)
        sol.transport()
        sol.f2m()
        sol.t += sol.dt
        plot_F(sol)
    plt.ioff()
    plt.show()

def test_relaxation():
    # parameters
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
    dx = 1. # spatial step
    la = 1. # velocity of the scheme
    Tf = 10
    rhoo = 1.
    s  = [0., 0., 0., 1.9, 1.8, 1.7, 1.75, 1.85, 1.95]

    rhoi = 1.
    qxi = -0.2
    qyi = 1.2

    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]
    dico_geometry = {'dim':dim,
                     'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':0},
                     'Elements':[]
                     }
    dico   = {'dim':dim,
              'Geometry':dico_geometry,
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
                                       -2*u[0][0] + 3*q2,
                                       u[0][0]+1.5*q2,
                                       u[0][1]/LA, u[0][2]/LA,
                                       qx2-qy2, qxy]),
                 'init':{0:init_rho, 1:init_qx, 2:init_qy},
                 'init_args':{0:(rhoi,), 1:(qxi,), 2:(qyi,)}
                 }
            }

    geom = pyLBMGeom.Geometry(dico)
    dom = pyLBMDom.Domain(geom,dico)
    sol = pyLBMSimu.Simulation(dico, geom)
    print sol.Scheme.Code_Relaxation
    q2 = qxi**2+qyi**2
    valeq = [rhoi, qxi, qyi, -2*rhoi+3*q2, rhoi+1.5*q2, qxi, qyi, qxi**2-qyi**2, qxi*qyi]
    sol.m[0][3:,:,:] = 0.
    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plot_m(sol,valeq)
    while (sol.t<Tf-0.5*sol.dt):
        sol.Scheme.m2f(sol.m, sol.F)
        sol.Scheme.f2m(sol.F, sol.m)
        sol.Scheme.relaxation(sol.m)
        sol.t += sol.dt
        plot_m(sol,valeq)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_transport()
