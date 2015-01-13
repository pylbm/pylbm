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
from pyLBM.elements import *
import pyLBM.geometry as pyLBMGeom
import pyLBM.stencil as pyLBMSten
import pyLBM.domain as pyLBMDom
import pyLBM.scheme as pyLBMScheme
import pyLBM.simulation as pyLBMSimu
import pyLBM.boundary as pyLBMBoundary

import numba


X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return rhoo * np.ones((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qx(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((x.shape[0], y.shape[0]), dtype='float64')

@numba.jit("void(f8[:, :], f8[:, :], i8[:, :], f8[:], i4, i4)", nopython=True) 
def bb(fk, fksym, indices, s, vx, vy):
    n = indices.shape[1]
    
    for i in xrange(n):
        iy = indices[0, i]
        ix = indices[1, i]
        if s[i] < .5:
            t = 2*s[i]
            fk[ix, iy] = t*fk[ix + vx, iy + vy] + (1.-t)*fksym[ix + 2*vx, iy + 2*vy]
        else:
            t = .5/s[i]
            fk[ix, iy] = t*fksym[ix + vx, iy + vy] + (1.-t)*fk[ix + vx, iy + vy]
            

@profile
def bouzidi(f, bv, num2index):
    v = bv.v
    k = v.num
    vsym = v.get_symmetric()
    ksym = num2index[vsym.num]
    #ksym = num2index[v.get_symmetric().num]
    
    bb(f[k], f[ksym], bv.indices, bv.distance, v.vx, v.vy)

    # mask = bv.distance < .5
    # iy = bv.indices[0, mask]
    # ix = bv.indices[1, mask]
    # s = 2.*bv.distance[mask]
    # f[k, ix, iy] = s*f[ksym, ix + v.vx, iy + v.vy] + (1.-s)*f[ksym, ix + 2*v.vx, iy + 2*v.vy]
    # mask = np.logical_not(mask)
    # iy = bv.indices[0, mask]
    # ix = bv.indices[1, mask]
    # s = 0.5/bv.distance[mask]
    # f[k, ix, iy] = s*f[ksym, ix + v.vx, iy + v.vy] + (1.-s)*f[k, ix + v.vx, iy + v.vy]

def bc_up(m, x, y):
    m[0][0] = 0.
    m[0][1] = 0.1#*4*x*(1. - x) 
    m[0][2] = 0.

def plot_vorticite(sol):
    V = sol.m[0][2][2:,1:-1] - sol.m[0][2][0:-2,1:-1] - sol.m[0][1][1:-1,2:] + sol.m[0][1][1:-1,0:-2]
    V /= np.sqrt(V**2+1.e-5)
    plt.imshow(np.float32(V.transpose()), origin='lower', cmap=cm.gray)
    plt.title('Vorticity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

def plot_velocity(sol):
    """
    NV = (sol.m[0][1][1:-1,1:-1]**2 + sol.m[0][2][1:-1,1:-1]**2)
    plt.imshow(np.float32(NV.transpose()), origin='lower', cmap=cm.jet, interpolation='nearest')
    plt.title('Norme of the velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)
    """
    pas = 4
    plt.clf()
    X, Y = np.meshgrid(sol.domain.x[0][1:-1:pas], sol.domain.x[1][1:-1:pas])
    #Y, X = np.meshgrid(sol.Domain.x[1][1:-1:pas], sol.Domain.x[0][1:-1:pas])
    u = sol.m[0][1,1:-1:pas,1:-1:pas].transpose()
    v = sol.m[0][2,1:-1:pas,1:-1:pas].transpose()
    #normu = np.sqrt(u**2+v**2)
    normu = np.sqrt(sol.m[0][1,1:-1,1:-1]**2+sol.m[0][2,1:-1,1:-1]**2).max()
    nv = u**2+v**2
    #plt.streamplot(X, Y, u, v, color=normu, linewidth=2, cmap=plt.cm.autumn)
    plt.quiver(X, Y, u, v, nv, pivot='mid', scale=normu*10)
    plt.title('Velocity at t = {0:f}'.format(sol.t))
    plt.draw()
    plt.pause(1.e-3)

@profile
def one_time_step(sol):
    sol.scheme.relaxation(sol.m)
    sol.scheme.m2f(sol.m, sol.F)
    sol.scheme.set_boundary_conditions(sol.F, sol.m, sol.bc)
    sol.scheme.transport(sol.F)

    sol.scheme.f2m(sol.F, sol.m)

    #f2m_jit(f, m)
    
    #print (sol.m[0] == m).all()

    #print sol.m[0][:100]

if __name__ == "__main__":
    # parameters
    NbImages = 100 # number of figures
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.
    dx = 1./256 # spatial step
    la = 1. # velocity of the scheme
    Tf = 20
    rhoo = 1.
    mu   = 1.e-3 #0.00185
    zeta = 1.e-3
    dummy = 3.0/(la*rhoo*dx)
    s3 = 1.0/(0.5+zeta*dummy)
    s4 = s3
    s5 = s4
    s6 = s4
    s7 = 1.0/(0.5+mu*dummy)
    s8 = s7
    s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*u[0][1]**2
    qy2 = dummy*u[0][2]**2
    q2  = qx2+qy2
    qxy = dummy*u[0][1]*u[0][2]

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,0,1,0]},
        'space_step':dx,
        'number_of_schemes':1,
        'scheme_velocity':la,
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
                                  -u[0][1]/LA, -u[0][2]/LA,
                                  qx2-qy2, qxy]),
        },
        'init':{'type':'moments', 0:{0:(initialization_rho,),
                                     1:(initialization_qx,),
                                     2:(initialization_qy,)
                                     }
        },
        'boundary_conditions':{
            0:{'method':{0: bouzidi}, 'value':None},
            1:{'method':{0: bouzidi}, 'value':bc_up}
        }
    }


    #dom = pyLBMDom.Domain(dico)
    #print dom
    sol = pyLBMSimu.Simulation(dico)
    #print sol.scheme.code_m2f
    #print sol.scheme.code_f2m
    #print sol.scheme.code_transport
    print sol.scheme.code_relaxation
    #print sol.scheme.code_equilibrium

    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0

    for i in xrange(50):
        one_time_step(sol)

    #import inspect
    #print sol.scheme.code_f2m

    #f2m_jit = numba.jit("void(f8[:, :, :], f8[:, :, :])", nopython=True)(f2m)
