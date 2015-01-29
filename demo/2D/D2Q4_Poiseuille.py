##############################################################################
# Solver D2Q4^3 for a Poiseuille flow
#
# d_t(p) + d_x(ux) + d_y(uy) = 0
# d_t(ux) + d_x(ux^2) + d_y(ux*uy) + d_x(p) = mu (d_xx+d_yy)(ux)
# d_t(uy) + d_x(ux*uy) + d_y(uy^2) + d_y(p) = mu (d_xx+d_yy)(uy)
#
# in a tunnel of width .5 and length 1.
#
#   ------------------------------------
#       ->      ->      ->      ->
#       -->     -->     -->     -->
#       ->      ->      ->      ->
#   ------------------------------------
#
# the solution is
# ux = umax (1 - 4 * (y/L)^2) if L is the width of the tunnel
# uy = 0
# p = -C x with C = mu * umax * 8/L^2
#
# the variables of the three D2Q4 are p, ux, and uy
# initialization with 0.
# boundary conditions
#     - ux=uy=0. on bottom and top
#     - p given on left and right to constrain the pressure gradient
#     - ux and uy given on left to accelerate the convergence (optional)
#
##############################################################################

import sys
import time
import os

import cmath
from math import pi, sqrt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi

import pyLBM

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')

u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def initialization_rho(x,y):
    return np.zeros((y.size, x.size), dtype='float64')

def initialization_qx(x,y):
    return np.zeros((y.size, x.size), dtype='float64')

def initialization_qy(x,y):
    return np.zeros((y.size, x.size), dtype='float64')

def bc_in(f, m, x, y, scheme):
    ######### BEGIN OF WARNING #########
    # the order depends on the compilater
    # through the variable nv_on_beg
    #m[:, 0] = rhoo + (xmin-0.5*Longueur) * grad_pression
    #m[:, 4] = rhoo*max_velocity * (1. - 4.*y**2/Largeur**2)
    #m[:, 8] = 0.
    m[0, :] = (x-0.5*Longueur) * grad_pression *cte
    m[4, :] = max_velocity * (1. - 4.*y**2/Largeur**2)
    m[8, :] = 0.
    #########  END OF WARNING  #########
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def bc_out(f, m, x, y, scheme):
    ######### BEGIN OF WARNING #########
    # the order depends on the compilater
    # through the variable nv_on_beg
    #m[:, 0] = rhoo + (xmax-0.5*Longueur) * grad_pression
    #m[:, 4] = 0.
    #m[:, 8] = 0.
    m[0, :] = (x-0.5*Longueur) * grad_pression *cte
    m[4, :] = 0.
    m[8, :] = 0.
    #########  END OF WARNING  #########
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def plot_coupe(sol, num):
    nx, ny = sol.domain.N
    x = sol.domain.x[1][1:-1]
    ya = sol.m[1][0][1:-1, 1]
    yb = sol.m[1][0][1:-1, nx/2]
    yc = sol.m[1][0][1:-1, -2]
    y = sol.domain.x[0][1:-1]
    z = sol.m[1][0][ny/2, 1:-1]
    plt.clf()
    plt.hold(True)
    plt.plot(x + xmin, ya, 'k-', label='x={0}'.format(xmin))
    plt.plot(x + 0.5*(xmin+xmax), yb, 'r-', label='x={0}'.format((xmin+xmax)/2))
    plt.plot(x + xmax, yc, 'b-', label='x={0}'.format(xmax))
    plt.plot(y, z, 'g-', label='y={0}'.format((ymin+ymax)/2))
    plt.hold(False)
    plt.title("slice of the solution, t = {0}".format(sol.t))
    plt.legend(loc=0)
    plt.draw()
    plt.pause(1.e-3)

def run(dico):
    sol = pyLBM.Simulation(dico)
    im = 0
    c = 0
    plot_coupe(sol,im)
    while (sol.t<Tf):
        sol.one_time_step()
        c += 1
        if c == 128:
            im += 1
            sol.f2m()
            plot_coupe(sol,im)
            c = 0
    plt.show()


    print "*"*50
    rho = sol.m[0][0][1:-1, 1:-1]
    qx = sol.m[1][0][1:-1, 1:-1]
    qy = sol.m[2][0][1:-1, 1:-1]
    x = sol.domain.x[0][1:-1]
    y = sol.domain.x[1][1:-1]
    x = x[np.newaxis, :]
    y = y[:, np.newaxis]
    coeff = sol.domain.dx / np.sqrt(Largeur*Longueur)
    Err_rho = coeff * np.linalg.norm(rho - (x-0.5*Longueur) * grad_pression)
    Err_qx = coeff * np.linalg.norm(qx - max_velocity * (1 - 4 * y**2 / Largeur**2))
    Err_qy = coeff * np.linalg.norm(qy)
    print "Norm of the error on rho: {0:10.3e}".format(Err_rho)
    print "Norm of the error on qx:  {0:10.3e}".format(Err_qx)
    print "Norm of the error on qy:  {0:10.3e}".format(Err_qy)

    plt.figure(2)
    plt.clf()
    plt.imshow(np.float32(qx - max_velocity * (1 - 4 * y**2 / Largeur**2)), origin='lower', cmap=cm.gray)
    plt.colorbar()
    plt.show()

    print "*"*50
    ttot = sol.cpu_time['total']
    print "total:      {0:10.3e}".format(ttot)
    t = sol.cpu_time['relaxation']
    print "relaxation: {0:10.3e} = {1:4.2f}%".format(t, t/ttot*100)
    t = sol.cpu_time['transport']
    print "transport:  {0:10.3e} = {1:4.2f}%".format(t, t/ttot*100)
    t = sol.cpu_time['f2m_m2f']
    print "f2m, m2f:   {0:10.3e} = {1:4.2f}%".format(t, t/ttot*100)
    t = sol.cpu_time['boundary_conditions']
    print "bouzidi:    {0:10.3e} = {1:4.2f}%".format(t, t/ttot*100)
    print "MLUPS: {0:5.1f}".format(sol.cpu_time['MLUPS'])
    print "*"*50

if __name__ == "__main__":
    # parameters
    Tf = 200.
    Longueur = 1.
    Largeur = .5
    xmin, xmax, ymin, ymax = 0., Longueur, -.5*Largeur, .5*Largeur
    dx = 1./64 # spatial step
    la = 1. # velocity of the scheme
    max_velocity = 0.1
    mu   = 0.00185
    zeta = 1.e-5
    #grad_pression = - max_velocity * 8.0 / (Largeur)**2 * 3. /(la**2) * mu
    grad_pression = -mu * max_velocity * 8./Largeur**2
    cte = 3.

    dummy = 3.0/(la*dx)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)

    vitesse = range(1, 5)
    polynomes = Matrix([1, LA*X, LA*Y, X**2-Y**2])

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0, 1, 0, 2]},
        'space_step':dx,
        'scheme_velocity':la,
        'inittype': 'moments',
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., s1, s1, 1.],
                    'equilibrium':Matrix([u[0][0], u[1][0], u[2][0], 0.]),
                    'init':{0:(initialization_rho,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., s2, s2, 1.],
                    'equilibrium':Matrix([u[1][0], u[1][0]**2 + u[0][0]/cte, u[1][0]*u[2][0], 0.]),
                    'init':{0:(initialization_qx,)},
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':[0., s2, s2, 1.],
                    'equilibrium':Matrix([u[2][0], u[1][0]*u[2][0], u[2][0]**2 + u[0][0]/cte, 0.]),
                    'init':{0:(initialization_qy,)},
                    },
        ],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.bouzidi_bounce_back,
                         1: pyLBM.bc.bouzidi_anti_bounce_back,
                         2: pyLBM.bc.bouzidi_anti_bounce_back
                         },
                'value':None,
            },
            1:{'method':{0: pyLBM.bc.bouzidi_anti_bounce_back,
                         1: pyLBM.bc.neumann_vertical,
                         2: pyLBM.bc.neumann_vertical
                         },
                'value':bc_out,
            },
            2:{'method':{0: pyLBM.bc.bouzidi_anti_bounce_back,
                         1: pyLBM.bc.bouzidi_anti_bounce_back,
                         2: pyLBM.bc.bouzidi_anti_bounce_back
                         },
                'value':bc_in,
            },
        },
    }

    run(dico)
