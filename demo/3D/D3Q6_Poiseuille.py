##############################################################################
# Solver D3Q6^4 for a Poiseuille flow
#
# d_t(p) + d_x(ux) + d_y(uy)  + d_z(uz)= 0
# d_t(ux) + d_x(ux^2) + d_y(ux*uy) + d_z(ux*uz) + d_x(p) = mu (d_xx+d_yy+d_zz)(ux)
# d_t(uy) + d_x(ux*uy) + d_y(uy^2) + d_z(uy*uz) + d_y(p) = mu (d_xx+d_yy+d_zz)(uy)
# d_t(uz) + d_x(ux*uz) + d_y(uy*uz) + d_z(uz^2) + d_z(p) = mu (d_xx+d_yy+d_zz)(uz)
#
# in a tunnel of width .5 and length 1. (periodic in z)
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
# uz = 0
# p = -C x with C = mu * umax * 8/L^2
#
# the variables of the four D3Q6 are p, ux, uy, and uz
# initialization with 0.
# boundary conditions
#     - ux=uy=uz=0. on bottom and top
#     - p given on left and right to constrain the pressure gradient
#     - ux, uy, and uz given on left to accelerate the convergence (optional)
#     - periodic conditions in z
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

from pyevtk.hl import imageToVTK
from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkFile, VtkRectilinearGrid


X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
rho, qx, qy, qz = sp.symbols('rho, qx, qy, qz')

def bc_in(f, m, x, y, z, scheme):
    ######### BEGIN OF WARNING #########
    # the order depends on the compilater
    # through the variable nv_on_beg
    m[:, 0] = (x-0.5*Longueur) * grad_pression *cte
    m[:, 4] = max_velocity * (1. - 4.*y**2/Largeur**2)
    m[:, 8] = 0.
    m[:, 12] = 0.
    #m[0, :] = (x-0.5*Longueur) * grad_pression *cte
    #m[4, :] = max_velocity * (1. - 4.*y**2/Largeur**2)
    #m[8, :] = 0.
    #m[12, :] = 0.
    #########  END OF WARNING  #########
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def bc_out(f, m, x, y, z, scheme):
    ######### BEGIN OF WARNING #########
    # the order depends on the compilater
    # through the variable nv_on_beg
    m[:, 0] = (x-0.5*Longueur) * grad_pression *cte
    m[:, 4] = 0.
    m[:, 8] = 0.
    m[:, 12] = 0.
    #m[0, :] = (x-0.5*Longueur) * grad_pression *cte
    #m[4, :] = 0.
    #m[8, :] = 0.
    #m[12, :] = 0.
    #########  END OF WARNING  #########
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def plot_coupe(sol, num):
    """
    nx, ny, nz = sol.domain.N
    x = sol.domain.x[1][1:-1]
    ya = sol.m[1][0][1, 1:-1, nz/2]
    yb = sol.m[1][0][nx/2, 1:-1, nz/2]
    yc = sol.m[1][0][-2, 1:-1, nz/2]
    y = sol.domain.x[0][1:-1]
    z = sol.m[1][0][1:-1, ny/2, nz/2]
    plt.clf()
    plt.hold(True)
    plt.plot(x + xmin, ya, 'k-', label='x={0}'.format(xmin))
    plt.plot(x + 0.5*(xmin+xmax), yb, 'r-', label='x={0}'.format((xmin+xmax)/2))
    plt.plot(x + xmax, yc, 'b-', label='x={0}'.format(xmax))
    plt.plot(y, z, 'g-', label='y={0}'.format((ymin+ymax)/2))
    plt.hold(False)
    plt.title("slice of the solution, t = {0}".format(sol.t))
    plt.legend(loc=0)
    """
    plt.imshow(sol.m[0][0][1:-1,1:-1,3].transpose())
    plt.draw()
    plt.pause(1.e-3)

def plot(sol, num):
    sol.time_info()
    nx, ny, nz = sol.domain.N

    start, end = (0, 0, 0), (nx-1, ny-1, nz-1)
    w = VtkFile("./data/image_{0}".format(num), VtkRectilinearGrid)
    w.openGrid(start = start, end = end)
    w.openPiece(start = start, end = end)

    pressure = sol.m[0][0][1:-1,1:-1,1:-1]
    x, y, z = sol.domain.x[0][1:-1], sol.domain.x[1][1:-1], sol.domain.x[2][1:-1]
    vx, vy, vz = sol.m[1][0][1:-1,1:-1,1:-1], sol.m[2][0][1:-1,1:-1,1:-1], sol.m[3][0][1:-1,1:-1,1:-1]

    pressure = pressure.ravel(order='F')
    vx = vx.ravel(order='F')
    vy = vy.ravel(order='F')
    vz = vz.ravel(order='F')

    # Point data
    w.openData("Point", scalars = "Pressure", vectors = "Velocity")
    w.addData("Pressure", pressure)
    w.addData("Velocity", (vx, vy, vz))
    w.closeData("Point")

    # Coordinates of cell vertices
    w.openElement("Coordinates")
    w.addData("x_coordinates", x);
    w.addData("y_coordinates", y);
    w.addData("z_coordinates", z);
    w.closeElement("Coordinates");

    w.closePiece()
    w.closeGrid()

    w.appendData(data = pressure)
    w.appendData(data = (vx,vy,vz))
    w.appendData(x).appendData(y).appendData(z)
    w.save()

def run(dico):
    sol = pyLBM.Simulation(dico)
    im = 0
    c = 0
    plot_coupe(sol,im)
    #plot(sol, im)
    while (sol.t<Tf):
        sol.one_time_step()
        c += 1
        if c == 16:
            im += 1
            sol.f2m()
            #plot_coupe(sol,im)
            plot(sol, im)
            c = 0
    plt.show()


    print "*"*50
    rho = sol.m[0][0][1:-1, 1:-1, 1:-1]
    qx = sol.m[1][0][1:-1, 1:-1, 1:-1]
    qy = sol.m[2][0][1:-1, 1:-1, 1:-1]
    qz = sol.m[3][0][1:-1, 1:-1, 1:-1]
    x = sol.domain.x[0][1:-1]
    y = sol.domain.x[1][1:-1]
    x = x[:, np.newaxis, np.newaxis]
    y = y[np.newaxis, :, np.newaxis]
    coeff = sol.domain.dx / np.sqrt(Largeur*Longueur)
    Err_rho = coeff * np.linalg.norm(rho - (x-0.5*Longueur) * grad_pression)
    Err_qx = coeff * np.linalg.norm(qx - max_velocity * (1 - 4 * y**2 / Largeur**2))
    Err_qy = coeff * np.linalg.norm(qy)
    Err_qz = coeff * np.linalg.norm(qz)
    print "Norm of the error on rho: {0:10.3e}".format(Err_rho)
    print "Norm of the error on qx:  {0:10.3e}".format(Err_qx)
    print "Norm of the error on qy:  {0:10.3e}".format(Err_qy)
    print "Norm of the error on qz:  {0:10.3e}".format(Err_qz)

    plt.figure(2)
    plt.clf()
    plt.imshow(np.float32(qx - max_velocity * (1 - 4 * y**2 / Largeur**2)), origin='lower', cmap=cm.gray)
    plt.colorbar()
    plt.show()

    sol.time_info()

if __name__ == "__main__":
    # parameters
    Tf = 50.
    Longueur = 1.
    Largeur = .5
    xmin, xmax, ymin, ymax = 0., Longueur, -.5*Largeur, .5*Largeur
    dx = 1./64 # spatial step
    zmin, zmax = -2*dx, 2*dx
    la = 1. # velocity of the scheme
    max_velocity = 0.1
    mu   = 1.e-3
    zeta = 1.e-5
    grad_pression = -mu * max_velocity * 8./Largeur**2
    cte = 10.

    dummy = 3.0/(la*dx)
    #s1 = 1.0/(0.5+zeta*dummy)
    #s2 = 1.0/(0.5+mu*dummy)
    sigma = 1./np.sqrt(12)
    s = 1./(.5+sigma)
    vs = [0., s, s, s, s, s]


    vitesse = range(1, 7)
    polynomes = Matrix([1, LA*X, LA*Y, LA*Z, X**2-Y**2, X**2-Z**2])

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':[1, 2, 0, 0, -1, -1]},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':vs,
                    'equilibrium':[rho, qx, qy, qz, 0., 0.],
                    'init':{rho: 0.},
                    'conserved_moments': rho,
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':vs,
                    'equilibrium':[qx, qx**2 + rho/cte, qx*qy, qx*qz, 0., 0.],
                    'init':{qx: 0.},
                    'conserved_moments': qx,
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':vs,
                    'equilibrium':[qy, qx*qy, qy**2 + rho/cte, qy*qz, 0., 0.],
                    'init':{qy: 0.},
                    'conserved_moments': qy,
                    },
                    {'velocities':vitesse,
                    'polynomials':polynomes,
                    'relaxation_parameters':vs,
                    'equilibrium':[qz, qx*qz, qy*qz, qz**2 + rho/cte, 0., 0.],
                    'init':{qz: 0.},
                    'conserved_moments': qz,
                    },
        ],
        'parameters':{'LA': la},
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.bouzidi_bounce_back,
                         1: pyLBM.bc.bouzidi_anti_bounce_back,
                         2: pyLBM.bc.bouzidi_anti_bounce_back,
                         3: pyLBM.bc.bouzidi_anti_bounce_back,
                         },
                'value':None,
            },
            1:{'method':{0: pyLBM.bc.bouzidi_anti_bounce_back,
                         1: pyLBM.bc.neumann_vertical,
                         2: pyLBM.bc.neumann_vertical,
                         3: pyLBM.bc.neumann_vertical,
                         },
                'value':bc_out,
            },
            2:{'method':{0: pyLBM.bc.bouzidi_anti_bounce_back,
                         1: pyLBM.bc.bouzidi_anti_bounce_back,
                         2: pyLBM.bc.bouzidi_anti_bounce_back,
                         3: pyLBM.bc.bouzidi_anti_bounce_back,
                         },
                'value':bc_in,
            },
        },
        'generator': pyLBM.generator.CythonGenerator,
    }

    run(dico)
