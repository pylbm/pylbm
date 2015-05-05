import pyLBM
from pyLBM.viewer import VtkViewer
import math
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
from pyevtk.hl import imageToVTK, gridToVTK
from pyevtk.vtk import VtkFile, VtkRectilinearGrid


X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
mass, qx, qy, qz = sp.symbols('mass,qx,qy,qz')

def initialization_rho(x, y, z):
    return np.ones((x.size, y.size, z.size))

def initialization_q(x, y, z):
    return np.zeros((x.size, y.size, z.size))

def bc_up(f, m, x, y, z, scheme):
    m[:, 3] = -math.sqrt(2)/20.
    m[:, 5] = -math.sqrt(2)/20.
    m[:, 7] = 0.
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def plot(sol, num):
    sol.time_info()
    sol.f2m()
    nx, ny, nz = sol.domain.N

    start, end = (0, 0, 0), (nx-1, ny-1, nz-1)
    w = VtkFile("./data/image_{0}".format(num), VtkRectilinearGrid)
    w.openGrid(start = start, end = end)
    w.openPiece(start = start, end = end)

    mass = sol.m[0][0][1:-1,1:-1,1:-1]
    x, y, z = sol.domain.x[0][1:-1], sol.domain.x[1][1:-1], sol.domain.x[2][1:-1]
    qx, qy, qz = sol.m[0][3][1:-1,1:-1,1:-1], sol.m[0][5][1:-1,1:-1,1:-1], sol.m[0][7][1:-1,1:-1,1:-1]

    mass = mass.ravel(order='F')
    qx = qx.ravel(order='F')
    qy = qy.ravel(order='F')
    qz = qz.ravel(order='F')

    # Point data
    w.openData("Point", scalars = "Mass", vectors = "Velocity")
    w.addData("Mass", mass)
    w.addData("Velocity", (qx, qy, qz))
    w.closeData("Point")

    # Coordinates of cell vertices
    w.openElement("Coordinates")
    w.addData("x_coordinates", x);
    w.addData("y_coordinates", y);
    w.addData("z_coordinates", z);
    w.closeElement("Coordinates");

    w.closePiece()
    w.closeGrid()

    w.appendData(data = mass)
    w.appendData(data = (qx,qy,qz))
    w.appendData(x).appendData(y).appendData(z)
    w.save()

dx = 1./128
la = 1.
rho0 = 1.
Re = 200
nu = 5./Re

s1 = 1.6
s2 = 1.2
s4 = 1.6
s9 = 1./(3*nu +.5)
s11 = s9
s14 = 1.2

r = X**2+Y**2+Z**2

dico = {
    'box':{'x':[0., 1.], 'y':[0., 1.], 'z':[0., 1.], 'label':[0, 0, 0, 0, 0, 1]},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[{
        'velocities':range(7) + range(19,27),
        'conserved_moments':[mass, qx, qy, qz],
        'polynomials':[
            1,
            r - 2, .5*(15*r**2-55*r+32),
            X, .5*(5*r-13)*X,
            Y, .5*(5*r-13)*Y,
            Z, .5*(5*r-13)*Z,
            3*X**2-r, Y**2-Z**2,
            X*Y, Y*Z, Z*X,
            X*Y*Z
        ],
        'relaxation_parameters':[0, s1, s2, 0, s4, 0, s4, 0, s4, s9, s9, s11, s11, s11, s14],
        'equilibrium':[
            mass,
            -mass + qx**2 + qy**2 + qz**2,
            -mass,
            qx,
            -7./3*qx,
            qy,
            -7./3*qy,
            qz,
            -7./3*qz,
            1./3*(2*qx**2-(qy**2+qz**2)),
            qy**2-qz**2,
            qx*qy,
            qy*qz,
            qz*qx,
            0
        ],
        'init':{
            mass:(initialization_rho,),
            qx:(initialization_q,),
            qy:(initialization_q,),
            qz:(initialization_q,)
        },
    }],
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.bouzidi_bounce_back}, 'value':None},
        1:{'method':{0: pyLBM.bc.bouzidi_bounce_back}, 'value':bc_up},
    },
    'parameters': {LA: la},
    'generator': pyLBM.generator.CythonGenerator,
}

## verification
# geom = pyLBM.Geometry(dico)
# geom.visualize(viewlabel=True)
# sten = pyLBM.Stencil(dico)
# print sten
# v = VtkViewer()
# sten.visualize(v)
# dom = pyLBM.Domain(dico)
# dom.visualize(opt=1)
# print dom


sol = pyLBM.Simulation(dico)
nx, ny, nz = sol.domain.Ng
sol.save = np.empty((nx, ny, nz))

im = 0
compt = 0
plot(sol,im)

while sol.t<5.:
    sol.one_time_step()
    compt += 1
    if compt == 16:
        im += 1
        compt = 0
        plot(sol,im)

sol.time_info()
