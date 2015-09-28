import pyLBM
import sympy as sp
import math
import mpi4py.MPI as mpi

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
mass, qx, qy, qz = sp.symbols('mass,qx,qy,qz')

def bc_up(f, m, x, y, z):
    m[3] = .01
    m[5] = 0.
    m[7] = 0.

def save(x, y, z, m, im):
    init_pvd = False
    if im == 1:
        init_pvd = True

    vtk = pyLBM.VTKFile('karman', './data', im, init_pvd=init_pvd)
    vtk.set_grid(x, y, z)
    vtk.add_scalar('mass', m[0][0])
    qx, qy, qz = m[0][3], m[0][5], m[0][7]
    vtk.add_vector('velocity', [qx, qy, qz])
    vtk.save()

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
    'box':{'x':[0., 2.], 'y':[0., 1.], 'z':[0., 1.], 'label':[1, 2, -1, -1, -1, -1]},
    'elements':[pyLBM.Sphere((.3,.5,.5), 0.125, 0)],
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
            mass:1.,
            qx: 0.,
            qy: 0.,
            qz: 0.
        },
    }],
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}},
        1:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}, 'value':bc_up},
        2:{'method':{0: pyLBM.bc.Neumann_vertical}},
    },
    'parameters': {LA: la},
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)

x, y, z = sol.domain.x[0], sol.domain.x[1], sol.domain.x[2]

im = 0
compt = 0
while sol.t < 200.:
    sol.one_time_step()
    compt += 1
    if compt == 100:
        if mpi.COMM_WORLD.Get_rank() == 0:
            print sol.time_info()
        im += 1
        save(x, y, z, sol.m, im)
        compt = 0
