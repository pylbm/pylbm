##############################################################################
#
# Solver D3Q6 for the advection equation on the 3D-torus
#
# d_t(u) + cx d_x(u) + cy d_y(u) + c_z d_z(u) = 0, t > 0, 0 < x,y,z < 1,
# u(t=0,x,y,z) = u0(x,y,z),
# u(t,x=0,y,z) = u(t,x=1,y,z) 0 < y,z < 1,
# u(t,x,y=0,z) = u(t,x,y=1,z) 0 < x,z < 1,
# u(t,x,y,z=0) = u(t,x,y,z=1) 0 < x,y < 1,
#
# the solution is
# u(t,x,y,z) = u0(x-cx*t,y-cy*t,z-cz*t)
#
##############################################################################
import numpy as np
import sympy as sp
import pyLBM

u, X, Y, Z, LA = sp.symbols('u,X,Y,Z,LA')

# advective velocity
ux, uy, uz = .5, .2, .1
# domain of the computation
xmin, xmax, ymin, ymax, zmin, zmax = 0., 1., 0., 1., 0., 1.

def u0(x, y, z):
    xm, ym, zm = .5*(xmin+xmax), .5*(ymin+ymax), .5*(zmin+zmax)
    return .5*np.ones((x.size, y.size, z.size)) \
          + .5*(((x-xm)**2+(y-ym)**2+(z-zm)**2)<.25**2)

def save(x, y, z, m, num):
    vtk = pyLBM.VTKFile(filename + '_{0}'.format(num), path)
    vtk.set_grid(x, y, z)
    vtk.add_scalar('u', m[0][0][1:-1,1:-1,1:-1])
    vtk.save()

s = 1.
la = 1.
dx = 1./128
d = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':-1},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[{
        'velocities': range(1,7),
        'conserved_moments':[u],
        'polynomials': [1, LA*X, LA*Y, LA*Z, X**2-Y**2, X**2-Z**2],
        'equilibrium': [u, ux*u, uy*u, uz*u, 0., 0.],
        'relaxation_parameters': [0., s, s, s, s, s],
        'init':{u:(u0,)},
    },],
    'parameters': {LA: la},
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(d)

x, y, z = sol.domain.x[0][1:-1], sol.domain.x[1][1:-1], sol.domain.x[2][1:-1]

filename = 'advection'
path = './data'

im = 0
while sol.t<1.:
    sol.one_time_step()
    sol.f2m()
    save(x, y, z, sol.m, im)
    im += 1

pyLBM.write_collection(filename, path, im)
