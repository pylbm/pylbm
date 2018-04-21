from __future__ import print_function
from __future__ import division
"""
 Solver D3Q6 for the advection equation on the 3D-torus

 d_t(u) + cx d_x(u) + cy d_y(u) + c_z d_z(u) = 0, t > 0, 0 < x,y,z < 1,
 u(t=0,x,y,z) = u0(x,y,z),
 u(t,x=0,y,z) = u(t,x=1,y,z) 0 < y,z < 1,
 u(t,x,y=0,z) = u(t,x,y=1,z) 0 < x,z < 1,
 u(t,x,y,z=0) = u(t,x,y,z=1) 0 < x,y < 1,

 the solution is
 u(t,x,y,z) = u0(x-cx*t,y-cy*t,z-cz*t)

 test: True
"""
from six.moves import range
import numpy as np
import sympy as sp
import pylbm

u, X, Y, Z, LA = sp.symbols('u, X, Y, Z, LA')

def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pylbm.H5File(sol.mpi_topo, 'advection', './advection', im)
    h5.set_grid(x, y, z)
    h5.add_scalar('u', sol.m[u])
    h5.save()

def run(dx, Tf, generator="cython", sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    sorder: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # advective velocity
    ux, uy, uz = .5, .2, .1
    # domain of the computation
    xmin, xmax, ymin, ymax, zmin, zmax = 0., 1., 0., 1., 0., 1.

    def u0(x, y, z):
        xm, ym, zm = .5*(xmin+xmax), .5*(ymin+ymax), .5*(zmin+zmax)
        return .5*np.ones((x.size, y.size, z.size)) \
              + .5*(((x-xm)**2+(y-ym)**2+(z-zm)**2)<.25**2)

    s = 1.
    la = 1.

    d = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{
            'velocities': list(range(1,7)),
            'conserved_moments':[u],
            'polynomials': [1, LA*X, LA*Y, LA*Z, X**2-Y**2, X**2-Z**2],
            'equilibrium': [u, ux*u, uy*u, uz*u, 0., 0.],
            'relaxation_parameters': [0., s, s, s, s, s],
            'init':{u:(u0,)},
        },],
        'parameters': {LA: la},
        'generator': generator,
    }

    sol = pylbm.Simulation(d, sorder=sorder)

    im = 0
    while sol.t < Tf:
        sol.one_time_step()

        if withPlot:
            im += 1
            save(sol, im)

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 1.
    run(dx, Tf)
