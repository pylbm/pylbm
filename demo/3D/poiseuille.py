from __future__ import print_function
from __future__ import division
"""
 Solver D3Q6^4 for a Poiseuille flow

 d_t(p) + d_x(ux) + d_y(uy)  + d_z(uz)= 0
 d_t(ux) + d_x(ux^2) + d_y(ux*uy) + d_z(ux*uz) + d_x(p) = mu (d_xx+d_yy+d_zz)(ux)
 d_t(uy) + d_x(ux*uy) + d_y(uy^2) + d_z(uy*uz) + d_y(p) = mu (d_xx+d_yy+d_zz)(uy)
 d_t(uz) + d_x(ux*uz) + d_y(uy*uz) + d_z(uz^2) + d_z(p) = mu (d_xx+d_yy+d_zz)(uz)

 in a tunnel of width .5 and length 1. (periodic in z)

   ------------------------------------
       ->      ->      ->      ->
       -->     -->     -->     -->
       ->      ->      ->      ->
   ------------------------------------

 the solution is
 ux = umax (1 - 4 * (y/L)^2) if L is the width of the tunnel
 uy = 0
 uz = 0
 p = -C x with C = mu * umax * 8/L^2

 the variables of the four D3Q6 are p, ux, uy, and uz
 initialization with 0.
 boundary conditions
     - ux=uy=uz=0. on bottom and top
     - p given on left and right to constrain the pressure gradient
     - ux, uy, and uz given on left to accelerate the convergence (optional)
     - periodic conditions in z

 test: True
"""
from six.moves import range
import numpy as np
import sympy as sp

import pylbm

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
p, ux, uy, uz = sp.symbols('p,ux,uy,uz')


def save(sol, num):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z

    h5 = pylbm.H5File(sol.mpi_topo, 'poiseuille', './poiseuille', num)
    h5.set_grid(x, y, z)
    h5.add_scalar('pressure', sol.m[p])
    qx_n, qy_n, qz_n = sol.m[ux], sol.m[uy], sol.m[uz]
    h5.add_vector('velocity', [qx_n, qy_n, qz_n])
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
    # parameters
    width = 1.
    height = .5
    xmin, xmax, ymin, ymax = 0., width, -.5*height, .5*height
    zmin, zmax = -2*dx, 2*dx
    la = 1. # velocity of the scheme
    max_velocity = 0.1
    mu   = 1.e-3
    zeta = 1.e-5
    grad_pressure = -mu * max_velocity * 8./height**2
    cte = 10.

    dummy = 3.0/(la*dx)
    #s1 = 1.0/(0.5+zeta*dummy)
    #s2 = 1.0/(0.5+mu*dummy)
    sigma = 1./np.sqrt(12)
    s = 1./(.5+sigma)
    vs = [0., s, s, s, s, s]


    velocities = list(range(1, 7))
    polynomes = [1, LA*X, LA*Y, LA*Z, X**2-Y**2, X**2-Z**2]

    def bc_in(f, m, x, y, z):
        m[p] = (x-0.5*width) * grad_pressure *cte
        m[ux] = max_velocity * (1. - 4.*y**2/height**2)

    def bc_out(f, m, x, y, z):
        m[p] = (x-0.5*width) * grad_pressure *cte

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':[1, 2, 0, 0, -1, -1]},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{
            'velocities':velocities,
            'conserved_moments':p,
            'polynomials':polynomes,
            'relaxation_parameters':vs,
            'equilibrium':[p, ux, uy, uz, 0., 0.],
            'init':{p:0.},
            },{
            'velocities':velocities,
            'conserved_moments':ux,
            'polynomials':polynomes,
            'relaxation_parameters':vs,
            'equilibrium':[ux, ux**2 + p/cte, ux*uy, ux*uz, 0., 0.],
            'init':{ux:0.},
            },{
            'velocities':velocities,
            'conserved_moments':uy,
            'polynomials':polynomes,
            'relaxation_parameters':vs,
            'equilibrium':[uy, uy*ux, uy**2 + p/cte, uy*uz, 0., 0.],
            'init':{uy:0.},
            },{
            'velocities':velocities,
            'conserved_moments':uz,
            'polynomials':polynomes,
            'relaxation_parameters':vs,
            'equilibrium':[uz, uz*ux, uz*uy, uz**2 + p/cte, 0., 0.],
            'init':{uz:0.},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Bouzidi_bounce_back,
                         1: pylbm.bc.Bouzidi_anti_bounce_back,
                         2: pylbm.bc.Bouzidi_anti_bounce_back,
                         3: pylbm.bc.Bouzidi_anti_bounce_back,
                         },
            },
            1:{'method':{0: pylbm.bc.Bouzidi_anti_bounce_back,
                         1: pylbm.bc.Neumann_x,
                         2: pylbm.bc.Neumann_x,
                         3: pylbm.bc.Neumann_x,
                         },
                'value':bc_out,
            },
            2:{'method':{0: pylbm.bc.Bouzidi_anti_bounce_back,
                         1: pylbm.bc.Bouzidi_anti_bounce_back,
                         2: pylbm.bc.Bouzidi_anti_bounce_back,
                         3: pylbm.bc.Bouzidi_anti_bounce_back,
                         },
                'value':bc_in,
            },
        },
        'parameters': {LA: la},
        'generator': generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    im = 0
    while sol.t < Tf:
        nrep = 100
        for i in range(nrep):
                 sol.one_time_step()
        im += 1
        save(sol, im)
    return sol

if __name__ == '__main__':
    dx = 1./256
    Tf= 50.
    run(dx, Tf)
