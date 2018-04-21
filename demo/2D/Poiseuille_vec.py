from __future__ import print_function
from __future__ import division
"""
 Solver D2Q(4,4,4) for a Poiseuille flow

 d_t(p) + d_x(ux) + d_y(uy) = 0
 d_t(ux) + d_x(ux^2) + d_y(ux*uy) + d_x(p) = mu (d_xx+d_yy)(ux)
 d_t(uy) + d_x(ux*uy) + d_y(uy^2) + d_y(p) = mu (d_xx+d_yy)(uy)

 in a tunnel of width .5 and length 1.

   ------------------------------------
       ->      ->      ->      ->
       -->     -->     -->     -->
       ->      ->      ->      ->
   ------------------------------------

 the solution is
 ux = umax (1 - 4 * (y/L)^2) if L is the width of the tunnel
 uy = 0
 p = -C x with C = mu * umax * 8/L^2

 the variables of the three D2Q4 are p, ux, and uy
 initialization with 0.
 boundary conditions
     - ux=uy=0. on bottom and top
     - p given on left and right to constrain the pressure gradient
     - ux and uy given on left to accelerate the convergence (optional)

 test: True
"""
from six.moves import range
import numpy as np
import sympy as sp
import pylbm

X, Y, LA = sp.symbols('X, Y, LA')
p, ux, uy = sp.symbols('p, ux, uy')

def bc_in(f, m, x, y, width, height, max_velocity, grad_pressure, cte):
    m[p] = (x-0.5*width) * grad_pressure * cte
    m[ux] = max_velocity * (1. - 4.*y**2/height**2)

def bc_out(f, m, x, y, width, grad_pressure, cte):
    m[p] = (x-0.5*width) * grad_pressure * cte

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
    la = 1. # velocity of the scheme
    max_velocity = 0.1
    mu   = 0.00185
    zeta = 1.e-5
    grad_pressure = -mu * max_velocity * 8./height**2
    cte = 3.

    dummy = 3.0/(la*dx)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)

    velocities = list(range(1, 5))
    polynomes = [1, LA*X, LA*Y, X**2-Y**2]

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[2, 1, 0, 0]},
        'space_step':dx,
        'scheme_velocity':la,
        'inittype': 'moments',
        'schemes':[{'velocities': velocities,
                    'polynomials': polynomes,
                    'relaxation_parameters': [0., s1, s1, 1.],
                    'equilibrium': [p, ux, uy, 0.],
                    'conserved_moments': p,
                    'init': {p: 0.},
                    },
                    {'velocities': velocities,
                    'polynomials': polynomes,
                    'relaxation_parameters': [0., s2, s2, 1.],
                    'equilibrium': [ux, ux**2 + p/cte, ux*uy, 0.],
                    'conserved_moments': ux,
                    'init': {ux: 0.},
                    },
                    {'velocities': velocities,
                    'polynomials': polynomes,
                    'relaxation_parameters': [0., s2, s2, 1.],
                    'equilibrium': [uy, ux*uy, uy**2 + p/cte, 0.],
                    'conserved_moments': uy,
                    'init': {uy: 0.},
                    },
        ],
        'parameters': {LA: la},
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Bouzidi_bounce_back,
                         1: pylbm.bc.Bouzidi_anti_bounce_back,
                         2: pylbm.bc.Bouzidi_anti_bounce_back
                         },
            },
            1:{'method':{0: pylbm.bc.Bouzidi_anti_bounce_back,
                         1: pylbm.bc.Neumann_x,
                         2: pylbm.bc.Neumann_x
                         },
                'value':(bc_out, (width, grad_pressure, cte))
            },
            2:{'method':{0: pylbm.bc.Bouzidi_anti_bounce_back,
                         1: pylbm.bc.Bouzidi_anti_bounce_back,
                         2: pylbm.bc.Bouzidi_anti_bounce_back
                         },
                'value':(bc_in, (width, height, max_velocity, grad_pressure, cte)),
            },
        },
        'generator': generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    while sol.t<Tf:
        sol.one_time_step()

    if withPlot:
        print("*"*50)
        p_n = sol.m[p]
        ux_n = sol.m[ux]
        uy_n = sol.m[uy]
        x, y = np.meshgrid(*sol.domain.coords, sparse=True, indexing='ij')
        coeff = sol.domain.dx / np.sqrt(width*height)
        Err_p = coeff * np.linalg.norm(p_n - (x-0.5*width) * grad_pressure)
        Err_ux = coeff * np.linalg.norm(ux_n - max_velocity * (1 - 4 * y**2 / height**2))
        Err_uy = coeff * np.linalg.norm(uy_n)
        print("Norm of the error on rho: {0:10.3e}".format(Err_p))
        print("Norm of the error on qx:  {0:10.3e}".format(Err_ux))
        print("Norm of the error on qy:  {0:10.3e}".format(Err_uy))

        # init viewer
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]

        ax.image(ux_n - max_velocity * (1 - 4 * y**2 / height**2))
        ax.title = "Error on ux"
        fig.show()

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 200.
    run(dx, Tf)
