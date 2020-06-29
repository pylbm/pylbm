# pylint: disable=invalid-name

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

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
import numpy as np
import sympy as sp
import pylbm

# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=invalid-name

X, Y, LA = sp.symbols('X, Y, lambda')
P, UX, UY = sp.symbols('p, ux, uy')


def bc_in(f, m, x, y, width, height, max_velocity, grad_pressure, cte):
    """ inner boundary condition """
    m[P] = (x-0.5*width) * grad_pressure * cte
    m[UX] = max_velocity * (1. - 4.*y**2/height**2)


def bc_out(f, m, x, y, width, grad_pressure, cte):
    """ outer boundary condition """
    m[P] = (x-0.5*width) * grad_pressure * cte


def run(space_step,
        final_time,
        generator="cython",
        sorder=None,
        with_plot=True):
    """
    Parameters
    ----------

    space_step: double
        spatial step

    final_time: double
        final time

    generator: string
        pylbm generator

    sorder: list
        storage order

    with_plot: boolean
        if True plot the solution otherwise just compute the solution


    Returns
    -------

    sol
        <class 'pylbm.simulation.Simulation'>

    """
    # parameters
    la = 1                # lattice velocity
    width = 1             # width of the domain
    height = .5           # height of the domain
    max_velocity = 0.1    # reference of the maximal velocity
    rhoo = 1              # reference value of the density
    mu = 0.00185          # bulk viscosity
    zeta = 1.e-5          # shear viscosity
    cte = 3.

    xmin, xmax, ymin, ymax = 0., width, -.5*height, .5*height
    grad_pressure = -mu * max_velocity * 8./height**2

    dummy = 3.0/(la*space_step)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)

    velocities = list(range(1, 5))
    polynomes = [1, X, Y, X**2-Y**2]

    simu_cfg = {
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [2, 1, 0, 0]
        },
        'space_step':space_step,
        'scheme_velocity':la,
        'schemes':[{'velocities': velocities,
                    'polynomials': polynomes,
                    'relaxation_parameters': [0., s1, s1, 1.],
                    'equilibrium': [p, ux, uy, 0.],
                    'conserved_moments': p,
                    },
                    {'velocities': velocities,
                    'polynomials': polynomes,
                    'relaxation_parameters': [0., s2, s2, 1.],
                    'equilibrium': [ux, ux**2 + p/cte, ux*uy, 0.],
                    'conserved_moments': ux,
                    },
                    {'velocities': velocities,
                    'polynomials': polynomes,
                    'relaxation_parameters': [0., s2, s2, 1.],
                    'equilibrium': [uy, ux*uy, uy**2 + p/cte, 0.],
                    'conserved_moments': uy,
                    },
        ],
        'parameters': {LA: la},
        'init': {p: 0.,
                 ux: 0.,
                 uy: 0.},
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.BouzidiBounceBack,
                         1: pylbm.bc.BouzidiAntiBounceBack,
                         2: pylbm.bc.BouzidiAntiBounceBack
                         },
            },
            1:{'method':{0: pylbm.bc.BouzidiAntiBounceBack,
                         1: pylbm.bc.NeumannX,
                         2: pylbm.bc.NeumannX
                         },
                'value':(bc_out, (width, grad_pressure, cte))
            },
            2:{'method':{0: pylbm.bc.BouzidiAntiBounceBack,
                         1: pylbm.bc.BouzidiAntiBounceBack,
                         2: pylbm.bc.BouzidiAntiBounceBack
                         },
                'value':(bc_in, (width, height, max_velocity, grad_pressure, cte)),
            },
        },
        'generator': generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)

    while sol.t<Tf:
        sol.one_time_step()

    if with_plot:
        print("*"*50)
        p_n = sol.m[p]
        ux_n = sol.m[ux]
        uy_n = sol.m[uy]
        x, y = np.meshgrid(*sol.domain.coords, sparse=True, indexing='ij')
        coeff = sol.domain.space_step / np.sqrt(width*height)
        Err_p = coeff * np.linalg.norm(p_n - (x-0.5*width) * grad_pressure)
        Err_ux = coeff * np.linalg.norm(ux_n - max_velocity * (1 - 4 * y**2 / height**2))
        Err_uy = coeff * np.linalg.norm(uy_n)
        print("Norm of the error on rho: {0:10.3e}".format(Err_p))
        print("Norm of the error on qx:  {0:10.3e}".format(Err_ux))
        print("Norm of the error on qy:  {0:10.3e}".format(Err_uy))

        # init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]

        ax.image(ux_n - max_velocity * (1 - 4 * y**2 / height**2))
        ax.title = "Error on ux"
        fig.show()

    return sol


if __name__ == '__main__':
    space_step = 1./128
    Tf = 200.
    run(space_step, Tf)
