
# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Solver D2Q9-D2Q5 for the Bousinesq approximation

Wang, J. and Wang, D. and Lallemand, P. and Luo, L.-S.
Lattice Boltzmann simulations of thermal convective flows in two dimensions
Computers & Mathematics with Applications. An International Journal
Volume 65, number 2, 2013, pp. 262--286
http://space_step.doi.org/10.1016/j.camwa.2012.07.001

"""

import numpy as np
import sympy as sp
import pylbm

# pylint: disable=unused-argument, invalid-name, redefined-outer-name

X, Y, LA = sp.symbols('X, Y, LA')
RHO, QX, QY, T = sp.symbols('rho, qx, qy, T')


def init_temperature(x, y, temp0):
    """ initial condition for the temperature """
    return temp0


def bc(f, m, x, y, temp0):
    """ boundary conditition (qx = qy = 0, T = temp0) """
    m[QX] = 0.
    m[QY] = 0.
    m[T] = temp0


def bc_in(f, m, x, y, temp0, tempin, ymax, rhoo, uo):
    """ inner boundary condition (qx = qx0, qy = 0, T parabolic profile) """
    m[QX] = rhoo*uo
    m[QY] = 0.
    m[T] = temp0 + (tempin - temp0) * (ymax-y) * (y-.8) * 100


def run(space_step,
        final_time,
        generator="numpy",
        sorder=None,
        with_plot=True):
    """
    Parameters
    ----------

    space_step: double
        spatial step

    final_time: double
        final time

    generator: string, optional
        pylbm generator

    sorder: list, optional
        storage order

    with_plot: boolean, optional
        if True plot the solution otherwise just compute the solution


    Returns
    -------

    sol
        <class 'pylbm.simulation.Simulation'>

    """
    # parameters
    temp0 = .5                 # reference temperature
    tempin = -.5               # cool temperature
    xmin, xmax = 0., 1.        # bounds of the domain in x
    ymin, ymax = 0., 1.        # bounds of the domain in y
    Ra = 2000                  # Rayleigh number
    Pr = 0.71                  # Prandt number
    # Ma = 0.01                  # Mach number
    alpha = .005               # thermal expansion coefficient
    la = 1.                    # lattice velocity
    rhoo = 1.                  # reference density
    g = 9.81                   # gravity
    uo = 0.025                 # reference velocity

    # shear viscosity
    nu = np.sqrt(
        Pr * alpha*g*(temp0-tempin)
        * (ymax-ymin) / Ra
    )
    kappa = nu/Pr              # thermal diffusivity
    eta = nu
    # relaxation times
    snu = 1./(.5+3*nu)
    seta = 1./(.5+3*eta)
    sq = 8*(2-snu)/(8-snu)
    se = seta
    sf = [0., 0., 0., seta, se, sq, sq, snu, snu]
    a = .5
    skappa = 1./(.5+10*kappa/(4+a))
    se = 1./(.5+np.sqrt(3)/3)
    snu = se
    sT = [0., skappa, skappa, se, snu]

    simu_cfg = {
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [1, 2, 0, 0]
        },
        'elements': [
            pylbm.Parallelogram([xmin, ymin], [.1, 0], [0, .8], label=0),
            pylbm.Parallelogram([xmax, ymin], [-.1, 0], [0, .8], label=0),
        ],
        'space_step': space_step,
        'lattice_velocity': la,
        'schemes': [
            {
                'velocities': list(range(9)),
                'conserved_moments': [RHO, QX, QY],
                'polynomials':[
                    1, X, Y,
                    3*(X**2+Y**2)-4,
                    (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2,
                    3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                    X**2-Y**2, X*Y
                ],
                'relaxation_parameters': sf,
                'equilibrium':[
                    RHO, QX, QY,
                    -2*RHO + 3*(QX**2+QY**2),
                    RHO - 3*(QX**2+QY**2),
                    -QX, -QY,
                    QX**2 - QY**2, QX*QY
                ],
                'source_terms': {QX: alpha*g*T},
            },
            {
                'velocities': list(range(5)),
                'conserved_moments': T,
                'polynomials': [1, X, Y, 5*(X**2+Y**2) - 4, (X**2-Y**2)],
                'equilibrium': [T, T*QX, T*QY, a*T, 0.],
                'relaxation_parameters': sT,
            },
        ],
        'init': {
            RHO: rhoo,
            QX: 0.,
            QY: 0.,
            T: (init_temperature, (temp0,))},
        'boundary_conditions': {
            0: {
                'method': {
                    0: pylbm.bc.BouzidiBounceBack,
                    1: pylbm.bc.BouzidiAntiBounceBack
                },
                'value': (bc, (temp0,))
            },
            1: {
                'method': {
                    0: pylbm.bc.BouzidiBounceBack,
                    1: pylbm.bc.BouzidiAntiBounceBack
                },
                'value': (bc_in, (temp0, tempin, ymax, rhoo, uo))
            },
            2: {
                'method': {
                    0: pylbm.bc.NeumannX,
                    1: pylbm.bc.NeumannX
                },
            },
        },
        'generator': generator,
    }

    sol = pylbm.Simulation(simu_cfg)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        im = ax.image(sol.m[T].transpose(), cmap='jet', clim=[tempin, temp0])
        ax.title = 'solution at t = {0:f}'.format(sol.t)
        ax.polygon(
            [
                [xmin/space_step, ymin/space_step],
                [xmin/space_step, (ymin+.8)/space_step],
                [(xmin+.1)/space_step, (ymin+.8)/space_step],
                [(xmin+.1)/space_step, ymin/space_step]
            ], 'k'
        )
        ax.polygon(
            [
                [(xmax-.1)/space_step, ymin/space_step],
                [(xmax-.1)/space_step, (ymin+.8)/space_step],
                [xmax/space_step, (ymin+.8)/space_step],
                [xmax/space_step, ymin/space_step]
            ], 'k'
        )

        def update(iframe):
            nrep = 32
            for _ in range(nrep):
                sol.one_time_step()
            im.set_data(sol.m[T].transpose())
            ax.title = f'temperature at $t = {sol.t:05.2f}$'

        fig.animate(update, interval=1)
        fig.show()
    else:
        with pylbm.progress_bar(int(final_time/sol.dt),
                                title='run') as pbar:
            while sol.t < final_time:
                sol.one_time_step()
                pbar()

    return sol


if __name__ == '__main__':
    space_step = 1./128
    final_time = 10.
    run(space_step, final_time)
