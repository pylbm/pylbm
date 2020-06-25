# pylint: disable=invalid-name

# # Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
vectorial D2Q4 solver for the MHD system (in 2D)

dt rho + dx . q = 0
dt q   + dx . ( qq/rho + p* I -BB ) = 0
dt E   + dx . ( (E+p*)q/rho - q/rho . B B ) = 0
dt B   + dx . (q/rho B - B q/rho ) = 0

with p* = p + B**2/2
     p  = (gamma-1)( E - q**2/(2rho) - B**2/2 ) (gamma=5/3)

periodical conditions on [0, 2 pi] x [0, 2 pi]

initial conditions
    rho = gamma**2
    qx  = -gamma**2 * sin(y)
    qy  =  gamma**2 * sin(x)
    p   = gamma
    Bx  = -sin(y)
    By  = sin(2x)
"""
import numpy as np
import sympy as sp
import pylbm

# pylint: disable=redefined-outer-name, invalid-name
# pylint: disable=unused-argument

hdf5_save = True

GA, X, Y, LA = sp.symbols('GA, X, Y, lambda')
RHO, QX, QY, E, BX, BY = sp.symbols('rho, qx, qy, E, Bx, By')
P, PS = sp.symbols('p, ps')


def init_rho(x, y, gamma):
    """ initial condition for the density """
    return gamma**2 * np.ones(x.shape)


def init_qx(x, y, gamma):
    """ initial condition for the x-momentum """
    return -gamma**2 * np.sin(y)


def init_qy(x, y, gamma):
    """ initial condition for the y-momentum """
    return gamma**2 * np.sin(x)


def init_Bx(x, y):
    """ initial condition for the x-magnetic field """
    return -np.sin(y)


def init_By(x, y):
    """ initial condition for the y-magnetic field """
    return np.sin(2*x)


def init_E(x, y, gamma):
    """ initial condition for the electric field """
    Ec = 0.5 * (
        init_qx(x, y, gamma)**2
        + init_qy(x, y, gamma)**2
    )/init_rho(x, y, gamma)
    EB = 0.5 * (init_Bx(x, y)**2 + init_By(x, y)**2)
    return Ec + EB + gamma/(gamma-1)


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
    xmin, xmax = 0., 2*np.pi       # x-bounds of the domain
    ymin, ymax = 0., 2*np.pi       # y-bounds of the domain
    gamma = 5./3.                  # ratio of specific heats
    la = 10.                       # lattice velocity

    # relaxation parameters
    s0, s1, s2, s3 = [1.95]*4
    s_rho = [0., s1, s1, s0]
    s_q = [0., s2, s2, s0]
    s_E = [0., s3, s3, s0]
    s_B = [0., s3, s3, s0]

    # pressure law
    P = (GA-1) * (
        E - (QX**2+QY**2)/(2*RHO) - (BX**2+BY**2)/2
    )
    PS = P + (BX**2+BY**2)/2
    VB = (QX*BX + QY*BY) / RHO

    simu_cfg = {
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': -1
        },
        'space_step': space_step,
        'lattice_velocity': la,
        'schemes': [
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': RHO,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s_rho,
                'equilibrium': [RHO, QX, QY, 0],
            },
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': QX,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s_q,
                'equilibrium':[
                    QX,
                    QX**2/RHO + PS - BX**2,
                    QX*QY/RHO - BX*BY,
                    0
                ],
            },
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': QY,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s_q,
                'equilibrium':[
                    QY,
                    QX*QY/RHO - BX*BY,
                    QY**2/RHO + PS - BY**2,
                    0
                ],
            },
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': E,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s_E,
                'equilibrium':[
                    E,
                    (E+PS)*QX/RHO - VB*BX,
                    (E+PS)*QY/RHO - VB*BY,
                    0
                ],
            },
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': BX,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s_B,
                'equilibrium':[
                    BX,
                    0,
                    (QY*BX - QX*BY) / RHO,
                    0
                ],
            },
            {
                'velocities': list(range(1, 5)),
                'conserved_moments': BY,
                'polynomials': [1, X, Y, X**2-Y**2],
                'relaxation_parameters': s_B,
                'equilibrium':[
                    BY,
                    (QX*BY - QY*BX) / RHO,
                    0,
                    0
                ],
            },
        ],
        'init': {
            RHO: (init_rho, (gamma,)),
            QX: (init_qx, (gamma,)),
            QY: (init_qy, (gamma,)),
            E: (init_E, (gamma,)),
            BX: init_Bx,
            BY: init_By
        },
        'parameters': {LA: la, GA: gamma},
        'generator': generator,
    }

    sol = pylbm.Simulation(simu_cfg)

    if with_plot:
        # init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        # N, M = sol.m[RHO].shape
        # na, nb = 1, N-1
        # ma, mb = 1, M-1
        im = ax.image(sol.m[RHO].transpose(), clim=[0.5, 7.2])
        ax.title = 'solution at t = {0:f}'.format(sol.t)

        def update(iframe):
            for _ in range(16):
                sol.one_time_step()
            im.set_data(sol.m[RHO].transpose())
            ax.title = 'solution at t = {0:f}'.format(sol.t)

        # run the simulation
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
    # pylint: disable=invalid-name
    space_step = 2.*np.pi/256
    final_time = 10.
    run(space_step, final_time)
