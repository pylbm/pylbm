# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Simulate the lid driven cavity

dt rho + dx qx + dy qy = 0
dt qx + dx (qx^2/rho + c^2 rho) + dy (qx*qy/rho) = 0
dt qy + dx (qx*qy/rho) + dy (qy^2/rho + c^2 rho) = 0
"""

import numpy as np
import sympy as sp
import pylbm

# pylint: disable=redefined-outer-name

X, Y = sp.symbols('X, Y')
RHO, QX, QY = sp.symbols('rho, qx, qy')
LA = sp.symbols('lambda', constants=True)


# pylint: disable=unused-argument
def bc_up(f, m, x, y, rho_o, driven_velocity):
    """
    boundary values on the top bound
    """
    m[RHO] = rho_o
    m[QX] = rho_o * driven_velocity
    m[QY] = 0.


def vorticity(sol):
    """
    compute the vorticity of the solution
    """
    qx_n = sol.m[QX] / sol.m[RHO]
    qy_n = sol.m[QY] / sol.m[RHO]
    vort = np.abs(
        qx_n[1:-1, 2:] - qx_n[1:-1, :-2] -
        qy_n[2:, 1:-1] + qy_n[:-2, 1:-1]
    )
    return vort


def norm_velocity(sol):
    """
    compute the norm of the velocity
    """
    qx_n = sol.m[QX] / sol.m[RHO]
    qy_n = sol.m[QY] / sol.m[RHO]
    nv = np.sqrt(qx_n**2 + qy_n**2)
    return nv


# pylint: disable=invalid-name
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
    scheme_name = 'Geier'
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.  # bounds of the domain
    la = 1.                                  # velocity of the scheme
    rho_o = 1.                               # reference value of the mass
    driven_velocity = 0.05                   # boundary value of the velocity
    mu = 5.e-6                               # bulk viscosity
    zeta = 100*mu                            # shear viscosity

    def moments_choice(scheme_name, mu, zeta):
        if scheme_name == 'dHumiere':
            dummy = 1./rho_o
            QX2 = dummy*QX**2
            QY2 = dummy*QY**2
            Q2 = QX2+QY2
            QXY = dummy*QX*QY
            polynomials = [
                1,
                X, Y,
                3*(X**2+Y**2)-4*LA**2,
                0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)*LA**2+8*LA**4),
                3*X*(X**2+Y**2)-5*X*LA**2, 3*Y*(X**2+Y**2)-5*Y*LA**2,
                X**2-Y**2, X*Y
            ]
            equilibrium = [
                RHO,
                QX, QY,
                -2*RHO*LA**2 + 3*Q2,
                RHO*LA**2 - 3*Q2,
                -QX*LA**2, -QY*LA**2,
                QX2 - QY2, QXY
            ]
            dummy = 3.0/(la*rho_o*space_step)
            sigma_1 = dummy*zeta
            sigma_2 = dummy*mu
            s_1 = 1/(.5+sigma_1)
            s_2 = 1/(.5+sigma_2)

        if scheme_name == 'Geier':
            UX, UY = QX/RHO, QY/RHO
            RHOU2 = RHO * (UX**2 + UY**2)
            polynomials = [
                1, X, Y,
                X**2 + Y**2,
                X*Y**2,
                Y*X**2,
                X**2*Y**2,
                X**2 - Y**2,
                X*Y,
            ]
            equilibrium = [
                RHO, QX, QY,
                RHOU2 + 2/3*RHO*LA**2,
                QX*(LA**2/3+UY**2),
                QY*(LA**2/3+UX**2),
                RHO*(LA**2/3+UX**2)*(LA**2/3+UY**2),
                RHO*(UX**2 - UY**2),
                RHO*UX*UY,
            ]
            dummy = 3.0/(la*rho_o*space_step)
            sigma_1 = dummy*(zeta - 2*mu/3)
            sigma_2 = dummy*mu
            s_1 = 1/(.5+sigma_1)
            s_2 = 1/(.5+sigma_2)

        if scheme_name == 'Lallemand':
            dummy = 1./rho_o
            QX2 = dummy*QX**2
            QY2 = dummy*QY**2
            Q2 = QX2+QY2
            QXY = dummy*QX*QY
            polynomials = [
                1, X, Y,
                X**2 + Y**2,
                X*(X**2+Y**2),
                Y*(X**2+Y**2),
                (X**2+Y**2)**2,
                X**2 - Y**2,
                X*Y,
            ]
            equilibrium = [
                RHO,
                QX, QY,
                Q2+2/3*LA**2*RHO,
                4/3*QX*LA**2,
                4/3*QY*LA**2,
                ((21*Q2+6*RHO*LA**2)*LA**2 - (6*Q2-2*RHO*LA**2))/9,
                QX2-QY2,
                QXY,
            ]
            dummy = 3.0/(la*rho_o*space_step)
            sigma_1 = dummy*zeta
            sigma_2 = dummy*mu
            s_1 = 1/(.5+sigma_1)
            s_2 = 1/(.5+sigma_2)

        s = [0., 0., 0., s_1, s_1, s_1, s_1, s_2, s_2]
        return polynomials, equilibrium, s

    polynomials, equilibrium, s = moments_choice(scheme_name, mu, zeta)

    simu_cfg = {
        'parameters': {LA: la},
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [0, 0, 0, 1]
        },
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': list(range(9)),
                'polynomials': polynomials,
                'relaxation_parameters': s,
                'equilibrium': equilibrium,
                'conserved_moments': [RHO, QX, QY],
            },
        ],
        'init': {RHO: rho_o,
                 QX: 0.,
                 QY: 0.},
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiBounceBack}},
            1: {
                'method': {0: pylbm.bc.BouzidiBounceBack},
                'value': (bc_up, (rho_o, driven_velocity))
            }
        },
        'generator': generator,
        'relative_velocity': [QX/RHO, QY/RHO],
        # 'show_code': True,
    }

    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        Re = rho_o*driven_velocity*2/mu
        print("Reynolds number {0:10.3e}".format(Re))

        # init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        axe = fig[0]
        axe.grid(visible=False)
        axe.xaxis_set_visible(False)
        axe.yaxis_set_visible(False)
        surf = axe.SurfaceImage(
            vorticity(sol), cmap='jet', clim=[0, .025]
            # norm_velocity(sol), cmap='jet', clim=[0, driven_velocity]
        )

        def update(iframe):
            nrep = 128
            for _ in range(nrep):
                sol.one_time_step()
            surf.update(vorticity(sol))
            # surf.update(norm_velocity(sol))
            axe.title = "Solution t={0:f}".format(sol.t)

        # run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./128
    final_time = 10
    run(space_step, final_time)
