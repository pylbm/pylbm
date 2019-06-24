
# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Simulate a flow in a bend

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
def bc_in(f, m, x, y, rho_o, u_o, ymin, ymax, radius):
    """
    boundary values on the left bound
    """
    y_bound = ymax - radius
    m[RHO] = rho_o
    m[QX] = rho_o*u_o * 4*(ymax-y)*(y-y_bound)/(ymax-y_bound)**2
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
    radius = 0.25                            # radius of the obstacle
    la = 1.                                  # velocity of the scheme
    rho_o = 1.                               # reference value of the mass
    u_o = 0.05                               # boundary value of the velocity
    mu = 2.5e-6                              # bulk viscosity
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
    xc = xmax - radius
    yc = ymax - radius

    simu_cfg = {
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [2, 0, 1, 0]
        },
        'elements': [
            pylbm.Parallelogram(
                (xmin, ymin),
                (xc, ymin),
                (xmin, yc),
                label=0
            )
        ],
        'space_step': space_step,
        'scheme_velocity': la,
        'schemes': [
            {
                'velocities': list(range(9)),
                'polynomials': polynomials,
                'relaxation_parameters': s,
                'equilibrium': equilibrium,
                'conserved_moments': [RHO, QX, QY],
            },
        ],
        'parameters': {LA: la},
        'init': {
            RHO: rho_o,
            QX: rho_o * u_o,
            QY: 0.
        },
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiBounceBack}},
            1: {'method': {0: pylbm.bc.NeumannY}},
            2: {
                'method': {
                    0: pylbm.bc.BouzidiBounceBack
                },
                'value': (bc_in, (rho_o, u_o, ymin, ymax, radius))
            },
        },
        'generator': generator,
        'relative_velocity': [QX/RHO, QY/RHO],
        # 'show_code': True
    }

    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        Re = rho_o*u_o*2*radius/mu
        print("Reynolds number {0:10.3e}".format(Re))

        # init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()

        axe = fig[0]
        axe.grid(visible=False)
        axe.xaxis_set_visible(False)
        axe.yaxis_set_visible(False)
        axe.polygon(
            [
                [0, 0],
                [0, yc/space_step-1],
                [xc/space_step-1, yc/space_step-1],
                [xc/space_step-1, 0]
            ], 'black'
        )

        surf = axe.SurfaceImage(
            vorticity(sol),
            cmap='jet', clim=[0, .1]
        )

        def update(iframe):  # pylint: disable=unused-argument
            nrep = 32
            for _ in range(nrep):
                sol.one_time_step()
            surf.update(vorticity(sol))
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
