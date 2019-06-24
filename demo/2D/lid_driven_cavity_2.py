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
import matplotlib.pyplot as plt
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


def flow_lines(sol, nlines, time_length, scale=0.5):
    """
    compute the flow lines of the solution

    Parameters
    ----------
    sol : :py:class:`Simulation<pylbm.simulation.Simulation>`
        the solution given by pylbm
    nlines : int (number of flow lines)
    time_length : double (time during which we follow the lines)
    scale : double (velocity scale (default 0.5))

    Returns
    -------
    list
        lines
    """
    u_x = sol.m[QX] / sol.m[RHO]
    u_y = sol.m[QY] / sol.m[RHO]
    # if scale is None:
    #     scale = max(np.linalg.norm(u_x, np.inf), np.linalg.norm(u_y, np.inf))
    lines = []
    xmin, xmax = sol.domain.geom.bounds[0]
    ymin, ymax = sol.domain.geom.bounds[1]
    dx = sol.domain.dx
    nx, ny = sol.domain.shape_in
    for _ in range(nlines):
        # begin a new line
        cont = True  # boolean to continue the line
        x = xmin + (xmax-xmin) * np.random.rand()
        y = ymin + (ymax-ymin) * np.random.rand()
        line_x, line_y = [x], [y]
        t = 0
        while cont:
            i, j = int((x-xmin)/(xmax-xmin)*nx), int((y-ymin)/(ymax-ymin)*ny)
            uxij, uyij = u_x[i, j], u_y[i, j]
            if uxij == 0 and uyij == 0:
                cont = False
            else:
                dt = dx*scale / np.sqrt(uxij**2+uyij**2)
                x += uxij*dt
                y += uyij*dt
                t += dt
                if x < xmin or x >= xmax or y < ymin or y >= ymax:
                    cont = False
                else:
                    line_x.append(x)
                    line_y.append(y)
            if t >= time_length:
                cont = False
        lines.append([np.array(line_x), np.array(line_y)])
    return lines


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
    while sol.t < final_time:
        sol.one_time_step()

    viewer = pylbm.viewer.matplotlib_viewer
    fig = viewer.Fig()

    axe = fig[0]
    axe.grid(visible=False)
    axe.xaxis_set_visible(False)
    axe.yaxis_set_visible(False)
    axe.SurfaceImage(
        vorticity(sol),
        cmap='jet', clim=[0, .1], alpha=0.25,
    )
    lines = flow_lines(sol, 10, 2)
    for linek in lines:
        axe.CurveLine(linek[0], linek[1], alpha=1)

    plt.show()
    return sol

if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./128
    final_time = 100
    run(space_step, final_time)
