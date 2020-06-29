<<<<<<< HEAD

=======
# pylint: disable=invalid-name

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
>>>>>>> 1b60335a5d53c6e3e2de1bb1a140303f0bf8f351

"""
Simulate the Poiseuille flow

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

test: True
"""

import sympy as sp
import pylbm

# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=invalid-name

<<<<<<< HEAD
import pylbm

X, Y, LA = sp.symbols('X, Y, lambda')
rho, qx, qy = sp.symbols('rho, qx, qy')
=======
X, Y, LA = sp.symbols('X, Y, lambda')
RHO, QX, QY = sp.symbols('rho, qx, qy')

>>>>>>> 1b60335a5d53c6e3e2de1bb1a140303f0bf8f351

def bc_in(f, m, x, y, width, height, max_velocity, grad_pressure):
    """ inner boundary condition """
    m[RHO] = (x-0.5*width) * grad_pressure
    m[QX] = max_velocity * (1. - 4.*y**2/height**2)


def bc_out(f, m, x, y, width, grad_pressure):
    """ outer boundary condition """
    m[RHO] = (x-0.5*width) * grad_pressure


<<<<<<< HEAD
def run(dx, Tf, generator="cython", sorder=None, with_plot=True):
=======
def run(space_step,
        final_time,
        generator="cython",
        sorder=None,
        with_plot=True):
>>>>>>> 1b60335a5d53c6e3e2de1bb1a140303f0bf8f351
    """
    Parameters
    ----------

    space_step: double
        spatial step

    final_time: double
        final time

<<<<<<< HEAD
    generator: pylbm generator
=======
    generator: string
        pylbm generator
>>>>>>> 1b60335a5d53c6e3e2de1bb1a140303f0bf8f351

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
    width = 2             # width of the domain
    height = 1            # height of the domain
    max_velocity = 0.1    # reference of the maximal velocity
    rhoo = 1              # reference value of the density
    mu = 0.00185          # bulk viscosity
    zeta = 1.e-5          # shear viscosity

    xmin, xmax, ymin, ymax = 0.0, width, -0.5*height, 0.5*height
    grad_pressure = - max_velocity * 8.0 / (height)**2 * 3.0/(la**2*rhoo) * mu
    dummy = 3.0/(la*rhoo*space_step)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)
<<<<<<< HEAD
    s  = [0.,0.,0.,s1,s1,s1,s1,s2,s2]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[2, 1, 0, 0]},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[{'velocities':list(range(9)),
                    'polynomials':[1,
                             LA*X, LA*Y,
                             3*(X**2+Y**2)-4,
                             0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                             3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                             X**2-Y**2, X*Y],
                    'relaxation_parameters':s,
                    'equilibrium':[rho,
                              qx, qy,
                              -2*rho + 3*q2,
                              rho - 3*q2,
                              -qx/LA, -qy/LA,
                              qx2 - qy2, qxy],
                    'conserved_moments': [rho, qx, qy],
                    }],
        'parameters':{LA:la},
        'init':{rho: 1.,
                qx: 0.,
                qy: 0.
                },
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.BouzidiBounceBack}},
            1:{'method':{0: pylbm.bc.NeumannX}},
            2:{'method':{0: pylbm.bc.BouzidiBounceBack},
               'value':(bc_in, (width, height, max_velocity, grad_pressure))}
=======
    s = [0, 0, 0, s1, s1, s1, s1, s2, s2]
    dummy = 1/(LA**2*rhoo)
    QX2 = dummy*QX**2
    QY2 = dummy*QY**2
    Q2 = QX2 + QY2
    QXY = dummy*QX*QY

    simu_cfg = {
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [2, 1, 0, 0]
        },
        'space_step': space_step,
        'lattice_velocity': la,
        'schemes': [
            {
                'velocities': list(range(9)),
                'conserved_moments': [RHO, QX, QY],
                'polynomials': [
                    1,
                    X, Y,
                    3*(X**2+Y**2)-4*LA**2,
                    0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)*LA**2+8*LA**4),
                    3*X*(X**2+Y**2)-5*X*LA**2, 3*Y*(X**2+Y**2)-5*Y*LA**2,
                    X**2-Y**2, X*Y
                ],
                'equilibrium': [
                    RHO,
                    QX, QY,
                    -2*RHO*LA**2 + 3*Q2,
                    RHO*LA**2 - 3*Q2,
                    -QX*LA**2, -QY*LA**2,
                    QX2 - QY2, QXY
                ],
                'relaxation_parameters': s,
            }
        ],
        'parameters': {LA: la},
        'init': {
            RHO: rhoo,
            QX: 0,
            QY: 0
        },
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiBounceBack}},
            1: {'method': {0: pylbm.bc.NeumannX}},
            2: {
                'method': {0: pylbm.bc.BouzidiBounceBack},
                'value': (bc_in, (width, height, max_velocity, grad_pressure))
            }
>>>>>>> 1b60335a5d53c6e3e2de1bb1a140303f0bf8f351
        },
        'generator': generator,
    }

<<<<<<< HEAD
    sol = pylbm.Simulation(dico, sorder=sorder)
=======
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)
>>>>>>> 1b60335a5d53c6e3e2de1bb1a140303f0bf8f351

    if with_plot:
        # init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]

        nt = int(sol.domain.shape_in[0]/2)
        y = sol.domain.y

        l1 = ax.CurveScatter(
            y, sol.m[QX][nt],
            color='orange', label=r'$D_2Q_9$'
        )
        ax.CurveLine(
            y, rhoo*max_velocity * (1.-4.*y**2/height**2),
            color='navy', label='exact', alpha=0.5
        )
        ax.title = f'Velocity at $t = {sol.t:3.0f}$'
        ax.legend()

        def update(iframe):
            for _ in range(128):
                sol.one_time_step()
            l1.update(sol.m[QX][nt])
            ax.title = f'Velocity at $t = {sol.t:3.0f}$'

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
    space_step = 1./128
    final_time = 20
    run(space_step, final_time)
