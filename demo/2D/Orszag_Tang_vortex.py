
"""
D2Q4 solver for the MHD system (in 2D)

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

GA, X, Y, LA = sp.symbols('GA, X, Y, lambda')
rho, qx, qy, E, Bx, By = sp.symbols('rho, qx, qy, E, Bx, By')
p, ps = sp.symbols('p, ps')

def init_rho(x, y, gamma):
    return gamma**2 * np.ones(x.shape)

def init_qx(x, y, gamma):
    return -gamma**2 * np.sin(y)

def init_qy(x, y, gamma):
    return gamma**2 * np.sin(x)

def init_Bx(x, y):
    return -np.sin(y)

def init_By(x, y):
    return np.sin(2*x)

def init_E(x, y, gamma):
    Ec = 0.5 * (init_qx(x, y, gamma)**2 + init_qy(x, y, gamma)**2)/init_rho(x, y, gamma)
    EB = 0.5 * (init_Bx(x, y)**2 + init_By(x, y)**2)
    return Ec + EB + gamma/(gamma-1)

def run(dx, Tf, generator="cython", sorder=None, with_plot=True):
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

    with_plot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    xmin, xmax, ymin, ymax = 0., 2*np.pi, 0., 2*np.pi
    gamma = 5./3.

    s0, s1, s2, s3 = [1.95]*4
    la = 10.
    s_rho = [0., s1, s1, s0]
    s_q = [0., s2, s2, s0]
    s_E = [0., s3, s3, s0]
    s_B = [0., s3, s3, s0]

    p = (GA-1) * (E - (qx**2+qy**2)/(2*rho) - (Bx**2+By**2)/2)
    ps = p + (Bx**2+By**2)/2
    vB = (qx*Bx + qy*By)/rho

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':list(range(1,5)),
                'conserved_moments':rho,
                'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
                'relaxation_parameters':s_rho,
                'equilibrium':[rho, qx, qy, 0.],
            },
            {
                'velocities':list(range(1,5)),
                'conserved_moments':qx,
                'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
                'relaxation_parameters':s_q,
                'equilibrium':[
                    qx,
                    qx**2/rho + ps - Bx**2,
                    qx*qy/rho - Bx*By,
                    0.
                ],
            },
            {
                'velocities':list(range(1,5)),
                'conserved_moments':qy,
                'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
                'relaxation_parameters':s_q,
                'equilibrium':[
                    qy,
                    qx*qy/rho - Bx*By,
                    qy**2/rho + ps - By**2,
                    0.
                ],
            },
            {
                'velocities':list(range(1,5)),
                'conserved_moments':E,
                'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
                'relaxation_parameters':s_E,
                'equilibrium':[
                    E,
                    (E+ps)*qx/rho - vB*Bx,
                    (E+ps)*qy/rho - vB*By,
                    0.
                ],
            },
            {
                'velocities':list(range(1,5)),
                'conserved_moments':Bx,
                'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
                'relaxation_parameters':s_B,
                'equilibrium':[
                    Bx,
                    0,
                    (qy*Bx - qx*By)/rho,
                    0.
                ],
            },
            {
                'velocities':list(range(1,5)),
                'conserved_moments':By,
                'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
                'relaxation_parameters':s_B,
                'equilibrium':[
                    By,
                    (qx*By - qy*Bx)/rho,
                    0,
                    0.
                ],
            },
        ],
        'init':{rho: (init_rho, (gamma,)),
                qx: (init_qx, (gamma,)),
                qy: (init_qy, (gamma,)),
                E: (init_E, (gamma,)),
                Bx: init_Bx,
                By: init_By},
        'parameters':{LA:la, GA:gamma},
        'generator': generator,
    }

    sol = pylbm.Simulation(dico)

    if with_plot:
        # init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        N, M = sol.m[rho].shape
        na, nb = 1, N-1
        ma, mb = 1, M-1
        im = ax.image(sol.m[rho][na:nb, ma:mb].transpose(), clim=[0.5, 7.2])
        ax.title = 'solution at t = {0:f}'.format(sol.t)

        def update(iframe):
            for k in range(16):
                sol.one_time_step()      # increment the solution of one time step
            im.set_data(sol.m[rho][na:nb, ma:mb].transpose())
            ax.title = 'solution at t = {0:f}'.format(sol.t)

        # run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < Tf:
           sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 2.*np.pi/256
    Tf = 10.
    run(dx, Tf)
