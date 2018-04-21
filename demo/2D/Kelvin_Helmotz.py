from __future__ import print_function
from __future__ import division
"""
test: True
"""
from six.moves import range
import numpy as np
import sympy as sp
import mpi4py.MPI as mpi
import pylbm

X, Y = sp.symbols('X, Y')
rho, qx, qy, LA = sp.symbols('rho, qx, qy, LA', real=True)

def bc_up(f, m, x, y, driven_velocity):
    m[qx] = driven_velocity

def vorticity(sol):
    #sol.f2m()
    qx_n = sol.m[qx]
    qy_n = sol.m[qy]
    vort = np.abs(qx_n[1:-1, 2:] - qx_n[1:-1, :-2]
                  - qy_n[2:, 1:-1] + qy_n[:-2, 1:-1])
    return vort.T

def qx0(x, y, U, k):
    return np.zeros_like(x) + U*np.tanh(k*(y-.25))*(y<=.5) + U*np.tanh(k*(.75 - y))*(y>.5)

def qy0(x, y, U, delta):
    return np.zeros_like(y) + U*delta*np.sin(2*np.pi*(x + .25)) 

def feq(v, u):
    c0 = LA/sp.sqrt(3)
    x, y = sp.symbols('x, y')
    vsymb = sp.Matrix([x, y])
    w = sp.Matrix([sp.Rational(4,9)] + [sp.Rational(1, 9)]*4 + [sp.Rational(1, 36)]*4)
    f = rho*(1 + u.dot(vsymb)/c0**2 + u.dot(vsymb)**2/(2*c0**4) - u.norm()**2/(2*c0**2))
    return sp.Matrix([w[iv]*f.subs([(x, vv[0]*LA), (y, vv[1]*LA)]) for iv, vv in enumerate(v)])

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
    rhoo = 1.
    U = .5
    k = 80
    delta = .05

    Ma = .04
    lamb = np.sqrt(3)/Ma
    mu = .0366
    nu = 1e-4

    sigma3 = 3*mu/(rhoo*lamb*dx)
    sigma4 = 3*nu/(rhoo*lamb*dx)
    s3 = 1./(sigma3+.5)
    s4 = 1./(sigma4+.5)
    s  = [0.,0.,0.,s3,s4,s4,s3,s3,s3]

    kelvin_helmoltz = {
        'parameters':{LA: lamb},
        'box':{'x':[0., 1.], 'y':[0., 1.], 'label':-1},
        'space_step': dx,
        'scheme_velocity':LA,
        'schemes':[
            {
                'velocities':list(range(9)),
                'polynomials':[
                    1, X, Y,
                    X**2 + Y**2,
                    X**2 - Y**2,
                    X*Y,
                    X*(X**2+Y**2),
                    Y*(X**2+Y**2),
                    (X**2+Y**2)**2                    
                ],
                'relaxation_parameters':s,
                'feq': (feq, (sp.Matrix([qx/rho, qy/rho]),)),
                # 'equilibrium':[
                #     rho, qx, qy, 
                #     (qx**2 + qy**2 + 2*rho**2/3)/rho, 
                #     (qx**2 - qy**2)/rho, 
                #     qx*qy/rho, 
                #     4*qx/3, 
                #     4*qy/3, 
                #     (15*qx**2 + 15*qy**2 + 8*rho**2)/(9*rho)
                # ],
                'conserved_moments': [rho, qx, qy],
                'init': {rho: 1., qx: (qx0, (U, k)), qy: (qy0, (U, delta))},
            },
        ],
        #'relative_velocity': [qx/rho, qy/rho],
        'generator': generator,
    }


    sol = pylbm.Simulation(kelvin_helmoltz, sorder=sorder)

    # f = feq(sp.Matrix([qx/rho, qy/rho]))
    # out = sol.scheme.M*f
    # out.simplify()
    # print(f)
    # print(sol.scheme.M)
    # print(out)
    if withPlot:
        # init viewer
        viewer = pylbm.viewer.matplotlibViewer
        fig = viewer.Fig()
        ax = fig[0]
        image = ax.image(vorticity, (sol,), cmap='jet')#, clim=[-60, 50])

        def update(iframe):
            nrep = 100
            for i in range(nrep):
                sol.one_time_step()

            image.set_data(vorticity(sol))
            ax.title = "Solution t={0:f}".format(sol.t)

        # run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < Tf:
           sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 0.6
    run(dx, Tf)
