"""
 Solver D1Q2 for the advection reaction equation on the 1D-torus

 d_t(u) + c d_x(u) = mu u(1-u), t > 0, 0 < x < 1, (c=1/4)
 u(t=0,x) = u0(x),
 u(t,x=0) = u(t,x=1)

 test: True
"""
import numpy as np
import sympy as sp
import pyLBM

t, X, LA, u = sp.symbols('t, X, LA, u')
C, MU = sp.symbols('C, MU')


def u0(x, xmin, xmax):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    return .25/largeur**10 * (x%1-milieu-largeur)**5 * (milieu-x%1-largeur)**5 * (abs(x%1-milieu)<=largeur)

def solution(t, x, xmin, xmax, c, mu):
    dt = np.tanh(0.5*mu*t)
    ui = u0(x - c*t, xmin, xmax)
    return (dt+2*ui-(1-2*ui)*dt)/(2-2*(1-2*ui)*dt)

def run(dx, Tf, generator=pyLBM.generator.NumpyGenerator, sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pyLBM generator

    store: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    # parameters
    xmin, xmax = 0., 1.   # bounds of the domain
    la = 1.               # scheme velocity (la = dx/dt)
    c = 0.25              # velocity of the advection
    mu = 1.               # parameter of the source term
    s = 1.9              # relaxation parameter

    # dictionary of the simulation
    dico = {
        'box':{'x':[xmin, xmax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
        {
            'velocities':[1,2],
            'conserved_moments':u,
            'polynomials':[1,LA*X],
            'relaxation_parameters':[0., s],
            'equilibrium':[u, C*u],
            'source_terms':{u:MU*u*t - MU*u**2},
            'init':{u:(u0,(xmin, xmax))},
        },
        ],
        'generator': generator,
        'parameters': {LA: la, C: c, MU: mu, 'time':t},
    }

    # simulation
    sol = pyLBM.Simulation(dico, sorder=sorder) # build the simulation
    print sol.scheme.generator.code


    # if withPlot:
    #     # create the viewer to plot the solution
    #     viewer = pyLBM.viewer.matplotlibViewer
    #     fig = viewer.Fig()
    #     ax = fig[0]
    #     ymin, ymax = -.2, 1.2
    #     ax.axis(xmin, xmax, ymin, ymax)
    #
    #     x = sol.domain.x[0][1:-1]
    #     l1 = ax.plot(x, sol.m[u][1:-1], width=2, color='b', label='D1Q2')[0]
    #     l2 = ax.plot(x, solution(sol.t, x, xmin, xmax, c, mu), width=2, color='k', label='exact')[0]
    #
    #     def update(iframe):
    #         if sol.t < Tf:                 # time loop
    #             sol.one_time_step()      # increment the solution of one time step
    #             l1.set_data(x, sol.m[u][1:-1])
    #             l2.set_data(x, solution(sol.t, x, xmin, xmax, c, mu))
    #             ax.title = 'solution at t = {0:f}'.format(sol.t)
    #             ax.legend()
    #
    #     fig.animate(update)
    #     fig.show()
    # else:
    #     while sol.t < Tf:
    #         sol.one_time_step()

    return sol

if __name__ == '__main__':
    dx = 1./128
    Tf = 3.
    sol = run(dx, Tf)
