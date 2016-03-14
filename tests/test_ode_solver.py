from __future__ import print_function
from __future__ import division
from six.moves import range
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyLBM

u, X, ALPHA = sp.symbols('u, x, ALPHA')

if __name__ == "__main__":
    # parameters
    dt = 1.e-1
    la = 1
    alpha = 0.1
    Nt = 100
    # data
    dx = la*dt
    xmin, xmax = 0., 5*dx
    dico = {
        'box':{'x':[xmin, xmax],},
        'space_step':dx,
        'scheme_velocity':la,
        'generator':pyLBM.generator.CythonGenerator,
        'ode_solver':pyLBM.generator.basic,
        'parameters':{ALPHA:alpha},
        'schemes':[
           {
               'velocities':[0,],
               'conserved_moments':u,
               'polynomials':[1,],
               'relaxation_parameters':[0,],
               'equilibrium':[u,],
               'source_terms':{u:-ALPHA*u},
               'init':{u:1.},
           },
        ],
    }

    sol = pyLBM.Simulation(dico)

    fnum = np.empty((Nt,))
    tnum = np.empty((Nt,))
    fnum[0] = sol.m[u][1]
    tnum[0] = sol.t
    for n in range(1,Nt):
        sol.one_time_step()
        fnum[n] = sol.m[u][1]
        tnum[n] = sol.t
    plt.figure(1)
    plt.plot(tnum, fnum, 'r*')
    plt.show()
