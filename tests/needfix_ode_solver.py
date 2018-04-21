from __future__ import print_function
from __future__ import division
from six.moves import range
import numpy as np
from scipy import stats
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pylbm

u, X = sp.symbols('u, x')

def solution(t, alpha):
    return np.exp(-alpha*t)

def verification(dt, alpha, f, ode_solver):
    c = 0.5*alpha*dt
    if ode_solver == pylbm.generator.basic or ode_solver == pylbm.generator.explicit_euler:
        coeff = 1-c
    elif ode_solver == pylbm.generator.heun or ode_solver == pylbm.generator.middle_point:
        coeff = 1-c+c**2/2
    elif ode_solver == pylbm.generator.RK4:
        coeff = 1-c+c**2/2-c**3/6+c**4/24
    else:
        print("Cannot test the ode scheme ", ode_solver.__name__)
        coeff = 0.
    assert(np.allclose(f, coeff**np.arange(0, 2*f.size, 2)))

def run(dt, alpha,
    generator = pylbm.generator.CythonGenerator,
    ode_solver = pylbm.generator.basic):
    # parameters
    Tf = 1
    la = 1.
    # data
    Nt = int(Tf/dt)
    dx = la*dt
    xmin, xmax = 0., 2*dx
    dico = {
        'box':{'x':[xmin, xmax],},
        'space_step':dx,
        'scheme_velocity':la,
        'generator':generator,
        'ode_solver':ode_solver,
        'schemes':[
           {
               'velocities':[0,],
               'conserved_moments':u,
               'polynomials':[1,],
               'relaxation_parameters':[0,],
               'equilibrium':[u,],
               'source_terms':{u:-alpha*u},
               'init':{u:1.},
           },
        ],
    }

    sol = pylbm.Simulation(dico)
    #print(sol.scheme.generator.code)
    fnum = np.empty((Nt,))
    tnum = np.empty((Nt,))
    fnum[0] = sol.m[u][1]
    tnum[0] = sol.t
    for n in range(1,Nt):
        sol.one_time_step()
        fnum[n] = sol.m[u][1]
        tnum[n] = sol.t
    verification(dt, alpha, fnum, ode_solver)
    return np.linalg.norm(fnum - solution(tnum, alpha), np.inf)

if __name__ == "__main__":
    alpha = 0.875
    ODES = [pylbm.generator.basic,
        pylbm.generator.explicit_euler,
        pylbm.generator.heun,
        pylbm.generator.middle_point,
        pylbm.generator.RK4
    ]
    print(" "*28 + " Numpy      Cython")
    for odes in ODES:
        DT = []
        ERnp = []
        ERcy = []
        for k in range(2, 10):
            dt = 2**(-k)
            DT.append(0.5*dt)
            ERnp.append(run(dt, alpha,
                generator = pylbm.generator.NumpyGenerator,
                ode_solver = odes))
            ERcy.append(run(dt, alpha,
                generator = pylbm.generator.CythonGenerator,
                ode_solver = odes))
        print("Slope for {0:14s}: {1:10.3e} {2:10.3e}".format(odes.__name__, stats.linregress(np.log2(DT), np.log2(ERnp))[0],  stats.linregress(np.log2(DT), np.log2(ERcy))[0]))
