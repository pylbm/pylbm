from __future__ import print_function
from __future__ import division
from six.moves import range
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyLBM

rho, q, E, X, LA = sp.symbols('rho,q,E,X,LA')

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

if __name__ == "__main__":
    # parameters
    gamma = 1.4
    xmin, xmax = 0., 1.
    dx = 1.e-3 # spatial step
    la = 3. # velocity of the scheme
    rho_L, rho_R, p_L, p_R, u_L, u_R = 1., 1./8., 1., 0.1, 0., 0.
    q_L = rho_L*u_L
    q_R = rho_R*u_R
    E_L = rho_L*u_L**2 + p_L/(gamma-1.)
    E_R = rho_R*u_R**2 + p_R/(gamma-1.)
    s_rho, s_q, s_E = 1.9, 1.5, 1.4

    dico1 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':rho,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_rho],
                'equilibrium':[rho, q],
                'init':{rho:(Riemann_pb, (rho_L, rho_R))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':q,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_q],
                'equilibrium':[q, (gamma-1.)*E+0.5*(3.-gamma)*q**2/rho],
                'init':{q:(Riemann_pb, (q_L, q_R))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':E,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_E],
                'equilibrium':[E, gamma*E*q/rho-0.5*(gamma-1.)*q**3/rho**2],
                'init':{E:(Riemann_pb, (E_L, E_R))},
            },
        ],
        'boundary_conditions':{
            0:{
                'method':{
                    0: pyLBM.bc.Neumann,
                    1: pyLBM.bc.Neumann,
                    2: pyLBM.bc.Neumann
                },
                'value':None
            },
        },
        'parameters':{LA:la},
        'generator': pyLBM.generator.NumpyGenerator,
    }

    dico2 = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':rho,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_rho],
                'equilibrium':[rho, q],
                'init':{rho:(Riemann_pb, (rho_L, rho_R))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':q,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_q],
                'equilibrium':[q, (gamma-1.)*E+0.5*(3.-gamma)*q**2/rho],
                'init':{q:(Riemann_pb, (q_L, q_R))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':E,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s_E],
                'equilibrium':[E, gamma*E*q/rho-0.5*(gamma-1.)*q**3/rho**2],
                'init':{E:(Riemann_pb, (E_L, E_R))},
            },
        ],
        'boundary_conditions':{
            0:{
                'method':{
                    0: pyLBM.bc.Neumann,
                    1: pyLBM.bc.Neumann,
                    2: pyLBM.bc.Neumann
                },
                'value':None
            },
        },
        'parameters':{LA:la},
        'generator': pyLBM.generator.CythonGenerator,
    }

    N = 2

    sol1 = pyLBM.Simulation(dico1)
    x = sol1.domain.x[0][1:-1]
    for k in range(N):
        sol1.one_time_step()
    sol1.f2m()
    rho1 = sol1.m[rho][1:-1]
    q1 = sol1.m[q][1:-1]
    E1 = sol1.m[E][1:-1]
    u1 = q1/rho1
    p1 = (gamma-1.)*(E1 - .5*rho1*u1**2)
    e1 = E1/rho1 - .5*u1**2

    sol2 = pyLBM.Simulation(dico2)
    for k in range(N):
        sol2.one_time_step()
    sol2.f2m()
    rho2 = sol2.m[rho][1:-1]
    q2 = sol2.m[q][1:-1]
    E2 = sol2.m[E][1:-1]
    u2 = q2/rho2
    p2 = (gamma-1.)*(E2 - .5*rho2*u2**2)
    e2 = E2/rho2 - .5*u2**2

    if sol1.t != sol2.t:
        print("Problem of time !!!")

    print(sol1.scheme.generator.code)
    print(sol2.scheme.generator.code)

    f, ax = plt.subplots(2, 3)
    ax[0,0].plot(x, rho1-rho2)
    ax[0,0].set_title('mass')
    ax[0,1].plot(x, u1-u2)
    ax[0,1].set_title('velocity')
    ax[0,2].plot(x, p1-p2)
    ax[0,2].set_title('pressure')
    ax[1,0].plot(x, E1-E2)
    ax[1,0].set_title('energy')
    ax[1,1].plot(x, q1-q2)
    ax[1,1].set_title('momentum')
    ax[1,2].plot(x, e1-e2)
    ax[1,2].set_title('internal energy')

    plt.show()
