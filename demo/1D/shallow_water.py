##############################################################################
#
# Solver D1Q2Q2 for the shallow water system on [0, 1]
#
# d_t(h) + d_x(q)    = 0, t > 0, 0 < x < 1,
# d_t(q) + d_x(q^2/h+gh^2/2) = 0, t > 0, 0 < x < 1,
# h(t=0,x) = h0(x), q(t=0,x) = q0(x),
# d_t(h)(t,x=0) = d_t(h)(t,x=1) = 0
# d_t(q)(t,x=0) = d_t(q)(t,x=1) = 0
#
# the initial condition is a picewise constant function
# in order to visualize the simulation of elementary waves
#
##############################################################################

import sys
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyLBM

h, q, X, LA, g = sp.symbols('h,q,X,LA,g')

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

def plot_init(num = 0):
    fig = plt.figure(num,figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    l1 = Line2D([], [], color='b', marker='*', linestyle='None')
    l2 = Line2D([], [], color='r', marker='d', linestyle='None')
    ax1.add_line(l1)
    ax2.add_line(l2)
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax1.set_ylim(.9*ymina, 1.1*ymaxa)
    ax2.set_ylim(.9*yminb, 1.1*ymaxb)
    t1 = ax1.text(0.5*(xmin+xmax), ymaxa, '')
    t2 = ax2.text(0.5*(xmin+xmax), ymaxb, '')
    return [l1, l2, t1, t2]

def plot(sol, l):
    sol.f2m()
    x = sol.domain.x[0][1:-1]
    l[0].set_data(x, sol.m[0][0][1:-1])
    l[1].set_data(x, sol.m[1][0][1:-1])
    l[2].set_text(r'$h$ at $t = {0:f}$'.format(sol.t))
    l[3].set_text(r'$q$ at $t = {0:f}$'.format(sol.t))
    plt.pause(1.e-3)

if __name__ == "__main__":
    # parameters
    xmin, xmax = 0., 1.  # bounds of the domain
    dx = 1./256          # spatial step
    la = 2.              # velocity of the scheme
    s = 1.7              # relaxation parameter
    Tf = 0.25            # final time

    hg, hd, qg, qd = 1., .25, 0.10, 0.10
    ymina, ymaxa, yminb, ymaxb = 0., 1., 0., .5

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':h,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[h, q],
                'init':{h:(Riemann_pb, (hg, hd))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':q,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[q, q**2/h+.5*g*h**2],
                'init':{q:(Riemann_pb, (qg, qd))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.neumann, 1: pyLBM.bc.neumann}, 'value':None},
        },
        'parameters':{LA:la, g:1.},
    }

    sol = pyLBM.Simulation(dico)
    l = plot_init()
    plot(sol, l)
    while (sol.t<Tf):
        sol.one_time_step()
        plot(sol, l)
    plt.show()
