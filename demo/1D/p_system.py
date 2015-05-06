##############################################################################
#
# Solver D1Q2Q2 for the p-system on [0, 1]
#
# d_t(ua) - d_x(ub)    = 0, t > 0, 0 < x < 1,
# d_t(ub) - d_x(p(ua)) = 0, t > 0, 0 < x < 1,
# ua(t=0,x) = ua0(x), ub(t=0,x) = ub0(x),
# d_t(ua)(t,x=0) = d_t(ua)(t,x=1) = 0
# d_t(ub)(t,x=0) = d_t(ub)(t,x=1) = 0
#
# the initial condition is a picewise constant function
# in order to visualize the simulation of elementary waves
#
##############################################################################

import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyLBM

ua, ub, X, LA = sp.symbols('ua,ub,X,LA,')

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
    l[2].set_text(r'$u_a$ at $t = {0:f}$'.format(sol.t))
    l[3].set_text(r'$u_b$ at $t = {0:f}$'.format(sol.t))
    plt.pause(1.e-3)

if __name__ == "__main__":
    # parameters
    gamma = 2./3.        # exponent in the p-function
    xmin, xmax = 0., 1.  # bounds of the domain
    dx = 1./256          # spatial step
    la = 2.              # velocity of the scheme
    s = 1.7              # relaxation parameter
    Tf = 0.25            # final time

    uag, uad, ubg, ubd = 1.50, 1.25, 1.50, 1.00
    ymina, ymaxa, yminb, ymaxb = 1., 1.75, 1., 1.5

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':ua,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[ua, -ub],
                'init':{ua:(Riemann_pb, (uag, uad))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':ub,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[ub, ua**(-gamma)],
                'init':{ub:(Riemann_pb, (ubg, ubd))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.neumann, 1: pyLBM.bc.neumann}, 'value':None},
        },
        'parameters':{LA:la},
    }

    sol = pyLBM.Simulation(dico)
    l = plot_init()
    plot(sol, l)
    while (sol.t<Tf):
        sol.one_time_step()
        plot(sol, l)
    plt.show()
