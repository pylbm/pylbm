##############################################################################
#
# Solver D1Q2 for the advection equation on the 1D-torus
#
# d_t(u) + c d_x(u) = 0, t > 0, 0 < x < 1, (c=1/4)
# u(t=0,x) = u0(x),
# u(t,x=0) = u(t,x=1)
#
# the solution is
# u(t,x) = u0(x-ct).
#
##############################################################################

import sympy as sp
import pylab as plt
import pyLBM

X, LA, u = sp.symbols('X,LA,u')

def u0(x):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    return 1.0/largeur**10 * (x%1-milieu-largeur)**5 * (milieu-x%1-largeur)**5 * (abs(x%1-milieu)<=largeur)

def plot_init(num = 0):
    fig = plt.figure(num,figsize=(16, 8))
    plt.clf()
    l1 = plt.plot([], [], 'b', label=r'$D_1Q_2$')[0]
    l2 = plt.plot([], [], 'k', label='exact')[0]
    plt.xlim(xmin, xmax)
    ymin, ymax = -.2, 1.2
    plt.ylim(ymin, ymax)
    plt.legend()
    return [l1, l2]

def plot(sol, l):
    sol.f2m()
    x = sol.domain.x[0][1:-1]
    l[0].set_data(x, sol.m[0][0][1:-1])
    l[1].set_data(x, u0(x-c*sol.t))
    plt.title('solution at t = {0:f}'.format(sol.t))
    plt.pause(1.e-3)

# parameters
xmin, xmax = 0., 1.   # bounds of the domain
dx = 1./128           # spatial step
la = 1.               # scheme velocity (la = dx/dt)
c = 0.25              # velocity of the advection
Tf = 4.               # final time
s = 1.99              # relaxation parameter

# dictionary of the simulation
dico = {
    'box':{'x':[xmin, xmax], 'label':-1},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':[1,2],
            'conserved_moments':[u],
            'polynomials':[1,LA*X],
            'relaxation_parameters':[0., s],
            'equilibrium':[u, c*u],
            'init':{u:(u0,)},
        },
    ],
    'parameters': {LA: la},
}

# simulation
sol = pyLBM.Simulation(dico) # build the simulation
l = plot_init(0)             # initialize the plot
plot(sol, l)                 # plot initial condition
while (sol.t<Tf):            # time loop
    sol.one_time_step()      # increment the solution of one time step
    plot(sol, l)             # plot the solution
plt.show()
