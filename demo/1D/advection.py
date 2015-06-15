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
import pyLBM

X, LA, u = sp.symbols('X,LA,u')

def u0(x):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    return 1.0/largeur**10 * (x%1-milieu-largeur)**5 * (milieu-x%1-largeur)**5 * (abs(x%1-milieu)<=largeur)

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
            'conserved_moments':u,
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

# create the viewer to plot the solution
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]
ymin, ymax = -.2, 1.2
ax.axis(xmin, xmax, ymin, ymax)

x = sol.domain.x[0][1:-1]
l1 = ax.plot(x, sol.m[0][0][1:-1], width=2, color='b')[0]
l2 = ax.plot(x, u0(x-c*sol.t), width=2, color='k')[0]

def update(iframe):
    if sol.t<Tf:                 # time loop
        sol.one_time_step()      # increment the solution of one time step
        sol.f2m()
        l1.set_data(x, sol.m[0][0][1:-1])
        l2.set_data(x, u0(x-c*sol.t))
        ax.title = 'solution at t = {0:f}'.format(sol.t)

fig.animate(update)
fig.show()
