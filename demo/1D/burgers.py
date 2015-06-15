##############################################################################
#
# Solver D1Q2 and D1Q3 for the Burger's equation on [-1, 1]
#
# d_t(u) + d_x(u^2/2) = 0, t > 0, 0 < x < 1,
# u(t=0,x) = u0(x),
# d_t(u)(t,x=0) = d_t(u)(t,x=1) = 0
#
# the initial condition is a Riemann problem,
# that is a picewise constant function
#
# u0(x) = uL if x<0, uR if x>0.
#
# The solution is a shock wave if uL>uR and a linear rarefaction wave if uL<uR
#
##############################################################################

import numpy as np
import sympy as sp

import pyLBM

X, LA, u = sp.symbols('X,LA,u')

def u0(x): # initial condition
    xm = 0.5*(xmin+xmax)
    u = np.empty(x.shape)
    u[x < xm] = uL
    u[x == xm] = .5*(uL+uR)
    u[x > xm] = uR
    return u

def solution(t,x): # solution
    xm = 0.5*(xmin+xmax)
    u = np.empty(x.shape)
    if uL >= uR or t==0: # shock wave
        xD = xm + .5*t*(uL+uR)
        u[x < xD] = uL
        u[x == xD] = .5*(uL+uR)
        u[x > xD] = uR
    else: # rarefaction wave
        xL = xm + t*uL
        xR = xm + t*uR
        u[x < xL] = uL
        u[x>=xL and x<=xR] = (uL * (xR-x[ind_D]) + uR * (x[ind_D]-xL)) / (xR-xL)
        u[x > xR] = uR
    return u

# parameters
xmin, xmax = -1., 1.  # bounds of the domain
uL =  0.3             # left value
uR =  0.0             # right value
L = 0.2               # length of the middle area
dx = 1./256           # spatial step
la = 1.               # scheme velocity (la = dx/dt)
Tf = .5               # final time
s = 1.8               # relaxation parameter

# dictionary for the D1Q2
dico1 = {
    'box':{'x':[xmin, xmax], 'label':0},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':[1,2],
            'conserved_moments':[u],
            'polynomials':[1,LA*X],
            'relaxation_parameters':[0., s],
            'equilibrium':[u, u**2/2],
            'init':{u:(u0,)},
        },
    ],
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.neumann}, 'value':None},
    },
    'parameters': {LA: la},
}
# dictionary for the D1Q3
dico2 = {
    'box':{'x':[xmin, xmax], 'label':0},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':range(3),
            'conserved_moments':u,
            'polynomials':[1,LA*X,LA**2*X**2],
            'relaxation_parameters':[0., s, s],
            'equilibrium':[u, u**2/2, LA**2*u/3 + 2*u**3/9],
            'init':{u:(u0,)},
        },
    ],
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.neumann}, 'value':None},
    },
    'parameters': {LA: la},
}
# simulation
sol1 = pyLBM.Simulation(dico1) # build the simulation with D1Q2
sol2 = pyLBM.Simulation(dico2) # build the simulation with D1Q3

# create the viewer to plot the solution
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]
ymin, ymax = min([uL,uR])-.1*abs(uL-uR), max([uL,uR])+.1*abs(uL-uR)
ax.axis(xmin, xmax, ymin, ymax)

x1 = sol1.domain.x[0][1:-1]
l1 = ax.plot(x1, sol1.m[0][0][1:-1], width=1, color='b')[0]
x2 = sol2.domain.x[0][1:-1]
l2 = ax.plot(x2, sol2.m[0][0][1:-1], width=1, color='r')[0]
le = ax.plot(x1, solution(sol1.t, x1), width=1, color='k')[0]

def update(iframe):
    if sol1.t<Tf:
        sol1.one_time_step()
        sol1.f2m()
        sol2.one_time_step()
        sol2.f2m()
        l1.set_data(x1, sol1.m[0][0][1:-1])
        l2.set_data(x2, sol2.m[0][0][1:-1])
        le.set_data(x1, solution(sol1.t, x1))
        ax.title = 'solution at t = {0:f}'.format(sol1.t)

fig.animate(update)
fig.show()
