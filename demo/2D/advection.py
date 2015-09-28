##############################################################################
#
# Solver D2Q4 for the advection equation on the 2D-torus
#
# d_t(u) + cx d_x(u) + cy d_y(u) = 0, t > 0, 0 < x,y < 1,
# u(t=0,x,y) = u0(x,y),
# u(t,x=0,y) = u(t,x=1,y) 0 < y < 1,
# u(t,x,y=0) = u(t,x,y=1) 0 < x < 1,
#
# the solution is
# u(t,x,y) = u0(x-cx*t,y-cy*t)
#
##############################################################################

import numpy as np
import sympy as sp

import pyLBM

u, X, Y, LA = sp.symbols('u,X,Y,LA')

def u0(x,y):
    return np.ones((x.shape[0], y.shape[0]), dtype='float64') \
           + .5 * ((x-0.25*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.01)

# parameters
xmin, xmax, ymin, ymax = 0., 1., 0., 1. # bounds of the domain
cx, cy = 0.2, 0.5                       # velocity of the advection
dx = 1./128                             # spatial step
la = 2.                                 # scheme velocity
Tf = 10                                 # final time
sigma_qx = 1./np.sqrt(12)
sigma_xy = sigma_qx
s_qx = 1./(0.5+sigma_qx)
s_xy = 1./(0.5+sigma_xy)
s  = [0., s_qx, s_qx, s_xy]             # relaxation parameters

dico = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':range(1,5),
            'conserved_moments':u,
            'polynomials':[1, LA*X, LA*Y, X**2-Y**2],
            'relaxation_parameters':s,
            'equilibrium':[u, cx*u, cy*u, 0],
            'init':{u:(u0,)},
        },
    ],
    'generator': pyLBM.generator.CythonGenerator,
    'parameters':{LA:la},
    }

sol = pyLBM.Simulation(dico, sorder=[1, 2, 0])

print sol.scheme.generator.code

# create the viewer to plot the solution
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]

im = ax.image(sol.m[0][0].transpose())
ax.title = 'solution at t = {0:f}'.format(sol.t)

compt = 0

def update(iframe):
    global compt
    if sol.t<Tf:                 # time loop
        sol.one_time_step()      # increment the solution of one time step
        compt += 1
        if compt == 128:
            compt = 0
            im.set_data(sol.m[0][0].transpose())
            ax.title = 'solution at t = {0:f}'.format(sol.t)

fig.animate(update, interval=1)
fig.show()
