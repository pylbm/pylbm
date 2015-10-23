import numpy as np
import sympy as sp

import pyLBM

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_in(f, m, x, y):
    m[rho] = (x-0.5*width) * grad_pressure
    m[qx] = max_velocity * (1. - 4.*y**2/height**2)
<<<<<<< Updated upstream

def bc_out(f, m, x, y):
    m[rho] = (x-0.5*width) * grad_pressure
=======
    m[qy] = 0.

def bc_out(f, m, x, y):
    m[rho] = (x-0.5*width) * grad_pressure
    m[qx] = 0.
    m[qy] = 0.
>>>>>>> Stashed changes

def update(iframe):
    sol.one_time_step()
    l1.set_data(y, sol.m[qx][nt, 1:-1])
    ax.title = 'Velocity at t = {0:f}'.format(sol.t)

# parameters
dim = 2 # spatial dimension
dx = 1./128 # spatial step
la = 1. # velocity of the scheme
Tf = 20
width = 2
height = 1
max_velocity = 0.1
rhoo = 1.
mu   = 0.00185
zeta = 1.e-4
xmin, xmax, ymin, ymax = 0.0, width, -0.5*height, 0.5*height
grad_pressure = - max_velocity * 8.0 / (height)**2 * 3.0/(la**2*rhoo) * mu
NbImages = 80 # number of figures
dummy = 3.0/(la*rhoo*dx)
s1 = 1.0/(0.5+zeta*dummy)
s2 = 1.0/(0.5+mu*dummy)
s  = [0.,0.,0.,s1,s1,s1,s1,s2,s2]
dummy = 1./(LA**2*rhoo)
qx2 = dummy*qx**2
qy2 = dummy*qy**2
q2  = qx2+qy2
qxy = dummy*qx*qy

dico = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[2, 1, 0, 0]},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[{'velocities':range(9),
                'polynomials':[1,
                         LA*X, LA*Y,
                         3*(X**2+Y**2)-4,
                         0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                         3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                         X**2-Y**2, X*Y],
                'relaxation_parameters':s,
                'equilibrium':[rho,
                          qx, qy,
                          -2*rho + 3*q2,
                          rho + 1.5*q2,
                          -qx/LA, -qy/LA,
                          qx2 - qy2, qxy],
                'conserved_moments': [rho, qx, qy],
                'init':{rho: 1.,
                        qx: 0.,
                        qy: 0.
                        },
                }],
    'parameters':{'LA':la},
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}},
        1:{'method':{0: pyLBM.bc.Neumann_vertical}},
        2:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}, 'value':bc_in}
    },
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)

# init viewer
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]

nt = int(sol.domain.N[0]/2)
y = sol.domain.x[1][1:-1]
l1 = ax.plot(y, sol.m[qx][nt, 1:-1], color='r', marker='+')[0]
l2 = ax.plot(y, rhoo*max_velocity * (1.-4.*y**2/height**2), color='k')
ax.title = 'Velocity at t = {0:f}'.format(sol.t)
#ax.axis(ymin, ymax, 0., 1.2*max_velocity)

# run the simulation
fig.animate(update, interval=1)
fig.show()
