
import numpy as np
import sympy as sp

import pyLBM

X, Y = sp.symbols('X, Y')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_up(f, m, x, y):
    m[0] = 0.
    m[1] = driven_velocity
    m[2] = 0.

def vorticity(sol):
    sol.f2m()
    qx = sol.m[0][1]
    qy = sol.m[0][2]
    vort = np.abs(qx[1:-1, 2:] - qx[1:-1, :-2]
                  - qy[2:, 1:-1] + qy[:-2, 1:-1])
    return vort.T

def update(iframe):
    nrep = 100
    for i in xrange(nrep):
         sol.one_time_step()

    image.set_data(vorticity(sol))
    ax.title = "Solution t={0:f}".format(sol.t)

mu   = 1.e-4
zeta = 1.e-4
driven_velocity = 0.2 # velocity of the upper border
dx = 1./256
dummy = 3.0/dx
s1 = 1.0/(0.5+zeta*dummy)
s2 = 1.0/(0.5+mu*dummy)
s  = [0.,0.,0.,s1,s1,s1,s1,s2,s2]
Tf = 10.

lid_cavity = {'box':{'x':[0., 1.], 'y':[0., 1.], 'label':[0, 0, 0, 1]},
              'space_step': dx,
              'scheme_velocity':1,
              'schemes':[{'velocities':range(9),
                          'polynomials':[1,
                                   X, Y,
                                   3*(X**2+Y**2)-4,
                                   0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                                   3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                                   X**2-Y**2, X*Y],
                          'relaxation_parameters':s,
                          'equilibrium':[rho, qx, qy,
                                        -2*rho + 3*qx**2 + 3*qy**2,
                                        rho + 3/2*qx**2 + 3/2*qy**2,
                                        -qx, -qy,
                                        qx**2 - qy**2, qx*qy],
                          'conserved_moments': [rho, qx, qy],
                          'init': {rho: 1., qx: 0., qy: 0.},
              }],
              'boundary_conditions':{
                 0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}},
                 1:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}, 'value':bc_up}
              },
              'generator': pyLBM.CythonGenerator,
              }

sol = pyLBM.Simulation(lid_cavity)

# init viewer
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]
image = ax.image(vorticity, (sol,), cmap='cubehelix', clim=[0, .1])

# run the simulation
fig.animate(update, interval=1)
fig.show()
