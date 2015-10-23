import numpy as np
import sympy as sp

import pyLBM

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_rect(f, m, x, y):
    m[rho] = 0.
    m[qx] = rhoo*uo
    m[qy] = 0.

def vorticity(sol):
    qx_n = sol.m[qx]
    qy_n = sol.m[qy]
    vort = np.abs(qx_n[1:-1, 2:] - qx_n[1:-1, :-2]
                  - qy_n[2:, 1:-1] + qy_n[:-2, 1:-1])
    return vort.T

def update(iframe):
    nrep = 100
    for i in xrange(nrep):
         sol.one_time_step()
    #print sol.time_info()
    image.set_data(vorticity(sol))
    ax.title = "Solution t={0:f}".format(sol.t)

# parameters
dim = 2 # spatial dimension
xmin, xmax, ymin, ymax = 0., 2., 0., 1.
radius = 0.125
dx = 0.005 # spatial step
la = 1. # velocity of the scheme
rhoo = 1.
uo = 0.05
mu   = 2.5e-5 #0.00185
zeta = 10*mu
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
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0, 1, 0, 0]},
    'elements':[pyLBM.Circle([.3, 0.5*(ymin+ymax)+2*dx], radius, label=2)],
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':range(9),
            'polynomials':[
                1,
                LA*X, LA*Y,
                3*(X**2+Y**2)-4,
                0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                X**2-Y**2, X*Y
            ],
            'relaxation_parameters':s,
            'equilibrium':[
                rho,
                qx, qy,
                -2*rho + 3*q2,
                rho - 3*q2,
                -qx/LA, -qy/LA,
                qx2 - qy2, qxy
            ],
            'conserved_moments': [rho, qx, qy],
            'init':{
                rho: rhoo,
                qx: uo,
                qy: 0.
            },
        },
    ],
    'parameters':{'LA':la},
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}, 'value':bc_rect},
        1:{'method':{0: pyLBM.bc.Neumann_vertical}},
        2:{'method':{0: pyLBM.bc.Bouzidi_bounce_back}},
    },
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)

Re = rhoo*uo*2*radius/mu
print "Reynolds number {0:10.3e}".format(Re)

# init viewer
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()

ax = fig[0]
ax.ellipse([.3/dx, 0.5*(ymin+ymax)/dx+2], [radius/dx, radius/dx], 'r')
image = ax.image(vorticity(sol), cmap='cubehelix', clim=[0, .05])

update(1)
# run the simulation
fig.animate(update, interval=1)
fig.show()
