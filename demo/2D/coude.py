import numpy as np
import sympy as sp
import pyLBM

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_in(f, m, x, y, scheme):
    if scheme.nv_on_beg:
        m[0] = rhoo
        m[1] = rhoo*uo * (ymax-y)*(y-0.75*(ymax-ymin))*8**2
        m[2] = 0.
    else:
        m[:, 0] = rhoo
        m[:, 1] = rhoo*uo * (ymax-y)*(y-0.75*(ymax-ymin))*8**2
        m[:, 2] = 0.
    scheme.equilibrium(m)
    scheme.m2f(m, f)

def vorticity(sol):
    sol.f2m()
    qx = sol.m[0][1]
    qy = sol.m[0][2]
    vort = np.abs(qx[1:-1, 2:] - qx[1:-1, :-2]
                  - qy[2:, 1:-1] + qy[:-2, 1:-1])
    return vort.T

def update(iframe):
    nrep = 256
    for i in xrange(nrep):
         sol.one_time_step()

    image.set_data(vorticity(sol))
    ax.title = "Solution t={0:f}".format(sol.t)

# parameters
dim = 2 # spatial dimension
xmin, xmax, ymin, ymax = 0., 1., 0., 1
rayon = 0.25*(xmax-xmin)
dx = 1./256 # spatial step
la = 1. # velocity of the scheme
rhoo = 1.
uo = 0.05
mu   = 1.e-5 #0.00185
zeta = 10*mu
dummy = 3.0/(la*rhoo*dx)
s3 = 1.0/(0.5+zeta*dummy)
s4 = s3
s5 = s4
s6 = s4
s7 = 1.0/(0.5+mu*dummy)
s8 = s7
s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
dummy = 1./(LA**2*rhoo)
qx2 = dummy*qx**2
qy2 = dummy*qy**2
q2  = qx2+qy2
qxy = dummy*qx*qy
xc = xmin + 0.75*(xmax-xmin)
yc = ymin + 0.75*(ymax-ymin)

dico = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[2, 0, 1, 0]},
    'elements':[pyLBM.Parallelogram((xmin,ymin),(xc,ymin),(xmin,yc), label=0)],
    'scheme_velocity':la,
    'space_step': dx,
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
                            #rho + 3/2*qx**2 + 3/2*qy**2,
                            rho - 3*qx**2 - 3*qy**2,
                            -qx, -qy,
                            qx**2 - qy**2, qx*qy],
                'conserved_moments': [rho, qx, qy],
                'init': {rho: rhoo, qx: rhoo*uo, qy: 0.},
    }],
    'boundary_conditions':{
       0:{'method':{0: pyLBM.bc.bouzidi_bounce_back}},
       1:{'method':{0: pyLBM.bc.neumann_horizontal}},
       2:{'method':{0: pyLBM.bc.bouzidi_bounce_back}, 'value':bc_in}
    },
    'generator': pyLBM.CythonGenerator,
  }

sol = pyLBM.Simulation(dico)

# init viewer
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig()
ax = fig[0]
image = ax.image(vorticity, (sol,), cmap='jet', clim=[0, .1])
ax.polygon([[xmin/dx, ymin/dx],[xmin/dx, yc/dx], [xc/dx, yc/dx], [xc/dx, ymin/dx]], 'k')

# run the simulation
fig.animate(update, interval=1)
fig.show()
