from __future__ import print_function, division
from six.moves import range
import numpy as np
import sympy as sp
import pyLBM

VTK_save = False

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy, T = sp.symbols('rho, qx, qy, T')

def init_T(x, y):
    return T0

def bc_in(f, m, x, y):
    m[qx] = rhoo*uo
    m[qy] = 0.
    m[T] = Tin

def bc_out(f, m, x, y):
    m[qx] = rhoo*uo
    m[qy] = 0.

def update(iframe):
    nrep = 128
    for i in xrange(nrep):
        sol.one_time_step()
    image.set_data(plot_field(sol))
    ax.title = "Solution t={0:f}".format(sol.t)

def plot_field(sol):
    return sol.m[T][1:-1,1:-1].T

def save(x, y, m, num):
    if num > 0:
        vtk = pyLBM.VTKFile(filename, path, num)
    else:
        vtk = pyLBM.VTKFile(filename, path, num, init_pvd = True)
    vtk.set_grid(x, y)
    vtk.add_scalar('T', m[T][1:-1,1:-1])
    #vtk.add_vector('velocity', [m[qx][1:-1,1:-1], m[qy][1:-1,1:-1]])
    vtk.save()

# parameters
T0 = .5
Tin = -.5
xmin, xmax, ymin, ymax = 0., 1., 0., 1.
Ra, Pr, Ma, alpha = 1.e7, 0.71, 0.01, 1.e-2
dx = 1./128 # spatial step
la = 1. # velocity of the scheme
rhoo = 1.
g = 9.81
uo = 0.1

nu = np.sqrt(Pr*alpha*g*abs(T0-Tin)*(xmax-xmin)**3/Ra)
kappa = nu/Pr
eta = nu
#print(nu, kappa)
dummy = 3./(la*rhoo*dx)
snu = 1./(.5+dummy*nu)
seta = 1./(.5+dummy*eta)
sq = 8*(2-snu)/(8-snu)
se = seta
sf = [0., 0., 0., seta, se, sq, sq, snu, snu]
#print(sf)
a = .5
skappa = 1./(.5+10/(la*rhoo*dx)*kappa/(4+a))
se = 1./(.5+np.sqrt(3)/3)
snu = se
sT = [0., skappa, skappa, se, snu]
#print(sT)

xb, yb, l, e = 0.5*(xmin+xmax), 0.9*ymin+0.1*ymax, 0.125*(xmax-xmin), 0.01
banc = pyLBM.Parallelogram([xb-l, yb-e], [2*l, 0], [0, 2*e], label=0)

dico = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0, 0, 0, 0]},
    'elements':[
        pyLBM.Parallelogram([xmin, 0.8], [ .01, 0], [0, .1], label=1),
        pyLBM.Parallelogram([xmax, 0.8], [-.01, 0], [0, .1], label=2),
        banc,
    ],
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':list(range(9)),
            'conserved_moments': [rho, qx, qy],
            'polynomials':[
                1, X, Y,
                3*(X**2+Y**2)-4,
                0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                X**2-Y**2, X*Y
            ],
            'relaxation_parameters':sf,
            'equilibrium':[
                rho, qx, qy,
                -2*rho + 3*(qx**2+qy**2),
                rho - 3*(qx**2+qy**2),
                -qx, -qy,
                qx**2 - qy**2, qx*qy
            ],
            'source_terms':{qy: alpha*g * T},
            'init':{rho: 1., qx: 0., qy: 0.},
        },
        {
            'velocities':list(range(5)),
            'conserved_moments':T,
            'polynomials':[1, X, Y, 5*(X**2+Y**2) - 4, (X**2-Y**2)],
            'equilibrium':[T, T*qx, T*qy, a*T, 0.],
            'relaxation_parameters':sT,
            'init':{T:(init_T,)},
        },
    ],
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Neumann}, 'value':None},
        1:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Bouzidi_anti_bounce_back}, 'value':bc_in},
        2:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Neumann_vertical}, 'value':bc_out},
        #2:{'method':{0: pyLBM.bc.Neumann_vertical, 1: pyLBM.bc.Neumann_vertical}, 'value':None},
    },
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)
#sol.domain.geom.visualize(viewlabel=True)

if VTK_save:
    filename = 'Air_Conditioning'
    path = './data_Air_Conditioning'
    im = 0
    x, y = sol.domain.x[1:-1], sol.domain.y[1:-1]
    save(x, y, sol.m, im)
    while sol.t<200.:
        for k in range(128):
            sol.one_time_step()
        im += 1
        save(x, y, sol.m, im)
else:
    # init viewer
    viewer = pyLBM.viewer.matplotlibViewer
    fig = viewer.Fig()
    ax = fig[0]
    Tmin, Tmax = min(Tin, T0), max(Tin, T0)
    image = ax.image(plot_field(sol), clim=[Tmin, Tmax])
    ax.polygon(-0.5+np.asarray([[0, .01/dx, .01/dx, 0], [.8/dx, .8/dx, .9/dx, .9/dx]]).T, color='b')
    ax.polygon(-0.5+np.asarray([[1./dx, .99/dx, .99/dx, 1./dx], [.8/dx, .8/dx, .9/dx, .9/dx]]).T, color='b')
    ax.polygon(-0.5+np.asarray([[(xb-l)/dx, (xb-l)/dx, (xb+l)/dx, (xb+l)/dx], [(yb-e)/dx, (yb+e)/dx, (yb+e)/dx, (yb-e)/dx]]).T, color='b')
    # run the simulation
    fig.animate(update, interval=1)
    fig.show()
