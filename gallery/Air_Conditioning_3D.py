from __future__ import print_function, division
from six.moves import range
import numpy as np
import sympy as sp
import pyLBM

VTK_save = True

X, Y, Z, LA = sp.symbols('X, Y, Z, LA')
mass, qx, qy, qz, T = sp.symbols('mass, qx, qy, qz, T')

def init_T(x, y, z):
    return T0

def bc_in(f, m, x, y, z):
    m[qx] = rhoo*uo
    m[qy] = 0.
    m[qz] = 0.
    m[T] = Tin

def bc_out(f, m, x, y, z):
    m[qx] = rhoo*uo
    m[qy] = 0.
    m[qz] = 0.

def update(iframe):
    nrep = 128
    for i in xrange(nrep):
        sol.one_time_step()
    image.set_data(plot_field(sol))
    ax.title = "Solution t={0:f}".format(sol.t)

def plot_field(sol):
    shape = sol._m.nspace
    f = sol.m[T][1:-1,1:-1, shape[-1]//2]
    return f.T

def save(x, y, z, m, num):
    if num > 0:
        vtk = pyLBM.VTKFile(filename, path, num)
    else:
        vtk = pyLBM.VTKFile(filename, path, num, init_pvd = True)
    vtk.set_grid(x, y, z)
    vtk.add_scalar('T', m[T][1:-1,1:-1,1:-1])
    #vtk.add_vector('velocity', [m[qx][1:-1,1:-1], m[qy][1:-1,1:-1], m[qz][1:-1,1:-1]])
    vtk.save()

# parameters
T0 = .5
Tin = -.5
xmin, xmax, ymin, ymax, zmin, zmax = 0., 1., 0., 1., 0., 1.
Ra, Pr, alpha = 1e5, 0.71, 0.001
dx = 1./128 # spatial step
la = 1. # velocity of the scheme
rhoo = 1.
g = 9.81
uo = 0.05

nu = np.sqrt(Pr*g*abs(T0-Tin)*(xmax-xmin)**3/Ra)
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
sigma_kappa = 3.*kappa/(la*rhoo*dx)
skappa = 1./(.5+sigma_kappa)
se = 1./(.5+np.sqrt(3)/3)
snu = se
sT = [0., skappa, skappa, skappa, se, se]

s1 = 1.6
s2 = 1.2
s4 = 1.6
s9 = 1./(3*nu +.5)
s11 = s9
s14 = 1.2

r = X**2+Y**2+Z**2

radius = .05

dico = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':0},
    'elements':[
        pyLBM.Cylinder_Circle([xmin, 0.8, (zmax+zmin)/2], [ 0., radius, 0], [0, 0, radius], [0.01, 0, 0], label=1),
        pyLBM.Cylinder_Circle([xmax, 0.8, (zmax+zmin)/2], [ 0., radius, 0], [0, 0, radius], [-0.01, 0, 0], label=2),
    ],
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
              {
                'velocities':list(range(7)) + list(range(19,27)),
                'conserved_moments':[mass, qx, qy, qz],
                'polynomials':[
                    1,
                    r - 2, .5*(15*r**2-55*r+32),
                    X, .5*(5*r-13)*X,
                    Y, .5*(5*r-13)*Y,
                    Z, .5*(5*r-13)*Z,
                    3*X**2-r, Y**2-Z**2,
                    X*Y, Y*Z, Z*X,
                    X*Y*Z
                ],
                'relaxation_parameters':[0, s1, s2, 0, s4, 0, s4, 0, s4, s9, s9, s11, s11, s11, s14],
                'equilibrium':[
                    mass,
                    -mass + qx**2 + qy**2 + qz**2,
                    -mass,
                    qx,
                    -7./3*qx,
                    qy,
                    -7./3*qy,
                    qz,
                    -7./3*qz,
                    1./3*(2*qx**2-(qy**2+qz**2)),
                    qy**2-qz**2,
                    qx*qy,
                    qy*qz,
                    qz*qx,
                    0
                ],
                'source_terms':{qy: alpha*g*T},
                'init':{
                    mass:1.,
                    qx: 0.,
                    qy: 0.,
                    qz: 0.
                },
            },
            {
                'velocities': list(range(1,7)),
                'conserved_moments':[T],
                'polynomials': [1, X, Y, Z, X**2-Y**2, X**2-Z**2],
                'equilibrium': [T, qx*T, qy*T, qz*T, 0., 0.],
                'relaxation_parameters': sT,
                'init':{T:(init_T,)},
            },
        ],
    'boundary_conditions':{
        #0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Bouzidi_anti_bounce_back}, 'value':bc},
        0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Neumann}, 'value':None},
        1:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Bouzidi_anti_bounce_back}, 'value':bc_in},
        2:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Neumann_x}, 'value':bc_out},
    },
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)
#sol.domain.geom.visualize(viewlabel=True)
#sol.domain.visualize(view_distance=False, view_bound=True, label=[1,2], view_in=False, view_out=False)

if VTK_save:
    filename = 'Air_Conditioning'
    path = './data_Air_Conditioning_3D'
    im = 0
    x, y, z = sol.domain.x[1:-1], sol.domain.y[1:-1], sol.domain.z[1:-1]
    save(x, y, z, sol.m, im)
    while sol.t<200.:
        for k in range(128):
            sol.one_time_step()
        im += 1
        save(x, y, z, sol.m, im)
else:
    # init viewer
    viewer = pyLBM.viewer.matplotlibViewer
    fig = viewer.Fig()
    ax = fig[0]
    Tmin, Tmax = min(Tin, T0), max(Tin, T0)
    image = ax.image(plot_field(sol), clim=[Tmin, Tmax])
    #ax.polygon(-0.5+np.asarray([[0, .01/dx, .01/dx, 0], [0, 0, .8/dx, .8/dx]]).T, color='b')
    #ax.polygon(-0.5+np.asarray([[1./dx, .99/dx, .99/dx, 1./dx], [0, 0, .8/dx, .8/dx]]).T, color='b')
    #ax.polygon(-0.5+np.asarray([[0, .01/dx, .01/dx, 0], [1./dx, 1./dx, .9/dx, .9/dx]]).T, color='b')
    #ax.polygon(-0.5+np.asarray([[1./dx, .99/dx, .99/dx, 1./dx], [1./dx, 1./dx, .9/dx, .9/dx]]).T, color='b')
    #ax.polygon(-0.5+np.asarray([[(xb-l)/dx, (xb-l)/dx, (xb+l)/dx, (xb+l)/dx], [(yb-e)/dx, (yb+e)/dx, (yb+e)/dx, (yb-e)/dx]]).T, color='b')
    # run the simulation
    fig.animate(update, interval=1)
    fig.show()
