import sys

import numpy as np
import sympy as sp

import pyLBM

import pylab as plt
import matplotlib.cm as cm

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def initialization_rho(x,y):
    return rhoo * np.ones((x.size, y.size), dtype='float64') + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)

def plot_init(num = 0):
    plt.ion()
    fig = plt.figure(num,figsize=(16, 8))
    plt.clf()
    ax = plt.subplot(111)
    l = ax.imshow([], origin='lower', cmap=cm.gray, interpolation='nearest')[0]
    return l

def update(iframe):
    for k in xrange(32):
        sol.one_time_step()      # increment the solution of one time step
    im.set_data(sol.m[0][0].transpose())
    ax.title = 'solution at t = {0:f}'.format(sol.t)

if __name__ == "__main__":
    rhoo = 1.
    deltarho = 1.
    Taille = 2.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille
    # parameters
    dx = 1./128 # spatial step
    la = 4 # velocity of the scheme
    g = 1.
    sigma = 1.e-4
    s_0qx = 2.#1./(0.5+sigma)
    s_0xy = 1.5
    s_1qx = 1.5
    s_1xy = 1.2
    s0  = [0., s_0qx, s_0qx, s_0xy]
    s1  = [0., s_1qx, s_1qx, s_1xy]

    vitesse = range(1,5)
    polynomes = [1, LA*X, LA*Y, X**2-Y**2]

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
        'space_step':dx,
        'scheme_velocity':la,
        'parameters':{LA:la},
        'schemes':[
            {
                'velocities':vitesse,
                'conserved_moments':rho,
                'polynomials':polynomes,
                'relaxation_parameters':s0,
                'equilibrium':[rho, qx, qy, 0.],
                'init':{rho:(initialization_rho,)},
            },
            {
                'velocities':vitesse,
                'conserved_moments':qx,
                'polynomials':polynomes,
                'relaxation_parameters':s1,
                'equilibrium':[qx, qx**2/rho + 0.5*g*rho**2, qx*qy/rho, 0.],
                'init':{qx:0.},
            },
            {
                'velocities':vitesse,
                'conserved_moments':qy,
                'polynomials':polynomes,
                'relaxation_parameters':s1,
                'equilibrium':[qy, qy*qx/rho, qy**2/rho + 0.5*g*rho**2, 0.],
                'init':{qy:0.},
            },
        ],
        'generator': pyLBM.generator.CythonGenerator,
        }

    sol = pyLBM.Simulation(dico)

    viewer = pyLBM.viewer.matplotlibViewer
    fig = viewer.Fig()
    ax = fig[0]

    im = ax.image(sol.m[0][0].transpose(), clim=[rhoo-.5*deltarho, rhoo+1.5*deltarho])
    ax.title = 'solution at t = {0:f}'.format(sol.t)

    fig.animate(update, interval=1)
    fig.show()
