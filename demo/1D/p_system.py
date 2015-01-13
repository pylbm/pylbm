import sys
import cmath
from math import pi, sqrt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, zeros
import mpi4py.MPI as mpi
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import pyLBM
import pyLBM.Geometry as pyLBMGeom
import pyLBM.Simulation as pyLBMSimu

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

def p(x):
    y   = - x**(-gamma)
    yp  =  gamma*x**(-gamma-1)
    ypp = -gamma*(gamma+1)*x**(-gamma-2)
    return (y,yp,ypp)

def intp(x,xo):
    alpha = -0.5*(gamma-1)
    y = sqrt(gamma)/alpha*(x**alpha-xo**alpha)
    return y

def f1(u1,ug1,ug2): # parametrisation de la 1-onde
    (pu1,dpu1,ddpu1) = p(u1)
    if u1 < ug1: # 1-choc
	(pug1,dpug1,ddpug1) = p(ug1)
	u2  = ug2 - sqrt((pu1-pug1)*(u1-ug1))
	up2 = -(dpu1*(u1-ug1)+pu1-pug1)/sqrt((pu1-pug1)*(u1-ug1))/2
    elif u1 > ug1: # 1-detente
	u2  = ug2 + intp(u1,ug1)
	up2 = sqrt(dpu1)
    else: # rien
	u2  = ug2
	up2 = sqrt(dpu1)
    return (u2,up2)

def f2(u1,ud1,ud2): # parametrisation de la 2-onde
    (pu1,dpu1,ddpu1) = p(u1)
    if u1 < ud1: # 2-choc
	(pud1,dpud1,ddpud1) = p(ud1)
	u2  = ud2 + sqrt((pu1-pud1)*(u1-ud1))
	up2 =  (dpu1*(u1-ud1)+pu1-pud1)/sqrt((pu1-pud1)*(u1-ud1))/2
    elif u1 > ud1: # 2-detente
	u2  = ud2 - intp(u1,ud1)
        up2 = - sqrt(dpu1)
    else:
	u2  = ud2
        up2 = - sqrt(dpu1)
    return (u2,up2)

def interstate(uag,ubg,uad,ubd):
    epsilon = 1e-10
    x = (uag+uad)/2
    fx  = 1.
    dfx = 0.
    while abs(fx)>epsilon:
        (f1x,df1x) = f1(x,uag,ubg)
        (f2x,df2x) = f2(x,uad,ubd)
	fx  = f1x-f2x
	dfx = df1x-df2x
	x = x - fx/dfx
    ua = x
    (ub, dummy) = f1(ua,uag,ubg)
    return (ua,ub)


if __name__ == "__main__":
    # init values
    try:
        numonde = int(sys.argv[1])
    except:
        numonde = 0
    # parameters
    gamma = 2./3. # exponent in the p-function
    dim = 1 # spatial dimension
    xmin, xmax = 0., 1.
    dx = 0.001 # spatial step
    la = 2. # velocity of the scheme
    if (numonde == 0): # 1-shock, 2-shock
        uag, uad, ubg, ubd = 1.50, 1.25, 1.50, 1.00
    elif (numonde == 1): # 1-shock, 2-rarefaction
        uag, uad, ubg, ubd = 1.50, 1.00, 1.25, 1.00
    elif (numonde == 2): # 1-rarefaction, 2-shock
        uag, uad, ubg, ubd = 1.00, 1.50, 1.00, 1.25
    elif (numonde == 3): # 1-rarefaction, 2-rarefaction
        uag, uad, ubg, ubd = 1.25, 1.00, 1.25, 1.50
    elif (numonde == 4): # 2-shock
        uag, uad, ubd = 1.00, 1.25, 1.25
        (ubg,dummy) = f2(uag, uad, ubd)
    elif (numonde == 5): # 2-rarefaction
        uag, uad, ubd = 1.25, 1.00, 1.25
        (ubg,dummy) = f2(uag, uad, ubd)
    elif (numonde == 6): # 1-shock
        uag, uad, ubg = 1.25, 1.00, 1.25
        (ubd,dummy) = f1(uad, uag, ubg)
    elif (numonde == 7): # 1-rarefaction
        uag, uad, ubg = 1.00, 1.25, 1.25
        (ubd,dummy) = f1(uad, uag, ubg)
    else:
        print "Odd initialization: numonde = " + str(numonde)
        sys.exit()
    Tf = 0.3 # final time
    NbImages = 10 # number of figures
    
    dico = {'dim':dim,
            'box':([xmin, xmax],),
            'space_step':dx,
            'scheme_velocity':la,
            'number_of_schemes':2,
            'init':'moments',
            0:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.9],
               'equilibrium':Matrix([u[0][0], -u[1][0]]),
               'init':{0:Riemann_pb},
               'init_args':{0:(uag, uad)}
               },
            1:{'velocities':[2,1],
               'polynomials':Matrix([1,LA*X]),
               'relaxation_parameters':[0.,1.9],
               'equilibrium':Matrix([u[1][0], u[0][0]**(-gamma)]),
               'init':{0:Riemann_pb},
               'init_args':{0:(ubg, ubd)}
               }
            }

    geom = pyLBMGeom.Geometry(dico)
    sol = pyLBMSimu.Simulation(dico, geom)

    fig = plt.figure(0,figsize=(16, 8))
    fig.clf()
    plt.ion()
    plt.subplot(121)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0][1:-1])
    plt.title("Solution mass at t={0:.3f}".format(sol.t),fontsize=14)
    plt.subplot(122)
    plt.plot(sol.Domain.x[0][1:-1],sol.m[1][0][1:-1])
    plt.title("Solution velocity at t={0:.3f}".format(sol.t),fontsize=14)
    plt.draw()
    plt.pause(1.e-3)
    compt = 0
    Ncompt = (int)(Tf/(NbImages*sol.dt))
    im = 0
    while (sol.t<=Tf):
        compt += 1
        sol.one_time_step()
        if (compt%Ncompt==0):
            im += 1
            fig.clf()
            plt.subplot(121)
            plt.plot(sol.Domain.x[0][1:-1],sol.m[0][0][1:-1])
            plt.title("Solution mass at t={0:.3f}".format(sol.t), fontsize=14)
            plt.subplot(122)
            plt.plot(sol.Domain.x[0][1:-1],sol.m[1][0][1:-1])
            plt.title("Solution velocity at t={0:.3f}".format(sol.t), fontsize=14)
            plt.draw()
            plt.pause(1.e-3)
    plt.ioff()
    plt.show()
