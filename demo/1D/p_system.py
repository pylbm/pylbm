import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyLBM

ua, ub, X, LA = sp.symbols('ua,ub,X,LA,')

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
    y = np.sqrt(gamma)/alpha*(x**alpha-xo**alpha)
    return y

def f1(u1,ug1,ug2): # parametrisation de la 1-onde
    (pu1,dpu1,ddpu1) = p(u1)
    if u1 < ug1: # 1-choc
	(pug1,dpug1,ddpug1) = p(ug1)
	u2  = ug2 - np.sqrt((pu1-pug1)*(u1-ug1))
	up2 = -(dpu1*(u1-ug1)+pu1-pug1)/np.sqrt((pu1-pug1)*(u1-ug1))/2
    elif u1 > ug1: # 1-detente
	u2  = ug2 + intp(u1,ug1)
	up2 = np.sqrt(dpu1)
    else: # rien
	u2  = ug2
	up2 = np.sqrt(dpu1)
    return (u2,up2)

def f2(u1,ud1,ud2): # parametrisation de la 2-onde
    (pu1,dpu1,ddpu1) = p(u1)
    if u1 < ud1: # 2-choc
	(pud1,dpud1,ddpud1) = p(ud1)
	u2  = ud2 + np.sqrt((pu1-pud1)*(u1-ud1))
	up2 =  (dpu1*(u1-ud1)+pu1-pud1)/np.sqrt((pu1-pud1)*(u1-ud1))/2
    elif u1 > ud1: # 2-detente
	u2  = ud2 - intp(u1,ud1)
        up2 = - np.sqrt(dpu1)
    else:
	u2  = ud2
        up2 = - np.sqrt(dpu1)
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

def plot_init(num = 0):
    fig = plt.figure(num,figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    l1 = Line2D([], [], color='b', marker='*', linestyle='None')
    l2 = Line2D([], [], color='r', marker='d', linestyle='None')
    ax1.add_line(l1)
    ax2.add_line(l2)
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax1.set_ylim(.9*ymina, 1.1*ymaxa)
    ax2.set_ylim(.9*yminb, 1.1*ymaxb)
    t1 = ax1.text(0.5*(xmin+xmax), ymaxa, '')
    t2 = ax2.text(0.5*(xmin+xmax), ymaxb, '')
    return [l1, l2, t1, t2]

def plot(sol, l):
    sol.f2m()
    x = sol.domain.x[0][1:-1]
    l[0].set_data(x, sol.m[0][0][1:-1])
    l[1].set_data(x, sol.m[1][0][1:-1])
    l[2].set_text(r'$u_a$ at $t = {0:f}$'.format(sol.t))
    l[3].set_text(r'$u_b$ at $t = {0:f}$'.format(sol.t))
    plt.pause(1.e-3)

if __name__ == "__main__":
    # parameters
    gamma = 2./3.        # exponent in the p-function
    xmin, xmax = 0., 1.  # bounds of the domain
    dx = 1./256          # spatial step
    la = 2.              # velocity of the scheme
    s = 1.7              # relaxation parameter
    Tf = 0.25            # final time

    # init values
    try:
        numonde = int(sys.argv[1])
    except:
        numonde = 0
    if (numonde == 0): # 1-shock, 2-shock
        uag, uad, ubg, ubd = 1.50, 1.25, 1.50, 1.00
        ymina, ymaxa, yminb, ymaxb = 1., 1.75, 1., 1.5
    elif (numonde == 1): # 1-shock, 2-rarefaction
        uag, uad, ubg, ubd = 1.50, 1.00, 1.25, 1.00
        ymina, ymaxa, yminb, ymaxb = 1., 1.75, 1., 1.5
    elif (numonde == 2): # 1-rarefaction, 2-shock
        uag, uad, ubg, ubd = 1.00, 1.50, 1.00, 1.25
        ymina, ymaxa, yminb, ymaxb = 1., 1.75, 1., 1.5
    elif (numonde == 3): # 1-rarefaction, 2-rarefaction
        uag, uad, ubg, ubd = 1.25, 1.00, 1.25, 1.50
        ymina, ymaxa, yminb, ymaxb = 1., 1.5, 1.2, 1.6
    elif (numonde == 4): # 2-shock
        uag, uad, ubd = 1.00, 1.25, 1.25
        (ubg,dummy) = f2(uag, uad, ubd)
        ymina, ymaxa, yminb, ymaxb = 1., 1.3, 1.25, 1.5
    elif (numonde == 5): # 2-rarefaction
        uag, uad, ubd = 1.25, 1.00, 1.25
        (ubg,dummy) = f2(uag, uad, ubd)
        ymina, ymaxa, yminb, ymaxb = 1., 1.3, 1., 1.3
    elif (numonde == 6): # 1-shock
        uag, uad, ubg = 1.25, 1.00, 1.25
        (ubd,dummy) = f1(uad, uag, ubg)
        ymina, ymaxa, yminb, ymaxb = 1., 1.3, 1., 1.3
    elif (numonde == 7): # 1-rarefaction
        uag, uad, ubg = 1.00, 1.25, 1.25
        (ubd,dummy) = f1(uad, uag, ubg)
        ymina, ymaxa, yminb, ymaxb = 1., 1.3, 1.25, 1.5
    else:
        print "Odd initialization: numonde = " + str(numonde)
        sys.exit()

    dico = {
        'box':{'x':[xmin, xmax], 'label':0},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':[1,2],
                'conserved_moments':ua,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[ua, -ub],
                'init':{ua:(Riemann_pb, (uag, uad))},
            },
            {
                'velocities':[1,2],
                'conserved_moments':ub,
                'polynomials':[1, LA*X],
                'relaxation_parameters':[0, s],
                'equilibrium':[ub, ua**(-gamma)],
                'init':{ub:(Riemann_pb, (ubg, ubd))},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.neumann, 1: pyLBM.bc.neumann}, 'value':None},
        },
        'parameters':{LA:la},
    }

    sol = pyLBM.Simulation(dico)
    l = plot_init()
    plot(sol, l)
    while (sol.t<Tf):
        sol.one_time_step()
        plot(sol, l)
    plt.show()
