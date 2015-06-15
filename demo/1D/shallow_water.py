##############################################################################
#
# Solver D1Q2Q2 for the shallow water system on [0, 1]
#
# d_t(h) + d_x(q)    = 0, t > 0, 0 < x < 1,
# d_t(q) + d_x(q^2/h+gh^2/2) = 0, t > 0, 0 < x < 1,
# h(t=0,x) = h0(x), q(t=0,x) = q0(x),
# d_t(h)(t,x=0) = d_t(h)(t,x=1) = 0
# d_t(q)(t,x=0) = d_t(q)(t,x=1) = 0
#
# the initial condition is a picewise constant function
# in order to visualize the simulation of elementary waves
#
##############################################################################

import sympy as sp
import pyLBM

h, q, X, LA, g = sp.symbols('h,q,X,LA,g')

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

# parameters
xmin, xmax = 0., 1.  # bounds of the domain
dx = 1./256          # spatial step
la = 2.              # velocity of the scheme
s = 1.7              # relaxation parameter
Tf = 0.25            # final time

hg, hd, qg, qd = 1., .25, 0.10, 0.10
ymina, ymaxa, yminb, ymaxb = 0., 1., 0., .5

dico = {
    'box':{'x':[xmin, xmax], 'label':0},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':[1,2],
            'conserved_moments':h,
            'polynomials':[1, LA*X],
            'relaxation_parameters':[0, s],
            'equilibrium':[h, q],
            'init':{h:(Riemann_pb, (hg, hd))},
        },
        {
            'velocities':[1,2],
            'conserved_moments':q,
            'polynomials':[1, LA*X],
            'relaxation_parameters':[0, s],
            'equilibrium':[q, q**2/h+.5*g*h**2],
            'init':{q:(Riemann_pb, (qg, qd))},
        },
    ],
    'boundary_conditions':{
        0:{'method':{0: pyLBM.bc.neumann, 1: pyLBM.bc.neumann}, 'value':None},
    },
    'parameters':{LA:la, g:1.},
}

sol = pyLBM.Simulation(dico)

# create the viewer to plot the solution
viewer = pyLBM.viewer.matplotlibViewer
fig = viewer.Fig(2, 1)
ax1 = fig[0]
ax1.axis(xmin, xmax, .9*ymina, 1.1*ymaxa)
ax2 = fig[1]
ax2.axis(xmin, xmax, .9*yminb, 1.1*ymaxb)

x = sol.domain.x[0][1:-1]
l1 = ax1.plot(x, sol.m[0][0][1:-1], color='b')[0]
l2 = ax2.plot(x, sol.m[1][0][1:-1], color='r')[0]
p1 = [0.5*(xmin+xmax), ymaxa]
t1 = ax1.text(r'$h$ at $t = {0:f}$'.format(sol.t), p1)[0]
p2 = [0.5*(xmin+xmax), ymaxb]
t2 = ax2.text(r'$q$ at $t = {0:f}$'.format(sol.t), p2)[0]

def update(iframe):
    if sol.t<Tf:
        sol.one_time_step()
        sol.f2m()
        l1.set_data(x, sol.m[0][0][1:-1])
        l2.set_data(x, sol.m[1][0][1:-1])
        t1.set_text(r'$u_a$ at $t = {0:f}$'.format(sol.t))
        t2.set_text(r'$u_b$ at $t = {0:f}$'.format(sol.t))

fig.animate(update)
fig.show()
