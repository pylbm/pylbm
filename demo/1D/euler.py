##############################################################################
#
# Solver D1Q2Q2 for the Euler system on [0, 1]
#
# d_t(rho)   + d_x(rho u)     = 0, t > 0, 0 < x < 1,
# d_t(rho u) + d_x(rho u^2+p) = 0, t > 0, 0 < x < 1,
# d_t(E)   + d_x((E+p) u)     = 0, t > 0, 0 < x < 1,
#
# where E and p are linked by E = 1/2 rho u^2 + p/(gamma-1)
#
# then p = (gamma-1)(E - rho u^2/2)
# rho u^2 + p = (gamma-1)E + rho u^2 (3-gamma)/2
# E + p = 1/2 rho u^2 + p (1)
#
# Initial and boundary conditions are:
#
# rho(t=0,x) = rho0(x), u(t=0,x) = u0(x), E(t=0,x) = E0(x)
# d_t(rho)(t,x=0) = d_t(rho)(t,x=1) = 0
# d_t(u)(t,x=0) = d_t(u)(t,x=1) = 0
# d_t(E)(t,x=0) = d_t(E)(t,x=1) = 0
#
# the initial condition is a picewise constant function
# in order to simulate the Sod's shock tube
#
##############################################################################

import sympy as sp
import pyLBM

rho, q, E, X, LA = sp.symbols('rho, q, E, X, LA')

def Riemann_pb(x, ug, ud):
    xm = 0.5*(xmin+xmax)
    return ug*(x<xm) + ud*(x>xm) + 0.5*(ug+ud)*(x==xm)

# parameters
gamma = 1.4
xmin, xmax = 0., 1.
dx = 1.e-3 # spatial step
la = 3. # velocity of the scheme
rho_L, rho_R, p_L, p_R, u_L, u_R = 1., 1./8., 1., 0.1, 0., 0.
q_L = rho_L*u_L
q_R = rho_R*u_R
E_L = rho_L*u_L**2 + p_L/(gamma-1.)
E_R = rho_R*u_R**2 + p_R/(gamma-1.)
Tf = 0.14 # final time
s_rho, s_q, s_E = 1.9, 1.5, 1.4

dico = {
    'box':{'x':[xmin, xmax], 'label':0},
    'space_step':dx,
    'scheme_velocity':la,
    'schemes':[
        {
            'velocities':[1,2],
            'conserved_moments':rho,
            'polynomials':[1, LA*X],
            'relaxation_parameters':[0, s_rho],
            'equilibrium':[rho, q],
            'init':{rho:(Riemann_pb, (rho_L, rho_R))},
        },
        {
            'velocities':[1,2],
            'conserved_moments':q,
            'polynomials':[1, LA*X],
            'relaxation_parameters':[0, s_q],
            'equilibrium':[q, (gamma-1.)*E+0.5*(3.-gamma)*q**2/rho],
            'init':{q:(Riemann_pb, (q_L, q_R))},
        },
        {
            'velocities':[1,2],
            'conserved_moments':E,
            'polynomials':[1, LA*X],
            'relaxation_parameters':[0, s_E],
            'equilibrium':[E, gamma*E*q/rho-0.5*(gamma-1.)*q**3/rho**2],
            'init':{E:(Riemann_pb, (E_L, E_R))},
        },
    ],
    'boundary_conditions':{
        0:{
            'method':{
                0: pyLBM.bc.neumann,
                1: pyLBM.bc.neumann,
                2: pyLBM.bc.neumann
            },
            'value':None
        },
    },
    'parameters':{LA:la},
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)

while (sol.t<Tf):
    sol.one_time_step()

sol.f2m()
sol.time_info()

x = sol.domain.x[0][1:-1]
rho = sol.m[0][0][1:-1]
q = sol.m[1][0][1:-1]
E = sol.m[2][0][1:-1]
u = q/rho
p = (gamma-1.)*(E - .5*rho*u**2)
e = E/rho - .5*u**2

viewer= pyLBM.viewer.matplotlibViewer
fig = viewer.Fig(2, 3)

fig[0,0].plot(x, rho)
fig[0,0].title = 'mass'
fig[0,1].plot(x, u)
#fig[0,1].title = 'velocity'
fig[0,2].plot(x, p)
#fig[0,2].title = 'pressure'
fig[1,0].plot(x, E)
#fig[1,0].title = 'energy'
fig[1,1].plot(x, q)
#fig[1,1].title = 'momentum'
fig[1,2].plot(x, e)
#fig[1,2].title = 'internal energy'

fig.show()
