from __future__ import print_function
from __future__ import division
from six.moves import range
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import pylbm

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def init_m(x, y, val):
    return val*np.ones((y.size, x.size))

def init_un(x, y):
    uu = np.zeros((y.size, x.size))
    uu[y.size//2, x.size//2] = 1.
    return uu

def test_transport():
    # parameters
    dim = 2 # spatial dimension
    xmin, xmax, ymin, ymax = -0.5, 4.5, -0.5, 4.5
    dx = 1. # spatial step
    la = 1. # velocity of the scheme
    Tf = 5

    rhoo = 1.
    s  = [0.,0.,0.,1.,1.,1.,1.,1.,1.]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy

    vitesse = list(range(9))

    polynomes = [1,
                 LA*X, LA*Y,
                 3*(X**2+Y**2)-4,
                 0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                 X**2-Y**2, X*Y]
    equilibre = [rho,
                 qx, qy,
                 -2*rho + 3*q2,
                 rho + 1.5*q2,
                 qx/LA, qy/LA,
                 qx2-qy2, qxy]

    dico   = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[-1]*4},#[0, 0, 1, 1]},
        'space_step':dx,
        'scheme_velocity':la,
        'inittype':'distributions',
        'schemes':[{'velocities':vitesse,
                      'polynomials':polynomes,
                      'relaxation_parameters':s,
                      'equilibrium':equilibre,
                      'conserved_moments': [rho, qx, qy],
                      'init':{0:(init_un,),
                              1:(init_un,),
                              2:(init_un,),
                              3:(init_un,),
                              4:(init_un,),
                              5:(init_un,),
                              6:(init_un,),
                              7:(init_un,),
                              8:(init_un,),
                              },
                    },
                    ],
        #'generator': pylbm.generator.CythonGenerator,
        'boundary_conditions':{
            0:{'method':{0: pylbm.bc.Bouzidi_bounce_back}},
            1:{'method':{0: pylbm.bc.Bouzidi_anti_bounce_back}},
        },
        'parameters':{'LA':1.},
        }


    sol = pylbm.Simulation(dico)

    while (sol.t<Tf-0.5*sol.dt):
        #sol.m2f()
        sol.boundary_condition()
        sol.transport()
        #sol.f2m()
        sol.t += sol.dt
        print(sol.t)
    print()
    print(sol.F[rho][1:-1, 1:-1])
    print()
    print(sol.F[qx][1:-1, 1:-1])
    print()
    print(sol.F[qy][1:-1, 1:-1])

# def test_relaxation():
#     # parameters
#     dim = 2 # spatial dimension
#     xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
#     dx = 1. # spatial step
#     la = 1. # velocity of the scheme
#     Tf = 10
#     rhoo = 1.
#     s  = [0., 0., 0., 1.9, 1.8, 1.7, 1.75, 1.85, 1.95]
#
#     rhoi = 1.
#     qxi = -0.2
#     qyi = 1.2
#
#     dummy = 1./(LA**2*rhoo)
#     qx2 = dummy*u[0][1]**2
#     qy2 = dummy*u[0][2]**2
#     q2  = qx2+qy2
#     qxy = dummy*u[0][1]*u[0][2]
#     dico_geometry = {'dim':dim,
#                      'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':0},
#                      'Elements':[]
#                      }
#     dico   = {'dim':dim,
#               'Geometry':dico_geometry,
#               'space_step':dx,
#               'scheme_velocity':la,
#               'number_of_schemes':1,
#               'init':'moments',
#               0:{'velocities':range(9),
#                  'polynomials':Matrix([1,
#                                        LA*X, LA*Y,
#                                        3*(X**2+Y**2)-4,
#                                        0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
#                                        3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
#                                        X**2-Y**2, X*Y]),
#                  'relaxation_parameters':s,
#                  'equilibrium':Matrix([u[0][0],
#                                        u[0][1], u[0][2],
#                                        -2*u[0][0] + 3*q2,
#                                        u[0][0]+1.5*q2,
#                                        u[0][1]/LA, u[0][2]/LA,
#                                        qx2-qy2, qxy]),
#                  'init':{0:init_rho, 1:init_qx, 2:init_qy},
#                  'init_args':{0:(rhoi,), 1:(qxi,), 2:(qyi,)}
#                  }
#             }
#
#     geom = pylbmGeom.Geometry(dico)
#     dom = pylbmDom.Domain(geom,dico)
#     sol = pylbmSimu.Simulation(dico, geom)
#     print sol.Scheme.Code_Relaxation
#     q2 = qxi**2+qyi**2
#     valeq = [rhoi, qxi, qyi, -2*rhoi+3*q2, rhoi+1.5*q2, qxi, qyi, qxi**2-qyi**2, qxi*qyi]
#     sol.m[0][3:,:,:] = 0.
#     fig = plt.figure(0,figsize=(16, 8))
#     fig.clf()
#     plt.ion()
#     plot_m(sol,valeq)
#     while (sol.t<Tf-0.5*sol.dt):
#         sol.Scheme.m2f(sol.m, sol.F)
#         sol.Scheme.f2m(sol.F, sol.m)
#         sol.Scheme.relaxation(sol.m)
#         sol.t += sol.dt
#         plot_m(sol,valeq)
#     plt.ioff()
#     plt.show()

if __name__ == "__main__":
    test_transport()
