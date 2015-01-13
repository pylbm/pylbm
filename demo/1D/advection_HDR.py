import sys
import cmath
from math import pi, sqrt, log10
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
import pyLBM.Domain as pyLBMDom
import pyLBM.Scheme as pyLBMScheme

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

def Riemann_pb(x):
    ug, ud = 0.5, 0.0 # left and right state
    xm = 0.5*(xmin+xmax)
    return ug*(x<=xm) + ud*(x>xm)

def Smooth(x):
    milieu = 0.5*(xmin+xmax)
    largeur = 0.1*(xmax-xmin)
    milieu -= 0.5*c*Tf
    return 1.0/largeur**10 * (x-milieu-largeur)**5 * (milieu-x-largeur)**5 * (abs(x-milieu)<=largeur)
    

def Calcul_D1Q2(k, FINIT, norm=1):
    dx = 2**(-k) # spatial step
    dicoQ2 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,1],
                 'polynomials':Matrix([1,LA*X]),
                 'relaxation_parameters':[0.,1.5],
                 'equilibrium':Matrix([u[0][0], c*u[0][0]]),
                 'init':{0:FINIT}
                 }
                }
    geom = pyLBMGeom.Geometry(dicoQ2)
    sol = pyLBMSimu.Simulation(dicoQ2, geom)
    while (sol.t<Tf):
        sol.one_time_step_fast()
    exacte = FINIT(sol.Domain.x[0][1:-1] - c*sol.t)
    if (norm == 1):
        Err = dx*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 1)
    elif (norm == 2):
        Err = sqrt(dx)*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 2)
    else:
        print 'Bad choice of norm'
        sys.exit()
    print 'Error for k={0:2d}: {1:10.3e}'.format(k,Err)
    return Err

def Calcul_D1Q3(k, FINIT, norm=1):
    dx = 2**(-k) # spatial step
    s1 = 1.
    sigma1 = 1./s1-0.5
    sigma2 = sqrt(sigma1**2+1./(64*sigma1**2)) - sigma1 + 1./(8*sigma1)
    sQ3 = [0., 1./(0.5+sigma1), 1./(0.5+sigma2)]
    dicoQ3 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':la,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,0,1],
                 'polynomials':Matrix([1,LA*X,LA**2*X**2]),
                 'relaxation_parameters':sQ3,
                 'equilibrium':Matrix([u[0][0], c*u[0][0], c**2*u[0][0]]),
                 'init':{0:FINIT}
                 }
            }
    """
    s1 = 2.
    s2 = s1
    sQ3 = [0., s1, s2]
    dicoQ3 = {'dim':dim,
              'box':([xmin, xmax],),
              'space_step':dx,
              'scheme_velocity':2,
              'number_of_schemes':1,
              'init':'moments',
              0:{'velocities':[2,0,1],
                 'polynomials':Matrix([1,LA*X,LA**2*X**2]),
                 'relaxation_parameters':sQ3,
                 'equilibrium':Matrix([u[0][0], c*u[0][0], (2*c**2+LA**2)/3*u[0][0]]),
                 'init':{0:Smooth}
                 #'init':{0:Riemann_pb},
                 #'init_args':{0:(ug, ud)}
                 }
            }
    """
    geom = pyLBMGeom.Geometry(dicoQ3)
    sol = pyLBMSimu.Simulation(dicoQ3, geom)
    sol.m[0][2,1:-1] -= 0.5/sol.Scheme.s[0][2] * c/sol.Scheme.la*(sol.Scheme.la**2-c**2) * (sol.m[0][0,2:]-sol.m[0][0,0:-2])
    while (sol.t<Tf):
        sol.one_time_step()
    exacte = FINIT(sol.Domain.x[0][1:-1] - c*sol.t)
    if (norm == 1):
        Err = dx*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 1)
    elif (norm == 2):
        Err = sqrt(dx)*np.linalg.norm(sol.m[0][0][1:-1] - exacte, 2)
    else:
        print 'Bad choice of norm'
        sys.exit()
    print 'Error for k={0:2d}: {1:10.3e}'.format(k,Err)
    return Err

if __name__ == "__main__":
    # parameters
    dim = 1 # spatial dimension
    xmin, xmax = 0., 1.
    FINIT = Riemann_pb
    la = 1. # velocity of the scheme
    c = 0.75 # velocity of the advection
    Tf = 0.4
    KK = range(3, 15)
    EK2 = []
    EK3 = []
    for k in KK:
        EK2.append(log10(Calcul_D1Q2(k,FINIT)))
        EK3.append(log10(Calcul_D1Q3(k,FINIT)))
    slope2 = (EK2[-1]-EK2[-2]) / (log10(2**(-KK[-1]))-log10(2**(-KK[-2])))
    slope3 = (EK3[-1]-EK3[-2]) / (log10(2**(-KK[-1]))-log10(2**(-KK[-2])))
    #fig = plt.figure(0,figsize=(8, 8))
    #fig.clf()
    #plt.ion()
    #plt.plot(KK, EK2, 'b*', KK, EK3, 'rd')
    #plt.title('Slope D1Q2 {0:5.3f}, D1Q3 {1:5.3f}'.format(slope2, slope3))
    #plt.ioff()
    #plt.draw()
    #plt.show()
    

