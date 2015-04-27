import sympy as sp
import pyLBM
from nose import tools

X, Y, Z, LA, g = sp.symbols('X,Y,Z,LA,g')
rho, qx, qy = sp.symbols('rho, qx, qy')
m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]

poly = [[1,
         LA*X, LA*Y,
         3*(X**2+Y**2)-4,
         0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
         3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
         X**2-Y**2, X*Y],
         [1,
         'LA*X', 'LA*Y',
         '3*(X**2+Y**2)-4',
         '0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)',
         '3*X*(X**2+Y**2)-5*X', '3*Y*(X**2+Y**2)-5*Y',
         'X**2-Y**2', 'X*Y']]

eq = [[m[0][0],
       m[0][1], m[0][2],
       -2*m[0][0] + 3*(m[0][1]**2+m[0][2]**2),
       m[0][0]+1.5*(m[0][1]**2+m[0][2]**2),
       -m[0][1]/LA, -m[0][2]/LA,
       m[0][1]**2-m[0][2]**2, m[0][1]*m[0][2]],
       ['m[0][0]',
       'm[0][1]', 'm[0][2]',
       '-2*m[0][0] + 3*(m[0][1]**2+m[0][2]**2)',
       'm[0][0]+1.5*(m[0][1]**2+m[0][2]**2)',
       '-m[0][1]/LA', '-m[0][2]/LA',
       'm[0][1]**2-m[0][2]**2', 'm[0][1]*m[0][2]'],
       [rho,
       qx, qy,
       -2*qx + 3*(qx**2+qy**2),
       qx+1.5*(qx**2+qy**2),
       -qx/LA, -qy/LA,
       qx**2-qy**2, qx*qy]
       ]

param = [{LA: 1., g: 1.}, {'LA': 1., 'g': 1.}]

consm = [None, [rho, qx, qy]]

# def test_D2Q9():
#     seq = []#[map(str, e) for e in eq]
#     for pa in param:
#         for p in poly:
#             for ie, e in enumerate(eq + seq):
#                 dico = {'dim':2,
#                         'scheme_velocity': 1.,
#                         'schemes':[{'velocities':range(9),
#                                     'polynomials':p,
#                                     'relaxation_parameters':[0]*9,
#                                     'equilibrium':e
#                                     }],
#                         'parameters':pa,
#                         'conserved_moments': consm[ie%2],
#                         }
#                 yield construct_scheme, dico

polyD2Q4 = [[1, LA*X, LA*Y, X**2-Y**2],
            ['1', 'LA*X', 'LA*Y', 'X**2-Y**2']]

eqD2Q4Q4Q4 = [[[m[0][0], m[1][0], m[2][0], 0.],
               [m[1][0], m[1][0]**2/m[0][0] + 0.5*g*m[0][0]**2, m[1][0]*m[2][0]/m[0][0], 0.],
               [m[2][0], m[1][0]*m[2][0]/m[0][0], m[2][0]**2/m[0][0] + 0.5*g*m[0][0]**2, 0.]],
               [[rho, qx, qy, 0.],
                [qx, qx**2/rho + .5*g*rho**2, qx*qy/rho, 0.],
                [qy, qx*qy/rho, qy**2/rho + 0.5*g*rho**2, 0.]]
            ]

consmD2Q4Q4Q4 = [[None, None, None], [rho, qx, qy]]

from sympy import Matrix

EQ_result = [Matrix([
[m[0][0]],
[m[1][0]],
[m[2][0]],
[    0.0]]), Matrix([
[                              m[1][0]],
[0.5*g*m[0][0]**2 + m[1][0]**2/m[0][0]],
[              m[1][0]*m[2][0]/m[0][0]],
[                                  0.0]]), Matrix([
[                              m[2][0]],
[              m[1][0]*m[2][0]/m[0][0]],
[0.5*g*m[0][0]**2 + m[2][0]**2/m[0][0]],
[                                  0.0]])]

def test_D2Q4Q4Q4():
    seq = []
    for e1 in eqD2Q4Q4Q4:
        print type(e1)
        seq.append([map(str, e2) for e2 in e1])

    for pa in param:
        for p in polyD2Q4:
            for ie, e in enumerate(eqD2Q4Q4Q4 + seq):
                schemes = []
                for js, s in enumerate(e):
                    schemes.append({
                    'velocities':range(1,5),
                    'polynomials':p,
                    'relaxation_parameters':[0]*4,
                    'equilibrium':s,
                    'conserved_moments': consmD2Q4Q4Q4[ie%2][js]
                    })

                dico = {'dim':2,
                        'scheme_velocity': 1.,
                        'schemes': schemes,
                        'parameters': pa,
                        }
                yield construct_scheme, dico

def construct_scheme(dico):
    s = pyLBM.Scheme(dico)
    tools.eq_(s._EQ, EQ_result)
