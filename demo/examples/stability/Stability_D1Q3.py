from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Stability of the D1Q3 for the advection
"""
from six.moves import range
import numpy as np
import pylab as plt
import sympy as sp
import pylbm

u, X = sp.symbols('u,X')

def scheme_constructor(ux, sq, sE):
    dico = {
        'dim':1,
        'scheme_velocity':1.,
        'schemes':[
            {
            'velocities':list(range(3)),
            'conserved_moments':u,
            'polynomials':[1, X, X**2],
            'equilibrium':[u, ux*u, (2*ux**2+1)/3*u],
            'relaxation_parameters':[0., sq, sE],
            },
        ],
        'stability':{
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    return pylbm.Scheme(dico)

def stability_array_in_s(ux):
    plt.figure(1)
    plt.clf()
    plt.axis('equal')
    plt.hold(True)
    N = 64
    vs_q = np.linspace(0., 2., N+1)
    vs_E = np.linspace(0., 2., N+1)
    mR = np.zeros((vs_q.size, vs_E.size))
    nb_calcul = 0
    mR, nb_calcul = stability_array_in_s_recur(vs_q, vs_E, ux, mR, [0,N,0,N], nb_calcul)
    plt.hold(False)
    print("Number of stability computations: {0:d}".format(nb_calcul))
    plt.show()

def stability_array_in_s_recur(vs_q, vs_E, ux, mR, l, nb_calcul):
    dummy = 0
    for i in l[0:2]:
        for j in l[2:4]:
            if mR[i, j] == 0:
                S = scheme_constructor(ux, vs_q[i], vs_E[j])
                nb_calcul += 1
                if S.is_L2_stable(Nk = 51):
                    plt.scatter(vs_q[i], vs_E[j], c = 'b', marker = 'o')
                    mR[i, j] = 1
                else:
                    plt.scatter(vs_q[i], vs_E[j], c = 'r', marker = 's')
                    mR[i, j] = -1
                plt.pause(1.e-5)
            dummy += mR[i, j]
    if ((l[1] - l[0] > 1) & (abs(dummy) < 4)) | (l[1] - l[0] > 16):
        laa = [l[0], (l[0]+l[1])/2, l[2], (l[2]+l[3])/2]
        lab = [l[0], (l[0]+l[1])/2, (l[2]+l[3])/2, l[3]]
        lba = [(l[0]+l[1])/2, l[1], l[2], (l[2]+l[3])/2]
        lbb = [(l[0]+l[1])/2, l[1], (l[2]+l[3])/2, l[3]]
        mR, nb_calcul = stability_array_in_s_recur(vs_q, vs_E, ux, mR, laa, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs_q, vs_E, ux, mR, lab, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs_q, vs_E, ux, mR, lba, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs_q, vs_E, ux, mR, lbb, nb_calcul)
    return mR, nb_calcul

if __name__ == "__main__":
    ux = .9
    stability_array_in_s(ux)
