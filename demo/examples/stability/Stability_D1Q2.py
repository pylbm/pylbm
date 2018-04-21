from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Stability of the D1Q2
"""
import numpy as np
import pylab as plt
import sympy as sp
import pylbm

u, X = sp.symbols('u,X')

def scheme_constructor(ux, s):
    dico = {
        'dim':1,
        'scheme_velocity':1.,
        'schemes':[
            {
            'velocities':[1, 2],
            'conserved_moments':u,
            'polynomials':[1, X],
            'relaxation_parameters':[0., s],
            'equilibrium':[u, ux*u],
            },
        ],
        'stability':{
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    return pylbm.Scheme(dico)

def vp_plot(ux):
    Nk = 100
    vkx = np.linspace(0., 2*np.pi, Nk+1)
    Ns = 50
    plt.figure(1)
    for s in np.linspace(0., 2., Ns+1):
        S = scheme_constructor(ux, s)
        R = 1.
        plt.clf()
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        plt.hold(True)
        plt.plot(np.cos(np.linspace(0., 2.*np.pi, 200)),
            np.sin(np.linspace(0., 2.*np.pi, 200)), 'r')
        for k in range(Nk):
            vp = S.vp_amplification_matrix((vkx[k],))
            rloc = max(np.abs(vp))
            plt.plot(vp.real, vp.imag, 'ko')
            if rloc>R+1.e-14:
                R = rloc
        if R>1+1.e-14:
            print("instable scheme for s={0:5.3f}".format(s))
        plt.hold(False)
        plt.title('eigenvalues for $s = {0:5.3f}$'.format(s))
        plt.pause(1.e-1)

def stability_array():
    plt.figure(1)
    plt.clf()
    plt.axis('equal')
    plt.hold(True)
    Nu = 32
    Ns = 64
    vux = np.linspace(0., 1., Nu+1)
    vs = np.linspace(0., 2., Ns+1)
    mR = np.zeros((vux.size, vs.size))
    nb_calcul = 0
    mR, nb_calcul = stability_array_recur(vs, vux, mR, [0,Nu,0,Ns], nb_calcul)
    plt.hold(False)
    print("Number of stability computations: {0:d}".format(nb_calcul))
    plt.show()

def stability_array_recur(vs, vux, mR, l, nb_calcul):
    dummy = 0
    for i in l[0:2]:
        for j in l[2:4]:
            if (mR[i, j] == 0):
                S = scheme_constructor(vux[i], vs[j])
                nb_calcul += 1
                if S.is_monotonically_stable():
                    plt.scatter([vux[i], -vux[i]], [vs[j], vs[j]], c = 'b', marker = 'o')
                    mR[i, j] = 1
                else:
                    plt.scatter([vux[i], -vux[i]], [vs[j], vs[j]], c = 'r', marker = 's')
                    mR[i, j] = -1
                plt.pause(1.e-5)
            dummy += mR[i, j]
    taille = max(l[1] - l[0], l[3] - l[2])
    if ((taille > 1) & (abs(dummy) < 4)) | (taille > 16):
        laa = [l[0], (l[0]+l[1])/2, l[2], (l[2]+l[3])/2]
        lab = [l[0], (l[0]+l[1])/2, (l[2]+l[3])/2, l[3]]
        lba = [(l[0]+l[1])/2, l[1], l[2], (l[2]+l[3])/2]
        lbb = [(l[0]+l[1])/2, l[1], (l[2]+l[3])/2, l[3]]
        mR, nb_calcul = stability_array_recur(vs, vux, mR, laa, nb_calcul)
        mR, nb_calcul = stability_array_recur(vs, vux, mR, lab, nb_calcul)
        mR, nb_calcul = stability_array_recur(vs, vux, mR, lba, nb_calcul)
        mR, nb_calcul = stability_array_recur(vs, vux, mR, lbb, nb_calcul)
    return mR, nb_calcul

if __name__ == "__main__":
    vp_plot(0.75)
    stability_array()
