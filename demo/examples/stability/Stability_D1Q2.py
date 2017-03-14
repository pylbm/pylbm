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
import pyLBM

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
    return pyLBM.Scheme(dico)

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

if __name__ == "__main__":
    vp_plot(0.75)
