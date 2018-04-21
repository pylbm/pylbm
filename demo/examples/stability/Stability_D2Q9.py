from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Stability of the D2Q9
"""
from six.moves import range
import numpy as np
import pylab as plt
import sympy as sp
import pylbm

rho, qx, qy, X, Y, LA = sp.symbols('rho,qx,qy,X,Y,LA')

def scheme_constructor(ux, uy, s_mu, s_eta):
    rhoo = 1.
    la = 1.
    s3 = s_mu
    s4 = s3
    s5 = s4
    s6 = s4
    s7 = s_eta
    s8 = s7
    s  = [0.,0.,0.,s3,s4,s5,s6,s7,s8]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy

    dico = {
        'dim':2,
        'scheme_velocity':la,
        'parameters':{LA:la},
        'schemes':[
            {
            'velocities':list(range(9)),
            'conserved_moments':[rho, qx, qy],
            'polynomials':[
                1,
                LA*X, LA*Y,
                3*(X**2+Y**2)-4,
                0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                X**2-Y**2, X*Y
            ],
            'relaxation_parameters':s,
            'equilibrium':[rho, qx, qy,
                -2*rho + 3*q2,
                rho+1.5*q2,
                -qx/LA, -qy/LA,
                qx2-qy2, qxy
            ],
            },
        ],
        'stability':{
            'linearization':{rho: rhoo, qx: ux, qy: uy},
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    return pylbm.Scheme(dico)

def vp_plot(ux, uy, s_mu, s_eta):
    S = scheme_constructor(ux, uy, s_mu, s_eta)
    Nk = 100
    Nangle = 25

    R = 1.
    plt.figure(1)
    i = 0
    for theta in np.linspace(0., 0.25*np.pi, Nangle):
        plt.clf()
        plt.hold(True)
        plt.plot(np.cos(np.linspace(0., 2.*np.pi, 200)),
            np.sin(np.linspace(0., 2.*np.pi, 200)), 'r')
        vkx = np.linspace(0., 2*np.pi, Nk) * np.cos(theta)
        vky = np.linspace(0., 2*np.pi, Nk) * np.sin(theta)
        for k in range(Nk):
            vp = S.vp_amplification_matrix((vkx[k], vky[k]))
            rloc = max(np.abs(vp))
            if rloc > R+1.e-14:
                R = rloc
                print("Spectral radius for theta = {0:5.3f}: {1:10.3e}".format(theta, rloc))
            plt.plot(vp.real, vp.imag, 'ko')
        i += 1
        plt.hold(False)
        plt.title('eigenvalues for $\Theta = {0:5.3f}$'.format(theta))
        plt.pause(1.e-1)
    print("Maximal spectral radius: {0:10.3e}".format(R))

def stability_array_in_u(s_mu, s_eta):
    plt.figure(1)
    plt.clf()
    plt.axis('equal')
    plt.hold(True)
    N = 32
    vux = np.linspace(0., 1., N+1)
    vuy = np.linspace(0., 1., N+1)
    mR = np.zeros((vux.size, vuy.size))
    nb_calcul = 0
    mR, nb_calcul = stability_array_in_u_recur(s_mu, s_eta, vux, vuy, mR, [0,N,0,N], nb_calcul)
    plt.hold(False)
    print("Number of stability computations: {0:d}".format(nb_calcul))
    plt.show()

def stability_array_in_u_recur(s_mu, s_eta, vux, vuy, mR, l, nb_calcul):
    dummy = 0
    for i in l[0:2]:
        for j in l[2:4]:
            if (mR[i, j] == 0) & (mR[j, i] == 0):
                S = scheme_constructor(vux[i], vuy[j], s_mu, s_eta)
                nb_calcul += 1
                if S.is_L2_stable(Nk = 51):
                    plt.scatter([vux[i], vux[i], -vux[i], -vux[i]],
                                [vuy[j], -vuy[j], vuy[j], -vuy[j]],
                                c = 'b', marker = 'o')
                    mR[i, j] = 1
                    plt.scatter([vux[j], vux[j], -vux[j], -vux[j]],
                                [vuy[i], -vuy[i], vuy[i], -vuy[i]],
                                c = 'b', marker = 'o')
                    mR[j, i] = 1
                else:
                    plt.scatter([vux[i], vux[i], -vux[i], -vux[i]],
                                [vuy[j], -vuy[j], vuy[j], -vuy[j]],
                                c = 'r', marker = 's')
                    mR[i, j] = -1
                    plt.scatter([vux[j], vux[j], -vux[j], -vux[j]],
                                [vuy[i], -vuy[i], vuy[i], -vuy[i]],
                                c = 'r', marker = 's')
                    mR[j, i] = -1
                plt.pause(1.e-5)
            dummy += mR[i, j]
    if ((l[1] - l[0] > 1) & (abs(dummy) < 4)) | (l[1] - l[0] > 16):
        laa = [l[0], (l[0]+l[1])/2, l[2], (l[2]+l[3])/2]
        lab = [l[0], (l[0]+l[1])/2, (l[2]+l[3])/2, l[3]]
        lba = [(l[0]+l[1])/2, l[1], l[2], (l[2]+l[3])/2]
        lbb = [(l[0]+l[1])/2, l[1], (l[2]+l[3])/2, l[3]]
        mR, nb_calcul = stability_array_in_u_recur(s_mu, s_eta, vux, vuy, mR, laa, nb_calcul)
        mR, nb_calcul = stability_array_in_u_recur(s_mu, s_eta, vux, vuy, mR, lab, nb_calcul)
        mR, nb_calcul = stability_array_in_u_recur(s_mu, s_eta, vux, vuy, mR, lba, nb_calcul)
        mR, nb_calcul = stability_array_in_u_recur(s_mu, s_eta, vux, vuy, mR, lbb, nb_calcul)
    return mR, nb_calcul

def stability_array_in_s(ux, uy):
    plt.figure(1)
    plt.clf()
    plt.axis('equal')
    plt.hold(True)
    N = 64
    vs_mu = np.linspace(0., 2., N+1)
    vs_eta = np.linspace(0., 2., N+1)
    mR = np.zeros((vs_mu.size, vs_eta.size))
    nb_calcul = 0
    mR, nb_calcul = stability_array_in_s_recur(vs_mu, vs_eta, ux, uy, mR, [0,N,0,N], nb_calcul)
    plt.hold(False)
    print("Number of stability computations: {0:d}".format(nb_calcul))
    plt.show()

def stability_array_in_s_recur(vs_mu, vs_eta, ux, uy, mR, l, nb_calcul):
    dummy = 0
    for i in l[0:2]:
        for j in l[2:4]:
            if mR[i, j] == 0:
                S = scheme_constructor(ux, uy, vs_mu[i], vs_eta[j])
                nb_calcul += 1
                if S.is_L2_stable(Nk = 51):
                    plt.scatter(vs_mu[i], vs_eta[j], c = 'b', marker = 'o')
                    mR[i, j] = 1
                else:
                    plt.scatter(vs_mu[i], vs_eta[j], c = 'r', marker = 's')
                    mR[i, j] = -1
                plt.pause(1.e-5)
            dummy += mR[i, j]
    if ((l[1] - l[0] > 1) & (abs(dummy) < 4)) | (l[1] - l[0] > 16):
        laa = [l[0], (l[0]+l[1])/2, l[2], (l[2]+l[3])/2]
        lab = [l[0], (l[0]+l[1])/2, (l[2]+l[3])/2, l[3]]
        lba = [(l[0]+l[1])/2, l[1], l[2], (l[2]+l[3])/2]
        lbb = [(l[0]+l[1])/2, l[1], (l[2]+l[3])/2, l[3]]
        mR, nb_calcul = stability_array_in_s_recur(vs_mu, vs_eta, ux, uy, mR, laa, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs_mu, vs_eta, ux, uy, mR, lab, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs_mu, vs_eta, ux, uy, mR, lba, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs_mu, vs_eta, ux, uy, mR, lbb, nb_calcul)
    return mR, nb_calcul

if __name__ == "__main__":
    ux, uy = 0.1, 0.1
    s_mu = 1.7
    s_eta = 1.5
    vp_plot(ux, uy, s_mu, s_eta)
    ####
    s_mu = 1.9
    s_eta = 1.
    stability_array_in_u(s_mu, s_eta)
    ####
    ux, uy = 0.1, 0.1
    stability_array_in_s(ux, uy)
