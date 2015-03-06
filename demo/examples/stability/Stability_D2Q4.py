import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import pylab as plt
import sympy as sp
from sympy.matrices import Matrix, zeros
import pyLBM.scheme as sch

X, Y, Z, LA = sp.symbols('X,Y,Z,LA')
u = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(25)] for i in xrange(10)]


def scheme_constructor(ux, uy, s, s3):
    dico1 = {
        'dim':2,
        'scheme_velocity':1.,
        'schemes':[
            {
            'velocities':range(1, 5),
            'polynomials':Matrix([1, LA*X, LA*Y, LA**2*(X**2-Y**2)]),
            'relaxation_parameters':[0., s, s, s3],
            'equilibrium':Matrix([u[0][0], ux*u[0][0], uy*u[0][0], 0.]),
            },
        ],
        'stability':{
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    dico2 = {
        'dim':2,
        'scheme_velocity':1.,
        'schemes':[
            {
            'velocities':range(1, 5),
            'polynomials':Matrix([1, LA*(X-ux), LA*(Y-uy), LA**2*((X-ux)**2-(Y-uy)**2)]),
            'relaxation_parameters':[0., s, s, s3],
            'equilibrium':Matrix([u[0][0], 0., 0., 0.]),
            },
        ],
        'stability':{
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    dico3 = {
        'dim':2,
        'scheme_velocity':1.,
        'schemes':[
            {
            'velocities':range(5, 9),
            'polynomials':Matrix([1, LA*X, LA*Y, LA**2*X*Y]),
            'relaxation_parameters':[0., s, s, s3],
            'equilibrium':Matrix([u[0][0], ux*u[0][0], uy*u[0][0], 0.]),
            },
        ],
        'stability':{
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    dico4 = {
        'dim':2,
        'scheme_velocity':1.,
        'schemes':[
            {
            'velocities':range(5, 9),
            'polynomials':Matrix([1, LA*(X-ux), LA*(Y-uy), LA**2*(X-ux)*(Y-uy)]),
            'relaxation_parameters':[0., s, s, s3],
            'equilibrium':Matrix([u[0][0], 0., 0., 0.]),
            },
        ],
        'stability':{
            'test_maximum_principle':False,
            'test_L2_stability':False,
        },
    }
    return sch.Scheme(dico1)

def vp_plot(ux, uy, s, s3):
    S = scheme_constructor(ux, uy, s, s3)
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
                print "Spectral radius for theta = {0:5.3f}: {1:10.3e}".format(theta, rloc)
            plt.plot(vp.real, vp.imag, 'ko')
        i += 1
        plt.hold(False)
        plt.title('eigenvalues for $\Theta = {0:5.3f}$'.format(theta))
        plt.pause(1.e-1)
    print "Maximal spectral radius: {0:10.3e}".format(R)

def stability_array_in_u(s, s3):
    plt.figure(1)
    plt.clf()
    plt.axis('equal')
    plt.hold(True)
    N = 32
    vux = np.linspace(0., 1., N+1)
    vuy = np.linspace(0., 1., N+1)
    mR = np.zeros((vux.size, vuy.size))
    nb_calcul = 0
    mR, nb_calcul = stability_array_in_u_recur(s, s3, vux, vuy, mR, [0,N,0,N], nb_calcul)
    plt.hold(False)
    print "Number of stability computations: {0:d}".format(nb_calcul)
    plt.show()

def stability_array_in_u_recur(s, s3, vux, vuy, mR, l, nb_calcul):
    dummy = 0
    for i in l[0:2]:
        for j in l[2:4]:
            if (mR[i, j] == 0) & (mR[j, i] == 0):
                S = scheme_constructor(vux[i], vuy[j], s, s3)
                nb_calcul += 1
                if S.is_stable_L2(Nk = 51):
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
        mR, nb_calcul = stability_array_in_u_recur(s, s3, vux, vuy, mR, laa, nb_calcul)
        mR, nb_calcul = stability_array_in_u_recur(s, s3, vux, vuy, mR, lab, nb_calcul)
        mR, nb_calcul = stability_array_in_u_recur(s, s3, vux, vuy, mR, lba, nb_calcul)
        mR, nb_calcul = stability_array_in_u_recur(s, s3, vux, vuy, mR, lbb, nb_calcul)
    return mR, nb_calcul

def stability_array_in_s(ux, uy):
    plt.figure(1)
    plt.clf()
    plt.axis('equal')
    plt.hold(True)
    N = 64
    vs = np.linspace(0., 2., N+1)
    vs3 = np.linspace(0., 2., N+1)
    mR = np.zeros((vs.size, vs3.size))
    nb_calcul = 0
    mR, nb_calcul = stability_array_in_s_recur(vs, vs3, ux, uy, mR, [0,N,0,N], nb_calcul)
    plt.hold(False)
    print "Number of stability computations: {0:d}".format(nb_calcul)
    plt.show()

def stability_array_in_s_recur(vs, vs3, ux, uy, mR, l, nb_calcul):
    dummy = 0
    for i in l[0:2]:
        for j in l[2:4]:
            if mR[i, j] == 0:
                S = scheme_constructor(ux, uy, vs[i], vs3[j])
                nb_calcul += 1
                if S.is_stable_L2(Nk = 51):
                    plt.scatter(vs[i], vs3[j], c = 'b', marker = 'o')
                    mR[i, j] = 1
                else:
                    plt.scatter(vs[i], vs3[j], c = 'r', marker = 's')
                    mR[i, j] = -1
                plt.pause(1.e-5)
            dummy += mR[i, j]
    if ((l[1] - l[0] > 1) & (abs(dummy) < 4)) | (l[1] - l[0] > 16):
        laa = [l[0], (l[0]+l[1])/2, l[2], (l[2]+l[3])/2]
        lab = [l[0], (l[0]+l[1])/2, (l[2]+l[3])/2, l[3]]
        lba = [(l[0]+l[1])/2, l[1], l[2], (l[2]+l[3])/2]
        lbb = [(l[0]+l[1])/2, l[1], (l[2]+l[3])/2, l[3]]
        mR, nb_calcul = stability_array_in_s_recur(vs, vs3, ux, uy, mR, laa, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs, vs3, ux, uy, mR, lab, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs, vs3, ux, uy, mR, lba, nb_calcul)
        mR, nb_calcul = stability_array_in_s_recur(vs, vs3, ux, uy, mR, lbb, nb_calcul)
    return mR, nb_calcul

if __name__ == "__main__":
    s = 1./(.5+1./np.sqrt(12))
    s3 = s#1./(.5+1./np.sqrt(3))
    ux, uy = .5, .5
    vp_plot(ux, uy, s, s3)
    ####
    s = 1./(.5+1./np.sqrt(12))
    s3 = s
    stability_array_in_u(s, s3)
    ####
    ux, uy = 0.5, 0.5
    stability_array_in_s(ux, uy)
