# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

def intersection_two_lines(p1, v1, p2, v2):
    """
    intersection between two lines defined by a point and a vector.
    """
    alpha = beta = None
    det = v1[1]*v2[0] - v1[0]*v2[1]
    if det != 0:
        invdet = 1./det
        c1 = p2[0] - p1[0]
        c2 = p2[1] - p1[1]
        alpha = (-v2[1]*c1 + v2[0]*c2)*invdet
        beta = (-v1[1]*c1 + v1[0]*c2)*invdet
    return alpha, beta

def distance_lines(x, y, v, p, vt, dmax, label):
    """
    return distance for several lines
    """
    v2 = np.asarray([x, y])
    alpha = 1e16*np.ones((x.size, y.size))
    border = -np.ones((x.size, y.size))
    for i in xrange(len(vt)):
        tmp1, tmp2 = intersection_two_lines(v2, v, p[i], vt[i])
        if tmp1 is not None:
            if dmax is None:
                ind = np.logical_and(tmp1>0, np.logical_and(tmp2>=0, tmp2<=1))
            else:
                ind = np.logical_and(np.logical_and(tmp1>0, tmp1<=dmax),
                                   np.logical_and(tmp2>=0, tmp2<=1))
            ind = np.where(np.logical_and(alpha>tmp1, ind))
            alpha[ind] = tmp1[ind]
            border[ind] = label[i]
    alpha[alpha == 1e16] = -1.
    return alpha, border

def distance_ellipse(x, y, v, c, v1, v2, dmax, label):
    """
    return the distance according
    a line defined by a point x, y and a vector v
    to an ellipse defined by a point c and two vectors v1 and v2
    """
    # build the equation of the ellipse
    # then write the second order equation in d
    # a d**2 + b d + c = 0
    # delta = b**2-4ac
    X = x - c[0]
    Y = y - c[1]
    vx2 = v1[0]**2 + v2[0]**2
    vy2 = v1[1]**2 + v2[1]**2
    vxy = v1[0]*v1[1] + v2[0]*v2[1]
    a = v[0]**2*vy2 + v[1]**2*vx2 - 2*v[0]*v[1]*vxy
    b = 2*X*v[0]*vy2 + 2*Y*v[1]*vx2 - 2*(X*v[1]+Y*v[0])*vxy
    c = X**2*vy2 + Y**2*vx2 - 2*X*Y*vxy - (v1[0]*v2[1]-v1[1]*v2[0])**2
    delta = b**2 - 4*a*c
    ind = delta>=0
    delta[ind] = np.sqrt(delta[ind])
    d1 = 1e16*np.ones(delta.shape)
    d2 = 1e16*np.ones(delta.shape)
    d1[ind] = (-b[ind]-delta[ind]) / (2*a)
    d2[ind] = (-b[ind]+delta[ind]) / (2*a)
    d1[d1<0] = 1e16
    d2[d2<0] = 1e16
    d = -np.ones(d1.shape)
    d[ind] = np.minimum(d1[ind], d2[ind])
    d[d==1e16] = -1

    alpha = -np.ones((x.size, y.size))
    border = -np.ones((x.size, y.size))
    if dmax is None:
        ind = d>0
    else:
        ind = np.logical_and(d>0, d<=dmax)
    alpha[ind] = d[ind]
    border[ind] = label[0]
    return alpha, border
