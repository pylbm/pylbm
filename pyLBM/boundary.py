# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from .logs import setLogger
log = setLogger(__name__)


class Boundary_Velocity:
    """
    Indices and distances for the label and the velocity ksym
    """
    def __init__(self, domain, label, ksym):
        self.label = label
        # on cherche les points de l'exterieur qui ont une vitesse qui rentre (indice ksym)
        # sur un bord labelise par label
        # on parcourt toutes les vitesses et on determine les points interieurs qui ont la vitesse
        # symmetrique (indice k) qui sort
        # puis on ecrit dans une liste reprenant l'ordre des vitesses du schema
        # - les indices des points exterieurs correspondants
        # - les distances associees
        self.v = domain.stencil.unique_velocities[ksym]
        v = self.v.get_symmetric()
        num = domain.stencil.unum2index[v.num]
        ind = np.where(domain.flag[num] == self.label)
        self.indices = np.array([ind[0] + v.vy, ind[1] + v.vx])
        self.distance = np.array(domain.distance[num, ind[0], ind[1]])

class Boundary:
    def __init__(self, domain, dico):
        self.domain = domain
        self.dico = dico

        # build the list of indices for each unique velocity and for each label
        self.bv = []
        for label in self.domain.geom.list_of_labels():
            dummy_bv = []
            for k in xrange(self.domain.stencil.unvtot):
                dummy_bv.append(Boundary_Velocity(self.domain, label, k))
            self.bv.append(dummy_bv)

        # build the list of boundary informations for each stencil and each label
        self.be = []
        self.method_bc = []
        self.value_bc = []

        dico_bound = dico.get('boundary_conditions',{})

        for label in self.domain.geom.list_of_labels():
            if (label == -1): # periodic conditions
                pass
            else: # non periodic conditions
                self.be.append([])
                self.method_bc.append([])
                self.value_bc.append(dico_bound[label].get('value', None))
                for n in xrange(self.domain.stencil.nstencils):
                    self.method_bc[-1].append(dico_bound[label]['method'][n])
                    self.be[-1].append([self.bv[label][self.domain.stencil.unum2index[numk]] for numk in self.domain.stencil.num[n]])


def bouzidi_bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance < .5
    iy = bv.indices[0, mask]
    ix = bv.indices[1, mask]
    s = 2.*bv.distance[mask]
    if nv_on_beg:
        f[k, iy, ix] = s*f[ksym, iy + v.vy, ix + v.vx] + (1.-s)*f[ksym, iy + 2*v.vy, ix + 2*v.vx]
        if feq is not None and np.any(mask):
            f[k, iy, ix] += feq[k, mask] - feq[ksym, mask]
    else:
        f[iy, ix, k] = s*f[iy + v.vy, ix + v.vx, ksym] + (1.-s)*f[iy + 2*v.vy, ix + 2*v.vx, ksym]
        if feq is not None and np.any(mask):
            f[iy, ix, k] += feq[mask, k] - feq[mask, ksym]

    #print "bb1"*20
    #print f[iy, ix, k].T

    mask = np.logical_not(mask)
    iy = bv.indices[0, mask]
    ix = bv.indices[1, mask]
    s = 0.5/bv.distance[mask]
    if nv_on_beg:
        f[k, iy, ix] = s*f[ksym, iy + v.vy, ix + v.vx] + (1.-s)*f[k, iy + v.vy, ix + v.vx]
        if feq is not None and np.any(mask):
            f[k, iy, ix] += feq[k, mask] - feq[ksym, mask]
    else:
        f[iy, ix, k] = s*f[iy + v.vy, ix + v.vx, ksym] + (1.-s)*f[iy + v.vy, ix + v.vx, k]
        #print s, ix, iy
        #f[iy, ix, k] = s*f[iy + v.vy, ix + v.vx, k]
        #f[iy, ix, k] = s*f[iy + v.vy, ix + v.vx, k]
        #print f[:, :, k].T
        if feq is not None and np.any(mask):
            f[iy, ix, k] += feq[mask, k] - feq[mask, ksym]

    #print "bb2"*20
    #print f[iy, ix, k].T
    #return f

def bouzidi_anti_bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance < .5
    iy = bv.indices[0, mask]
    ix = bv.indices[1, mask]
    s = 2.*bv.distance[mask]
    f[k, ix, iy] = - s*f[ksym, ix + v.vx, iy + v.vy] - (1.-s)*f[ksym, ix + 2*v.vx, iy + 2*v.vy]
    if feq is not None:
        f[k, ix, iy] += feq[k, mask] + feq[ksym, mask]

    mask = np.logical_not(mask)
    iy = bv.indices[0, mask]
    ix = bv.indices[1, mask]
    s = 0.5/bv.distance[mask]
    f[k, ix, iy] = - s*f[ksym, ix + v.vx, iy + v.vy] - (1.-s)*f[k, ix + v.vx, iy + v.vy]
    if feq is not None:
        f[k, ix, iy] += feq[k, mask] + feq[ksym, mask]

def neumann_vertical(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    iy = bv.indices[0]
    ix = bv.indices[1]
    if nv_on_beg:
        f[k, iy, ix] = f[k, iy, ix + v.vx]
    else:
        f[iy, ix, k] = f[iy, ix + v.vx, k]
    #print 'neumann1'*10
    #print f[iy, ix, k].T

def neumann_horizontal(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    iy = bv.indices[0]
    ix = bv.indices[1]
    if nv_on_beg:
        f[k, iy, ix] = f[k, iy + v.vy, ix]
    else:
        f[iy, ix, k] = f[iy + v.vy, ix, k]

def neumann(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    iy = bv.indices[0]
    ix = bv.indices[1]
    f[k, iy, ix] = f[k, iy + v.vy, ix + v.vx]

if __name__ == "__main__":
    from pyLBM.elements import *
    import geometry, domain
    import numpy as np

    dim = 2
    dx = .1
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.

    dico_geometry = {'dim':dim,
                     'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':[0,0,1,0]},
                     'Elements':[0],
                     0:{'Element':Circle([0.5*(xmin+xmax),0.5*(ymin+ymax)], 0.3),
                        'del':True,
                        'label':2}
                     }

    dico   = {'dim':dim,
              'eometry':dico_geometry,
              'space_step':dx,
              'number_of_schemes':1,
              0:{'velocities':range(9),}
              }

    geom = Geometry.Geometry(dico)
    dom = Domain.Domain(geom,dico)
    b = Boundary(dom, 2, 0)
    print b.indices
    print
    print b.distance
