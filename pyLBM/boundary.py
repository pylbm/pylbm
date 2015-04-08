# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from .logs import setLogger

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
        self.indices = np.array(ind)
        if self.indices.size != 0:
            self.indices += np.asarray(v.v[::-1])[:, np.newaxis]
        self.distance = np.array(domain.distance[(num,) + ind])

class Boundary:
    def __init__(self, domain, dico):
        self.log = setLogger(__name__)
        self.domain = domain
        self.dico = dico

        # build the list of indices for each unique velocity and for each label
        self.bv = {}
        for label in self.domain.geom.list_of_labels():
            dummy_bv = []
            for k in xrange(self.domain.stencil.unvtot):
                dummy_bv.append(Boundary_Velocity(self.domain, label, k))
            self.bv[label] = dummy_bv

        # build the list of boundary informations for each stencil and each label
        self.be = []
        self.method_bc = []
        self.value_bc = []

        dico_bound = dico.get('boundary_conditions',{})

        for label in self.domain.geom.list_of_labels():
            if label == -1: # periodic conditions
                pass
            elif label == -2: # interface conditions
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
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v[::-1])]
    i3 = [i + 2*j for i, j in zip(i1, v.v[::-1])]
    s = 2.*bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = s*f[[ksym] + i2] + (1.-s)*f[[ksym] + i3]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] - feq[ksym, mask]
    else:
        f[i1 + [k]] = s*f[i2 + [ksym]] + (1.-s)*f[i3 + [ksym]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] - feq[mask, ksym]

    mask = np.logical_not(mask)
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v[::-1])]
    s = 0.5/bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = s*f[[ksym] + i2] + (1.-s)*f[[k] + i2]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] - feq[ksym, mask]
    else:
        f[i1 + [k]] = s*f[i2 + [ksym]] + (1.-s)*f[i2 + [k]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] - feq[mask, ksym]

def bouzidi_anti_bounce_back(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    ksym = num2index[v.get_symmetric().num]

    mask = bv.distance < .5
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v[::-1])]
    i3 = [i + 2*j for i, j in zip(i1, v.v[::-1])]
    s = 2.*bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = -s*f[[ksym] + i2] - (1.-s)*f[[ksym] + i3]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] + feq[ksym, mask]
    else:
        f[i1 + [k]] = -s*f[i2 + [ksym]] - (1.-s)*f[i3 + [ksym]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] + feq[mask, ksym]

    mask = np.logical_not(mask)
    i1 = list(bv.indices[:, mask])
    i2 = [i + j for i, j in zip(i1, v.v[::-1])]
    s = 0.5/bv.distance[mask]

    if nv_on_beg:
        f[[k] + i1] = -s*f[[ksym] + i2] - (1.-s)*f[[k] + i2]
        if feq is not None and np.any(mask):
            f[[k] + i1] += feq[k, mask] + feq[ksym, mask]
    else:
        f[i1 + [k]] = -s*f[i2 + [ksym]] - (1.-s)*f[i2 + [k]]
        if feq is not None and np.any(mask):
            f[i1 + [k]] += feq[mask, k] + feq[mask, ksym]

def neumann_vertical(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    # TODO: find a way to avoid these copies
    i1 = list(bv.indices.copy())
    i2 = list(bv.indices.copy())
    i2[-1] += v.vx

    if nv_on_beg:
        f[[k] + i1] = f[[k] + i2]
    else:
        f[i1 + [k]] = f[i2 + [k]]

def neumann_horizontal(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    # TODO: find a way to avoid these copies
    i1 = list(bv.indices.copy())
    i2 = list(bv.indices.copy())
    dim = bv.indices.shape[0]
    i2[0 if dim == 2 else 1] += v.vy

    if nv_on_beg:
        f[[k] + i1] = f[[k] + i2]
    else:
        f[i1 + [k]] = f[i2 + [k]]

def neumann(f, bv, num2index, feq, nv_on_beg):
    v = bv.v
    k = num2index[v.num]
    # TODO: find a way to avoid this copy
    i1 = list(bv.indices.copy())
    i2 = [i + j for i, j in zip(i1, v.v[::-1])]

    if nv_on_beg:
        f[[k] + i1] = f[[k] + i2]
    else:
        f[i1 + [k]] = f[i2 + [k]]

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
